import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query
from pydantic import BaseModel
from scipy.spatial.distance import cdist
import logging
from collections import defaultdict
from numpy.linalg import LinAlgError
from fastapi import HTTPException
import math

# ----------------------------
# Configuration
# ----------------------------

DATA_DIR = "data"
MODEL_NAME = "all-MiniLM-L6-v2"
NUM_RESULTS = 10

# ----------------------------
# Logging
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("search_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------
# Response models
# ----------------------------

class Entry(BaseModel):
    uri: str
    ontology: str
    source: str

class SearchResult(BaseModel):
    label: str
    score: float
    entries: List[Entry]

class SearchResponse(BaseModel):
    query: str
    source: str
    results: List[SearchResult]

# ----------------------------
# Load model and app
# ----------------------------

model = SentenceTransformer(MODEL_NAME)
app = FastAPI(title="Semantic Search API (Multi-Source)")

# ----------------------------
# Load sources
# ----------------------------

sources: Dict[str, Dict[str, object]] = {}

logger.info("📂 Searching for datasets in the data folder...")
for subdir in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, subdir)
    if not os.path.isdir(path):
        continue

    try:
        vectors = np.load(os.path.join(path, "vectors.npy"))
        with open(os.path.join(path, "labels.json"), "r", encoding="utf-8") as f:
            labels = json.load(f)
        # metadata = pd.read_parquet(os.path.join(path, "metadata.parquet"))
        metadata = pd.read_parquet(os.path.join(path, "metadata.parquet"))
        metadata.reset_index(drop=True, inplace=True)  

        sources[subdir] = {
            "vectors": vectors,
            "labels": labels,
            "metadata": metadata
        }
        logger.info(f"✅ Source loaded: {subdir} ({len(labels)} vectors)")
    except Exception as e:
        logger.warning(f"⚠️ Could not load '{subdir}': {e}")

# ----------------------------
# Endpoints
# ----------------------------

@app.get("/sources")
def list_sources():
    return list(sources.keys())

@app.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., min_length=1),
    k: int = Query(NUM_RESULTS, gt=0, le=100),
    source: Optional[str] = Query(None)
):
    logger.info(f"🔎 Query: '{query}' in source: '{source or 'ALL'}'")
    query_vec = model.encode([query])

    datasets = [source] if source else list(sources.keys())
    results = []

    for src in datasets:
        if src not in sources:
            continue

        data = sources[src]
        vectors = data["vectors"]
        labels = data["labels"]
        metadata = data["metadata"]

        dists = cdist(query_vec, vectors, metric="cosine")[0]
        sorted_indices = np.argsort(dists)

        for idx in sorted_indices:
            label = labels[idx]
            score = float(dists[idx])
            entries = metadata[metadata["label"] == label][["uri", "ontology"]].drop_duplicates()
            entries_list = [Entry(uri=row.uri, ontology=row.ontology, source=src) for row in entries.itertuples(index=False)]
            results.append(SearchResult(label=label, score=round(score, 5), entries=entries_list))
            if len(results) >= k:
                break

    logger.info(f"📤 Results returned: {len(results)}")
    return SearchResponse(query=query, source=source or "all", results=results)

@app.get("/alignments", response_model=List[SearchResult])
def find_alignments(
    label: str = Query(..., min_length=1),
    threshold: float = Query(0.0001, ge=0.0, le=1.0),
    source: Optional[str] = Query(None)
):
    logger.info(f"🔗 Finding alignments for '{label}' with threshold ≤ {threshold}")
    query_vec = model.encode([label])
    datasets = [source] if source else list(sources.keys())
    grouped_results = defaultdict(lambda: {"score": float("inf"), "entries": []})

    for src in datasets:
        if src not in sources:
            continue

        data = sources[src]
        vectors = data["vectors"]
        labels = data["labels"]
        metadata = data["metadata"]

        dists = cdist(query_vec, vectors, metric="cosine")[0]

        for idx, score in enumerate(dists):
            if score <= threshold:
                matched_label = labels[idx]
                norm_label = matched_label.lower()
                entries = metadata[metadata["label"].str.lower() == norm_label][["uri", "ontology"]].drop_duplicates()
                entries_list = [Entry(uri=row.uri, ontology=row.ontology, source=src) for row in entries.itertuples(index=False)]

                if score < grouped_results[norm_label]["score"]:
                    grouped_results[norm_label]["score"] = score
                grouped_results[norm_label]["entries"].extend(entries_list)

    results = []
    for norm_label, data in grouped_results.items():
        unique_entries = {(e.uri, e.ontology, e.source): e for e in data["entries"]}
        results.append(SearchResult(label=norm_label, score=round(data["score"], 5), entries=list(unique_entries.values())))

    logger.info(f"🔍 Alignments found: {len(results)}")
    return results


@app.get("/dsr_class")
def compute_dsr_class(source: str, ontology: str):
    logger.info(f"📏 Calculating DSR for classes in ontology '{ontology}' within source '{source}'")

    if source not in sources:
        logger.warning(f"❌ Source '{source}' not found.")
        raise HTTPException(status_code=404, detail=f"Source '{source}' not available.")

    data = sources[source]
    metadata = data["metadata"].reset_index(drop=True)  # 🔧 Ensure alignment
    vectors = data["vectors"]

    matching_rows = metadata[metadata["ontology"] == ontology]

    if matching_rows.empty:
        logger.warning(f"❌ Ontology '{ontology}' not found in source '{source}'.")
        return {
            "source": source,
            "ontology": ontology,
            "Nc": 0,
            "DSC": None,
            "DSRC": None,
            "mu": None,
            "status": "ontology_not_found"
        }

    indices = matching_rows.index.to_numpy()

    if np.max(indices) >= vectors.shape[0]:
        logger.error("❌ Out-of-range indices: metadata and vectors are misaligned.")
        return {
            "source": source,
            "ontology": ontology,
            "Nc": len(indices),
            "DSC": None,
            "DSRC": None,
            "mu": None,
            "status": "index_mismatch"
        }

    selected_vectors = vectors[indices]
    Nc = selected_vectors.shape[0]

    if Nc < 2:
        logger.warning("⚠️ Insufficient number of classes to calculate DSR.")
        return {
            "source": source,
            "ontology": ontology,
            "Nc": Nc,
            "DSC": None,
            "DSRC": None,
            "status": "insufficient_classes"
        }

    if np.allclose(np.var(selected_vectors, axis=0), 0):
        logger.warning("⚠️ Vectors with zero variance. Degenerate covariance matrix.")
        return {
            "source": source,
            "ontology": ontology,
            "Nc": Nc,
            "DSC": None,
            "DSRC": None,
            "status": "degenerate_vectors"
        }

    try:
        mu = np.mean(selected_vectors, axis=0)
        Sigma = np.cov(selected_vectors, rowvar=False)

        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except LinAlgError:
            logger.warning("⚠️ Non-invertible covariance. Applying regularization.")
            Sigma += np.eye(Sigma.shape[0]) * 1e-6
            Sigma_inv = np.linalg.inv(Sigma)

        diffs = selected_vectors - mu
        dists = np.sqrt(np.einsum('ij,jk,ik->i', diffs, Sigma_inv, diffs))

        if np.any(np.isnan(dists)) or np.any(dists < 0):
            logger.warning(f"⚠️ Mahalanobis produced invalid values (NaN or negative).")
            return {
                "source": source,
                "ontology": ontology,
                "Nc": Nc,
                "DSC": None,
                "DSRC": None,
                "status": "mahalanobis_invalid"
            }

        DSC = np.mean(dists)
        DSRC = DSC / np.sqrt(Nc)

        logger.info(f"📐 DSR_C = {DSRC:.5f} successfully calculated.")

        return {
            "source": source,
            "ontology": ontology,
            "Nc": Nc,
            "DSC": round(DSC, 5),
            "DSRC": round(DSRC, 5),
            "status": "ok"
        }

    except Exception as e:
        logger.exception(f"❌ Unexpected error while calculating DSR: {e}")
        raise HTTPException(status_code=500, detail="Internal error calculating DSR")
    
@app.get("/centroid")
def get_ontology_centroid(source: str, ontology: str):
    logger.info(f"📍 Calculating centroid for ontology '{ontology}' in source '{source}'")

    if source not in sources:
        logger.warning(f"❌ Source '{source}' not found.")
        raise HTTPException(status_code=404, detail=f"Source '{source}' not available.")

    data = sources[source]
    metadata = data["metadata"].reset_index(drop=True)
    vectors = data["vectors"]

    matching_rows = metadata[metadata["ontology"] == ontology]
    if matching_rows.empty:
        logger.warning(f"❌ Ontology '{ontology}' not found in source '{source}'.")
        return {
            "source": source,
            "ontology": ontology,
            "Nc": 0,
            "centroid": None,
            "status": "ontology_not_found"
        }

    indices = matching_rows.index.to_numpy()
    if np.max(indices) >= vectors.shape[0]:
        logger.error("❌ Out-of-range indices: metadata and vectors are misaligned.")
        return {
            "source": source,
            "ontology": ontology,
            "Nc": len(indices),
            "centroid": None,
            "status": "index_mismatch"
        }

    selected_vectors = vectors[indices]
    Nc = selected_vectors.shape[0]

    if Nc == 0:
        return {
            "source": source,
            "ontology": ontology,
            "Nc": 0,
            "centroid": None,
            "status": "empty_vector_set"
        }

    centroid = np.mean(selected_vectors, axis=0).tolist()
    logger.info(f"📍 Centroid calculated for {Nc} vectors")

    return {
        "source": source,
        "ontology": ontology,
        "Nc": Nc,
        "centroid": centroid,
        "status": "ok"
    }
