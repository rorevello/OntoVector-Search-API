import os
import json
import logging
import numpy as np
import requests
import gc
import pandas as pd
from tqdm import tqdm
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
from rdflib import Graph, RDF, OWL, RDFS
from rdflib.util import guess_format
import obonet
from more_itertools import chunked
from pathlib import Path

# ----------------------------
# Configuration
# ----------------------------

MODEL_NAME = "all-MiniLM-L6-v2"
DATA_PATH = "onto.json"
ERRORS_PATH = "errors_.json"
TMP_DIR = "tmp_vectors"
CHUNK_SIZE = 10000
BATCH_SIZE = 64
OUTPUT_VECTORS = "vectors.npy"
OUTPUT_LABELS = "labels.json"
OUTPUT_METADATA = "metadata.parquet"

# ----------------------------
# Logging
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("vectorize_ontologies.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

model = SentenceTransformer(MODEL_NAME)
errors = {}


def url_exists(url):
    try:
        r = requests.head(url, allow_redirects=True, timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def extract_labels_from_obo(url):
    result = {}
    try:
        graph = obonet.read_obo(url)
        for node_id, data in graph.nodes(data=True):
            if 'name' in data:
                label = data['name'].strip()
                if label:
                    result.setdefault(label, []).append(node_id)
    except Exception as e:
        errors[url] = f"OBO ERROR: {e}"
        logger.warning(f"⚠️ Error processing (OBO) '{url}': {e}")
    return result

def extract_labels_from_owl(url):
    result = {}
    try:
        g = Graph()
        fmt = guess_format(url) or "xml"
        g.parse(url, format=fmt)
        for cls in g.subjects(RDF.type, OWL.Class):
            labels = [str(label).strip() for label in g.objects(cls, RDFS.label)]
            for label in labels:
                if label:
                    result.setdefault(label, []).append(str(cls))
    except Exception as e:
        errors[url] = f"OWL ERROR: {e}"
        logger.warning(f"⚠️ Error processing (OWL) '{url}': {e}")
    return result

def extract_labels_from_url(url):
    if not url_exists(url):
        errors[url] = "URL not accessible"
        logger.warning(f"⚠️ URL not accessible: {url}")
        return {}
    if url.endswith(".obo"):
        return extract_labels_from_obo(url)
    else:
        return extract_labels_from_owl(url)

def load_urls_from_obo_json(path_json):
    if not os.path.exists(path_json):
        raise FileNotFoundError(f"File '{path_json}' not found.")
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    urls = set()


    # Extract URLs from dicc_w3id and bioportal
    for a, b in data.items():
        for c in b:
            try:
                urls.add(c)
            except:
                logger.warning(f"⚠️ Error 'line 123' processing URL: {c}")
    
    return list(urls)
    

def build_vector_components():
    os.makedirs(TMP_DIR, exist_ok=True)
    urls = load_urls_from_obo_json(DATA_PATH)
    logger.info(f"🔗 Unique URLs found: {len(urls)}")

    label_to_entries = {}
    for url in tqdm(urls, desc="Processing ontologies"):
        extracted = extract_labels_from_url(url)
        for label, uris in extracted.items():
            entries = [{"uri": uri, "ontology": url} for uri in uris]
            label_to_entries.setdefault(label, []).extend(entries)

    labels = list(label_to_entries.keys())
    logger.info(f"📌 Total unique labels: {len(labels)}")
    logger.info("💡 Vectorizing and storing temporarily...")

    all_labels = []
    all_vectors = []
    all_metadata = []

    for i, batch in enumerate(chunked(labels, CHUNK_SIZE)):
        logger.info(f"🧩 Block {i + 1}: {len(batch)} labels")
        try:
            vectors = model.encode(batch, batch_size=BATCH_SIZE, show_progress_bar=False)
        except Exception as e:
            logger.error(f"❌ Error during vectorization of block {i + 1}: {e}")
            continue

        for label, vector in zip(batch, vectors):
            all_labels.append(label)
            all_vectors.append(vector)

            entry_list = label_to_entries.get(label, [])
            if entry_list:
                entry = entry_list[0]  
                all_metadata.append({
                    "label": label,
                    "uri": entry["uri"],
                    "ontology": entry["ontology"]
                })
            else:
                logger.warning(f"Label without associated entries: {label}")
                all_metadata.append({
                    "label": label,
                    "uri": "unknown",
                    "ontology": "unknown"
                })

        del vectors
        gc.collect()

    n_labels, n_vectors, n_metadata = len(all_labels), len(all_vectors), len(all_metadata)
    assert n_labels == n_vectors == n_metadata, \
        f"❌ Misalignment: labels={n_labels}, vectors={n_vectors}, metadata={n_metadata}"
    logger.info(f"Alignment verified: {n_labels} vectors")

    logger.info(f"Saving vectors to: {OUTPUT_VECTORS}")
    np.save(OUTPUT_VECTORS, np.array(all_vectors))

    logger.info(f"Saving labels to: {OUTPUT_LABELS}")
    with open(OUTPUT_LABELS, "w", encoding="utf-8") as f:
        json.dump(all_labels, f, indent=2, ensure_ascii=False)

    logger.info(f"Saving metadata to: {OUTPUT_METADATA}")
    df = pd.DataFrame(all_metadata)
    df.to_parquet(OUTPUT_METADATA, index=False)

    with open(ERRORS_PATH, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    logger.info("Vectorization and export completed.")

if __name__ == "__main__":
    build_vector_components()
