"""
Incremental MiniBatchKMeans clustering on Milvus vector data using scikit-learn.
Supports batch loading from Milvus and optional batch PCA.
Saves trained models and metadata to src/sklearn_models/minibatch_kmeans/
"""

import os
import joblib
import json
import time
import numpy as np
from pymilvus import connections, Collection
from src.data_ingestion import config
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

# Settings
VECTOR_COLLECTION_NAME = config.COLLECTION_NAME
MODEL_DIR = "src/sklearn_models/minibatch_kmeans"
MODEL_FILE = os.path.join(MODEL_DIR, "minibatch_kmeans.joblib")
PCA_FILE = os.path.join(MODEL_DIR, "incremental_pca.joblib")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.joblib")
METADATA_FILE = os.path.join(MODEL_DIR, "metadata.json")

BATCH_SIZE = 50000       # Number of vectors per batch
N_CLUSTERS = 10           # Change as needed
USE_PCA = True            # Set False to skip PCA
PCA_COMPONENTS = 128      # Dimensionality after PCA
VERBOSE = True

def log(msg):
    if VERBOSE:
        print(msg)

def ensure_model_dir():
    if os.path.exists(MODEL_DIR) and os.path.exists(MODEL_FILE):
        log(f"Model already exists at '{MODEL_FILE}'. Skipping training.")
        return False
    os.makedirs(MODEL_DIR, exist_ok=True)
    return True

def load_vectors_batch(col, start_id, end_id):
    """Load a batch of vectors from Milvus collection by id range."""
    expr = f"id >= {start_id} and id <= {end_id}"
    results = col.query(expr=expr, output_fields=["emb"])
    batch_vectors = [r["emb"] for r in results]
    return np.array(batch_vectors)

def main():
    if not ensure_model_dir():
        return

    log("Connecting to Milvus...")
    connections.connect('default', host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    col = Collection(VECTOR_COLLECTION_NAME)
    col.load()
    num_entities = col.num_entities
    log(f"Collection '{VECTOR_COLLECTION_NAME}' has {num_entities} vectors.")

    scaler = StandardScaler()
    pca = IncrementalPCA(n_components=PCA_COMPONENTS) if USE_PCA else None
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=BATCH_SIZE, random_state=42, verbose=1)

    start_time = time.time()
    total_processed = 0

    # First pass: scaling and optional PCA fitting
    log("\n--- Scaling and optional PCA fitting ---")
    for start in range(0, num_entities, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_entities)
        batch_vectors = load_vectors_batch(col, start+1, end)
        total_processed += batch_vectors.shape[0]
        log(f"Batch {start//BATCH_SIZE+1}: Loaded {batch_vectors.shape[0]} vectors")

        batch_vectors = scaler.partial_fit(batch_vectors).transform(batch_vectors)
        if USE_PCA:
            pca.partial_fit(batch_vectors)
        log(f"Batch {start//BATCH_SIZE+1}: Scaled{' and PCA-fitted' if USE_PCA else ''}")

    # Second pass: incremental MiniBatchKMeans
    log("\n--- Training MiniBatchKMeans ---")
    for start in range(0, num_entities, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_entities)
        batch_vectors = load_vectors_batch(col, start+1, end)
        batch_vectors = scaler.transform(batch_vectors)
        if USE_PCA:
            batch_vectors = pca.transform(batch_vectors)

        kmeans.partial_fit(batch_vectors)
        log(f"Batch {start//BATCH_SIZE+1}: MiniBatchKMeans partial_fit done")

    # Save models
    joblib.dump(scaler, SCALER_FILE)
    if USE_PCA:
        joblib.dump(pca, PCA_FILE)
    joblib.dump(kmeans, MODEL_FILE)

    # Save metadata
    metadata = {
        "vector_collection_name": VECTOR_COLLECTION_NAME,
        "num_vectors_processed": total_processed,
        "batch_size": BATCH_SIZE,
        "n_clusters": N_CLUSTERS,
        "use_pca": USE_PCA,
        "pca_components": PCA_COMPONENTS if USE_PCA else None,
        "model_files": {
            "scaler": SCALER_FILE,
            "pca": PCA_FILE if USE_PCA else None,
            "kmeans": MODEL_FILE
        },
        "training_time_seconds": round(time.time() - start_time, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

    log(f"\nTraining complete! Models and metadata saved to '{MODEL_DIR}'")
    log(json.dumps(metadata, indent=4))

if __name__ == "__main__":
    main()
