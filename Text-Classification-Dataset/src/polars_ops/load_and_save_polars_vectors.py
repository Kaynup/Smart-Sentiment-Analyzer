"""
Milvus vectors to Polars Parquet with crash-resume support.

- Saves each batch immediately.
- Resumes from last saved row if interrupted.
- Keeps raw text and embeddings (no cleaning).
"""

import os
import polars as pl
from pymilvus import connections, Collection
from src.data_ingestion import config

# -----------------------------
# SETTINGS
# -----------------------------
VECTOR_COLLECTION_NAME = config.COLLECTION_NAME
PARQUET_DIR = "exports/parquet"
BATCH_SIZE = 50000
VERBOSE = True

def log(msg):
    if VERBOSE:
        print(msg)

# -----------------------------
# CONNECT TO MILVUS
# -----------------------------
log("Connecting to Milvus...")
connections.connect('default', host=config.MILVUS_HOST, port=config.MILVUS_PORT)
col = Collection(VECTOR_COLLECTION_NAME)
col.load()
num_entities = col.num_entities
log(f"Collection '{VECTOR_COLLECTION_NAME}' has {num_entities} vectors.")

# -----------------------------
# RESUME CHECK
# -----------------------------
os.makedirs(PARQUET_DIR, exist_ok=True)
parquet_files = sorted([f for f in os.listdir(PARQUET_DIR) if f.endswith(".parquet")])

if parquet_files:
    # Find the last saved row ID
    last_file = os.path.join(PARQUET_DIR, parquet_files[-1])
    df_last = pl.read_parquet(last_file)
    last_id = df_last["id"].max()
    start_id = last_id + 1
    log(f"Resuming from ID {start_id}")
else:
    start_id = 1
    log("Starting from the beginning...")

if start_id > num_entities:
    log("All rows already saved! Nothing to do.")
    exit(0)

# -----------------------------
# FETCH AND SAVE IN BATCHES
# -----------------------------
for batch_start in range(start_id, num_entities + 1, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE - 1, num_entities)
    log(f"Fetching batch: {batch_start}-{batch_end} ...")

    expr = f"id >= {batch_start} and id <= {batch_end}"
    results = col.query(expr=expr, output_fields=["id", "text", "emb"])

    if not results:
        log("No results in this batch, skipping.")
        continue

    ids = [r["id"] for r in results]
    texts = [r["text"] for r in results]  # raw text
    vectors = [r["emb"] for r in results]

    df_batch = pl.DataFrame({
        "id": ids,
        "text": texts,
        "emb": vectors
    })

    # Save immediately
    parquet_file = os.path.join(PARQUET_DIR, f"vectors_{batch_start}_{batch_end}.parquet")
    df_batch.write_parquet(parquet_file)
    log(f"Saved batch to {parquet_file}")

log("All done! All data saved in Parquet format.")
