from pathlib import Path

BASE_DIR = Path(__file__).parent

# CSV file paths (adjusted to point to project root)
CSV_FILES = [
    str(BASE_DIR.parent.parent / "precleaned-chunks" / f"precleaned_chunk_{i}.csv")
    for i in range(1, 6)
]

TEXT_COLUMN = "body_cleaned"

# Embedding model
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Milvus connection
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530

# Collection settings
print("\n-------From config.py-------")
COLLECTION_NAME = input("Enter collection name: ")
VECTOR_DIM = 384
INDEX_FILE_SIZE = 1024
METRIC_TYPE = "IP"

# Batch sizes
BATCH_SIZE = 2048
CHUNK_SIZE = 8000

# Export directory
EXPORT_DIR = str(BASE_DIR.parent / "exports")
