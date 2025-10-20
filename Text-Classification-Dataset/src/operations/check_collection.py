"""
check_collection.py
Inspect and query your Milvus text embeddings collection.
"""

import sys
import os
sys.path.append(os.getcwd())

from pymilvus import connections, Collection
from src.data_ingestion import config

def connect_to_milvus():
    """Connect to Milvus server."""
    connections.connect('default', host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    print(f"Connected to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}")

def show_schema():
    """Print collection schema."""
    col = Collection(config.COLLECTION_NAME)
    print("Collection schema:")
    print(col.schema)

def show_num_entities():
    """Print total number of vectors in the collection."""
    col = Collection(config.COLLECTION_NAME)
    print(f"Entities in '{config.COLLECTION_NAME}': {col.num_entities}")

def show_sample_rows(n=5):
    """Print first N rows of the collection."""
    col = Collection(config.COLLECTION_NAME)
    col.load()  # load into memory
    expr = f"id < {n+1}"
    results = col.query(expr=expr, output_fields=['id', 'text'])
    print(f"First {n} rows:")
    for row in results:
        print(row)

def main():
    connect_to_milvus()
    print("\n--- SCHEMA ---")
    show_schema()
    print("\n--- ENTITY COUNT ---")
    show_num_entities()
    print("\n--- SAMPLE ROWS ---")
    show_sample_rows(n=5)

if __name__ == "__main__":
    main()
