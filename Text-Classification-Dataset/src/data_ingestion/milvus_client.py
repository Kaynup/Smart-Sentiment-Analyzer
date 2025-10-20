from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
from . import config

def connect():
    # Remove existing default connection if it exists
    if connections.has_connection("default"):
        connections.remove_connection("default")
    
    # Now connect safely
    connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT)

def create_collection(collection_name=config.COLLECTION_NAME, dim=config.VECTOR_DIM):
    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        return Collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="emb", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description="Text embeddings collection")
    collection = Collection(name=collection_name, schema=schema, using='default')
    print(f"Created collection {collection_name}")
    return collection

def create_index(collection, field_name='emb', index_type='IVF_FLAT',
                 metric_type=config.METRIC_TYPE, params={"nlist": 1024}):
    index_params = {"index_type": index_type, "metric_type": metric_type, "params": params}
    collection.create_index(field_name, index_params)
    print("Index created.")
    return collection

def load_collection(collection):
    collection.load()
    print("Collection loaded into memory.")

def insert_batch(collection, ids, texts, embeddings):
    import numpy as np

    ids = np.array(ids, dtype=np.int64)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Insert as a list of columns in schema order: [id, text, emb]
    collection.insert([ids, texts, embeddings])
    collection.flush()  # persist insert
