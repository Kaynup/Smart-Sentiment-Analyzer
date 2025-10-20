import pytest
import src.data_ingestion.milvus_client as milvus
from pymilvus import utility, Collection
import src.data_ingestion.embedder as embedder
import src.data_ingestion.config as config

@pytest.fixture(scope="module")
def temp_collection_name():
    return "test_collection"

def test_ingest_small(temp_collection_name):
    milvus.connect()

    # Ensure fresh collection
    if utility.has_collection(temp_collection_name):
        utility.drop_collection(temp_collection_name)

    collection = milvus.create_collection(collection_name=temp_collection_name, dim=config.VECTOR_DIM)
    texts = ["pytest sample one", "pytest sample two"]
    ids = [1, 2]

    model = embedder.Embedder(config.EMBED_MODEL_NAME)
    embeddings = model.embed_batch(texts)

    milvus.insert_batch(collection, ids, texts, embeddings)
    milvus.create_index(collection)
    milvus.load_collection(collection)

    assert collection.num_entities == 2
