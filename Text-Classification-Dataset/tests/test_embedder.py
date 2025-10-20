import pytest
import src.data_ingestion.embedder as embedder
import src.data_ingestion.config as config

def test_embedder_dim():
    model = embedder.Embedder(config.EMBED_MODEL_NAME)
    vectors = model.embed_batch(["hello world", "pytest check"], normalize=True)
    assert len(vectors) == 2
    assert len(vectors[0]) == config.VECTOR_DIM
