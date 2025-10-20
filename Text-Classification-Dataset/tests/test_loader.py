import pytest
import src.data_ingestion.loader as loader
import src.data_ingestion.config as config

def test_stream_texts():
    gen = loader.stream_texts(config.CSV_FILES, text_col=config.TEXT_COLUMN, chunksize=100)
    texts = [next(gen) for _ in range(5)]  # take 5 rows
    assert all(isinstance(t, str) for t in texts)
    assert len(texts) == 5
