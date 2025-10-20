import pytest
from pymilvus import connections, utility
import src.data_ingestion.config as config

def test_milvus_connection():
    connections.connect(host="localhost", port=config.MILVUS_PORT)
    version = utility.get_server_version()
    assert version.startswith("2."), f"Unexpected version: {version}"
