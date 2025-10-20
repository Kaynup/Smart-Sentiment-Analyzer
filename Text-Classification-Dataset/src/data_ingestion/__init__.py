"""
Data Ingestion Package for Milvus:
- Load text from CSVs
- Preprocess and embed with sentence-transformers
- Insert into Milvus
- Cluster embeddings with KMeans and HDBSCAN
"""

__version__ = "0.1.0"

from .config import COLLECTION_NAME, CSV_FILES, EMBED_MODEL_NAME
from .pipeline import run_ingestion
