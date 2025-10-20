from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_ingestion import config
from typing import List, Optional

class AdvancedKNNSearcher:
    def __init__(self,
                 milvus_host: str = config.MILVUS_HOST,
                 milvus_port: int = config.MILVUS_PORT,
                 collection_name: str = config.COLLECTION_NAME,
                 embed_model_name: str = config.EMBED_MODEL_NAME,
                 vector_dim: int = config.VECTOR_DIM,
                 metric_type: str = config.METRIC_TYPE,
                 nprobe: int = 10):
        # Milvus connection
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        self.metric_type = metric_type
        self.nprobe = nprobe
        self.vector_dim = vector_dim

        connections.connect('default', host=self.milvus_host, port=self.milvus_port)
        self.collection = Collection(self.collection_name)
        self.collection.load()

        # Load embedding model
        self.model = SentenceTransformer(embed_model_name)

    def embed_text(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        return vectors

    def search(self,
               query_texts: List[str],
               top_k: int = 5,
               metric_type: Optional[str] = None,
               nprobe: Optional[int] = None,
               filter_expr: Optional[str] = None):
        """
        Advanced KNN search.

        Parameters:
            query_texts: List of query strings
            top_k: Number of top similar results to return
            metric_type: Optional metric override ('IP', 'L2', 'COSINE')
            nprobe: Optional override of nprobe
            filter_expr: Optional Milvus filter expression (e.g., "id > 1000")
        """
        nprobe = nprobe or self.nprobe
        metric_type = metric_type or self.metric_type

        vectors = self.embed_text(query_texts)
        results = self.collection.search(
            data=vectors.tolist(),
            anns_field="emb",
            param={"metric_type": metric_type, "params": {"nprobe": nprobe}},
            limit=top_k,
            output_fields=["id", "text"],
            expr=filter_expr
        )
        return results  # List of results per query

if __name__ == "__main__":
    searcher = AdvancedKNNSearcher()

    # Interactive inputs
    query_input = input("\nEnter one or multiple texts (comma-separated) to search: ")
    query_texts = [q.strip() for q in query_input.split(",") if q.strip()]
    top_k = int(input("Enter number of similar texts: "))
    metric_type = input("Enter metric type (IP / L2 / COSINE) [default IP]: ").strip() or None
    nprobe_input = input("Enter nprobe (press Enter to use default 10): ").strip()
    nprobe = int(nprobe_input) if nprobe_input else None
    filter_expr = input("Enter filter expression (optional, e.g., 'id > 1000'): ").strip() or None

    results_batch = searcher.search(query_texts, top_k=top_k, metric_type=metric_type, nprobe=nprobe, filter_expr=filter_expr)

    # Print results
    for i, res_list in enumerate(results_batch):
        print(f"\nQuery [{i+1}]: {query_texts[i]}")
        for res in res_list:
            print(f"ID: {res.id} | Score: {res.score:.4f} | Text: {res.text}")
