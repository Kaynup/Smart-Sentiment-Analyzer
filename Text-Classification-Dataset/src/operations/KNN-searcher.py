from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_ingestion import config

class SimilarTextSearcher:
    def __init__(self,
                 milvus_host=config.MILVUS_HOST,
                 milvus_port=config.MILVUS_PORT,
                 collection_name=config.COLLECTION_NAME,
                 embed_model_name=config.EMBED_MODEL_NAME,
                 vector_dim=config.VECTOR_DIM,
                 metric_type=config.METRIC_TYPE,
                 nprobe=10):
        # Milvus connection
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        self.metric_type = metric_type
        self.nprobe = nprobe
        self.vector_dim = vector_dim

        # Connect to Milvus
        connections.connect('default', host=self.milvus_host, port=self.milvus_port)
        self.collection = Collection(self.collection_name)
        self.collection.load()

        # Load embedding model
        self.model = SentenceTransformer(embed_model_name)

    def embed_text(self, texts, normalize=True):
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        return vectors

    def search(self, query_text, top_k=5, nprobe=None):
        nprobe = nprobe or self.nprobe
        query_vector = self.embed_text([query_text])[0].tolist()
        results = self.collection.search(
            data=[query_vector],
            anns_field="emb",
            param={"metric_type": self.metric_type, "params": {"nprobe": nprobe}},
            limit=top_k,
            output_fields=["id", "text"]
        )
        return results[0]  # Only one query, so return first result set

if __name__ == "__main__":
    searcher = SimilarTextSearcher()

    query_text = input("\nEnter text to get near-similar texts: ")
    top_k = int(input("Enter number of similar texts: "))
    nprobe = input("Enter nprobe (press Enter to use default 10): ")
    nprobe = int(nprobe) if nprobe.strip() else None

    results = searcher.search(query_text, top_k=top_k, nprobe=nprobe)
    print("\nTop similar texts:\n")
    for res in results:
        print(f"ID: {res.id} | Score: {res.score:.4f} | Text: {res.text}")
