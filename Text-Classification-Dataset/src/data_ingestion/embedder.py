from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str, normalize: bool = True) -> List[float]:
        """
        Embed a single text into a vector.

        Args:
            text (str): Input text to embed.
            normalize (bool): Whether to normalize the embedding vector.

        Returns:
            List[float]: Embedding vector.
        """
        vector = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        if normalize:
            norm = np.linalg.norm(vector)
            if norm == 0:
                norm = 1.0
            vector = vector / norm
        return vector.tolist()

    def embed_batch(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """
        Embed a batch of texts into vectors.

        Args:
            texts (List[str]): List of texts.
            normalize (bool): Whether to normalize the embedding vectors.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        return vectors.tolist()
