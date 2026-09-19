import hashlib
import math
from typing import Any

MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """SentenceTransformer wrapper with a deterministic test fallback."""

    def __init__(self, model_name: str = MODEL_NAME, model: Any = None, dimensions: int = 384, offline: bool = False):
        self._model = model
        self.model_name = model_name
        self.dimensions = dimensions
        self.offline = offline

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dimensions = self._model.get_sentence_embedding_dimension()
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self.offline:
            return [self._fallback(text) for text in texts]
        if self._model is not None:
            return self._load().encode(texts, convert_to_numpy=True).tolist()
        try:
            return self._load().encode(texts, convert_to_numpy=True).tolist()
        except ImportError:
            return [self._fallback(text) for text in texts]

    def _fallback(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for token in text.lower().split():
            index = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimensions
            values[index] += 1.0
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]