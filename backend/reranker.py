import time
from typing import Any


CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL_NAME, model: Any = None):
        self.model_name = model_name
        self._model = model
        self.last_latency_ms = 0.0

    def rerank(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        started = time.perf_counter()
        if not results:
            self.last_latency_ms = 0.0
            return []
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except ImportError:
                self._model = False
        if self._model:
            scores = self._model.predict([(query, item["text"]) for item in results])
            for item, score in zip(results, scores):
                item["rerank_score"] = float(score)
        else:
            query_words = set(query.lower().split())
            for item in results:
                item["rerank_score"] = sum(word in item["text"].lower() for word in query_words)
        reranked = sorted(results, key=lambda item: item["rerank_score"], reverse=True)[:top_k]
        self.last_latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return reranked