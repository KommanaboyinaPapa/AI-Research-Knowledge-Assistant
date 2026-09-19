import re
import logging
import time
from dataclasses import dataclass

from .embeddings import EmbeddingModel
from .reranker import CrossEncoderReranker
from .vector_store import VectorMetadata, VectorStore


logger = logging.getLogger("rag")


@dataclass
class RetrievalConfig:
    top_k: int = 5
    max_distance: float = 2.0
    keyword_weight: float = 0.25


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: EmbeddingModel, reranker: CrossEncoderReranker | None = None, config: RetrievalConfig | None = None):
        self.store, self.embedder = store, embedder
        self.reranker = reranker or CrossEncoderReranker()
        self.config = config or RetrievalConfig()

    def semantic_search(self, query: str, limit: int) -> tuple[list[tuple[VectorMetadata, float]], float]:
        started = time.perf_counter()
        query_vector = self.embedder.encode([query])[0]
        candidates = self.store.search(query_vector, limit)
        return candidates, round((time.perf_counter() - started) * 1000, 2)

    def keyword_search(self, query: str, limit: int) -> tuple[list[tuple[VectorMetadata, float | None]], float]:
        started = time.perf_counter()
        words = set(re.findall(r"\w+", query.lower()))
        scored = []
        for metadata in self.store.metadata:
            score = sum(word in metadata.text.lower() for word in words)
            if score:
                scored.append((score, metadata))
        scored.sort(key=lambda item: item[0], reverse=True)
        candidates = [(metadata, None) for _, metadata in scored[:limit]]
        return candidates, round((time.perf_counter() - started) * 1000, 2)

    def search(self, query: str, top_k: int | None = None) -> dict:
        limit = top_k or self.config.top_k
        semantic_candidates, semantic_latency = self.semantic_search(query, limit * 2)
        keyword_candidates, keyword_latency = self.keyword_search(query, limit * 2)

        # Merge both candidate lists while keeping one copy of each chunk.
        candidates_by_id = {}
        faiss_candidates = []
        for rank, (metadata, distance) in enumerate(semantic_candidates, start=1):
            faiss_candidates.append({"rank": rank, "chunk_id": metadata.chunk_id, "distance": distance})
            if distance <= self.config.max_distance:
                candidates_by_id[metadata.chunk_id] = (metadata, distance)
        for metadata, distance in keyword_candidates:
            # A zero threshold means no candidate is eligible, including keyword matches.
            if self.config.max_distance > 0:
                candidates_by_id.setdefault(metadata.chunk_id, (metadata, distance))

        words = set(re.findall(r"\w+", query.lower()))
        results = []
        for metadata, distance in candidates_by_id.values():
            keyword_score = sum(word in metadata.text.lower() for word in words) / max(len(words), 1)
            results.append({"chunk_id": metadata.chunk_id, "document_id": metadata.document_id, "filename": metadata.filename, "text": metadata.text, "page": metadata.page, "chunk_number": metadata.chunk_number, "distance": distance, "keyword_score": keyword_score})
        reranked = self.reranker.rerank(query, results, limit)
        logger.info("reranking_completed", extra={"latency_ms": self.reranker.last_latency_ms, "count": len(reranked)})
        return {
            "results": reranked,
            "fallback": not reranked,
            "latency_ms": round(semantic_latency + keyword_latency + self.reranker.last_latency_ms, 2),
            "semantic_latency_ms": semantic_latency,
            "keyword_latency_ms": keyword_latency,
            "reranking_latency_ms": self.reranker.last_latency_ms,
            "faiss_candidates": faiss_candidates,
            "hybrid_candidate_count": len(candidates_by_id),
            "hybrid_candidate_ids": list(candidates_by_id),
            "final_selected_chunks": [item["chunk_id"] for item in reranked],
        }


def metadata_from_chunks(document_id: str, filename: str, chunks: list[str], pages: list[int | None] | None = None) -> list[VectorMetadata]:
    return [
        VectorMetadata(f"{document_id}-{number}", document_id, filename, text, pages[number] if pages else None, number + 1)
        for number, text in enumerate(chunks)
    ]


def compare_documents(
    query: str,
    stores: list[VectorStore],
    embedder: EmbeddingModel,
    document_ids: set[str],
    top_k: int = 3,
) -> dict[str, list[dict]]:
    """Retrieve separate result groups so selected documents can be compared."""
    comparison: dict[str, list[dict]] = {}
    shared_reranker = CrossEncoderReranker(model=False)
    for store in stores:
        if not store.metadata or store.metadata[0].document_id not in document_ids:
            continue
        result = HybridRetriever(
            store,
            embedder,
            shared_reranker,
            RetrievalConfig(top_k=top_k),
        ).search(query, top_k)
        filename = store.metadata[0].filename
        comparison[filename] = result["results"]
    return comparison