import json

from backend.chunker import ChunkConfig, chunk_text
from backend.embeddings import EmbeddingModel
from backend.evaluation import evaluate
from backend.generator import FALLBACK, generate_answer
from backend.generator import stream_answer
from backend.reranker import CrossEncoderReranker
from backend.retrieval import HybridRetriever, RetrievalConfig, compare_documents, metadata_from_chunks
from backend.vector_store import VectorStore


def make_retriever(threshold=2.0):
    texts = ["Python is useful for data science.", "The solar system has planets."]
    embedder = EmbeddingModel(offline=True)
    store = VectorStore(embedder.encode(texts), metadata_from_chunks("1", "sample.txt", texts))
    # Keep tests offline; production can load the CrossEncoder model normally.
    return HybridRetriever(store, embedder, CrossEncoderReranker(model=False), RetrievalConfig(max_distance=threshold))


def test_sentence_aware_configurable_chunking():
    chunks = chunk_text("One sentence. Two sentence. Three sentence.", ChunkConfig(max_chars=20, overlap_sentences=1))
    assert all(chunk.endswith(".") for chunk in chunks)
    assert len(chunks) == 3


def test_embeddings_have_stable_shape():
    vectors = EmbeddingModel(offline=True).encode(["hello", "world"])
    assert len(vectors) == 2 and len(vectors[0]) == 384


def test_retrieval_and_reranking():
    result = make_retriever().search("What is Python used for?")
    assert result["results"][0]["filename"] == "sample.txt"
    assert "rerank_score" in result["results"][0]
    assert result["results"][0]["rerank_score"] >= result["results"][1]["rerank_score"]
    assert result["reranking_latency_ms"] >= 0
    assert result["faiss_candidates"][0]["rank"] == 1


def test_cross_encoder_reranking_orders_candidates_by_score():
    class FakeCrossEncoder:
        def predict(self, pairs):
            return [0.2, 0.9]

    candidates = [{"text": "first"}, {"text": "second"}]
    reranked = CrossEncoderReranker(model=FakeCrossEncoder()).rerank("question", candidates, 2)
    assert [item["text"] for item in reranked] == ["second", "first"]


def test_hybrid_search_merges_semantic_and_keyword_candidates_without_duplicates():
    retriever = make_retriever()
    semantic, _ = retriever.semantic_search("Python", 2)
    keyword, _ = retriever.keyword_search("Python", 2)
    result = retriever.search("Python", 2)
    assert semantic
    assert keyword
    assert result["hybrid_candidate_count"] == len(set(result["hybrid_candidate_ids"]))
    assert result["semantic_latency_ms"] >= 0
    assert result["keyword_latency_ms"] >= 0


def test_threshold_fallback():
    result = make_retriever(threshold=0).search("What is Python used for?")
    assert result["fallback"] is True


def test_store_json_round_trip(tmp_path):
    embedder = EmbeddingModel(offline=True)
    store = VectorStore(embedder.encode(["hello"]), metadata_from_chunks("1", "a.txt", ["hello"]))
    store.save(tmp_path)
    assert (tmp_path / "index.faiss").exists()
    loaded = VectorStore.load(tmp_path)
    assert loaded.metadata[0].text == "hello"
    assert loaded.search(store.embeddings[0], 1)[0][0].filename == "a.txt"


def test_metadata_includes_chunk_number_and_survives_retrieval():
    embedder = EmbeddingModel(offline=True)
    store = VectorStore(embedder.encode(["Python supports data science."]), metadata_from_chunks("doc-1", "notes.txt", ["Python supports data science."]))
    assert store.metadata[0].chunk_number == 1
    result = HybridRetriever(store, embedder, CrossEncoderReranker(model=False)).search("Python", 1)
    assert result["results"][0]["document_id"] == "doc-1"
    assert result["results"][0]["chunk_number"] == 1


def test_metadata_preserves_pdf_page_number():
    metadata = metadata_from_chunks("doc-2", "research.pdf", ["page text"], [4])
    assert metadata[0].filename == "research.pdf"
    assert metadata[0].page == 4
    assert metadata[0].chunk_number == 1


def test_multi_document_comparison_returns_separate_groups():
    embedder = EmbeddingModel(offline=True)
    first = VectorStore(embedder.encode(["Python supports data science."]), metadata_from_chunks("1", "python.txt", ["Python supports data science."]))
    second = VectorStore(embedder.encode(["Planets orbit the Sun."]), metadata_from_chunks("2", "space.txt", ["Planets orbit the Sun."]))
    comparison = compare_documents("What is Python?", [first, second], embedder, {"1", "2"})
    assert set(comparison) == {"python.txt", "space.txt"}
    assert comparison["python.txt"][0]["filename"] == "python.txt"


def test_generation_fallback_reports_latency():
    result = generate_answer("What is Python?", [])
    assert result["answer"] == FALLBACK
    assert result["generation_latency_ms"] >= 0


def test_streaming_generation_is_grounded_and_progressive(monkeypatch):
    import google

    captured = {}

    class FakeEvent:
        def __init__(self, text):
            self.delta = type("Step", (), {"delta": type("Text", (), {"text": text})()})()

    class FakeInteractions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return [FakeEvent("Deep "), FakeEvent("learning uses layers.")]

    class FakeClient:
        interactions = FakeInteractions()

    monkeypatch.setattr(google, "genai", type("GenAI", (), {"Client": lambda api_key: FakeClient()}), raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    results = [{"filename": "notes.txt", "document_id": "1", "page": None, "chunk_number": 1, "text": "Deep learning uses neural networks."}]
    events = [json.loads(event) for event in stream_answer("What is deep learning?", results)]
    assert [event["text"] for event in events if event["type"] == "text"] == ["Deep ", "learning uses layers."]
    assert "notes.txt" in captured["input"]
    assert "Answer only from the provided context" in captured["input"]


def test_evaluation_reports_each_question_and_latency():
    result = evaluate(make_retriever(), [("What is Python?", "Python")], k=1)
    assert len(result["results"]) == 1
    assert result["results"][0]["hit"] is True
    assert result["results"][0]["latency_ms"] >= 0