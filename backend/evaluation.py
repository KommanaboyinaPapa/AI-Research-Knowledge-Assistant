import time


DEFAULT_QUESTIONS = [
    ("What is artificial intelligence?", "Artificial intelligence"),
    ("What is machine learning?", "Machine learning"),
    ("What does deep learning use?", "Deep learning"),
    ("What is NLP used for?", "Natural language processing"),
    ("What can computer vision identify?", "Computer vision"),
    ("Why is Python popular?", "Python"),
    ("What does data science combine?", "Data science"),
    ("Why is good data important?", "Good data"),
    ("What does RAG find?", "retrieval-augmented"),
    ("What helps search a large collection?", "chunks"),
]


def evaluate(retriever, questions=None, k: int = 3) -> dict:
    questions = questions or DEFAULT_QUESTIONS
    hits = 0
    latencies = []
    question_results = []
    for question, expected in questions:
        started = time.perf_counter()
        result = retriever.search(question, k)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        latencies.append(latency_ms)
        hit = any(expected.lower() in item["text"].lower() for item in result["results"])
        hits += hit
        question_results.append({"question": question, "expected": expected, "hit": hit, "latency_ms": latency_ms})
    return {
        "questions": len(questions),
        "k": k,
        "recall_at_k": hits / len(questions) if questions else 0,
        "hit_rate": hits / len(questions) if questions else 0,
        "average_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "results": question_results,
    }