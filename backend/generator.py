import os
import time
import json
import logging


FALLBACK = "I could not find the answer in the provided documents."
logger = logging.getLogger("rag")


def _grounded_prompt(question: str, results: list[dict]) -> str:
    """Build the same grounded prompt for regular and streaming answers."""
    context = "\n\n".join(
        f"[Source: {item['filename']}; document_id: {item['document_id']}; "
        f"page: {item['page'] or 'unavailable'}; chunk: {item['chunk_number']}]\n"
        f"{item['text']}"
        for item in results
    )
    return f"Answer only from the provided context. Cite relevant sources using filename, page when available, and chunk number. If the answer is not supported by the context, say exactly: {FALLBACK}\nQuestion: {question}\nContext:\n{context}"


def generate_answer(question: str, results: list[dict], history: list[dict] | None = None) -> dict:
    started = time.perf_counter()
    if not results:
        return {
            "answer": FALLBACK,
            "citations": [],
            "generation_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("missing_api_configuration")
        return {
            "answer": results[0]["text"],
            "citations": results[:3],
            "fallback": True,
            "generation_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    from google import genai

    logger.info("gemini_generation_started")
    prompt = _grounded_prompt(question, results)
    interaction = genai.Client(api_key=api_key).interactions.create(model="gemini-3.6-flash", input=prompt)
    logger.info("gemini_generation_completed")
    return {
        "answer": interaction.output_text,
        "citations": results[:3],
        "generation_latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def stream_answer(question: str, results: list[dict]):
    """Yield newline-delimited JSON events while Gemini generates an answer."""
    started = time.perf_counter()
    if not results:
        yield json.dumps({"type": "text", "text": FALLBACK})
        yield json.dumps({"type": "final", "citations": [], "generation_latency_ms": round((time.perf_counter() - started) * 1000, 2)})
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("missing_api_configuration")
        yield json.dumps({"type": "text", "text": results[0]["text"]})
        yield json.dumps({"type": "final", "citations": results[:3], "fallback": True, "generation_latency_ms": round((time.perf_counter() - started) * 1000, 2)})
        return

    from google import genai

    logger.info("gemini_generation_started")
    interaction_stream = genai.Client(api_key=api_key).interactions.create(
        model="gemini-3.6-flash",
        input=_grounded_prompt(question, results),
        stream=True,
    )
    for event in interaction_stream:
        delta = getattr(event, "delta", None)
        delta = getattr(delta, "delta", delta)
        text = getattr(delta, "text", None) or (delta if isinstance(delta, str) else None)
        if text:
            yield json.dumps({"type": "text", "text": text})
    logger.info("gemini_generation_completed")
    yield json.dumps({"type": "final", "citations": results[:3], "generation_latency_ms": round((time.perf_counter() - started) * 1000, 2)})