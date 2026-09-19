import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkConfig:
    max_chars: int = 300
    overlap_chars: int = 50
    overlap_sentences: int | None = None


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def chunk_text(text: str, config: ChunkConfig | None = None) -> list[str]:
    """Build chunks on sentence boundaries while retaining configurable overlap."""
    config = config or ChunkConfig()
    if config.max_chars < 1 or config.overlap_chars < 0:
        raise ValueError("max_chars must be positive and overlap_chars cannot be negative")
    sentences = split_sentences(text)
    chunks: list[str] = []
    start = 0
    while start < len(sentences):
        current: list[str] = []
        size = 0
        for sentence in sentences[start:]:
            added = len(sentence) + (1 if current else 0)
            if current and size + added > config.max_chars:
                break
            current.append(sentence)
            size += added
        chunks.append(" ".join(current))
        next_start = start + len(current)
        if next_start >= len(sentences):
            break

        overlap_count = 0
        overlap_length = 0
        for sentence in reversed(current[:-1]):
            added_length = len(sentence) + (1 if overlap_count else 0)
            if config.overlap_sentences is None and overlap_length + added_length > config.overlap_chars:
                break
            if config.overlap_sentences is not None and overlap_count >= config.overlap_sentences:
                break
            overlap_length += added_length
            overlap_count += 1

        # Keep one complete sentence when no sentence fits the overlap target.
        if overlap_count == 0 and len(current) > 1:
            overlap_count = 1
        start = max(start + 1, next_start - overlap_count)
    return chunks