import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_chunks() -> list[str]:
    """Read sample.txt and split it into sentence-aware chunks."""
    file_path = Path(__file__).parent.parent / "data" / "sample.txt"
    text = file_path.read_text(encoding="utf-8").strip()

    chunk_size = 300
    overlap_size = 50
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    start = 0

    while start < len(sentences):
        current_sentences = []
        current_length = 0

        # Add complete sentences until the chunk is approximately 300 characters.
        for sentence in sentences[start:]:
            added_length = len(sentence) + (1 if current_sentences else 0)
            if current_sentences and current_length + added_length > chunk_size:
                break
            current_sentences.append(sentence)
            current_length += added_length

        chunks.append(" ".join(current_sentences))
        next_start = start + len(current_sentences)

        # Overlap keeps context between chunks for better retrieval.
        overlap_length = 0
        overlap_count = 0
        for sentence in reversed(current_sentences[:-1]):
            added_length = len(sentence) + (1 if overlap_count else 0)
            if overlap_length + added_length > overlap_size:
                break
            overlap_length += added_length
            overlap_count += 1

        # Reuse one complete sentence if no shorter sentence fits the target overlap.
        if overlap_count == 0 and len(current_sentences) > 1:
            overlap_count = 1

        start = next_start - overlap_count

    return chunks


def main() -> None:
    chunks = load_chunks()
    question = "What is deep learning?"

    # Use one model to create embeddings for both chunks and the question.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = model.encode(chunks)
    question_embedding = model.encode([question])

    # FAISS expects vectors as 32-bit floating-point NumPy arrays.
    chunk_embeddings = np.asarray(chunk_embeddings, dtype="float32")
    question_embedding = np.asarray(question_embedding, dtype="float32")

    # Both embedding types must have the same number of dimensions.
    if chunk_embeddings.shape[1] != question_embedding.shape[1]:
        raise ValueError("Chunk and question embeddings have different dimensions.")

    # Add every chunk embedding to a FAISS L2 distance index.
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)

    # Find the three chunks closest to the question embedding.
    distances, indices = index.search(question_embedding, 3)

    print(f"Question: {question}")
    print(f"Embedding shape: {question_embedding.shape}")
    print("\nTop 3 relevant chunks:")
    for rank, (chunk_index, distance) in enumerate(
        zip(indices[0], distances[0]), start=1
    ):
        print(f"\nResult {rank}")
        print(f"Chunk number: {chunk_index + 1}")
        print(f"Chunk text: {chunks[chunk_index]}")
        print(f"Distance: {distance:.4f}")


if __name__ == "__main__":
    main()
