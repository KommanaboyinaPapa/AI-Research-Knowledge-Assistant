import os
import re
from pathlib import Path

import faiss
import numpy as np
from google import genai
from sentence_transformers import SentenceTransformer


def load_chunks() -> list[str]:
    """Read sample.txt and split it into the project's sentence-aware chunks."""
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

        # Overlap keeps context between neighboring chunks for better retrieval.
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
    # Define the question that will be answered from the retrieved documents.
    question = "What is quantum computing?"
    chunks = load_chunks()

    # Use the same embedding model for documents and the question.
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = embedding_model.encode(chunks)
    question_embedding = embedding_model.encode([question])

    # FAISS uses 32-bit floating-point NumPy arrays for its vector index.
    chunk_embeddings = np.asarray(chunk_embeddings, dtype="float32")
    question_embedding = np.asarray(question_embedding, dtype="float32")

    # The index and the question must use vectors with the same dimensions.
    if chunk_embeddings.shape[1] != question_embedding.shape[1]:
        raise ValueError("Chunk and question embeddings have different dimensions.")

    # Store all chunk embeddings in a FAISS L2 distance index.
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)

    top_k = 3
    MAX_DISTANCE = 1.0

    # Retrieve the three chunks most similar to the question.
    distances, indices = index.search(question_embedding, top_k)

    # A distance threshold prevents unrelated chunks from being used as context.
    accepted_results = [
        (chunk_index, distance)
        for chunk_index, distance in zip(indices[0], distances[0])
        if distance <= MAX_DISTANCE
    ]

    # Do not call Gemini when FAISS found no sufficiently similar chunks.
    if not accepted_results:
        print("I could not find the answer in the provided documents.")
        return

    retrieved_chunks = []
    for chunk_index, distance in accepted_results:
        retrieved_chunks.append(
            f"Chunk {chunk_index + 1} (distance: {distance:.4f}):\n"
            f"{chunks[chunk_index]}"
        )
    context = "\n\n".join(retrieved_chunks)

    # Read the API key from the environment rather than hardcoding it.
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the GEMINI_API_KEY environment variable.")

    # Create a Gemini client and ask it to answer using only the retrieved context.
    client = genai.Client(api_key=api_key)
    prompt = f"""Answer the question using ONLY the provided context.
If the answer is not present in the context, say exactly:
I could not find the answer in the provided documents.

Question: {question}

Provided context:
{context}
"""
    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
    )

    print(f"User Question: {question}")
    print("\nRetrieved Chunks:")
    print(context)
    print("\nGenerated Answer:")
    print(interaction.output_text)


if __name__ == "__main__":
    main()
