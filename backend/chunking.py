import re
from pathlib import Path


# Chunks make long documents easier and faster for a RAG system to search.
file_path = Path(__file__).parent.parent / "data" / "sample.txt"
text = file_path.read_text(encoding="utf-8").strip()

chunk_size = 300
overlap_size = 50

# Sentence-aware chunks keep ideas together instead of blindly cutting every 100 characters.
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

    # Overlap keeps context between chunks, which helps retrieval understand nearby ideas.
    overlap_length = 0
    overlap_count = 0
    for sentence in reversed(current_sentences[:-1]):
        added_length = len(sentence) + (1 if overlap_count else 0)
        if overlap_length + added_length > overlap_size:
            break
        overlap_length += added_length
        overlap_count += 1

    # If no short sentence fits, overlap one complete sentence rather than cutting one.
    if overlap_count == 0 and len(current_sentences) > 1:
        overlap_count = 1

    start = next_start - overlap_count

print("Sentence-aware text chunks:")
for chunk_number, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {chunk_number}:")
    print(chunk)
