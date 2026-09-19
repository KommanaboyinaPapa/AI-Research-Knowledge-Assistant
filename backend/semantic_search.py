import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def main() -> None:
	# Create a small collection of documents for the retrieval demo.
	documents = [
		"Artificial intelligence is the field of building machines that perform tasks requiring human intelligence.",
		"Machine learning is a subset of artificial intelligence that learns patterns from data.",
		"Deep learning uses multi-layer neural networks to learn useful representations from large datasets.",
		"The solar system contains the Sun and the planets that orbit it.",
	]
	question = "What is a subset of artificial intelligence?"

	# Encode the documents and store their vectors in a FAISS L2 index.
	model = SentenceTransformer("all-MiniLM-L6-v2")
	document_embeddings = model.encode(documents)
	document_embeddings = np.asarray(document_embeddings, dtype="float32")
	index = faiss.IndexFlatL2(document_embeddings.shape[1])
	index.add(document_embeddings)

	# Encode the question and retrieve the two closest documents.
	question_embedding = model.encode([question])
	question_embedding = np.asarray(question_embedding, dtype="float32")
	distances, indices = index.search(question_embedding, 2)

	print(f"Question: {question}")
	print(f"Embedding shape: {question_embedding.shape}")
	print("Retrieved documents:")
	for rank, (document_index, distance) in enumerate(
		zip(indices[0], distances[0]), start=1
	):
		print(f"{rank}. {documents[document_index]}")
		print(f"   Distance: {distance:.4f}")


if __name__ == "__main__":
	main()
