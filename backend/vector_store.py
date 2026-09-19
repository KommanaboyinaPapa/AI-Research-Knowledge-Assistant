import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class VectorMetadata:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    page: int | None = None
    chunk_number: int = 0


class VectorStore:
    def __init__(self, embeddings: Any, metadata: list[VectorMetadata] | None = None):
        self.embeddings = embeddings
        self.metadata = metadata or []
        self._index = None
        try:
            import faiss
            import numpy as np

            matrix = np.asarray(embeddings, dtype="float32")
            if len(matrix):
                self._index = faiss.IndexFlatL2(matrix.shape[1])
                self._index.add(matrix)
        except ImportError:
            pass

    def search(self, query: list[float], top_k: int = 5) -> list[tuple[VectorMetadata, float]]:
        if not self.metadata:
            return []
        if self._index is not None:
            import numpy as np

            distances, indices = self._index.search(np.asarray([query], dtype="float32"), top_k)
            return [(self.metadata[i], float(distance)) for i, distance in zip(indices[0], distances[0]) if i >= 0]
        scored = []
        for item, vector in zip(self.metadata, self.embeddings):
            distance = sum((left - right) ** 2 for left, right in zip(query, vector))
            scored.append((item, distance))
        return sorted(scored, key=lambda item: item[1])[:top_k]

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "metadata.json").write_text(json.dumps([asdict(item) for item in self.metadata]), encoding="utf-8")
        (target / "embeddings.json").write_text(json.dumps(self.embeddings), encoding="utf-8")
        if self._index is not None:
            import faiss

            faiss.write_index(self._index, str(target / "index.faiss"))

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        source = Path(directory)
        metadata = [VectorMetadata(**item) for item in json.loads((source / "metadata.json").read_text(encoding="utf-8"))]
        embeddings = json.loads((source / "embeddings.json").read_text(encoding="utf-8"))
        store = cls(embeddings, metadata)
        index_path = source / "index.faiss"
        if index_path.exists():
            try:
                import faiss

                store._index = faiss.read_index(str(index_path))
            except ImportError:
                pass
        return store