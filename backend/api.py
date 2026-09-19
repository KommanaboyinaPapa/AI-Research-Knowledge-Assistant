import json
import hashlib
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from .chunker import ChunkConfig, chunk_text
from .embeddings import EmbeddingModel
from .evaluation import evaluate
from .extraction import extract_pages, extract_text
from .generator import generate_answer, stream_answer
from .errors import RAGError
from .auth import current_user, login_user, register_user
from .config import get_settings, validate_startup
from .logging_config import configure_logging
from .query_rewrite import rewrite_query
from .retrieval import HybridRetriever, compare_documents, metadata_from_chunks
from .vector_store import VectorStore

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import StreamingResponse
except ImportError:
    FastAPI = None


DATA_DIR = Path(os.getenv("RAG_DATA_DIR", Path(__file__).parent.parent / "storage"))
DOCUMENTS_FILE = DATA_DIR / "documents.json"
embedder = EmbeddingModel()
stores: list[VectorStore] = []
documents: list[dict] = []
logger = configure_logging()


def load_saved_state() -> None:
    """Restore document records and vector stores when the server starts."""
    if DOCUMENTS_FILE.exists():
        documents.extend(json.loads(DOCUMENTS_FILE.read_text(encoding="utf-8")))
    for document in documents:
        store_path = DATA_DIR / document["id"]
        if (store_path / "metadata.json").exists():
            stores.append(VectorStore.load(store_path))


def _retriever(owner_id: str | None = None) -> HybridRetriever:
    metadata, vectors = [], []
    for store in stores:
        if owner_id is not None:
            owner = next((item for item in documents if item["id"] == store.metadata[0].document_id), {}) if store.metadata else {}
            if owner.get("owner_id") != owner_id:
                continue
        metadata.extend(store.metadata)
        vectors.extend(store.embeddings)
    return HybridRetriever(VectorStore(vectors, metadata), embedder)


def create_app():
    if FastAPI is None:
        raise RuntimeError("Install requirements.txt to run the FastAPI server.")
    settings = validate_startup()
    app = FastAPI(title="AI Research Knowledge Assistant")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.post("/api/auth/register")
    async def register(payload: dict):
        return register_user(payload.get("email", ""), payload.get("password", ""))

    @app.post("/api/auth/login")
    async def login(payload: dict):
        return login_user(payload.get("email", ""), payload.get("password", ""))

    @app.get("/api/auth/me")
    async def me(request: Request):
        return current_user(request)

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        logger.info("request_received", extra={"request_id": request_id, "path": request.url.path, "method": request.method})
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_failed", extra={"request_id": request_id, "path": request.url.path})
            raise
        response.headers["X-Request-ID"] = request_id
        logger.info("request_completed", extra={"request_id": request_id, "path": request.url.path, "status_code": response.status_code, "latency_ms": round((time.perf_counter() - started) * 1000, 2)})
        return response

    @app.exception_handler(RAGError)
    async def rag_error_handler(request: Request, error: RAGError):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(error.error_type, extra={"request_id": request_id, "path": request.url.path})
        return JSONResponse(status_code=error.status_code, content={"error_type": error.error_type, "message": error.message, "request_id": request_id}, headers={"X-Request-ID": request_id})

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, error: HTTPException):
        request_id = getattr(request.state, "request_id", "unknown")
        error_type = "invalid_request" if error.status_code < 500 else "internal_error"
        return JSONResponse(status_code=error.status_code, content={"error_type": error_type, "message": str(error.detail), "request_id": request_id}, headers={"X-Request-ID": request_id})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, error: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception("unexpected_error", extra={"request_id": request_id, "path": request.url.path})
        return JSONResponse(status_code=500, content={"error_type": "internal_error", "message": "An unexpected server error occurred.", "request_id": request_id}, headers={"X-Request-ID": request_id})

    @app.get("/api/health")
    def health():
        return {"status": "ok", "ready": True, "environment": settings.environment, "documents": len(documents)}

    @app.get("/api/documents")
    def list_documents(request: Request):
        user = current_user(request)
        return [document for document in documents if document.get("owner_id") == user["id"]]

    @app.post("/api/documents/upload")
    async def upload_document(request: Request, file: UploadFile = File(...)):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        user = current_user(request)
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".pdf", ".docx", ".txt"}:
            raise RAGError("invalid_file_type", "Please upload a PDF, DOCX, or TXT file.", 400)
        if not content:
            raise RAGError("empty_document", "The uploaded document is empty.", 400)
        content_hash = hashlib.sha256(content).hexdigest()
        if any(document.get("content_hash") == content_hash and document.get("owner_id") == user["id"] for document in documents):
            raise HTTPException(status_code=409, detail="This document has already been uploaded.")
        document_id = str(uuid.uuid4())
        path = DATA_DIR / f"{document_id}{suffix}"
        path.write_bytes(content)
        try:
            page_texts = extract_pages(path)
            text = "\n".join(page_texts) if page_texts is not None else extract_text(path)
        except Exception as error:
            path.unlink(missing_ok=True)
            raise RAGError("corrupted_document", "The document could not be read. Please check the file and try again.", 400) from error
        if not text.strip():
            path.unlink(missing_ok=True)
            raise RAGError("empty_document", "The document does not contain readable text.", 400)
        chunks = chunk_text(text, ChunkConfig())
        try:
            vectors = embedder.encode(chunks)
        except Exception as error:
            path.unlink(missing_ok=True)
            raise RAGError("document_processing_failure", "The document could not be processed.", 500) from error
        pages = None
        if page_texts is not None:
            page_ends = []
            end = 0
            for page_text in page_texts:
                end += len(page_text) + 1
                page_ends.append(end)
            pages = []
            search_start = 0
            for chunk in chunks:
                position = text.find(chunk, search_start)
                page = next((number for number, page_end in enumerate(page_ends, start=1) if position < page_end), None)
                pages.append(page)
                search_start = max(search_start + 1, position + len(chunk))
        store = VectorStore(vectors, metadata_from_chunks(document_id, file.filename or path.name, chunks, pages))
        store.save(DATA_DIR / document_id)
        stores.append(store)
        record = {
            "id": document_id,
            "filename": file.filename,
            "file_type": path.suffix.lower().lstrip("."),
            "upload_timestamp": datetime.now(timezone.utc).isoformat(),
            "chunks": len(chunks),
            "status": "ready",
            "content_hash": content_hash,
            "owner_id": user["id"],
        }
        documents.append(record)
        DOCUMENTS_FILE.write_text(json.dumps(documents), encoding="utf-8")
        logger.info("document_processed", extra={"request_id": request.state.request_id, "document_id": document_id, "document_filename": file.filename, "count": len(chunks)})
        return record

    @app.delete("/api/documents/{document_id}")
    def delete_document(request: Request, document_id: str):
        user = current_user(request)
        matching = next((document for document in documents if document["id"] == document_id), None)
        if matching is None or matching.get("owner_id") != user["id"]:
            raise HTTPException(status_code=404, detail="Document not found.")
        documents.remove(matching)
        stores[:] = [store for store in stores if not store.metadata or store.metadata[0].document_id != document_id]
        for path in DATA_DIR.glob(f"{document_id}.*"):
            path.unlink(missing_ok=True)
        store_path = DATA_DIR / document_id
        for path in store_path.glob("*") if store_path.exists() else []:
            path.unlink(missing_ok=True)
        if store_path.exists():
            store_path.rmdir()
        DOCUMENTS_FILE.write_text(json.dumps(documents), encoding="utf-8")
        return {"deleted": document_id}

    @app.post("/api/chat")
    async def chat(request: Request, payload: dict):
        request_started = time.perf_counter()
        original_question = payload.get("question", "")
        history = payload.get("history", [])
        selected_documents = set(payload.get("document_ids", []))
        question = rewrite_query(original_question, history)
        user = current_user(request)
        try:
            retrieval = _retriever(user["id"]).search(question)
        except Exception as error:
            raise RAGError("retrieval_failure", "The documents could not be searched right now.", 500) from error
        logger.info("retrieval_completed", extra={"request_id": request.state.request_id, "latency_ms": retrieval["latency_ms"], "count": len(retrieval["results"])})
        if len(selected_documents) > 1:
            owned_ids = {item["id"] for item in documents if item.get("owner_id") == user["id"]}
            if not selected_documents.issubset(owned_ids):
                raise RAGError("document_access_denied", "You cannot access another user's documents.", 403)
            owned_stores = [store for store in stores if store.metadata and store.metadata[0].document_id in selected_documents]
            retrieval["comparison"] = compare_documents(question, owned_stores, embedder, selected_documents)
            retrieval["results"] = [
                item
                for result_group in retrieval["comparison"].values()
                for item in result_group
            ]
        elif selected_documents:
            owned_ids = {item["id"] for item in documents if item.get("owner_id") == user["id"]}
            if not selected_documents.issubset(owned_ids):
                raise RAGError("document_access_denied", "You cannot access another user's documents.", 403)
            retrieval["results"] = [item for item in retrieval["results"] if item["document_id"] in selected_documents]
        try:
            answer = generate_answer(original_question, retrieval["results"], history)
        except Exception as error:
            raise RAGError("generation_failure", "The answer could not be generated right now.", 502) from error
        total_latency = round((time.perf_counter() - request_started) * 1000, 2)
        return {
            **answer,
            "rewritten_question": question,
            "retrieval": retrieval,
            "retrieval_latency_ms": retrieval["latency_ms"],
            "total_latency_ms": total_latency,
            "latency_ms": total_latency,
        }

    @app.post("/api/chat/stream")
    async def chat_stream(request: Request, payload: dict):
        """Stream grounded answer events while keeping citations in the final event."""
        original_question = payload.get("question", "")
        history = payload.get("history", [])
        selected_documents = set(payload.get("document_ids", []))
        question = rewrite_query(original_question, history)
        user = current_user(request)
        try:
            retrieval = _retriever(user["id"]).search(question)
        except Exception as error:
            raise RAGError("retrieval_failure", "The documents could not be searched right now.", 500) from error
        if selected_documents:
            owned_ids = {item["id"] for item in documents if item.get("owner_id") == user["id"]}
            if not selected_documents.issubset(owned_ids):
                raise RAGError("document_access_denied", "You cannot access another user's documents.", 403)
            retrieval["results"] = [item for item in retrieval["results"] if item["document_id"] in selected_documents]

        def events():
            yield json.dumps({"type": "retrieval", "request_id": request.state.request_id, "retrieval": retrieval}) + "\n"
            try:
                for event in stream_answer(original_question, retrieval["results"]):
                    yield event + "\n"
            except Exception as error:
                logger.exception("streaming_failure", extra={"request_id": request.state.request_id})
                yield json.dumps({"type": "error", "error_type": "streaming_failure", "message": "The answer stream failed. Please try again.", "request_id": request.state.request_id}) + "\n"

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.get("/api/evaluation")
    def evaluation():
        return evaluate(_retriever())

    return app


load_saved_state()
app = create_app() if FastAPI is not None else None