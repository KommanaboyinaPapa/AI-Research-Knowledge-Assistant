import json

import pytest

from backend.query_rewrite import rewrite_query
from backend.config import get_settings, validate_startup


def auth_headers(client, email="test@example.com"):
    client.post("/api/auth/register", json={"email": email, "password": "password123"})
    token = client.post("/api/auth/login", json={"email": email, "password": "password123"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_follow_up_query_rewrite_uses_previous_user_question():
    history = [{"role": "user", "content": "What is deep learning?"}]
    rewritten = rewrite_query("How does it work?", history)
    assert "What is deep learning?" in rewritten
    assert "How does it work?" in rewritten


def test_standalone_query_is_not_rewritten():
    assert rewrite_query("What is machine learning?", []) == "What is machine learning?"


def test_health_endpoint():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from backend.api import create_app

    response = TestClient(create_app()).get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["ready"] is True


def test_production_startup_requires_strong_jwt_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    with pytest.raises(Exception) as error:
        validate_startup(get_settings())
    assert getattr(error.value, "error_type", "") == "missing_auth_configuration"


def test_cors_allows_configured_origin_and_rejects_other_origin(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import backend.api as api

    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://app.example.com")
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    client = TestClient(api.create_app())
    allowed = client.options("/api/health", headers={"Origin": "https://app.example.com", "Access-Control-Request-Method": "GET"})
    blocked = client.options("/api/health", headers={"Origin": "https://other.example.com", "Access-Control-Request-Method": "GET"})
    assert allowed.headers.get("access-control-allow-origin") == "https://app.example.com"
    assert "access-control-allow-origin" not in blocked.headers


def test_registration_login_me_and_protected_documents(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import backend.api as api
    import backend.auth as auth

    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users.json")
    monkeypatch.setattr(api, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(api, "DOCUMENTS_FILE", tmp_path / "data" / "documents.json")
    monkeypatch.setattr(api, "stores", [])
    monkeypatch.setattr(api, "documents", [])
    response = TestClient(api.create_app()).post("/api/auth/register", json={"email": "a@example.com", "password": "password123"})
    assert response.status_code == 200
    login = TestClient(api.create_app()).post("/api/auth/login", json={"email": "a@example.com", "password": "password123"})
    assert login.status_code == 200
    token = login.json()["access_token"]
    client = TestClient(api.create_app())
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    assert client.get("/api/documents").status_code == 401
    assert client.get("/api/documents", headers={"Authorization": f"Bearer {token}"}).json() == []


def test_invalid_credentials_are_rejected(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import backend.api as api
    import backend.auth as auth

    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users.json")
    client = TestClient(api.create_app())
    client.post("/api/auth/register", json={"email": "b@example.com", "password": "password123"})
    response = client.post("/api/auth/login", json={"email": "b@example.com", "password": "wrongpass"})
    assert response.status_code == 401
    assert response.json()["error_type"] == "invalid_credentials"


def test_users_cannot_access_each_others_documents(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import backend.api as api
    import backend.auth as auth
    from backend.embeddings import EmbeddingModel
    import backend.retrieval as retrieval
    from backend.reranker import CrossEncoderReranker

    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users.json")
    monkeypatch.setattr(api, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(api, "DOCUMENTS_FILE", tmp_path / "data" / "documents.json")
    monkeypatch.setattr(api, "stores", [])
    monkeypatch.setattr(api, "documents", [])
    monkeypatch.setattr(api, "embedder", EmbeddingModel(offline=True))
    monkeypatch.setattr(retrieval, "CrossEncoderReranker", lambda: CrossEncoderReranker(model=False))
    client = TestClient(api.create_app())
    client.post("/api/auth/register", json={"email": "owner@example.com", "password": "password123"})
    client.post("/api/auth/register", json={"email": "other@example.com", "password": "password123"})
    owner_token = client.post("/api/auth/login", json={"email": "owner@example.com", "password": "password123"}).json()["access_token"]
    other_token = client.post("/api/auth/login", json={"email": "other@example.com", "password": "password123"}).json()["access_token"]
    upload = client.post("/api/documents/upload", headers={"Authorization": f"Bearer {owner_token}"}, files={"file": ("private.txt", b"private notes", "text/plain")})
    document_id = upload.json()["id"]
    assert client.get("/api/documents", headers={"Authorization": f"Bearer {other_token}"}).json() == []
    forbidden_chat = client.post("/api/chat", headers={"Authorization": f"Bearer {other_token}"}, json={"question": "test", "document_ids": [document_id]})
    assert forbidden_chat.status_code == 403
    assert client.delete(f"/api/documents/{document_id}", headers={"Authorization": f"Bearer {other_token}"}).status_code == 404


def test_upload_list_and_chat_endpoints(monkeypatch, tmp_path):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import backend.api as api
    import backend.retrieval as retrieval
    from backend.embeddings import EmbeddingModel
    from backend.reranker import CrossEncoderReranker

    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api, "DOCUMENTS_FILE", tmp_path / "documents.json")
    monkeypatch.setattr(api, "stores", [])
    monkeypatch.setattr(api, "documents", [])
    monkeypatch.setattr(api, "embedder", EmbeddingModel(offline=True))
    monkeypatch.setattr(retrieval, "CrossEncoderReranker", lambda: CrossEncoderReranker(model=False))
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    import backend.auth as auth
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users.json")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    client = TestClient(api.create_app())
    headers = auth_headers(client)
    upload = client.post(
        "/api/documents/upload",
        headers=headers, files={"file": ("notes.txt", b"Python is useful for data science.", "text/plain")},
    )
    assert upload.status_code == 200
    document = upload.json()
    assert document["filename"] == "notes.txt"
    assert client.get("/api/documents", headers=headers).json()[0]["id"] == document["id"]

    chat = client.post("/api/chat", headers=headers, json={"question": "What is Python?"})
    assert chat.status_code == 200
    assert chat.json()["citations"][0]["filename"] == "notes.txt"
    assert chat.json()["citations"][0]["chunk_number"] == 1


def test_document_persistence_duplicate_prevention_deletion_and_reload(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import backend.api as api
    import backend.retrieval as retrieval
    from backend.embeddings import EmbeddingModel
    from backend.reranker import CrossEncoderReranker

    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api, "DOCUMENTS_FILE", tmp_path / "documents.json")
    monkeypatch.setattr(api, "stores", [])
    monkeypatch.setattr(api, "documents", [])
    monkeypatch.setattr(api, "embedder", EmbeddingModel(offline=True))
    monkeypatch.setattr(retrieval, "CrossEncoderReranker", lambda: CrossEncoderReranker(model=False))
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    import backend.auth as auth
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users.json")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    client = TestClient(api.create_app())
    headers = auth_headers(client)
    upload = client.post("/api/documents/upload", headers=headers, files={"file": ("persistent.txt", b"Python is useful.", "text/plain")})
    document = upload.json()
    assert document["status"] == "ready"
    assert document["file_type"] == "txt"
    assert document["upload_timestamp"]
    assert (tmp_path / document["id"] / "index.faiss").exists()

    duplicate = client.post("/api/documents/upload", headers=headers, files={"file": ("copy.txt", b"Python is useful.", "text/plain")})
    assert duplicate.status_code == 409

    saved_documents = list(api.documents)
    saved_stores = list(api.stores)
    api.documents.clear()
    api.stores.clear()
    api.load_saved_state()
    assert api.documents == saved_documents
    assert api.stores[0].metadata[0].filename == "persistent.txt"

    deleted = client.delete(f"/api/documents/{document['id']}", headers=headers)
    assert deleted.status_code == 200
    assert client.get("/api/documents", headers=headers).json() == []
    assert not (tmp_path / document["id"]).exists()


def test_streaming_chat_endpoint_returns_progressive_events(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import backend.api as api
    import backend.retrieval as retrieval
    from backend.embeddings import EmbeddingModel
    from backend.reranker import CrossEncoderReranker
    from backend.vector_store import VectorStore
    from backend.retrieval import metadata_from_chunks

    monkeypatch.setattr(api, "stores", [VectorStore(EmbeddingModel(offline=True).encode(["Python is useful."]), metadata_from_chunks("1", "notes.txt", ["Python is useful."]))])
    monkeypatch.setattr(api, "embedder", EmbeddingModel(offline=True))
    monkeypatch.setattr(retrieval, "CrossEncoderReranker", lambda: CrossEncoderReranker(model=False))
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    import backend.auth as auth
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users-stream.json")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = TestClient(api.create_app())
    headers = auth_headers(client)
    user_id = client.get("/api/auth/me", headers=headers).json()["id"]
    monkeypatch.setattr(api, "documents", [{"id": "1", "owner_id": user_id}])
    response = client.post("/api/chat/stream", headers=headers, json={"question": "What is Python?"})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["type"] == "retrieval"
    assert any(event["type"] == "text" for event in events)
    assert events[-1]["type"] == "final"
    assert events[-1]["citations"][0]["filename"] == "notes.txt"


def test_streaming_chat_endpoint_reports_generator_errors(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import backend.api as api
    import backend.retrieval as retrieval
    from backend.embeddings import EmbeddingModel
    from backend.reranker import CrossEncoderReranker
    from backend.vector_store import VectorStore
    from backend.retrieval import metadata_from_chunks

    monkeypatch.setattr(api, "stores", [VectorStore(EmbeddingModel(offline=True).encode(["Python is useful."]), metadata_from_chunks("1", "notes.txt", ["Python is useful."]))])
    monkeypatch.setattr(api, "embedder", EmbeddingModel(offline=True))
    monkeypatch.setattr(retrieval, "CrossEncoderReranker", lambda: CrossEncoderReranker(model=False))
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    import backend.auth as auth
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users-error.json")

    def failed_stream(question, results):
        raise RuntimeError("test failure")
        yield

    monkeypatch.setattr(api, "stream_answer", failed_stream)
    client = TestClient(api.create_app())
    headers = auth_headers(client)
    user_id = client.get("/api/auth/me", headers=headers).json()["id"]
    monkeypatch.setattr(api, "documents", [{"id": "1", "owner_id": user_id}])
    response = client.post("/api/chat/stream", headers=headers, json={"question": "What is Python?"})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]["type"] == "error"
    assert events[-1]["message"] == "The answer stream failed. Please try again."
    assert events[-1]["request_id"]


def test_invalid_upload_returns_clean_error_with_request_id(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import backend.api as api

    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api, "DOCUMENTS_FILE", tmp_path / "documents.json")
    monkeypatch.setattr(api, "stores", [])
    monkeypatch.setattr(api, "documents", [])
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    import backend.auth as auth
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users.json")
    client = TestClient(api.create_app())
    headers = auth_headers(client)
    response = client.post("/api/documents/upload", headers=headers, files={"file": ("bad.exe", b"x", "application/octet-stream")})
    assert response.status_code == 400
    assert response.json()["error_type"] == "invalid_file_type"
    assert response.json()["request_id"] == response.headers["X-Request-ID"]

    empty = client.post("/api/documents/upload", headers=headers, files={"file": ("empty.txt", b"", "text/plain")})
    assert empty.status_code == 400
    assert empty.json()["error_type"] == "empty_document"


def test_corrupted_document_returns_clean_error(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import backend.api as api

    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api, "DOCUMENTS_FILE", tmp_path / "documents.json")
    monkeypatch.setattr(api, "stores", [])
    monkeypatch.setattr(api, "documents", [])
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    import backend.auth as auth
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users.json")
    client = TestClient(api.create_app())
    response = client.post("/api/documents/upload", headers=auth_headers(client), files={"file": ("bad.pdf", b"not a pdf", "application/pdf")})
    assert response.status_code == 400
    assert response.json()["error_type"] == "corrupted_document"


def test_retrieval_and_generation_errors_are_classified(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import backend.api as api
    import backend.auth as auth

    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users-errors.json")
    client = TestClient(api.create_app())
    headers = auth_headers(client, "errors@example.com")
    monkeypatch.setattr(api, "_retriever", lambda: (_ for _ in ()).throw(RuntimeError("search failed")))
    response = client.post("/api/chat", headers=headers, json={"question": "test"})
    assert response.status_code == 500
    assert response.json()["error_type"] == "retrieval_failure"

    monkeypatch.setattr(api, "_retriever", lambda owner_id=None: type("Retriever", (), {"search": lambda self, query: {"results": [{"text": "answer", "filename": "a.txt"}], "latency_ms": 1}})())
    monkeypatch.setattr(api, "generate_answer", lambda *args: (_ for _ in ()).throw(RuntimeError("generation failed")))
    response = client.post("/api/chat", headers=headers, json={"question": "test"})
    assert response.status_code == 502
    assert response.json()["error_type"] == "generation_failure"