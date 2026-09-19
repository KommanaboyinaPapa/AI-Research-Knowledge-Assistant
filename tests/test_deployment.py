import pytest


def test_production_settings_read_environment(monkeypatch, tmp_path):
    from backend.config import get_settings

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("PORT", "9123")
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://app.example.com")
    settings = get_settings()
    assert settings.environment == "production"
    assert settings.port == 9123
    assert settings.data_dir == tmp_path
    assert settings.cors_origins == ("https://app.example.com",)


def test_production_startup_accepts_configured_jwt(monkeypatch, tmp_path):
    from backend.config import get_settings, validate_startup

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("JWT_SECRET", "x" * 32)
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    settings = validate_startup(get_settings())
    assert settings.data_dir.exists()


def test_health_does_not_expose_configuration_secrets():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from backend.api import create_app

    response = TestClient(create_app()).get("/api/health")
    body = response.text
    assert response.status_code == 200
    assert "JWT_SECRET" not in body
    assert "GEMINI_API_KEY" not in body