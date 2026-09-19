import os
from dataclasses import dataclass, field
from pathlib import Path

from .errors import RAGError


@dataclass(frozen=True)
class Settings:
    environment: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", os.getenv("RAG_DATA_DIR", "storage"))))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(origin.strip() for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173").split(",") if origin.strip()))
    jwt_secret: str = field(default_factory=lambda: os.getenv("JWT_SECRET", ""))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))


def get_settings() -> Settings:
    return Settings()


def validate_startup(settings: Settings | None = None) -> Settings:
    settings = settings or get_settings()
    if settings.environment.lower() == "production" and len(settings.jwt_secret) < 32:
        raise RAGError("missing_auth_configuration", "JWT_SECRET must be configured with at least 32 characters in production.", 500)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
