import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

from .errors import RAGError

USERS_FILE = Path(os.getenv("RAG_USERS_FILE", Path(os.getenv("RAG_DATA_DIR", Path(__file__).parent.parent / "storage")) / "users.json"))
ITERATIONS = 310_000


def _users() -> list[dict]:
    if not USERS_FILE.exists():
        return []
    return json.loads(USERS_FILE.read_text(encoding="utf-8"))


def _save_users(users: list[dict]) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users), encoding="utf-8")


def _secret() -> str:
    secret = os.getenv("JWT_SECRET")
    if not secret:
        raise RAGError("missing_auth_configuration", "Authentication is not configured.", 500)
    return secret


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"{base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(digest).decode()}"


def verify_password(password: str, stored: str) -> bool:
    salt_text, digest_text = stored.split("$", 1)
    salt = base64.urlsafe_b64decode(salt_text.encode())
    expected = base64.urlsafe_b64decode(digest_text.encode())
    actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return hmac.compare_digest(actual, expected)


def _encode(payload: dict) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    def part(value: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode()).rstrip(b"=").decode()
    message = f"{part(header)}.{part(payload)}"
    signature = hmac.new(_secret().encode(), message.encode(), hashlib.sha256).digest()
    return f"{message}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"


def _decode(token: str) -> dict:
    try:
        header_text, payload_text, signature_text = token.split(".")
        message = f"{header_text}.{payload_text}"
        expected = hmac.new(_secret().encode(), message.encode(), hashlib.sha256).digest()
        actual = base64.urlsafe_b64decode(signature_text + "=" * (-len(signature_text) % 4))
        if not hmac.compare_digest(actual, expected):
            raise ValueError
        payload = json.loads(base64.urlsafe_b64decode(payload_text + "=" * (-len(payload_text) % 4)))
        if payload.get("exp", 0) < time.time():
            raise RAGError("invalid_token", "Your session has expired. Please log in again.", 401)
        return payload
    except RAGError:
        raise
    except Exception as error:
        raise RAGError("invalid_token", "Your login token is invalid.", 401) from error


def register_user(email: str, password: str) -> dict:
    email = email.strip().lower()
    if not email or len(password) < 8:
        raise RAGError("invalid_registration", "Use an email and a password with at least 8 characters.", 400)
    users = _users()
    if any(user["email"] == email for user in users):
        raise RAGError("user_exists", "An account with this email already exists.", 409)
    user = {"id": secrets.token_hex(16), "email": email, "password_hash": hash_password(password), "created_at": time.time()}
    users.append(user)
    _save_users(users)
    return {"id": user["id"], "email": user["email"]}


def login_user(email: str, password: str) -> dict:
    user = next((item for item in _users() if item["email"] == email.strip().lower()), None)
    if not user or not verify_password(password, user["password_hash"]):
        raise RAGError("invalid_credentials", "Email or password is incorrect.", 401)
    token = _encode({"sub": user["id"], "email": user["email"], "exp": time.time() + int(os.getenv("JWT_EXPIRY_MINUTES", "60")) * 60})
    return {"access_token": token, "token_type": "bearer", "user": {"id": user["id"], "email": user["email"]}}


def current_user(request) -> dict:
    authorization = request.headers.get("Authorization", "")
    if not authorization.lower().startswith("bearer "):
        raise RAGError("authentication_required", "Please log in to continue.", 401)
    payload = _decode(authorization[7:].strip())
    user = next((item for item in _users() if item["id"] == payload.get("sub")), None)
    if not user:
        raise RAGError("invalid_token", "Your login token is invalid.", 401)
    return {"id": user["id"], "email": user["email"]}
