# Deployment

## GitHub preparation

Push the source tree, including `backend/`, `frontend/`, `tests/`, `requirements.txt`, and `render.yaml`. Do not commit `.env`, virtual environments, `node_modules/`, `frontend/dist/`, uploaded documents, or generated JSON/FAISS files under `storage/`. The included `.gitignore` covers these local runtime artifacts. Review staged files before the first push.

The repository is not deployed by this document. Create a GitHub repository and push the project using your normal Git workflow.

## Environment variables

Set these in the backend service environment. Never put secret values in source control or in `.env.example`.

- `APP_ENV=production`
- `JWT_SECRET`: a randomly generated secret with at least 32 characters
- `GEMINI_API_KEY`: the Google Gemini API key used for answer generation
- `CORS_ALLOWED_ORIGINS`: one or more exact frontend origins, comma-separated
- `DATA_DIR=/tmp/data` for the current Render Free deployment
- `RAG_USERS_FILE=/tmp/data/users.json` for the current Render Free deployment
- `RAG_DATA_DIR`: legacy-compatible fallback for local configurations that still use the previous variable name
- `JWT_EXPIRY_MINUTES`: optional token lifetime in minutes; the application default is 60
- `PORT`: supplied by Render; do not hardcode it in the start command

`GEMINI_API_KEY` is not needed by `GET /api/health`, and the backend has a non-Gemini fallback for local operation and tests.

## Render backend

Create a Render Web Service from the GitHub repository with:

- Runtime: `Python`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn backend.api:app --host 0.0.0.0 --port $PORT`
- Health check path: `/api/health`

The root `render.yaml` contains these settings and does not configure a Render Disk. Render Free uses the writable ephemeral directory `/tmp/data`; uploaded documents, users, and indexes can disappear after restart or redeploy. Set the secret and frontend-origin values in Render's environment settings rather than in YAML.

## CORS

Set `CORS_ALLOWED_ORIGINS` to the deployed frontend origin, for example `https://app.example.com`. Multiple exact origins may be separated by commas. Do not use `*` with credentials enabled.

## FAISS and local storage

The application intentionally preserves its existing local persistence behavior: uploaded files, `documents.json`, user records, embeddings, metadata, and FAISS indexes are stored under `DATA_DIR` and reloaded on startup. `RAG_DATA_DIR` remains a legacy-compatible fallback when `DATA_DIR` is unset. Render Free uses `/tmp/data`, which is ephemeral, so uploads and indexes can disappear after restart or redeploy. The current design is not configured for multiple backend instances or managed shared storage.

## Security precautions

- Use long, random values for `JWT_SECRET` and rotate secrets through the provider's environment settings.
- Keep `GEMINI_API_KEY` server-side; never expose it through frontend `VITE_*` variables.
- Use HTTPS and exact CORS origins in production.
- Review upload limits, rate limiting, backups, dependency updates, and persistent-disk capacity before making the service public.
- Check the staged GitHub files for secrets and generated runtime data before pushing.