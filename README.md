# AI Research Knowledge Assistant

A beginner-friendly RAG application for asking grounded questions about private PDF, DOCX, and TXT research documents. It combines document extraction, sentence-aware chunking, SentenceTransformer embeddings, FAISS retrieval, hybrid keyword search, CrossEncoder reranking, Gemini generation, citations, authentication, persistence, and a React interface.

## Problem Statement

Research documents are difficult to search manually, especially when information is spread across several files. A useful assistant should find relevant evidence, answer from that evidence, and show users where the answer came from.

## Why RAG?

An LLM by itself may rely on its training data, miss private documents, or produce unsupported answers. Retrieval-augmented generation first finds relevant passages from the user's documents and then gives those passages to Gemini as context. This makes answers more traceable and supports document-specific questions.

## Architecture

```text
User
  -> React + TypeScript frontend
  -> FastAPI API + authentication
  -> Document extraction (PDF/DOCX/TXT)
  -> Sentence-aware chunking
  -> SentenceTransformer embeddings
  -> FAISS IndexFlatL2 + persistent metadata
  -> Semantic search + keyword search
  -> Candidate merge and duplicate removal
  -> Distance threshold
  -> CrossEncoder reranking
  -> Grounded context with source metadata
  -> Gemini Interactions API
  -> Answer + citations, optionally streamed
```

## End-to-End RAG Pipeline

1. An authenticated user uploads a PDF, DOCX, or TXT file.
2. Text is extracted and PDF page boundaries are retained when available.
3. Text is split into sentence-aware chunks of about 300 characters with complete-sentence overlap near 50 characters.
4. `all-MiniLM-L6-v2` creates an embedding for every chunk.
5. Embeddings are stored in a FAISS `IndexFlatL2` index.
6. A question is embedded with the same model.
7. Semantic FAISS results and lightweight keyword results are merged and deduplicated.
8. Candidates outside the configured L2 distance threshold are removed.
9. A lightweight CrossEncoder scores each remaining question/chunk pair and reranks them.
10. The final chunks and their source metadata are sent to Gemini.
11. Gemini is instructed to answer only from the retrieved context.
12. The API returns an answer, citations, retrieval details, and latency. Streaming sends text progressively and citations in the final event.

## Technology Stack

- Python
- FastAPI and Uvicorn
- SentenceTransformers: `all-MiniLM-L6-v2` and CrossEncoder reranking
- FAISS `IndexFlatL2`
- NumPy
- Google GenAI Python SDK and Gemini Interactions API
- `pypdf` and `python-docx`
- React, TypeScript, Vite, Tailwind CSS, and Lucide icons
- Pytest

## Implemented Features

### Document extraction and chunking

`backend/extraction.py` reads TXT files, extracts PDF page text, and reads DOCX paragraphs. Unsupported file types, empty files, unreadable documents, and processing failures receive safe API errors.

`backend/chunker.py` detects sentence endings with Python's standard library and groups complete sentences near the configured character limit. Overlap preserves nearby context for retrieval. Keeping sentence boundaries is less likely to separate a definition from its explanation than blindly cutting every fixed number of characters.

### Embeddings and FAISS

`backend/embeddings.py` wraps `SentenceTransformer("all-MiniLM-L6-v2")`. The same model embeds documents and questions, ensuring matching vector dimensions. Tests can explicitly use its deterministic offline mode; production uses SentenceTransformers.

`backend/vector_store.py` stores vectors in FAISS `IndexFlatL2` and associates every vector with metadata: document ID, filename, page when available, chunk number, and text. Indexes are saved as `index.faiss`; metadata and JSON embeddings are saved beside them.

### Retrieval and reranking

`backend/retrieval.py` performs semantic FAISS search and lightweight keyword search for the same query. It merges candidates by chunk ID, removes duplicates, applies the existing distance threshold, and sends eligible candidates to the configurable CrossEncoder model `cross-encoder/ms-marco-MiniLM-L-6-v2`.

FAISS provides fast relevance by vector distance. Keyword search helps with exact names and terms. Hybrid retrieval can recover useful candidates that either method alone might rank too low. CrossEncoder reranking reads the question and chunk together, producing a more focused final ordering.

### Grounded Gemini generation and streaming

`backend/generator.py` uses the current Google GenAI Interactions API with `gemini-3.6-flash`. The API key is read only from `GEMINI_API_KEY`. The prompt includes retrieved source metadata and instructs Gemini to answer only from context and use the existing fallback when evidence is missing:

```text
I could not find the answer in the provided documents.
```

`POST /api/chat/stream` sends newline-delimited JSON events as text is generated. The React client appends text progressively, shows a generating state, handles stream errors, and displays final citations. The original `POST /api/chat` endpoint remains available.

### Citations and conversations

Answers expose structured citations with filename, document ID, chunk number, distance, and PDF page when available. TXT and DOCX sources show filename and chunk number. Follow-up questions are rewritten with the previous user question when they are short or depend on pronouns such as "it" or "that".

When multiple owned document IDs are selected, each document is retrieved separately and results are grouped for comparison before being combined for grounded answer generation.

### Persistence and document lifecycle

Uploads are stored under `RAG_DATA_DIR`. Document records contain ID, filename, type, timestamp, chunk count, processing status, owner ID, and a content hash. The server reloads JSON records and FAISS indexes at startup, rejects duplicate content for the same user, and supports deletion through the API.

### Authentication and isolation

Authentication answers who a user is; authorization decides what that user may access. Registration stores a salted PBKDF2 password hash, never the password. Login returns a signed JWT using `JWT_SECRET`. Protected document, chat, comparison, and streaming routes validate bearer tokens and filter records by owner ID.

### Error handling, logging, and latency

Centralized FastAPI handlers return safe JSON containing an error type, friendly message, and request ID. Request middleware adds `X-Request-ID` and structured JSON logs. Logs include lifecycle events and latency but do not include API keys, authorization headers, passwords, or document contents. Retrieval, reranking, generation, and total request latency are tracked separately where applicable.

## Folder Structure

```text
backend/
  api.py                 # FastAPI routes and request lifecycle
  auth.py                # Password hashing and JWT handling
  chunker.py             # Configurable sentence-aware chunks
  config.py              # Environment-backed settings
  embeddings.py          # SentenceTransformer wrapper
  errors.py              # Safe application errors
  evaluation.py          # Ten-question retrieval evaluation
  extraction.py          # PDF, DOCX, and TXT extraction
  generator.py           # Grounded Gemini and streaming generation
  logging_config.py      # Safe structured logging
  query_rewrite.py       # Simple conversational query rewriting
  reranker.py            # CrossEncoder scoring
  retrieval.py           # Semantic, keyword, hybrid, and comparison retrieval
  vector_store.py        # FAISS and metadata persistence
data/sample.txt          # Example research text
frontend/                # React + TypeScript + Vite application
tests/                   # Backend, API, auth, deployment, and RAG tests
render.yaml              # Optional Render service configuration
requirements.txt         # Python dependencies
.env.example             # Local configuration template
```

## API Endpoints

- `GET /api/health` - public readiness response without secrets
- `POST /api/auth/register` - create an account
- `POST /api/auth/login` - receive a JWT
- `GET /api/auth/me` - read the authenticated user
- `GET /api/documents` - list the current user's documents
- `POST /api/documents/upload` - upload and index a document
- `DELETE /api/documents/{document_id}` - delete an owned document
- `POST /api/chat` - non-streaming grounded chat
- `POST /api/chat/stream` - newline-delimited streaming grounded chat
- `GET /api/evaluation` - run the included ten-question retrieval evaluation

## Evaluation and Verified Results

The evaluation module contains 10 questions with expected topic terms. For each question it runs retrieval at a selected K, records whether an expected passage was retrieved, and measures retrieval latency. It reports Recall@K, Hit Rate, average latency, and per-question results. The project does not claim fabricated production quality metrics; run `GET /api/evaluation` against the documents in your environment for current retrieval measurements.

The last verified local checks reported:

- Full Python suite: `31 passed`
- Frontend production build: passed
- Python compilation: passed

These are local automated test results, not a claim about answer quality on an external corpus.

## Configuration

Copy `.env.example` to `.env` for local development. Important variables are:

- `APP_ENV` - `development` or `production`
- `PORT` - local port; Render supplies its own `PORT`
- `GEMINI_API_KEY` - optional locally, required for live Gemini generation
- `JWT_SECRET` - use a random secret of at least 32 characters in production
- `JWT_EXPIRY_MINUTES` - JWT lifetime
- `RAG_DATA_DIR` - document, index, and metadata storage directory
- `RAG_USERS_FILE` - persistent user JSON path
- `CORS_ALLOWED_ORIGINS` - comma-separated exact frontend origins

`.env`, frontend environment files, storage, build output, and secrets are ignored by Git. Never put an API key in source code or `VITE_*` variables unless it is intentionally public; `GEMINI_API_KEY` belongs only on the backend.

## Deployment

### Render backend

1. Create a Render Web Service from the repository.
2. Build command: `pip install -r requirements.txt`.
3. Start command: `uvicorn backend.api:app --host 0.0.0.0 --port $PORT`.
4. Set `APP_ENV=production`, `JWT_SECRET`, `GEMINI_API_KEY`, and `CORS_ALLOWED_ORIGINS` in Render environment settings.
5. Set the health check path to `/api/health`.

The included `render.yaml` describes the same service and an optional Render persistent disk. Do not deploy from this README automatically.

### Vercel frontend

1. Import the repository into Vercel.
2. Set the project root to `frontend`.
3. Use the existing `npm run build` command.
4. Set `VITE_API_URL` to the deployed Render API URL, based on `frontend/.env.example`.

## Storage Limitations

The current implementation uses local JSON files and FAISS files. A cloud instance without a persistent disk has ephemeral storage, so documents, user records, and indexes can disappear after a restart or redeploy. `render.yaml` includes a persistent disk configuration, but persistence only exists when that disk is actually enabled for the service. No managed database, object storage, shared vector service, or multi-instance coordination is configured. Multiple backend instances should not be used with local storage unless shared durable storage and indexing coordination are added.

## Security Considerations

- Keep `GEMINI_API_KEY` and `JWT_SECRET` in backend environment settings.
- Use a long random JWT secret and HTTPS in production.
- Configure exact trusted frontend origins instead of wildcard CORS.
- Passwords are stored as salted PBKDF2 hashes.
- Bearer tokens and ownership checks protect documents and chat.
- Error responses hide stack traces; logs exclude secrets and full document content.
- Review upload size limits, rate limiting, secret rotation, backups, and dependency updates before a public launch.

## Known Limitations

- Local JSON and FAISS persistence is not a substitute for managed shared storage.
- The simple query rewriting heuristic does not understand every conversational reference.
- Keyword search is intentionally lightweight and is not a full BM25 implementation.
- The deterministic fallback is useful for local tests but is not a replacement for Gemini quality.
- PDF page attribution is based on extracted text positions and may be imperfect for unusual PDFs.
- The built-in evaluation uses a small example corpus and topic-term matching, not human-judged answer quality.
- There is no background job queue, upload size policy, rate limiter, or multi-instance index locking.

## Future Improvements

- Move users and document records to a managed database.
- Move source files and indexes to durable object/shared storage.
- Add background ingestion jobs and progress status.
- Add stronger query rewriting and conversation persistence.
- Add a real BM25 index and calibrated reranker/threshold configuration.
- Expand evaluation with a larger, human-reviewed benchmark and answer faithfulness checks.
- Add rate limiting, upload limits, observability dashboards, backups, and automated deployment checks.

## How I would explain this project in an interview

I built a research assistant that answers questions from a user's own PDF, DOCX, and TXT documents. The backend extracts text, keeps PDF page information, creates sentence-aware overlapping chunks, and embeds them with `all-MiniLM-L6-v2`. For retrieval, FAISS finds semantically similar chunks and a lightweight keyword search catches exact terms; I merge and deduplicate those candidates, apply an L2 distance threshold, and use a CrossEncoder to rerank the survivors. Gemini receives only the retrieved context and source metadata, so the answer is grounded and can cite a filename, page, and chunk. The system supports streaming answers, conversational follow-ups, multi-document comparison, persistent FAISS storage, and JWT-based user isolation. I evaluate retrieval with ten questions using Recall@K, Hit Rate, and measured latency, and the automated project suite currently has 31 passing tests. I would describe the system as interview-ready and extensible, not as perfectly production-scalable, because local FAISS and JSON storage still require durable shared infrastructure for larger deployments.
