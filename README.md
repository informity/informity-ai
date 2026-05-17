# Informity AI

<p align="center">
  <img src="./assets/demo/informity-ai-demo.gif" alt="Informity AI demo" width="960" />
</p>

[![Version](https://img.shields.io/github/v/tag/informity/informity-ai?label=version)](https://github.com/informity/informity-ai/releases)
[![Python](https://img.shields.io/badge/python-3.13+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-macOS-black?logo=apple)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![CI](https://github.com/informity/informity-ai/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/informity/informity-ai/actions/workflows/ci.yml)

Privacy-first local document intelligence for macOS.
Informity scans and indexes local files, then answers questions with a local RAG pipeline backed by local embeddings and local LLM inference. Your documents and vectors stay on your machine.

## Contents

- [Highlights](#highlights)
- [Quick Start](#quick-start)
- [Installation (optional one-time setup)](#installation-optional-one-time-setup)
- [Data Location](#data-location)
- [PDF Processing](#pdf-processing)
- [Offline Mode](#offline-mode)
- [MCP Server (Read-only)](#mcp-server-read-only)
- [Chat Scope Contract](#chat-scope-contract)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Release Scripts](#release-scripts)
- [Development](#development)
- [License](#license)

## Highlights

- Local-first RAG over local files
- Offline-first runtime (full privacy mode)
- Temporary per-chat uploaded files in Researcher mode (`+` attach, `x` remove)
- SQLite + sqlite-vec for metadata + embeddings
- Local LLM inference via `xllamacpp` (Metal/GPU on macOS)
- Optional Ollama runtime provider (localhost daemon)

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Build frontend and run the app (single command — recommended for testing)
make app

# Or run in two steps:
make frontend-build   # Build React UI → src/frontend/dist/
make run              # Start backend on http://127.0.0.1:8420

# Development with auto-reload (code changes restart the server)
make dev
```

**Note:** The app serves the React frontend from `src/frontend/dist/`. Run `make frontend-build` before `make run` or `make dev`.

## Installation (optional one-time setup)

You can either **run the app and let it download models on first use** (see Offline mode below), or **run a one-time install** so everything is downloaded up front and the app always uses cached models.

**Option A — Install script (recommended for clean setup)**  
Run once to install Python deps and download the embedding model, reranker (cross-encoder), and optional LLM into app data, then lock the app to cached-only:

```bash
./scripts/install_app.sh
```

For first-run setup testing (install app/runtime dependencies only, no models preinstalled):

```bash
INFORMITY_INSTALL_PROFILE=dev INFORMITY_INSTALL_SKIP_MODELS=1 ./scripts/install_app.sh
```

- Uses `scripts/install.conf.json` for model IDs: `embedding_model`, `reranker_model` (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`), and optional LLM (default: **Qwen3.6 35B A3B** Q4_K_M via `repo_id` / `filename`).
- Downloads all models to `~/.informity/` by default (override with `INFORMITY_APP_DATA_DIR`) and writes `config.json` with `full_privacy=true` (no network after install).
- After this, the app will **never** auto-download; it only uses what’s already in app data. With those settings enabled, the app makes **no network requests after install** (no Hugging Face or internet contact).

**Uninstall**  
To remove all user data and downloaded content and return to a fresh distribution state (as after cloning), run from repo root: `./scripts/install_uninstall_app.sh` or `make uninstall`. This removes the app data directory (config, database, embedding cache, LLM models, vectors, logs), the virtualenv (`.venv`), and local caches. Run `./scripts/install_app.sh` again to reinstall.

**Reset (in-app)**  
Settings → Reset restores all settings to factory defaults (including default LLM: Qwen3.6 35B A3B). Index → Reset deletes all indexed data and chat history and also resets settings to the same defaults. For a full local cleanup and reset, run `./scripts/install_uninstall_app.sh` (or `make uninstall`). Then run `./scripts/install_app.sh` to reinstall.

**Option B — First-run auto-download**  
Just run the app. On first search/index/chat it may download the embedding model, reranker, and LLM if not already present. In Settings → Full Privacy Mode you can turn **“Enable”** on so future runs are fully offline.

You do **not** need to remove auto-download from the app: the install script is for users who want a single, explicit setup step and then strictly cached-only behaviour.

## Data Location

All application data (database, vectors, LLM and embedding models, logs, config) is stored under a single app data directory:

- **Default:** `~/.informity/`
- **Override:** Set `INFORMITY_APP_DATA_DIR` to use a custom path (e.g. an external drive or CI-isolated `./data`).

Directory layout:

```
~/.informity/
  config.json              # Saved settings
  db/                      # SQLite DB and WAL files
    informity.db
  storage/
    uploads/               # Temporary chat-scoped uploaded files
  logs/                    # Runtime log files
  models/
    llm/                   # LLM models (*.gguf files)
  cache/                   # Unified cache (not committed)
    huggingface/           # Embedding + reranker (cross-encoder) models
      hub/                 # Model blobs + snapshots (required)
      modules/             # Custom model code, created at first load (required)
    docling/               # Docling models for document extraction
```

**One cache only (avoid duplicates)**
Informity uses **only** the app data cache directory (`cache/` under app data). It does not use the default Hugging Face cache (`~/.cache/huggingface/hub`). If you have the same models in both places, you can remove the copy under `~/.cache/huggingface/hub` to free space.

**If embedding or reranker fails with "cache incomplete"** (e.g. missing `snapshots/` under the model folder), remove the incomplete model dir and re-download: run `./scripts/install_app.sh` or set `INFORMITY_FULL_PRIVACY=false` and run a scan/chat once so the missing model is downloaded.


## PDF Processing

PDFs are processed using **docling**, which provides superior structure preservation including tables, formulas, reading order detection, and built-in OCR support. Scanned and image-only PDFs are handled automatically without requiring external OCR tools.


## Offline Mode

The app is **offline-first by default**. With **Full Privacy Mode** on (Settings → Full Privacy Mode), no network is used at runtime; all models are loaded from local storage.

- **Two models in the Hugging Face cache** (`cache/huggingface/hub/` under app data): (1) **Embedding model** (`nomic-ai/nomic-embed-text-v1.5`) for document and query vectors; (2) **Reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for re-ranking search results. Settings → System shows both for transparency.
- With `full_privacy=true` (default after install), embedding and reranker are loaded only from this cache. Set `INFORMITY_FULL_PRIVACY=false` (or turn off in Settings) once to allow downloads, then turn Full Privacy Mode back on for offline use.
- **LLM (GGUF):** App default model is **Qwen3.6 35B A3B** (`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`), stored in `models/llm/` under the app data directory. With `llm_local_only=true` (default), the app only loads from this directory and never downloads. Place your `.gguf` file there, or set `INFORMITY_LLM_LOCAL_ONLY=false` once to allow a one-time download, then set it back to true.  
  Note: the optional installer seed in `scripts/install.conf.json` points to Qwen3.6 35B A3B.

After models are in place, the app runs fully offline with no internet required.

## MCP Server (Read-only)

Informity includes a built-in read-only MCP server so external AI clients (for example Claude Desktop) can query your indexed library.

Enable it in the app at **Settings -> System -> MCP Server**.

Current read-only tools:

- `informity_health`
- `informity_files_list`
- `informity_search_semantic`
- `informity_filter_options`
- `informity_index_status`
- `informity_scan_status`

Tool behavior notes:

- `informity_files_list` and `informity_search_semantic` default to `limit=50` and allow explicit values up to `200`.
- `informity_index_status` reports corpus counts with chat-upload records (`upload.local`) excluded, matching MCP corpus visibility.
- `informity_filter_options` returns currently available categories and file type extensions from the index.
- For `informity_search_semantic`, `category` and `file_types` are optional filters; start with only `query` + `limit` unless you specifically need filtering.
- `category` values are: `document`, `plaintext`, `data`, `web`, `other`.
- `file_types` accepts both dot extensions (`.pdf`) and common aliases (`pdf`, `docx`, `md`), case-insensitive.
- If a filtered semantic search returns zero results, the tool includes `hints` with applied filters and recovery guidance.

Access levels:

- `metadata_only` (recommended default)
- `search_snippets`
- `full_content`

For external MCP clients, `search_snippets` is generally the best balance of privacy and answer quality.

Security notes:

- Keep MCP disabled unless you need it.
- Informity MCP is strictly read-only: only an allowlisted set of read tools can be executed.
- `http` transport requires a bearer token.
- External clients operate outside Informity's internal Full Privacy controls, so only connect trusted local clients.
- MCP interaction logs are written to `~/.informity/logs/app.mcp.log` (or your configured `logs_dir`).

### HTTP transport quick test (`curl`)

After enabling MCP with `http` transport in Settings, test from a terminal:

```bash
curl -sS http://127.0.0.1:8431/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer imcp_YOUR_TOKEN_HERE" \
  -d '{
    "jsonrpc":"2.0",
    "id":3,
    "method":"tools/call",
    "params":{
      "name":"informity_search_semantic",
      "arguments":{
        "query":"artificial intelligence",
        "limit":3
      }
    }
  }'
```

### Claude Desktop config

For local development from this repository, use `uv run` so Claude starts the MCP server in your project environment:

```json
{
  "mcpServers": {
    "informity": {
      "command": "/Users/<you>/.local/bin/uv",
      "args": [
        "run",
        "--project",
        "/absolute/path/to/informity-ai",
        "informity-mcp"
      ]
    }
  }
}
```

For packaged installs (when `informity-mcp` is on `PATH`), use:

```json
{
  "mcpServers": {
    "informity": {
      "command": "informity-mcp"
    }
  }
}
```

Quick validation prompts in Claude Desktop:

- "Use `informity_health` and return raw JSON."
- "Use `informity_index_status` and summarize counts in one sentence."
- "Use `informity_search_semantic` with query `termination clause` and limit `3`."

## Provider Selection

Informity supports two LLM runtime providers:

- `local_gguf` (default): in-process `xllamacpp` with GGUF models in `models/llm/`.
- `ollama`: uses an Ollama daemon (default URL `http://127.0.0.1:11434`) and `llm_model_id` (for example `qwen3.6:35b`).

Known Ollama model aliases are mapped to Informity tuned profiles:

- `qwen3.6:35b*` -> Qwen3.6 35B A3B profile
- `qwen3:14b*`, `qwen3.5:14b*` -> Qwen3 14B profile
- `qwen3.5:9b*` -> Qwen3.5 9B profile

Unknown model IDs use a conservative Ollama default profile.

Setup/readiness is provider-aware:

- First-run setup remains the existing local-model onboarding flow (`local_gguf` path).
- Ollama is an advanced post-setup integration enabled from Settings.
- After setup, choosing `ollama` bypasses local GGUF gating and uses:
  - Ollama daemon reachable
  - configured Ollama model available locally
  - required dependency caches (embedding, reranker, docling)

## Chat Scope Contract

`POST /api/chat` supports optional scoped researcher retrieval with:

- `scoped_file_ids`: one-or-more indexed file IDs
- `scoped_upload_ids`: one-or-more chat-scoped upload IDs (Researcher mode only)

Notes:

- Legacy `file_id` is no longer accepted.
- When `scoped_file_ids` is provided, researcher retrieval is constrained to that file set.
- Uploaded files are temporary and chat-scoped (`POST /api/chat/uploads`, `GET /api/chat/chats/{chat_id}/uploads`, `DELETE /api/chat/uploads/{upload_id}`).
- When uploads are present and no explicit subset is chosen, retrieval defaults to all ready uploaded files in that chat.
- Removing the last upload auto-falls back to scanned-documents retrieval for subsequent turns.
- Assistant mode remains retrieval-free.

## Tech Stack

- **Python 3.13** — all core logic
- **FastAPI + uvicorn** — async API server
- **React + Vite** — frontend (served from `src/frontend/dist/` when built)
- **SQLite** via aiosqlite — metadata, config, chat history, vector storage (via sqlite-vec extension)
- **sqlite-vec** — vector storage extension for SQLite (embeddings stored in `vec_chunks` table)
- **sentence-transformers** (nomic-embed-text-v1.5) — embedding generation; (ms-marco-MiniLM-L-6-v2) — optional cross-encoder re-ranking
- **xllamacpp** (with Metal/GPU) — local LLM inference (app default: Qwen3.6 35B A3B Q4_K_M)

## Project Structure

```
src/frontend/                       # React + Vite UI (build output: dist/)
src/informity/
├── main.py                         # FastAPI app entry point, lifespan, health
├── config.py                       # Settings via pydantic-settings (config.json + env)
├── db/
│   ├── models.py                   # Pydantic models (IndexedFile, Chunk, ScanRecord, ChatMessage, etc.)
│   ├── sqlite.py                   # SQLite connection, schema, queries (aiosqlite)
│   └── vectors.py                  # SQLite vector storage via sqlite-vec (ChunkEmbedding, VectorStore)
├── scanner/
│   ├── crawler.py                  # Filesystem traversal, SHA-256 hashes, compare_with_db
│   ├── watcher.py                  # watchdog file change monitoring
│   └── extractors/                 # Docling extractor (PDF, DOCX, PPTX, XLSX, HTML, CSV) + EPUB extractor + text extractor
├── indexer/
│   ├── chunker.py                  # Parent-child chunking (child ~150 tokens, parent ~512 tokens)
│   ├── embedder.py                 # Embedding generation (nomic-embed-text-v1.5)
│   ├── classifier.py               # Auto-tagging, categorization, year extraction
│   ├── post_process.py             # Hyphenation repair (index-time only)
│   ├── reranker.py                 # Cross-encoder re-ranking (mandatory for all queries)
│   ├── adaptive_tuning.py          # Corpus-aware top-k tuning cache
│   ├── term_dictionary_builder.py  # Builds term/acronym dictionary from indexed corpus
│   └── pipeline.py                 # index_file, reindex_file, remove_file — orchestration
├── llm/
│   ├── engine.py                   # LLM inference (xllamacpp, Metal)
│   ├── model_adapter.py            # Per-model profiles (Qwen3 14B, Qwen3.5 9B, Qwen3.6 35B A3B)
│   ├── rag.py                      # QueryRouter — dispatches to handlers based on intent
│   ├── query_classifier.py         # Deterministic slot extraction + NLP/promptcue intent routing
│   ├── retrieval.py                # Unified retrieval pipeline (vector search → rerank)
│   ├── term_dictionary.py          # Runtime query expansion via corpus term dictionary
│   ├── intent_router.py            # Promptcue-backed intent classification router
│   ├── classification_policy.py    # Intent routing policy and normalization
│   ├── promptcue_adapter.py        # Adapter for promptcue intent classification
│   ├── chat_mode.py                # Assistant vs Researcher mode routing policy
│   ├── contract_gate.py            # Final closeout contract validation/repair
│   ├── contract_prompt_parser.py   # Parses required output section cues from user prompts
│   ├── metrics_payload.py          # Normalized diagnostics metrics payload helpers
│   ├── nlp_heuristics.py           # Minimal deterministic lexical cues
│   ├── prompt_builder.py           # Prompt construction and budget management
│   ├── streaming.py                # LLM stream wrapper
│   ├── metadata_filters.py         # Unified metadata filter extraction (year, category, extension)
│   ├── system_prompts.py           # Centralized system prompt templates
│   ├── timeout_policy.py           # Request timeout policy mapping by mode/intent
│   ├── user_messages.py            # Centralized user-facing message strings
│   ├── web_search.py               # Tavily/Linkup-backed web search adapter and status handling
│   ├── rag_runtime/                # RAG execution sub-pipeline (retrieval + generation phases)
│   └── handlers/                   # Query handlers (metadata, rag, simple)
└── api/
    ├── schemas.py                  # Request/response Pydantic models
    ├── operation_state.py          # Long-running operation flags (scan, reset)
    ├── setup_state.py              # First-run setup state management
    ├── chat_orchestrator.py        # Chat request orchestration entry point
    ├── chat_continuation.py        # Continuation/duplicate detection helpers
    ├── chat_sse.py                 # SSE event formatting for chat streams
    ├── chat_closeout.py            # Post-generation chat record finalization
    ├── chat_stream_registry.py     # Active stream registry (cancel support)
    ├── routes_scan.py              # POST /api/scan, GET /api/scan/status, GET /api/scan/errors, GET /api/files
    ├── routes_index.py             # POST /api/index/rebuild, GET /api/index/status, POST /api/index/reset
    │                               # GET|POST /api/index/term-dictionary/status|rebuild|purge
    ├── routes_search.py            # POST /api/search
    ├── routes_chat.py              # POST /api/chat (SSE), GET/PUT/DELETE conversations
    ├── routes_settings.py          # GET/PUT /api/settings, POST /api/settings/reset, env-vars, file-types
    ├── routes_system.py            # GET /api/diagnostics, GET /api/diagnostics/summary, POST /api/shutdown
    └── env_vars_metadata.py        # INFORMITY_* env var groups for Configuration page
src/informity/diagnostics/          # Diagnostics package
├── issue_types.py                  # IssueType enum
├── observer.py                     # EvalMetrics dataclass, detect_issues(), populate_signals()
└── resource_snapshot.py            # System resource snapshot at trace time
```

## Release Scripts

`scripts/` is committed and includes both developer setup scripts and maintainer release automation.

Maintainer-focused build/release scripts:

```bash
# Generate Tauri icons from source logo
make tauri-icons

# Build backend sidecar used for Tauri packaging
make tauri-backend

# Build + sign + notarize macOS release (requires local signing credentials)
make tauri-build-mac

# Build Linux release artifacts (.deb + AppImage)
make tauri-build-linux
```

Runtime chat diagnostics metrics are also persisted in SQLite (`response_diagnostics_metrics`) for operational observability. Use `GET /api/diagnostics/summary` for aggregated counts/rates/query-type breakdowns over a time window.

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check src/

# Format
uv run ruff format src/

# Frontend development (hot reload)
make frontend        # Vite dev server on port 5173 — run backend separately (make run or make dev)
make frontend-build  # Build React for production
make app             # Build frontend + run backend (single command for testing)
```

## License

Informity AI is licensed under the [MIT License](LICENSE).
