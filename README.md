# Informity AI

A privacy-first macOS desktop app that scans your local files, indexes them with vector embeddings, and lets you chat with your documents using a local LLM. All data stays on your machine.

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Build frontend and run the app (single command — recommended for testing)
make app

# Or run in two steps:
make frontend-build   # Build React UI → src/frontend/dist/
make run             # Start backend on http://127.0.0.1:8420

# Development with auto-reload (code changes restart the server)
make dev
```

**Note:** The app serves the React frontend from `src/frontend/dist/`. Run `make frontend-build` before `make run` or `make dev`. The vanilla backup is archived at `.archive/frontend-bak/`.

## Installation (optional one-time setup)

You can either **run the app and let it download models on first use** (see Offline mode below), or **run a one-time install** so everything is downloaded up front and the app always uses cached models.

**Option A — Install script (recommended for clean setup)**  
Run once to install Python deps and download the embedding model, reranker (cross-encoder), optional LLM, and classifier LLM into app data, then lock the app to cached-only:

```bash
./scripts/install.sh
```

- Uses `scripts/install.conf.json` for model IDs: `embedding_model`, `reranker_model` (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`), optional LLM (default: **Qwen 30B** Q5_K_M via `repo_id` / `filename`), and classifier LLM (default: **Qwen2.5-3B** Q4_K_M for query classification).
- Creates `data/` (or `INFORMITY_APP_DATA_DIR`), downloads all models there, and writes `config.json` with `full_privacy=true` (no network after install).
- After this, the app will **never** auto-download; it only uses what’s already in app data. With those settings enabled, the app makes **no network requests after install** (no Hugging Face or internet contact).

**Uninstall**  
To remove all user data and downloaded content and return to a fresh distribution state (as after cloning), run from repo root: `./scripts/uninstall.sh` or `make uninstall`. This removes the app data directory (config, database, embedding cache, LLM models, vectors, logs), the virtualenv (`.venv`), and local caches. Run `./scripts/install.sh` again to reinstall.

**Reset (in-app)**  
Settings → Reset restores all settings to factory defaults (including default LLM: Qwen 30B). Index → Reset deletes all indexed data and chat history and also resets settings to the same defaults. For a full disk reset (remove all app data but keep `.venv`), run `./scripts/reset.sh`.

**Option B — First-run auto-download**  
Just run the app. On first search/index/chat it may download the embedding model, reranker, LLM, and classifier LLM if not already present. In Settings → Full Privacy Mode you can turn **“Enable”** on so future runs are fully offline.

You do **not** need to remove auto-download from the app: the install script is for users who want a single, explicit setup step and then strictly cached-only behaviour.

## Data location

All application data (database, vectors, LLM and embedding models, logs, config) is stored under a single app data directory. By default this is:

- **Default:** `data/` in the directory where you run the app (e.g. project root). Override with `INFORMITY_APP_DATA_DIR` for production (e.g. **macOS:** `~/Library/Application Support/Informity AI/`).

To keep **everything inside the project folder** (e.g. `informity-ai/data/`), set:

```bash
# Optional: use a custom app data path (default is ./data when run from project root)
export INFORMITY_APP_DATA_DIR=./data
```

Then the layout is:

```
informity-ai/
  data/
    config.json            # Saved settings
    db/                    # SQLite DB and WAL files
      informity.db
      informity.db-shm
      informity.db-wal
    # Note: vectors are stored in SQLite database (vec_chunks table), not a separate directory
    logs/                  # app.log, app.error.log
  .cache/                  # Unified cache directory (at repo root, not committed)
    llm/                   # RAG LLM models (*.gguf files)
    query-classifier/      # Query classification LLM model (*.gguf file)
    diagnostics/           # Diagnostics LLM models (*.gguf files)
    huggingface/           # Embedding + reranker (cross-encoder) cache
      hub/                 # Model blobs + snapshots (required)
      modules/             # Custom model code, created at first load (required)
    docling/               # Docling models (docling creates its own structure inside)
```

Use the same variable in production if you want a custom path (e.g. an external drive).

**Unified cache structure (fully flat)**  
All models and cache are stored under `.cache/` at the repo root (not committed to git) in a flat structure:
- `.cache/chat-llm/` - Chat/RAG LLM models (*.gguf files)
- `.cache/query-classifier-llm/` - Query classification model (*.gguf file)
- `tools/diagnostics/models/` - Diagnostics LLM models (*.gguf files)
- `.cache/huggingface/hub/` - HuggingFace cache (embedding and reranker models)
- `.cache/docling/` - Docling models for document extraction (docling creates its own subdirectories inside)

**One cache only (avoid duplicates)**  
Informity uses **only** the unified cache directory (`.cache/` at repo root). It does not use the default Hugging Face cache (`~/.cache/huggingface/hub`). If you have the same models in both places (e.g. from an earlier run or another tool), you can remove the copy under `~/.cache/huggingface/hub` to free space; Informity will keep using the unified cache copy.

**If embedding or reranker fails with "cache incomplete"** (e.g. missing `snapshots/` under the model folder), remove the incomplete model dir and re-download: run `./scripts/install.sh` or set `INFORMITY_FULL_PRIVACY=false` and run a scan/chat once so the missing model is downloaded.


## PDF Processing

PDFs are processed using **docling**, which provides superior structure preservation including tables, formulas, reading order detection, and built-in OCR support. Scanned and image-only PDFs are handled automatically without requiring external OCR tools.


## Offline mode

The app is **offline-first by default**. With **Full Privacy Mode** on (Settings → Full Privacy Mode), no network is used at runtime; all models are loaded from local storage.

- **Two models in the Hugging Face cache** (`.cache/huggingface/hub/`): (1) **Embedding model** (`nomic-ai/nomic-embed-text-v1.5`) for document and query vectors; (2) **Reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for re-ranking search results. Settings → System shows both for transparency.
- With `full_privacy=true` (default after install), embedding and reranker are loaded only from this cache. Set `INFORMITY_FULL_PRIVACY=false` (or turn off in Settings) once to allow downloads, then turn Full Privacy Mode back on for offline use.
- **LLM (GGUF):** Default model is **Qwen 30B** (Q5_K_M), stored in `.cache/chat-llm/`. With `llm_local_only=true` (default), the app only loads from this directory and never downloads. Place your `.gguf` file there, or set `INFORMITY_LLM_LOCAL_ONLY=false` once to allow a one-time download, then set it back to true.

After models are in place, the app runs fully offline with no internet required.

## Tech Stack

- **Python 3.13** — all core logic
- **FastAPI + uvicorn** — async API server
- **React + Vite** — frontend (served from `src/frontend/dist/` when built)
- **SQLite** via aiosqlite — metadata, config, chat history, vector storage (via sqlite-vec extension)
- **sqlite-vec** — vector storage extension for SQLite (embeddings stored in `vec_chunks` table)
- **sentence-transformers** (nomic-embed-text-v1.5) — embedding generation; (ms-marco-MiniLM-L-6-v2) — optional cross-encoder re-ranking
- **llama-cpp-python** (with Metal/GPU) — local LLM inference (default: Qwen 30B Q5_K_M)

## Project Structure

```
src/frontend/          # React + Vite UI (build output: dist/)
src/informity/
├── main.py                 # FastAPI app entry point, lifespan, health
├── config.py               # Settings via pydantic-settings (config.json + env)
├── db/
│   ├── models.py           # Pydantic models (IndexedFile, Chunk, ScanRecord, ChatMessage, etc.)
│   ├── sqlite.py           # SQLite connection, schema, queries (aiosqlite)
│   └── vectors.py          # SQLite vector storage via sqlite-vec (ChunkEmbedding, VectorStore)
├── scanner/
│   ├── crawler.py          # Filesystem traversal, SHA-256 hashes, compare_with_db
│   ├── watcher.py          # watchdog file change monitoring
│   └── extractors/         # Unified docling extractor (PDF, DOCX, PPTX, XLSX, HTML, CSV) + text extractor
├── indexer/
│   ├── chunker.py         # Parent-child chunking (child ~150 tokens, parent ~600 tokens)
│   ├── embedder.py        # Embedding generation (nomic-embed-text-v1.5)
│   ├── classifier.py      # Auto-tagging, categorization, year extraction
│   ├── post_process.py    # Hyphenation repair (index-time only)
│   ├── reranker.py        # Cross-encoder re-ranking (mandatory for all queries)
│   └── pipeline.py        # index_file, reindex_file, remove_file — orchestration
├── llm/
│   ├── engine.py          # LLM inference (llama-cpp-python, Metal)
│   ├── model_adapter.py   # Per-model profiles (Qwen 30B, Qwen3 14B)
│   ├── rag.py             # QueryRouter — dispatches to handlers based on intent
│   ├── query_classifier.py # Structured slot extraction + decision tree
│   ├── retrieval.py       # Unified retrieval pipeline (vector search → rerank)
│   ├── prompt_builder.py  # Prompt construction
│   ├── streaming.py       # Minimal streaming
│   ├── metadata_filters.py # Unified metadata filter extraction
│   └── handlers/          # Query handlers (metadata, rag, simple)
└── api/
    ├── schemas.py         # Request/response Pydantic models
    ├── operation_state.py # Long-running operation flags (scan, reset)
    ├── routes_scan.py     # POST /api/scan, GET /api/scan/status, GET /api/files
    ├── routes_index.py    # POST /api/index/rebuild, GET /api/index/status, POST /api/index/reset
    ├── routes_search.py   # POST /api/search
    ├── routes_chat.py     # POST /api/chat (SSE), GET/PUT/DELETE conversations
    ├── routes_settings.py # GET/PUT /api/settings, POST /api/settings/reset, env-vars, file-types
    ├── routes_system.py   # GET /api/diagnostics, GET /api/diagnostics/summary, POST /api/shutdown
    └── env_vars_metadata.py  # INFORMITY_* env var groups for Configuration page
src/diagnostics/          # Diagnostics add-on package (sibling to informity)
├── issue_types.py        # IssueType enum (6 types)
├── observer.py           # EvalMetrics dataclass, detect_issues(), populate_signals()
└── tools/                # Evaluation pipeline tools (tools/diagnostics/)
    ├── evaluate.py        # Runs queries, collects metrics, writes traces
    ├── analyze.py         # Aggregates metrics, generates reports
    ├── generate_queries.py # Builds query sets from index
    ├── pipeline.py        # End-to-end orchestrator
    └── golden_set.py      # Pre-flight validation queries
```

## Diagnostics Evaluation

Informity includes a diagnostics evaluation system for testing RAG performance and identifying issues. The pipeline does **not** require the main application to be running; it uses the same index and config.

**Pipeline (recommended):** Generates queries from your index, runs golden set + regular evaluation, then analysis. Output streams to the terminal; use `--quiet` to show only the final message. Redirect output if you need a log (e.g. `2>&1 | tee run.log`).

```bash
# Run full pipeline (default: generate 15 queries, balanced strategy)
uv run python tools/diagnostics/pipeline.py

# Custom query count/strategy with deterministic seed
uv run python tools/diagnostics/pipeline.py --num-queries 20 --query-strategy balanced --seed 42

# Run with a custom query suite file (skips generation)
uv run python tools/diagnostics/pipeline.py --run-id custom-suite --queries-file .internal/TEST-QUERIES.json

# Minimal output
uv run python tools/diagnostics/pipeline.py --quiet
```

**Query generation:** Regular queries are generated from the index with strategies `balanced`, `coverage`, `focused`, `reasoning`, and `phase5`. Balanced (default) includes a curated versioned regression bank (`tools/diagnostics/query_banks/regression_v1.json`) plus generated metadata/focused/coverage/aggregation/comparison queries.

**Run steps individually:**

```bash
# Generate queries only (optional; pipeline does this by default)
uv run python tools/diagnostics/generate_queries.py --run-id {run_id} --num-queries 15 --strategy balanced --seed 42

# Run evaluation
uv run python tools/diagnostics/evaluate.py --run-id {run_id} --queries-file data/diagnostics/runs/{run_id}/queries/queries.json

# Analyze results
uv run python tools/diagnostics/analyze.py --run-id {run_id}
```

Results are saved to `data/diagnostics/runs/{run_id}/`:
- `queries/` - `queries.json` (generated regular queries)
- `traces/` - Trace files per query×model
- `results/` - `run.json`, `report.md`, `report.json`, `pipeline_manifest.json`

Metrics are read from trace files (app-compliant, no protocol pollution). The evaluation system uses OTel-named fields via `openinference-semantic-conventions` for future compatibility.

## Tooling Layout

`tools/` is organized by purpose:

- `tools/diagnostics/` - diagnostics pipeline and orchestration (`pipeline.py`, `run_all.sh`, `run_test_queries_suite.py`, etc.)
- `tools/qa/` - release/quality gate helpers (`docs_lint.py`, `secret_scan.py`)
- `tools/performance/` - performance benchmark scripts
- `tools/smoke/` - manual smoke scripts (not part of `pytest tests/`)
- `tools/maintenance/` - operational maintenance utilities

Convenience commands:

```bash
# Smoke scripts
make smoke-basic
make smoke-infra

# Maintenance
make maintenance-index-check
make maintenance-index-repair
make maintenance-download-nltk
make maintenance-reinstall-packages
make maintenance-chunk-structure
make maintenance-legacy-chunks
make maintenance-orphaned-chunks
make maintenance-migrate-hf-cache

# Diagnostics run control
make diagnostics-stop
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

Informity AI is licensed under the [MIT License](LICENSE). Third-party dependencies are under their own licenses; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for details and copyleft (GPL/AGPL) components.
