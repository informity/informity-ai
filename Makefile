# ==============================================================================
# Informity AI — Makefile
# Common development commands. Requires `uv` package manager.
# ==============================================================================

.DEFAULT_GOAL := help
.PHONY: help run dev test lint format baseline diagnostics-evaluate diagnostics-analyze diagnostics-pipeline diagnostics-stop reset-db reset-all clean-data clean install install-dev uninstall frontend frontend-build tauri-icons tauri-backend tauri-dev tauri-build app qa-quick qa-full qa-security qa-secrets qa-lint qa-typecheck qa-docs smoke-basic smoke-infra smoke-pdf maintenance-index-check maintenance-index-repair maintenance-download-nltk maintenance-reinstall-packages maintenance-chunk-structure maintenance-orphaned-chunks

# ==============================================================================
# Configuration
# ==============================================================================

HOST          := 127.0.0.1
PORT          := 8420
APP_DISPLAY_NAME := Informity AI
DIR_MODELS := models
DIR_LLM := llm
DIR_HUGGINGFACE := huggingface
DIR_HUB := hub
DIR_DOCLING := docling

# Use ~/.informity as the default runtime data root so
# `make run` / `make dev` match bundled-app behavior (same as config.py default).
APP_DATA_DIR  := $(HOME)/.informity
APP_CACHE_DIR := $(APP_DATA_DIR)/cache
APP_MODELS_ROOT_DIR := $(APP_DATA_DIR)/$(DIR_MODELS)
APP_MODELS_DIR := $(APP_MODELS_ROOT_DIR)/$(DIR_LLM)
APP_HF_HUB_DIR := $(APP_CACHE_DIR)/$(DIR_HUGGINGFACE)/$(DIR_HUB)
APP_DOCLING_CACHE_DIR := $(APP_CACHE_DIR)/$(DIR_DOCLING)
export INFORMITY_APP_DATA_DIR := $(APP_DATA_DIR)
export INFORMITY_CACHE_DIR := $(APP_CACHE_DIR)
export INFORMITY_MODELS_DIR := $(APP_MODELS_DIR)

# ==============================================================================
# Targets
# ==============================================================================

help: ## Show this help message
	@echo "Informity AI — Development Commands"
	@echo ""
	@echo "  Data directory: $(APP_DATA_DIR)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""

install: ## Install runtime dependencies and download models into app data
	./scripts/install_app.sh

install-dev: ## Install runtime + dev dependencies and download models into app data
	INFORMITY_INSTALL_PROFILE=dev ./scripts/install_app.sh

uninstall: ## Remove all user data, downloaded models, and .venv (fresh distribution state)
	./scripts/install_uninstall_app.sh

run: ## Run the application server (no reload — use for production or heavy indexing)
	uv run python -m informity.main

dev: ## Run with auto-reload for development (code changes restart the server)
	INFORMITY_DEV_RELOAD=true uv run uvicorn informity.main:app --host $(HOST) --port $(PORT) --reload --log-level info

frontend: ## Run Vite dev server (hot reload) — use with backend: make run or make dev in another terminal
	cd src/frontend && npm run dev

frontend-build: ## Build frontend for production (output: src/frontend/dist/)
	cd src/frontend && npm run build

tauri-dev: ## Run desktop shell in development mode (requires Rust toolchain + Tauri CLI)
	cd src/frontend && npm run tauri:dev

tauri-icons: ## Generate Tauri icon assets from the master logo
	uv run python scripts/build_generate_tauri_icons.py

tauri-backend: ## Build Python backend sidecar artifact for Tauri packaging
	./scripts/build_tauri_backend_sidecar.sh

tauri-build: ## Build desktop bundle artifacts (requires Rust toolchain + Tauri CLI)
	cd src/frontend && npm run tauri:build

app: frontend-build run ## Build frontend and run app on http://127.0.0.1:8420 (single command for testing)

test: ## Run the test suite
	uv run python -m pytest tests/ -v

qa-quick: ## On-demand quick quality gate (backend reliability smoke tests + frontend type/test/build)
	python3 tools/qa/docs_lint.py
	uv run python -m pytest tests/test_api_security.py tests/test_index_integrity.py tests/test_rag_split_contract.py tests/test_diagnostics_metrics_contract.py tests/test_json_serializer_contract.py -v
	cd src/frontend && npm run typecheck && npm run test && npm run build

qa-full: ## On-demand full quality gate (all backend tests + frontend type/test/build)
	python3 tools/qa/docs_lint.py
	uv run python -m pytest tests/ -v
	cd src/frontend && npm run typecheck && npm run test && npm run build

qa-docs: ## On-demand internal docs lint (headers, statuses, internal links)
	python3 tools/qa/docs_lint.py

qa-lint: ## On-demand code style checks (non-release-blocking while baseline debt exists)
	uv run ruff check src/ tests/

qa-typecheck: ## On-demand TypeScript strict checks (tracked separately from release gate)
	cd src/frontend && npm run typecheck

qa-security: qa-secrets ## On-demand security gate (secret scan + dependency vulnerability audit)
	@echo "Running pip-audit (CVE-2025-69872 ignored until upstream diskcache fix is available)."
	uv run --with pip-audit pip-audit --ignore-vuln CVE-2025-69872

qa-secrets: ## On-demand local secret scan using regex heuristics
	uv run python tools/qa/secret_scan.py

baseline: ## Run performance baseline (server must be running: make dev)
	uv run python tools/performance/performance_baseline.py --output tools/performance-baseline.md

diagnostics-evaluate: ## Run evaluation queries against all models
	uv run python tools/diagnostics/evaluate.py

diagnostics-analyze: ## Analyze diagnostics metrics and generate report
	uv run python tools/diagnostics/analyze.py --days 30 --type evaluation --diagnose --format markdown

diagnostics-pipeline: ## Run full diagnostics pipeline (evaluate → analyze → generate tasks)
	uv run python tools/diagnostics/pipeline.py

diagnostics-stop: ## Stop active diagnostics run via lock/PID control
	uv run python tools/diagnostics/stop.py

smoke-basic: ## Run lightweight smoke checks for index/search/chat wiring
	uv run python tools/smoke/smoke_basic_operations.py

smoke-infra: ## Run infrastructure smoke checks for scanner/indexer/db contracts
	uv run python tools/smoke/smoke_infrastructure.py

smoke-pdf: ## Run PDF extractor smoke script against provided PDF paths/dir
	uv run python tools/smoke/smoke_pdf_extraction.py

maintenance-index-check: ## Check cross-store index integrity issues
	uv run python tools/maintenance/index_integrity.py

maintenance-index-repair: ## Repair cross-store index integrity issues
	uv run python tools/maintenance/index_integrity.py --repair

maintenance-download-nltk: ## Download NLTK stopwords corpus (temporary privacy toggle)
	uv run python tools/maintenance/download_nltk_data.py

maintenance-reinstall-packages: ## Recreate .venv and reinstall dependencies
	uv run python tools/maintenance/reinstall_packages.py

maintenance-chunk-structure: ## Analyze chunk parent/child structure for integrity anomalies
	uv run python tools/maintenance/chunk_structure_analysis.py

maintenance-orphaned-chunks: ## Diagnose orphaned chunks (missing/invalid parent_id links)
	uv run python tools/maintenance/orphaned_chunks_diagnostic.py

lint: ## Run linter checks (ruff)
	uv run ruff check src/ tests/

format: ## Auto-format code (ruff)
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

reset-db: ## Delete the SQLite database (will be recreated on next run)
	rm -f "$(APP_DATA_DIR)/db/informity.db"
	rm -f "$(APP_DATA_DIR)/db/informity.db-journal"
	rm -f "$(APP_DATA_DIR)/db/informity.db-wal"
	rm -f "$(APP_DATA_DIR)/db/informity.db-shm"
	rm -f "$(APP_DATA_DIR)/informity.db"
	rm -f "$(APP_DATA_DIR)/informity.db-journal"
	rm -f "$(APP_DATA_DIR)/informity.db-wal"
	rm -f "$(APP_DATA_DIR)/informity.db-shm"
	@echo "Database reset. It will be recreated on next app start."

reset-all: reset-db ## Reset indexed data (vectors are stored in SQLite)
	@echo "All data reset."

clean-data: ## Remove unnecessary files from data/ (locks, .no_exist under HF cache). Keeps config, db, vectors, models, embedding and reranker.
	@echo "Cleaning data/ (locks, .no_exist under cache/huggingface)..."
	rm -rf "$(APP_HF_HUB_DIR)/.locks"
	find "$(APP_HF_HUB_DIR)" -type d -name ".no_exist" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean-data done."

clean: ## Remove build artifacts, caches, and local data
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find src tests -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/
	@echo "Cleaned build artifacts and caches."
