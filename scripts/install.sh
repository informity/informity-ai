#!/usr/bin/env bash
# ==============================================================================
# Informity AI — One-time install
# Installs Python deps and downloads embedding, reranker, optional LLM, and
# classifier LLM into app data, then configures the app to always use cached
# models (no auto-download at runtime).
# Run from repo root: ./scripts/install.sh   or   bash scripts/install.sh
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

APP_DISPLAY_NAME="Informity AI"
DIR_CACHE="cache"
DIR_MODELS="models"
DIR_CHAT_LLM="chat-llm"
DIR_QUERY_CLASSIFIER_LLM="query-classifier-llm"

# Install profile:
# - runtime (default): application runtime deps only
# - dev: runtime + optional dev tooling (pytest, ruff, pyinstaller, etc.)
INSTALL_PROFILE="${INFORMITY_INSTALL_PROFILE:-runtime}"

# App data directory: same default as bundled desktop app.
APP_DATA_DIR="${INFORMITY_APP_DATA_DIR:-$HOME/Library/Application Support/$APP_DISPLAY_NAME}"
export INFORMITY_APP_DATA_DIR="$APP_DATA_DIR"
# Keep cache and model paths aligned with bundled desktop runtime defaults.
export INFORMITY_CACHE_DIR="${INFORMITY_CACHE_DIR:-$APP_DATA_DIR/$DIR_CACHE}"
export INFORMITY_MODELS_DIR="${INFORMITY_MODELS_DIR:-$APP_DATA_DIR/$DIR_MODELS/$DIR_CHAT_LLM}"
export INFORMITY_QUERY_CLASSIFIER_MODELS_DIR="${INFORMITY_QUERY_CLASSIFIER_MODELS_DIR:-$APP_DATA_DIR/$DIR_MODELS/$DIR_QUERY_CLASSIFIER_LLM}"

echo "Informity AI — Install"
echo "  Repo root:    $REPO_ROOT"
echo "  App data dir: $APP_DATA_DIR"
echo "  Profile:      $INSTALL_PROFILE"
echo ""

# ------------------------------------------------------------------------------
# 1. Ensure uv is available
# ------------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Create virtual environment with uv-managed Python (supports SQLite extensions)
# ------------------------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.13 (uv-managed, supports SQLite extensions)..."
    uv venv --python 3.13
else
    echo "Virtual environment already exists (.venv/)"
fi

# ------------------------------------------------------------------------------
# 3. Install Python dependencies
# ------------------------------------------------------------------------------
case "$INSTALL_PROFILE" in
    runtime)
        echo "Installing Python runtime dependencies (uv sync)..."
        uv sync
        ;;
    dev)
        echo "Installing Python dependencies with dev extras (uv sync --all-extras)..."
        uv sync --all-extras
        ;;
    *)
        echo "Invalid INFORMITY_INSTALL_PROFILE: $INSTALL_PROFILE (expected: runtime|dev)"
        exit 1
        ;;
esac

# ------------------------------------------------------------------------------
# 4. Install frontend dependencies (including TypeScript)
# ------------------------------------------------------------------------------
echo ""
echo "Installing frontend dependencies (npm install)..."
(cd src/frontend && npm install)

# ------------------------------------------------------------------------------
# 5. Download models into app data and set offline config
# ------------------------------------------------------------------------------
# Temporarily disable privacy so bootstrap can download models. Bootstrap will
# re-enable privacy at the end when all models are cached.
echo ""
echo "Downloading models (embedding, reranker, docling, optional LLM, classifier LLM from scripts/install.conf.json)..."
export INFORMITY_FULL_PRIVACY=false
export INFORMITY_EMBEDDING_OFFLINE=false
export INFORMITY_LLM_LOCAL_ONLY=false
uv run python scripts/bootstrap_models.py

echo ""
echo "Done. Start the app with:"
echo "  export INFORMITY_APP_DATA_DIR=\"$APP_DATA_DIR\""
echo "  uv run uvicorn informity.main:app --host 127.0.0.1 --port 8420"
echo "Or from repo root: make run"
echo ""
echo "The app will use cached models only (embedding_offline=true, llm_local_only=true) and will not contact Hugging Face or the internet—no network requests after install."
