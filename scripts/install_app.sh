#!/usr/bin/env bash
# ==============================================================================
# Informity AI — Install
# Installs Python/frontend deps and downloads embedding, reranker, and optional
# LLM into app data, then configures cached/offline runtime defaults.
#
# Optional flags:
# - INFORMITY_INSTALL_PROFILE=runtime|dev (default: runtime)
# - INFORMITY_INSTALL_CLEAN=1           (run uninstall first)
# - INFORMITY_INSTALL_VERIFY=1          (init DB + run tests at end)
# - INFORMITY_INSTALL_SKIP_MODELS=1     (install deps only; do not download models)
#   Example: INFORMITY_INSTALL_PROFILE=dev INFORMITY_INSTALL_SKIP_MODELS=1 ./scripts/install_app.sh
#
# CLI shortcuts (equivalent to env vars):
# - --no-models
# - --clean
# - --verify
# - --dev | --runtime
# - -h | --help
#
# Run from repo root: ./scripts/install_app.sh   or   bash scripts/install_app.sh
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/install_common_paths.sh"

DIR_CACHE="cache"
DIR_MODELS="models"
DIR_LLM="llm"

INSTALL_PROFILE="${INFORMITY_INSTALL_PROFILE:-runtime}"
INSTALL_CLEAN="${INFORMITY_INSTALL_CLEAN:-0}"
INSTALL_VERIFY="${INFORMITY_INSTALL_VERIFY:-0}"
INSTALL_SKIP_MODELS="${INFORMITY_INSTALL_SKIP_MODELS:-0}"

print_help() {
    cat <<'EOF'
Informity AI install script

Usage:
  ./scripts/install_app.sh [options]

Options:
  --no-models   Install dependencies only; skip model downloads.
  --clean       Run uninstall cleanup before install.
  --verify      Initialize DB and run tests at the end.
  --dev         Install dev profile (uv sync --all-extras).
  --runtime     Install runtime profile (default).
  -h, --help    Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-models)
            INSTALL_SKIP_MODELS=1
            ;;
        --clean)
            INSTALL_CLEAN=1
            ;;
        --verify)
            INSTALL_VERIFY=1
            ;;
        --dev)
            INSTALL_PROFILE="dev"
            ;;
        --runtime)
            INSTALL_PROFILE="runtime"
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            print_help
            exit 1
            ;;
    esac
    shift
done

# App data directory: same default as bundled desktop app.
APP_DATA_DIR="${INFORMITY_APP_DATA_DIR:-$INFORMITY_DEFAULT_APP_DATA_DIR}"
export INFORMITY_APP_DATA_DIR="$APP_DATA_DIR"
# Keep cache and model paths aligned with bundled desktop runtime defaults.
export INFORMITY_CACHE_DIR="${INFORMITY_CACHE_DIR:-$APP_DATA_DIR/$DIR_CACHE}"
export INFORMITY_MODELS_DIR="${INFORMITY_MODELS_DIR:-$APP_DATA_DIR/$DIR_MODELS/$DIR_LLM}"

echo "Informity AI — Install"
echo "  Repo root:    $REPO_ROOT"
echo "  App data dir: $APP_DATA_DIR"
echo "  Profile:      $INSTALL_PROFILE"
echo "  Clean first:  $INSTALL_CLEAN"
echo "  Verify:       $INSTALL_VERIFY"
echo "  Skip models:  $INSTALL_SKIP_MODELS"
echo ""

if [[ "$INSTALL_CLEAN" == "1" ]]; then
    echo "Running uninstall cleanup first (INFORMITY_INSTALL_CLEAN=1)..."
    ./scripts/install_uninstall_app.sh
    echo ""
fi

# ------------------------------------------------------------------------------
# 1. Ensure uv is available
# ------------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Create virtual environment with uv-managed Python
# ------------------------------------------------------------------------------
if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment with Python 3.13 (uv-managed)..."
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
# 4. Install frontend dependencies
# ------------------------------------------------------------------------------
echo ""
echo "Installing frontend dependencies (npm install)..."
(cd src/frontend && npm install)

# ------------------------------------------------------------------------------
# 5. Download models into app data and set offline config (optional)
#    Set INFORMITY_INSTALL_SKIP_MODELS=1 to simulate shipped-app first run
#    where dependencies are installed but models are missing and setup UI
#    drives model download.
# ------------------------------------------------------------------------------
if [[ "$INSTALL_SKIP_MODELS" == "1" ]]; then
    echo ""
    echo "Skipping model download (INFORMITY_INSTALL_SKIP_MODELS=1)."
    echo "First run will show setup and prompt model download."
else
    echo ""
    echo "Downloading models (embedding, reranker, docling, optional LLM from scripts/install.conf.json)..."
    export INFORMITY_FULL_PRIVACY=false
    export INFORMITY_EMBEDDING_OFFLINE=false
    export INFORMITY_LLM_LOCAL_ONLY=false
    uv run python scripts/install_bootstrap_models.py
fi

# ------------------------------------------------------------------------------
# 6. Optional verification
# ------------------------------------------------------------------------------
if [[ "$INSTALL_VERIFY" == "1" ]]; then
    echo ""
    echo "Initializing database for verification..."
    uv run python -c "
import asyncio
from informity.config import settings
settings.ensure_directories()
from informity.db.sqlite import init_db
asyncio.run(init_db())
print('Database initialized.')
"

    echo ""
    echo "Running tests (uv run pytest tests/ -v)..."
    uv run pytest tests/ -v
fi

echo ""
echo "Done. Start the app with:"
echo "  export INFORMITY_APP_DATA_DIR=\"$APP_DATA_DIR\""
echo "  uv run uvicorn informity.main:app --host 127.0.0.1 --port 8420"
echo "Or from repo root: make run"
echo ""
if [[ "$INSTALL_SKIP_MODELS" == "1" ]]; then
    echo "Models were not preinstalled. App will run setup flow on first launch."
else
    echo "The app will use cached models only (embedding_offline=true, llm_local_only=true) and will not contact Hugging Face or the internet after install."
fi
echo ""
