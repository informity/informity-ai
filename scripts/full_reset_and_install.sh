#!/usr/bin/env bash
# ==============================================================================
# Informity AI — Full Reset and Install
# Resets all data, venv, node_modules, and .cache (except model directories), then
# runs install and tests. Use for a clean slate while preserving large LLM models.
# Run from repo root: ./scripts/full_reset_and_install.sh
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

APP_DATA_DIR="${INFORMITY_APP_DATA_DIR:-$REPO_ROOT/data}"
if [[ "$APP_DATA_DIR" != /* ]]; then
    APP_DATA_DIR="$REPO_ROOT/$APP_DATA_DIR"
fi

echo "Informity AI — Full Reset and Install"
echo "  Repo root:    $REPO_ROOT"
echo "  App data dir: $APP_DATA_DIR"
TOOLS_DIR="tools"
DIAGNOSTICS_DIR_NAME="diagnostics"
DIAGNOSTICS_MODELS_DIR_NAME="models"
DIAGNOSTICS_DIR="$REPO_ROOT/$TOOLS_DIR/$DIAGNOSTICS_DIR_NAME/$DIAGNOSTICS_MODELS_DIR_NAME"
echo "  Preserving:   .cache/chat-llm/, .cache/query-classifier-llm/, $TOOLS_DIR/$DIAGNOSTICS_DIR_NAME/$DIAGNOSTICS_MODELS_DIR_NAME/"
echo ""

# ------------------------------------------------------------------------------
# 1. Remove app data (config, database, logs)
# ------------------------------------------------------------------------------
if [[ -d "$APP_DATA_DIR" ]]; then
    echo "Removing app data: $APP_DATA_DIR"
    rm -rf "$APP_DATA_DIR"
else
    echo "App data dir not present: $APP_DATA_DIR"
fi

# ------------------------------------------------------------------------------
# 2. Remove virtualenv
# ------------------------------------------------------------------------------
VENV_DIR="$REPO_ROOT/.venv"
if [[ -d "$VENV_DIR" ]]; then
    echo "Removing virtualenv: $VENV_DIR"
    rm -rf "$VENV_DIR"
else
    echo "Virtualenv not present: $VENV_DIR"
fi

# ------------------------------------------------------------------------------
# 3. Remove node_modules
# ------------------------------------------------------------------------------
NODE_MODULES="$REPO_ROOT/src/frontend/node_modules"
if [[ -d "$NODE_MODULES" ]]; then
    echo "Removing node_modules: $NODE_MODULES"
    rm -rf "$NODE_MODULES"
else
    echo "node_modules not present: $NODE_MODULES"
fi

# ------------------------------------------------------------------------------
# 4. Reset .cache EXCEPT model directories
# ------------------------------------------------------------------------------
CACHE_DIR="$REPO_ROOT/.cache"
LLM_DIR="$CACHE_DIR/chat-llm"
QUERY_CLASSIFIER_DIR="$CACHE_DIR/query-classifier-llm"
if [[ -d "$CACHE_DIR" ]]; then
    echo "Resetting .cache (keeping model directories)..."
    for item in "$CACHE_DIR"/*; do
        if [[ -e "$item" ]] && [[ "$item" != "$LLM_DIR" ]] && [[ "$item" != "$QUERY_CLASSIFIER_DIR" ]]; then
            echo "  Removing: $item"
            rm -rf "$item"
        fi
    done
    if [[ -d "$LLM_DIR" ]]; then
        echo "  Preserved: $LLM_DIR"
    fi
    if [[ -d "$QUERY_CLASSIFIER_DIR" ]]; then
        echo "  Preserved: $QUERY_CLASSIFIER_DIR"
    fi
    if [[ -d "$DIAGNOSTICS_DIR" ]]; then
        echo "  Preserved: $DIAGNOSTICS_DIR"
    fi
else
    echo ".cache not present"
fi

# ------------------------------------------------------------------------------
# 5. Remove test/build caches
# ------------------------------------------------------------------------------
for dir in .pytest_cache .ruff_cache htmlcov; do
    if [[ -d "$REPO_ROOT/$dir" ]]; then
        echo "Removing $dir"
        rm -rf "$REPO_ROOT/$dir"
    fi
done
[[ -f "$REPO_ROOT/.coverage" ]] && rm -f "$REPO_ROOT/.coverage"

echo ""
echo "Reset complete. Running install..."
echo ""

# ------------------------------------------------------------------------------
# 6. Run install script (creates venv, npm install, downloads models)
# ------------------------------------------------------------------------------
export INFORMITY_APP_DATA_DIR="$APP_DATA_DIR"
./scripts/install.sh

echo ""
echo "Initializing database for tests..."
uv run python -c "
import asyncio
from informity.config import settings
settings.ensure_directories()
from informity.db.sqlite import init_db
asyncio.run(init_db())
print('Database initialized.')
"

echo ""
echo "Running tests..."
echo ""

# ------------------------------------------------------------------------------
# 7. Run all tests
# ------------------------------------------------------------------------------
uv run pytest tests/ -v

echo ""
echo "Done. Full reset, install, and tests complete."
echo ""
