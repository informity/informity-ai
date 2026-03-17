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

APP_DISPLAY_NAME="Informity AI"
DIR_MODELS="models"
DIR_CACHE=".cache"
DIR_HUGGINGFACE="huggingface"
DIR_DOCLING="docling"

APP_DATA_DIR="${INFORMITY_APP_DATA_DIR:-$HOME/Library/Application Support/$APP_DISPLAY_NAME}"
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
echo "  Preserving:   $APP_DATA_DIR/models/, $TOOLS_DIR/$DIAGNOSTICS_DIR_NAME/$DIAGNOSTICS_MODELS_DIR_NAME/"
echo ""

# ------------------------------------------------------------------------------
# 1. Remove app data (config, database, logs), preserving app_data/models
# ------------------------------------------------------------------------------
if [[ -d "$APP_DATA_DIR" ]]; then
    echo "Removing app data: $APP_DATA_DIR"
    MODELS_DIR="$APP_DATA_DIR/$DIR_MODELS"
    PRESERVED_MODELS_TMP=""
    if [[ -d "$MODELS_DIR" ]]; then
        PRESERVED_MODELS_TMP="$(mktemp -d)"
        mv "$MODELS_DIR" "$PRESERVED_MODELS_TMP/$DIR_MODELS"
        echo "  Preserved: $MODELS_DIR"
    fi

    rm -rf "$APP_DATA_DIR"

    if [[ -n "$PRESERVED_MODELS_TMP" ]] && [[ -d "$PRESERVED_MODELS_TMP/$DIR_MODELS" ]]; then
        mkdir -p "$APP_DATA_DIR"
        mv "$PRESERVED_MODELS_TMP/$DIR_MODELS" "$APP_DATA_DIR/$DIR_MODELS"
        rmdir "$PRESERVED_MODELS_TMP" 2>/dev/null || true
    fi
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
# 4. Reset .cache EXCEPT Hugging Face/docling artifacts needed for warm cache
# ------------------------------------------------------------------------------
CACHE_DIR="$REPO_ROOT/$DIR_CACHE"
if [[ -d "$CACHE_DIR" ]]; then
    echo "Resetting .cache (keeping huggingface/docling caches)..."
    for item in "$CACHE_DIR"/*; do
        if [[ -e "$item" ]] && [[ "$item" != "$CACHE_DIR/$DIR_HUGGINGFACE" ]] && [[ "$item" != "$CACHE_DIR/$DIR_DOCLING" ]]; then
            echo "  Removing: $item"
            rm -rf "$item"
        fi
    done
    if [[ -d "$CACHE_DIR/$DIR_HUGGINGFACE" ]]; then
        echo "  Preserved: $CACHE_DIR/$DIR_HUGGINGFACE"
    fi
    if [[ -d "$CACHE_DIR/$DIR_DOCLING" ]]; then
        echo "  Preserved: $CACHE_DIR/$DIR_DOCLING"
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
export INFORMITY_INSTALL_PROFILE=dev
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
