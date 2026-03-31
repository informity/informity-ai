#!/usr/bin/env bash
# ==============================================================================
# Informity AI — Reset user data state
# Removes app_data user-data contents (config, database, vectors-in-SQLite, logs, diagnostics)
# so the app behaves like first run for user data/config.
# Preserves persistent model directories under app_data/models/ and diagnostics models
# under tools/diagnostics/models/.
# Virtualenv is left intact so you can run the app immediately after reset.
# Run from repo root: ./scripts/reset.sh   or   bash scripts/reset.sh
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common_paths.sh"

DIR_MODELS="models"

# Same default as install and the app (~/.informity)
APP_DATA_DIR="${INFORMITY_APP_DATA_DIR:-$INFORMITY_DEFAULT_APP_DATA_DIR}"
if [[ "$APP_DATA_DIR" != /* ]]; then
    APP_DATA_DIR="$REPO_ROOT/$APP_DATA_DIR"
fi

echo "Informity AI — Reset user data state"
echo "  Repo root:    $REPO_ROOT"
echo "  App data dir: $APP_DATA_DIR"
echo ""

# ------------------------------------------------------------------------------
# Remove user data only (config, database, vectors in SQLite, logs, diagnostics).
# Preserve app_data/models/ (chat + classifier models) and tools/diagnostics/models/.
# ------------------------------------------------------------------------------
if [[ -d "$APP_DATA_DIR" ]]; then
    echo "Removing user data: $APP_DATA_DIR"
    MODELS_DIR="$APP_DATA_DIR/$DIR_MODELS"
    PRESERVED_MODELS_TMP=""
    if [[ -d "$MODELS_DIR" ]]; then
        PRESERVED_MODELS_TMP="$(mktemp -d)"
        mv "$MODELS_DIR" "$PRESERVED_MODELS_TMP/$DIR_MODELS"
        echo "  Preserving: $MODELS_DIR"
    fi

    rm -rf "$APP_DATA_DIR"

    if [[ -n "$PRESERVED_MODELS_TMP" ]] && [[ -d "$PRESERVED_MODELS_TMP/$DIR_MODELS" ]]; then
        mkdir -p "$APP_DATA_DIR"
        mv "$PRESERVED_MODELS_TMP/$DIR_MODELS" "$APP_DATA_DIR/$DIR_MODELS"
        rmdir "$PRESERVED_MODELS_TMP" 2>/dev/null || true
    fi

    echo "Done. Application is reset; next start will use default config and no user data."
    echo "  app_data/models/ and tools/diagnostics/models/ are preserved."
else
    echo "App data dir not present: $APP_DATA_DIR (already reset for user data)."
fi

echo ""
echo "Start the app with: make run   or   uv run uvicorn informity.main:app --host 127.0.0.1 --port 8420"
echo "To download models again for offline use: ./scripts/install.sh   or   uv run python scripts/bootstrap_models.py"
echo ""
