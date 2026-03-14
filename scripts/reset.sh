#!/usr/bin/env bash
# ==============================================================================
# Informity AI — Reset user data state
# Removes app_data contents (config, database, vectors-in-SQLite, logs, diagnostics)
# so the app behaves like first run for user data/config. Application assets remain:
# cache/model directories at repo root (for example .cache/ and tools/diagnostics/models/).
# Virtualenv is left intact so you can run the app immediately after reset.
# Run from repo root: ./scripts/reset.sh   or   bash scripts/reset.sh
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Same default as install and the app (data/ relative to repo root)
APP_DATA_DIR="${INFORMITY_APP_DATA_DIR:-$REPO_ROOT/data}"
if [[ "$APP_DATA_DIR" != /* ]]; then
    APP_DATA_DIR="$REPO_ROOT/$APP_DATA_DIR"
fi

echo "Informity AI — Reset user data state"
echo "  Repo root:    $REPO_ROOT"
echo "  App data dir: $APP_DATA_DIR"
echo ""

# ------------------------------------------------------------------------------
# Remove user data only (config, database, vectors in SQLite, logs, diagnostics).
# Application assets (.cache/ and tools/diagnostics/models/) are preserved.
# ------------------------------------------------------------------------------
if [[ -d "$APP_DATA_DIR" ]]; then
    echo "Removing user data: $APP_DATA_DIR"
    echo "  (Preserving application assets: .cache/ and tools/diagnostics/models/)"
    rm -rf "$APP_DATA_DIR"
    echo "Done. Application is reset; next start will use default config and no user data."
    echo "  .cache/ and tools/diagnostics/models/ are preserved."
else
    echo "App data dir not present: $APP_DATA_DIR (already reset for user data)."
fi

echo ""
echo "Start the app with: make run   or   uv run uvicorn informity.main:app --host 127.0.0.1 --port 8420"
echo "To download models again for offline use: ./scripts/install.sh   or   uv run python scripts/bootstrap_models.py"
echo ""
