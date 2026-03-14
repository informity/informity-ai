#!/usr/bin/env bash
# ==============================================================================
# Informity AI — Uninstall
# Removes all user data and downloaded content, returning the tree to the state
# of a fresh distribution (as if you had just cloned the repo).
# Run from repo root: ./scripts/uninstall.sh   or   bash scripts/uninstall.sh
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Same default as install and the app (data/ relative to repo root)
APP_DATA_DIR="${INFORMITY_APP_DATA_DIR:-$REPO_ROOT/data}"
if [[ "$APP_DATA_DIR" != /* ]]; then
    APP_DATA_DIR="$REPO_ROOT/$APP_DATA_DIR"
fi

echo "Informity AI — Uninstall"
echo "  Repo root:    $REPO_ROOT"
echo "  App data dir: $APP_DATA_DIR"
echo ""

REMOVED=()

# ------------------------------------------------------------------------------
# 1. Remove app data (config, database, cache, models, vectors, logs)
# ------------------------------------------------------------------------------
if [[ -d "$APP_DATA_DIR" ]]; then
    echo "Removing app data: $APP_DATA_DIR"
    rm -rf "$APP_DATA_DIR"
    REMOVED+=("$APP_DATA_DIR")
else
    echo "App data dir not present: $APP_DATA_DIR"
fi

# ------------------------------------------------------------------------------
# 2. Remove virtualenv (so next install runs uv sync from scratch)
# ------------------------------------------------------------------------------
VENV_DIR="$REPO_ROOT/.venv"
if [[ -d "$VENV_DIR" ]]; then
    echo "Removing virtualenv: $VENV_DIR"
    rm -rf "$VENV_DIR"
    REMOVED+=("$VENV_DIR")
else
    echo "Virtualenv not present: $VENV_DIR"
fi

# ------------------------------------------------------------------------------
# 3. Remove local caches that are not part of the distribution
# ------------------------------------------------------------------------------
for dir in .pytest_cache .ruff_cache htmlcov; do
    if [[ -d "$REPO_ROOT/$dir" ]]; then
        echo "Removing $dir"
        rm -rf "$REPO_ROOT/$dir"
        REMOVED+=("$REPO_ROOT/$dir")
    fi
done
if [[ -f "$REPO_ROOT/.coverage" ]]; then
    rm -f "$REPO_ROOT/.coverage"
    REMOVED+=("$REPO_ROOT/.coverage")
fi

echo ""
if [[ ${#REMOVED[@]} -gt 0 ]]; then
    echo "Uninstall complete. Removed:"
    for path in "${REMOVED[@]}"; do
        echo "  - $path"
    done
else
    echo "Nothing to remove (already at fresh distribution state)."
fi
echo ""
echo "To install again, run: ./scripts/install.sh"
echo ""
