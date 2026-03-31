#!/usr/bin/env bash
# ==============================================================================
# Informity AI — Uninstall (full local cleanup)
# Removes app data, local environments, and generated caches/artifacts so the
# repo returns to a fresh local state.
# Run from repo root: ./scripts/uninstall.sh   or   bash scripts/uninstall.sh
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/common_paths.sh"

# Same default as install and the app (~/.informity)
APP_DATA_DIR="${INFORMITY_APP_DATA_DIR:-$INFORMITY_DEFAULT_APP_DATA_DIR}"
if [[ "$APP_DATA_DIR" != /* ]]; then
    APP_DATA_DIR="$REPO_ROOT/$APP_DATA_DIR"
fi

echo "Informity AI — Uninstall"
echo "  Repo root:    $REPO_ROOT"
echo "  App data dir: $APP_DATA_DIR"
echo ""

REMOVED=()

# ------------------------------------------------------------------------------
# 1. Remove app data (includes setup state, config, db, logs, cache, chats, models)
# ------------------------------------------------------------------------------
if [[ -d "$APP_DATA_DIR" ]]; then
    echo "Removing app data: $APP_DATA_DIR"
    rm -rf "$APP_DATA_DIR"
    REMOVED+=("$APP_DATA_DIR")
else
    echo "App data dir not present: $APP_DATA_DIR"
fi

# ------------------------------------------------------------------------------
# 2. Remove local runtime/dependency environments
# ------------------------------------------------------------------------------
for dir in ".venv" "src/frontend/node_modules"; do
    if [[ -d "$REPO_ROOT/$dir" ]]; then
        echo "Removing $dir"
        rm -rf "$REPO_ROOT/$dir"
        REMOVED+=("$REPO_ROOT/$dir")
    fi
done

# ------------------------------------------------------------------------------
# 3. Remove local caches/build artifacts
# ------------------------------------------------------------------------------
for dir in ".cache" ".pytest_cache" ".ruff_cache" "htmlcov" "src/frontend/dist"; do
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
    echo "Nothing to remove (already at fresh local state)."
fi
echo ""
echo "To install again, run: ./scripts/install.sh"
echo ""
