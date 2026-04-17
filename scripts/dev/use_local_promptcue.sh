#!/usr/bin/env bash
set -euo pipefail

PROMPTCUE_SRC="${1:-/Users/dgerasimenko/Projects/informity.com/promptcue/src}"

if [ ! -d "$PROMPTCUE_SRC/promptcue" ]; then
  echo "[ERROR] PromptCue source not found at: $PROMPTCUE_SRC" >&2
  echo "Usage: scripts/dev/use_local_promptcue.sh [/absolute/path/to/promptcue/src]" >&2
  exit 1
fi

if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$PROMPTCUE_SRC:$PYTHONPATH"
else
  export PYTHONPATH="$PROMPTCUE_SRC"
fi

echo "[INFO] PYTHONPATH set for this shell: $PROMPTCUE_SRC"
IMPORT_PATH="$(uv run python - <<'PY'
import promptcue
print(promptcue.__file__)
PY
)"

echo "[INFO] promptcue import path: $IMPORT_PATH"
case "$IMPORT_PATH" in
  "$PROMPTCUE_SRC"/*)
    echo "[OK] Local PromptCue wiring active."
    ;;
  *)
    echo "[ERROR] promptcue did not resolve from local source." >&2
    echo "[ERROR] Expected prefix: $PROMPTCUE_SRC" >&2
    exit 2
    ;;
esac

echo "\nRun commands in this shell session, e.g.:"
echo "  uv run pytest tests/test_promptcue_adapter.py tests/test_query_classifier.py -q"
echo "\nTo disable temporary wiring:"
echo "  unset PYTHONPATH"
