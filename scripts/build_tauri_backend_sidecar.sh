#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BACKEND_NAME="informity-backend"
if [[ "${OS:-}" == "Windows_NT" ]] || [[ "$(uname -s)" =~ MINGW|MSYS|CYGWIN ]]; then
  BACKEND_NAME="${BACKEND_NAME}.exe"
fi

DIST_DIR="$ROOT_DIR/build/tauri-backend/dist"
WORK_DIR="$ROOT_DIR/build/tauri-backend/work"
SPEC_DIR="$ROOT_DIR/build/tauri-backend/spec"
OUT_DIR="$ROOT_DIR/src/frontend/src-tauri/backend"

mkdir -p "$DIST_DIR" "$WORK_DIR" "$SPEC_DIR" "$OUT_DIR"

verify_sidecar_contents() {
  local sidecar_bin="$1"
  local listing_file

  listing_file="$(mktemp)"
  uv run --with pyinstaller python -m PyInstaller.utils.cliutils.archive_viewer -r -b "$sidecar_bin" >"$listing_file" 2>&1 || true

  local -a required_patterns=(
    "docling/models/plugins/__init__\\.py"
    "docling_ibm_models/__init__\\.py"
    "docx/__init__\\.py"
    "docx/document\\.py"
    "docling-[0-9].*\\.dist-info/METADATA"
    "docling_core-[0-9].*\\.dist-info/METADATA"
    "docling_parse/.*\\.(so|dylib|pyd)"
    "docling_parse/pdf_resources/"
    "pandas/_libs/algos\\.cpython-"
    "sqlite_vec/__init__\\.py"
    "sqlite_vec/vec0\\.(so|dylib|dll)"
    "promptcue-[0-9].*\\.dist-info/METADATA"
    "promptcue/__init__\\.py"
    "thinkstrip-[0-9].*\\.dist-info/METADATA"
    "thinkstrip/__init__\\.py"
  )

  for pattern in "${required_patterns[@]}"; do
    if ! grep -Eq "$pattern" "$listing_file"; then
      echo "ERROR: sidecar verification failed (missing: $pattern)" >&2
      rm -f "$listing_file"
      return 1
    fi
  done

  rm -f "$listing_file"
}

echo "Building Tauri backend sidecar (${BACKEND_NAME})..."
uv run --with pyinstaller pyinstaller \
  --noconfirm \
  --clean \
  --onefile \
  --name "$BACKEND_NAME" \
  --exclude-module pytest \
  --exclude-module pytest_asyncio \
  --exclude-module tkinter \
  --exclude-module _tkinter \
  --exclude-module nltk \
  --exclude-module nltk.test \
  --exclude-module thinc.tests \
  --exclude-module torch.fx.passes.tests \
  --collect-data xllamacpp \
  --collect-data tiktoken \
  --collect-all docling \
  --collect-all docling_parse \
  --collect-all docling_ibm_models \
  --collect-all docx \
  --collect-all promptcue \
  --collect-all thinkstrip \
  --collect-binaries xllamacpp \
  --copy-metadata docling \
  --copy-metadata docling-core \
  --copy-metadata docling-parse \
  --copy-metadata docling-ibm-models \
  --copy-metadata promptcue \
  --copy-metadata thinkstrip \
  --hidden-import pandas \
  --collect-all sqlite_vec \
  --hidden-import tiktoken_ext \
  --hidden-import tiktoken_ext.openai_public \
  --paths "$ROOT_DIR/src" \
  --distpath "$DIST_DIR" \
  --workpath "$WORK_DIR" \
  --specpath "$SPEC_DIR" \
  "$ROOT_DIR/src/informity/main.py"

verify_sidecar_contents "$DIST_DIR/$BACKEND_NAME"

cp "$DIST_DIR/$BACKEND_NAME" "$OUT_DIR/$BACKEND_NAME"
chmod +x "$OUT_DIR/$BACKEND_NAME" || true

echo "Sidecar ready: $OUT_DIR/$BACKEND_NAME"
