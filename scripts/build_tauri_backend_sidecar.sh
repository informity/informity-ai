#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BACKEND_NAME="informity-backend"
if [[ "${OS:-}" == "Windows_NT" ]] || [[ "$(uname -s)" =~ MINGW|MSYS|CYGWIN ]]; then
  BACKEND_NAME="${BACKEND_NAME}.exe"
fi
BACKEND_BUNDLE_DIR="informity-backend-bundle"

DIST_DIR="$ROOT_DIR/build/tauri-backend/dist"
WORK_DIR="$ROOT_DIR/build/tauri-backend/work"
SPEC_DIR="$ROOT_DIR/build/tauri-backend/spec"
OUT_DIR="$ROOT_DIR/src/frontend/src-tauri/backend"

mkdir -p "$DIST_DIR" "$WORK_DIR" "$SPEC_DIR" "$OUT_DIR"

CHARSET_MYPYC_MODULE="$(uv run python - <<'PY'
from __future__ import annotations
from pathlib import Path
import sys

matches: list[Path] = []
for base in map(Path, sys.path):
    if not base.is_dir():
        continue
    if "site-packages" not in str(base):
        continue
    matches.extend(sorted(base.glob("*__mypyc.cpython-*.so")))

print(matches[0].stem.split(".cpython-")[0] if matches else "")
PY
)"
if [[ -n "$CHARSET_MYPYC_MODULE" ]]; then
  echo "Detected charset-normalizer mypyc module: $CHARSET_MYPYC_MODULE"
else
  echo "WARN: charset-normalizer mypyc helper module was not detected; continuing without explicit hidden import."
fi

verify_sidecar_contents() {
  local sidecar_dir="$1"
  local listing_file

  if [[ ! -d "$sidecar_dir" ]]; then
    echo "ERROR: sidecar verification failed (missing directory: $sidecar_dir)" >&2
    return 1
  fi
  if [[ ! -x "$sidecar_dir/$BACKEND_NAME" ]]; then
    echo "ERROR: sidecar verification failed (missing executable: $sidecar_dir/$BACKEND_NAME)" >&2
    return 1
  fi

  listing_file="$(mktemp)"
  # Include both files and directories to support namespace-style packages
  # that may omit __init__.py while still shipping required resources.
  find "$sidecar_dir" \( -type f -o -type d \) | sed "s|$sidecar_dir/||" >"$listing_file"

  local -a required_patterns=(
    "docling/models/"
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
    "charset_normalizer/md\\.cpython-.*\\.(so|dylib|pyd)"
    "PIL/_imagingmorph\\.cpython-.*\\.(so|dylib|pyd)"
  )

  # Some platforms/builds do not ship a standalone __mypyc extension module.
  # Only require it when we detected and requested a concrete hidden import.
  if [[ -n "${CHARSET_MYPYC_MODULE:-}" ]]; then
    required_patterns+=(".*__mypyc\\.cpython-.*\\.(so|dylib|pyd)")
  fi

  for pattern in "${required_patterns[@]}"; do
    if ! grep -Eq "$pattern" "$listing_file"; then
      echo "ERROR: sidecar verification failed (missing: $pattern)" >&2
      rm -f "$listing_file"
      return 1
    fi
  done

  rm -f "$listing_file"
}

sign_sidecar_macho_files() {
  local sidecar_dir="$1"
  local identity="${APPLE_SIGNING_IDENTITY:-}"

  if [[ -z "$identity" ]]; then
    echo "APPLE_SIGNING_IDENTITY not set; skipping sidecar Mach-O signing."
    return 0
  fi
  if [[ ! -d "$sidecar_dir" ]]; then
    echo "ERROR: sidecar signing failed (missing directory: $sidecar_dir)" >&2
    return 1
  fi

  echo "Signing sidecar Mach-O files with identity: $identity"
  local signed_count=0
  while IFS= read -r -d '' candidate; do
    if file "$candidate" | grep -q "Mach-O"; then
      codesign --force --sign "$identity" --options runtime --timestamp "$candidate"
      signed_count=$((signed_count + 1))
    fi
  done < <(find "$sidecar_dir" -type f -print0)

  echo "Signed $signed_count sidecar Mach-O files."
}

echo "Building Tauri backend sidecar (${BACKEND_NAME})..."
uv run --with pyinstaller pyinstaller \
  --noconfirm \
  --clean \
  --onedir \
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
  --collect-all charset_normalizer \
  --collect-all PIL \
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
  --hidden-import PIL._imagingmorph \
  ${CHARSET_MYPYC_MODULE:+--hidden-import "$CHARSET_MYPYC_MODULE"} \
  --paths "$ROOT_DIR/src" \
  --distpath "$DIST_DIR" \
  --workpath "$WORK_DIR" \
  --specpath "$SPEC_DIR" \
  "$ROOT_DIR/src/informity/main.py"

verify_sidecar_contents "$DIST_DIR/$BACKEND_NAME"

rm -rf "$OUT_DIR/$BACKEND_BUNDLE_DIR" "$OUT_DIR/$BACKEND_NAME"
cp -R "$DIST_DIR/$BACKEND_NAME" "$OUT_DIR/$BACKEND_BUNDLE_DIR"
chmod +x "$OUT_DIR/$BACKEND_BUNDLE_DIR/$BACKEND_NAME" || true
sign_sidecar_macho_files "$OUT_DIR/$BACKEND_BUNDLE_DIR"

echo "Sidecar ready: $OUT_DIR/$BACKEND_BUNDLE_DIR/"
