#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/src/frontend"
BUNDLE_DIR="$FRONTEND_DIR/src-tauri/target/release/bundle"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: Linux release build must run on Linux." >&2
  exit 1
fi

cd "$FRONTEND_DIR"
npm run tauri:build:linux

echo "Linux release artifacts ready:"
echo "  DEB:      $BUNDLE_DIR/deb/"
echo "  AppImage: $BUNDLE_DIR/appimage/"
