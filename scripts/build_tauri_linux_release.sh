#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/src/frontend"
BUNDLE_DIR="$FRONTEND_DIR/src-tauri/target/release/bundle"

BUNDLE_TYPE="${1:-deb}"

case "$BUNDLE_TYPE" in
  deb|rpm)
    ;;
  *)
    echo "ERROR: Unsupported Linux bundle type: $BUNDLE_TYPE (expected deb or rpm)." >&2
    exit 1
    ;;
esac

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: Linux release build must run on Linux." >&2
  exit 1
fi

cd "$FRONTEND_DIR"
npm run tauri:build -- --bundles "$BUNDLE_TYPE,appimage"

echo "Linux release artifacts ready:"
echo "  ${BUNDLE_TYPE^^}:      $BUNDLE_DIR/$BUNDLE_TYPE/"
echo "  AppImage: $BUNDLE_DIR/appimage/"
