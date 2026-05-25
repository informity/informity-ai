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

DMG_DIR="$BUNDLE_DIR/dmg"
if [[ -d "$DMG_DIR" ]]; then
  DMG_PATH="$(find "$DMG_DIR" -maxdepth 1 -type f -name 'Informity_AI_*_arm64.dmg' | sort | tail -n 1)"
  if [[ -n "$DMG_PATH" ]]; then
    LATEST_DMG_PATH="$DMG_DIR/Informity_AI_latest_aarch64.dmg"
    cp -f "$DMG_PATH" "$LATEST_DMG_PATH"
    echo "Created latest DMG alias: $LATEST_DMG_PATH"
  else
    echo "No versioned arm64 DMG found in $DMG_DIR; skipping latest DMG alias copy."
  fi
fi

echo "Linux release artifacts ready:"
echo "  DEB:      $BUNDLE_DIR/deb/"
echo "  AppImage: $BUNDLE_DIR/appimage/"
