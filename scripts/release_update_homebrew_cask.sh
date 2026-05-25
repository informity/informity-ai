#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/release_update_homebrew_cask.sh \
    --version <version> \
    --sha256 <sha256> \
    [--tap-dir <path>] \
    [--cask-path <relative path>] \
    [--dry-run]

Example:
  scripts/release_update_homebrew_cask.sh \
    --version 0.13.3 \
    --sha256 6b860b60e87e0547e7bec3953042864acef0dfbb9eea70da0abea0bbfe5598eb \
    --tap-dir ../homebrew-tap
EOF
}

VERSION=""
SHA256=""
TAP_DIR="${TAP_DIR:-}"
CASK_PATH="${CASK_PATH:-Casks/informity-ai.rb}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="${2:-}"
      shift 2
      ;;
    --sha256)
      SHA256="${2:-}"
      shift 2
      ;;
    --tap-dir)
      TAP_DIR="${2:-}"
      shift 2
      ;;
    --cask-path)
      CASK_PATH="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$VERSION" || -z "$SHA256" ]]; then
  echo "ERROR: --version and --sha256 are required." >&2
  usage
  exit 1
fi

if [[ -z "$TAP_DIR" ]]; then
  echo "ERROR: --tap-dir (or TAP_DIR env var) is required." >&2
  exit 1
fi

if [[ ! -d "$TAP_DIR/.git" ]]; then
  echo "ERROR: tap repo not found at: $TAP_DIR" >&2
  exit 1
fi

TARGET_FILE="$TAP_DIR/$CASK_PATH"
if [[ ! -f "$TARGET_FILE" ]]; then
  echo "ERROR: cask file not found: $TARGET_FILE" >&2
  exit 1
fi

echo "Updating Homebrew cask:"
echo "  Tap dir:     $TAP_DIR"
echo "  Cask file:   $TARGET_FILE"
echo "  New version: $VERSION"
echo "  New sha256:  $SHA256"

perl -0777 -i -pe 's/version\s+"[^"]+"/version "'"$VERSION"'"/g' "$TARGET_FILE"
perl -0777 -i -pe 's/sha256\s+arm:\s+"[^"]+"/sha256 arm: "'"$SHA256"'"/g' "$TARGET_FILE"

if [[ "$DRY_RUN" == "1" ]]; then
  echo ""
  echo "Dry run enabled. Updated file preview:"
  git -C "$TAP_DIR" --no-pager diff -- "$CASK_PATH" || true
  exit 0
fi

if git -C "$TAP_DIR" diff --quiet -- "$CASK_PATH"; then
  echo "No cask changes detected; skipping commit."
  exit 0
fi

git -C "$TAP_DIR" add "$CASK_PATH"
git -C "$TAP_DIR" commit -m "chore(cask): bump informity-ai to v$VERSION"
git -C "$TAP_DIR" push

echo "Cask update pushed successfully."
