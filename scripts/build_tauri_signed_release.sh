#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${INFORMITY_CODESIGN_ENV_FILE:-$ROOT_DIR/.env.codesign}"
FRONTEND_DIR="$ROOT_DIR/src/frontend"
TAURI_DIR="$FRONTEND_DIR/src-tauri"
BUNDLE_DIR="$TAURI_DIR/target/release/bundle"
APP_PATH="$BUNDLE_DIR/macos/Informity AI.app"
APP_ZIP_PATH="$BUNDLE_DIR/macos/Informity AI.app.zip"
DMG_GLOB="$BUNDLE_DIR/dmg/"'*.dmg'

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: Signing env file not found: $ENV_FILE" >&2
  echo "Create it from .env.codesign.example and retry." >&2
  exit 1
fi

# shellcheck source=/dev/null
set -a
source "$ENV_FILE"
set +a

required_vars=(
  APPLE_SIGNING_IDENTITY
  APPLE_API_ISSUER
  APPLE_API_KEY
  APPLE_API_KEY_PATH
)
for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "ERROR: Required variable '$var_name' is missing in $ENV_FILE" >&2
    exit 1
  fi
done

if [[ ! -f "$APPLE_API_KEY_PATH" ]]; then
  echo "ERROR: APPLE_API_KEY_PATH does not exist: $APPLE_API_KEY_PATH" >&2
  exit 1
fi

echo "Using signing identity: $APPLE_SIGNING_IDENTITY"
echo "Building signed app and DMG (Tauri notarization disabled during bundle)..."

# Build/sign app + dmg first. Keep APPLE_API_* unset during tauri build to avoid
# create-dmg notarization hook failures; notarize app and dmg explicitly below.
(
  export APPLE_SIGNING_IDENTITY
  unset APPLE_API_ISSUER APPLE_API_KEY APPLE_API_KEY_PATH
  cd "$FRONTEND_DIR"
  npm run tauri:build:mac
)

if [[ ! -d "$APP_PATH" ]]; then
  echo "ERROR: App bundle not found at $APP_PATH" >&2
  exit 1
fi

DMG_PATH="$(find "$BUNDLE_DIR/dmg" -maxdepth 1 -type f -name '*.dmg' | sort | tail -n 1)"
if [[ -z "$DMG_PATH" ]]; then
  echo "ERROR: No DMG found in $BUNDLE_DIR/dmg" >&2
  exit 1
fi

echo "Notarizing app: $APP_PATH"
rm -f "$APP_ZIP_PATH"
ditto -c -k --sequesterRsrc --keepParent "$APP_PATH" "$APP_ZIP_PATH"

xcrun notarytool submit "$APP_ZIP_PATH" \
  --key "$APPLE_API_KEY_PATH" \
  --key-id "$APPLE_API_KEY" \
  --issuer "$APPLE_API_ISSUER" \
  --wait

echo "Stapling app..."
xcrun stapler staple "$APP_PATH"

echo "Notarizing DMG: $DMG_PATH"
xcrun notarytool submit "$DMG_PATH" \
  --key "$APPLE_API_KEY_PATH" \
  --key-id "$APPLE_API_KEY" \
  --issuer "$APPLE_API_ISSUER" \
  --wait

echo "Stapling DMG..."
xcrun stapler staple "$DMG_PATH"

echo "Verifying app signature..."
codesign --verify --deep --strict --verbose=2 "$APP_PATH"

echo "Verifying Gatekeeper acceptance..."
spctl -a -t exec -vv "$APP_PATH"
spctl -a -t open --context context:primary-signature -vv "$DMG_PATH"

echo "Signed + notarized release artifacts ready:"
echo "  APP: $APP_PATH"
echo "  DMG: $DMG_PATH"
