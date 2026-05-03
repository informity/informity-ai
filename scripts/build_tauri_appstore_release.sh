#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${INFORMITY_APPSTORE_ENV_FILE:-$ROOT_DIR/.env.appstore}"
FRONTEND_DIR="$ROOT_DIR/src/frontend"
TAURI_DIR="$FRONTEND_DIR/src-tauri"
TARGET_TRIPLE="${TAURI_APPSTORE_TARGET:-universal-apple-darwin}"
APP_NAME="${TAURI_APP_NAME:-Informity AI}"
APPSTORE_CONFIG_REL="src-tauri/tauri.appstore.conf.json"
APPSTORE_CONFIG_PATH="$FRONTEND_DIR/$APPSTORE_CONFIG_REL"
BUNDLE_ROOT="$TAURI_DIR/target/$TARGET_TRIPLE/release/bundle"
APP_PATH="$BUNDLE_ROOT/macos/$APP_NAME.app"
PKG_PATH="$BUNDLE_ROOT/macos/$APP_NAME.pkg"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: App Store env file not found: $ENV_FILE" >&2
  echo "Create it from .env.appstore.example and retry." >&2
  exit 1
fi

# shellcheck source=/dev/null
set -a
source "$ENV_FILE"
set +a

required_vars=(
  APPLE_APPSTORE_SIGNING_IDENTITY
  APPLE_APPSTORE_INSTALLER_IDENTITY
  APPLE_API_ISSUER
  APPLE_API_KEY
)
for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "ERROR: Required variable '$var_name' is missing in $ENV_FILE" >&2
    exit 1
  fi
done

if [[ ! -f "$APPSTORE_CONFIG_PATH" ]]; then
  echo "ERROR: App Store Tauri config not found: $APPSTORE_CONFIG_PATH" >&2
  exit 1
fi

if [[ ! -f "${APPLE_API_KEY_PATH:-}" ]]; then
  echo "ERROR: APPLE_API_KEY_PATH is missing or file does not exist: ${APPLE_API_KEY_PATH:-}" >&2
  exit 1
fi

if [[ -n "${APPLE_APPSTORE_PROVISION_PROFILE_PATH:-}" && ! -f "$APPLE_APPSTORE_PROVISION_PROFILE_PATH" ]]; then
  echo "ERROR: APPLE_APPSTORE_PROVISION_PROFILE_PATH does not exist: $APPLE_APPSTORE_PROVISION_PROFILE_PATH" >&2
  exit 1
fi

echo "Using App Store app signing identity: $APPLE_APPSTORE_SIGNING_IDENTITY"
echo "Using App Store installer identity: $APPLE_APPSTORE_INSTALLER_IDENTITY"

(
  export APPLE_SIGNING_IDENTITY="$APPLE_APPSTORE_SIGNING_IDENTITY"
  unset APPLE_API_ISSUER APPLE_API_KEY APPLE_API_KEY_PATH
  cd "$FRONTEND_DIR"
  npm run tauri:build -- --no-bundle
  npx tauri bundle --bundles app --target "$TARGET_TRIPLE" --config "$APPSTORE_CONFIG_REL"
)

if [[ ! -d "$APP_PATH" ]]; then
  echo "ERROR: App bundle not found at $APP_PATH" >&2
  exit 1
fi

echo "Building signed App Store PKG: $PKG_PATH"
rm -f "$PKG_PATH"
xcrun productbuild \
  --sign "$APPLE_APPSTORE_INSTALLER_IDENTITY" \
  --component "$APP_PATH" \
  /Applications \
  "$PKG_PATH"

echo "Uploading PKG to App Store Connect..."
xcrun altool --upload-app \
  --type macos \
  --file "$PKG_PATH" \
  --apiKey "$APPLE_API_KEY" \
  --apiIssuer "$APPLE_API_ISSUER"

echo "App Store upload submitted successfully:"
echo "  APP: $APP_PATH"
echo "  PKG: $PKG_PATH"
