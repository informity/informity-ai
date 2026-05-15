#!/usr/bin/env bash
set -euo pipefail

DMG_DIR="src/frontend/src-tauri/target/release/bundle/dmg"
DMG_ICON_SRC="src/frontend/src-tauri/icons/dmg-volume-icon.icns"
APP_ICON_SRC="src/frontend/src-tauri/icons/icon.icns"
TMP_DIR="/tmp/informity-dmg-volume-icon"

if [[ ! -d "$DMG_DIR" ]]; then
  exit 0
fi

if [[ -f "$DMG_ICON_SRC" ]]; then
  ICON_SRC="$DMG_ICON_SRC"
elif [[ -f "$APP_ICON_SRC" ]]; then
  ICON_SRC="$APP_ICON_SRC"
else
  echo "DMG volume icon step skipped: missing both $DMG_ICON_SRC and $APP_ICON_SRC"
  exit 0
fi

mkdir -p "$TMP_DIR"

for dmg in "$DMG_DIR"/*.dmg; do
  [[ -e "$dmg" ]] || continue

  attach_output="$(hdiutil attach "$dmg" -nobrowse -readonly)"
  device="$(printf '%s\n' "$attach_output" | awk '/^\/dev\// {print $1; exit}')"
  mount_point="$(printf '%s\n' "$attach_output" | awk -F '\t' '/\t\// {print $NF; exit}')"

  if [[ -z "$device" || -z "$mount_point" ]]; then
    echo "Failed to parse mount info for $(basename "$dmg")"
    if [[ -n "$device" ]]; then
      hdiutil detach "$device" >/dev/null 2>&1 || true
    fi
    continue
  fi

  # Make writable copy because release DMGs are typically compressed read-only.
  rw_dmg="$TMP_DIR/rw.$(basename "$dmg")"
  hdiutil detach "$device" >/dev/null
  hdiutil convert "$dmg" -format UDRW -o "$rw_dmg" >/dev/null

  rw_attach_output="$(hdiutil attach "$rw_dmg" -nobrowse)"
  rw_device="$(printf '%s\n' "$rw_attach_output" | awk '/^\/dev\// {print $1; exit}')"
  rw_mount_point="$(printf '%s\n' "$rw_attach_output" | awk -F '\t' '/\t\// {print $NF; exit}')"

  if [[ -z "$rw_device" || -z "$rw_mount_point" ]]; then
    echo "Failed to mount writable DMG for $(basename "$dmg")"
    [[ -n "$rw_device" ]] && hdiutil detach "$rw_device" >/dev/null 2>&1 || true
    continue
  fi

  cp "$ICON_SRC" "$rw_mount_point/.VolumeIcon.icns"
  SetFile -a C "$rw_mount_point"

  hdiutil detach "$rw_device" >/dev/null

  final_dmg="$TMP_DIR/final.$(basename "$dmg")"
  hdiutil convert "$rw_dmg" -format UDZO -o "$final_dmg" >/dev/null
  mv "$final_dmg" "$dmg"

  echo "Set DMG volume icon for $(basename "$dmg")"
done

rm -rf "$TMP_DIR"
