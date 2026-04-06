#!/usr/bin/env bash
set -euo pipefail

DMG_DIR="src-tauri/target/release/bundle/dmg"

if [[ ! -d "$DMG_DIR" ]]; then
  exit 0
fi

for f in "$DMG_DIR"/Informity\ AI_*.dmg; do
  [[ -e "$f" ]] || continue
  renamed="${f//Informity AI/Informity_AI}"
  if [[ "$renamed" != "$f" ]]; then
    mv "$f" "$renamed"
    echo "Renamed DMG artifact: $(basename "$f") -> $(basename "$renamed")"
  fi
done
