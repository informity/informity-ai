# Backend Sidecar Layout

Phase 2 packs the backend runtime as a sidecar artifact in this directory.

Expected packaged artifact layout (PyInstaller `onedir`):
- `informity-backend-bundle/` (macOS/Linux)
  - `informity-backend` executable
- `informity-backend-bundle/` (Windows)
  - `informity-backend.exe` executable

The Tauri runtime loads the sidecar executable from:
- `resources/backend/informity-backend-bundle/<binary-name>` (`onedir`)

Build command:
- `make tauri-backend`

This runs `scripts/build_tauri_backend_sidecar.sh`, which uses PyInstaller to
produce an `onedir` sidecar bundle in this directory.
