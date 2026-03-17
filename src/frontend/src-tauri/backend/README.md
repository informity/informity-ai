# Backend Sidecar Layout

Phase 2 packs the backend runtime as a sidecar artifact in this directory.

Expected packaged artifact name:
- `informity-backend` (macOS/Linux)
- `informity-backend.exe` (Windows)

The Tauri runtime loads the sidecar from `resources/backend/` in packaged builds.

Build command:
- `make tauri-backend`

This runs `scripts/build_tauri_backend_sidecar.sh`, which uses PyInstaller to
produce the sidecar binary in this directory.
