# Running Informity AI as a Container (Podman / Docker)

This covers building and running the Informity backend as a headless OCI container.
The result is a FastAPI server on port 8420 — no GUI, no Tauri shell.

Tested target: **rootless Podman on x86 Linux** (Kicksecure / Debian derivative).
The same Dockerfile works with Docker and on arm64 (Apple Silicon) for local testing.

---

## Prerequisites

- Podman 4+ (or Docker — commands are interchangeable, swap `podman` → `docker`)
- Git
- 10 GB+ free disk space (models are large)

---

## 1. Clone the repo and switch to the branch

```bash
git clone https://github.com/informity/informity-ai.git
cd informity-ai
git checkout contrib/podman
```

---

## 2. Build the image

```bash
podman build -t informity-backend .
```

This installs all Python dependencies via `uv` inside the image.
Expect 10–20 minutes on first build (downloading torch, docling, xllamacpp, etc.).

---

## 3. Download models (first time only)

Models are stored in a named volume and only need to be downloaded once.

```bash
podman volume create informity-data

podman run --rm \
  -v informity-data:/data \
  -e INFORMITY_APP_DATA_DIR=/data \
  -e INFORMITY_FULL_PRIVACY=false \
  -e INFORMITY_EMBEDDING_OFFLINE=false \
  -e INFORMITY_LLM_LOCAL_ONLY=false \
  localhost/informity-backend:latest \
  python scripts/install_bootstrap_models.py
```

This pulls the embedding model, reranker, and LLM into the volume.
**Note:** The default LLM (Qwen3 ~35B) is large and will run on CPU on x86 without
a CUDA-capable GPU. Inference will be slow. Consider switching to a smaller model
in `scripts/install.conf.json` before running the above step.

---

## 4a. Run standalone (quick test)

```bash
podman run -d \
  --name informity \
  -p 8420:8420 \
  -v informity-data:/data \
  -e INFORMITY_APP_DATA_DIR=/data \
  localhost/informity-backend:latest
```

Verify it started:

```bash
curl http://localhost:8420/setup/status
```

---

## 4b. Deploy as a Podman Quadlet (systemd, rootless)

Copy the unit file into your Quadlet directory:

```bash
cp contrib/podman/informity.container ~/.config/containers/systemd/
```

Reload systemd and start the service:

```bash
systemctl --user daemon-reload
systemctl --user start informity
systemctl --user status informity
```

Enable on boot:

```bash
systemctl --user enable informity
loginctl enable-linger $USER   # keep user services running without a login session
```

---

## 5. Verify

```bash
# Basic liveness
curl http://localhost:8420/setup/status

# Diagnostics (models loaded, DB stats, etc.)
curl http://localhost:8420/diagnostics
```

---

## Networking (Quadlet with Caddy)

The provided `informity.container` joins `caddy.network` with alias `informity`,
matching the same pattern as `open-webui.container`. Once running, other containers
on that network can reach the backend at:

```
http://informity:8420
```

If `caddy.network` does not exist yet, create it first:

```bash
podman network create caddy
```

---

## Logs

```bash
# Quadlet / systemd
journalctl --user -u informity -f

# Standalone container
podman logs -f informity
```

---

## Stop / remove

```bash
# Standalone
podman stop informity && podman rm informity

# Quadlet
systemctl --user stop informity
```

---

## Notes

- **LLM inference on CPU:** `xllamacpp` uses Metal on macOS and CUDA on Linux (if
  available). Without a CUDA GPU on x86, all LLM inference runs on CPU and will be
  significantly slower than Apple Silicon. The embedding model and document indexing
  are unaffected.
- **Data persistence:** everything (models, SQLite DB, config, logs) lives in the
  `informity-data` volume at `/data` inside the container. Deleting the container
  does not delete the volume.
- **Port conflict:** if something else is already on 8420, override with
  `-p <host-port>:8420` and set `Environment=INFORMITY_PORT=8420` in the Quadlet
  (the container-side port stays 8420 regardless).
