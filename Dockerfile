# syntax=docker/dockerfile:1
# Multi-stage build for the Informity AI backend service.
# Produces a headless FastAPI server on port 8420 — no Tauri/GUI.
#
# Build:  docker build -t informity-backend .
# Run:    docker run -p 8420:8420 -v ~/.informity:/data informity-backend

# ── Stage 1: dependency install ───────────────────────────────────────────────
FROM python:3.13-slim AS deps

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install all runtime deps (--frozen ensures lockfile is respected)
RUN uv sync --no-dev --frozen --python 3.13

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed deps, source, and scripts from build stage
COPY --from=deps /app/.venv /app/.venv
COPY --from=deps /app/src /app/src
COPY --from=deps /app/scripts /app/scripts

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Bind to all interfaces so the container port is reachable
ENV INFORMITY_HOST=0.0.0.0
ENV INFORMITY_PORT=8420

# Data directory — mount a volume here to persist models, DB, and config
ENV INFORMITY_APP_DATA_DIR=/data
VOLUME ["/data"]

EXPOSE 8420

CMD ["python", "-m", "informity.main"]
