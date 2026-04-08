# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

USER root
WORKDIR /app

# 1. Install system tools needed for building pybullet
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential curl && \
    rm -rf /var/lib/apt/lists/* 

# 2. FIXED: Correct uv installer URL
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# 3. Setup environment code
COPY . /app/env
WORKDIR /app/env

# 4. Create venv and install everything from requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv .venv && \
    uv pip install -r requirements.txt

# --- Final runtime stage ---
FROM ${BASE_IMAGE}

USER root
# Install runtime libraries for PyBullet's OpenGL dependencies
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
USER user

WORKDIR /app

# Copy the built venv and code
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

# Configure environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check for Hugging Face Spaces
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
