# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

USER root
WORKDIR /app

# Install system dependencies required to compile pybullet and git tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential curl && \
    rm -rf /var/lib/apt/lists/* 

# Ensure uv is available for dependency management
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Copy environment code
COPY . /app/env
WORKDIR /app/env

# Force creation of a virtual environment and install from requirements.txt
# This ensures pybullet and other manual additions are actually included
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv .venv && \
    uv pip install -r requirements.txt

# Final runtime stage
FROM ${BASE_IMAGE}

USER root
# Install runtime libraries that pybullet sometimes looks for
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
USER user

WORKDIR /app

# Copy the virtual environment and code from builder
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

# Set PATH and PYTHONPATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
