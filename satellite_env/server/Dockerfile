# Use the official hackathon base image
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

USER root
WORKDIR /app

# 1. Install EVERYTHING needed to build C++ physics engines
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git build-essential curl python3-dev libgl1-mesa-dev && \
    rm -rf /var/lib/apt/lists/* # 2. Install uv simply and directly
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod +x /install.sh && /install.sh && rm /install.sh
ENV PATH="/root/.local/bin/:$PATH"

# 3. Setup code
COPY . /app/env
WORKDIR /app/env

# 4. Create venv and install. We use --no-cache to avoid HF permission errors.
RUN uv venv .venv && \
    uv pip install --no-cache -r requirements.txt

# --- Final runtime stage ---
FROM ${BASE_IMAGE}

USER root
# Install modern GL libraries for Trixie/Debian 13
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglvnd0 libglx0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the built venv and code from builder
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

# Set permissions for the HF user
RUN chown -R 1000:1000 /app
USER 1000

# Configure environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Start the server
WORKDIR /app/env
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]