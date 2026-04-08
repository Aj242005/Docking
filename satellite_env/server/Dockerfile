# Use the official hackathon base image
FROM ghcr.io/meta-pytorch/openenv-base:latest

USER root
WORKDIR /app

# 1. Install system dependencies for PyBullet & OpenGL
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git build-essential curl python3-dev \
    libgl1 libglvnd0 libglx0 libgl1-mesa-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. Install uv
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod +x /install.sh && /install.sh && rm /install.sh
ENV PATH="/root/.local/bin/:$PATH"

# 3. Copy everything
COPY . /app/env
WORKDIR /app/env

# 4. Install dependencies
RUN uv venv .venv && \
    . .venv/bin/activate && \
    uv pip install --no-cache wheel setuptools numpy && \
    uv pip install --no-cache -r server/requirements.txt

# 5. Fix permissions for Hugging Face's internal user
RUN chown -R 1000:1000 /app
USER 1000

# 6. Setup Environment Variables
ENV PATH="/app/env/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# 7. Start the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]