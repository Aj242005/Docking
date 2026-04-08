# Use the official hackathon base image
FROM ghcr.io/meta-pytorch/openenv-base:latest

USER root

# 1. Set the working directory to /app (Hugging Face standard)
WORKDIR /app

# 2. Install system dependencies for PyBullet & OpenGL
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git build-essential curl python3-dev \
    libgl1 libglvnd0 libglx0 libgl1-mesa-dev && \
    rm -rf /var/lib/apt/lists/*

# 3. Install uv
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod +x /install.sh && /install.sh && rm /install.sh
ENV PATH="/root/.local/bin/:$PATH"

# 4. Copy ONLY requirements.txt first
# This is a Docker best practice: it prevents re-installing dependencies every time you change your code.
COPY requirements.txt /app/requirements.txt

# 5. Create virtual environment and install dependencies
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install --no-cache wheel setuptools numpy && \
    uv pip install --no-cache -r /app/requirements.txt

# 6. Copy the rest of your application code
COPY . /app

# 7. Fix permissions for Hugging Face Spaces
# Hugging Face runs containers as user 1000. We must grant this user ownership of /app.
RUN chown -R 1000:1000 /app

# 8. Switch to the Hugging Face user
USER 1000

# 9. Setup Environment Variables
# Point exactly to the new virtual environment's bin folder
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# 10. Start the server
# Hugging Face Spaces default to port 7860. Ensure uvicorn binds to it.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]