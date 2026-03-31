FROM python:3.12-slim

WORKDIR /app

# Install system dependencies + uv + gcsfuse
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl gnupg lsb-release \
    procps iproute2 ethtool \
    && GCSFUSE_REPO="gcsfuse-$(lsb_release -c -s)" \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /etc/apt/keyrings/gcsfuse.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/gcsfuse.gpg] https://packages.cloud.google.com/apt ${GCSFUSE_REPO} main" \
        > /etc/apt/sources.list.d/gcsfuse.list \
    && apt-get update && apt-get install -y --no-install-recommends gcsfuse \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# --- Dependency layer (cached unless requirements.txt changes) ---
COPY requirements.txt .
RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    -r requirements.txt

# --- Source layer ---
COPY . .
RUN uv pip install --system --no-cache --no-deps .
