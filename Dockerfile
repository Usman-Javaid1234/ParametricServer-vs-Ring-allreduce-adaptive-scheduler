# Dockerfile
# CS332 | Distributed SGD Project
# Single image used for both workers and orchestrator.

FROM python:3.11-slim

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 \
    net-tools \
    iputils-ping \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Source ----
COPY src/ /app/src/

# ---- Results & data dirs ----
RUN mkdir -p /results /data/cifar10

# ---- Entrypoint: workers use this ----
CMD ["python", "/app/src/worker.py"]
