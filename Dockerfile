# Dockerfile — CS332 | Distributed SGD Project
# GPU-enabled image: CUDA 11.8 + cuDNN base, PyTorch with CUDA support.
# Falls back gracefully to CPU if no GPU is present.

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    iproute2 \
    net-tools \
    iputils-ping \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.11 /usr/bin/python \
    && ln -sf python3.11 /usr/bin/python3

# ---- Python deps ----
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Source ----
COPY src/ /app/src/

# ---- Results & data dirs ----
RUN mkdir -p /results /data/cifar10

CMD ["python", "/app/src/worker.py"]