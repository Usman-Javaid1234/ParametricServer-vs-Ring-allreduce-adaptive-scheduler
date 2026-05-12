# Dockerfile — CS332 | Distributed SGD + QSGD Compression
# CUDA 11.8 base image — required for torch+cu118 to actually use the GPU.
# python:3.11-slim has no CUDA runtime so GPU is invisible even with cu118 wheels.

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 + pip + minimal tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    iproute2 curl net-tools \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/

RUN mkdir -p /results /data/cifar10

CMD ["python", "/app/src/worker.py"]