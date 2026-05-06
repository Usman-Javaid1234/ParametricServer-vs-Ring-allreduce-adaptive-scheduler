#!/bin/bash
# scripts/run_baseline.sh
# Phase 1 baseline experiments: Ring AR and PS-BSP at n=4 workers
#
# Runs:
#   1. Ring AR  — 4 workers, no compression, no faults
#   2. PS-BSP   — 4 workers, no compression, no faults
#
# Results land in ./results/ as CSV + JSON summary files.
#
# Usage:
#   bash scripts/run_baseline.sh
#
# CS332 | Distributed SGD Project

set -euo pipefail

WORLD_SIZE=4
EPOCHS=5
BATCH_SIZE=64
LR=0.01

echo "========================================"
echo " CS332 | Distributed SGD | Phase 1 Baseline"
echo " World size : $WORLD_SIZE workers"
echo " Epochs     : $EPOCHS"
echo " Batch size : $BATCH_SIZE per worker"
echo "========================================"

# ---- Experiment 1: Ring AllReduce ----
echo ""
echo ">>> [1/2] Ring AllReduce baseline"
BACKEND=ring_ar \
WORLD_SIZE=$WORLD_SIZE \
EPOCHS=$EPOCHS \
BATCH_SIZE=$BATCH_SIZE \
LR=$LR \
RUN_ID=baseline_ring_ar \
docker compose up --abort-on-container-exit

echo ">>> Ring AR done. Collecting results..."
docker compose down

# ---- Experiment 2: Parameter Server BSP ----
echo ""
echo ">>> [2/2] Parameter Server BSP baseline"
BACKEND=ps \
WORLD_SIZE=$WORLD_SIZE \
EPOCHS=$EPOCHS \
BATCH_SIZE=$BATCH_SIZE \
LR=$LR \
TAU=0 \
RUN_ID=baseline_ps_bsp \
docker compose up --abort-on-container-exit

echo ">>> PS-BSP done. Collecting results..."
docker compose down

echo ""
echo "========================================"
echo " Baseline runs complete!"
echo " Results in: ./results/"
echo " Run: python scripts/plot_baseline.py to visualise"
echo "========================================"
