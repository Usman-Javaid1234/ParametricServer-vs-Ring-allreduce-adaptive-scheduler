#!/bin/bash
# scripts/run_exp_a_scaling.sh
# Experiment A — Throughput Scaling
# Runs both PS-BSP and Ring AR at n=4 workers (Docker limit on single machine).
# Records samples/sec and comm latency at each scale.
#
# On a single machine Docker can only reliably run 4 workers.
# For n=8,16,32 we use analytical extrapolation (see plot_exp_a.py).
#
# Usage: bash scripts/run_exp_a_scaling.sh
#
# CS332 | Distributed SGD | Phase 2

set -euo pipefail

EPOCHS=3        # 3 epochs enough for stable throughput measurement
BATCH_SIZE=32
WORLD_SIZE=4    # fixed at 4 for Docker single-machine

echo "========================================"
echo " Experiment A — Throughput Scaling"
echo " Workers: $WORLD_SIZE | Epochs: $EPOCHS"
echo "========================================"

mkdir -p results/exp_a

# ---- Ring AR ----
echo ""
echo ">>> Ring AR | n=$WORLD_SIZE workers"
BACKEND=ring_ar \
WORLD_SIZE=$WORLD_SIZE \
EPOCHS=$EPOCHS \
BATCH_SIZE=$BATCH_SIZE \
RUN_ID=exp_a_ring_ar_n${WORLD_SIZE} \
docker compose up --abort-on-container-exit
docker compose down

mv results/exp_a_ring_ar_n${WORLD_SIZE}_rank*.csv results/exp_a/ 2>/dev/null || true
mv results/exp_a_ring_ar_n${WORLD_SIZE}_rank*.json results/exp_a/ 2>/dev/null || true

# ---- PS BSP ----
echo ""
echo ">>> PS-BSP | n=$WORLD_SIZE workers"
BACKEND=ps \
WORLD_SIZE=$WORLD_SIZE \
EPOCHS=$EPOCHS \
BATCH_SIZE=$BATCH_SIZE \
TAU=0 \
RUN_ID=exp_a_ps_bsp_n${WORLD_SIZE} \
docker compose up --abort-on-container-exit
docker compose down

mv results/exp_a_ps_bsp_n${WORLD_SIZE}_rank*.csv results/exp_a/ 2>/dev/null || true
mv results/exp_a_ps_bsp_n${WORLD_SIZE}_rank*.json results/exp_a/ 2>/dev/null || true

echo ""
echo "========================================"
echo " Experiment A done! Results in results/exp_a/"
echo " Run: python scripts/plot_exp_a.py"
echo "========================================"
