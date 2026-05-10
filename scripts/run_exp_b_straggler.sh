#!/bin/bash
# scripts/run_exp_b_straggler.sh
# Experiment B — Straggler Sensitivity
#
# Injects artificial delays on worker rank 3 via STRAGGLER_RANK env var.
# Tests delays: 50ms, 200ms, 500ms on 1 straggler worker.
# Tests both PS-BSP, PS-SSP (tau=2) and Ring AR.
#
# Usage: bash scripts/run_exp_b_straggler.sh
# CS332 | Distributed SGD | Phase 2

set -euo pipefail

EPOCHS=3
BATCH_SIZE=32
WORLD_SIZE=4
STRAGGLER_RANK=3       # always slow down the last worker
STRAGGLER_START=20     # start injecting after 20 iters (warmup)

DELAYS=(50 200 500)

echo "========================================"
echo " Experiment B — Straggler Sensitivity"
echo " Straggler rank: $STRAGGLER_RANK"
echo " Delays: ${DELAYS[*]} ms"
echo "========================================"

mkdir -p results/exp_b

for DELAY in "${DELAYS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo " Delay = ${DELAY}ms"
    echo "----------------------------------------"

    # ---- Ring AR + straggler ----
    echo ">>> Ring AR | delay=${DELAY}ms"
    BACKEND=ring_ar \
    WORLD_SIZE=$WORLD_SIZE \
    EPOCHS=$EPOCHS \
    BATCH_SIZE=$BATCH_SIZE \
    STRAGGLER_RANK=$STRAGGLER_RANK \
    STRAGGLER_DELAY_MS=$DELAY \
    STRAGGLER_START_ITER=$STRAGGLER_START \
    RUN_ID=exp_b_ring_ar_delay${DELAY} \
    docker compose up --abort-on-container-exit
    docker compose down
    mv results/exp_b_ring_ar_delay${DELAY}_rank*.csv results/exp_b/ 2>/dev/null || true
    mv results/exp_b_ring_ar_delay${DELAY}_rank*.json results/exp_b/ 2>/dev/null || true

    # ---- PS-BSP + straggler ----
    echo ">>> PS-BSP | delay=${DELAY}ms"
    BACKEND=ps \
    WORLD_SIZE=$WORLD_SIZE \
    EPOCHS=$EPOCHS \
    BATCH_SIZE=$BATCH_SIZE \
    TAU=0 \
    STRAGGLER_RANK=$STRAGGLER_RANK \
    STRAGGLER_DELAY_MS=$DELAY \
    STRAGGLER_START_ITER=$STRAGGLER_START \
    RUN_ID=exp_b_ps_bsp_delay${DELAY} \
    docker compose up --abort-on-container-exit
    docker compose down
    mv results/exp_b_ps_bsp_delay${DELAY}_rank*.csv results/exp_b/ 2>/dev/null || true
    mv results/exp_b_ps_bsp_delay${DELAY}_rank*.json results/exp_b/ 2>/dev/null || true

    # ---- PS-SSP tau=2 + straggler ----
    echo ">>> PS-SSP (tau=2) | delay=${DELAY}ms"
    BACKEND=ps \
    WORLD_SIZE=$WORLD_SIZE \
    EPOCHS=$EPOCHS \
    BATCH_SIZE=$BATCH_SIZE \
    TAU=2 \
    STRAGGLER_RANK=$STRAGGLER_RANK \
    STRAGGLER_DELAY_MS=$DELAY \
    STRAGGLER_START_ITER=$STRAGGLER_START \
    RUN_ID=exp_b_ps_ssp2_delay${DELAY} \
    docker compose up --abort-on-container-exit
    docker compose down
    mv results/exp_b_ps_ssp2_delay${DELAY}_rank*.csv results/exp_b/ 2>/dev/null || true
    mv results/exp_b_ps_ssp2_delay${DELAY}_rank*.json results/exp_b/ 2>/dev/null || true

done

echo ""
echo "========================================"
echo " Experiment B done! Results in results/exp_b/"
echo " Run: python scripts/plot_exp_b.py"
echo "========================================"
