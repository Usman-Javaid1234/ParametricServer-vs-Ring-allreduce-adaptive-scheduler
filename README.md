# Gradient Compression in Ring AllReduce
### CS332 — Parallel & Distributed Computing | NUST-SEECS | BSCS-13E

> **Research Question:** Does compression-induced accuracy degradation increase with ring size, and at what compression ratios does this effect become significant?

---

## Overview

This project implements **QSGD gradient quantization** integrated into **Ring AllReduce** distributed training, and provides the first empirical characterization of the **compression ratio × ring size interaction**.

We train ResNet-18 on CIFAR-10 across 8 experimental configurations (2 ring sizes × 4 bit-widths) and demonstrate that larger rings tolerate more aggressive compression due to quantization noise averaging across workers.

### Key Findings

| Compression | 2 Workers (acc drop) | 4 Workers (acc drop) | Difference |
|---|---|---|---|
| 8-bit (4×) | 14.95% | 10.92% | ~4% |
| 4-bit (8×) | 38.20% | 19.21% | ~19% |
| 2-bit (16×) | 57.51% | 36.56% | **~21%** |

**4-worker rings degrade 20.95% less than 2-worker rings at 16× compression** — confirming the ring-size interaction is practically significant.

---

## Project Structure

```
.
├── src/
│   ├── worker.py                # Training loop — pluggable sync backend
│   ├── compressed_ring_ar.py    # CompressedRingARBackend (QSGD + AllReduce)
│   ├── qsgd.py                  # QSGD quantize/dequantize + compression math
│   ├── ring_ar.py               # NativeRingAR + ManualRingAR (Phase 1)
│   ├── metrics.py               # Per-iteration CSV + JSON metrics logging
│   ├── fault_injector.py        # SlowdownInjector + CrashInjector
│   └── orchestrator.py          # Heartbeat monitor + failure detector
│
├── results/                     # Auto-generated experiment outputs (CSVs, JSONs, checkpoints)
├── plots/                       # Auto-generated plots from analyze.py
│
├── Dockerfile                   # CUDA 11.8 base image
├── docker-compose.yml           # Multi-worker container setup
├── requirements.txt             # Python dependencies (torch cu118)
├── run_experiments.py           # Automated experiment grid runner
└── analyze.py                   # Results analysis + plot generation
```

---

## Architecture

### Compression Pipeline (per iteration)

```
Each Worker
    │
    ▼
Forward + Backward Pass
    │
    ▼
Flatten Gradients (float32, ~44.7MB for ResNet-18)
    │
    ▼  QSGD quantize(num_bits)
Compressed tensor (int32) + L2 norm scalar
    │
    ▼  dist.all_reduce SUM  ← NCCL Ring AllReduce
Summed compressed gradients (all workers)
    │
    ▼  ÷ world_size
Averaged compressed gradients
    │
    ▼  QSGD dequantize
Reconstructed float32 gradients
    │
    ▼
Optimizer Step (SGD)
```

### QSGD Quantization

For a gradient tensor `g` with `b` bits:
1. Compute L2 norm `ν = ‖g‖₂`
2. Normalize: `ĝ = g / ν`
3. Scale to `s = 2^b - 1` levels
4. Stochastic round: `q = ⌊v⌋ + Bernoulli(v - ⌊v⌋) × sign(g)`
5. Transmit `(q as float32, ν)` via `dist.all_reduce`

**Key property:** `E[Q(g)] = g` (unbiased) — guarantees Ring AllReduce compatibility.

### Compression Ratios

| Bit-width | Compression | Theoretical Savings | Use Case |
|---|---|---|---|
| 32-bit | 1× | — | Baseline |
| 8-bit | 4× | 75% | Recommended — minimal accuracy loss |
| 4-bit | 8× | 87.5% | Moderate loss, large ring sizes only |
| 2-bit | 16× | 93.75% | Not recommended — significant degradation |

---

## Quickstart

### Prerequisites

- Docker + Docker Compose
- NVIDIA GPU with CUDA 11.8+
- NVIDIA Container Toolkit (`nvidia-smi` works inside Docker)

Verify GPU access in Docker:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 1. Build

```bash
git clone <repo-url>
cd <repo>
mkdir -p results plots
docker compose build
```

### 2. Smoke Test (2 workers, 8-bit, 2 epochs)

```bash
WORLD_SIZE=2 NUM_BITS=8 EPOCHS=2 BATCH_SIZE=64 RUN_ID=smoketest \
docker compose up orchestrator worker_0 worker_1
```

Expected output:
```
worker_0 | Epoch 1/2 | train_loss=X.XX | val_loss=X.XX | val_acc=XX.XX% | bits=8 | workers=2
worker_0 | Epoch 2/2 | ...
worker_0 | Rank 0: training complete.
```

### 3. Run Full Experiment Grid

```bash
python run_experiments.py
```

This runs all 10 experiments automatically (8 grid + 2 failure scenario):

```
Experiment grid: 2 worker configs × 4 bit widths = 8 runs
Epochs per run : 20
Total runs     : 10
Est. time      : ~100–160 min total
```

### 4. Generate Plots & Analysis

```bash
pip install pandas matplotlib seaborn numpy
python analyze.py
```

Plots saved to `./plots/`. Summary table printed to terminal with all accuracy and latency numbers.

---

## Running Individual Experiments

### Manual Docker Compose

```bash
# Baseline (32-bit, 4 workers)
WORLD_SIZE=4 NUM_BITS=32 EPOCHS=20 RUN_ID=exp docker compose up

# 4-bit compression, 2 workers
WORLD_SIZE=2 NUM_BITS=4 EPOCHS=20 RUN_ID=exp \
docker compose up orchestrator worker_0 worker_1

# 8-bit compression, 4 workers
WORLD_SIZE=4 NUM_BITS=8 EPOCHS=20 RUN_ID=exp docker compose up

# Failure scenario (crash rank 1 at iteration 200)
WORLD_SIZE=4 NUM_BITS=4 CRASH_RANK=1 CRASH_ITER=200 RUN_ID=failure \
docker compose up

# CPU-only (no GPU required)
WORLD_SIZE=2 NUM_BITS=8 DIST_BACKEND=gloo docker compose up orchestrator worker_0 worker_1
```

### Partial Grid

```bash
# Only 2-worker runs
python run_experiments.py --workers 2

# Only specific bit-widths
python run_experiments.py --bits 32 8

# Quick test (2 epochs)
python run_experiments.py --epochs 2

# Dry run — print commands without executing
python run_experiments.py --dry-run

# Skip failure scenario
python run_experiments.py --skip-failure
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WORLD_SIZE` | 4 | Number of workers |
| `NUM_BITS` | 8 | Quantization bits (2, 4, 8, 32) |
| `EPOCHS` | 10 | Training epochs |
| `BATCH_SIZE` | 128 | Per-worker batch size |
| `LR` | 0.01 | SGD learning rate |
| `RUN_ID` | exp | Results file prefix |
| `DIST_BACKEND` | nccl | `nccl` (GPU) or `gloo` (CPU) |
| `CRASH_RANK` | — | Worker rank to crash (fault injection) |
| `CRASH_ITER` | 200 | Iteration to trigger crash |
| `STRAGGLER_RANK` | — | Worker rank to slow down |
| `STRAGGLER_DELAY_MS` | 0 | Slowdown delay in ms |
| `RESULTS_DIR` | /results | Output directory inside container |

---

## Results

Results are written to `./results/` with the naming convention:

```
{RUN_ID}_w{WORLD_SIZE}_b{NUM_BITS}_rank{RANK}_epochs.csv      # per-epoch val accuracy
{RUN_ID}_w{WORLD_SIZE}_b{NUM_BITS}_rank{RANK}_iterations.csv  # per-iteration metrics
{RUN_ID}_w{WORLD_SIZE}_b{NUM_BITS}_rank{RANK}_summary.json    # final summary
```

### Generated Plots

| Plot | Description |
|---|---|
| `convergence_curves.png` | Validation accuracy vs epoch for all configurations |
| `sensitivity.png` | **Key plot** — accuracy drop vs compression ratio by ring size |
| `comm_latency.png` | Average communication latency per configuration |
| `throughput.png` | Training throughput per configuration |
| `bytes_saved.png` | Theoretical communication volume and savings |
| `failure_scenario.png` | Loss curve and val accuracy: clean vs crash run |

---

## Fault Tolerance

Ring AllReduce is **completely fault-intolerant**. When a worker crashes:

1. The crashed worker stops sending heartbeats
2. Orchestrator suspects failure after **5.0s** of silence
3. Orchestrator confirms failure after **10.0s**
4. Remaining workers are permanently blocked at `dist.all_reduce` barrier
5. Training halts — no automatic recovery

This is a fundamental property of ring topologies, not an implementation limitation. Contrast with Parameter Server architectures where the server can exclude failed workers and continue.

---

## Dependencies

```
torch==2.2.2+cu118
torchvision==0.17.2+cu118
numpy==1.26.4
pandas==2.2.2
matplotlib==3.9.0
seaborn==0.13.2
tqdm==4.66.4
```

---

## References

1. Alistarh et al. *QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding.* NeurIPS 2017. [arXiv:1610.02132](https://arxiv.org/abs/1610.02132)
2. Xin & Canini. *Global-QSGD: Allreduce-Compatible Quantization.* EuroMLSys 2025. [arXiv:2305.18627](https://arxiv.org/abs/2305.18627)
3. Lin et al. *Deep Gradient Compression.* ICLR 2018.
4. Dryden et al. *Communication Quantization for Data-Parallel Training.* [arXiv:1901.01544](https://arxiv.org/abs/1901.01544)

---

## Team

BSCS-13E — NUST School of Electrical Engineering and Computer Science
CS332 Parallel & Distributed Computing — Spring 2026