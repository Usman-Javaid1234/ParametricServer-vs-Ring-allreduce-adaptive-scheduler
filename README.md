# Distributed SGD: Parameter Server vs Ring AllReduce
**CS332 — Parallel and Distributed Computing | NUST-SEECS | BSCS-13E**

---

## Project Structure

```
distributed_sgd/
├── src/
│   ├── worker.py          # Shared training loop (pluggable backend)
│   ├── ps_server.py       # Parameter Server (BSP + SSP modes)
│   ├── ring_ar.py         # Ring AllReduce (native + manual)
│   ├── orchestrator.py    # Heartbeat monitor + straggler detector
│   ├── metrics.py         # CSV + JSON metrics collector
│   └── entrypoint.py      # Docker container entrypoint
├── scripts/
│   ├── run_baseline.sh    # Phase 1 baseline runner
│   ├── plot_baseline.py   # Visualise baseline results
│   └── setup_wan_emulation.sh  # tc-netem WAN emulation
├── results/               # Output CSV/JSON/plots (git-ignored)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Docker Setup (Step-by-Step)

### Step 1 — Install Docker Desktop

**Windows / Mac:**
1. Go to https://www.docker.com/products/docker-desktop
2. Download Docker Desktop for your OS
3. Run the installer (accept all defaults)
4. Launch Docker Desktop — wait for the whale icon in the taskbar to stop animating
5. Open a terminal and verify:
   ```bash
   docker --version
   # Expected: Docker version 25.x.x or later
   docker compose version
   # Expected: Docker Compose version v2.x.x
   ```

**Ubuntu/Debian Linux:**
```bash
# Install Docker Engine
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Run Docker without sudo (log out and back in after this)
sudo usermod -aG docker $USER

# Verify
docker --version
docker compose version
```

---

### Step 2 — Clone / Copy the Project

```bash
# Copy the distributed_sgd/ folder to your machine, then:
cd distributed_sgd/
```

---

### Step 3 — Build the Docker Image

This downloads PyTorch and all dependencies (~2 GB, one time only):

```bash
docker compose build
```

Expected output (last few lines):
```
 => [worker 6/6] COPY src/ /app/src/    0.1s
 => exporting to image                  2.3s
 => => naming to docker.io/library/distributed_sgd-worker
```

---

### Step 4 — Run Phase 1 Baseline

**Option A — Quick single command:**
```bash
bash scripts/run_baseline.sh
```
This runs Ring AR then PS-BSP, each with 4 workers for 5 epochs.
Results land in `./results/`.

**Option B — Manual (more control):**

Ring AllReduce baseline:
```bash
BACKEND=ring_ar WORLD_SIZE=4 EPOCHS=5 RUN_ID=baseline_ring_ar \
    docker compose up --abort-on-container-exit
docker compose down
```

Parameter Server BSP baseline:
```bash
BACKEND=ps WORLD_SIZE=4 EPOCHS=5 TAU=0 RUN_ID=baseline_ps_bsp \
    docker compose up --abort-on-container-exit
docker compose down
```

---

### Step 5 — View Results

```bash
# Install matplotlib locally (just for plotting)
pip install matplotlib pandas seaborn

# Plot the results
python scripts/plot_baseline.py --results-dir ./results

# Or inspect raw CSV
ls results/
cat results/baseline_ring_ar_rank0_summary.json
```

---

## Environment Variables (full reference)

| Variable | Default | Description |
|---|---|---|
| `BACKEND` | `ring_ar` | `ring_ar` or `ps` |
| `WORLD_SIZE` | `4` | Number of workers |
| `EPOCHS` | `5` | Training epochs |
| `BATCH_SIZE` | `64` | Per-worker batch size |
| `LR` | `0.01` | Learning rate |
| `TAU` | `0` | SSP staleness bound (0=BSP) |
| `RUN_ID` | `baseline` | Prefix for result files |
| `DIST_BACKEND` | `gloo` | `gloo` (CPU) or `nccl` (GPU) |

---

## Scaling Workers

```bash
# 8 workers
WORLD_SIZE=8 docker compose up --scale worker_1=1 --scale worker_2=1 \
    --scale worker_3=1  # add worker_4..7 entries to docker-compose.yml

# Or use torchrun for multi-process on a single machine:
torchrun --nproc_per_node=4 src/worker.py \
    --backend ring_ar --model resnet50 --dataset cifar10 --epochs 5
```

---

## WAN Emulation (optional)

To simulate 100 Mbps / 20ms latency between workers:

```bash
# Inside each worker container
docker exec -it distributed_sgd-worker_1-1 bash
bash /app/scripts/setup_wan_emulation.sh --latency 20ms --bandwidth 100mbit
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `docker: command not found` | Install Docker Desktop (Step 1) |
| `permission denied /var/run/docker.sock` | Run `sudo usermod -aG docker $USER` then log out/in |
| Port 29500 already in use | `docker compose down` then retry, or change `MASTER_PORT` |
| Workers hang at `dist.init_process_group` | Check all containers can ping `worker_0`; verify `WORLD_SIZE` matches actual worker count |
| Out of memory | Reduce `BATCH_SIZE` to 32 or 16 |
| CIFAR-10 download fails | Mount a pre-downloaded dataset volume or check internet access inside container |

---

## What Phase 1 Produces

After `run_baseline.sh` completes you will have:

```
results/
├── baseline_ring_ar_rank0_iterations.csv   # per-iteration metrics
├── baseline_ring_ar_rank0_epochs.csv       # per-epoch loss + val acc
├── baseline_ring_ar_rank0_summary.json     # final summary stats
├── baseline_ps_bsp_rank0_iterations.csv
├── baseline_ps_bsp_rank0_epochs.csv
├── baseline_ps_bsp_rank0_summary.json
└── plots/
    └── baseline_results.png                # 4-panel comparison figure
```

These files are the **Comparative Baseline Analysis** slide for Deliverable 3.

---

## Phase Roadmap

| Phase | Status | What to run |
|---|---|---|
| 1 — Baseline | ✅ Complete | `run_baseline.sh` |
| 2 — Experiments A & B | 🔜 Next | `run_exp_a_scaling.sh`, `run_exp_b_straggler.sh` |
| 3 — Fault Recovery & Compression | 🔜 | `run_exp_c_fault.sh`, `run_exp_d_compression.sh` |
| 4 — Hybrid Prototype | 🔜 | `run_exp_e_hybrid.sh` |
