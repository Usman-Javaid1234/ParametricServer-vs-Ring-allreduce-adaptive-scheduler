"""
scripts/plot_baseline.py
Plots Phase 1 baseline results from CSV files in ./results/.

Produces:
  1. Loss curves (PS-BSP vs Ring AR)
  2. Throughput (samples/sec) over training
  3. Communication latency per iteration
  4. Summary bar chart

Usage:
    python scripts/plot_baseline.py --results-dir ./results
"""

import argparse
import json
import glob
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"ring_ar": "#2563EB", "ps": "#DC2626"}
LABELS = {"ring_ar": "Ring AllReduce", "ps": "Parameter Server (BSP)"}


# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> dict:
    """Load all iteration and epoch CSVs, keyed by arch."""
    data = {}
    for arch in ("ring_ar", "ps"):
        iter_files  = sorted(results_dir.glob(f"*{arch}*_rank0_iterations.csv"))
        epoch_files = sorted(results_dir.glob(f"*{arch}*_rank0_epochs.csv"))

        if not iter_files:
            print(f"[WARNING] No iteration data found for arch={arch}")
            continue

        iters  = pd.concat([pd.read_csv(f) for f in iter_files],  ignore_index=True)
        epochs = pd.concat([pd.read_csv(f) for f in epoch_files], ignore_index=True)
        data[arch] = {"iters": iters, "epochs": epochs}
    return data


def plot_baseline(results_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_results(results_dir)

    if not data:
        print("No results found. Run the baseline experiments first.")
        return

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    # ---- (A) Training loss per epoch ----
    ax1 = fig.add_subplot(gs[0, 0])
    for arch, d in data.items():
        ep = d["epochs"]
        ax1.plot(ep["epoch"], ep["train_loss"],
                 color=COLORS[arch], label=LABELS[arch], linewidth=2)
    ax1.set_title("(A) Training Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()

    # ---- (B) Validation accuracy per epoch ----
    ax2 = fig.add_subplot(gs[0, 1])
    for arch, d in data.items():
        ep = d["epochs"]
        if "val_acc" in ep.columns:
            ax2.plot(ep["epoch"], ep["val_acc"],
                     color=COLORS[arch], label=LABELS[arch], linewidth=2)
    ax2.set_title("(B) Validation Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    # ---- (C) Throughput over iterations ----
    ax3 = fig.add_subplot(gs[1, 0])
    for arch, d in data.items():
        it = d["iters"]
        # Rolling average to smooth noise
        it["smooth_tp"] = it["throughput_samp_s"].rolling(20, min_periods=1).mean()
        ax3.plot(it.index, it["smooth_tp"],
                 color=COLORS[arch], label=LABELS[arch], linewidth=1.5, alpha=0.85)
    ax3.set_title("(C) Throughput (samples/sec)")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Samples / sec")
    ax3.legend()

    # ---- (D) Communication latency distribution ----
    ax4 = fig.add_subplot(gs[1, 1])
    comm_data, labels = [], []
    for arch, d in data.items():
        comm_data.append(d["iters"]["comm_latency_ms"].dropna().values)
        labels.append(LABELS[arch])
    ax4.boxplot(comm_data, labels=labels, notch=True,
                patch_artist=True,
                boxprops=dict(facecolor="#E5E7EB"),
                medianprops=dict(color="#1F2937", linewidth=2))
    ax4.set_title("(D) Comm Latency Distribution (ms)")
    ax4.set_ylabel("Latency (ms)")

    fig.suptitle(
        "Phase 1 Baseline: PS-BSP vs Ring AllReduce | n=4 workers | ResNet-50 CIFAR-10",
        fontsize=13, fontweight="bold"
    )

    out_path = out_dir / "baseline_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()

    # ---- Print summary table ----
    print("\n=== Summary ===")
    for arch, d in data.items():
        iters = d["iters"]
        epochs = d["epochs"]
        print(f"\n{LABELS[arch]}:")
        print(f"  Avg throughput   : {iters['throughput_samp_s'].mean():.0f} samp/s")
        print(f"  Avg comm latency : {iters['comm_latency_ms'].mean():.2f} ms")
        print(f"  Avg iter time    : {iters['iter_time_ms'].mean():.2f} ms")
        if "val_acc" in epochs.columns and len(epochs) > 0:
            print(f"  Final val acc    : {epochs['val_acc'].iloc[-1]:.2f}%")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    parser.add_argument("--out-dir",     type=Path, default=Path("./results/plots"))
    args = parser.parse_args()
    plot_baseline(args.results_dir, args.out_dir)
