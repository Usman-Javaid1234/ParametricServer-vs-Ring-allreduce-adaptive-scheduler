"""
scripts/plot_results.py
Plots Phase 1 baseline results from the backed-up directory structure:
  results/ps/        — PS-BSP results (baseline_ps_bsp_rank0_*.csv)
  results/ring_ar/   — Ring AR results (baseline_rank0_*.csv)

Usage:
    python scripts/plot_results.py --results-dir ./results
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"ring_ar": "#2563EB", "ps": "#DC2626"}
LABELS = {"ring_ar": "Ring AllReduce", "ps": "Parameter Server (BSP)"}


def load_arch(directory: Path, arch: str) -> dict | None:
    """Load rank-0 iteration and epoch CSVs from a directory."""
    iter_files  = sorted(directory.glob("*rank0*iterations.csv"))
    epoch_files = sorted(directory.glob("*rank0*epochs.csv"))

    if not iter_files:
        print(f"[WARNING] No iteration CSV found in {directory}")
        return None
    if not epoch_files:
        print(f"[WARNING] No epoch CSV found in {directory}")
        return None

    iters  = pd.read_csv(iter_files[0])
    epochs = pd.read_csv(epoch_files[0])

    # Drop duplicate epoch rows — keep the one with val_loss filled
    if "epoch" in epochs.columns:
        epochs = (epochs
                  .sort_values("timestamp")
                  .drop_duplicates(subset=["epoch"], keep="last")
                  .reset_index(drop=True))

    # Drop rows missing val metrics
    epochs = epochs.dropna(subset=["val_loss", "val_acc"]).reset_index(drop=True)

    print(f"[{arch}] {len(iters)} iterations | {len(epochs)} epochs | "
          f"file: {iter_files[0].name}")
    return {"iters": iters, "epochs": epochs}


def plot(results_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load each arch from its subdirectory
    data = {}
    for arch, subdir in [("ring_ar", "ring_ar"), ("ps", "ps")]:
        d = results_dir / subdir
        if not d.exists():
            print(f"[WARNING] Directory not found: {d}")
            continue
        result = load_arch(d, arch)
        if result:
            data[arch] = result

    if not data:
        print("No data found. Check that results/ps/ and results/ring_ar/ exist.")
        return

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.32)

    # ---- (A) Training Loss per Epoch ----
    ax1 = fig.add_subplot(gs[0, 0])
    for arch, d in data.items():
        ep = d["epochs"]
        ax1.plot(ep["epoch"], ep["train_loss"],
                 color=COLORS[arch], label=LABELS[arch],
                 linewidth=2, marker="o", markersize=5)
    ax1.set_title("(A) Training Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()

    # ---- (B) Validation Accuracy per Epoch ----
    ax2 = fig.add_subplot(gs[0, 1])
    for arch, d in data.items():
        ep = d["epochs"]
        ax2.plot(ep["epoch"], ep["val_acc"],
                 color=COLORS[arch], label=LABELS[arch],
                 linewidth=2, marker="o", markersize=5)
    ax2.set_title("(B) Validation Accuracy per Epoch (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    # ---- (C) Throughput over Iterations ----
    ax3 = fig.add_subplot(gs[1, 0])
    for arch, d in data.items():
        it = d["iters"].copy()
        it["smooth"] = it["throughput_samp_s"].rolling(20, min_periods=1).mean()
        ax3.plot(range(len(it)), it["smooth"],
                 color=COLORS[arch], label=LABELS[arch],
                 linewidth=1.5, alpha=0.9)
    ax3.set_title("(C) Throughput (samples/sec)")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Samples / sec")
    ax3.legend()

    # ---- (D) Communication Latency Distribution ----
    ax4 = fig.add_subplot(gs[1, 1])
    vals, lbls = [], []
    for arch, d in data.items():
        vals.append(d["iters"]["comm_latency_ms"].dropna().values)
        lbls.append(LABELS[arch])
    bplot = ax4.boxplot(vals, labels=lbls, notch=False, patch_artist=True,
                        medianprops=dict(color="#1F2937", linewidth=2))
    for patch, arch in zip(bplot["boxes"], data.keys()):
        patch.set_facecolor(COLORS[arch])
        patch.set_alpha(0.6)
    ax4.set_title("(D) Comm Latency Distribution (ms)")
    ax4.set_ylabel("Latency (ms)")

    fig.suptitle(
        "Phase 1 Baseline: PS-BSP vs Ring AllReduce\n"
        "n=4 workers | ResNet-18 | CIFAR-10 | batch=32",
        fontsize=13, fontweight="bold"
    )

    out_path = out_dir / "baseline_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.show()

    # ---- Summary Table ----
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for arch, d in data.items():
        iters  = d["iters"]
        epochs = d["epochs"]
        print(f"\n{LABELS[arch]}:")
        print(f"  Epochs completed   : {len(epochs)}")
        print(f"  Final train loss   : {epochs['train_loss'].iloc[-1]:.4f}")
        print(f"  Final val loss     : {epochs['val_loss'].iloc[-1]:.4f}")
        print(f"  Final val acc      : {epochs['val_acc'].iloc[-1]:.2f}%")
        print(f"  Avg throughput     : {iters['throughput_samp_s'].mean():.1f} samp/s")
        print(f"  Avg comm latency   : {iters['comm_latency_ms'].mean():.1f} ms")
        print(f"  Avg iter time      : {iters['iter_time_ms'].mean():.1f} ms")
        print(f"  Total iterations   : {len(iters)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("./Baselines results"))
    parser.add_argument("--out-dir",     type=Path, default=Path("./Baselines results/plots"))
    args = parser.parse_args()
    plot(args.results_dir, args.out_dir)