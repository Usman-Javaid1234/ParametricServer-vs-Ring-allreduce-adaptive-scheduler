"""
scripts/plot_exp_b.py
Experiment B — Straggler Sensitivity Results.

Plots:
  (A) Avg iteration time vs straggler delay for PS-BSP, PS-SSP, Ring AR
  (B) Throughput degradation (%) vs straggler delay
  (C) Comm latency increase vs straggler delay
  (D) Loss curves under worst straggler (500ms) for all three modes

Usage:
    python scripts/plot_exp_b.py --results-dir ./results/exp_b
"""

import argparse
import json
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

CONFIGS = {
    "ring_ar":  {"color": "#2563EB", "label": "Ring AllReduce",    "pattern": "ring_ar"},
    "ps_bsp":   {"color": "#DC2626", "label": "PS-BSP (τ=0)",      "pattern": "ps_bsp"},
    "ps_ssp2":  {"color": "#F59E0B", "label": "PS-SSP (τ=2)",      "pattern": "ps_ssp2"},
}
DELAYS = [0, 50, 200, 500]   # 0 = no straggler (baseline)


def load_summary(results_dir: Path, config_key: str, delay: int) -> dict | None:
    pattern = CONFIGS[config_key]["pattern"]
    if delay == 0:
        # No-straggler baseline — check parent results dir
        files = sorted(results_dir.parent.glob(f"*{pattern}*rank0*summary.json"))
    else:
        files = sorted(results_dir.glob(f"*{pattern}*delay{delay}*rank0*summary.json"))
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def load_iters(results_dir: Path, config_key: str, delay: int) -> pd.DataFrame | None:
    pattern = CONFIGS[config_key]["pattern"]
    if delay == 0:
        files = sorted(results_dir.parent.glob(f"*{pattern}*rank0*iterations.csv"))
    else:
        files = sorted(results_dir.glob(f"*{pattern}*delay{delay}*rank0*iterations.csv"))
    if not files:
        return None
    return pd.read_csv(files[0])


def load_epochs(results_dir: Path, config_key: str, delay: int) -> pd.DataFrame | None:
    pattern = CONFIGS[config_key]["pattern"]
    if delay == 0:
        files = sorted(results_dir.parent.glob(f"*{pattern}*rank0*epochs.csv"))
    else:
        files = sorted(results_dir.glob(f"*{pattern}*delay{delay}*rank0*epochs.csv"))
    if not files:
        return None
    df = pd.read_csv(files[0])
    df = (df.sort_values("timestamp")
            .drop_duplicates(subset=["epoch"], keep="last")
            .dropna(subset=["val_loss"])
            .reset_index(drop=True))
    return df


def plot(results_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics across all delays and configs
    records = {k: {"iter_time": [], "throughput": [], "comm_latency": []}
               for k in CONFIGS}

    for config_key in CONFIGS:
        for delay in DELAYS:
            summary = load_summary(results_dir, config_key, delay)
            iters   = load_iters(results_dir, config_key, delay)
            if summary:
                records[config_key]["iter_time"].append(
                    iters["iter_time_ms"].mean() if iters is not None else None)
                records[config_key]["throughput"].append(
                    summary["avg_throughput_samp_s"])
                records[config_key]["comm_latency"].append(
                    summary["avg_comm_latency_ms"])
            else:
                records[config_key]["iter_time"].append(None)
                records[config_key]["throughput"].append(None)
                records[config_key]["comm_latency"].append(None)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    delays_plot = [d for d in DELAYS if d > 0]   # x-axis for straggler plots

    # ---- (A) Avg iteration time vs delay ----
    ax1 = fig.add_subplot(gs[0, 0])
    for config_key, cfg in CONFIGS.items():
        vals = records[config_key]["iter_time"]
        # baseline (delay=0) as horizontal reference
        baseline = vals[0]
        y = [v for v in vals[1:] if v is not None]
        x = [d for d, v in zip(DELAYS[1:], vals[1:]) if v is not None]
        if x:
            ax1.plot(x, y, color=cfg["color"], label=cfg["label"],
                     linewidth=2, marker="o", markersize=6)
            if baseline:
                ax1.axhline(baseline, color=cfg["color"],
                            linestyle="--", alpha=0.4, linewidth=1)
    ax1.set_title("(A) Avg Iteration Time vs Straggler Delay")
    ax1.set_xlabel("Straggler Delay (ms)")
    ax1.set_ylabel("Avg Iter Time (ms)")
    ax1.set_xticks(delays_plot)
    ax1.legend(fontsize=9)

    # ---- (B) Throughput degradation (%) ----
    ax2 = fig.add_subplot(gs[0, 1])
    for config_key, cfg in CONFIGS.items():
        vals    = records[config_key]["throughput"]
        baseline = vals[0]
        if not baseline:
            continue
        y = [((baseline - v) / baseline * 100) if v else None
             for v in vals[1:]]
        x = [d for d, v in zip(DELAYS[1:], y) if v is not None]
        y = [v for v in y if v is not None]
        if x:
            ax2.plot(x, y, color=cfg["color"], label=cfg["label"],
                     linewidth=2, marker="o", markersize=6)
    ax2.set_title("(B) Throughput Degradation vs Straggler Delay")
    ax2.set_xlabel("Straggler Delay (ms)")
    ax2.set_ylabel("Throughput Drop (%)")
    ax2.set_xticks(delays_plot)
    ax2.legend(fontsize=9)

    # ---- (C) Communication latency vs delay ----
    ax3 = fig.add_subplot(gs[1, 0])
    for config_key, cfg in CONFIGS.items():
        vals = records[config_key]["comm_latency"]
        y = [v for v in vals[1:] if v is not None]
        x = [d for d, v in zip(DELAYS[1:], vals[1:]) if v is not None]
        if x:
            ax3.plot(x, y, color=cfg["color"], label=cfg["label"],
                     linewidth=2, marker="o", markersize=6)
    ax3.set_title("(C) Comm Latency vs Straggler Delay")
    ax3.set_xlabel("Straggler Delay (ms)")
    ax3.set_ylabel("Avg Comm Latency (ms)")
    ax3.set_xticks(delays_plot)
    ax3.legend(fontsize=9)

    # ---- (D) Loss curves under 500ms straggler ----
    ax4 = fig.add_subplot(gs[1, 1])
    worst_delay = 500
    for config_key, cfg in CONFIGS.items():
        epochs = load_epochs(results_dir, config_key, worst_delay)
        if epochs is not None and len(epochs) > 0:
            ax4.plot(epochs["epoch"], epochs["val_acc"],
                     color=cfg["color"], label=cfg["label"],
                     linewidth=2, marker="o", markersize=5)
    ax4.set_title(f"(D) Val Accuracy Under {worst_delay}ms Straggler")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Validation Accuracy (%)")
    ax4.legend(fontsize=9)

    fig.suptitle(
        "Experiment B — Straggler Sensitivity\n"
        "1 straggler worker (rank 3) | n=4 workers | PS-BSP vs PS-SSP(τ=2) vs Ring AR",
        fontsize=12, fontweight="bold"
    )

    out_path = out_dir / "exp_b_straggler.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.show()

    # Summary
    print("\n=== Experiment B Summary ===")
    for config_key, cfg in CONFIGS.items():
        print(f"\n{cfg['label']}:")
        baseline_tp = records[config_key]["throughput"][0]
        for i, delay in enumerate(DELAYS):
            tp = records[config_key]["throughput"][i]
            if tp is None:
                continue
            drop = ((baseline_tp - tp) / baseline_tp * 100) if baseline_tp else 0
            print(f"  delay={delay:>4}ms | throughput={tp:.1f} samp/s "
                  f"| drop={drop:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("./results/exp_b"))
    parser.add_argument("--out-dir",     type=Path, default=Path("./results/exp_b/plots"))
    args = parser.parse_args()
    plot(args.results_dir, args.out_dir)
