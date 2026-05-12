"""
analyze.py — Experiment Results Analysis & Plotting
CS332 | Distributed SGD + QSGD Compression

Generates all plots required by the rubric:
  1. convergence_curves.png     — val_acc vs epoch per bit-width, per ring size
  2. comm_latency.png           — avg comm latency per configuration
  3. throughput.png             — avg throughput per configuration
  4. sensitivity.png            — accuracy degradation vs compression ratio by ring size (KEY PLOT)
  5. bytes_saved.png            — communication bytes saved vs baseline
  6. failure_scenario.png       — loss curve with crash vs clean baseline

Usage:
  python analyze.py                          # reads from ./results, saves plots to ./plots
  python analyze.py --results ./results      # custom results dir
  python analyze.py --out ./plots            # custom output dir
"""

import os
import json
import argparse
import glob
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8f8f8",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    1.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "legend.framealpha": 0.9,
})

# Colour palette — one colour per bit-width, consistent across all plots
BIT_COLOURS = {32: "#2d6a9f", 8: "#2ca02c", 4: "#ff7f0e", 2: "#d62728"}
BIT_LABELS  = {32: "32-bit (baseline)", 8: "8-bit", 4: "4-bit", 2: "2-bit"}
BIT_ORDER   = [32, 8, 4, 2]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_epochs(results_dir: Path) -> pd.DataFrame:
    """Load all *_rank0_epochs.csv files into one DataFrame with run metadata."""
    frames = []
    for path in sorted(results_dir.glob("exp_w*_b*_rank0_epochs.csv")):
        stem = path.stem  # e.g. exp_w2_b32_rank0_epochs
        parts = stem.split("_")
        # parts: ['exp', 'w2', 'b32', 'rank0', 'epochs']
        try:
            world_size = int(parts[1][1:])
            num_bits   = int(parts[2][1:])
        except (IndexError, ValueError):
            print(f"  Skipping unrecognised filename: {path.name}")
            continue
        df = pd.read_csv(path)
        df["world_size"] = world_size
        df["num_bits"]   = num_bits
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No epoch CSV files found in {results_dir}")
    return pd.concat(frames, ignore_index=True)


def load_summaries(results_dir: Path) -> pd.DataFrame:
    """Load all *_rank0_summary.json files into one DataFrame."""
    rows = []
    for path in sorted(results_dir.glob("exp_w*_b*_rank0_summary.json")):
        stem  = path.stem
        parts = stem.split("_")
        try:
            world_size = int(parts[1][1:])
            num_bits   = int(parts[2][1:])
        except (IndexError, ValueError):
            continue
        with open(path) as f:
            d = json.load(f)
        d["world_size"] = world_size
        d["num_bits"]   = num_bits
        rows.append(d)

    if not rows:
        raise FileNotFoundError(f"No summary JSON files found in {results_dir}")
    return pd.DataFrame(rows)


def load_iterations(results_dir: Path, pattern: str) -> pd.DataFrame:
    """Load iteration CSVs matching a glob pattern."""
    frames = []
    for path in sorted(results_dir.glob(pattern)):
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compression_ratio(num_bits: int) -> float:
    return 32 / num_bits


def bytes_per_step(num_elements: int, num_bits: int, world_size: int) -> float:
    """Theoretical bytes sent per GPU per step using Ring AllReduce formula."""
    if num_bits == 32:
        payload = num_elements * 4
    else:
        import math
        payload = math.ceil(num_elements * num_bits / 8) + 4  # +4 for scale
    return 2 * (world_size - 1) / world_size * payload


# ── Plot 1: Convergence curves ────────────────────────────────────────────────

def plot_convergence(epochs_df: pd.DataFrame, out: Path):
    world_sizes = sorted(epochs_df["world_size"].unique())
    fig, axes = plt.subplots(1, len(world_sizes), figsize=(7 * len(world_sizes), 5),
                             sharey=True)
    if len(world_sizes) == 1:
        axes = [axes]

    for ax, ws in zip(axes, world_sizes):
        sub = epochs_df[epochs_df["world_size"] == ws]
        for bits in BIT_ORDER:
            bsub = sub[sub["num_bits"] == bits].sort_values("epoch")
            if bsub.empty:
                continue
            ax.plot(bsub["epoch"], bsub["val_acc"],
                    color=BIT_COLOURS[bits], label=BIT_LABELS[bits],
                    linewidth=2, marker="o", markersize=3)

        ax.set_title(f"{ws} Workers")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy (%)" if ws == world_sizes[0] else "")
        ax.legend(loc="lower right")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle("Convergence: Validation Accuracy vs Epoch", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = out / "convergence_curves.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 2: Comm latency bar chart ────────────────────────────────────────────

def plot_comm_latency(summaries: pd.DataFrame, out: Path):
    world_sizes = sorted(summaries["world_size"].unique())
    x     = np.arange(len(world_sizes))
    width = 0.18
    offsets = np.linspace(-(len(BIT_ORDER)-1)/2, (len(BIT_ORDER)-1)/2, len(BIT_ORDER)) * width

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, bits in enumerate(BIT_ORDER):
        vals = [
            summaries[(summaries["world_size"] == ws) &
                      (summaries["num_bits"] == bits)]["avg_comm_latency_ms"].mean()
            for ws in world_sizes
        ]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=BIT_LABELS[bits], color=BIT_COLOURS[bits], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ws} Workers" for ws in world_sizes])
    ax.set_ylabel("Avg Comm Latency (ms)")
    ax.set_title("Communication Latency per Configuration")
    ax.legend()
    fig.tight_layout()
    out_path = out / "comm_latency.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 3: Throughput bar chart ──────────────────────────────────────────────

def plot_throughput(summaries: pd.DataFrame, out: Path):
    world_sizes = sorted(summaries["world_size"].unique())
    x      = np.arange(len(world_sizes))
    width  = 0.18
    offsets = np.linspace(-(len(BIT_ORDER)-1)/2, (len(BIT_ORDER)-1)/2, len(BIT_ORDER)) * width

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, bits in enumerate(BIT_ORDER):
        vals = [
            summaries[(summaries["world_size"] == ws) &
                      (summaries["num_bits"] == bits)]["avg_throughput_samp_s"].mean()
            for ws in world_sizes
        ]
        ax.bar(x + offsets[i], vals, width,
               label=BIT_LABELS[bits], color=BIT_COLOURS[bits], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ws} Workers" for ws in world_sizes])
    ax.set_ylabel("Avg Throughput (samples/sec)")
    ax.set_title("Training Throughput per Configuration")
    ax.legend()
    fig.tight_layout()
    out_path = out / "throughput.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 4: Sensitivity — KEY PLOT ───────────────────────────────────────────

def plot_sensitivity(summaries: pd.DataFrame, epochs_df: pd.DataFrame, out: Path):
    """
    Accuracy degradation vs compression ratio, one line per ring size.
    This is the novel Gap 2 finding: does sensitivity to compression
    increase as ring size grows?
    """
    world_sizes  = sorted(summaries["world_size"].unique())
    ws_colours   = {2: "#1f77b4", 4: "#ff7f0e", 8: "#2ca02c"}
    ws_markers   = {2: "o", 4: "s", 8: "^"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: final val_acc vs compression ratio ──
    ax = axes[0]
    for ws in world_sizes:
        sub = summaries[summaries["world_size"] == ws].copy()
        sub["compression_ratio"] = sub["num_bits"].apply(compression_ratio)
        sub = sub.sort_values("compression_ratio")

        # Baseline accuracy (32-bit) for this world size
        baseline = sub[sub["num_bits"] == 32]["final_val_acc"].values
        if len(baseline) == 0:
            continue
        baseline = baseline[0]

        ax.plot(sub["compression_ratio"], sub["final_val_acc"],
                color=ws_colours.get(ws, "gray"),
                marker=ws_markers.get(ws, "o"),
                linewidth=2, markersize=7,
                label=f"{ws} Workers")

    ax.axvline(x=1, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Compression Ratio (32-bit / N-bit)")
    ax.set_ylabel("Final Validation Accuracy (%)")
    ax.set_title("Accuracy vs Compression Ratio by Ring Size")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}×"))

    # ── Right: accuracy DROP vs compression ratio ──
    ax = axes[1]
    for ws in world_sizes:
        sub = summaries[summaries["world_size"] == ws].copy()
        baseline_row = sub[sub["num_bits"] == 32]
        if baseline_row.empty:
            continue
        baseline_acc = baseline_row["final_val_acc"].values[0]

        sub["compression_ratio"] = sub["num_bits"].apply(compression_ratio)
        sub["acc_drop"]          = baseline_acc - sub["final_val_acc"]
        sub = sub.sort_values("compression_ratio")

        ax.plot(sub["compression_ratio"], sub["acc_drop"],
                color=ws_colours.get(ws, "gray"),
                marker=ws_markers.get(ws, "o"),
                linewidth=2, markersize=7,
                label=f"{ws} Workers")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Compression Ratio (32-bit / N-bit)")
    ax.set_ylabel("Accuracy Drop vs 32-bit Baseline (%)")
    ax.set_title("Sensitivity: Accuracy Degradation by Ring Size")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}×"))

    fig.suptitle("Gap 2 Finding: Compression Ratio × Ring Size Interaction",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = out / "sensitivity.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 5: Bytes saved ───────────────────────────────────────────────────────

def plot_bytes_saved(summaries: pd.DataFrame, out: Path):
    # ResNet-18 ~11.17M parameters
    num_elements = 11_173_962
    world_sizes  = sorted(summaries["world_size"].unique())

    rows = []
    for ws in world_sizes:
        baseline_bytes = bytes_per_step(num_elements, 32, ws)
        for bits in BIT_ORDER:
            comp_bytes  = bytes_per_step(num_elements, bits, ws)
            saving_pct  = (1 - comp_bytes / baseline_bytes) * 100
            ratio       = baseline_bytes / comp_bytes
            rows.append({
                "world_size":    ws,
                "num_bits":      bits,
                "baseline_mb":   baseline_bytes / 1e6,
                "compressed_mb": comp_bytes / 1e6,
                "saving_pct":    saving_pct,
                "ratio":         ratio,
            })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: bytes per step (grouped bar)
    ax = axes[0]
    x      = np.arange(len(world_sizes))
    width  = 0.18
    offsets = np.linspace(-(len(BIT_ORDER)-1)/2, (len(BIT_ORDER)-1)/2, len(BIT_ORDER)) * width
    for i, bits in enumerate(BIT_ORDER):
        sub  = df[df["num_bits"] == bits]
        vals = [sub[sub["world_size"] == ws]["compressed_mb"].values[0]
                for ws in world_sizes]
        ax.bar(x + offsets[i], vals, width,
               label=BIT_LABELS[bits], color=BIT_COLOURS[bits], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ws} Workers" for ws in world_sizes])
    ax.set_ylabel("Data Transferred per Step (MB)")
    ax.set_title("Communication Volume per Step")
    ax.legend()

    # Right: % saving vs baseline
    ax = axes[1]
    for i, bits in enumerate(BIT_ORDER):
        if bits == 32:
            continue
        sub  = df[df["num_bits"] == bits].sort_values("world_size")
        ax.plot(sub["world_size"], sub["saving_pct"],
                color=BIT_COLOURS[bits], label=BIT_LABELS[bits],
                linewidth=2, marker="o", markersize=7)
    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Communication Savings vs 32-bit (%)")
    ax.set_title("Communication Savings by Configuration")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    fig.suptitle("Workload Modeling: Communication Volume",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = out / "bytes_saved.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 6: Failure scenario ──────────────────────────────────────────────────

def plot_failure(results_dir: Path, out: Path):
    # Load clean baseline iterations
    clean_files = list(results_dir.glob("failure_clean_*_rank0_iterations.csv"))
    crash_files = list(results_dir.glob("failure_scenario_*_rank0_iterations.csv"))

    if not clean_files or not crash_files:
        print("  Skipping failure plot — missing failure_clean or failure_scenario files.")
        return

    clean_df = pd.read_csv(clean_files[0])
    crash_df = pd.read_csv(crash_files[0])

    # Smooth loss with rolling average for readability
    window = 20

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: loss curves
    ax = axes[0]
    clean_smooth = clean_df["loss"].rolling(window, min_periods=1).mean()
    crash_smooth = crash_df["loss"].rolling(window, min_periods=1).mean()

    ax.plot(clean_df.index, clean_smooth,
            color="#2d6a9f", linewidth=2, label="Clean run (no crash)")
    ax.plot(crash_df.index, crash_smooth,
            color="#d62728", linewidth=2, label="Crash: rank 1 killed @ iter 200",
            linestyle="--")

    # Mark crash point
    crash_iter = 200
    ax.axvline(x=crash_iter, color="#d62728", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.text(crash_iter + 10, ax.get_ylim()[1] * 0.95,
            "Worker crash", color="#d62728", fontsize=9)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss: Clean vs Failure")
    ax.legend()

    # Right: epoch-level val_acc comparison
    ax = axes[1]
    clean_epoch_files = list(results_dir.glob("failure_clean_*_rank0_epochs.csv"))
    crash_epoch_files = list(results_dir.glob("failure_scenario_*_rank0_epochs.csv"))

    if clean_epoch_files:
        ce = pd.read_csv(clean_epoch_files[0])
        ax.plot(ce["epoch"], ce["val_acc"],
                color="#2d6a9f", linewidth=2, marker="o", markersize=5,
                label="Clean run")

    if crash_epoch_files:
        cr = pd.read_csv(crash_epoch_files[0])
        ax.plot(cr["epoch"], cr["val_acc"],
                color="#d62728", linewidth=2, marker="s", markersize=5,
                linestyle="--", label="After crash")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Val Accuracy: Clean vs Failure")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle("Failure Scenario: Worker Crash at Iteration 200 (4-bit, 4 Workers)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = out / "failure_scenario.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(summaries: pd.DataFrame):
    print("\n" + "=" * 75)
    print(f"{'Config':<20} {'Val Acc':>9} {'Comm (ms)':>11} "
          f"{'Throughput':>12} {'Wall Time':>11}")
    print("=" * 75)
    for ws in sorted(summaries["world_size"].unique()):
        for bits in BIT_ORDER:
            row = summaries[(summaries["world_size"] == ws) &
                            (summaries["num_bits"] == bits)]
            if row.empty:
                continue
            r = row.iloc[0]
            cfg = f"w={ws} b={bits}"
            print(f"  {cfg:<18} {r['final_val_acc']:>8.2f}%"
                  f" {r['avg_comm_latency_ms']:>10.1f}ms"
                  f" {r['avg_throughput_samp_s']:>10.1f}/s"
                  f" {r['total_wall_time_s']/60:>9.1f}min")
        print()

    # Measurable improvement statement
    print("─" * 75)
    print("MEASURABLE IMPROVEMENT (vs 32-bit baseline):")
    for ws in sorted(summaries["world_size"].unique()):
        baseline = summaries[(summaries["world_size"] == ws) &
                             (summaries["num_bits"] == 32)]
        if baseline.empty:
            continue
        b_acc  = baseline.iloc[0]["final_val_acc"]
        b_comm = baseline.iloc[0]["avg_comm_latency_ms"]
        for bits in [8, 4, 2]:
            row = summaries[(summaries["world_size"] == ws) &
                            (summaries["num_bits"] == bits)]
            if row.empty:
                continue
            r = row.iloc[0]
            acc_drop  = b_acc - r["final_val_acc"]
            comm_save = (1 - r["avg_comm_latency_ms"] / b_comm) * 100
            ratio     = compression_ratio(bits)
            print(f"  w={ws} {bits}-bit: {ratio:.0f}× compression | "
                  f"comm savings={comm_save:+.1f}% | "
                  f"acc drop={acc_drop:+.2f}%")
    print("=" * 75)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("./results"))
    parser.add_argument("--out",     type=Path, default=Path("./plots"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {args.results}")
    epochs_df  = load_epochs(args.results)
    summaries  = load_summaries(args.results)

    configs = summaries.groupby(["world_size", "num_bits"]).size().reset_index()
    print(f"Found {len(configs)} configurations, {len(epochs_df)} epoch records\n")

    print("Generating plots...")
    plot_convergence(epochs_df, args.out)
    plot_comm_latency(summaries, args.out)
    plot_throughput(summaries, args.out)
    plot_sensitivity(summaries, epochs_df, args.out)
    plot_bytes_saved(summaries, args.out)
    plot_failure(args.results, args.out)

    print_summary_table(summaries)
    print(f"\nAll plots saved to: {args.out}/")


if __name__ == "__main__":
    main()
