"""
plot_exp_b.py
Experiment B — Straggler Sensitivity Results.

Directory structure expected (relative to this script's location):
  ps/delay50/        exp_b_ps_bsp_delay50_rank0_*.csv
  ps/delay200/       exp_b_ps_bsp_delay200_rank0_*.csv
  ps/delay500/       exp_b_ps_bsp_delay500_rank0_*.csv
  ps/ssp2_delay50/   exp_b_ps_ssp2_delay50_rank0_*.csv
  ps/ssp2_delay200/  exp_b_ps_ssp2_delay200_rank0_*.csv
  ps/ssp2_delay500/  exp_b_ps_ssp2_delay500_rank0_*.csv
  ring_ar/delay50/   exp_b_ring_ar_delay50_rank0_*.csv
  ring_ar/delay200/  exp_b_ring_ar_delay200_rank0_*.csv
  ring_ar/delay500/  exp_b_ring_ar_delay500_rank0_*.csv

Also needs baseline (no straggler) summaries from Baselines results folder.
If not found, delay=0 baseline is omitted from plots.

Usage (run from inside "Experiment B results" folder):
    python plot_exp_b.py

Or from anywhere:
    python plot_exp_b.py --results-dir "path/to/Experiment B results"
                         --baseline-dir "path/to/Baselines results"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

CONFIGS = {
    "ring_ar": {
        "color": "#2563EB",
        "label": "Ring AllReduce",
        "dir":   "ring_ar",
        "prefix": "exp_b_ring_ar",
    },
    "ps_bsp": {
        "color": "#DC2626",
        "label": "PS-BSP (τ=0)",
        "dir":   "ps",
        "prefix": "exp_b_ps_bsp",
    },
    "ps_ssp2": {
        "color": "#F59E0B",
        "label": "PS-SSP (τ=2)",
        "dir":   "ps",
        "prefix": "exp_b_ps_ssp2",
    },
}

DELAYS = [50, 200, 500]

# Subdir name per config per delay
SUBDIRS = {
    "ring_ar": {50: "delay50",      200: "delay200",      500: "delay500"},
    "ps_bsp":  {50: "delay50",      200: "delay200",      500: "delay500"},
    "ps_ssp2": {50: "ssp2_delay50", 200: "ssp2_delay200", 500: "ssp2_delay500"},
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_summary(results_dir: Path, config_key: str, delay: int) -> dict | None:
    cfg    = CONFIGS[config_key]
    subdir = SUBDIRS[config_key][delay]
    folder = results_dir / cfg["dir"] / subdir
    files  = sorted(folder.glob(f"{cfg['prefix']}_delay{delay if 'ssp' not in subdir else ''}{delay if 'ssp' in subdir else ''}_rank0_summary.json"))
    # Simpler glob — just find any summary json for rank0
    files  = sorted(folder.glob("*rank0*summary.json"))
    if not files:
        print(f"  [MISSING] {folder} — no summary.json")
        return None
    with open(files[0]) as f:
        return json.load(f)


def load_iters(results_dir: Path, config_key: str, delay: int) -> pd.DataFrame | None:
    cfg    = CONFIGS[config_key]
    subdir = SUBDIRS[config_key][delay]
    folder = results_dir / cfg["dir"] / subdir
    files  = sorted(folder.glob("*rank0*iterations.csv"))
    if not files:
        return None
    return pd.read_csv(files[0])


def load_epochs(results_dir: Path, config_key: str, delay: int) -> pd.DataFrame | None:
    cfg    = CONFIGS[config_key]
    subdir = SUBDIRS[config_key][delay]
    folder = results_dir / cfg["dir"] / subdir
    files  = sorted(folder.glob("*rank0*epochs.csv"))
    if not files:
        return None
    df = pd.read_csv(files[0])
    df = (df.sort_values("timestamp")
            .drop_duplicates(subset=["epoch"], keep="last")
            .dropna(subset=["val_loss"])
            .reset_index(drop=True))
    return df


def load_baseline_summary(baseline_dir: Path, config_key: str) -> dict | None:
    """Load no-straggler baseline from Baselines results folder."""
    if baseline_dir is None or not baseline_dir.exists():
        return None
    # ring_ar baseline is in ring_ar/ subdir, ps in ps/ subdir
    if config_key == "ring_ar":
        files = sorted((baseline_dir / "ring_ar").glob("*rank0*summary.json"))
    else:
        files = sorted((baseline_dir / "ps").glob("*bsp*rank0*summary.json"))
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot(results_dir: Path, baseline_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    print(f"Results dir : {results_dir}")
    print(f"Baseline dir: {baseline_dir}")

    # ------------------------------------------------------------------
    # Collect metrics across all delays for each config
    # ------------------------------------------------------------------
    data = {}   # config_key → {delay → {iter_time, throughput, comm_ms}}

    for config_key in CONFIGS:
        data[config_key] = {}
        for delay in DELAYS:
            summary = load_summary(results_dir, config_key, delay)
            iters   = load_iters(results_dir, config_key, delay)
            if summary and iters is not None:
                data[config_key][delay] = {
                    "throughput":  summary["avg_throughput_samp_s"],
                    "comm_ms":     summary["avg_comm_latency_ms"],
                    "iter_ms":     iters["iter_time_ms"].mean(),
                    "wall_time":   summary["total_wall_time_s"],
                }
                print(f"  [{config_key}] delay={delay}ms | "
                      f"tp={summary['avg_throughput_samp_s']:.1f} | "
                      f"comm={summary['avg_comm_latency_ms']:.1f}ms")
            else:
                data[config_key][delay] = None

    # Baseline (delay=0)
    baselines = {}
    for config_key in ("ring_ar", "ps_bsp"):
        s = load_baseline_summary(baseline_dir, config_key)
        if s:
            baselines[config_key] = s["avg_throughput_samp_s"]
            print(f"  [baseline {config_key}] tp={s['avg_throughput_samp_s']:.1f}")
    # SSP2 baseline same as BSP (no straggler → identical)
    if "ps_bsp" in baselines:
        baselines["ps_ssp2"] = baselines["ps_bsp"]

    # ------------------------------------------------------------------
    # Build plot arrays
    # ------------------------------------------------------------------
    x = np.array(DELAYS)

    def get_metric(config_key, metric):
        return [data[config_key][d][metric]
                if data[config_key][d] else np.nan
                for d in DELAYS]

    fig = plt.figure(figsize=(15, 11))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    # ---- (A) Avg iteration time vs straggler delay ----
    ax1 = fig.add_subplot(gs[0, 0])
    for config_key, cfg in CONFIGS.items():
        y = get_metric(config_key, "iter_ms")
        ax1.plot(x, y, color=cfg["color"], label=cfg["label"],
                 linewidth=2.5, marker="o", markersize=8)
    ax1.set_title("(A) Avg Iteration Time vs Straggler Delay")
    ax1.set_xlabel("Straggler Delay (ms)")
    ax1.set_ylabel("Avg Iteration Time (ms)")
    ax1.set_xticks(DELAYS)
    ax1.legend()

    # ---- (B) Throughput degradation % vs delay ----
    ax2 = fig.add_subplot(gs[0, 1])
    for config_key, cfg in CONFIGS.items():
        base = baselines.get(config_key)
        if base is None:
            # Use delay=50 as proxy baseline if no external baseline
            d50 = data[config_key].get(50)
            base = d50["throughput"] if d50 else None
        if base is None:
            continue
        y = [(((base - data[config_key][d]["throughput"]) / base * 100)
               if data[config_key][d] else np.nan)
              for d in DELAYS]
        ax2.plot(x, y, color=cfg["color"], label=cfg["label"],
                 linewidth=2.5, marker="o", markersize=8)
    ax2.set_title("(B) Throughput Degradation vs Straggler Delay")
    ax2.set_xlabel("Straggler Delay (ms)")
    ax2.set_ylabel("Throughput Drop (%)")
    ax2.set_xticks(DELAYS)
    ax2.legend()

    # ---- (C) Communication latency vs delay ----
    ax3 = fig.add_subplot(gs[1, 0])
    for config_key, cfg in CONFIGS.items():
        y = get_metric(config_key, "comm_ms")
        ax3.plot(x, y, color=cfg["color"], label=cfg["label"],
                 linewidth=2.5, marker="o", markersize=8)
    ax3.set_title("(C) Avg Comm Latency vs Straggler Delay")
    ax3.set_xlabel("Straggler Delay (ms)")
    ax3.set_ylabel("Avg Comm Latency (ms)")
    ax3.set_xticks(DELAYS)
    ax3.legend()

    # ---- (D) Val accuracy under worst straggler (500ms) ----
    ax4 = fig.add_subplot(gs[1, 1])
    worst_delay = 500
    plotted = False
    for config_key, cfg in CONFIGS.items():
        epochs = load_epochs(results_dir, config_key, worst_delay)
        if epochs is not None and len(epochs) > 0:
            ax4.plot(epochs["epoch"], epochs["val_acc"],
                     color=cfg["color"], label=cfg["label"],
                     linewidth=2.5, marker="o", markersize=6)
            plotted = True
    if not plotted:
        ax4.text(0.5, 0.5, "No epoch data found\nfor delay=500ms",
                 ha="center", va="center", transform=ax4.transAxes)
    ax4.set_title(f"(D) Validation Accuracy | {worst_delay}ms Straggler")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Validation Accuracy (%)")
    ax4.legend()

    fig.suptitle(
        "Experiment B — Straggler Sensitivity Analysis\n"
        "1 straggler (rank 3) | n=4 workers | ResNet-18 | CIFAR-10",
        fontsize=13, fontweight="bold"
    )

    out_path = out_dir / "exp_b_straggler.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.show()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT B SUMMARY")
    print("=" * 60)
    for config_key, cfg in CONFIGS.items():
        print(f"\n{cfg['label']}:")
        base = baselines.get(config_key)
        for delay in DELAYS:
            d = data[config_key][delay]
            if d is None:
                print(f"  delay={delay:>4}ms | NO DATA")
                continue
            drop = (((base - d["throughput"]) / base * 100)
                    if base else 0.0)
            print(f"  delay={delay:>4}ms | "
                  f"throughput={d['throughput']:>6.1f} samp/s | "
                  f"drop={drop:>5.1f}% | "
                  f"comm={d['comm_ms']:>6.1f}ms | "
                  f"iter={d['iter_ms']:>7.1f}ms")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir,
        help="Path to 'Experiment B results' folder (default: script location)"
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=script_dir.parent / "Baselines results",
        help="Path to 'Baselines results' folder for no-straggler reference"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=script_dir / "plots",
        help="Where to save the output plot"
    )
    args = parser.parse_args()
    plot(args.results_dir, args.baseline_dir, args.out_dir)
