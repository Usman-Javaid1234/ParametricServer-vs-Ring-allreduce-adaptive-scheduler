"""
plot_exp_a.py
Experiment A — Throughput Scaling Results.

Directory structure expected (relative to this script's location):
  ps/       exp_a_ps_bsp_n4_rank0_*.csv
  ring_ar/  exp_a_ring_ar_n4_rank0_*.csv

Usage (run from inside "Experiment A results" folder):
    python plot_exp_a.py

Or from anywhere:
    python plot_exp_a.py --results-dir "path/to/Experiment A results"
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
COLORS = {"ring_ar": "#2563EB", "ps": "#DC2626"}
LABELS = {"ring_ar": "Ring AllReduce", "ps": "Parameter Server (BSP)"}

CONFIGS = {
    "ring_ar": {"dir": "ring_ar", "pattern": "exp_a_ring_ar_n4"},
    "ps":      {"dir": "ps",      "pattern": "exp_a_ps_bsp_n4"},
}


def load_summary(results_dir: Path, config_key: str) -> dict | None:
    cfg   = CONFIGS[config_key]
    folder = results_dir / cfg["dir"]
    files  = sorted(folder.glob("*rank0*summary.json"))
    if not files:
        print(f"[MISSING] {folder} — no summary.json")
        return None
    with open(files[0]) as f:
        return json.load(f)


def load_iters(results_dir: Path, config_key: str) -> pd.DataFrame | None:
    cfg    = CONFIGS[config_key]
    folder = results_dir / cfg["dir"]
    files  = sorted(folder.glob("*rank0*iterations.csv"))
    if not files:
        print(f"[MISSING] {folder} — no iterations.csv")
        return None
    return pd.read_csv(files[0])


def amdahl_speedup(n, f_serial):
    return 1.0 / (f_serial + (1 - f_serial) / n)


def plot(results_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    print(f"Results dir: {results_dir}")

    # Load empirical data for n=4
    empirical = {}
    for config_key in CONFIGS:
        summary = load_summary(results_dir, config_key)
        iters   = load_iters(results_dir, config_key)
        if summary and iters is not None:
            empirical[config_key] = {"summary": summary, "iters": iters}
            print(f"  [{config_key}] "
                  f"throughput={summary['avg_throughput_samp_s']:.1f} samp/s | "
                  f"comm={summary['avg_comm_latency_ms']:.1f}ms | "
                  f"wall_time={summary['total_wall_time_s']:.0f}s")

    if not empirical:
        print("No data found.")
        return

    # Compute f_serial = comm_time / total_iter_time
    f_serials = {}
    for config_key, d in empirical.items():
        iters     = d["iters"]
        avg_comm  = iters["comm_latency_ms"].mean()
        avg_iter  = iters["iter_time_ms"].mean()
        f_serial  = avg_comm / avg_iter
        f_serials[config_key] = f_serial
        print(f"  [{config_key}] f_serial={f_serial:.3f} "
              f"(comm={avg_comm:.1f}ms / iter={avg_iter:.1f}ms)")

    n_values  = [4, 8, 16, 32]
    n_range   = np.linspace(1, 32, 300)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    # ---- (A) Measured throughput at n=4 ----
    ax1 = fig.add_subplot(gs[0, 0])
    archs = list(empirical.keys())
    tps   = [empirical[a]["summary"]["avg_throughput_samp_s"] for a in archs]
    bars  = ax1.bar([LABELS[a] for a in archs], tps,
                    color=[COLORS[a] for a in archs], alpha=0.8, width=0.5)
    for bar, val in zip(bars, tps):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", va="bottom", fontweight="bold")
    ax1.set_title("(A) Measured Throughput | n=4 workers")
    ax1.set_ylabel("Samples / sec")
    ax1.set_ylim(0, max(tps) * 1.3)

    # ---- (B) Amdahl predicted scaling to n=32 ----
    ax2 = fig.add_subplot(gs[0, 1])
    for config_key, d in empirical.items():
        tp_n4    = d["summary"]["avg_throughput_samp_s"]
        s_n4     = amdahl_speedup(4, f_serials[config_key])
        tp_n1    = tp_n4 / s_n4
        tp_pred  = tp_n1 * amdahl_speedup(np.array(n_values), f_serials[config_key])
        ax2.plot(n_values, tp_pred,
                 color=COLORS[config_key], label=LABELS[config_key],
                 linewidth=2.5, marker="o", markersize=7)
        # Mark measured point at n=4
        ax2.scatter([4], [tp_n4], color=COLORS[config_key],
                    s=150, zorder=5, marker="*")
    ax2.set_title("(B) Predicted Scaling (Amdahl's Law)\n★ = measured at n=4")
    ax2.set_xlabel("Number of Workers (n)")
    ax2.set_ylabel("Throughput (samp/s)")
    ax2.set_xticks(n_values)
    ax2.legend()

    # ---- (C) Communication latency distribution ----
    ax3 = fig.add_subplot(gs[1, 0])
    vals, lbls = [], []
    for config_key, d in empirical.items():
        vals.append(d["iters"]["comm_latency_ms"].dropna().values)
        lbls.append(LABELS[config_key])
    bplot = ax3.boxplot(vals, labels=lbls, patch_artist=True,
                        medianprops=dict(color="#1F2937", linewidth=2))
    for patch, config_key in zip(bplot["boxes"], empirical.keys()):
        patch.set_facecolor(COLORS[config_key])
        patch.set_alpha(0.6)
    ax3.set_title("(C) Comm Latency Distribution | n=4")
    ax3.set_ylabel("Latency (ms)")

    # ---- (D) Amdahl speedup curves ----
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(n_range, n_range, "k--", linewidth=1, alpha=0.4, label="Linear (ideal)")
    for config_key, d in empirical.items():
        speedup = amdahl_speedup(n_range, f_serials[config_key])
        ax4.plot(n_range, speedup,
                 color=COLORS[config_key], label=LABELS[config_key],
                 linewidth=2.5)
    ax4.set_title(
        f"(D) Amdahl Speedup Curves\n"
        f"f_serial — Ring AR: {f_serials.get('ring_ar', 0):.2f} | "
        f"PS: {f_serials.get('ps', 0):.2f}"
    )
    ax4.set_xlabel("Number of Workers (n)")
    ax4.set_ylabel("Speedup S(n)")
    ax4.set_xticks(n_values)
    ax4.legend()

    fig.suptitle(
        "Experiment A — Throughput Scaling: PS-BSP vs Ring AllReduce\n"
        "Empirical n=4 + Amdahl's Law extrapolation to n=32 | ResNet-18 | CIFAR-10",
        fontsize=12, fontweight="bold"
    )

    out_path = out_dir / "exp_a_scaling.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.show()

    # Summary table
    print("\n" + "=" * 55)
    print("EXPERIMENT A SUMMARY")
    print("=" * 55)
    for config_key, d in empirical.items():
        s = d["summary"]
        f = f_serials[config_key]
        print(f"\n{LABELS[config_key]}:")
        print(f"  Throughput (n=4)   : {s['avg_throughput_samp_s']:.1f} samp/s")
        print(f"  Comm latency       : {s['avg_comm_latency_ms']:.1f} ms")
        print(f"  Avg iter time      : {d['iters']['iter_time_ms'].mean():.1f} ms")
        print(f"  f_serial           : {f:.3f}")
        print(f"  Predicted n=8      : {amdahl_speedup(8,  f):.2f}x speedup")
        print(f"  Predicted n=16     : {amdahl_speedup(16, f):.2f}x speedup")
        print(f"  Predicted n=32     : {amdahl_speedup(32, f):.2f}x speedup")


if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir,
        help="Path to 'Experiment A results' folder (default: script location)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=script_dir / "plots",
        help="Where to save the output plot"
    )
    args = parser.parse_args()
    plot(args.results_dir, args.out_dir)
