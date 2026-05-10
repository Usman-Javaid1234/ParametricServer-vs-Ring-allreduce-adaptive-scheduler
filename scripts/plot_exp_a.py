"""
scripts/plot_exp_a.py
Experiment A — Throughput Scaling Results.

Plots:
  (A) Measured throughput at n=4 workers (empirical)
  (B) Analytical scaling model extrapolated to n=4,8,16,32
      using Amdahl's Law with measured f_serial
  (C) Communication latency comparison PS vs Ring AR
  (D) Speedup relative to n=1 (analytical)

Usage:
    python scripts/plot_exp_a.py --results-dir ./results/exp_a
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
COLORS = {"ring_ar": "#2563EB", "ps": "#DC2626", "ps_ssp": "#F59E0B"}
LABELS = {"ring_ar": "Ring AllReduce", "ps": "PS-BSP", "ps_ssp": "PS-SSP"}


def load_summary(results_dir: Path, arch: str, n: int) -> dict | None:
    files = sorted(results_dir.glob(f"*{arch}*n{n}*rank0*summary.json"))
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def load_iters(results_dir: Path, arch: str, n: int) -> pd.DataFrame | None:
    files = sorted(results_dir.glob(f"*{arch}*n{n}*rank0*iterations.csv"))
    if not files:
        return None
    return pd.read_csv(files[0])


def amdahl_speedup(n_workers, f_serial):
    """S(n) = 1 / (f + (1-f)/n)"""
    return 1.0 / (f_serial + (1 - f_serial) / n_workers)


def plot(results_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    n_values = [4, 8, 16, 32]
    empirical_n = 4   # only n=4 is measured; rest are analytical

    # Load empirical data for n=4
    empirical = {}
    for arch in ("ring_ar", "ps"):
        summary = load_summary(results_dir, arch, empirical_n)
        iters   = load_iters(results_dir, arch, empirical_n)
        if summary and iters is not None:
            empirical[arch] = {"summary": summary, "iters": iters}
            print(f"[{arch}] n={empirical_n} | "
                  f"throughput={summary['avg_throughput_samp_s']:.1f} samp/s | "
                  f"comm={summary['avg_comm_latency_ms']:.1f}ms")

    if not empirical:
        print("No empirical data found in", results_dir)
        return

    # Compute f_serial from measured comm / total iter time
    f_serials = {}
    for arch, d in empirical.items():
        iters = d["iters"]
        avg_comm = iters["comm_latency_ms"].mean()
        avg_iter = iters["iter_time_ms"].mean()
        f_serial = avg_comm / avg_iter
        f_serials[arch] = f_serial
        print(f"[{arch}] f_serial={f_serial:.3f} "
              f"(comm={avg_comm:.1f}ms / iter={avg_iter:.1f}ms)")

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # ---- (A) Empirical throughput at n=4 ----
    ax1 = fig.add_subplot(gs[0, 0])
    archs = list(empirical.keys())
    tps   = [empirical[a]["summary"]["avg_throughput_samp_s"] for a in archs]
    bars  = ax1.bar([LABELS[a] for a in archs], tps,
                    color=[COLORS[a] for a in archs], alpha=0.8, width=0.5)
    for bar, val in zip(bars, tps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontweight="bold")
    ax1.set_title("(A) Measured Throughput | n=4 workers")
    ax1.set_ylabel("Samples / sec")
    ax1.set_ylim(0, max(tps) * 1.25)

    # ---- (B) Analytical Amdahl scaling to n=32 ----
    ax2 = fig.add_subplot(gs[0, 1])
    n_range = np.array(n_values)
    for arch in ("ring_ar", "ps"):
        if arch not in f_serials:
            continue
        # Base throughput at n=1 (extrapolate from n=4 measurement)
        tp_n4    = empirical[arch]["summary"]["avg_throughput_samp_s"]
        s_n4     = amdahl_speedup(4, f_serials[arch])
        tp_n1    = tp_n4 / s_n4
        speedups = amdahl_speedup(n_range, f_serials[arch])
        tp_pred  = tp_n1 * speedups
        ax2.plot(n_range, tp_pred, color=COLORS[arch], label=LABELS[arch],
                 linewidth=2, marker="o", markersize=6)
        # Mark the empirical point
        ax2.scatter([4], [tp_n4], color=COLORS[arch], s=120,
                    zorder=5, marker="*", label=f"{LABELS[arch]} (measured)")
    ax2.set_title("(B) Predicted Scaling (Amdahl's Law)")
    ax2.set_xlabel("Number of Workers (n)")
    ax2.set_ylabel("Throughput (samp/s)")
    ax2.set_xticks(n_values)
    ax2.legend(fontsize=8)

    # ---- (C) Communication latency distribution ----
    ax3 = fig.add_subplot(gs[1, 0])
    vals, lbls = [], []
    for arch, d in empirical.items():
        vals.append(d["iters"]["comm_latency_ms"].dropna().values)
        lbls.append(LABELS[arch])
    bplot = ax3.boxplot(vals, labels=lbls, patch_artist=True,
                        medianprops=dict(color="#1F2937", linewidth=2))
    for patch, arch in zip(bplot["boxes"], empirical.keys()):
        patch.set_facecolor(COLORS[arch])
        patch.set_alpha(0.6)
    ax3.set_title("(C) Comm Latency Distribution | n=4")
    ax3.set_ylabel("Latency (ms)")

    # ---- (D) Amdahl speedup curves ----
    ax4 = fig.add_subplot(gs[1, 1])
    n_fine = np.linspace(1, 32, 200)
    ax4.plot(n_fine, n_fine, "k--", linewidth=1, alpha=0.4, label="Linear (ideal)")
    for arch in ("ring_ar", "ps"):
        if arch not in f_serials:
            continue
        speedup = amdahl_speedup(n_fine, f_serials[arch])
        ax4.plot(n_fine, speedup, color=COLORS[arch], label=LABELS[arch], linewidth=2)
        # Mark f_serial on legend
    ax4.set_title(f"(D) Amdahl Speedup\n"
                  f"f_serial: Ring={f_serials.get('ring_ar',0):.2f} | "
                  f"PS={f_serials.get('ps',0):.2f}")
    ax4.set_xlabel("Number of Workers (n)")
    ax4.set_ylabel("Speedup S(n)")
    ax4.set_xticks(n_values)
    ax4.legend()

    fig.suptitle(
        "Experiment A — Throughput Scaling: PS-BSP vs Ring AllReduce\n"
        "Empirical n=4 + Amdahl's Law extrapolation to n=32",
        fontsize=12, fontweight="bold"
    )

    out_path = out_dir / "exp_a_scaling.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.show()

    # Summary
    print("\n=== Experiment A Summary ===")
    for arch, d in empirical.items():
        s = d["summary"]
        print(f"\n{LABELS[arch]} | n=4:")
        print(f"  Throughput     : {s['avg_throughput_samp_s']:.1f} samp/s")
        print(f"  Comm latency   : {s['avg_comm_latency_ms']:.1f} ms")
        print(f"  f_serial       : {f_serials[arch]:.3f}")
        print(f"  Predicted n=16 : {amdahl_speedup(16, f_serials[arch]):.2f}x speedup")
        print(f"  Predicted n=32 : {amdahl_speedup(32, f_serials[arch]):.2f}x speedup")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("./results/exp_a"))
    parser.add_argument("--out-dir",     type=Path, default=Path("./results/exp_a/plots"))
    args = parser.parse_args()
    plot(args.results_dir, args.out_dir)
