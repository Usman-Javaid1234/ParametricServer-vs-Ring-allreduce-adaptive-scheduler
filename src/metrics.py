"""
metrics.py — Per-iteration metrics collection.
Logs to CSV (always) + Prometheus push gateway (if available).
CS332 | Distributed SGD Project | Phase 1
"""

import os
import csv
import time
import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))


class MetricsCollector:
    """
    Records per-iteration and per-epoch metrics to CSV files.

    Files produced (under RESULTS_DIR):
      {run_id}_rank{rank}_iterations.csv   — one row per iteration
      {run_id}_rank{rank}_epochs.csv       — one row per epoch
      {run_id}_rank{rank}_summary.json     — final summary
    """

    ITER_FIELDS = [
        "timestamp", "epoch", "iteration",
        "loss", "throughput_samp_s",
        "comm_latency_ms", "iter_time_ms",
        "rank", "world_size", "arch",
    ]

    EPOCH_FIELDS = [
        "timestamp", "epoch",
        "train_loss", "val_loss", "val_acc",
        "rank", "world_size", "arch",
    ]

    def __init__(
        self,
        run_id: str,
        rank: int,
        arch: str,
        world_size: int,
        results_dir: Optional[Path] = None,
    ):
        self.run_id     = run_id
        self.rank       = rank
        self.arch       = arch
        self.world_size = world_size
        self.start_time = time.time()

        base = results_dir or RESULTS_DIR
        base.mkdir(parents=True, exist_ok=True)

        self._iter_path    = base / f"{run_id}_rank{rank}_iterations.csv"
        self._epoch_path   = base / f"{run_id}_rank{rank}_epochs.csv"
        self._summary_path = base / f"{run_id}_rank{rank}_summary.json"

        # Safety: never silently overwrite existing results.
        if self._iter_path.exists() or self._epoch_path.exists():
            import time as _time
            suffix = int(_time.time())
            log.warning(f"Results files already exist for run_id={run_id} "
                        f"rank={rank} — appending suffix _{suffix} to avoid overwrite.")
            self._iter_path    = base / f"{run_id}_rank{rank}_{suffix}_iterations.csv"
            self._epoch_path   = base / f"{run_id}_rank{rank}_{suffix}_epochs.csv"
            self._summary_path = base / f"{run_id}_rank{rank}_{suffix}_summary.json"

        self._iter_f  = open(self._iter_path,  "w", newline="")
        self._epoch_f = open(self._epoch_path, "w", newline="")

        self._iter_writer  = csv.DictWriter(self._iter_f,  fieldnames=self.ITER_FIELDS)
        self._epoch_writer = csv.DictWriter(self._epoch_f, fieldnames=self.EPOCH_FIELDS)
        self._iter_writer.writeheader()
        self._epoch_writer.writeheader()

        # Running totals for summary
        self._total_iters       = 0
        self._total_comm_ms     = 0.0
        self._total_throughput  = 0.0
        self._last_val_acc      = 0.0

        log.info(
            f"MetricsCollector init | run_id={run_id} rank={rank} "
            f"arch={arch} world_size={world_size}"
        )

    # -----------------------------------------------------------------------

    def record_iteration(
        self,
        epoch: int,
        iteration: int,
        loss: float,
        throughput: float,
        comm_latency_ms: float,
        iter_time_ms: float,
    ):
        row = {
            "timestamp":        time.time(),
            "epoch":            epoch,
            "iteration":        iteration,
            "loss":             round(loss, 6),
            "throughput_samp_s": round(throughput, 2),
            "comm_latency_ms":  round(comm_latency_ms, 3),
            "iter_time_ms":     round(iter_time_ms, 3),
            "rank":             self.rank,
            "world_size":       self.world_size,
            "arch":             self.arch,
        }
        self._iter_writer.writerow(row)
        self._iter_f.flush()

        self._total_iters      += 1
        self._total_comm_ms    += comm_latency_ms
        self._total_throughput += throughput

    def record_epoch_train(self, epoch: int, train_loss: float):
        """Called by ALL ranks at end of each epoch — records train loss only."""
        row = {
            "timestamp":   time.time(),
            "epoch":       epoch,
            "train_loss":  round(train_loss, 6),
            "val_loss":    "",   # only rank 0 fills these
            "val_acc":     "",
            "rank":        self.rank,
            "world_size":  self.world_size,
            "arch":        self.arch,
        }
        self._epoch_writer.writerow(row)
        self._epoch_f.flush()

    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_acc: float,
    ):
        row = {
            "timestamp":   time.time(),
            "epoch":       epoch,
            "train_loss":  round(train_loss, 6),
            "val_loss":    round(val_loss, 6),
            "val_acc":     round(val_acc, 4),
            "rank":        self.rank,
            "world_size":  self.world_size,
            "arch":        self.arch,
        }
        self._epoch_writer.writerow(row)
        self._epoch_f.flush()
        self._last_val_acc = val_acc

    def finalize(self):
        elapsed = time.time() - self.start_time
        summary = {
            "run_id":               self.run_id,
            "rank":                 self.rank,
            "arch":                 self.arch,
            "world_size":           self.world_size,
            "total_iterations":     self._total_iters,
            "avg_comm_latency_ms":  round(
                self._total_comm_ms / max(self._total_iters, 1), 3
            ),
            "avg_throughput_samp_s": round(
                self._total_throughput / max(self._total_iters, 1), 2
            ),
            "final_val_acc":        self._last_val_acc,
            "total_wall_time_s":    round(elapsed, 2),
        }
        with open(self._summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        self._iter_f.close()
        self._epoch_f.close()
        log.info(f"Metrics finalised → {self._summary_path}")
        log.info(json.dumps(summary, indent=2))