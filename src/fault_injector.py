"""
fault_injector.py — Straggler simulation + worker crash injection.
CS332 | Distributed SGD Project | Phase 2 & 3

Two classes:
  SlowdownInjector — adds per-iteration delay to simulate a straggler (Exp B)
  CrashInjector   — kills the worker process after N iterations (Exp C)
"""

import os
import sys
import time
import logging
import threading

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment B — Straggler delay injection
# ---------------------------------------------------------------------------

class SlowdownInjector:
    """
    Injects delay at the START of each iteration on the straggler worker,
    BEFORE sync_gradients() is called. All other workers wait at the
    all_reduce barrier, so the effect shows in comm_latency_ms for ALL ranks.

    Environment variables:
      STRAGGLER_RANK        — which worker rank to slow (default: none)
      STRAGGLER_DELAY_MS    — ms to sleep per iteration (default: 0)
      STRAGGLER_START_ITER  — iteration to start injection (default: 10)
    """

    def __init__(self, rank: int):
        self.rank       = rank
        self.enabled    = False
        self.delay_ms   = 0.0
        self.start_iter = 10
        self._fired     = 0

        straggler_rank = os.environ.get("STRAGGLER_RANK", "")
        delay_ms       = float(os.environ.get("STRAGGLER_DELAY_MS", "0"))
        start_iter     = int(os.environ.get("STRAGGLER_START_ITER", "10"))

        if straggler_rank and delay_ms > 0:
            if rank == int(straggler_rank):
                self.enabled    = True
                self.delay_ms   = delay_ms
                self.start_iter = start_iter
                log.warning(
                    f"[SlowdownInjector] Rank {rank} STRAGGLER | "
                    f"delay={delay_ms}ms | starts at iter {start_iter}"
                )

    def pre_iter_delay(self, iteration: int):
        """Call BEFORE forward pass so barrier wait is visible to all ranks."""
        if not self.enabled or iteration < self.start_iter:
            return
        time.sleep(self.delay_ms / 1000.0)
        self._fired += 1

    def maybe_delay(self, iteration: int):
        self.pre_iter_delay(iteration)

    def is_straggler(self) -> bool:
        return self.enabled

    def fired_count(self) -> int:
        return self._fired


# ---------------------------------------------------------------------------
# Experiment C — Worker crash injection
# ---------------------------------------------------------------------------

class CrashInjector:
    """
    Kills this worker process after CRASH_ITER iterations by calling sys.exit(1).
    Only fires on the rank specified by CRASH_RANK env var.

    This simulates a real worker crash — the process dies, heartbeat stops,
    orchestrator detects failure after FAILURE_TIMEOUT seconds, and the
    recovery path is triggered.

    Environment variables:
      CRASH_RANK  — which worker rank to kill (default: none)
      CRASH_ITER  — which iteration to crash at (default: 100)

    Recovery behaviour:
      PS-BSP  — server excludes worker, continues with n-1 workers
      Ring AR — ring is broken, training halts (demonstrates Ring AR weakness)
    """

    def __init__(self, rank: int):
        self.rank        = rank
        self.enabled     = False
        self.crash_iter  = 100
        self._crashed    = False
        self._crash_time = None

        crash_rank = os.environ.get("CRASH_RANK", "")
        crash_iter = int(os.environ.get("CRASH_ITER", "100"))

        if crash_rank and rank == int(crash_rank):
            self.enabled    = True
            self.crash_iter = crash_iter
            log.warning(
                f"[CrashInjector] Rank {rank} will CRASH at iteration {crash_iter}"
            )

    def maybe_crash(self, iteration: int):
        """
        Call this every iteration. On the target iteration, logs the crash
        event and exits the process immediately.
        """
        if not self.enabled or self._crashed:
            return
        if iteration >= self.crash_iter:
            self._crashed    = True
            self._crash_time = time.time()
            log.error(
                f"[CrashInjector] Rank {self.rank} CRASHING at iteration {iteration} "
                f"(simulated worker failure)"
            )
            # Write crash event to results for the plot script
            crash_log = os.environ.get(
                "RESULTS_DIR", "/results"
            ) + f"/crash_event_rank{self.rank}.json"
            try:
                import json
                with open(crash_log, "w") as f:
                    json.dump({
                        "rank":      self.rank,
                        "iteration": iteration,
                        "timestamp": self._crash_time,
                        "arch":      os.environ.get("BACKEND", "unknown"),
                        "run_id":    os.environ.get("RUN_ID", "unknown"),
                    }, f)
            except Exception:
                pass
            # Die — this is the crash
            sys.exit(1)

    def is_crash_enabled(self) -> bool:
        return self.enabled