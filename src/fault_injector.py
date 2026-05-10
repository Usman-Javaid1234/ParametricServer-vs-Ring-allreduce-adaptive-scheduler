"""
fault_injector.py — Straggler simulation for Experiment B.
CS332 | Distributed SGD Project | Phase 2

Two injection modes:
  1. delay   — adds artificial compute delay inside the training loop
               (no tc-netem needed, works on any OS including Windows/WSL2)
  2. network — uses tc-netem to throttle the network interface
               (Linux only, requires NET_ADMIN capability)

Workers import SlowdownInjector and call maybe_delay() each iteration.
The orchestrator reads STRAGGLER_RANK and STRAGGLER_DELAY_MS env vars
and signals the right worker.
"""

import os
import time
import logging
import threading

log = logging.getLogger(__name__)


class SlowdownInjector:
    """
    Injected into the training loop of each worker.
    If this worker's rank matches STRAGGLER_RANK, it sleeps
    STRAGGLER_DELAY_MS milliseconds every iteration after
    STRAGGLER_START_ITER iterations have passed.

    Environment variables:
      STRAGGLER_RANK        — which worker rank to slow down (default: none)
      STRAGGLER_DELAY_MS    — how many ms to sleep per iteration (default: 0)
      STRAGGLER_START_ITER  — which iteration to start injection (default: 50)
      STRAGGLER_N_WORKERS   — how many workers to slow (1 or 2, default: 1)
    """

    def __init__(self, rank: int):
        self.rank        = rank
        self.enabled     = False
        self.delay_ms    = 0.0
        self.start_iter  = 50
        self._iteration  = 0
        self._lock       = threading.Lock()

        # Read config from env
        straggler_rank = os.environ.get("STRAGGLER_RANK", "")
        delay_ms       = float(os.environ.get("STRAGGLER_DELAY_MS", "0"))
        start_iter     = int(os.environ.get("STRAGGLER_START_ITER", "50"))
        n_workers      = int(os.environ.get("STRAGGLER_N_WORKERS", "1"))

        if straggler_rank and delay_ms > 0:
            # Support slowing 1 or 2 workers
            # STRAGGLER_RANK=3 → slow rank 3
            # STRAGGLER_N_WORKERS=2 → slow ranks 3 and 2
            base_rank = int(straggler_rank)
            slow_ranks = [base_rank]
            if n_workers == 2:
                slow_ranks.append((base_rank - 1) % 4)

            if rank in slow_ranks:
                self.enabled    = True
                self.delay_ms   = delay_ms
                self.start_iter = start_iter
                log.warning(
                    f"[FaultInjector] Rank {rank} will be slowed by "
                    f"{delay_ms}ms/iter starting at iter {start_iter}"
                )

    def maybe_delay(self, iteration: int):
        """Call this at the end of each training iteration."""
        if not self.enabled:
            return
        if iteration < self.start_iter:
            return
        time.sleep(self.delay_ms / 1000.0)

    def is_straggler(self) -> bool:
        return self.enabled