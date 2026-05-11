"""
fault_injector.py — Straggler simulation for Experiment B.
CS332 | Distributed SGD Project | Phase 2

Injects delay BEFORE the gradient sync (not after) so it directly
inflates the all_reduce wait time visible to all other workers.
This makes the effect clearly measurable in comm_latency_ms.
"""

import os
import time
import logging

log = logging.getLogger(__name__)


class SlowdownInjector:
    """
    Injects delay at the START of each iteration on the straggler worker,
    BEFORE sync_gradients() is called. This way all other workers are forced
    to wait at the all_reduce barrier, making the effect show up in
    comm_latency_ms for ALL ranks — not just iter_time_ms of rank 3.

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
                    f"[FaultInjector] Rank {rank} STRAGGLER | "
                    f"delay={delay_ms}ms | starts at iter {start_iter}"
                )

    def pre_iter_delay(self, iteration: int):
        """
        Call this BEFORE forward pass and sync_gradients.
        Sleeping here forces other workers to wait at the all_reduce barrier,
        making the straggler effect visible in comm_latency_ms for all ranks.
        """
        if not self.enabled or iteration < self.start_iter:
            return
        time.sleep(self.delay_ms / 1000.0)
        self._fired += 1

    # Keep maybe_delay as alias for backward compat
    def maybe_delay(self, iteration: int):
        self.pre_iter_delay(iteration)

    def is_straggler(self) -> bool:
        return self.enabled

    def fired_count(self) -> int:
        return self._fired