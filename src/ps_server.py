"""
ps_server.py — Parameter Server with BSP and SSP consistency modes.
CS332 | Distributed SGD Project | Phase 1

Architecture:
  - One server process (rank=0 in the 'ps' process group OR standalone RPC server)
  - Workers push gradients via all_reduce (Phase 1 emulation)
  - SSP mode: tracks per-worker iteration counters, enforces staleness bound τ
  - Exposes health endpoint on TCP port for orchestrator heartbeat checks

Phase 1 note:
  Full torch.distributed.rpc push/pull is in Phase 2.
  Here we use dist.all_reduce as an emulated PS to get the baseline
  working immediately with minimal infra. The interface is identical —
  workers call backend.sync_gradients() — so swapping to true RPC
  requires no changes to worker.py.
"""

import os
import time
import threading
import logging
import socket
import json
import torch
import torch.distributed as dist
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Staleness tracker (used by SSP mode)
# ---------------------------------------------------------------------------

class StalenessTracker:
    """
    Tracks the current iteration number for each worker and enforces
    the SSP staleness bound τ.

    A worker at iteration t is blocked if:
        t - min_iteration_of_all_workers > τ
    """

    def __init__(self, world_size: int, tau: int):
        self.world_size = world_size
        self.tau        = tau
        self._lock      = threading.Lock()
        # iteration counters: worker_id → iteration number
        self._iterations: Dict[int, int] = {i: 0 for i in range(world_size)}
        self._failed_workers: set = set()

    def update(self, worker_id: int, iteration: int):
        with self._lock:
            self._iterations[worker_id] = iteration

    def mark_failed(self, worker_id: int):
        with self._lock:
            self._failed_workers.add(worker_id)
            del self._iterations[worker_id]
        log.warning(f"Worker {worker_id} marked as failed; excluded from sync.")

    def should_block(self, worker_id: int) -> bool:
        """Returns True if this worker is too far ahead and should block."""
        if self.tau == 0:
            return False        # BSP: never block individually — barrier handles it
        with self._lock:
            if not self._iterations:
                return False
            min_iter = min(self._iterations.values())
            my_iter  = self._iterations.get(worker_id, 0)
            return (my_iter - min_iter) > self.tau

    def wait_if_needed(self, worker_id: int, poll_interval: float = 0.01):
        """Spin-wait until this worker is within τ of the slowest worker."""
        while self.should_block(worker_id):
            time.sleep(poll_interval)

    def get_staleness(self, worker_id: int) -> int:
        with self._lock:
            if not self._iterations:
                return 0
            min_iter = min(self._iterations.values())
            my_iter  = self._iterations.get(worker_id, 0)
            return max(0, my_iter - min_iter)

    def status(self) -> dict:
        with self._lock:
            return {
                "iterations":    dict(self._iterations),
                "failed":        list(self._failed_workers),
                "min_iteration": min(self._iterations.values()) if self._iterations else 0,
                "max_staleness": max(
                    v - min(self._iterations.values())
                    for v in self._iterations.values()
                ) if len(self._iterations) > 1 else 0,
            }


# ---------------------------------------------------------------------------
# PS Server health endpoint
# ---------------------------------------------------------------------------

class HealthServer(threading.Thread):
    """
    Tiny TCP health server on port 29501.
    Orchestrator connects and reads JSON status.
    """

    def __init__(self, tracker: StalenessTracker, port: int = 29501):
        super().__init__(daemon=True)
        self.tracker = tracker
        self.port    = port

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", self.port))
            s.listen(5)
            log.info(f"PS health server listening on :{self.port}")
            while True:
                try:
                    conn, _ = s.accept()
                    with conn:
                        status = self.tracker.status()
                        conn.sendall(json.dumps(status).encode() + b"\n")
                except Exception as e:
                    log.debug(f"Health server error: {e}")


# ---------------------------------------------------------------------------
# PS Server main class
# ---------------------------------------------------------------------------

class ParameterServer:
    """
    Parameter Server coordinating gradient aggregation.

    Phase 1: uses dist.all_reduce as aggregation primitive.
    Phase 2: will use torch.distributed.rpc for true push/pull.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        tau: int = 0,          # 0 = BSP
        dist_backend: str = "gloo",
    ):
        self.rank        = rank
        self.world_size  = world_size
        self.tau         = tau
        self.mode        = "BSP" if tau == 0 else f"SSP(τ={tau})"
        self.tracker     = StalenessTracker(world_size, tau)
        self._iteration  = 0

        log.info(f"ParameterServer init | rank={rank} world_size={world_size} mode={self.mode}")

        # Start health endpoint on server rank (rank 0 in PS topology)
        if rank == 0:
            HealthServer(self.tracker).start()

    # -----------------------------------------------------------------------

    def sync_gradients(self, model: torch.nn.Module, worker_id: int) -> float:
        """
        Synchronize gradients for one training iteration.
        Returns communication latency in milliseconds.
        """
        # SSP: block if this worker is too far ahead
        if self.tau > 0:
            self.tracker.wait_if_needed(worker_id)

        t0 = time.perf_counter()

        # ---- Gradient aggregation (all_reduce = PS fan-in + fan-out emulation) ----
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

        # BSP: global barrier — every worker must finish before anyone proceeds
        if self.tau == 0:
            dist.barrier()

        comm_ms = (time.perf_counter() - t0) * 1000.0

        # Update staleness tracker
        self._iteration += 1
        self.tracker.update(worker_id, self._iteration)

        return comm_ms

    def remove_worker(self, worker_id: int):
        """Called by orchestrator when a worker failure is confirmed."""
        self.tracker.mark_failed(worker_id)
        self.world_size -= 1
        log.warning(
            f"Worker {worker_id} removed. Active workers: {self.world_size}"
        )

    def get_staleness(self, worker_id: int) -> int:
        return self.tracker.get_staleness(worker_id)

    def status(self) -> dict:
        return {
            "mode":       self.mode,
            "iteration":  self._iteration,
            "tracker":    self.tracker.status(),
        }
