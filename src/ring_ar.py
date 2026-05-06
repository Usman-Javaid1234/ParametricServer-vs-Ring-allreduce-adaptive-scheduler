"""
ring_ar.py — Ring AllReduce: Reduce-Scatter + AllGather phases.
CS332 | Distributed SGD Project | Phase 1

Two modes:
  1. native   — wraps torch.distributed.all_reduce (NCCL/Gloo backend).
               This is what Phase 1 uses.
  2. manual   — explicit chunk-level send/recv ring implementation.
               Used in Phase 2 for per-chunk latency profiling and
               validating the 2(n-1)/n × |M| bandwidth formula.

The worker.py calls backend.sync_gradients(model) — same interface as ps_server.
"""

import time
import math
import logging
import torch
import torch.distributed as dist
from typing import List

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Native Ring AR (wraps dist.all_reduce — uses NCCL Ring AR under the hood)
# ---------------------------------------------------------------------------

class NativeRingAR:
    """
    Uses torch.distributed.all_reduce which internally implements Ring AR
    when the NCCL or Gloo backend is active.
    Simplest and fastest for Phase 1.
    """

    def __init__(self, rank: int, world_size: int):
        self.rank       = rank
        self.world_size = world_size

    def sync_gradients(self, model: torch.nn.Module) -> float:
        t0 = time.perf_counter()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size
        return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Manual Ring AR (explicit chunk send/recv — for profiling and validation)
# ---------------------------------------------------------------------------

class ManualRingAR:
    """
    Explicit Ring AllReduce implementation following the two-phase protocol:

    Phase 1 — Reduce-Scatter (n-1 steps):
      Each worker sends chunk[send_idx] to right neighbor,
      receives chunk[recv_idx] from left neighbor, accumulates.

    Phase 2 — AllGather (n-1 steps):
      Each worker sends its fully-reduced chunk to right neighbor,
      receives a chunk from left neighbor (copy, not accumulate).

    Data sent per GPU = 2 × (n-1)/n × |M|  ← bandwidth-optimal
    """

    def __init__(self, rank: int, world_size: int):
        self.rank        = rank
        self.world_size  = world_size
        self.right       = (rank + 1) % world_size
        self.left        = (rank - 1) % world_size

    # -----------------------------------------------------------------------

    def _flatten_gradients(self, model: torch.nn.Module) -> torch.Tensor:
        """Concatenate all parameter gradients into a single 1-D tensor."""
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.view(-1))
            else:
                grads.append(torch.zeros_like(param.data).view(-1))
        return torch.cat(grads)

    def _unflatten_gradients(
        self, flat: torch.Tensor, model: torch.nn.Module
    ):
        """Write flat gradient vector back into model.param.grad."""
        offset = 0
        for param in model.parameters():
            numel = param.data.numel()
            if param.grad is not None:
                param.grad.data.copy_(flat[offset: offset + numel].view_as(param.data))
            offset += numel

    # -----------------------------------------------------------------------

    def sync_gradients(self, model: torch.nn.Module) -> float:
        t0 = time.perf_counter()

        n = self.world_size
        flat = self._flatten_gradients(model)

        # Pad to be divisible by n
        remainder = flat.numel() % n
        if remainder:
            pad_size = n - remainder
            flat = torch.cat([flat, flat.new_zeros(pad_size)])
        chunk_size = flat.numel() // n

        # Split into n chunks
        chunks: List[torch.Tensor] = list(flat.split(chunk_size))

        # ---- Phase 1: Reduce-Scatter ----
        for step in range(n - 1):
            send_idx = (self.rank - step) % n
            recv_idx = (self.rank - step - 1) % n

            send_buf = chunks[send_idx].clone()
            recv_buf = torch.zeros_like(chunks[recv_idx])

            # Non-blocking send / blocking recv to avoid deadlock
            send_req = dist.isend(send_buf, dst=self.right)
            dist.recv(recv_buf, src=self.left)
            send_req.wait()

            chunks[recv_idx] += recv_buf          # accumulate

        # ---- Phase 2: AllGather ----
        for step in range(n - 1):
            send_idx = (self.rank - step + 1) % n
            recv_idx = (self.rank - step) % n

            send_buf = chunks[send_idx].clone()
            recv_buf = torch.zeros_like(chunks[recv_idx])

            send_req = dist.isend(send_buf, dst=self.right)
            dist.recv(recv_buf, src=self.left)
            send_req.wait()

            chunks[recv_idx].copy_(recv_buf)      # overwrite (not accumulate)

        # Reassemble and divide by n (average)
        flat_result = torch.cat(chunks)
        flat_result /= n

        # Strip padding and write back
        original_numel = sum(
            p.data.numel() for p in model.parameters() if p.grad is not None
        )
        flat_result = flat_result[:original_numel]
        self._unflatten_gradients(flat_result, model)

        comm_ms = (time.perf_counter() - t0) * 1000.0

        # Theoretical bandwidth check (logged at debug level)
        model_bytes = flat.numel() * 4  # float32
        theoretical_bytes = 2 * (n - 1) / n * model_bytes
        log.debug(
            f"ManualRingAR | rank={self.rank} | "
            f"model={model_bytes/1e6:.1f}MB | "
            f"theoretical_sent={theoretical_bytes/1e6:.1f}MB | "
            f"comm={comm_ms:.1f}ms"
        )

        return comm_ms

    # -----------------------------------------------------------------------

    def bandwidth_formula(self, model_size_bytes: int) -> dict:
        """
        Returns theoretical bandwidth metrics for this configuration.
        Used in performance modeling (Section 5 of design doc).
        """
        n = self.world_size
        data_per_gpu = 2 * (n - 1) / n * model_size_bytes
        return {
            "workers":              n,
            "model_size_bytes":     model_size_bytes,
            "data_per_gpu_bytes":   data_per_gpu,
            "data_per_gpu_MB":      data_per_gpu / 1e6,
            "phases":               2,
            "steps_per_phase":      n - 1,
            "total_steps":          2 * (n - 1),
        }
