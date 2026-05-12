"""
compressed_ring_ar.py — QSGD-Compressed Ring AllReduce Backend
CS332 | Distributed SGD Project | Phase 2 — Compression Layer

Wraps torch.distributed.all_reduce with QSGD quantization/dequantization.
Plugs into worker.py via the same sync_gradients(model) interface as
RingARBackend — zero changes required to the training loop.

Communication flow per iteration:

  [Each worker]
       │
       ▼
  flatten gradients → float32 vector (11MB for ResNet-18)
       │
       ▼  quantize(num_bits)
  compressed int32 vector + scale scalar
       │
       ▼  dist.all_reduce (SUM) on compressed ints
          dist.all_reduce (SUM) on scale (for averaging)
       │
       ▼  divide by world_size → averaged compressed gradients
       │
       ▼  dequantize(num_bits)
  reconstructed float32 vector
       │
       ▼
  write back into model.param.grad

Why two all_reduce calls (compressed + scale)?
  The scale (L2 norm) must also be averaged across workers so dequantization
  uses a consistent scale. It's a single float — negligible overhead.

Bytes sent per worker per iteration:
  Baseline (32-bit):  2 × (n-1)/n × |M| × 4 bytes
  Compressed (k-bit): 2 × (n-1)/n × (|M| × k/8 + 4) bytes
  Compression ratio:  ~32/k  (e.g. 4-bit → ~8× less data)

Note on all_reduce with int32:
  dist.all_reduce SUM on integer tensors is exact — no floating point
  error in the ring aggregation step. Quantization error only enters
  at the quantize() and dequantize() steps, which is the intended design.
"""

import time
import logging
import torch
import torch.distributed as dist
from typing import Optional

from qsgd import quantize, dequantize, compressed_bytes, compression_ratio

log = logging.getLogger(__name__)


class CompressedRingARBackend:
    """
    Drop-in replacement for RingARBackend in worker.py.

    Constructor args:
      rank        — this worker's rank
      world_size  — total number of workers
      num_bits    — quantization bit-width (2, 4, 8, or 32)
                    32 = no compression (identical to plain RingARBackend)

    Usage in worker.py:
      # Replace:
      sync_backend = RingARBackend(rank, world_size)
      # With:
      sync_backend = CompressedRingARBackend(rank, world_size, num_bits=args.num_bits)

    Everything else stays the same — same sync_gradients(model) call,
    same return value (comm_latency_ms), same MetricsCollector integration.
    """

    def __init__(self, rank: int, world_size: int, num_bits: int = 8):
        assert num_bits in (2, 4, 8, 32), \
            f"num_bits must be 2, 4, 8, or 32. Got {num_bits}."

        self.rank        = rank
        self.world_size  = world_size
        self.num_bits    = num_bits
        self._iteration  = 0

        # Lazy-computed model size (set on first sync call)
        self._total_numel: Optional[int] = None

        mode = "BASELINE (no compression)" if num_bits == 32 else f"QSGD {num_bits}-bit"
        log.info(
            f"CompressedRingARBackend | rank={rank} | "
            f"world_size={world_size} | mode={mode}"
        )

    # -----------------------------------------------------------------------
    # Gradient helpers — same as ManualRingAR in ring_ar.py
    # -----------------------------------------------------------------------

    def _flatten_gradients(self, model: torch.nn.Module) -> torch.Tensor:
        """Concatenate all parameter gradients into a single 1-D float32 tensor."""
        parts = []
        for p in model.parameters():
            if p.grad is not None:
                parts.append(p.grad.data.cpu().float().view(-1))
            else:
                parts.append(torch.zeros(p.data.numel(), dtype=torch.float32))
        flat = torch.cat(parts)
        if self._total_numel is None:
            self._total_numel = flat.numel()
        return flat

    def _unflatten_gradients(
        self, flat: torch.Tensor, model: torch.nn.Module
    ):
        """Write flat float32 gradient vector back into model.param.grad."""
        offset = 0
        for p in model.parameters():
            numel = p.data.numel()
            g = flat[offset: offset + numel].view_as(p.data).to(p.device)
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.data.copy_(g)
            offset += numel

    # -----------------------------------------------------------------------
    # Core sync — called every iteration from train_epoch()
    # -----------------------------------------------------------------------

    def sync_gradients(self, model: torch.nn.Module) -> float:
        """
        Compress → AllReduce → Decompress → write back.

        Returns:
          comm_latency_ms — wall-clock time of the AllReduce calls only,
                            matching the metric semantics of RingARBackend.
        """
        t_start = time.perf_counter()

        # ---- Step 1: Flatten ----
        flat = self._flatten_gradients(model)   # float32, CPU

        # ---- Step 2: Quantize ----
        t_compress = time.perf_counter()
        compressed, scale = quantize(flat, self.num_bits)
        # compressed: int32 tensor | scale: 1-element float32 tensor
        t_compress_ms = (time.perf_counter() - t_compress) * 1000.0

        # ---- Step 3: AllReduce ----
        # NCCL requires tensors on GPU. Gloo requires CPU.
        # We detect the backend and move tensors accordingly.
        # After all_reduce, move back to CPU for dequantize/unflatten.
        t_comm = time.perf_counter()

        backend = dist.get_backend()
        use_gpu = (backend == "nccl")
        device  = torch.device("cuda") if use_gpu else torch.device("cpu")

        if self.num_bits == 32:
            comm_tensor = compressed.to(device)
            scale_gpu   = scale.to(device)
            dist.all_reduce(comm_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(scale_gpu,   op=dist.ReduceOp.SUM)
            compressed  = comm_tensor.cpu()
            scale       = scale_gpu.cpu()
        else:
            # Cast int32 → float32 for transport (NCCL/Gloo don't support int32 reduce)
            # Integer values fit exactly in float32 (safe up to 2^24)
            compressed_float = compressed.float().to(device)
            scale_gpu        = scale.to(device)
            dist.all_reduce(compressed_float, op=dist.ReduceOp.SUM)
            dist.all_reduce(scale_gpu,        op=dist.ReduceOp.SUM)
            compressed = compressed_float.cpu().to(torch.int32)
            scale      = scale_gpu.cpu()

        comm_ms = (time.perf_counter() - t_comm) * 1000.0

        # ---- Step 4: Average (divide by world_size) ----
        if self.num_bits == 32:
            flat_result = compressed / self.world_size
        else:
            compressed_avg = (compressed.float() / self.world_size).round().to(torch.int32)
            scale_avg      = scale / self.world_size
            # ---- Step 5: Dequantize ----
            flat_result = dequantize(compressed_avg, scale_avg, self.num_bits)

        # ---- Step 6: Write back into model ----
        self._unflatten_gradients(flat_result, model)

        total_ms = (time.perf_counter() - t_start) * 1000.0

        # ---- Logging (every 100 iterations on rank 0) ----
        if self.rank == 0 and self._iteration % 100 == 0:
            numel = self._total_numel or flat.numel()
            orig_mb       = numel * 4 / 1e6
            comp_mb       = compressed_bytes(numel, self.num_bits) / 1e6
            ratio         = compression_ratio(numel, self.num_bits)
            log.info(
                f"[CompressedRingAR] iter={self._iteration} | "
                f"bits={self.num_bits} | "
                f"original={orig_mb:.1f}MB | "
                f"compressed={comp_mb:.1f}MB | "
                f"ratio={ratio:.1f}x | "
                f"comm={comm_ms:.1f}ms | "
                f"compress_overhead={t_compress_ms:.1f}ms | "
                f"total={total_ms:.1f}ms"
            )

        self._iteration += 1
        return comm_ms   # matches RingARBackend return type

    # -----------------------------------------------------------------------
    # Info helpers — for experiment grid logging
    # -----------------------------------------------------------------------

    def get_compression_info(self) -> dict:
        """Returns compression metadata for MetricsCollector / experiment logs."""
        numel = self._total_numel or 0
        return {
            "num_bits":          self.num_bits,
            "world_size":        self.world_size,
            "model_numel":       numel,
            "original_mb":       round(numel * 4 / 1e6, 2),
            "compressed_mb":     round(compressed_bytes(numel, self.num_bits) / 1e6, 2),
            "compression_ratio": round(compression_ratio(numel, self.num_bits), 2),
        }
