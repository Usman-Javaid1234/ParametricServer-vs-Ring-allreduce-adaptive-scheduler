"""
qsgd.py — QSGD Gradient Quantization (Alistarh et al., NeurIPS 2017)
CS332 | Distributed SGD Project | Phase 2 — Compression Layer

Implements unbiased stochastic quantization compatible with Ring AllReduce.

Key property (why this works with all_reduce):
  E[quantize(x)] = x   (unbiased)
  quantize(a) + quantize(b) ≈ quantize(a + b)  (linear under expectation)

  This means compressed tensors can be summed across workers in the ring
  without decompressing at each hop — the ring's all_reduce is valid on
  compressed values, and the result is an unbiased estimate of the true sum.

Usage:
  compressed, scale = quantize(tensor, num_bits)
  reconstructed     = dequantize(compressed, scale, num_bits)

num_bits controls the compression ratio:
  32-bit → no compression (baseline)
   8-bit → 4× compression
   4-bit → 8× compression
   2-bit → 16× compression

References:
  Alistarh et al. "QSGD: Communication-Efficient SGD via Gradient
  Quantization and Encoding." NeurIPS 2017. arxiv.org/abs/1610.02132

  Xin & Canini. "Global-QSGD: Allreduce-Compatible Quantization."
  EuroMLSys 2025. arxiv.org/abs/2305.18627
"""

import torch
import math
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core QSGD quantize / dequantize
# ---------------------------------------------------------------------------

def quantize(tensor: torch.Tensor, num_bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    QSGD stochastic quantization of a flat gradient tensor.

    Algorithm:
      1. Compute L2 norm (global scale) — preserves magnitude information.
      2. Normalize tensor to [-1, 1] range.
      3. Scale to [0, levels] where levels = 2^num_bits - 1.
      4. Stochastic rounding: floor(v) + Bernoulli(v - floor(v)).
         This makes quantization UNBIASED — E[result] = original value.
      5. Restore sign.

    Args:
      tensor:   1-D float32 gradient tensor (already flattened).
      num_bits: Quantization bit-width. 32 = bypass (no compression).

    Returns:
      (quantized, scale):
        quantized — int8/int16/int32 tensor (values in [0, levels]).
        scale     — scalar float32 tensor (the L2 norm, for dequantization).
    """
    if num_bits == 32:
        # Baseline: no compression — return tensor as-is with dummy scale
        return tensor.clone(), torch.ones(1, dtype=torch.float32)

    assert 1 <= num_bits <= 8, f"num_bits must be in [1, 8] or 32, got {num_bits}"

    levels = (2 ** num_bits) - 1   # e.g. 4-bit → 15 levels

    # Step 1: Global L2 norm (scale factor)
    norm = torch.norm(tensor.float())
    if norm == 0:
        # Zero gradient — return zero quantized tensor
        zeros = torch.zeros_like(tensor, dtype=torch.int32)
        return zeros, torch.zeros(1, dtype=torch.float32)

    # Step 2-3: Normalize and scale
    normalized = tensor.float() / norm          # values in [-1, 1]
    scaled     = normalized.abs() * levels      # values in [0, levels]

    # Step 4: Stochastic rounding (the key to unbiasedness)
    floor_vals = scaled.floor()
    residual   = scaled - floor_vals            # fractional part in [0, 1)
    random     = torch.rand_like(residual)      # uniform [0, 1)
    quantized  = floor_vals + (random < residual).float()  # Bernoulli rounding

    # Step 5: Restore sign — store as signed integer
    signs      = tensor.sign()                  # -1, 0, or +1
    quantized  = quantized * signs              # signed quantized values

    # Cast to int32 for storage (int8 would overflow with sign for levels > 127)
    quantized_int = quantized.to(torch.int32)

    return quantized_int, norm.unsqueeze(0)


def dequantize(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    num_bits: int,
) -> torch.Tensor:
    """
    Reconstruct float32 gradient from QSGD quantized tensor.

    Args:
      quantized: Integer tensor from quantize() — values in [-levels, levels].
      scale:     L2 norm scalar from quantize().
      num_bits:  Must match the num_bits used in quantize().

    Returns:
      Reconstructed float32 gradient tensor (same shape as original input
      to quantize()).
    """
    if num_bits == 32:
        # Baseline: tensor was not compressed, return as float32
        return quantized.float()

    levels = (2 ** num_bits) - 1

    # Reverse: divide by levels to get normalized, multiply by norm
    reconstructed = quantized.float() / levels * scale.item()
    return reconstructed


# ---------------------------------------------------------------------------
# Convenience: bytes accounting
# ---------------------------------------------------------------------------

def compressed_bytes(num_elements: int, num_bits: int) -> int:
    """
    Theoretical bytes for a compressed gradient vector.

    For num_bits=32 (baseline):  num_elements × 4 bytes (float32)
    For num_bits=N:              ceil(num_elements × N / 8) bytes
                                 + 4 bytes for the scale (norm)

    Used in workload modeling / bandwidth plots.
    """
    if num_bits == 32:
        return num_elements * 4
    payload_bits = num_elements * num_bits
    payload_bytes = math.ceil(payload_bits / 8)
    scale_bytes = 4   # one float32 norm
    return payload_bytes + scale_bytes


def compression_ratio(num_elements: int, num_bits: int) -> float:
    """Ratio of original size to compressed size. Higher = more compression."""
    original = num_elements * 4   # float32 baseline
    compressed = compressed_bytes(num_elements, num_bits)
    return original / compressed


# ---------------------------------------------------------------------------
# Sanity check (run directly: python qsgd.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("QSGD Sanity Check")
    print("=" * 60)

    torch.manual_seed(42)
    # Simulate a ResNet-18 gradient vector (~11M floats)
    n = 11_173_962
    grad = torch.randn(n)

    for bits in [2, 4, 8, 32]:
        q, scale = quantize(grad, bits)
        recon    = dequantize(q, scale, bits)

        # Relative error
        err = (grad - recon).norm() / grad.norm()

        # Compression ratio
        ratio = compression_ratio(n, bits)

        print(
            f"  {bits:2d}-bit | "
            f"compression={ratio:.1f}x | "
            f"relative_error={err:.4f} | "
            f"compressed_MB={compressed_bytes(n, bits)/1e6:.2f}"
        )

    print()
    print("Unbiasedness check (2-bit, small tensor, 10000 trials):")
    torch.manual_seed(0)
    x = torch.randn(100)
    reconstructions = []
    for _ in range(10_000):
        q, s = quantize(x, 2)
        reconstructions.append(dequantize(q, s, 2))
    mean_recon = torch.stack(reconstructions).mean(dim=0)
    bias = (mean_recon - x).abs().mean().item()
    print(f"  Mean absolute bias: {bias:.6f}  (should be < 0.01 for unbiased)")
    print("=" * 60)
