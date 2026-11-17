"""Switches for the elementwise fusions of the Qwen-Image-2.1 DiT block (forward_reference path).

QWEN21_FUSION is a comma-separated subset of {select, qknorm, residual}; unset = all on, empty = none (the
composed ops, for A/B tests):
- select:   in the cached denoise pass every token is a target token, so the per-token modulation rows are one
            broadcast row instead of four full-sequence tensors built with mx.where (same values).
- qknorm:   Q/K RMSNorm + RoPE + head-major layout in the existing fused_qk_norm_rope kernel instead of
            reshape/transpose, RMSNorm, and the float32 RoPE with stack/reshape.
- residual: x + tanh(gate) * y as one compiled kernel.
"""

import os

import mlx.core as mx

ALL = ("select", "qknorm", "residual")


def enabled(name: str) -> bool:
    value = os.environ.get("QWEN21_FUSION")
    if value is None:
        return True
    return name in {part.strip() for part in value.split(",")}


@mx.compile
def gated_residual(x: mx.array, gate: mx.array, y: mx.array) -> mx.array:
    return x + mx.tanh(gate) * y
