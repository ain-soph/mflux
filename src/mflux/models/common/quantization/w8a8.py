"""W8A8 linear layers for ComfyUI "int8_convrot" checkpoints, on the Metal TensorOps INT8 matmul (Apple M5).

A convrot checkpoint stores each projection rotated by a block-diagonal Hadamard matrix (256-wide groups along
the input dimension) and quantized to int8 with one float32 scale per output row. At run time the activations
get the same rotation and a per-row int8 quantization, so the rotations cancel in the product:
y = (q(x H) q(W H)^T) * x_scale * w_scale ~ x W^T. The rotation spreads outliers across each group, which is
what keeps int8 activations accurate.

The kernels (fused rotate-quantize; mlx-serve's INT8 TensorOps GEMM, MIT) are in MLX: mx.fast.convrot_* of the
ain-soph/mlx `serving` branch.
"""

import mlx.core as mx
import numpy as np
from mlx import nn

GROUP = 256

_H4 = np.ones((4, 4), dtype=np.float32)
np.fill_diagonal(_H4[:, ::-1], -1)
H256 = mx.array(np.kron(np.kron(_H4, _H4), np.kron(_H4, _H4)) / 16)


@mx.compile
def rotate_quantize_reference(x: mx.array) -> tuple[mx.array, mx.array]:
    rotated = (x.astype(mx.float32).reshape(-1, GROUP) @ H256).reshape(x.shape)
    scale = mx.maximum(mx.max(mx.abs(rotated), axis=-1) / 127, 1e-20)
    return mx.clip(mx.round(rotated / scale[:, None]), -127, 127).astype(mx.int8), scale


def rotate_quantize(x: mx.array) -> tuple[mx.array, mx.array]:
    """Per-row int8 quantization of x H256 (block-diagonal along the last axis) with float32 scales, in one MLX
    kernel (mx.fast.convrot_quantize, ain-soph/mlx serving). The int8 rows are allocated up to a multiple of 128 (the
    largest GEMM tile height); the extra rows only feed output rows that are never stored."""
    return mx.fast.convrot_quantize(x)


def matmul(x: mx.array, weight: mx.array, scale: mx.array) -> mx.array:
    """x (rows, in) in any float dtype @ an int8 convrot weight (out, in) with float32 scales (out,).

    The rotate-quantize and INT8 TensorOps GEMM kernels live in MLX (mx.fast.convrot_*, ain-soph/mlx serving)."""
    rows, n = x.shape[0], weight.shape[0]
    bn = 8192 // (128 if rows < 8192 else 64)
    if n < bn:
        # narrower than one column tile, which the kernel cannot place inside the weight: float product
        rotated = (x.astype(mx.float32).reshape(-1, GROUP) @ H256).reshape(x.shape)
        return (rotated @ (weight.astype(mx.float32) * scale[:, None]).T).astype(x.dtype)
    xq, xs = rotate_quantize(x)
    return mx.fast.convrot_matmul_quantized(xq, xs, weight, scale, x.dtype)


class W8A8Linear(nn.Module):
    """Drop-in for a bias-free nn.Linear holding an int8 convrot weight and its per-row scales."""

    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()
        if input_dims % GROUP:
            raise ValueError(f"W8A8 needs the input width to be a multiple of {GROUP}, got {input_dims}.")
        self.weight = mx.zeros((output_dims, input_dims), dtype=mx.int8)
        self.scale = mx.zeros((output_dims,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        y = matmul(x.reshape(-1, shape[-1]), self.weight, self.scale)
        return y.reshape(*shape[:-1], self.weight.shape[0])

    @staticmethod
    def fused(x: mx.array, layers: tuple["W8A8Linear", ...]) -> list[mx.array]:
        """Several projections of the same input with one activation quantization and one GEMM. The weights are
        packed as a view when they lie back to back in a zero-copy file (QKV, gate/up), else concatenated."""
        from mflux.models.common.weights.loading import zero_copy

        weights = [layer.weight for layer in layers]
        packed = zero_copy.adjacent_view(weights, cache=False)
        if packed is None:
            packed = mx.concatenate(weights, axis=0)
        scale = mx.concatenate([layer.scale for layer in layers])
        shape = x.shape
        y = matmul(x.reshape(-1, shape[-1]), packed, scale).reshape(*shape[:-1], packed.shape[0])
        sizes = [w.shape[0] for w in weights]
        return mx.split(y, [sum(sizes[: i + 1]) for i in range(len(sizes) - 1)], axis=-1)

    @staticmethod
    def replace(module: nn.Module, name: str) -> "W8A8Linear":
        linear = getattr(module, name) if not name.isdigit() else module[int(name)]
        w8a8 = W8A8Linear(linear.weight.shape[1], linear.weight.shape[0])
        if name.isdigit():
            module[int(name)] = w8a8
        else:
            setattr(module, name, w8a8)
        return w8a8


class ConvRotEmbedding(nn.Module):
    """Embedding table stored as int8 convrot rows (rotated along the hidden size, one float32 scale per row).
    A lookup dequantizes the selected rows and undoes the rotation (H256 is symmetric and orthonormal)."""

    def __init__(self, num_embeddings: int, dims: int):
        super().__init__()
        self.weight = mx.zeros((num_embeddings, dims), dtype=mx.int8)
        self.scale = mx.zeros((num_embeddings,), dtype=mx.float32)

    def __call__(self, ids: mx.array) -> mx.array:
        rows = self.weight[ids].astype(mx.float32) * self.scale[ids][..., None]
        shape = rows.shape
        return (rows.reshape(-1, GROUP) @ H256).reshape(shape).astype(mx.bfloat16)
