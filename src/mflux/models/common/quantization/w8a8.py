"""W8A8 linear layers for ComfyUI "int8_convrot" checkpoints, on the Metal TensorOps INT8 matmul (Apple M5).

A convrot checkpoint stores each projection rotated by a block-diagonal Hadamard matrix (256-wide groups along
the input dimension) and quantized to int8 with one float32 scale per output row. At run time the activations
get the same rotation and a per-row int8 quantization, so the rotations cancel in the product:
y = (q(x H) q(W H)^T) * x_scale * w_scale ~ x W^T. The rotation spreads outliers across each group, which is
what keeps int8 activations accurate.

The kernels (fused rotate-quantize; mlx-serve's INT8 TensorOps GEMM, David Dalcu, MIT) are in MLX: mx.fast.convrot_*
of the ain-soph/mlx `serving` branch, as used by h3-image-mlx. GPUs without TensorOps (before Apple M5) run the same
layers through a float fallback: the activations are rotated and multiplied by the dequantized int8 weights, which
gives the same result up to the activation rounding.
"""

import re
from functools import cache

import mlx.core as mx
import numpy as np
from mlx import nn

GROUP = 256

_H4 = np.ones((4, 4), dtype=np.float32)
np.fill_diagonal(_H4[:, ::-1], -1)
H256 = mx.array(np.kron(np.kron(_H4, _H4), np.kron(_H4, _H4)) / 16)


@cache
def tensor_ops_available() -> bool:
    """The INT8 TensorOps GEMM needs an Apple M5 (or later) GPU."""
    name = str(mx.device_info().get("device_name", ""))
    return bool(re.search(r"\bApple M([5-9]|\d{2})\b", name))


def rotate(x: mx.array) -> mx.array:
    """x H256 (block-diagonal along the last axis) in float32."""
    return (x.astype(mx.float32).reshape(-1, GROUP) @ H256).reshape(x.shape)


def rotate_quantize_reference(x: mx.array) -> tuple[mx.array, mx.array]:
    rotated = rotate(x)
    scale = mx.maximum(mx.max(mx.abs(rotated), axis=-1) / 127, 1e-20)
    return mx.clip(mx.round(rotated / scale[:, None]), -127, 127).astype(mx.int8), scale


def rotate_quantize(x: mx.array, pad_rows: int = 1) -> tuple[mx.array, mx.array]:
    """Per-row int8 quantization of x H256 (block-diagonal along the last axis) with float32 scales, in one MLX
    kernel (mx.fast.convrot_quantize: one threadgroup per row, Hadamard-256 as base-4 butterflies).

    MLX allocates the int8 rows up to a multiple of 128 (the largest GEMM tile height); this returns them up to a
    multiple of pad_rows. The extra rows only feed output rows that are never stored."""
    rows = x.shape[0]
    xq, xs = mx.fast.convrot_quantize(x)
    return xq[: -(-rows // pad_rows) * pad_rows], xs


def matmul(x: mx.array, weight: mx.array, scale: mx.array) -> mx.array:
    """x (rows, in) in any float dtype @ an int8 convrot weight (out, in) with float32 scales (out,)."""
    if x.shape[1] != weight.shape[1]:
        # The kernel takes its K from x, so another width would read the weight rows with the wrong stride.
        raise ValueError(f"W8A8 input width {x.shape[1]} does not match the weight {tuple(weight.shape)}.")
    rows, n = x.shape[0], weight.shape[0]
    bn = 8192 // (128 if rows < 8192 else 64)
    if not tensor_ops_available() or n < bn:
        # Rotated activations times the dequantized weight: (x H)(q(W H) s)^T ~ x W^T. Also used for outputs
        # narrower than one column tile, which the kernel cannot place inside the weight.
        w = weight.astype(mx.float32) * scale[:, None]
        return (rotate(x) @ w.T).astype(x.dtype)
    xq, xs = mx.fast.convrot_quantize(x)  # rows padded to a multiple of 128, which every tile height divides
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
    def fused(x: mx.array, packed: tuple[mx.array, mx.array, list[int]]) -> list[mx.array]:
        """Several projections of the same input (packed by pack()) with one activation quantization and one GEMM."""
        weight, scale, sizes = packed
        shape = x.shape
        y = matmul(x.reshape(-1, shape[-1]), weight, scale).reshape(*shape[:-1], weight.shape[0])
        return mx.split(y, [sum(sizes[: i + 1]) for i in range(len(sizes) - 1)], axis=-1)

    @staticmethod
    def fused_layers(x: mx.array, layers: tuple["W8A8Linear", ...]) -> list[mx.array]:
        """As fused() for layers that were not packed at load (zero-copy weights): the weights are packed as a view
        when they lie back to back in the mapped file (QKV, the gate_up halves), else concatenated for this call, so
        a streamed block keeps no copy of its weights."""
        from mflux.models.common.weights.loading import zero_copy

        weights = [layer.weight for layer in layers]
        packed = zero_copy.adjacent_view(weights, cache=False)
        if packed is None:
            packed = mx.concatenate(weights, axis=0)
        scale = mx.concatenate([layer.scale for layer in layers])
        return W8A8Linear.fused(x, (packed, scale, [w.shape[0] for w in weights]))

    @staticmethod
    def pack(layers: tuple["W8A8Linear", ...]) -> tuple[mx.array, mx.array, list[int]]:
        """Concatenate the rows of layers that share an input, then point each layer at its slice of the result."""
        weight = mx.concatenate([layer.weight for layer in layers], axis=0)
        scale = mx.concatenate([layer.scale for layer in layers])
        mx.eval(weight, scale)
        sizes, start = [], 0
        for layer in layers:
            rows = layer.weight.shape[0]
            layer.weight, layer.scale = weight[start : start + rows], scale[start : start + rows]
            sizes.append(rows)
            start += rows
        return weight, scale, sizes

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
