"""Qwen-Image-2.1 text encoder with 8-bit weights and bfloat16 activations (W8A16).

MFLUX_QWEN21_TE_Q8=<dir or file> loads the Qwen3-VL text encoder from a file written by this module: the 36
language layers' projections (q/k/v/o, gate/up/down) and the generation head (lm_head) are MLX 8-bit
`QuantizedLinear` layers (group 32), everything else (token embedding, norms, vision tower) stays bfloat16. The
activations are not quantized, so the error is the weights' rounding alone, unlike the int8_convrot text encoder
(MFLUX_QWEN21_TE_W8A8), whose per-row int8 activations dominate its error.

The weights are symmetric like llama.cpp Q8_0: blocks of 32 along the input dimension, d = max|w| / 127 rounded
*up* to bfloat16, q = round(w / d) in [-127, 127]. MLX's affine 8-bit format stores u = q + 128 with scale = d and
bias = -128 * d, which reproduces every q * d exactly; scales and biases are bfloat16 so that bfloat16 activations
take MLX's fast quantized_matmul path (float16 scales with bfloat16 activations are slow). MLX's own default 8-bit
(min/max affine) is avoided: its per-group offset error gave 5-9x the Q8_0 error on Qwen3-Reranker-8B.

The file is an ordinary page-aligned safetensors checkpoint, so it loads zero-copy and streams layer by layer like
the bfloat16 one. Keys keep the checkpoint's names (model.language_model..., model.visual..., lm_head) with
.scales / .biases added next to each quantized .weight.

  python -m mflux.models.qwen21.weights.qwen21_q8 SRC_TEXT_ENCODER_DIR OUT_DIR
"""

import json
import os
import struct
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.weights.loading import zero_copy

GROUP = 32
TEXT_PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def checkpoint() -> Path | None:
    path = os.environ.get("MFLUX_QWEN21_TE_Q8")
    return Path(path) if path else None


def _quantized(key: str) -> bool:
    if key == "lm_head.weight":
        return True
    return key.startswith("model.language_model.layers.") and key.endswith(
        tuple("." + p + ".weight" for p in TEXT_PROJECTIONS))


def _bf16_up(d: np.ndarray) -> np.ndarray:
    """float32 -> the smallest bfloat16 >= d, as float32 (d >= 0)."""
    bits = d.astype(np.float32).view(np.uint32)
    up = (bits & 0xFFFF0000) + np.where(bits & 0xFFFF, 0x10000, 0).astype(np.uint32)
    return up.astype(np.uint32).view(np.float32)


def quantize(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """float32 [out, in] -> (uint32 [out, in/4], bfloat16-as-uint16 scales, biases [out, in/32])."""
    out_dim, in_dim = w.shape
    blocks = w.reshape(out_dim, in_dim // GROUP, GROUP)
    d = _bf16_up(np.abs(blocks).max(axis=-1) / 127)
    inv = np.where(d == 0, 0, 1 / np.where(d == 0, 1, d)).astype(np.float32)
    q = np.clip(np.rint(blocks * inv[..., None]), -127, 127)
    u = (q + 128).astype(np.uint32).reshape(out_dim, in_dim // 4, 4)
    packed = u[..., 0] | (u[..., 1] << 8) | (u[..., 2] << 16) | (u[..., 3] << 24)
    bias = (-128 * d).astype(np.float32)  # d times a power of two: exact in bfloat16
    to_bf16 = lambda x: (x.view(np.uint32) >> 16).astype(np.uint16)  # noqa: E731 (both are exact bfloat16)
    return packed.astype(np.uint32), to_bf16(d), to_bf16(bias)


def _read(path: Path):
    header, start = zero_copy.read_header(path)
    header.pop("__metadata__", None)
    data = np.memmap(path, dtype=np.uint8, mode="r")
    return header, start, data


def convert(src_dir: Path, out_dir: Path) -> Path:
    src = sorted(Path(src_dir).glob("*.safetensors"))
    if len(src) != 1:
        raise ValueError(f"{src_dir} must hold exactly one safetensors file (align it first)")
    header, start = zero_copy.read_header(src[0])
    header.pop("__metadata__", None)
    source = open(src[0], "rb")  # plain reads, so the 16 GB source does not count as resident memory

    def read(a, b):
        source.seek(start + a)
        return np.frombuffer(source.read(b - a), dtype=np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    plan = []  # (key, dtype, shape, nbytes, producer)
    for key, spec in header.items():
        a, b = spec["data_offsets"]
        raw = lambda a=a, b=b: read(a, b)  # noqa: E731
        if _quantized(key):
            if spec["dtype"] != "BF16":
                raise ValueError(f"{key}: expected BF16, got {spec['dtype']}")
            o, i = spec["shape"]
            cache = {}

            def produce(part, raw=raw, o=o, i=i, cache=cache):
                if not cache:
                    w = (raw().view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(o, i)
                    cache.update(zip(("weight", "scales", "biases"), quantize(w)))
                value = cache.pop(part)
                return value.tobytes()

            base = key[: -len(".weight")]
            plan.append((base + ".weight", "U32", [o, i // 4], o * i, lambda p=produce: p("weight")))
            plan.append((base + ".scales", "BF16", [o, i // GROUP], o * i // GROUP * 2, lambda p=produce: p("scales")))
            plan.append((base + ".biases", "BF16", [o, i // GROUP], o * i // GROUP * 2, lambda p=produce: p("biases")))
        else:
            plan.append((key, spec["dtype"], spec["shape"], b - a, lambda raw=raw: raw().tobytes()))
    meta, offset = {}, 0
    for key, dtype, shape, nbytes, _ in plan:
        meta[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps({"__metadata__": {"format": "pt", "mflux_q8": "w8a16 group32 symmetric bf16-scales"}, **meta},
                      separators=(",", ":")).encode()
    tmp = out_dir / "model.unaligned.safetensors"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        for key, _, _, nbytes, produce in plan:
            chunk = produce()
            if len(chunk) != nbytes:
                raise RuntimeError(f"{key}: {len(chunk)} bytes, expected {nbytes}")
            f.write(chunk)
    for name in ("config.json", "generation_config.json"):
        if (Path(src_dir) / name).exists():
            (out_dir / name).write_text((Path(src_dir) / name).read_text())
    return zero_copy.align(out_dir)


def convert_text_encoder(text_encoder) -> None:
    """Swap the quantized projections for MLX 8-bit QuantizedLinear layers (weights come from `load_text_encoder`)."""
    def swap(parent, name):
        linear = getattr(parent, name)
        out_dims, in_dims = linear.weight.shape
        setattr(parent, name, nn.QuantizedLinear(in_dims, out_dims, bias=False, group_size=GROUP, bits=8))

    for layer in text_encoder.language_model.layers:
        for path in TEXT_PROJECTIONS:
            parent, name = path.split(".")
            swap(getattr(layer, parent), name)
    swap(text_encoder, "lm_head")


def load_text_encoder(text_encoder, path: Path) -> None:
    from mflux.models.common_models.qwen3_vl.qwen3_vl_multimodal import Qwen3VLMultimodal

    if path.is_dir():
        files = sorted(path.glob("*.safetensors"))
        if len(files) != 1:
            raise ValueError(f"{path} must hold exactly one safetensors file")
        path = files[0]
    weights = zero_copy.load(path) if zero_copy.enabled() and zero_copy.page_aligned(path) else mx.load(str(path))
    params = {}
    for key, value in weights.items():
        if key.startswith("lm_head."):
            target = key
        elif key.startswith("model.visual."):
            target = "visual." + key[len("model.visual."):]
            value = Qwen3VLMultimodal.vision_weight(target, value)
        elif key.startswith("model.language_model."):
            target = "language_model." + key[len("model.language_model."):]
        else:
            raise ValueError(f"{path}: unexpected key {key}")
        params[target] = value
    expected = {p for p, _ in tree_flatten(text_encoder.parameters()) if not p.endswith(".inv_freq")}
    missing, extra = expected - set(params), set(params) - expected
    if missing or extra:
        raise ValueError(f"{path}: missing {sorted(missing)[:5]} ({len(missing)}), unexpected {sorted(extra)[:5]} ({len(extra)})")
    text_encoder.update(tree_unflatten(list(params.items())), strict=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(convert(Path(sys.argv[1]), Path(sys.argv[2])))
