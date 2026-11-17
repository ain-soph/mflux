"""Qwen-Image-2.1 DiT and text encoder from ComfyUI int8_convrot checkpoints (Comfy-Org/Qwen-Image-2.1, qwen_image_2.1[_turbo]_int8_convrot).

MFLUX_QWEN21_W8A8=<file> replaces the projections of every transformer block (to_q/k/v, to_out, gate/up/out of the
feed-forward) by W8A8Linear and loads the DiT from that file instead of the checkpoint's transformer/ folder; the
text encoder and the VAE still come from the checkpoint. The non-block layers stay bfloat16 as the file stores them.
The file can be page-aligned (zero_copy.align) to load it zero-copy and stream it like the bfloat16 weights.

MFLUX_QWEN21_TE_W8A8=<file> does the same for the Qwen3-VL text encoder (Comfy-Org qwen3vl_8b_int8_convrot): the 36
language layers' projections and the generation head (lm_head) become W8A8Linear and the token embedding a
ConvRotEmbedding; the vision tower stays bfloat16 as stored.
"""

import json
import os
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten

from mflux.models.common.quantization.w8a8 import GROUP, ConvRotEmbedding, W8A8Linear
from mflux.models.common.weights.loading import zero_copy

PROJECTIONS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out")
# The checkpoint fuses the feed-forward input projections as gate_up = [gate_layer; proj] (rows): undoing the rotation,
# the first half matches gate_layer (cos 0.99996) and the second proj; the per-row int8 weights are within 0.9% of bf16.
FUSED = {"img_mlp.gate_up": ("img_mlp.gate_layer", "img_mlp.proj")}


TEXT_PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def checkpoint() -> Path | None:
    path = os.environ.get("MFLUX_QWEN21_W8A8")
    return Path(path) if path else None


def text_checkpoint() -> Path | None:
    path = os.environ.get("MFLUX_QWEN21_TE_W8A8")
    return Path(path) if path else None


def convert_text_encoder(text_encoder) -> None:
    model = text_encoder.language_model
    for layer in model.layers:
        for path in TEXT_PROJECTIONS:
            parent, name = path.split(".")
            W8A8Linear.replace(getattr(layer, parent), name)
    vocab, dims = model.embed_tokens.weight.shape
    model.embed_tokens = ConvRotEmbedding(vocab, dims)
    # the generation head (enhance_prompt, auto_mask, verify) is int8 convrot too
    W8A8Linear.replace(text_encoder, "lm_head")


def _check_quant(weights: dict) -> None:
    for key in [k for k in weights if k.endswith(".comfy_quant")]:
        config = json.loads(bytes(memoryview(weights.pop(key).astype(mx.uint8))).decode())
        # e.g. {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
        if not config.get("convrot") or not str(config.get("format", "")).startswith("int8") or int(
            config.get("convrot_groupsize", 0)
        ) != GROUP:
            raise ValueError(f"{key}: unsupported quantization {config}")


def load_text_encoder(text_encoder, path: Path) -> None:
    from mlx.utils import tree_flatten

    from mflux.models.common_models.qwen3_vl.qwen3_vl_multimodal import Qwen3VLMultimodal

    weights = _read(path)
    _check_quant(weights)
    params = {}
    for key, value in weights.items():
        if key.startswith("lm_head."):
            target = key
        elif key.startswith("model.visual."):
            target = "visual." + key[len("model.visual."):]
            value = Qwen3VLMultimodal.vision_weight(target, value)
        elif key.startswith("model."):
            target = "language_model." + key[len("model."):]
        else:
            raise ValueError(f"{path}: unexpected key {key}")
        if target.endswith(".weight_scale"):
            target, value = target[: -len(".weight_scale")] + ".scale", value.reshape(-1)
        params[target] = value
    # inv_freq buffers are computed by the modules, not stored
    expected = {p for p, _ in tree_flatten(text_encoder.parameters()) if not p.endswith(".inv_freq")}
    missing, extra = expected - set(params), set(params) - expected
    if missing or extra:
        raise ValueError(f"{path}: missing {sorted(missing)[:5]} ({len(missing)}), unexpected {sorted(extra)[:5]} ({len(extra)})")
    text_encoder.update(tree_unflatten(list(params.items())), strict=False)


def convert(transformer) -> None:
    for block in transformer.transformer_blocks:
        for path in PROJECTIONS:
            *parents, name = path.split(".")
            module = block
            for part in parents:
                module = module[int(part)] if part.isdigit() else getattr(module, part)
            W8A8Linear.replace(module, name)


def _read(path: Path) -> dict[str, mx.array]:
    if path.is_dir():
        files = sorted(path.glob("*.safetensors"))
        if len(files) != 1:
            raise ValueError(f"{path} must hold exactly one safetensors file")
        path = files[0]
    if zero_copy.enabled() and zero_copy.page_aligned(path):
        return zero_copy.load(path)
    return mx.load(str(path))


def load(transformer, path: Path) -> None:
    from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer

    weights = _read(path)
    # ComfyUI attention hints (Turbo: {"attention": "comfy_kitchen_int8"} on blocks 1-31) are not used here.
    for key in [k for k in weights if k.endswith(".comfy_attention.config")]:
        del weights[key]
    _check_quant(weights)
    params = {}
    for key, value in weights.items():
        prefix, _, leaf = key.rpartition(".")
        name = "scale" if leaf == "weight_scale" else leaf
        if leaf == "weight_scale":
            value = value.reshape(-1)
        fused = next((n for n in FUSED if prefix.endswith("." + n)), None)
        if fused is None:
            params[f"{prefix}.{name}"] = value
            continue
        block, parts = prefix[: -len(fused)], FUSED[fused]
        # registered views, so streaming releases each half with its block
        halves = zero_copy.split_rows(value, len(parts)) if leaf == "weight" else mx.split(value, len(parts))
        for part, half in zip(parts, halves):
            params[f"{block}{part}.{name}"] = half
    params = Qwen21Initializer._normalize_transformer_weights(params)
    quantized = sum(1 for k, v in params.items() if v.dtype == mx.int8)
    expected = len(transformer.transformer_blocks) * len(PROJECTIONS)
    if quantized != expected:
        raise ValueError(f"{path}: {quantized} int8 projections, expected {expected}")
    transformer.update(tree_unflatten(list(params.items())), strict=False)
