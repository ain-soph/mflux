"""Qwen-Image-2.1 DiT from ComfyUI int8_convrot checkpoints (Comfy-Org/Qwen-Image-2.1, qwen_image_2.1[_turbo]_int8_convrot).

MFLUX_QWEN21_W8A8=<file or directory> replaces the projections of every transformer block (to_q/k/v, to_out, gate/up/out
of the feed-forward) by W8A8Linear and loads the DiT from that file instead of the checkpoint's transformer/ folder; the
text encoder and the VAE still come from the checkpoint. The non-block layers stay bfloat16 as the file stores them.
QKV and gate/up are packed after loading, so each runs as one GEMM with one activation quantization.
"""

import json
import os
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.quantization.w8a8 import GROUP, ConvRotEmbedding, W8A8Linear
from mflux.models.common.weights.loading import zero_copy

PROJECTIONS = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.gate_layer",
    "img_mlp.proj",
    "img_mlp.out",
)
# The checkpoint fuses the feed-forward input projections as gate_up = [gate_layer; proj] (rows): undoing the rotation,
# the first half matches gate_layer (cos 0.99996) and the second proj; the per-row int8 weights are within 0.9% of bf16.
FUSED = {"img_mlp.gate_up": ("img_mlp.gate_layer", "img_mlp.proj")}
# (ain-soph fork) MFLUX_QWEN21_TE_W8A8: the Qwen3-VL text encoder's projections in its 36 language layers.
TEXT_PROJECTIONS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


class Qwen21W8A8:
    """Loading the DiT from an int8_convrot file (MFLUX_QWEN21_W8A8): checks, conversion and weight loading."""

    @staticmethod
    def checkpoint() -> Path | None:
        path = os.environ.get("MFLUX_QWEN21_W8A8")
        return Path(path).expanduser() if path else None

    @staticmethod
    def download_patterns(patterns: list[str]) -> list[str]:
        """The checkpoint files to fetch: without the float DiT shards when the DiT comes from MFLUX_QWEN21_W8A8."""
        if Qwen21W8A8.checkpoint() is None:
            return patterns
        return [p for p in patterns if p != "transformer/*.safetensors"]

    @staticmethod
    def check_savable(model) -> None:
        """save_model would write int8 weights and scales as an ordinary checkpoint that cannot be reloaded."""
        if getattr(model, "_w8a8_checkpoint", None) is not None:
            raise ValueError(
                "A model loaded with MFLUX_QWEN21_W8A8 cannot be saved; its DiT is the int8_convrot file "
                f"({model._w8a8_checkpoint}). Load it without the variable to save the checkpoint."
            )

    @staticmethod
    def validate(quantize: int | None, lora_paths: list[str] | None) -> Path | None:
        """The MFLUX_QWEN21_W8A8 file, checked before anything is downloaded; None when the variable is unset."""
        path = Qwen21W8A8.checkpoint()
        if path is None:
            return None
        if quantize is not None or lora_paths:
            raise ValueError("MFLUX_QWEN21_W8A8 loads an int8 DiT; it cannot be combined with quantize or LoRAs.")
        if not path.exists():
            raise FileNotFoundError(f"MFLUX_QWEN21_W8A8 points to a missing file: {path}")
        return path

    @staticmethod
    def prepare(model, definition, quantize: int | None, lora_paths: list[str] | None):
        """Before loading: with MFLUX_QWEN21_W8A8 set, swap the block projections for W8A8Linear and return a weight
        definition without the transformer (Qwen21W8A8.load() reads it from the int8 file), plus that file; else (definition, None)."""
        path = Qwen21W8A8.validate(quantize, lora_paths)
        if path is None:
            return definition, None
        Qwen21W8A8.convert(model.transformer)
        model._w8a8_checkpoint = path

        class WithoutTransformer(definition):
            @staticmethod
            def get_components():
                return [c for c in definition.get_components() if c.name != "transformer"]

        return WithoutTransformer, path

    @staticmethod
    def convert(transformer) -> None:
        for block in transformer.transformer_blocks:
            for path in PROJECTIONS:
                *parents, name = path.split(".")
                module = block
                for part in parents:
                    module = module[int(part)] if part.isdigit() else getattr(module, part)
                W8A8Linear.replace(module, name)

    @staticmethod
    def _check_quant(weights: dict) -> None:
        quant_keys = [k for k in weights if k.endswith(".comfy_quant")]
        for key in quant_keys:
            config = json.loads(bytes(memoryview(weights.pop(key).astype(mx.uint8))).decode())
            # e.g. {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
            if (
                not config.get("convrot")
                or not str(config.get("format", "")).startswith("int8")
                or int(config.get("convrot_groupsize", 0)) != GROUP
            ):
                raise ValueError(f"{key}: unsupported quantization {config}")
        # W8A8Linear rotates the activations, so an int8 weight stored without the rotation would give x H W^T
        # instead of x W^T without any error: every int8 weight needs its convrot entry.
        int8 = [
            key.removesuffix(".weight")
            for key, value in weights.items()
            if key.endswith(".weight") and value.dtype == mx.int8
        ]
        missing = sorted(f"{prefix}.comfy_quant" for prefix in int8 if f"{prefix}.comfy_quant" not in quant_keys)
        if missing:
            raise ValueError(
                f"int8 weights without quantization metadata: missing {missing[:5]} ({len(missing)}); an int8_convrot "
                f'file stores e.g. {{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": {GROUP}}} there'
            )

    @staticmethod
    def _same_kind(expected: mx.Dtype, found: mx.Dtype) -> bool:
        # int8 weights must stay int8; every other parameter (scales, biases, norms) must be floating point,
        # so an integer scale such as int32 is rejected instead of being multiplied in.
        if expected == mx.int8:
            return found == mx.int8
        return mx.issubdtype(found, mx.floating)

    @staticmethod
    def _check_complete(path: Path, transformer, params: dict[str, mx.array]) -> None:
        # update(strict=False) would leave a missing parameter (such as a projection's scale) at its zero
        # initial value and would accept a wrong shape, so the file must match the transformer exactly.
        expected = dict(tree_flatten(transformer.parameters()))
        missing = sorted(set(expected) - set(params))
        unexpected = sorted(set(params) - set(expected))
        mismatched = sorted(
            k
            for k in set(expected) & set(params)
            if expected[k].shape != params[k].shape or not Qwen21W8A8._same_kind(expected[k].dtype, params[k].dtype)
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                f"{path} does not match the transformer: missing {missing[:5]} ({len(missing)}), "
                f"unexpected {unexpected[:5]} ({len(unexpected)}), wrong shape or dtype {mismatched[:5]} ({len(mismatched)})"
            )

    @staticmethod
    def _read(path: Path) -> dict[str, mx.array]:
        if path.is_dir():
            files = sorted(path.glob("*.safetensors"))
            if len(files) != 1:
                raise ValueError(f"{path} must hold exactly one safetensors file")
            path = files[0]
        # a page-aligned file (zero_copy.align) loads as views of one mapping, so it can be streamed per block
        if zero_copy.enabled() and zero_copy.page_aligned(path):
            return zero_copy.load(path)
        return mx.load(str(path))

    @staticmethod
    def load(transformer, path: Path) -> None:
        from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer

        weights = Qwen21W8A8._read(path)
        # ComfyUI attention hints (Turbo: {"attention": "comfy_kitchen_int8"} on blocks 1-31) are not used here.
        for key in [k for k in weights if k.endswith(".comfy_attention.config")]:
            del weights[key]
        Qwen21W8A8._check_quant(weights)
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
            # zero-copy: registered views, so streaming releases each half with its block and fused_layers can
            # pack them again without a copy
            halves = zero_copy.split_rows(value, len(parts)) if leaf == "weight" else mx.split(value, len(parts))
            for part, half in zip(parts, halves):
                params[f"{block}{part}.{name}"] = half
        params = Qwen21Initializer._normalize_transformer_weights(params)
        Qwen21W8A8._check_complete(path, transformer, params)
        transformer.update(tree_unflatten(list(params.items())), strict=False)
        if any(zero_copy.is_view(v) for k, v in params.items() if v.dtype == mx.int8):
            return  # zero-copy weights stay views; the blocks pack them per call (W8A8Linear.fused_layers)
        for block in transformer.transformer_blocks:
            attn, mlp = block.attn, block.img_mlp
            attn._w8a8_qkv = W8A8Linear.pack((attn.to_q, attn.to_k, attn.to_v))
            mlp._w8a8_gate_proj = W8A8Linear.pack((mlp.gate_layer, mlp.proj))

    # (ain-soph fork) MFLUX_QWEN21_TE_W8A8=<file> does the same for the Qwen3-VL text encoder (Comfy-Org
    # qwen3vl_8b_int8_convrot): the 36 language layers' projections and the generation head (lm_head, used by
    # enhance_prompt, auto_mask and verify) become W8A8Linear and the token embedding a ConvRotEmbedding; the vision
    # tower stays bfloat16 as stored.
    @staticmethod
    def text_checkpoint() -> Path | None:
        path = os.environ.get("MFLUX_QWEN21_TE_W8A8")
        return Path(path) if path else None

    @staticmethod
    def prepare_text_encoder(model, definition, quantize: int | None, small_text_encoder: bool):
        """Before loading: with MFLUX_QWEN21_TE_W8A8 set, swap the text encoder's layers and return a weight
        definition without the text encoder plus that file; else (definition, None)."""
        path = None if small_text_encoder else Qwen21W8A8.text_checkpoint()
        if path is None:
            return definition, None
        if quantize is not None:
            raise ValueError("MFLUX_QWEN21_TE_W8A8 already quantizes the text encoder; do not pass quantize.")
        if not path.exists():
            raise FileNotFoundError(f"MFLUX_QWEN21_TE_W8A8 points to a missing file: {path}")
        Qwen21W8A8.convert_text_encoder(model.text_encoder)

        class WithoutTextEncoder(definition):
            @staticmethod
            def get_components():
                return [c for c in definition.get_components() if c.name != "text_encoder"]

        return WithoutTextEncoder, path

    @staticmethod
    def convert_text_encoder(text_encoder) -> None:
        model = text_encoder.language_model
        for layer in model.layers:
            for path in TEXT_PROJECTIONS:
                parent, name = path.split(".")
                W8A8Linear.replace(getattr(layer, parent), name)
        vocab, dims = model.embed_tokens.weight.shape
        model.embed_tokens = ConvRotEmbedding(vocab, dims)
        W8A8Linear.replace(text_encoder, "lm_head")

    @staticmethod
    def load_text_encoder(text_encoder, path: Path) -> None:
        from mflux.models.common_models.qwen3_vl.qwen3_vl_multimodal import Qwen3VLMultimodal

        weights = Qwen21W8A8._read(path)
        Qwen21W8A8._check_quant(weights)
        params = {}
        for key, value in weights.items():
            if key.startswith("lm_head."):
                target = key
            elif key.startswith("model.visual."):
                target = "visual." + key[len("model.visual.") :]
                value = Qwen3VLMultimodal.vision_weight(target, value)
            elif key.startswith("model."):
                target = "language_model." + key[len("model.") :]
            else:
                raise ValueError(f"{path}: unexpected key {key}")
            if target.endswith(".weight_scale"):
                target, value = target[: -len(".weight_scale")] + ".scale", value.reshape(-1)
            params[target] = value
        # inv_freq buffers are computed by the modules, not stored
        expected = {p for p, _ in tree_flatten(text_encoder.parameters()) if not p.endswith(".inv_freq")}
        missing, extra = expected - set(params), set(params) - expected
        if missing or extra:
            raise ValueError(
                f"{path}: missing {sorted(missing)[:5]} ({len(missing)}), unexpected {sorted(extra)[:5]} ({len(extra)})"
            )
        text_encoder.update(tree_unflatten(list(params.items())), strict=False)
