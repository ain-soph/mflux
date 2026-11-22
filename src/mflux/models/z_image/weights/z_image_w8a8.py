"""(ain-soph fork) Z-Image DiT with W8A8 projections (ConvRot int8 weights and activations, Apple M5 TensorOps GEMM).

MFLUX_ZIMAGE_W8A8=<file> replaces the attention (to_q/k/v, to_out) and feed-forward (w1/w2/w3) projections of every
transformer block (noise refiner, context refiner, main layers) by W8A8Linear and loads the whole DiT from that file
instead of the checkpoint's transformer/ folder; the text encoder and the VAE still come from the checkpoint. The other
DiT layers (embedders, adaLN modulation, norms, final layer) stay bfloat16 as the file stores them.

Write the file once from the official checkpoint (the weights are rotated by a block-diagonal Hadamard-256 matrix and
quantized per output row, mx.fast.convrot_quantize_weight):

    python -m mflux.models.z_image.weights.z_image_w8a8 Tongyi-MAI/Z-Image-Turbo OUT.safetensors

The file is page-aligned, so with MFLUX_ZERO_COPY=1 it is mapped instead of read, and with MFLUX_ZERO_COPY_STREAM=1 the
transformer holds the mapped weights only while it denoises (see zero_copy.streamed).
"""

import os
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.quantization.w8a8 import GROUP, W8A8Linear
from mflux.models.common.weights.loading import zero_copy

ENV = "MFLUX_ZIMAGE_W8A8"
PROJECTIONS = (
    "attention.to_q",
    "attention.to_k",
    "attention.to_v",
    "attention.to_out.0",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
)
FORMAT = "mflux-z-image-w8a8-convrot256"


class ZImageW8A8:
    @staticmethod
    def checkpoint() -> Path | None:
        path = os.environ.get(ENV)
        return Path(path).expanduser() if path else None

    @staticmethod
    def validate(quantize: int | None, lora_paths: list[str] | None) -> Path | None:
        """The MFLUX_ZIMAGE_W8A8 file, checked before anything is loaded; None when the variable is unset."""
        path = ZImageW8A8.checkpoint()
        if path is None:
            return None
        if quantize is not None or lora_paths:
            raise ValueError(f"{ENV} loads an int8 DiT; it cannot be combined with quantize or LoRAs.")
        if not path.is_file():
            raise FileNotFoundError(f"{ENV} points to a missing file: {path}")
        return path

    @staticmethod
    def without_transformer(definition):
        class WithoutTransformer(definition):
            @staticmethod
            def get_components():
                return [c for c in definition.get_components() if c.name != "transformer"]

            @staticmethod
            def get_download_patterns():
                return [p for p in definition.get_download_patterns() if not p.startswith("transformer/")]

        return WithoutTransformer

    @staticmethod
    def blocks(transformer):
        return [*transformer.noise_refiner, *transformer.context_refiner, *transformer.layers]

    @staticmethod
    def convert(transformer, quantize_weights: bool = False) -> None:
        """Swap each block projection for W8A8Linear; with quantize_weights, fill it from the float weight."""
        for block in ZImageW8A8.blocks(transformer):
            for path in PROJECTIONS:
                *parents, name = path.split(".")
                module = block
                for part in parents:
                    module = module[int(part)] if part.isdigit() else getattr(module, part)
                linear = module[int(name)] if name.isdigit() else getattr(module, name)
                if linear.weight.shape[1] % GROUP or "bias" in linear:
                    raise ValueError(f"{path}: W8A8 needs a bias-free projection with a multiple of {GROUP} inputs")
                weight = linear.weight
                w8a8 = W8A8Linear.replace(module, name)
                if quantize_weights:
                    w8a8.weight, w8a8.scale = mx.fast.convrot_quantize_weight(weight)
                    mx.eval(w8a8.weight, w8a8.scale)

    @staticmethod
    def load(transformer, path: Path) -> None:
        # a page-aligned file loads as views of one mapping, so the transformer can be streamed
        if zero_copy.enabled() and zero_copy.page_aligned(path):
            params = zero_copy.load(path)
        else:
            params = mx.load(str(path))
        expected = dict(tree_flatten(transformer.parameters()))
        missing = sorted(set(expected) - set(params))
        unexpected = sorted(set(params) - set(expected))
        mismatched = sorted(
            k
            for k in set(expected) & set(params)
            if expected[k].shape != params[k].shape
            or (expected[k].dtype == mx.int8) != (params[k].dtype == mx.int8)
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                f"{path} does not match the Z-Image transformer: missing {missing[:5]} ({len(missing)}), "
                f"unexpected {unexpected[:5]} ({len(unexpected)}), wrong shape or dtype {mismatched[:5]} ({len(mismatched)})"
            )
        transformer.update(tree_unflatten(list(params.items())), strict=False)

    @staticmethod
    def save(model_path: str, out: Path) -> None:
        from mflux.models.common.config import ModelConfig
        from mflux.models.z_image.variants.z_image import ZImage

        os.environ.pop(ENV, None)
        model = ZImage(model_path=model_path, model_config=ModelConfig.z_image_turbo())
        transformer = model.transformer
        del model
        ZImageW8A8.convert(transformer, quantize_weights=True)
        params = dict(tree_flatten(transformer.parameters()))
        mx.eval(list(params.values()))
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(out), params, {"format": FORMAT}, page_align=True)
        if not zero_copy.page_aligned(out):
            raise RuntimeError(f"{out} was not written page-aligned")
        print(f"{out}: {len(params)} tensors, {out.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ZImageW8A8.save(sys.argv[1], Path(sys.argv[2]))
