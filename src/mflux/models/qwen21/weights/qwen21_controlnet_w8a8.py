"""(ain-soph fork) The Qwen-Image-2.1 Fun ControlNet-Union branch with W8A8 projections (ConvRot int8 weights and
activations, mx.fast.convrot_* on the Apple M5 TensorOps GEMM), as the DiT runs with MFLUX_QWEN21_W8A8.

Write the file once from the original checkpoint (alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union, bfloat16):

    python -m mflux.models.qwen21.weights.qwen21_controlnet_w8a8 SRC.safetensors OUT.safetensors [--classes qkv,out,gate_up,down]

and pass OUT as the ControlNet path (controlnet_path, or QwenImage21Controlnet.from_model). The loader recognises the
file by its metadata and swaps the projection classes it names for W8A8Linear; every other tensor (norms, the control
input projection, the classes left out) stays bfloat16 as the original stores it.

Projection classes (each is one entry of --classes):
    qkv      attn.to_q, attn.to_k, attn.to_v (one GEMM on the adjacent weights)
    out      attn.to_out.0
    gate_up  img_mlp.gate_layer, img_mlp.proj (one GEMM)
    down     img_mlp.out
    skip     after_proj and the first block's before_proj (with bias)

The projections are quantized from the stored bfloat16 values (widened to float32 exactly, so each weight is rounded
once) on the CPU stream: the float32 rotation is never a TF32 product and the file does not depend on the GPU. The file
is page-aligned with each packing group back to back, so with MFLUX_ZERO_COPY=1 it is mapped instead of read and with
MFLUX_ZERO_COPY_STREAM=1 each control block holds its weights only while it runs (QwenImage21ControlNet.__call__).
"""

import argparse
import json
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.quantization.w8a8 import GROUP, W8A8Linear
from mflux.models.common.weights.loading import zero_copy

FORMAT = "mflux-qwen21-controlnet-w8a8-convrot256"
CLASSES = {
    "qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
    "out": ("attn.to_out.0",),
    "gate_up": ("img_mlp.gate_layer", "img_mlp.proj"),
    "down": ("img_mlp.out",),
    "skip": ("after_proj", "before_proj"),
}
DEFAULT_CLASSES = ("qkv", "out", "gate_up", "down")


class W8A8LinearBias(W8A8Linear):
    """W8A8Linear plus a bias added in the activation dtype (the control branch's before_proj / after_proj)."""

    def __init__(self, input_dims: int, output_dims: int):
        super().__init__(input_dims, output_dims)
        self.bias = mx.zeros((output_dims,), dtype=mx.bfloat16)

    def __call__(self, x: mx.array) -> mx.array:
        y = super().__call__(x)
        return y + self.bias.astype(y.dtype)


def file_classes(path: Path) -> tuple[str, ...] | None:
    """The projection classes a W8A8 ControlNet file quantizes, or None for any other file (the original checkpoint)."""
    header, _ = zero_copy.read_header(Path(path))
    meta = header.get("__metadata__") or {}
    if meta.get("format") != FORMAT:
        return None
    classes = tuple(c for c in meta.get("w8a8", "").split(",") if c)
    unknown = [c for c in classes if c not in CLASSES]
    if unknown:
        raise ValueError(f"{path}: unknown W8A8 projection classes {unknown}")
    return classes


def _modules(controlnet, classes):
    """(parent module, attribute name, path) of every projection of the given classes."""
    out = []
    for index, block in enumerate(controlnet.control_blocks):
        for cls in classes:
            for path in CLASSES[cls]:
                *parents, name = path.split(".")
                module = block
                for part in parents:
                    module = module[int(part)] if part.isdigit() else getattr(module, part)
                if name.isdigit() or name in module:
                    out.append((module, name, f"control_blocks.{index}.{path}"))
    return out


def convert(controlnet, classes, quantize_weights: bool = False) -> None:
    """Swap the projections of `classes` for W8A8Linear (W8A8LinearBias where the projection has a bias); with
    quantize_weights, fill them from the float weights they replace."""
    for module, name, path in _modules(controlnet, classes):
        linear = module[int(name)] if name.isdigit() else getattr(module, name)
        if linear.weight.shape[1] % GROUP:
            raise ValueError(f"{path}: W8A8 needs a multiple of {GROUP} inputs")
        cls = W8A8LinearBias if "bias" in linear else W8A8Linear
        replacement = cls(linear.weight.shape[1], linear.weight.shape[0])
        if quantize_weights:
            replacement.weight, replacement.scale = mx.fast.convrot_quantize_weight(linear.weight.astype(mx.float32))
            if "bias" in linear:
                replacement.bias = linear.bias
            mx.eval(replacement.parameters())
        if name.isdigit():
            module[int(name)] = replacement
        else:
            setattr(module, name, replacement)


def load(controlnet, path: Path) -> None:
    """Load a W8A8 ControlNet file into `controlnet` (built from the transformer config), converting the classes the
    file names first. A page-aligned file loads as zero-copy views when MFLUX_ZERO_COPY=1."""
    classes = file_classes(path)
    if classes is None:
        raise ValueError(f"{path} is not a {FORMAT} file")
    convert(controlnet, classes)
    params = zero_copy.load(path) if zero_copy.enabled() and zero_copy.page_aligned(path) else mx.load(str(path))
    expected = dict(tree_flatten(controlnet.parameters()))
    missing = sorted(set(expected) - set(params))
    unexpected = sorted(set(params) - set(expected))
    mismatched = sorted(
        k
        for k in set(expected) & set(params)
        if expected[k].shape != params[k].shape or (expected[k].dtype == mx.int8) != (params[k].dtype == mx.int8)
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            f"{path} does not match the ControlNet: missing {missing[:5]} ({len(missing)}), "
            f"unexpected {unexpected[:5]} ({len(unexpected)}), wrong shape or dtype {mismatched[:5]} ({len(mismatched)})"
        )
    controlnet.update(tree_unflatten(list(params.items())), strict=False)


def save(source: Path, out: Path, classes=DEFAULT_CLASSES, transformer_config: dict | None = None) -> None:
    from mflux.models.qwen21.variants.controlnet.qwen_image21_controlnet_transformer import QwenImage21ControlNet

    config = transformer_config or {
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "num_layers": 32,
        "mlp_ratio": 3,
    }
    with mx.stream(mx.cpu):
        controlnet = QwenImage21ControlNet(config)
        # the stored dtype (bfloat16), not a cast: the float32 widening in convert() is exact
        controlnet.load_weights(list(mx.load(str(source)).items()), strict=True)
        convert(controlnet, classes, quantize_weights=True)
        params = dict(tree_flatten(controlnet.parameters()))
        mx.eval(list(params.values()))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # zero_copy.align writes the tensors page-aligned with each packing group (q/k/v, gate/proj) back to back
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        staged = Path(tmp) / "model.safetensors"
        mx.save_safetensors(
            str(staged), params, {"format": FORMAT, "w8a8": ",".join(classes), "source": Path(source).name}
        )
        aligned = zero_copy.align(Path(tmp))
        aligned.rename(out)
    if not zero_copy.page_aligned(out) or file_classes(out) != tuple(classes):
        raise RuntimeError(f"{out} was not written as expected")
    print(f"{out}: {len(params)} tensors, {out.stat().st_size / 1e9:.2f} GB, W8A8 classes {','.join(classes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="the original ControlNet safetensors file (bfloat16)")
    parser.add_argument("out", type=Path)
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help=f"subset of {','.join(CLASSES)}")
    parser.add_argument("--transformer-config", type=Path, help="the base checkpoint's transformer/config.json")
    args = parser.parse_args()
    classes = tuple(c for c in args.classes.split(",") if c)
    unknown = [c for c in classes if c not in CLASSES]
    if unknown:
        parser.error(f"unknown classes {unknown}")
    config = json.loads(args.transformer_config.read_text()) if args.transformer_config else None
    save(args.source, args.out, classes, config)


if __name__ == "__main__":
    main()
