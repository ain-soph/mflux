"""Zero-copy weights: page-aligned safetensors mapped read-only and wrapped as MLX arrays without a copy.

With MFLUX_ZERO_COPY=1, a safetensors file whose tensors all start on a page boundary (see `align`) is not
read into MLX buffers. Each tensor becomes an MLX array over a read-only mmap of the file, so the GPU reads the
weights straight from the page cache: an idle process holds no wired memory for them, the kernel may drop the
pages under memory pressure (they are re-read from disk on the next use), and several processes that map the
same file share one copy.

Projection packing (QKV, gate/up) normally concatenates the weights into a new buffer every generation.
`align` writes the tensors of each packing group next to each other with no padding, so `adjacent_view` can
return the packed weight as one more view of the mapping instead of a copy.

Usage: python -m mflux.models.common.weights.loading.zero_copy DIR [DIR...]
  Replaces the *.safetensors in each component directory by one page-aligned model file (verified byte for
  byte before the originals are deleted). The result is an ordinary safetensors checkpoint.
"""

import json
import mmap
import os
import re
import struct
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

PAGE = os.sysconf("SC_PAGE_SIZE")
DTYPES = {
    "BF16": mx.bfloat16,
    "F16": mx.float16,
    "F32": mx.float32,
    "F64": mx.float64,
    "I8": mx.int8,
    "U8": mx.uint8,
    "I16": mx.int16,
    "I32": mx.int32,
    "I64": mx.int64,
    "U32": mx.uint32,
    "BOOL": mx.bool_,
}
# Tensors that projection packing concatenates, in packing order; align keeps each group contiguous.
PACK_GROUPS = (
    (".attn.to_q.weight", ".attn.to_k.weight", ".attn.to_v.weight"),
    (".img_mlp.gate_layer.weight", ".img_mlp.proj.weight"),
)

# id(array) -> (array, mapping, absolute offset, nbytes) for every view handed out here
_origin: dict[int, tuple] = {}
# One view per (mapping, range, dtype, shape): packing every generation reuses the same arrays instead of
# registering new ones (which would never be freed).
_views: dict[tuple, mx.array] = {}


def enabled() -> bool:
    return os.environ.get("MFLUX_ZERO_COPY") == "1"


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(length)), 8 + length


def page_aligned(path: Path) -> bool:
    header, start = read_header(path)
    return start % PAGE == 0 and all(
        (start + spec["data_offsets"][0]) % PAGE == 0 for key, spec in header.items() if key != "__metadata__"
    )


def _view(mapping: mmap.mmap, offset: int, nbytes: int, dtype: mx.Dtype, shape: list[int]) -> mx.array:
    # Metal wraps host memory without a copy only for page-aligned, page-sized ranges: map whole pages
    # and slice the tail off.
    padded = -(-nbytes // PAGE) * PAGE
    raw = mx.from_dlpack(np.frombuffer(mapping, np.uint8, padded, offset))[:nbytes]
    return raw.view(dtype).reshape(shape) if dtype != mx.uint8 else raw.reshape(shape)


def _registered(mapping: mmap.mmap, offset: int, nbytes: int, dtype: mx.Dtype, shape: list[int]) -> mx.array:
    key = (id(mapping), offset, nbytes, dtype, tuple(shape))
    array = _views.get(key)
    if array is None:
        array = _view(mapping, offset, nbytes, dtype, shape)
        _views[key] = array
        _origin[id(array)] = (array, mapping, offset, nbytes)
    return array


def load(path: Path) -> dict[str, mx.array]:
    """All tensors of a page-aligned safetensors file as zero-copy views of one read-only mapping."""
    header, start = read_header(path)
    with open(path, "rb") as f:
        mapping = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    out = {}
    for key, spec in header.items():
        if key == "__metadata__":
            continue
        a, b = spec["data_offsets"]
        if b == a:
            out[key] = mx.zeros(spec["shape"], DTYPES[spec["dtype"]])
            continue
        out[key] = _registered(mapping, start + a, b - a, DTYPES[spec["dtype"]], spec["shape"])
    return out


def adjacent_view(weights: list[mx.array]) -> mx.array | None:
    """The row-wise concatenation of `weights` as a view, when they are back-to-back in one mapping."""
    entries = [_origin.get(id(w)) for w in weights]
    if any(e is None or e[0] is not w for e, w in zip(entries, weights)):
        return None
    mapping, offset = entries[0][1], entries[0][2]
    position = offset
    for e in entries:
        if e[1] is not mapping or e[2] != position:
            return None
        position += e[3]
    first = weights[0]
    if any(w.dtype != first.dtype or w.shape[1:] != first.shape[1:] for w in weights):
        return None
    rows = sum(w.shape[0] for w in weights)
    return _registered(mapping, offset, position - offset, first.dtype, [rows, *first.shape[1:]])


def split_rows(packed: mx.array, parts: int) -> list[mx.array]:
    """mx.split(packed, parts, axis=0), as registered views when `packed` came from adjacent_view, so the
    next generation packs the same weights without a copy again."""
    entry = _origin.get(id(packed))
    if entry is None or entry[0] is not packed:
        return mx.split(packed, parts, axis=0)
    _, mapping, offset, nbytes = entry
    rows, step = packed.shape[0] // parts, nbytes // parts
    return [_registered(mapping, offset + i * step, step, packed.dtype, [rows, *packed.shape[1:]]) for i in range(parts)]


def _order(keys: list[str]) -> list[str]:
    """Natural order, with every packing group's members consecutive and in packing order."""

    def natural(key: str):
        return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", key)]

    keys = sorted(keys, key=natural)
    placed, out = set(), []
    for key in keys:
        if key in placed:
            continue
        group = next((g for g in PACK_GROUPS if key.endswith(g)), None)
        members = [key]
        if group is not None:
            prefix = key[: -len(next(s for s in group if key.endswith(s)))]
            candidates = [prefix + s for s in group]
            if all(c in keys for c in candidates):
                members = candidates
        for m in members:
            placed.add(m)
            out.append(m)
    return out


def align(component_dir: Path) -> Path:
    """Rewrite a component's safetensors as one page-aligned file; verify; delete the originals."""
    component_dir = Path(component_dir)
    shards = sorted(p for p in component_dir.glob("*.safetensors") if not p.name.startswith("._"))
    if len(shards) == 1 and page_aligned(shards[0]):
        print(f"{component_dir}: already page-aligned")
        return shards[0]
    sources = {}
    metadata = {}
    for shard in shards:
        header, start = read_header(shard)
        metadata.update(header.pop("__metadata__", {}) or {})
        for key, spec in header.items():
            sources[key] = (shard, start, spec)
    # One file named like a single-file checkpoint of that component type (diffusers and transformers both
    # look for it when there is no index).
    name = "diffusion_pytorch_model.safetensors" if any("diffusion_pytorch_model" in s.name for s in shards) else "model.safetensors"
    target = component_dir / (name + ".aligning")
    keys = _order(list(sources))
    header, offset = {}, 0
    for key in keys:
        _, _, spec = sources[key]
        nbytes = spec["data_offsets"][1] - spec["data_offsets"][0]
        header[key] = {"dtype": spec["dtype"], "shape": spec["shape"], "data_offsets": [offset, offset + nbytes]}
        offset += -(-nbytes // PAGE) * PAGE
    if metadata:
        header = {"__metadata__": metadata, **header}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-(8 + len(blob))) % PAGE)
    data_start = 8 + len(blob)
    with open(target, "wb") as out:
        out.write(struct.pack("<Q", len(blob)) + blob)
        for key in keys:
            shard, start, spec = sources[key]
            a, b = spec["data_offsets"]
            out.seek(data_start + header[key]["data_offsets"][0])
            with open(shard, "rb") as src:
                src.seek(start + a)
                out.write(src.read(b - a))
        out.truncate(data_start + offset)
    # Verify every tensor byte for byte before touching the originals.
    new_header, new_start = read_header(target)
    with open(target, "rb") as f:
        for key in keys:
            shard, start, spec = sources[key]
            a, b = spec["data_offsets"]
            na, nb = new_header[key]["data_offsets"]
            with open(shard, "rb") as src:
                src.seek(start + a)
                f.seek(new_start + na)
                if src.read(b - a) != f.read(nb - na):
                    target.unlink()
                    raise RuntimeError(f"verification failed for {key}")
    assert page_aligned(target)
    for shard in shards:
        shard.unlink()
    index = component_dir / (shards[0].name.split("-0000")[0] + ".safetensors.index.json")
    if len(shards) > 1 and index.exists():
        index.unlink()
    final = component_dir / name
    target.rename(final)
    print(f"{component_dir}: {len(shards)} file(s) -> {final.name}, {len(keys)} tensors, page {PAGE}")
    return final


if __name__ == "__main__":
    for d in sys.argv[1:]:
        align(Path(d))
