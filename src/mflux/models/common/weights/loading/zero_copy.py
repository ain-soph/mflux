"""Zero-copy weights: page-aligned safetensors mapped read-only and wrapped as MLX arrays without a copy.

With MFLUX_ZERO_COPY=1, a safetensors file whose tensors all start on a page boundary (see `align`) is not
read into MLX buffers. Each tensor becomes an MLX array over a read-only mapping of the file (MLX's
SafetensorsFile, ain-soph/mlx `serving`), so the GPU reads the
weights straight from the page cache: an idle process holds no wired memory for them, the kernel may drop the
pages under memory pressure (they are re-read from disk on the next use), and several processes that map the
same file share one copy.

Projection packing (QKV, gate/up) normally concatenates the weights into a new buffer every generation.
`align` writes the tensors of each packing group next to each other with no padding, so `adjacent_view` can
return the packed weight as one more view of the mapping instead of a copy.

With MFLUX_ZERO_COPY_STREAM=1 as well, the layers wrapped in `streamed(layer)` (the Qwen-Image-2.1 DiT blocks and
the Qwen3-VL text-encoder layers) hold their weights only while they run: the views are created on entry and
dropped on exit. MLX makes every live buffer resident (wired) while the GPU works, so otherwise a generation
wires all weights at once; streamed, it wires about one layer plus the activations. Projection packing is
skipped in this mode (measured equally fast without it).

Usage: python -m mflux.models.common.weights.loading.zero_copy DIR [DIR...]
  Replaces the *.safetensors in each component directory by one page-aligned model file (verified byte for
  byte before the originals are deleted). The result is an ordinary safetensors checkpoint.
"""

import json
from contextlib import contextmanager
import os
import re
import struct
import sys
from pathlib import Path

import mlx.core as mx

PAGE = os.sysconf("SC_PAGE_SIZE")
# Tensors that projection packing concatenates, in packing order; align keeps each group contiguous.
PACK_GROUPS = (
    (".attn.to_q.weight", ".attn.to_k.weight", ".attn.to_v.weight"),
    (".img_mlp.gate_layer.weight", ".img_mlp.proj.weight"),
)

# Where an array handed out here comes from: (file, keys, part). keys is one tensor name, or several that lie back
# to back in the file and are packed row-wise (file.get_rows); part is (i, n) for the i-th of n row blocks of a
# single tensor, else None.
_Source = tuple
# id(array) -> (array, source) for every view handed out here
_origin: dict[int, tuple] = {}
# One view per source: packing every generation reuses the same arrays instead of registering new ones (which would
# never be freed).
_views: dict[tuple, mx.array] = {}
# One open file (header read and file mapped once) per path
_files: dict[str, "mx.SafetensorsFile"] = {}


def enabled() -> bool:
    return os.environ.get("MFLUX_ZERO_COPY") == "1"


def streaming() -> bool:
    return enabled() and os.environ.get("MFLUX_ZERO_COPY_STREAM") == "1"


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(length)), 8 + length


def page_aligned(path: Path) -> bool:
    """Written by `align`: the data starts on a page and every tensor whose size is a whole number of pages
    starts on a page (the small rest, norms and biases, follows them and is copied when loaded)."""
    header, start = read_header(path)
    specs = [spec for key, spec in header.items() if key != "__metadata__"]
    return start % PAGE == 0 and all(
        (start + s["data_offsets"][0]) % PAGE == 0
        for s in specs
        if (s["data_offsets"][1] - s["data_offsets"][0]) % PAGE == 0
    )


def _file(path: Path) -> "mx.SafetensorsFile":
    key = str(Path(path).resolve())
    f = _files.get(key)
    if f is None:
        f = _files[key] = mx.SafetensorsFile(key, zero_copy=True)
    return f


def _make(source: _Source) -> mx.array:
    f, keys, part = source
    if len(keys) > 1:
        return f.get_rows(list(keys))
    array = f[keys[0]]
    if part is not None:
        i, n = part
        array = mx.split(array, n, axis=0)[i]
    return array


def _registered(source: _Source) -> mx.array:
    key = (id(source[0]), source[1], source[2])
    array = _views.get(key)
    if array is None:
        array = _make(source)
        _views[key] = array
        _origin[id(array)] = (array, source)
    return array


def load(path: Path) -> dict[str, mx.array]:
    """All tensors of a page-aligned safetensors file; those MLX maps (they start on a page) are zero-copy views of
    one read-only mapping, the small rest is read."""
    f = _file(path)
    header, start = read_header(path)
    size = os.path.getsize(path)
    out = {}
    for key in f.keys():
        a, b = header[key]["data_offsets"]
        padded = -(-(b - a) // PAGE) * PAGE
        if f.mapped and b > a and (start + a) % PAGE == 0 and start + a + padded <= size:
            out[key] = _registered((f, (key,), None))
        else:
            out[key] = f[key]
    return out


# id(module) -> [(parameter path, source)] of its zero-copy parameters
_streamed: dict[int, list[tuple]] = {}


def _placeholders(specs: list[tuple]) -> list[tuple[str, mx.array]]:
    return [(path, mx.zeros((0,), dtype)) for path, _, dtype in specs]


@contextmanager
def streamed(module):
    """Give `module` its zero-copy weights for the duration of the block (streaming mode only).

    The caller must evaluate everything it needs from the module inside the block, or the weights stay
    alive (correct, but not released) until it does."""
    if not streaming():
        yield
        return
    from mlx.utils import tree_flatten, tree_unflatten

    specs = _streamed.get(id(module))
    if specs is None:
        specs = []
        for path, array in tree_flatten(module.parameters()):
            entry = _origin.get(id(array))
            if entry is not None and entry[0] is array:
                source = entry[1]
                specs.append((path, source, array.dtype))
                _origin.pop(id(array))
                _views.pop((id(source[0]), source[1], source[2]), None)
        _streamed[id(module)] = specs
    views = []
    for _, source, _ in specs:
        view = _make(source)
        _origin[id(view)] = (view, source)  # so adjacent_view can pack them; dropped again on exit
        views.append(view)
    module.update(tree_unflatten([(spec[0], view) for spec, view in zip(specs, views)]), strict=False)
    try:
        yield
    finally:
        module.update(tree_unflatten(_placeholders(specs)), strict=False)
        for view in views:
            _origin.pop(id(view), None)


def adjacent_view(weights: list[mx.array], cache: bool = True) -> mx.array | None:
    """The row-wise concatenation of `weights` as a view, when they are back-to-back in one mapping: separate
    tensors written next to each other (QKV, see `align`), or the row blocks of one tensor that `split_rows` made.

    cache=False returns a fresh view that nothing else keeps alive (for streamed layers)."""
    entries = [_origin.get(id(w)) for w in weights]
    if any(e is None or e[0] is not w for e, w in zip(entries, weights)):
        return None
    sources = [e[1] for e in entries]
    f = sources[0][0]
    if any(s[0] is not f for s in sources):
        return None
    if all(s[2] is not None for s in sources):
        # all row blocks of one tensor, in order: the tensor itself
        key, n = sources[0][1], sources[0][2][1]
        if len(sources) != n or any(s[1] != key or s[2] != (i, n) for i, s in enumerate(sources)):
            return None
        source = (f, key, None)
    elif all(s[2] is None and len(s[1]) == 1 for s in sources):
        source = (f, tuple(s[1][0] for s in sources), None)
        if f.get_rows(list(source[1])) is None:     # not back to back in the file
            return None
    else:
        return None
    return _registered(source) if cache else _make(source)


def split_rows(packed: mx.array, parts: int) -> list[mx.array]:
    """mx.split(packed, parts, axis=0), as registered views when `packed` came from this module, so the next
    generation packs the same weights without a copy again."""
    entry = _origin.get(id(packed))
    if entry is None or entry[0] is not packed:
        return mx.split(packed, parts, axis=0)
    f, keys, part = entry[1]
    if part is not None:
        return mx.split(packed, parts, axis=0)
    if len(keys) == parts:          # a packed group: its own tensors
        return [_registered((f, (k,), None)) for k in keys]
    if len(keys) == 1:              # row blocks of one tensor
        return [_registered((f, keys, (i, parts))) for i in range(parts)]
    return mx.split(packed, parts, axis=0)


def _contiguous(path: Path) -> bool:
    header, _ = read_header(path)
    spans = sorted(s["data_offsets"] for k, s in header.items() if k != "__metadata__")
    return spans[0][0] == 0 and all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


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
    if len(shards) == 1 and page_aligned(shards[0]) and _contiguous(shards[0]):
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
    # Safetensors readers require the tensors back to back with no gaps, so alignment comes from the order:
    # tensors that are whole pages first (each then starts on a page), the small rest at the end.
    def size(key):
        a, b = sources[key][2]["data_offsets"]
        return b - a

    ordered = _order(list(sources))
    keys = [k for k in ordered if size(k) % PAGE == 0] + [k for k in ordered if size(k) % PAGE]
    header, offset = {}, 0
    for key in keys:
        spec = sources[key][2]
        header[key] = {"dtype": spec["dtype"], "shape": spec["shape"], "data_offsets": [offset, offset + size(key)]}
        offset += size(key)
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
