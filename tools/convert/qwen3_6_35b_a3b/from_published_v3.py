"""Re-frame the published Qwen3.6-35B-A3B container-v3 artifact as this runtime's v2 artifact.

Canonical invocation::

    python -m tools.convert.qwen3_6_35b_a3b.from_published_v3 \
      --published /path/to/qwen3_6_35b_a3b.ninfer \
      --out out/qwen3_6_35b_a3b.ninfer

The published ``neroued/Qwen3.6-35B-A3B-NInfer`` artifact (sha256 pinned below) stores the same
940 objects as this target's v2 contract -- identical fused tensors, numeric formats, row-split
layouts, and resource bytes -- but frames them as container version 3: objects are named by id
and logical names live in a binding table of row ranges. This tool proves, for every v2 object,
that the v3 binding parts it fuses cover one v3 object in exactly the v2 row order, shape, format,
and encoded size, then copies the payload bytes unchanged into a v2 container. No value is
decoded or re-encoded.

One resource differs by design: the published ``chat_template.jinja`` is a later NInfer revision
of the Qwen template, while this runtime's frontend admits the official Qwen3.6 template. The
tool therefore packages the official template carried verbatim in the same artifact's
``tokenizer_config.json`` and checks it against that pinned digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import struct
from pathlib import Path
from typing import Sequence

from tools.artifact.container import ArtifactIdentity, ArtifactWriter
from tools.artifact.layouts import encoded_size
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.artifact.container import ResourceSpec as StoredResourceSpec

from . import inventory

PUBLISHED_REPOSITORY = "neroued/Qwen3.6-35B-A3B-NInfer"
PUBLISHED_REVISION = "ee4495803bc4f8015b8a7e22d4cf9b67de8e27c6"
PUBLISHED_SHA256 = "3e33297645dc33557751be1a3c407a74ed7c00f34909b5d4e8cfdce91b3dbe84"
PUBLISHED_BYTES = 22790484480

_V3_MAGIC = b"NINFER\x00\x03"
# sha256 of the official Qwen3.6 chat template this runtime's frontend admits.
OFFICIAL_TEMPLATE_SHA256 = "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
_FORMATS = {
    "bf16": "BF16",
    "fp32": "FP32",
    "int32": "I32",
    "q4_g64_fp16": "Q4G64_F16S",
    "q5_g64_fp16": "Q5G64_F16S",
    "q6_g64_fp16": "Q6G64_F16S",
    "q8_g32_fp16": "W8G32_F16S",
}
_LAYOUTS = {"contiguous_le_v1": "contiguous-le-v1", "row_split_k128_v1": "row-split-k128-v1"}
_RESOURCES = {
    "frontend/tokenizer.json": "resource/text/tokenizer.json",
    "frontend/tokenizer_config.json": "resource/text/tokenizer_config.json",
    "frontend/chat_template.jinja": "resource/text/chat_template.jinja",
    "frontend/generation_config.json": "resource/text/generation_config.json",
    "frontend/preprocessor_config.json": "resource/vision/preprocessor_config.json",
    "frontend/video_preprocessor_config.json": "resource/vision/video_preprocessor_config.json",
}


class PublishedArtifact:
    def __init__(self, path: Path):
        self.path = path
        self._file = path.open("rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        if self._map[:8] != _V3_MAGIC:
            raise ValueError(f"{path} is not a NInfer container-v3 artifact")
        header_bytes = struct.unpack_from("<Q", self._map, 8)[0]
        self.header = json.loads(self._map[32:32 + header_bytes])
        self.payload_base = 32 + header_bytes
        payload_bytes = self.header["files"][0]["payload_bytes"]
        if self.payload_base + payload_bytes != len(self._map):
            raise ValueError("container-v3 payload does not end at the end of the file")
        self.objects = {obj["id"]: obj for obj in self.header["objects"]}
        self.bindings = self.header["bindings"]

    def payload(self, object_id: str) -> memoryview:
        obj = self.objects[object_id]
        begin = self.payload_base + obj["offset"]
        return memoryview(self._map)[begin:begin + obj["bytes"]]


def _v3_parts(v2_name: str) -> list[str]:
    """v3 binding names whose row ranges, in order, form the v2 object."""

    prefix, _, leaf = v2_name.rpartition("/")
    if v2_name == "text/draft_head":
        return ["proposal/head"]
    if v2_name == "text/draft_head_token_ids":
        return ["proposal/token_ids"]
    if v2_name.startswith("mtp/layer/"):
        v2_name = v2_name.replace("mtp/layer/", "mtp/layers/0/", 1)
        prefix, _, leaf = v2_name.rpartition("/")
    if prefix.endswith("/gdn") and leaf == "a_b_projection":
        return [f"{prefix}/a_projection", f"{prefix}/b_projection"]
    if prefix.endswith("/gdn") and leaf == "query_key_value_z":
        return [f"{prefix}/{part}" for part in ("query", "key", "value", "z")]
    if prefix.endswith("/attention") and leaf == "query_key_gate_value":
        return [f"{prefix}/{part}" for part in ("query", "key", "gate", "value")]
    if prefix.endswith("/attention") and leaf == "query_key_value":
        return [f"{prefix}/{part}" for part in ("query", "key", "value")]
    if prefix.endswith("/attention") and leaf == "qkv":
        return [f"{prefix}/{part}" for part in ("query", "key", "value")]
    if prefix.endswith("/attention") and leaf == "qkv_bias":
        return [f"{prefix}/{part}_bias" for part in ("query", "key", "value")]
    if prefix.endswith("/moe") and leaf == "router_shared_gate":
        return [f"{prefix}/router", f"{prefix}/shared_score"]
    if prefix.endswith("/moe") and leaf == "routed_gate_up":
        return [f"{prefix}/experts/{e}/{p}" for e in range(256) for p in ("gate", "up")]
    if prefix.endswith("/moe") and leaf == "routed_down":
        return [f"{prefix}/experts/{e}/down" for e in range(256)]
    if prefix.endswith("/moe") and leaf == "shared_gate_up":
        return [f"{prefix}/shared/gate", f"{prefix}/shared/up"]
    if prefix.endswith("/moe") and leaf == "shared_down":
        return [f"{prefix}/shared/down"]
    if prefix.endswith("/mlp") and leaf == "gate_up":
        return [f"{prefix}/gate", f"{prefix}/up"]
    if v2_name.startswith("vision/") and prefix.rsplit("/", 1)[-1] in ("norm1", "norm2", "norm"):
        # v2 vision/layers/N/norm1/weight == v3 vision/layers/N/norm1_weight (merger: norm_weight).
        return [f"{prefix}_{leaf}"]
    return [v2_name]


def resolve_object(published: PublishedArtifact, spec) -> str:
    """The v3 object id holding v2 tensor `spec`, after proving it is exactly that tensor."""

    parts = _v3_parts(spec.name)
    rows, columns = (spec.shape[0], 1) if len(spec.shape) == 1 else (spec.shape[0], spec.shape[-1])
    total = 1
    for dim in spec.shape:
        total *= dim
    object_id = None
    cursor = 0
    for name in parts:
        binding = published.bindings.get(name)
        if binding is None:
            raise ValueError(f"{spec.name}: published artifact has no binding {name!r}")
        if "object" in binding:
            if len(parts) != 1:
                raise ValueError(f"{spec.name}: whole-object binding {name!r} inside a fused tensor")
            object_id, begin, end = binding["object"], 0, total
        else:
            if len(binding["parts"]) != 1:
                raise ValueError(f"{spec.name}: binding {name!r} spans several objects")
            part = binding["parts"][0]
            begin, end = part["range"]
            if object_id is None:
                object_id = part["object"]
            elif part["object"] != object_id:
                raise ValueError(f"{spec.name}: fused parts come from different objects")
        if begin != cursor or end <= begin or (columns > 1 and (begin % columns or end % columns)):
            raise ValueError(f"{spec.name}: part {name!r} is not the next whole-row range")
        cursor = end
    if cursor != total:
        raise ValueError(f"{spec.name}: fused parts cover {cursor} of {total} elements")
    obj = published.objects[object_id]
    if obj["kind"] != "tensor" or tuple(obj["shape"]) != tuple(spec.shape):
        raise ValueError(f"{spec.name}: published object {object_id} has shape {obj.get('shape')}")
    if _FORMATS[obj["format"]] != spec.format or _LAYOUTS[obj["layout"]] != spec.layout:
        raise ValueError(f"{spec.name}: published object {object_id} is {obj['format']}/{obj['layout']}")
    if obj["bytes"] != encoded_size(spec.layout, spec.format, spec.shape):
        raise ValueError(f"{spec.name}: published object {object_id} has {obj['bytes']} bytes")
    del rows
    return object_id


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(64 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def convert(published_path: Path, out_path: Path, *, verify_hash: bool = True) -> None:
    if published_path.stat().st_size != PUBLISHED_BYTES:
        raise ValueError(f"{published_path} is not the pinned {PUBLISHED_REPOSITORY} artifact (size)")
    if verify_hash and file_sha256(published_path) != PUBLISHED_SHA256:
        raise ValueError(f"{published_path} does not match the pinned sha256 {PUBLISHED_SHA256}")
    published = PublishedArtifact(published_path)
    resources = {
        spec.name: bytes(published.payload(_RESOURCES[spec.name])) for spec in inventory.RESOURCE_SPECS
    }
    official_template = json.loads(resources["frontend/tokenizer_config.json"])["chat_template"].encode()
    if hashlib.sha256(official_template).hexdigest() != OFFICIAL_TEMPLATE_SHA256:
        raise ValueError("tokenizer_config.json does not carry the official Qwen3.6 chat template")
    resources["frontend/chat_template.jinja"] = official_template
    plan = family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)
    sources: dict[str, str] = {}
    used: set[str] = set(_RESOURCES.values())
    for spec in inventory.TENSOR_SPECS:
        object_id = resolve_object(published, spec)
        if object_id in used:
            raise ValueError(f"{spec.name}: published object {object_id} is already mapped")
        used.add(object_id)
        sources[spec.name] = object_id
    if used != set(published.objects):
        raise ValueError(f"{len(set(published.objects) - used)} published objects have no v2 object")
    print(f"mapped all {len(plan.specs)} objects; writing {out_path}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ArtifactWriter(out_path, ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID), plan.specs) as writer:
        for index, spec in enumerate(plan.specs, 1):
            if isinstance(spec, StoredResourceSpec):
                writer.write(spec.name, resources[spec.name])
            else:
                writer.write(spec.name, published.payload(sources[spec.name]))
            if index % 100 == 0 or index == len(plan.specs):
                print(f"[{index}/{len(plan.specs)}] {spec.name}", flush=True)
    print(f"complete: {out_path.stat().st_size} bytes", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--published", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--skip-source-hash", action="store_true",
                        help="skip the sha256 check of the 22.8 GB published artifact (size is still checked)")
    args = parser.parse_args(argv)
    convert(args.published, args.out, verify_hash=not args.skip_source_hash)


if __name__ == "__main__":
    main()
