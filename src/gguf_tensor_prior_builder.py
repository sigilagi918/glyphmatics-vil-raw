#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

HOME = Path.home()

GGUF_VALUE_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

GGML_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    31: "Q4_0_4_4",
    32: "Q4_0_4_8",
    33: "Q4_0_8_8",
    34: "TQ1_0",
    35: "TQ2_0",
}

TEXT_EXT = ".gguf"

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha_file_stream(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

class Reader:
    def __init__(self, path: Path):
        self.path = path
        self.f = path.open("rb")
        self.size = path.stat().st_size

    def close(self):
        self.f.close()

    def pos(self) -> int:
        return self.f.tell()

    def read(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"Unexpected EOF while reading {n} bytes at offset {self.pos()}")
        return b

    def u8(self) -> int:
        return struct.unpack("<B", self.read(1))[0]

    def i8(self) -> int:
        return struct.unpack("<b", self.read(1))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.read(2))[0]

    def i16(self) -> int:
        return struct.unpack("<h", self.read(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.read(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.read(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.read(8))[0]

    def gguf_string(self) -> str:
        n = self.u64()
        if n > 256 * 1024 * 1024:
            raise ValueError(f"String length too large: {n}")
        return self.read(n).decode("utf-8", errors="replace")

    def skip_value_payload(self, typ: int) -> Any:
        if typ == 0:
            return self.u8()
        if typ == 1:
            return self.i8()
        if typ == 2:
            return self.u16()
        if typ == 3:
            return self.i16()
        if typ == 4:
            return self.u32()
        if typ == 5:
            return self.i32()
        if typ == 6:
            return self.f32()
        if typ == 7:
            return bool(self.u8())
        if typ == 8:
            return self.gguf_string()
        if typ == 10:
            return self.u64()
        if typ == 11:
            return self.i64()
        if typ == 12:
            return self.f64()
        if typ == 9:
            elem_type = self.u32()
            length = self.u64()
            values = []
            keep = min(length, 32)
            for i in range(length):
                v = self.skip_value_payload(elem_type)
                if i < keep:
                    values.append(v)
            return {
                "array_type": GGUF_VALUE_TYPES.get(elem_type, f"unknown_{elem_type}"),
                "length": length,
                "sample": values,
                "truncated": length > keep,
            }
        raise ValueError(f"Unsupported GGUF metadata type: {typ}")

def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(HOME))
    except Exception:
        return str(path)

def parse_gguf(path: Path, hash_file: bool = False) -> Dict[str, Any]:
    r = Reader(path)
    try:
        magic = r.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not GGUF magic: {magic!r}")

        version = r.u32()
        tensor_count = r.u64()
        kv_count = r.u64()

        metadata = {}
        metadata_types = {}
        alignment = 32

        for _ in range(kv_count):
            key = r.gguf_string()
            typ = r.u32()
            value = r.skip_value_payload(typ)
            metadata[key] = value
            metadata_types[key] = GGUF_VALUE_TYPES.get(typ, f"unknown_{typ}")

            if key == "general.alignment":
                try:
                    alignment = int(value)
                except Exception:
                    alignment = 32

        tensors = []
        type_counts = Counter()
        rank_counts = Counter()
        name_prefix_counts = Counter()
        shape_counts = Counter()

        for _ in range(tensor_count):
            name = r.gguf_string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            ggml_type_id = r.u32()
            offset = r.u64()

            ggml_type = GGML_TYPES.get(ggml_type_id, f"UNKNOWN_{ggml_type_id}")

            prefix = name.split(".")[0] if "." in name else name
            shape_key = "x".join(str(x) for x in dims)

            type_counts[ggml_type] += 1
            rank_counts[str(n_dims)] += 1
            name_prefix_counts[prefix] += 1
            shape_counts[shape_key] += 1

            tensors.append({
                "name": name,
                "rank": n_dims,
                "shape": dims,
                "type_id": ggml_type_id,
                "type": ggml_type,
                "offset": offset,
                "prefix": prefix,
                "shape_key": shape_key,
            })

        tensor_info_end = r.pos()
        data_start = tensor_info_end
        if alignment > 0:
            data_start = ((tensor_info_end + alignment - 1) // alignment) * alignment

        record = {
            "path": safe_rel(path),
            "file": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha_file_stream(path) if hash_file else None,
            "magic": "GGUF",
            "version": version,
            "tensor_count": tensor_count,
            "metadata_kv_count": kv_count,
            "alignment": alignment,
            "tensor_info_end": tensor_info_end,
            "data_start_estimated": data_start,
            "metadata": metadata,
            "metadata_types": metadata_types,
            "tensor_type_counts": dict(type_counts),
            "tensor_rank_counts": dict(rank_counts),
            "tensor_prefix_counts": dict(name_prefix_counts),
            "tensor_shape_counts_top": dict(shape_counts.most_common(100)),
            "tensors": tensors,
        }
        return record
    finally:
        r.close()

def glyph_token(i: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    a = alphabet[(i // 26) % 26]
    b = alphabet[i % 26]
    return f"Γ{a}{b}"

def build_prior(records: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_keys = Counter()
    tensor_names = Counter()
    tensor_prefixes = Counter()
    tensor_shapes = Counter()
    tensor_types = Counter()
    model_arches = Counter()

    for rec in records:
        for k in rec.get("metadata", {}).keys():
            metadata_keys[k] += 1

        arch = rec.get("metadata", {}).get("general.architecture")
        if isinstance(arch, str):
            model_arches[arch] += 1

        for t in rec.get("tensors", []):
            tensor_names[t["name"]] += 1
            tensor_prefixes[t["prefix"]] += 1
            tensor_shapes[t["shape_key"]] += 1
            tensor_types[t["type"]] += 1

    dictionary = []
    candidates: List[Tuple[str, str, int]] = []

    for k, c in metadata_keys.items():
        if c >= 1:
            candidates.append(("metadata_key", k, c))
    for k, c in tensor_prefixes.items():
        if c >= 1:
            candidates.append(("tensor_prefix", k, c))
    for k, c in tensor_shapes.items():
        if c >= 1:
            candidates.append(("tensor_shape", k, c))
    for k, c in tensor_types.items():
        if c >= 1:
            candidates.append(("tensor_type", k, c))
    for k, c in tensor_names.items():
        if c >= 2:
            candidates.append(("tensor_name", k, c))

    candidates.sort(key=lambda x: (x[2], len(x[1])), reverse=True)

    for i, (kind, value, count) in enumerate(candidates[:4096]):
        dictionary.append({
            "glyph": glyph_token(i),
            "kind": kind,
            "value": value,
            "count": count,
            "value_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        })

    total_model_bytes = sum(r["size_bytes"] for r in records)

    glyphlines = []
    glyphlines.append("pGGUFZvhz")
    glyphlines.append("MODE GGUF_STRUCTURAL_TENSOR_PRIOR")
    glyphlines.append("RULE METADATA_AND_TENSOR_STRUCTURE_NOT_FULL_BYTES")
    glyphlines.append("NO_BASE64")
    glyphlines.append("NO_BYTE_TO_BRAILLE")
    glyphlines.append("NO_BYTE_TO_HANZI")
    glyphlines.append(f"FILES {len(records)}")
    glyphlines.append(f"DICT {len(dictionary)}")
    glyphlines.append(f"MODEL_BYTES {total_model_bytes}")

    for d in dictionary:
        glyphlines.append(json.dumps(d, ensure_ascii=False, sort_keys=True))

    glyphline_text = "\n".join(glyphlines) + "\n"
    glyphline_bytes = len(glyphline_text.encode("utf-8"))
    ratio = total_model_bytes / glyphline_bytes if glyphline_bytes else 0.0
    reduction = 100.0 - ((glyphline_bytes / total_model_bytes) * 100.0) if total_model_bytes else 0.0

    dataset = {
        "format": "GlyphMatics GGUF Tensor Structural Prior",
        "version": "0.8.0-dev",
        "created_at": int(time.time()),
        "scope": "GGUF metadata, tensor names, tensor shapes, quantization type distribution, and offsets",
        "boundary": "This is structural prior compression, not full GGUF byte-lossless tensor reconstruction.",
        "records": records,
        "dictionary": dictionary,
        "summary": {
            "file_count": len(records),
            "total_model_bytes": total_model_bytes,
            "dictionary_entries": len(dictionary),
            "glyphline_bytes": glyphline_bytes,
            "semantic_structure_ratio_vs_model_bytes": ratio,
            "semantic_structure_reduction_percent": reduction,
            "architectures": dict(model_arches),
            "tensor_types": dict(tensor_types),
            "top_tensor_prefixes": dict(tensor_prefixes.most_common(100)),
            "top_tensor_shapes": dict(tensor_shapes.most_common(100)),
            "top_metadata_keys": dict(metadata_keys.most_common(100)),
        },
    }

    raw = json.dumps(dataset, sort_keys=True, ensure_ascii=False)
    dataset["summary"]["dataset_digest"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    dataset_path = out_dir / "gguf_tensor_structural_prior.json"
    glyphline_path = out_dir / "gguf_tensor_structural_prior.glyphlines.txt"
    summary_path = out_dir / "summary.json"

    dataset_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    glyphline_path.write_text(glyphline_text, encoding="utf-8")
    summary_path.write_text(json.dumps(dataset["summary"], indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "status": "built",
        "dataset": str(dataset_path),
        "glyphlines": str(glyphline_path),
        "summary": str(summary_path),
        **dataset["summary"],
    }

def find_gguf_files(roots: List[Path]) -> List[Path]:
    found = []
    seen = set()

    for root in roots:
        root = root.expanduser()
        if not root.exists():
            continue

        if root.is_file() and root.suffix.lower() == ".gguf":
            files = [root]
        else:
            files = list(root.rglob("*.gguf"))

        for p in files:
            try:
                rp = p.resolve()
                if str(rp) not in seen and rp.is_file():
                    seen.add(str(rp))
                    found.append(rp)
            except Exception:
                pass

    return sorted(found, key=lambda p: p.stat().st_size, reverse=True)

def main() -> None:
    ap = argparse.ArgumentParser(description="Build GlyphMatics GGUF tensor structural prior from GGUF metadata and tensor table.")
    ap.add_argument("--root", action="append", default=[], help="Root folder or .gguf file. Can repeat.")
    ap.add_argument("--out", default=str(HOME / "glyphmatics_gguf_tensor_prior"))
    ap.add_argument("--limit", type=int, default=0, help="Max GGUF files to parse. 0 = all.")
    ap.add_argument("--hash-file", action="store_true", help="Hash full GGUF files. Slower for large models.")
    args = ap.parse_args()

    roots = [Path(x).expanduser() for x in args.root] if args.root else [HOME / "models"]
    files = find_gguf_files(roots)

    if args.limit and args.limit > 0:
        files = files[:args.limit]

    if not files:
        raise SystemExit("No .gguf files found. Use --root /path/to/models or --root model.gguf")

    records = []
    failures = []

    for p in files:
        try:
            print(f"[GGUF] parsing {p}")
            records.append(parse_gguf(p, hash_file=args.hash_file))
        except Exception as e:
            failures.append({"file": str(p), "error": str(e)})
            print(f"[WARN] failed {p}: {e}")

    result = build_prior(records, Path(args.out).expanduser())
    result["failures"] = failures

    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
