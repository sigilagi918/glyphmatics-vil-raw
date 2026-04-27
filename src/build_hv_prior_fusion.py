#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import struct
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

HOME = Path.home()

DEFAULT_ROOTS = [
    HOME / "kaggle",
    HOME / "arc3_glyphmatic",
    HOME / "arcagi3_glyph_encoded",
    HOME / "glyphmatics-vil-raw",
    HOME / "vil-canonical-glyph-system",
    HOME / "vil-canonical-glyph-system-repo",
    HOME / "vil-glyphmatic-demo",
    HOME / "sigilagi",
    HOME / "sigilagi_real",
    HOME / "llama.cpp",
    HOME / "models",
    HOME / "qwen",
]

TEXT_EXTS = {
    ".py", ".ipynb", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".sh", ".bash", ".zsh", ".c", ".h", ".cpp",
    ".hpp", ".rs", ".go", ".js", ".ts", ".html", ".css", ".java", ".kt",
    ".sql", ".xml",
}

WEIGHT_EXTS = {
    ".gguf", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx",
    ".npz", ".npy", ".tflite",
}

SKIP_PARTS = {
    ".git", ".cache", "__pycache__", "node_modules", ".venv", "venv",
    "tok-venv", "vilenv", "site-packages", ".mypy_cache", ".pytest_cache",
}

MAX_TEXT_BYTES = 3_000_000

HORIZONTAL_LAYERS = {
    "data_ingest": ["dataset", "crawler", "scrape", "download", "ingest", "loader", "corpus", "shard"],
    "tokenization": ["tokenizer", "sentencepiece", "bpe", "token", "vocab", "merge"],
    "model_weights": ["gguf", "safetensors", "model", "weights", "checkpoint", "qwen", "llama", "lora"],
    "inference_runtime": ["inference", "runner", "llama.cpp", "server", "generate", "chat", "predict"],
    "agent_runtime": ["agent", "planner", "executor", "tool", "action", "runtime", "orchestrator"],
    "benchmark_solver": ["arc", "agi", "kaggle", "submission", "score", "solver", "eval"],
    "vector_memory": ["vector", "embedding", "rag", "index", "search", "retrieval"],
    "visual_artifact": ["image", "png", "pgm", "ocr", "visual", "canvas", "render"],
    "tensor_network": ["tensor", "mps", "quantum", "grover", "qubit", "gguf", "matrix"],
    "deployment": ["docker", "systemd", "service", "github", "huggingface", "space", "deploy", "pages"],
    "ui_web": ["html", "css", "js", "react", "ui", "web", "page"],
    "security_verification": ["verify", "hash", "sha", "digest", "lock", "seal", "lineage", "proof"],
}

VERTICAL_STAGES = [
    "source",
    "ingest",
    "normalize",
    "tokenize",
    "train",
    "weights",
    "inference",
    "agent",
    "evaluate",
    "package",
    "deploy",
    "verify",
    "visualize",
]

def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def sha_file_sample(path: Path, max_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()

def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(HOME))
    except Exception:
        return str(path)

def should_skip(path: Path) -> bool:
    return bool(set(path.parts) & SKIP_PARTS)

def read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="replace")

def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.rstrip()).strip()

def classify_horizontal(path: Path, text_sample: str = "") -> List[str]:
    s = (str(path) + "\n" + text_sample[:8000]).lower()
    layers = []
    for layer, keys in HORIZONTAL_LAYERS.items():
        if any(k in s for k in keys):
            layers.append(layer)
    return layers or ["general"]

def classify_vertical_stage(path: Path, text_sample: str = "") -> List[str]:
    s = (str(path) + "\n" + text_sample[:8000]).lower()
    stages = []

    tests = {
        "source": ["src/", ".py", ".c", ".cpp", ".rs", ".go", ".js", ".ts"],
        "ingest": ["ingest", "download", "scrape", "crawler", "loader", "dataset"],
        "normalize": ["normalize", "clean", "canonical", "schema", "preprocess"],
        "tokenize": ["tokenizer", "tokenize", "bpe", "sentencepiece", "vocab"],
        "train": ["train", "trainer", "fit", "epoch", "loss", "optimizer"],
        "weights": ["gguf", "safetensors", ".pt", ".pth", ".bin", "weights", "checkpoint"],
        "inference": ["inference", "generate", "predict", "runner", "chat", "server"],
        "agent": ["agent", "planner", "executor", "action", "tool"],
        "evaluate": ["eval", "score", "benchmark", "test", "verify"],
        "package": ["pack", "artifact", "gma", "bundle", "archive"],
        "deploy": ["github", "huggingface", "space", "deploy", "pages", "service"],
        "verify": ["verify", "hash", "digest", "seal", "lock", "lineage", "proof"],
        "visualize": ["image", "png", "pgm", "visual", "render", "canvas", "ocr"],
    }

    for stage, keys in tests.items():
        if any(k in s for k in keys):
            stages.append(stage)

    return stages or ["source"]

def project_key(path: Path) -> str:
    rel = safe_rel(path)
    parts = rel.split("/")
    return parts[0] if parts else "unknown"

def ast_signature(code: str) -> Dict[str, Any]:
    sig = {
        "imports": [],
        "functions": [],
        "classes": [],
        "calls": Counter(),
    }

    try:
        tree = ast.parse(code)
    except Exception:
        return sig

    def call_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = call_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                sig["imports"].append(n.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for n in node.names:
                sig["imports"].append(f"{mod}.{n.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig["functions"].append(node.name)
        elif isinstance(node, ast.ClassDef):
            sig["classes"].append(node.name)
        elif isinstance(node, ast.Call):
            n = call_name(node.func)
            if n:
                sig["calls"][n] += 1

    sig["imports"] = sorted(set(sig["imports"]))
    sig["functions"] = sorted(set(sig["functions"]))
    sig["classes"] = sorted(set(sig["classes"]))
    sig["calls"] = dict(sig["calls"].most_common(100))
    return sig

def extract_notebook_code(text: str) -> str:
    try:
        obj = json.loads(text)
    except Exception:
        return text

    if not isinstance(obj, dict) or "cells" not in obj:
        return text

    blocks = []
    for cell in obj.get("cells", []):
        if cell.get("cell_type") == "code":
            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            src = str(src).strip()
            if src:
                blocks.append(src)
    return "\n\n".join(blocks)

def parse_safetensors_header(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("too small")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len > 64 * 1024 * 1024:
            raise ValueError("safetensors header too large")
        header = f.read(header_len).decode("utf-8", errors="replace")
        obj = json.loads(header)

    tensors = []
    metadata = obj.get("__metadata__", {})
    dtype_counts = Counter()
    shape_counts = Counter()

    for name, info in obj.items():
        if name == "__metadata__":
            continue
        dtype = info.get("dtype")
        shape = info.get("shape", [])
        offsets = info.get("data_offsets", [])
        dtype_counts[str(dtype)] += 1
        shape_key = "x".join(map(str, shape))
        shape_counts[shape_key] += 1
        tensors.append({
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "shape_key": shape_key,
            "data_offsets": offsets,
        })

    return {
        "format": "safetensors",
        "metadata": metadata,
        "tensor_count": len(tensors),
        "dtype_counts": dict(dtype_counts),
        "shape_counts_top": dict(shape_counts.most_common(100)),
        "tensors": tensors[:1000],
        "truncated_tensors": len(tensors) > 1000,
    }

def parse_npz_header(path: Path) -> Dict[str, Any]:
    # Does not load array values; reads zip member names/sizes.
    with zipfile.ZipFile(path, "r") as z:
        members = []
        for info in z.infolist():
            members.append({
                "name": info.filename,
                "compressed_size": info.compress_size,
                "file_size": info.file_size,
            })
    return {
        "format": "npz",
        "members": members,
        "member_count": len(members),
    }

def parse_gguf_struct(path: Path) -> Dict[str, Any]:
    # Reuse v0.8 parser if available.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from gguf_tensor_prior_builder import parse_gguf
        rec = parse_gguf(path, hash_file=False)
        return {
            "format": "gguf",
            "version": rec.get("version"),
            "tensor_count": rec.get("tensor_count"),
            "metadata_kv_count": rec.get("metadata_kv_count"),
            "alignment": rec.get("alignment"),
            "metadata_keys": list(rec.get("metadata", {}).keys()),
            "tensor_type_counts": rec.get("tensor_type_counts"),
            "tensor_rank_counts": rec.get("tensor_rank_counts"),
            "tensor_prefix_counts": rec.get("tensor_prefix_counts"),
            "tensor_shape_counts_top": rec.get("tensor_shape_counts_top"),
            "tensors_sample": rec.get("tensors", [])[:250],
            "truncated_tensors": len(rec.get("tensors", [])) > 250,
        }
    except Exception as e:
        return {
            "format": "gguf",
            "parse_error": str(e),
        }

def parse_weight(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    base = {
        "path": safe_rel(path),
        "size_bytes": path.stat().st_size,
        "extension": ext,
        "sample_sha256_first_1mb": sha_file_sample(path),
        "note": "structural metadata only; raw weights are not copied",
    }

    try:
        if ext == ".gguf":
            base["structure"] = parse_gguf_struct(path)
        elif ext == ".safetensors":
            base["structure"] = parse_safetensors_header(path)
        elif ext == ".npz":
            base["structure"] = parse_npz_header(path)
        else:
            base["structure"] = {
                "format": ext.lstrip("."),
                "parse_mode": "filename_size_sample_hash_only",
            }
    except Exception as e:
        base["structure"] = {
            "format": ext.lstrip("."),
            "parse_error": str(e),
        }

    return base

def glyph_token(i: int, prefix: str) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    a = alphabet[(i // 26) % 26]
    b = alphabet[i % 26]
    return f"{prefix}{a}{b}"

def scan_roots(roots: List[Path], max_files: int = 0) -> Dict[str, Any]:
    records = []
    weight_records = []
    line_counter = Counter()
    import_counter = Counter()
    function_counter = Counter()
    class_counter = Counter()
    horizontal = defaultdict(list)
    vertical = defaultdict(lambda: defaultdict(list))

    seen = set()

    for root in roots:
        root = root.expanduser()
        if not root.exists():
            continue

        candidates = [root] if root.is_file() else list(root.rglob("*"))

        for path in candidates:
            if max_files and len(records) + len(weight_records) >= max_files:
                break

            if should_skip(path) or not path.is_file():
                continue

            ext = path.suffix.lower()
            rp = path.resolve()
            if str(rp) in seen:
                continue
            seen.add(str(rp))

            if ext in WEIGHT_EXTS:
                wr = parse_weight(path)
                layers = classify_horizontal(path, json.dumps(wr)[:8000])
                stages = classify_vertical_stage(path, json.dumps(wr)[:8000])
                wr["horizontal_layers"] = layers
                wr["vertical_stages"] = stages
                wr["project"] = project_key(path)
                weight_records.append(wr)

                for layer in layers:
                    horizontal[layer].append(wr["path"])
                for stage in stages:
                    vertical[wr["project"]][stage].append(wr["path"])
                continue

            if ext not in TEXT_EXTS:
                continue

            try:
                if path.stat().st_size > MAX_TEXT_BYTES:
                    continue
                text = read_text(path)
            except Exception:
                continue

            functional = extract_notebook_code(text) if ext == ".ipynb" else text
            sample = functional[:10000]
            layers = classify_horizontal(path, sample)
            stages = classify_vertical_stage(path, sample)
            proj = project_key(path)

            lines = [normalize_line(x) for x in functional.splitlines()]
            lines = [x for x in lines if x]

            for line in lines:
                if len(line) >= 8:
                    line_counter[line] += 1

            sig = ast_signature(functional) if ext in {".py", ".ipynb"} else {}
            for x in sig.get("imports", []):
                import_counter[x] += 1
            for x in sig.get("functions", []):
                function_counter[x] += 1
            for x in sig.get("classes", []):
                class_counter[x] += 1

            rec = {
                "path": safe_rel(path),
                "project": proj,
                "extension": ext,
                "size_bytes": path.stat().st_size,
                "functional_bytes": len(functional.encode("utf-8")),
                "functional_sha256": sha_text(functional),
                "horizontal_layers": layers,
                "vertical_stages": stages,
                "python_signature": sig,
            }
            records.append(rec)

            for layer in layers:
                horizontal[layer].append(rec["path"])
            for stage in stages:
                vertical[proj][stage].append(rec["path"])

    return {
        "records": records,
        "weights": weight_records,
        "line_counter": line_counter,
        "import_counter": import_counter,
        "function_counter": function_counter,
        "class_counter": class_counter,
        "horizontal": horizontal,
        "vertical": vertical,
    }

def build_dictionaries(scan: Dict[str, Any], max_dict: int) -> Dict[str, Any]:
    line_items = []
    for line, count in scan["line_counter"].items():
        if count >= 2:
            savings = (len(line) - 5) * (count - 1)
            if savings > 0:
                line_items.append((savings, count, line))
    line_items.sort(reverse=True, key=lambda x: (x[0], x[1], len(x[2])))

    horizontal_dict = []
    for i, (layer, paths) in enumerate(sorted(scan["horizontal"].items())):
        horizontal_dict.append({
            "glyph": glyph_token(i, "H"),
            "layer": layer,
            "count": len(paths),
            "sample_paths": paths[:50],
        })

    vertical_dict = []
    i = 0
    for project, stages in sorted(scan["vertical"].items()):
        for stage, paths in sorted(stages.items()):
            vertical_dict.append({
                "glyph": glyph_token(i, "V"),
                "project": project,
                "stage": stage,
                "count": len(paths),
                "sample_paths": paths[:50],
            })
            i += 1

    structural_dict = []
    for i, (_, count, line) in enumerate(line_items[:max_dict]):
        structural_dict.append({
            "glyph": glyph_token(i, "S"),
            "kind": "repeated_code_line",
            "count": count,
            "value": line,
            "value_sha256": sha_text(line),
        })

    weight_dict = []
    i = 0
    for wr in scan["weights"]:
        st = wr.get("structure", {})
        if st.get("format") == "gguf":
            for key, val in (st.get("tensor_type_counts") or {}).items():
                weight_dict.append({"glyph": glyph_token(i, "W"), "kind": "gguf_tensor_type", "value": key, "count": val})
                i += 1
            for key, val in (st.get("tensor_shape_counts_top") or {}).items():
                weight_dict.append({"glyph": glyph_token(i, "W"), "kind": "gguf_tensor_shape", "value": key, "count": val})
                i += 1
        elif st.get("format") == "safetensors":
            for key, val in (st.get("dtype_counts") or {}).items():
                weight_dict.append({"glyph": glyph_token(i, "W"), "kind": "safetensors_dtype", "value": key, "count": val})
                i += 1
            for key, val in (st.get("shape_counts_top") or {}).items():
                weight_dict.append({"glyph": glyph_token(i, "W"), "kind": "safetensors_shape", "value": key, "count": val})
                i += 1

    return {
        "horizontal_dictionary": horizontal_dict,
        "vertical_dictionary": vertical_dict,
        "structural_dictionary": structural_dict,
        "weight_dictionary": weight_dict[:max_dict],
    }

def build(roots: List[Path], out_dir: Path, max_dict: int, max_files: int) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scan = scan_roots(roots, max_files=max_files)
    dictionaries = build_dictionaries(scan, max_dict=max_dict)

    total_text_bytes = sum(r["functional_bytes"] for r in scan["records"])
    total_weight_bytes = sum(r["size_bytes"] for r in scan["weights"])
    total_observed_bytes = total_text_bytes + total_weight_bytes

    horizontal_prior = {
        "format": "GlyphMatics Horizontal Consolidation Prior",
        "version": "0.9.0-dev",
        "definition": "Groups competitor artifacts by equivalent functional layer across systems.",
        "layers": {k: v for k, v in sorted(scan["horizontal"].items())},
        "dictionary": dictionaries["horizontal_dictionary"],
    }

    vertical_prior = {
        "format": "GlyphMatics Vertical Integration Prior",
        "version": "0.9.0-dev",
        "definition": "Maps artifacts into source→ingest→tokenize→weights→inference→agent→deploy→verify stacks.",
        "projects": {p: dict(stages) for p, stages in sorted(scan["vertical"].items())},
        "dictionary": dictionaries["vertical_dictionary"],
    }

    weight_prior = {
        "format": "GlyphMatics Weight Structural Prior",
        "version": "0.9.0-dev",
        "definition": "Indexes public/owned model weight structures without copying raw tensors.",
        "boundary": "Structural metadata only; not byte-lossless full weight reconstruction.",
        "weights": scan["weights"],
        "dictionary": dictionaries["weight_dictionary"],
    }

    structural_prior = {
        "format": "GlyphMatics Cross-Competitor Structural Dictionary",
        "version": "0.9.0-dev",
        "definition": "Repeated public/owned code/document structures reduced to glyph dictionary.",
        "records": scan["records"],
        "dictionary": dictionaries["structural_dictionary"],
        "top_imports": scan["import_counter"].most_common(200),
        "top_functions": scan["function_counter"].most_common(200),
        "top_classes": scan["class_counter"].most_common(200),
    }

    glyphlines = [
        "pHVZvhz",
        "MODE HORIZONTAL_VERTICAL_PRIOR_FUSION",
        "RULE SHARED_PRIOR_NOT_RAW_ENTROPY",
        "NO_PRIVATE_DATA",
        "NO_STOLEN_REPOS",
        "NO_BASE64",
        "NO_HIDDEN_PAYLOAD",
        "NO_BYTE_TO_BRAILLE",
        "NO_BYTE_TO_HANZI",
        f"TEXT_FILES {len(scan['records'])}",
        f"WEIGHT_FILES {len(scan['weights'])}",
        f"TEXT_BYTES {total_text_bytes}",
        f"WEIGHT_BYTES {total_weight_bytes}",
        f"TOTAL_OBSERVED_BYTES {total_observed_bytes}",
        f"HORIZONTAL_GLYPHS {len(dictionaries['horizontal_dictionary'])}",
        f"VERTICAL_GLYPHS {len(dictionaries['vertical_dictionary'])}",
        f"STRUCTURAL_GLYPHS {len(dictionaries['structural_dictionary'])}",
        f"WEIGHT_GLYPHS {len(dictionaries['weight_dictionary'])}",
    ]

    for section_name in ["horizontal_dictionary", "vertical_dictionary", "structural_dictionary", "weight_dictionary"]:
        glyphlines.append(f"SECTION {section_name}")
        for item in dictionaries[section_name]:
            glyphlines.append(json.dumps(item, ensure_ascii=False, sort_keys=True))

    glyphline_text = "\n".join(glyphlines) + "\n"
    glyphline_bytes = len(glyphline_text.encode("utf-8"))

    ratio = total_observed_bytes / glyphline_bytes if glyphline_bytes else 0.0
    reduction = 100.0 - ((glyphline_bytes / total_observed_bytes) * 100.0) if total_observed_bytes else 0.0

    summary = {
        "format": "GlyphMatics HV Prior Fusion Summary",
        "version": "0.9.0-dev",
        "created_at": int(time.time()),
        "roots": [str(r) for r in roots],
        "text_file_count": len(scan["records"]),
        "weight_file_count": len(scan["weights"]),
        "total_text_bytes": total_text_bytes,
        "total_weight_bytes": total_weight_bytes,
        "total_observed_bytes": total_observed_bytes,
        "glyphline_bytes": glyphline_bytes,
        "semantic_structure_ratio_vs_observed_bytes": ratio,
        "semantic_structure_reduction_percent": reduction,
        "horizontal_glyphs": len(dictionaries["horizontal_dictionary"]),
        "vertical_glyphs": len(dictionaries["vertical_dictionary"]),
        "structural_glyphs": len(dictionaries["structural_dictionary"]),
        "weight_glyphs": len(dictionaries["weight_dictionary"]),
        "boundary": "Public/owned artifacts only. Weights are structurally indexed, not raw-copied.",
    }

    digest_source = json.dumps({
        "summary": summary,
        "horizontal_prior": horizontal_prior,
        "vertical_prior": vertical_prior,
        "weight_prior": weight_prior,
        "structural_prior": structural_prior,
    }, sort_keys=True, ensure_ascii=False)
    summary["dataset_digest"] = sha_text(digest_source)

    files = {
        "summary.json": summary,
        "horizontal_consolidation_prior.json": horizontal_prior,
        "vertical_integration_prior.json": vertical_prior,
        "weight_structural_prior.json": weight_prior,
        "cross_competitor_structural_prior.json": structural_prior,
    }

    for name, obj in files.items():
        (out_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    (out_dir / "hv_prior_fusion.glyphlines.txt").write_text(glyphline_text, encoding="utf-8")

    return {
        "status": "built",
        "out_dir": str(out_dir),
        **summary,
    }

def main() -> None:
    ap = argparse.ArgumentParser(description="Build GlyphMatics horizontal consolidation + vertical integration prior fusion.")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--out", default=str(HOME / "glyphmatics_hv_prior_fusion"))
    ap.add_argument("--max-dict", type=int, default=4096)
    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()

    roots = [Path(x).expanduser() for x in args.root] if args.root else DEFAULT_ROOTS
    result = build(roots, Path(args.out).expanduser(), args.max_dict, args.max_files)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
