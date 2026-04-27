#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import struct
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

HOME = Path.home()

CHAT_FAMILIES = {
    "llama": ["llama", "meta-llama", "llama.cpp", "llama3", "llama-3"],
    "qwen": ["qwen", "qwen2", "qwen3", "dashscope"],
    "deepseek": ["deepseek", "deepseek-r1", "deepseek-v3"],
    "mistral": ["mistral", "mixtral", "codestral", "ministral"],
    "gemma": ["gemma", "google/gemma"],
    "phi": ["phi", "phi-3", "phi-4", "microsoft/phi"],
    "grok": ["grok", "xai", "grok-1"],
    "claude_interface": ["claude", "anthropic"],
    "chatgpt_interface": ["chatgpt", "openai", "gpt-4", "gpt-5"],
    "gemini_interface": ["gemini", "google-ai", "google genai"],
    "cohere": ["cohere", "command-r", "command_r"],
    "falcon": ["falcon", "tiiuae"],
    "yi": ["yi-", "01-ai", "yi_"],
    "granite": ["granite", "ibm-granite"],
    "stablelm": ["stablelm", "stabilityai"],
    "smollm": ["smollm", "smolvlm"],
}

PRIOR_LAYERS = {
    "chat_protocol": ["messages", "role", "system", "user", "assistant", "chat_template", "conversation"],
    "tool_calling": ["tool_call", "function_call", "tools", "arguments", "json_schema", "schema"],
    "agent_runtime": ["agent", "planner", "executor", "orchestrator", "actions", "observe", "act"],
    "memory_rag": ["rag", "retrieval", "embedding", "vector", "index", "memory", "context"],
    "inference_runtime": ["generate", "inference", "runner", "sampling", "temperature", "top_p", "top_k"],
    "tokenizer": ["tokenizer", "bpe", "sentencepiece", "vocab", "merges", "chat_template"],
    "weights": ["gguf", "safetensors", "checkpoint", "weights", ".bin", ".pt", ".pth", ".npz"],
    "eval_benchmark": ["eval", "benchmark", "score", "arc", "agi", "kaggle", "mmlu", "gsm8k"],
    "deployment": ["space", "huggingface", "github", "docker", "server", "api", "endpoint"],
    "safety_alignment": ["safety", "policy", "alignment", "guard", "moderation", "refusal"],
}

TEXT_EXTS = {
    ".py", ".ipynb", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".sh", ".c", ".h", ".cpp", ".hpp",
    ".rs", ".go", ".js", ".ts", ".html", ".css", ".xml", ".sql",
}

WEIGHT_EXTS = {
    ".gguf", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx",
    ".npz", ".npy", ".tflite",
}

SKIP_PARTS = {
    ".git", ".cache", "__pycache__", "node_modules", ".venv", "venv",
    "tok-venv", "vilenv", "site-packages", ".mypy_cache", ".pytest_cache",
}

DEFAULT_ROOTS = [
    HOME / "models",
    HOME / "llama.cpp",
    HOME / "qwen",
    HOME / "grok-1",
    HOME / "sigilagi",
    HOME / "sigilagi_real",
    HOME / "glyphmatics-vil-raw",
    HOME / "vil-canonical-glyph-system",
    HOME / "kaggle",
    HOME / "arc3_glyphmatic",
]

MAX_TEXT_BYTES = 2_000_000


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


def compact_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def classify_family(path: Path, sample: str = "") -> List[str]:
    blob = (str(path) + "\n" + sample[:10000]).lower()
    hits = []
    for fam, keys in CHAT_FAMILIES.items():
        if any(k.lower() in blob for k in keys):
            hits.append(fam)
    return hits or ["general_chat_agent"]


def classify_layers(path: Path, sample: str = "") -> List[str]:
    blob = (str(path) + "\n" + sample[:10000]).lower()
    hits = []
    for layer, keys in PRIOR_LAYERS.items():
        if any(k.lower() in blob for k in keys):
            hits.append(layer)
    return hits or ["general"]


def project_key(path: Path) -> str:
    rel = safe_rel(path)
    parts = rel.split("/")
    return parts[0] if parts else "unknown"


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


def python_signature(code: str) -> Dict[str, Any]:
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
            name = call_name(node.func)
            if name:
                sig["calls"][name] += 1

    sig["imports"] = sorted(set(sig["imports"]))
    sig["functions"] = sorted(set(sig["functions"]))
    sig["classes"] = sorted(set(sig["classes"]))
    sig["calls"] = dict(sig["calls"].most_common(100))
    return sig


def parse_safetensors_header(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("too small")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len > 64 * 1024 * 1024:
            raise ValueError("header too large")
        header = f.read(header_len).decode("utf-8", errors="replace")
    obj = json.loads(header)

    dtype_counts = Counter()
    shape_counts = Counter()
    tensor_count = 0

    for name, info in obj.items():
        if name == "__metadata__":
            continue
        tensor_count += 1
        dtype_counts[str(info.get("dtype"))] += 1
        shape = info.get("shape", [])
        shape_counts["x".join(map(str, shape))] += 1

    return {
        "format": "safetensors",
        "tensor_count": tensor_count,
        "dtype_counts": dict(dtype_counts),
        "shape_counts_top": dict(shape_counts.most_common(100)),
        "metadata_keys": list((obj.get("__metadata__", {}) or {}).keys()),
    }


def parse_npz_header(path: Path) -> Dict[str, Any]:
    members = []
    with zipfile.ZipFile(path, "r") as z:
        for info in z.infolist():
            members.append({
                "name": info.filename,
                "compressed_size": info.compress_size,
                "file_size": info.file_size,
            })
    return {
        "format": "npz",
        "member_count": len(members),
        "members_sample": members[:250],
        "truncated": len(members) > 250,
    }


def parse_gguf_header(path: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from gguf_tensor_prior_builder import parse_gguf
        rec = parse_gguf(path, hash_file=False)
        return {
            "format": "gguf",
            "version": rec.get("version"),
            "tensor_count": rec.get("tensor_count"),
            "metadata_kv_count": rec.get("metadata_kv_count"),
            "tensor_type_counts": rec.get("tensor_type_counts"),
            "tensor_rank_counts": rec.get("tensor_rank_counts"),
            "tensor_prefix_counts": rec.get("tensor_prefix_counts"),
            "tensor_shape_counts_top": rec.get("tensor_shape_counts_top"),
            "metadata_keys": list((rec.get("metadata") or {}).keys()),
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
        "project": project_key(path),
        "extension": ext,
        "size_bytes": path.stat().st_size,
        "sample_sha256_first_1mb": sha_file_sample(path),
        "boundary": "structural index only; raw tensors are not copied",
    }

    try:
        if ext == ".gguf":
            base["structure"] = parse_gguf_header(path)
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

    sample = json.dumps(base, ensure_ascii=False)
    base["families"] = classify_family(path, sample)
    base["prior_layers"] = classify_layers(path, sample)
    return base


def scan(roots: List[Path], max_files: int = 0) -> Dict[str, Any]:
    text_records = []
    weight_records = []

    families = defaultdict(list)
    layers = defaultdict(list)
    family_layer = defaultdict(lambda: defaultdict(list))

    imports = Counter()
    functions = Counter()
    classes = Counter()
    repeated_lines = Counter()

    seen = set()
    count = 0

    for root in roots:
        root = root.expanduser()
        if not root.exists():
            continue

        candidates = [root] if root.is_file() else root.rglob("*")

        for path in candidates:
            if max_files and count >= max_files:
                break

            if not path.is_file() or should_skip(path):
                continue

            ext = path.suffix.lower()
            rp = path.resolve()
            if str(rp) in seen:
                continue
            seen.add(str(rp))

            if ext in WEIGHT_EXTS:
                rec = parse_weight(path)
                weight_records.append(rec)
                count += 1

                for fam in rec["families"]:
                    families[fam].append(rec["path"])
                    for layer in rec["prior_layers"]:
                        family_layer[fam][layer].append(rec["path"])
                for layer in rec["prior_layers"]:
                    layers[layer].append(rec["path"])
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
            sample = functional[:12000]

            fams = classify_family(path, sample)
            pls = classify_layers(path, sample)

            sig = python_signature(functional) if ext in {".py", ".ipynb"} else {
                "imports": [],
                "functions": [],
                "classes": [],
                "calls": {},
            }

            for x in sig.get("imports", []):
                imports[x] += 1
            for x in sig.get("functions", []):
                functions[x] += 1
            for x in sig.get("classes", []):
                classes[x] += 1

            for line in functional.splitlines():
                line = compact_ws(line)
                if len(line) >= 12:
                    repeated_lines[line] += 1

            rec = {
                "path": safe_rel(path),
                "project": project_key(path),
                "extension": ext,
                "size_bytes": path.stat().st_size,
                "functional_bytes": len(functional.encode("utf-8")),
                "functional_sha256": sha_text(functional),
                "families": fams,
                "prior_layers": pls,
                "python_signature": sig,
            }

            text_records.append(rec)
            count += 1

            for fam in fams:
                families[fam].append(rec["path"])
                for layer in pls:
                    family_layer[fam][layer].append(rec["path"])
            for layer in pls:
                layers[layer].append(rec["path"])

    return {
        "text_records": text_records,
        "weight_records": weight_records,
        "families": families,
        "layers": layers,
        "family_layer": family_layer,
        "imports": imports,
        "functions": functions,
        "classes": classes,
        "repeated_lines": repeated_lines,
    }


def glyph_token(i: int, prefix: str) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    a = alphabet[(i // 26) % 26]
    b = alphabet[i % 26]
    return f"{prefix}{a}{b}"


def build_dictionaries(scan_data: Dict[str, Any], max_dict: int) -> Dict[str, Any]:
    family_dict = []
    for i, (fam, paths) in enumerate(sorted(scan_data["families"].items())):
        family_dict.append({
            "glyph": glyph_token(i, "C"),
            "family": fam,
            "count": len(paths),
            "sample_paths": paths[:50],
        })

    layer_dict = []
    for i, (layer, paths) in enumerate(sorted(scan_data["layers"].items())):
        layer_dict.append({
            "glyph": glyph_token(i, "L"),
            "layer": layer,
            "count": len(paths),
            "sample_paths": paths[:50],
        })

    family_layer_dict = []
    i = 0
    for fam, layer_map in sorted(scan_data["family_layer"].items()):
        for layer, paths in sorted(layer_map.items()):
            family_layer_dict.append({
                "glyph": glyph_token(i, "F"),
                "family": fam,
                "layer": layer,
                "count": len(paths),
                "sample_paths": paths[:50],
            })
            i += 1

    repeated = []
    for line, count in scan_data["repeated_lines"].items():
        if count >= 2:
            savings = (len(line) - 8) * (count - 1)
            if savings > 0:
                repeated.append((savings, count, line))
    repeated.sort(reverse=True, key=lambda x: (x[0], x[1], len(x[2])))

    structural_dict = []
    for i, (_, count, line) in enumerate(repeated[:max_dict]):
        structural_dict.append({
            "glyph": glyph_token(i, "S"),
            "kind": "repeated_structure",
            "count": count,
            "value_sha256": sha_text(line),
            "value": line,
        })

    weight_dict = []
    i = 0
    for rec in scan_data["weight_records"]:
        st = rec.get("structure", {})
        fmt = st.get("format", rec.get("extension", "unknown"))
        for fam in rec.get("families", []):
            weight_dict.append({
                "glyph": glyph_token(i, "W"),
                "kind": "weight_file",
                "family": fam,
                "format": fmt,
                "size_bytes": rec.get("size_bytes"),
                "path": rec.get("path"),
                "sample_sha256_first_1mb": rec.get("sample_sha256_first_1mb"),
            })
            i += 1

        for k, v in (st.get("tensor_type_counts") or {}).items():
            weight_dict.append({
                "glyph": glyph_token(i, "W"),
                "kind": "tensor_type",
                "format": fmt,
                "value": k,
                "count": v,
            })
            i += 1

        for k, v in (st.get("shape_counts_top") or st.get("tensor_shape_counts_top") or {}).items():
            weight_dict.append({
                "glyph": glyph_token(i, "W"),
                "kind": "tensor_shape",
                "format": fmt,
                "value": k,
                "count": v,
            })
            i += 1

    return {
        "family_dictionary": family_dict,
        "layer_dictionary": layer_dict,
        "family_layer_dictionary": family_layer_dict,
        "structural_dictionary": structural_dict,
        "weight_dictionary": weight_dict[:max_dict],
    }


def build(roots: List[Path], out_dir: Path, max_dict: int, max_files: int) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_data = scan(roots, max_files=max_files)
    dicts = build_dictionaries(scan_data, max_dict=max_dict)

    total_text_bytes = sum(x["functional_bytes"] for x in scan_data["text_records"])
    total_weight_bytes = sum(x["size_bytes"] for x in scan_data["weight_records"])
    total_observed_bytes = total_text_bytes + total_weight_bytes

    family_prior = {
        "format": "GlyphMatics Public Chat-Agent Family Prior",
        "version": "0.9.1-dev",
        "definition": "Groups public/owned local artifacts by major chat model family or chat interface family.",
        "families": {k: v for k, v in sorted(scan_data["families"].items())},
        "dictionary": dicts["family_dictionary"],
    }

    layer_prior = {
        "format": "GlyphMatics Chat-Agent Capability Layer Prior",
        "version": "0.9.1-dev",
        "definition": "Groups public/owned local artifacts by chat-agent capability layer.",
        "layers": {k: v for k, v in sorted(scan_data["layers"].items())},
        "dictionary": dicts["layer_dictionary"],
    }

    family_layer_prior = {
        "format": "GlyphMatics Chat-Agent Family x Layer Prior",
        "version": "0.9.1-dev",
        "definition": "Cross-product of model/chat-agent family and capability layer.",
        "families": {
            fam: {layer: paths for layer, paths in sorted(layer_map.items())}
            for fam, layer_map in sorted(scan_data["family_layer"].items())
        },
        "dictionary": dicts["family_layer_dictionary"],
    }

    structural_prior = {
        "format": "GlyphMatics Public Chat-Agent Structural Prior",
        "version": "0.9.1-dev",
        "definition": "Repeated public/owned structures reduced to glyph dictionary.",
        "records": scan_data["text_records"],
        "dictionary": dicts["structural_dictionary"],
        "top_imports": scan_data["imports"].most_common(200),
        "top_functions": scan_data["functions"].most_common(200),
        "top_classes": scan_data["classes"].most_common(200),
    }

    weight_prior = {
        "format": "GlyphMatics Public Chat-Agent Weight Structural Prior",
        "version": "0.9.1-dev",
        "definition": "Indexes public/owned model-weight structures without copying raw tensors.",
        "boundary": "Raw tensors are not copied. Only file type, size, sample hash, and parseable metadata/header structures are stored.",
        "records": scan_data["weight_records"],
        "dictionary": dicts["weight_dictionary"],
    }

    glyphlines = [
        "pCHATZvhz",
        "MODE PUBLIC_MAJOR_CHAT_AGENT_PRIOR_FAMILIES",
        "RULE PUBLIC_OR_OWNED_ONLY",
        "RULE SHARED_PRIOR_NOT_RAW_ENTROPY",
        "NO_PRIVATE_DATA",
        "NO_STOLEN_REPOS",
        "NO_RAW_TENSOR_COPY",
        "NO_BASE64",
        "NO_HIDDEN_PAYLOAD",
        "NO_BYTE_TO_BRAILLE",
        "NO_BYTE_TO_HANZI",
        f"TEXT_FILES {len(scan_data['text_records'])}",
        f"WEIGHT_FILES {len(scan_data['weight_records'])}",
        f"TEXT_BYTES {total_text_bytes}",
        f"WEIGHT_BYTES {total_weight_bytes}",
        f"TOTAL_OBSERVED_BYTES {total_observed_bytes}",
        f"FAMILY_GLYPHS {len(dicts['family_dictionary'])}",
        f"LAYER_GLYPHS {len(dicts['layer_dictionary'])}",
        f"FAMILY_LAYER_GLYPHS {len(dicts['family_layer_dictionary'])}",
        f"STRUCTURAL_GLYPHS {len(dicts['structural_dictionary'])}",
        f"WEIGHT_GLYPHS {len(dicts['weight_dictionary'])}",
    ]

    for section in [
        "family_dictionary",
        "layer_dictionary",
        "family_layer_dictionary",
        "structural_dictionary",
        "weight_dictionary",
    ]:
        glyphlines.append(f"SECTION {section}")
        for item in dicts[section]:
            glyphlines.append(json.dumps(item, ensure_ascii=False, sort_keys=True))

    glyphline_text = "\n".join(glyphlines) + "\n"
    glyphline_bytes = len(glyphline_text.encode("utf-8"))

    ratio = total_observed_bytes / glyphline_bytes if glyphline_bytes else 0.0
    reduction = 100.0 - ((glyphline_bytes / total_observed_bytes) * 100.0) if total_observed_bytes else 0.0

    summary = {
        "format": "GlyphMatics Public Major Chat-Agent Prior Family Summary",
        "version": "0.9.1-dev",
        "created_at": int(time.time()),
        "roots": [str(x) for x in roots],
        "text_file_count": len(scan_data["text_records"]),
        "weight_file_count": len(scan_data["weight_records"]),
        "total_text_bytes": total_text_bytes,
        "total_weight_bytes": total_weight_bytes,
        "total_observed_bytes": total_observed_bytes,
        "glyphline_bytes": glyphline_bytes,
        "semantic_structure_ratio_vs_observed_bytes": ratio,
        "semantic_structure_reduction_percent": reduction,
        "family_glyphs": len(dicts["family_dictionary"]),
        "layer_glyphs": len(dicts["layer_dictionary"]),
        "family_layer_glyphs": len(dicts["family_layer_dictionary"]),
        "structural_glyphs": len(dicts["structural_dictionary"]),
        "weight_glyphs": len(dicts["weight_dictionary"]),
        "boundary": "Public/owned artifacts only. Interface priors and structural weight priors only.",
    }

    digest_source = json.dumps({
        "summary": summary,
        "family_prior": family_prior,
        "layer_prior": layer_prior,
        "family_layer_prior": family_layer_prior,
        "structural_prior": structural_prior,
        "weight_prior": weight_prior,
    }, sort_keys=True, ensure_ascii=False)
    summary["dataset_digest"] = sha_text(digest_source)

    outputs = {
        "summary.json": summary,
        "chat_agent_family_prior.json": family_prior,
        "chat_agent_layer_prior.json": layer_prior,
        "chat_agent_family_layer_prior.json": family_layer_prior,
        "chat_agent_structural_prior.json": structural_prior,
        "chat_agent_weight_structural_prior.json": weight_prior,
    }

    for name, obj in outputs.items():
        (out_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    (out_dir / "public_chat_agent_priors.glyphlines.txt").write_text(glyphline_text, encoding="utf-8")

    return {
        "status": "built",
        "out_dir": str(out_dir),
        **summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GlyphMatics public major chat-agent prior families.")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--out", default=str(HOME / "glyphmatics_public_chat_agent_priors"))
    ap.add_argument("--max-dict", type=int, default=4096)
    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()

    roots = [Path(x).expanduser() for x in args.root] if args.root else DEFAULT_ROOTS

    result = build(
        roots=roots,
        out_dir=Path(args.out).expanduser(),
        max_dict=args.max_dict,
        max_files=args.max_files,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
