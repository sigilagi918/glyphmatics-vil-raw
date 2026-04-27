#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

HOME = Path.home()

DEFAULT_SCAN_ROOTS = [
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
]

TEXT_EXTS = {
    ".py", ".ipynb", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".sh", ".bash", ".zsh", ".c", ".h", ".cpp",
    ".hpp", ".rs", ".go", ".js", ".ts", ".html", ".css", ".java", ".kt",
}

SKIP_PARTS = {
    ".git", ".cache", "__pycache__", "node_modules", ".venv", "venv",
    "tok-venv", "vilenv", "site-packages", ".mypy_cache", ".pytest_cache",
}

MAX_FILE_BYTES = 3_000_000

CANONICAL_GLYPHS = [
    ("G0", "Origin"),
    ("G1", "Split"),
    ("G2", "Bind"),
    ("G3", "Flow"),
    ("G4", "Gate"),
    ("G5", "Memory"),
    ("G6", "Signal"),
    ("G7", "Transform"),
    ("G8", "Anchor"),
    ("G9", "Cycle"),
    ("G10", "Collapse"),
    ("G11", "Expand"),
    ("G12", "Sync"),
    ("G13", "Drift"),
    ("G14", "Lock"),
    ("G15", "Key"),
]

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(HOME))
    except Exception:
        return str(path)

def is_allowed_file(path: Path) -> bool:
    if set(path.parts) & SKIP_PARTS:
        return False
    if not path.is_file():
        return False
    if path.suffix.lower() not in TEXT_EXTS:
        return False
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return False
    except Exception:
        return False
    return True

def read_file(path: Path) -> str:
    raw = path.read_bytes()
    return raw.decode("utf-8", errors="replace")

def normalize_line(line: str) -> str:
    line = line.rstrip()
    line = re.sub(r"\s+", " ", line).strip()
    return line

def extract_notebook_code(text: str) -> str:
    try:
        obj = json.loads(text)
    except Exception:
        return text

    if not isinstance(obj, dict) or "cells" not in obj:
        return text

    blocks = []
    for cell in obj.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        src = str(src).strip()
        if src:
            blocks.append(src)

    return "\n\n".join(blocks)

def ast_signature_python(code: str) -> Dict[str, Any]:
    sig = {
        "imports": [],
        "functions": [],
        "classes": [],
        "calls": Counter(),
        "assign_targets": Counter(),
    }

    try:
        tree = ast.parse(code)
    except Exception:
        return sig

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                sig["imports"].append(n.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for n in node.names:
                sig["imports"].append(f"{mod}.{n.name}")
        elif isinstance(node, ast.FunctionDef):
            sig["functions"].append(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            sig["functions"].append(node.name)
        elif isinstance(node, ast.ClassDef):
            sig["classes"].append(node.name)
        elif isinstance(node, ast.Call):
            name = call_name(node.func)
            if name:
                sig["calls"][name] += 1
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                name = target_name(t)
                if name:
                    sig["assign_targets"][name] += 1

    sig["imports"] = sorted(set(sig["imports"]))
    sig["functions"] = sorted(set(sig["functions"]))
    sig["classes"] = sorted(set(sig["classes"]))
    sig["calls"] = dict(sig["calls"].most_common(100))
    sig["assign_targets"] = dict(sig["assign_targets"].most_common(100))
    return sig

def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None

def target_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = target_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None

def classify_path(path: Path, text: str) -> str:
    s = str(path).lower()
    t = text[:10_000].lower()

    if "arc" in s or "arc" in t:
        return "arc_agi"
    if "llama" in s or "gguf" in s or "model" in s:
        return "llm_model_stack"
    if "glyph" in s or "sigil" in s:
        return "glyphmatics_sigil"
    if "kaggle" in s or "submission" in s:
        return "competition_submission"
    if path.suffix == ".ipynb":
        return "notebook"
    if path.suffix == ".py":
        return "python_code"
    if path.suffix in {".md", ".txt"}:
        return "documentation"
    return "general"

def glyph_token(i: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    a = alphabet[(i // 26) % 26]
    b = alphabet[i % 26]
    return f"Ψ{a}{b}"

def build_dataset(roots: List[Path], out_dir: Path, max_dict: int = 2048) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    line_counter = Counter()
    phrase_counter = Counter()
    ast_index = {}
    class_counts = Counter()
    import_counts = Counter()
    function_counts = Counter()

    for root in roots:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not is_allowed_file(path):
                continue

            text = read_file(path)
            functional = extract_notebook_code(text) if path.suffix.lower() == ".ipynb" else text
            classification = classify_path(path, functional)

            lines = [normalize_line(x) for x in functional.splitlines()]
            lines = [x for x in lines if x]

            for line in lines:
                line_counter[line] += 1

            # Short phrase windows for symbolic prior.
            for line in lines:
                words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[{}()[\].,:=+\-*/<>]", line)
                for n in (2, 3, 4, 5):
                    for i in range(0, max(0, len(words) - n + 1)):
                        phrase_counter[" ".join(words[i:i+n])] += 1

            py_sig = {}
            if path.suffix.lower() in {".py", ".ipynb"}:
                py_sig = ast_signature_python(functional)
                for x in py_sig.get("imports", []):
                    import_counts[x] += 1
                for x in py_sig.get("functions", []):
                    function_counts[x] += 1
                for x in py_sig.get("classes", []):
                    class_counts[x] += 1

            record = {
                "path": safe_rel(path),
                "classification": classification,
                "size_bytes": path.stat().st_size,
                "text_sha256": sha_text(text),
                "functional_sha256": sha_text(functional),
                "functional_bytes": len(functional.encode("utf-8")),
                "line_count": len(lines),
                "python_signature": py_sig,
            }
            files.append(record)
            ast_index[record["path"]] = py_sig

    dictionary_candidates = []

    for line, count in line_counter.items():
        if count >= 2 and len(line) >= 8:
            savings = (len(line) - 5) * (count - 1)
            if savings > 0:
                dictionary_candidates.append(("line", savings, count, line))

    for phrase, count in phrase_counter.items():
        if count >= 3 and len(phrase) >= 8:
            savings = (len(phrase) - 5) * (count - 1)
            if savings > 0:
                dictionary_candidates.append(("phrase", savings, count, phrase))

    dictionary_candidates.sort(key=lambda x: (x[1], x[2], len(x[3])), reverse=True)

    dictionary = []
    seen_values = set()

    for i, (kind, savings, count, value) in enumerate(dictionary_candidates):
        if len(dictionary) >= max_dict:
            break
        if value in seen_values:
            continue
        seen_values.add(value)
        dictionary.append({
            "glyph": glyph_token(len(dictionary)),
            "kind": kind,
            "count": count,
            "savings_estimate": savings,
            "value": value,
            "value_sha256": sha_text(value),
        })

    total_functional_bytes = sum(f["functional_bytes"] for f in files)
    dict_text = "\n".join(f"{d['glyph']}={d['value']}" for d in dictionary)
    dict_bytes = len(dict_text.encode("utf-8"))

    glyphline_program = "\n".join([
        "pCOMPETITORZvhz",
        "MODE SHARED_PRIOR_DATASET",
        "RULE STRUCTURE_NOT_BYTES",
        "NO_PRIVATE_DATA",
        "NO_BASE64",
        "NO_HIDDEN_PAYLOAD",
        f"FILES {len(files)}",
        f"DICT {len(dictionary)}",
        f"BYTES {total_functional_bytes}",
    ]) + "\n"

    glyphline_bytes = len(glyphline_program.encode("utf-8")) + dict_bytes
    compression_ratio = total_functional_bytes / glyphline_bytes if glyphline_bytes else 0.0
    reduction_percent = 100 - ((glyphline_bytes / total_functional_bytes) * 100) if total_functional_bytes else 0.0

    manifest = {
        "format": "GlyphMatics Competitor Shared Prior Dataset",
        "version": "0.1.0",
        "created_at": int(time.time()),
        "rule": "compression via shared prior + program induction + symbolic dictionary",
        "license_boundary": "public_or_owned_sources_only",
        "roots": [str(r) for r in roots],
        "file_count": len(files),
        "dictionary_entries": len(dictionary),
        "total_functional_bytes": total_functional_bytes,
        "dictionary_bytes": dict_bytes,
        "glyphline_bytes_estimated": glyphline_bytes,
        "semantic_compression_ratio_estimated": compression_ratio,
        "semantic_reduction_percent_estimated": reduction_percent,
        "canonical_glyph_basis": [{"glyph": g, "role": r} for g, r in CANONICAL_GLYPHS],
        "top_imports": import_counts.most_common(100),
        "top_functions": function_counts.most_common(100),
        "top_classes": class_counts.most_common(100),
        "classifications": Counter(f["classification"] for f in files),
        "dataset_digest": None,
    }

    dataset = {
        "manifest": manifest,
        "files": files,
        "dictionary": dictionary,
    }

    dataset["manifest"]["classifications"] = dict(dataset["manifest"]["classifications"])
    digest_source = json.dumps(dataset, sort_keys=True, ensure_ascii=False)
    dataset["manifest"]["dataset_digest"] = sha_text(digest_source)

    (out_dir / "competitor_shared_prior_dataset.json").write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (out_dir / "competitor_shared_prior_glyphline.txt").write_text(
        glyphline_program + "\n" + dict_text + "\n",
        encoding="utf-8",
    )

    summary = {
        "status": "built",
        "out_dir": str(out_dir),
        "dataset": str(out_dir / "competitor_shared_prior_dataset.json"),
        "glyphline": str(out_dir / "competitor_shared_prior_glyphline.txt"),
        "file_count": len(files),
        "dictionary_entries": len(dictionary),
        "total_functional_bytes": total_functional_bytes,
        "glyphline_bytes_estimated": glyphline_bytes,
        "semantic_compression_ratio_estimated": compression_ratio,
        "semantic_reduction_percent_estimated": reduction_percent,
        "dataset_digest": dataset["manifest"]["dataset_digest"],
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def main() -> None:
    ap = argparse.ArgumentParser(description="Build GlyphMatics competitor shared-prior dataset from public/owned local artifacts.")
    ap.add_argument("--out", default=str(HOME / "glyphmatics_competitor_prior"))
    ap.add_argument("--max-dict", type=int, default=2048)
    ap.add_argument("--root", action="append", default=[])
    args = ap.parse_args()

    roots = [Path(x).expanduser() for x in args.root] if args.root else DEFAULT_SCAN_ROOTS
    summary = build_dataset(roots, Path(args.out).expanduser(), args.max_dict)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
