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
from typing import Any, Dict, List, Optional, Tuple

HOME = Path.home()

DEFAULT_ROOTS = [
    HOME / "kaggle",
    HOME / "arc3_glyphmatic",
    HOME / "arcagi3_glyph_encoded",
    HOME / "glyphmatics-vil-raw" / "artifacts" / "arcagi3",
]

SKIP_PARTS = {
    ".git",
    ".cache",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "site-packages",
}

SCORE_PATTERNS = [
    r"(?:public[_\s-]*score|lb[_\s-]*score|leaderboard[_\s-]*score|score)\D{0,24}([01](?:\.\d+)?)",
    r"([01]\.\d{2,6})\D{0,16}(?:public[_\s-]*score|lb[_\s-]*score|score)",
    r"(?:v|score|s)[_\-]?([01]\.\d{2,6})",
]

ARC_KEYWORDS = [
    "arc",
    "agi",
    "arcagi",
    "arc-agi",
    "arc3",
    "submission",
    "solver",
    "grid",
    "object",
    "color",
    "transform",
    "bfs",
    "planner",
    "heuristic",
]


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


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


def extract_notebook(path: Path) -> Dict[str, Any]:
    raw = read_text(path)
    try:
        nb = json.loads(raw)
    except Exception as e:
        raise ValueError(f"invalid notebook json: {e}")

    cells = nb.get("cells", [])
    code_blocks = []
    markdown_blocks = []

    for cell in cells:
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        src = str(src)

        if cell.get("cell_type") == "code":
            if src.strip():
                code_blocks.append(src)
        elif cell.get("cell_type") == "markdown":
            if src.strip():
                markdown_blocks.append(src)

    code = "\n\n".join(code_blocks)
    markdown = "\n\n".join(markdown_blocks)

    return {
        "raw": raw,
        "code": code,
        "markdown": markdown,
        "cell_count": len(cells),
        "code_cell_count": len(code_blocks),
        "markdown_cell_count": len(markdown_blocks),
        "metadata": nb.get("metadata", {}),
    }


def score_candidates(text: str, path: Path) -> List[float]:
    blob = f"{path.name}\n{path.stem}\n{text[:250000]}"
    found = []

    for pat in SCORE_PATTERNS:
        for m in re.finditer(pat, blob, flags=re.IGNORECASE):
            try:
                val = float(m.group(1))
                if 0.0 <= val <= 1.0:
                    found.append(val)
            except Exception:
                pass

    # Conservative ARC score fallback: decimals in likely score-bearing filename.
    lower_name = path.name.lower()
    if any(k in lower_name for k in ["score", "lb", "public", "submission", "arc"]):
        for m in re.finditer(r"(?<!\d)(0\.\d{2,6}|1\.0+)(?!\d)", lower_name):
            try:
                val = float(m.group(1))
                if 0.0 <= val <= 1.0:
                    found.append(val)
            except Exception:
                pass

    return found


def detect_score(notebook_text: str, path: Path) -> Optional[float]:
    vals = score_candidates(notebook_text, path)
    if not vals:
        return None
    return max(vals)


def arc_relevance(text: str, path: Path) -> int:
    blob = (str(path) + "\n" + text[:50000]).lower()
    return sum(blob.count(k) for k in ARC_KEYWORDS)


def ast_signature(code: str) -> Dict[str, Any]:
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

    def call_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = call_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return None

    def target_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
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
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                n = target_name(t)
                if n:
                    sig["assign_targets"][n] += 1

    sig["imports"] = sorted(set(sig["imports"]))
    sig["functions"] = sorted(set(sig["functions"]))
    sig["classes"] = sorted(set(sig["classes"]))
    sig["calls"] = dict(sig["calls"].most_common(200))
    sig["assign_targets"] = dict(sig["assign_targets"].most_common(200))
    return sig


def classify_solver_layers(code: str, path: Path) -> List[str]:
    blob = (str(path) + "\n" + code[:200000]).lower()
    tests = {
        "object_detection": ["object", "component", "connected", "blob", "region", "mask"],
        "grid_transform": ["grid", "transform", "rotate", "flip", "crop", "pad", "scale"],
        "color_reasoning": ["color", "palette", "recolor", "replace_color", "background"],
        "pattern_completion": ["pattern", "repeat", "symmetry", "mirror", "period", "tile"],
        "search_planning": ["bfs", "dfs", "astar", "a*", "beam", "search", "planner"],
        "heuristics": ["heuristic", "score", "cost", "distance", "loss", "rank"],
        "program_synthesis": ["program", "dsl", "primitive", "op", "operation", "candidate"],
        "ensemble": ["ensemble", "vote", "merge", "fallback", "cascade", "dual"],
        "submission_io": ["submission", "json", "predict", "test", "train", "task"],
        "neural_symbolic": ["cnn", "torch", "model", "embedding", "classifier", "neural"],
    }

    out = []
    for layer, keys in tests.items():
        if any(k in blob for k in keys):
            out.append(layer)
    return out or ["general_arc_solver"]


def find_notebooks(roots: List[Path]) -> List[Path]:
    found = []
    seen = set()

    for root in roots:
        root = root.expanduser()
        if not root.exists():
            continue

        if root.is_file() and root.suffix.lower() == ".ipynb":
            candidates = [root]
        else:
            candidates = list(root.rglob("*.ipynb"))

        for p in candidates:
            if should_skip(p):
                continue
            try:
                rp = p.resolve()
                if str(rp) not in seen and rp.is_file():
                    seen.add(str(rp))
                    found.append(rp)
            except Exception:
                pass

    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def analyze_notebook(path: Path) -> Dict[str, Any]:
    nb = extract_notebook(path)
    code = nb["code"]
    full = nb["raw"][:500000]
    score = detect_score(full + "\n" + code + "\n" + nb["markdown"], path)
    relevance = arc_relevance(code + "\n" + nb["markdown"], path)
    sig = ast_signature(code)
    layers = classify_solver_layers(code, path)

    return {
        "path": safe_rel(path),
        "absolute_path": str(path),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "mtime": int(path.stat().st_mtime),
        "detected_score": score,
        "score_detected": score is not None,
        "arc_relevance": relevance,
        "cell_count": nb["cell_count"],
        "code_cell_count": nb["code_cell_count"],
        "markdown_cell_count": nb["markdown_cell_count"],
        "code_bytes": len(code.encode("utf-8")),
        "code_sha256": sha_text(code),
        "notebook_sample_sha256": sha_text(nb["raw"][:500000]),
        "solver_layers": layers,
        "python_signature": sig,
        "code": code,
    }


def rank_notebooks(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Highest detected score first; if score missing, ARC relevance and recency decide.
    return sorted(
        records,
        key=lambda r: (
            r["detected_score"] if r["detected_score"] is not None else -1.0,
            r["arc_relevance"],
            r["code_bytes"],
            r["mtime"],
        ),
        reverse=True,
    )


def glyph_token(i: int, prefix: str) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    a = alphabet[(i // 26) % 26]
    b = alphabet[i % 26]
    return f"{prefix}{a}{b}"


def build_prior(selected: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    repeated_lines = Counter()
    imports = Counter()
    functions = Counter()
    classes = Counter()
    calls = Counter()
    assign_targets = Counter()
    layer_counts = Counter()

    notebook_records = []

    for rec in selected:
        code = rec["code"]

        for line in code.splitlines():
            line = compact_ws(line)
            if len(line) >= 12:
                repeated_lines[line] += 1

        sig = rec.get("python_signature", {})
        for x in sig.get("imports", []):
            imports[x] += 1
        for x in sig.get("functions", []):
            functions[x] += 1
        for x in sig.get("classes", []):
            classes[x] += 1
        for x, c in sig.get("calls", {}).items():
            calls[x] += c
        for x, c in sig.get("assign_targets", {}).items():
            assign_targets[x] += c
        for layer in rec.get("solver_layers", []):
            layer_counts[layer] += 1

        clean = dict(rec)
        clean.pop("code", None)
        notebook_records.append(clean)

    structural_candidates = []
    for line, count in repeated_lines.items():
        if count >= 2:
            savings = (len(line) - 8) * (count - 1)
            if savings > 0:
                structural_candidates.append((savings, count, line))
    structural_candidates.sort(reverse=True, key=lambda x: (x[0], x[1], len(x[2])))

    dictionary = []
    idx = 0

    for name, count in imports.most_common(300):
        dictionary.append({"glyph": glyph_token(idx, "A"), "kind": "import", "value": name, "count": count})
        idx += 1

    for name, count in functions.most_common(500):
        dictionary.append({"glyph": glyph_token(idx, "A"), "kind": "function", "value": name, "count": count})
        idx += 1

    for name, count in classes.most_common(300):
        dictionary.append({"glyph": glyph_token(idx, "A"), "kind": "class", "value": name, "count": count})
        idx += 1

    for name, count in calls.most_common(500):
        dictionary.append({"glyph": glyph_token(idx, "A"), "kind": "call", "value": name, "count": count})
        idx += 1

    for layer, count in layer_counts.most_common():
        dictionary.append({"glyph": glyph_token(idx, "A"), "kind": "solver_layer", "value": layer, "count": count})
        idx += 1

    for _, count, line in structural_candidates[:1000]:
        dictionary.append({
            "glyph": glyph_token(idx, "A"),
            "kind": "repeated_code_line",
            "count": count,
            "value_sha256": sha_text(line),
            "value": line,
        })
        idx += 1

    total_notebook_bytes = sum(x["size_bytes"] for x in selected)
    total_code_bytes = sum(x["code_bytes"] for x in selected)

    glyphlines = [
        "pARC5Zvhz",
        "MODE ARCAGI3_TOP5_PUBLIC_NOTEBOOK_KNOWN_PRIOR",
        "RULE PUBLIC_OR_OWNED_NOTEBOOKS_ONLY",
        "RULE SHARED_PRIOR_NOT_RAW_ENTROPY",
        "NO_PRIVATE_DATA",
        "NO_STOLEN_NOTEBOOKS",
        "NO_BASE64",
        "NO_HIDDEN_PAYLOAD",
        "NO_BYTE_TO_BRAILLE",
        "NO_BYTE_TO_HANZI",
        f"NOTEBOOKS {len(selected)}",
        f"NOTEBOOK_BYTES {total_notebook_bytes}",
        f"CODE_BYTES {total_code_bytes}",
        f"DICTIONARY {len(dictionary)}",
    ]

    for i, rec in enumerate(notebook_records, start=1):
        glyphlines.append(
            "TOP_NOTEBOOK "
            + json.dumps({
                "rank": i,
                "path": rec["path"],
                "detected_score": rec["detected_score"],
                "arc_relevance": rec["arc_relevance"],
                "code_sha256": rec["code_sha256"],
                "solver_layers": rec["solver_layers"],
            }, sort_keys=True)
        )

    glyphlines.append("SECTION dictionary")
    for item in dictionary:
        glyphlines.append(json.dumps(item, ensure_ascii=False, sort_keys=True))

    glyphline_text = "\n".join(glyphlines) + "\n"
    glyphline_bytes = len(glyphline_text.encode("utf-8"))

    ratio_vs_notebook = total_notebook_bytes / glyphline_bytes if glyphline_bytes else 0.0
    ratio_vs_code = total_code_bytes / glyphline_bytes if glyphline_bytes else 0.0

    prior = {
        "format": "GlyphMatics ARCAGI3 Top-5 Public Notebook Known Prior",
        "version": "0.9.3-dev",
        "created_at": int(time.time()),
        "glyphline": "pARC5Zvhz",
        "boundary": "Public/owned local notebooks only. This is a known-prior symbolic structural dataset, not arbitrary byte compression.",
        "ranking_rule": "Highest detected score first; fallback to ARC relevance, code size, and mtime if score missing.",
        "selected_notebooks": notebook_records,
        "dictionary": dictionary,
        "top_imports": imports.most_common(200),
        "top_functions": functions.most_common(300),
        "top_classes": classes.most_common(200),
        "top_calls": calls.most_common(300),
        "solver_layer_counts": dict(layer_counts),
    }

    summary = {
        "format": "GlyphMatics ARCAGI3 Top-5 Known Prior Summary",
        "version": "0.9.3-dev",
        "created_at": int(time.time()),
        "notebook_count": len(selected),
        "total_notebook_bytes": total_notebook_bytes,
        "total_code_bytes": total_code_bytes,
        "glyphline_bytes": glyphline_bytes,
        "semantic_structure_ratio_vs_notebook_bytes": ratio_vs_notebook,
        "semantic_structure_ratio_vs_code_bytes": ratio_vs_code,
        "semantic_structure_reduction_vs_notebook_percent": (
            100.0 - ((glyphline_bytes / total_notebook_bytes) * 100.0)
            if total_notebook_bytes else 0.0
        ),
        "dictionary_entries": len(dictionary),
        "selected": [
            {
                "rank": i + 1,
                "path": r["path"],
                "detected_score": r["detected_score"],
                "score_detected": r["score_detected"],
                "arc_relevance": r["arc_relevance"],
                "code_bytes": r["code_bytes"],
                "code_sha256": r["code_sha256"],
            }
            for i, r in enumerate(notebook_records)
        ],
    }

    digest_source = json.dumps({"summary": summary, "prior": prior}, sort_keys=True, ensure_ascii=False)
    summary["dataset_digest"] = sha_text(digest_source)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "arcagi3_top5_known_prior.json").write_text(json.dumps(prior, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "arcagi3_top5_known_prior.glyphlines.txt").write_text(glyphline_text, encoding="utf-8")

    return {"status": "built", "out_dir": str(out_dir), **summary}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build known prior from the top 5 highest-scoring local public/owned ARCAGI3 notebooks.")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--notebook", action="append", default=[], help="Explicit notebook path. Can repeat.")
    ap.add_argument("--out", default=str(HOME / "glyphmatics_arcagi3_top5_known_prior"))
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    roots = [Path(x).expanduser() for x in args.root] if args.root else DEFAULT_ROOTS
    files = find_notebooks(roots)

    for nb in args.notebook:
        p = Path(nb).expanduser()
        if p.exists() and p.suffix.lower() == ".ipynb":
            files.append(p.resolve())

    uniq = []
    seen = set()
    for f in files:
        s = str(f.resolve())
        if s not in seen:
            seen.add(s)
            uniq.append(f)

    if not uniq:
        raise SystemExit("No notebooks found. Pass --root ~/kaggle or --notebook file.ipynb")

    records = []
    failures = []

    for p in uniq:
        try:
            rec = analyze_notebook(p)
            if rec["arc_relevance"] > 0 or rec["score_detected"]:
                records.append(rec)
        except Exception as e:
            failures.append({"path": str(p), "error": str(e)})

    ranked = rank_notebooks(records)
    selected = ranked[: args.top_k]

    if not selected:
        raise SystemExit("No ARCAGI3-relevant notebooks found.")

    result = build_prior(selected, Path(args.out).expanduser())
    result["candidate_notebooks"] = len(records)
    result["failures"] = failures[:50]

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
