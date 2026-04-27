#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

HOME = Path.home()

DEFAULT_QUERIES = [
    "ARC AGI 3 submission",
    "ARC Prize submission",
    "ARC AGI solver",
    "ARC AGI ensemble",
    "ARC AGI BFS",
    "ARC AGI program synthesis",
    "ARC AGI object detection",
    "ARC AGI notebook",
    "ARC AGI public score",
    "Kaggle ARC AGI",
]

SCORE_PATTERNS = [
    r"(?:public[_\s-]*score|lb[_\s-]*score|leaderboard[_\s-]*score|score)\D{0,24}([01](?:\.\d+)?)",
    r"([01]\.\d{2,6})\D{0,20}(?:public[_\s-]*score|lb[_\s-]*score|leaderboard|score)",
    r"(?:score|lb|public|submission)[_\-\s]?([01]\.\d{2,6})",
]

ARC_LAYER_KEYS = {
    "object_detection": ["object", "component", "connected", "blob", "mask", "region"],
    "grid_transform": ["grid", "rotate", "flip", "crop", "pad", "scale", "transform"],
    "color_reasoning": ["color", "palette", "recolor", "background"],
    "pattern_completion": ["pattern", "symmetry", "mirror", "repeat", "period", "tile"],
    "search_planning": ["bfs", "dfs", "astar", "a*", "beam", "search", "planner"],
    "heuristics": ["heuristic", "score", "cost", "distance", "rank", "loss"],
    "program_synthesis": ["program", "dsl", "primitive", "candidate", "operation"],
    "ensemble": ["ensemble", "vote", "merge", "fallback", "cascade"],
    "submission_io": ["submission", "predict", "test", "train", "task", "json"],
    "neural_symbolic": ["torch", "cnn", "model", "embedding", "classifier", "neural"],
}

def run(cmd: List[str], cwd: Optional[Path] = None, check: bool = False) -> subprocess.CompletedProcess:
    print("[RUN]", " ".join(shlex.quote(x) for x in cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )

def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_ref(ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", ref.strip("/"))

def read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="replace")

def extract_notebook_code(path: Path) -> Dict[str, Any]:
    raw = read_text(path)
    try:
        nb = json.loads(raw)
    except Exception:
        return {
            "raw_sha256": sha_text(raw),
            "code": raw,
            "code_cell_count": 0,
            "markdown_cell_count": 0,
            "cell_count": 0,
        }

    code_blocks = []
    md_blocks = []

    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        src = str(src)

        if cell.get("cell_type") == "code" and src.strip():
            code_blocks.append(src)
        elif cell.get("cell_type") == "markdown" and src.strip():
            md_blocks.append(src)

    code = "\n\n".join(code_blocks)
    md = "\n\n".join(md_blocks)

    return {
        "raw_sha256": sha_text(raw),
        "code": code,
        "markdown": md,
        "code_sha256": sha_text(code),
        "markdown_sha256": sha_text(md),
        "code_cell_count": len(code_blocks),
        "markdown_cell_count": len(md_blocks),
        "cell_count": len(nb.get("cells", [])),
    }

def detect_score(text: str) -> Optional[float]:
    vals = []
    for pat in SCORE_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            try:
                v = float(m.group(1))
                if 0 <= v <= 1:
                    vals.append(v)
            except Exception:
                pass
    return max(vals) if vals else None

def classify_layers(text: str) -> List[str]:
    blob = text.lower()
    out = []
    for layer, keys in ARC_LAYER_KEYS.items():
        if any(k in blob for k in keys):
            out.append(layer)
    return out or ["general_submission_notebook"]

def compact_line(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def kaggle_auth_check() -> None:
    cp = run(["kaggle", "--version"])
    if cp.returncode != 0:
        raise SystemExit(
            "[ERR] kaggle CLI missing. Run:\n"
            "  pip install kaggle\n"
            "Then place kaggle.json in ~/.kaggle/kaggle.json"
        )

    token = HOME / ".kaggle" / "kaggle.json"
    if not token.exists():
        print("[WARN] ~/.kaggle/kaggle.json not found.")
        print("Kaggle CLI may still work if KAGGLE_USERNAME/KAGGLE_KEY env vars are set.")

def parse_kernel_list_csv(text: str) -> List[Dict[str, str]]:
    # Kaggle CLI --csv usually emits header rows. Fall back gracefully.
    lines = [x for x in text.splitlines() if x.strip()]
    if not lines:
        return []

    try:
        rows = list(csv.DictReader(lines))
        return [dict(r) for r in rows if any(r.values())]
    except Exception:
        return []

def extract_ref(row: Dict[str, str]) -> Optional[str]:
    keys = list(row.keys())
    for k in keys:
        lk = k.lower()
        if lk in {"ref", "kernel", "id", "slug"}:
            v = row.get(k, "").strip()
            if "/" in v:
                return v

    for v in row.values():
        v = str(v).strip()
        if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", v):
            return v

    return None

def discover_kernels(queries: List[str], pages: int, page_size: int, sleep_s: float) -> List[Dict[str, Any]]:
    seen = {}
    for q in queries:
        for page in range(1, pages + 1):
            cmd = [
                "kaggle", "kernels", "list",
                "-s", q,
                "-p", str(page),
                "--page-size", str(page_size),
                "--kernel-type", "notebook",
                "--sort-by", "relevance",
                "-v",
            ]
            cp = run(cmd)
            if cp.returncode != 0:
                print("[WARN] list failed:", q, "page", page)
                print("[STDOUT]", cp.stdout.strip()[-2000:])
                print("[STDERR]", cp.stderr.strip()[-4000:])
                if "401" in cp.stdout or "Unauthorized" in cp.stdout or "401" in cp.stderr or "Unauthorized" in cp.stderr:
                    raise SystemExit("[ERR] Kaggle authentication failed: 401 Unauthorized. Refresh Kaggle auth before harvesting.")
                continue

            rows = parse_kernel_list_csv(cp.stdout)
            print(f"[DISCOVER] query={q!r} page={page} rows={len(rows)}")

            for row in rows:
                ref = extract_ref(row)
                if not ref:
                    continue
                seen[ref] = {
                    "ref": ref,
                    "query": q,
                    "row": row,
                }

            time.sleep(sleep_s)

    return list(seen.values())

def pull_kernel(ref: str, vault: Path, sleep_s: float) -> Dict[str, Any]:
    out_dir = vault / safe_ref(ref)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -m asks for metadata too.
    cp = run(["kaggle", "kernels", "pull", ref, "-p", str(out_dir), "-m"])

    result = {
        "ref": ref,
        "dir": str(out_dir),
        "returncode": cp.returncode,
        "stdout": cp.stdout[-4000:],
        "stderr": cp.stderr[-4000:],
        "ok": cp.returncode == 0,
    }

    time.sleep(sleep_s)
    return result

def analyze_pulled_kernel(ref: str, folder: Path) -> Dict[str, Any]:
    files = [p for p in folder.rglob("*") if p.is_file()]
    notebooks = [p for p in files if p.suffix.lower() == ".ipynb"]
    scripts = [p for p in files if p.suffix.lower() in {".py", ".r", ".rmd"}]
    metadata_files = [p for p in files if p.name == "kernel-metadata.json"]

    metadata = {}
    for mf in metadata_files[:1]:
        try:
            metadata = json.loads(read_text(mf))
        except Exception:
            metadata = {}

    code_chunks = []
    file_records = []

    for p in notebooks:
        nb = extract_notebook_code(p)
        code_chunks.append(nb.get("code", ""))
        file_records.append({
            "path": str(p),
            "kind": "notebook",
            "size_bytes": p.stat().st_size,
            "sha256": sha_file(p),
            "code_sha256": nb.get("code_sha256"),
            "cell_count": nb.get("cell_count"),
            "code_cell_count": nb.get("code_cell_count"),
            "markdown_cell_count": nb.get("markdown_cell_count"),
        })

    for p in scripts:
        text = read_text(p)
        code_chunks.append(text)
        file_records.append({
            "path": str(p),
            "kind": "script",
            "size_bytes": p.stat().st_size,
            "sha256": sha_file(p),
            "code_sha256": sha_text(text),
        })

    all_code = "\n\n".join(code_chunks)
    score = detect_score(json.dumps(metadata, ensure_ascii=False) + "\n" + all_code)
    layers = classify_layers(all_code + "\n" + json.dumps(metadata, ensure_ascii=False))

    return {
        "ref": ref,
        "folder": str(folder),
        "metadata": metadata,
        "file_records": file_records,
        "notebook_count": len(notebooks),
        "script_count": len(scripts),
        "total_code_bytes": len(all_code.encode("utf-8")),
        "code_sha256": sha_text(all_code),
        "detected_score": score,
        "solver_layers": layers,
        "code": all_code,
    }

def build_prior(analyses: List[Dict[str, Any]], out_dir: Path, keep_raw_in_prior: bool) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked = sorted(
        analyses,
        key=lambda r: (
            r["detected_score"] if r["detected_score"] is not None else -1,
            r["total_code_bytes"],
            r["notebook_count"] + r["script_count"],
        ),
        reverse=True,
    )

    repeated = Counter()
    layer_counts = Counter()
    ref_records = []

    for rec in ranked:
        for layer in rec["solver_layers"]:
            layer_counts[layer] += 1

        for line in rec.get("code", "").splitlines():
            line = compact_line(line)
            if len(line) >= 12:
                repeated[line] += 1

        clean = {
            "ref": rec["ref"],
            "metadata": rec["metadata"],
            "file_records": rec["file_records"],
            "notebook_count": rec["notebook_count"],
            "script_count": rec["script_count"],
            "total_code_bytes": rec["total_code_bytes"],
            "code_sha256": rec["code_sha256"],
            "detected_score": rec["detected_score"],
            "solver_layers": rec["solver_layers"],
        }
        if keep_raw_in_prior:
            clean["code"] = rec.get("code", "")
        ref_records.append(clean)

    dictionary = []
    idx = 0

    def glyph(i: int) -> str:
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "N" + alphabet[(i // 26) % 26] + alphabet[i % 26]

    for layer, count in layer_counts.most_common():
        dictionary.append({"glyph": glyph(idx), "kind": "solver_layer", "value": layer, "count": count})
        idx += 1

    candidates = []
    for line, count in repeated.items():
        if count >= 2:
            savings = (len(line) - 8) * (count - 1)
            if savings > 0:
                candidates.append((savings, count, line))
    candidates.sort(reverse=True, key=lambda x: (x[0], x[1], len(x[2])))

    for _, count, line in candidates[:5000]:
        dictionary.append({
            "glyph": glyph(idx),
            "kind": "repeated_submission_line",
            "count": count,
            "value_sha256": sha_text(line),
            "value": line,
        })
        idx += 1

    total_code_bytes = sum(r["total_code_bytes"] for r in ranked)

    glyphlines = [
        "pKAGSUBZvhz",
        "MODE PUBLIC_SUBMISSION_NOTEBOOK_PRIOR",
        "RULE KAGGLE_API_PUBLIC_NOTEBOOKS",
        "RULE LOCAL_VAULT_CAN_STORE_RAW",
        "RULE REPO_PRIOR_DEFAULTS_TO_STRUCTURE_HASHES",
        "NO_PRIVATE_DATA",
        "NO_STOLEN_NOTEBOOKS",
        "NO_SECRETS",
        "NO_BASE64",
        "NO_HIDDEN_PAYLOAD",
        "NO_BYTE_TO_BRAILLE",
        "NO_BYTE_TO_HANZI",
        f"KERNELS {len(ranked)}",
        f"TOTAL_CODE_BYTES {total_code_bytes}",
        f"DICTIONARY {len(dictionary)}",
    ]

    for rec in ref_records:
        glyphlines.append(json.dumps({
            "ref": rec["ref"],
            "score": rec["detected_score"],
            "code_sha256": rec["code_sha256"],
            "layers": rec["solver_layers"],
            "code_bytes": rec["total_code_bytes"],
        }, sort_keys=True, ensure_ascii=False))

    glyphlines.append("SECTION dictionary")
    for item in dictionary:
        glyphlines.append(json.dumps(item, sort_keys=True, ensure_ascii=False))

    glyphline_text = "\n".join(glyphlines) + "\n"
    glyphline_bytes = len(glyphline_text.encode("utf-8"))

    prior = {
        "format": "GlyphMatics Public Submission Notebook Prior",
        "version": "0.9.5-dev",
        "glyphline": "pKAGSUBZvhz",
        "created_at": int(time.time()),
        "boundary": "Public Kaggle notebooks collected through official CLI. Repo prior stores structure/hashes by default; raw local vault remains local.",
        "keep_raw_in_prior": keep_raw_in_prior,
        "records": ref_records,
        "dictionary": dictionary,
    }

    summary = {
        "format": "GlyphMatics Public Submission Notebook Prior Summary",
        "version": "0.9.5-dev",
        "created_at": int(time.time()),
        "kernel_count": len(ranked),
        "total_code_bytes": total_code_bytes,
        "glyphline_bytes": glyphline_bytes,
        "semantic_structure_ratio_vs_code_bytes": total_code_bytes / glyphline_bytes if glyphline_bytes else 0.0,
        "semantic_structure_reduction_percent": 100.0 - ((glyphline_bytes / total_code_bytes) * 100.0) if total_code_bytes else 0.0,
        "dictionary_entries": len(dictionary),
        "top_refs": [
            {
                "ref": r["ref"],
                "detected_score": r["detected_score"],
                "total_code_bytes": r["total_code_bytes"],
                "code_sha256": r["code_sha256"],
                "solver_layers": r["solver_layers"],
            }
            for r in ranked[:50]
        ],
    }

    digest_source = json.dumps({"summary": summary, "prior": prior}, sort_keys=True, ensure_ascii=False)
    summary["dataset_digest"] = sha_text(digest_source)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "public_submission_notebook_prior.json").write_text(json.dumps(prior, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "public_submission_notebook_prior.glyphlines.txt").write_text(glyphline_text, encoding="utf-8")

    return {"status": "built", "out_dir": str(out_dir), **summary}

def main() -> None:
    ap = argparse.ArgumentParser(description="Collect public Kaggle submission notebooks and build a GlyphMatics prior.")
    ap.add_argument("--query", action="append", default=[])
    ap.add_argument("--queries-file", default="")
    ap.add_argument("--vault", default=str(HOME / "glyphmatics_public_submission_notebook_vault"))
    ap.add_argument("--out", default=str(HOME / "glyphmatics_public_submission_notebook_prior"))
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--page-size", type=int, default=20)
    ap.add_argument("--max-pull", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--discover-only", action="store_true")
    ap.add_argument("--keep-raw-in-prior", action="store_true")
    args = ap.parse_args()

    queries = list(args.query) or DEFAULT_QUERIES
    if args.queries_file:
        qf = Path(args.queries_file).expanduser()
        if qf.exists():
            queries += [x.strip() for x in qf.read_text().splitlines() if x.strip() and not x.strip().startswith("#")]

    kaggle_auth_check()

    vault = Path(args.vault).expanduser()
    vault.mkdir(parents=True, exist_ok=True)

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    discovered = discover_kernels(queries, pages=args.pages, page_size=args.page_size, sleep_s=args.sleep)
    (out / "discovered_kernel_refs.json").write_text(json.dumps(discovered, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[DISCOVERED]", len(discovered))

    if args.discover_only:
        print(json.dumps({"status": "discovered", "count": len(discovered), "out": str(out)}, indent=2))
        return

    pull_results = []
    analyses = []

    for item in discovered[: args.max_pull]:
        ref = item["ref"]
        pr = pull_kernel(ref, vault=vault, sleep_s=args.sleep)
        pull_results.append(pr)
        if not pr["ok"]:
            print("[WARN] pull failed:", ref)
            print("[STDOUT]", pr.get("stdout", "")[-2000:])
            print("[STDERR]", pr.get("stderr", "")[-4000:])
            continue

        folder = Path(pr["dir"])
        try:
            analyses.append(analyze_pulled_kernel(ref, folder))
        except Exception as e:
            print("[WARN] analyze failed", ref, e)

    (out / "pull_results.json").write_text(json.dumps(pull_results, indent=2, ensure_ascii=False), encoding="utf-8")

    result = build_prior(analyses, out_dir=out, keep_raw_in_prior=args.keep_raw_in_prior)
    result["discovered_count"] = len(discovered)
    result["pulled_count"] = sum(1 for x in pull_results if x["ok"])
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
