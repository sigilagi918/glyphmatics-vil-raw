#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

HOME = Path.home()

DEFAULT_ROOTS = [
    HOME / "glyphmatics-vil-raw",
    HOME / "vil-canonical-glyph-system",
    HOME / "vil-canonical-glyph-system-repo",
    HOME / "vil-glyphmatic-demo",
    HOME / "sigilagi",
    HOME / "sigilagi_real",
    HOME / "llama.cpp",
    HOME / "kaggle",
    HOME / "arc3_glyphmatic",
]

CODE_EXTS = {
    ".py", ".sh", ".bash", ".zsh",
    ".c", ".h", ".cpp", ".hpp",
    ".rs", ".go",
    ".js", ".ts", ".jsx", ".tsx",
    ".html", ".css",
    ".java", ".kt",
    ".sql", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".md", ".txt",
    ".ipynb",
}

SKIP_PARTS = {
    ".git", ".cache", "__pycache__", "node_modules", ".venv", "venv",
    "tok-venv", "vilenv", "site-packages", ".mypy_cache", ".pytest_cache",
    "proof_runs", "downloads", "Downloads",
}

SENSITIVE_PATTERNS = [
    "token", "secret", "password", "passwd", "credential", "apikey", "api_key",
    "private_key", "id_rsa", "id_ed25519", "authorized_keys", ".env", "cookies",
    "hosts.yml", "gh/hosts.yml", "huggingface/token",
]

LICENSE_FILENAMES = [
    "LICENSE", "LICENSE.md", "LICENSE.txt",
    "COPYING", "COPYING.md", "NOTICE",
]

PERMISSIVE_LICENSE_HINTS = {
    "mit": ["mit license", "permission is hereby granted"],
    "apache-2.0": ["apache license", "version 2.0"],
    "bsd": ["bsd license", "redistribution and use in source and binary forms"],
    "isc": ["isc license", "permission to use, copy, modify, and/or distribute"],
    "unlicense": ["the unlicense", "unencumbered public domain"],
    "cc0": ["creative commons zero", "cc0"],
}

RESTRICTIVE_LICENSE_HINTS = {
    "gpl": ["gnu general public license"],
    "agpl": ["gnu affero general public license"],
    "lgpl": ["gnu lesser general public license"],
}

MAX_FILE_BYTES_DEFAULT = 500_000


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(HOME))
    except Exception:
        return str(path)


def is_sensitive(path: Path) -> bool:
    s = str(path).lower()
    return any(x.lower() in s for x in SENSITIVE_PATTERNS)


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & SKIP_PARTS) or is_sensitive(path)


def read_text(path: Path, max_bytes: Optional[int] = None) -> str:
    data = path.read_bytes()
    if max_bytes is not None:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


def find_repo_root(path: Path, search_roots: List[Path]) -> Path:
    current = path if path.is_dir() else path.parent
    search_resolved = [r.expanduser().resolve() for r in search_roots if r.exists()]

    while True:
        if (current / ".git").exists() or any((current / name).exists() for name in LICENSE_FILENAMES):
            return current

        if current.parent == current:
            break

        if current.resolve() in search_resolved:
            return current

        current = current.parent

    # fallback: nearest supplied root containing path
    rp = path.resolve()
    candidates = []
    for root in search_resolved:
        try:
            rp.relative_to(root)
            candidates.append(root)
        except Exception:
            pass
    if candidates:
        return sorted(candidates, key=lambda x: len(str(x)), reverse=True)[0]

    return path.parent


def detect_license(repo_root: Path) -> Dict[str, Any]:
    found = []

    for name in LICENSE_FILENAMES:
        p = repo_root / name
        if p.exists() and p.is_file() and not is_sensitive(p):
            try:
                text = read_text(p, max_bytes=200_000)
            except Exception:
                continue
            lower = text.lower()

            license_kind = "unknown"
            redistribution = "unknown"

            for kind, hints in PERMISSIVE_LICENSE_HINTS.items():
                if all(h in lower for h in hints[:1]) or any(h in lower for h in hints):
                    license_kind = kind
                    redistribution = "permissive"
                    break

            if license_kind == "unknown":
                for kind, hints in RESTRICTIVE_LICENSE_HINTS.items():
                    if any(h in lower for h in hints):
                        license_kind = kind
                        redistribution = "copyleft_or_restrictive"
                        break

            found.append({
                "path": safe_rel(p),
                "sha256": sha_text(text),
                "kind": license_kind,
                "redistribution": redistribution,
                "sample": text[:500],
            })

    if not found:
        return {
            "present": False,
            "kind": "unknown",
            "redistribution": "unknown",
            "files": [],
        }

    # Prefer permissive if any permissive license exists.
    for item in found:
        if item["redistribution"] == "permissive":
            return {
                "present": True,
                "kind": item["kind"],
                "redistribution": "permissive",
                "files": found,
            }

    return {
        "present": True,
        "kind": found[0]["kind"],
        "redistribution": found[0]["redistribution"],
        "files": found,
    }


def extract_notebook_code(text: str) -> str:
    try:
        nb = json.loads(text)
    except Exception:
        return text

    if not isinstance(nb, dict) or "cells" not in nb:
        return text

    blocks = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            src = str(src).strip()
            if src:
                blocks.append(src)
    return "\n\n".join(blocks)


def classify_code_layer(path: Path, text: str) -> List[str]:
    blob = (str(path) + "\n" + text[:20000]).lower()
    tests = {
        "runtime": ["runtime", "vm", "execute", "interpreter", "runner"],
        "agent": ["agent", "planner", "tool", "action", "orchestrator"],
        "model": ["model", "weights", "tensor", "gguf", "inference", "generate"],
        "dataset": ["dataset", "corpus", "loader", "download", "scrape", "ingest"],
        "benchmark": ["kaggle", "arc", "agi", "score", "eval", "submission"],
        "visual": ["image", "png", "pgm", "ocr", "visual", "canvas", "render"],
        "web_ui": ["html", "css", "javascript", "react", "button", "canvas"],
        "security": ["hash", "sha256", "digest", "verify", "lock", "seal"],
        "packaging": ["pack", "artifact", "manifest", "bundle", "archive"],
        "docs": ["readme", "license", "documentation", "docs"],
    }

    layers = []
    for layer, keys in tests.items():
        if any(k in blob for k in keys):
            layers.append(layer)
    return layers or ["general_code"]


def project_key(path: Path) -> str:
    rel = safe_rel(path)
    return rel.split("/", 1)[0] if rel else "unknown"


def find_code_files(roots: List[Path]) -> List[Path]:
    found = []
    seen = set()

    for root in roots:
        root = root.expanduser()
        if not root.exists():
            continue

        candidates = [root] if root.is_file() else root.rglob("*")

        for p in candidates:
            if not p.is_file() or should_skip(p):
                continue

            if p.suffix.lower() not in CODE_EXTS:
                continue

            try:
                rp = p.resolve()
                if str(rp) not in seen:
                    seen.add(str(rp))
                    found.append(rp)
            except Exception:
                pass

    return sorted(found, key=lambda p: str(p))


def glyph_token(i: int, prefix: str = "K") -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    a = alphabet[(i // 26) % 26]
    b = alphabet[i % 26]
    return f"{prefix}{a}{b}"


def build(roots: List[Path], out_dir: Path, max_file_bytes: int, include_unknown_license: bool, max_files: int) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_code_files(roots)
    if max_files > 0:
        files = files[:max_files]

    repo_license_cache: Dict[str, Dict[str, Any]] = {}

    records = []
    skipped = []
    layer_counts = Counter()
    project_counts = Counter()
    extension_counts = Counter()
    line_counts = Counter()
    total_code_bytes = 0
    included_code_bytes = 0

    for p in files:
        repo_root = find_repo_root(p, roots)
        repo_key = str(repo_root)

        if repo_key not in repo_license_cache:
            repo_license_cache[repo_key] = detect_license(repo_root)

        lic = repo_license_cache[repo_key]
        ext = p.suffix.lower()

        try:
            size = p.stat().st_size
        except Exception:
            continue

        if size > max_file_bytes:
            skipped.append({
                "path": safe_rel(p),
                "reason": "too_large",
                "size_bytes": size,
            })
            continue

        allowed_actual_code = lic.get("redistribution") == "permissive" or include_unknown_license

        if not allowed_actual_code:
            skipped.append({
                "path": safe_rel(p),
                "reason": "license_not_permissive_or_unknown",
                "license": lic,
            })
            continue

        try:
            raw = read_text(p)
        except Exception as e:
            skipped.append({
                "path": safe_rel(p),
                "reason": "read_failed",
                "error": str(e),
            })
            continue

        code = extract_notebook_code(raw) if ext == ".ipynb" else raw
        code_sha = sha_text(code)
        total_code_bytes += len(code.encode("utf-8"))
        included_code_bytes += len(code.encode("utf-8"))

        layers = classify_code_layer(p, code)
        for layer in layers:
            layer_counts[layer] += 1
        project_counts[project_key(p)] += 1
        extension_counts[ext] += 1

        for line in code.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if len(line) >= 12:
                line_counts[line] += 1

        records.append({
            "path": safe_rel(p),
            "project": project_key(p),
            "repo_root": safe_rel(repo_root),
            "extension": ext,
            "size_bytes": size,
            "code_bytes": len(code.encode("utf-8")),
            "code_sha256": code_sha,
            "source_file_sha256": sha_file(p),
            "license": lic,
            "layers": layers,
            "actual_code": code,
        })

    dictionary = []
    idx = 0

    for project, count in project_counts.most_common():
        dictionary.append({"glyph": glyph_token(idx), "kind": "project", "value": project, "count": count})
        idx += 1

    for layer, count in layer_counts.most_common():
        dictionary.append({"glyph": glyph_token(idx), "kind": "code_layer", "value": layer, "count": count})
        idx += 1

    for ext, count in extension_counts.most_common():
        dictionary.append({"glyph": glyph_token(idx), "kind": "extension", "value": ext, "count": count})
        idx += 1

    repeated_candidates = []
    for line, count in line_counts.items():
        if count >= 2:
            savings = (len(line) - 8) * (count - 1)
            if savings > 0:
                repeated_candidates.append((savings, count, line))

    repeated_candidates.sort(reverse=True, key=lambda x: (x[0], x[1], len(x[2])))

    for _, count, line in repeated_candidates[:2000]:
        dictionary.append({
            "glyph": glyph_token(idx),
            "kind": "repeated_public_code_line",
            "count": count,
            "value_sha256": sha_text(line),
            "value": line,
        })
        idx += 1

    glyphlines = [
        "pCODEZvhz",
        "MODE PUBLIC_CODE_KNOWN_PRIOR",
        "RULE ACTUAL_CODE_ONLY_IF_OWNED_OR_REDIStributABLE",
        "RULE SHARED_PRIOR_NOT_RAW_ENTROPY",
        "NO_PRIVATE_DATA",
        "NO_SECRETS",
        "NO_STOLEN_REPOS",
        "NO_BASE64",
        "NO_HIDDEN_PAYLOAD",
        "NO_BYTE_TO_BRAILLE",
        "NO_BYTE_TO_HANZI",
        f"FILES {len(records)}",
        f"INCLUDED_CODE_BYTES {included_code_bytes}",
        f"DICTIONARY {len(dictionary)}",
    ]

    for rec in records:
        glyphlines.append(json.dumps({
            "path": rec["path"],
            "code_sha256": rec["code_sha256"],
            "license_kind": rec["license"].get("kind"),
            "layers": rec["layers"],
        }, ensure_ascii=False, sort_keys=True))

    glyphline_text = "\n".join(glyphlines) + "\n"
    glyphline_bytes = len(glyphline_text.encode("utf-8"))

    prior = {
        "format": "GlyphMatics Public Code Known Prior",
        "version": "0.9.4-dev",
        "glyphline": "pCODEZvhz",
        "created_at": int(time.time()),
        "boundary": "Actual code included only when owned/permissively licensed or explicitly overridden by user.",
        "include_unknown_license": include_unknown_license,
        "records": records,
        "dictionary": dictionary,
        "skipped": skipped[:2000],
    }

    summary = {
        "format": "GlyphMatics Public Code Known Prior Summary",
        "version": "0.9.4-dev",
        "created_at": int(time.time()),
        "roots": [str(x) for x in roots],
        "files_included": len(records),
        "files_skipped": len(skipped),
        "included_code_bytes": included_code_bytes,
        "glyphline_bytes": glyphline_bytes,
        "semantic_structure_ratio_vs_included_code_bytes": included_code_bytes / glyphline_bytes if glyphline_bytes else 0.0,
        "semantic_structure_reduction_percent": (
            100.0 - ((glyphline_bytes / included_code_bytes) * 100.0)
            if included_code_bytes else 0.0
        ),
        "dictionary_entries": len(dictionary),
        "project_counts": dict(project_counts),
        "layer_counts": dict(layer_counts),
        "extension_counts": dict(extension_counts),
        "boundary": "Actual source code is stored only for owned or redistributable public code unless --include-unknown-license is used.",
    }

    digest_source = json.dumps({"summary": summary, "records": records, "dictionary": dictionary}, sort_keys=True, ensure_ascii=False)
    summary["dataset_digest"] = sha_text(digest_source)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "public_code_known_prior.json").write_text(json.dumps(prior, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "public_code_known_prior.glyphlines.txt").write_text(glyphline_text, encoding="utf-8")

    return {"status": "built", "out_dir": str(out_dir), **summary}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GlyphMatics actual public/owned code known prior.")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--out", default=str(HOME / "glyphmatics_public_code_known_prior"))
    ap.add_argument("--max-file-bytes", type=int, default=MAX_FILE_BYTES_DEFAULT)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--include-unknown-license", action="store_true", help="Include actual code even if license is unknown. Use only for owned code or code you are allowed to redistribute.")
    args = ap.parse_args()

    roots = [Path(x).expanduser() for x in args.root] if args.root else DEFAULT_ROOTS

    result = build(
        roots=roots,
        out_dir=Path(args.out).expanduser(),
        max_file_bytes=args.max_file_bytes,
        include_unknown_license=args.include_unknown_license,
        max_files=args.max_files,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
