#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise SystemExit("Missing pillow. Install with: pip install pillow")

HOME = Path.home()

DEFAULT_ROOTS = [
    HOME / "kaggle",
    HOME / "arc3_glyphmatic",
    HOME / "arc_agi3_data",
    HOME / "arc_tasks",
    HOME,
]

KEYWORDS = [
    "arc",
    "agi3",
    "kaggle",
    "submission",
    "final",
    "dual",
    "plan",
    "score",
    "optimized",
    "agent",
    "notebook",
    "v5",
    "v6",
    "v30",
    "forge",
    "glyph",
]

FONT_CANDIDATES = [
    "/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/system/fonts/RobotoMono-Regular.ttf",
    "/system/fonts/DroidSansMono.ttf",
]

def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def safe_rel(p: Path) -> str:
    try:
        return str(p.relative_to(HOME))
    except Exception:
        return str(p)

def find_notebooks(roots: List[Path]) -> List[Path]:
    seen = set()
    out = []

    skip_parts = {
        ".git",
        ".cache",
        "node_modules",
        "venv",
        ".venv",
        "tok-venv",
        "vilenv",
        "site-packages",
        "__pycache__",
    }

    for root in roots:
        if not root.exists():
            continue

        for p in root.rglob("*.ipynb"):
            parts = set(p.parts)
            if parts & skip_parts:
                continue

            rp = p.resolve()
            if str(rp) in seen:
                continue

            seen.add(str(rp))
            out.append(rp)

    return out

def score_notebook(path: Path) -> float:
    s = str(path).lower()
    score = 0.0

    for kw in KEYWORDS:
        if kw in s:
            score += 10.0

    try:
        size = path.stat().st_size
        mtime = path.stat().st_mtime
    except Exception:
        return score

    # Prefer real notebooks but avoid huge output-only notebooks.
    score += min(size / 50_000, 20)

    # Recency boost.
    age_days = max(0, (time.time() - mtime) / 86400)
    score += max(0, 30 - age_days)

    # Penalize checkpoints.
    if ".ipynb_checkpoints" in s:
        score -= 100

    return score

def load_notebook_code(path: Path) -> Tuple[str, List[str]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return "", []

    cells = obj.get("cells", [])
    blocks = []

    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        src = cell.get("source", "")
        if isinstance(src, list):
            text = "".join(src)
        else:
            text = str(src)

        text = text.rstrip()
        if not text.strip():
            continue

        blocks.append(f"# ---- CELL {i} ----\n{text}")

    code = "\n\n".join(blocks).strip() + "\n"
    lines = code.splitlines()
    return code, lines

def token_name(i: int) -> str:
    # OCR-stable ASCII glyph token.
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    a = alphabet[(i // 26) % 26]
    b = alphabet[i % 26]
    return f"G{a}{b}"

def build_dictionary(all_lines: List[str], max_tokens: int = 512) -> Dict[str, str]:
    counts = Counter(all_lines)

    candidates = []
    for line, count in counts.items():
        stripped = line.strip()
        if count < 2:
            continue
        if len(stripped) < 12:
            continue
        if stripped.startswith("# ---- CELL"):
            continue

        # Estimated savings: repeated literal length minus token length.
        savings = (len(line) - 5) * (count - 1)
        if savings > 0:
            candidates.append((savings, count, line))

    candidates.sort(reverse=True, key=lambda x: (x[0], x[1], len(x[2])))

    dictionary = {}
    for i, (_, _, line) in enumerate(candidates[:max_tokens]):
        dictionary[token_name(i)] = line

    return dictionary

def escape_literal(s: str) -> str:
    # JSON string gives exact text while staying visible.
    return json.dumps(s, ensure_ascii=False)

def unescape_literal(s: str) -> str:
    return json.loads(s)

def encode_notebooks(notebooks: List[Path], out_dir: Path, max_tokens: int) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    all_lines = []

    for p in notebooks:
        code, lines = load_notebook_code(p)
        if not code.strip():
            continue

        rec = {
            "path": safe_rel(p),
            "source_file": str(p),
            "code_sha256": sha_text(code),
            "code_bytes": len(code.encode("utf-8")),
            "line_count": len(lines),
            "lines": lines,
        }
        records.append(rec)
        all_lines.extend(lines)

    dictionary = build_dictionary(all_lines, max_tokens=max_tokens)
    reverse = {v: k for k, v in dictionary.items()}

    glyphlines = []
    glyphlines.append("GMARC3V1")
    glyphlines.append("MODE FUNCTIONAL_CODE_ONLY")
    glyphlines.append("RULE GLYPHMATICS_STRUCTURAL_COMPRESSION_ONLY")
    glyphlines.append("NO_BASE64")
    glyphlines.append("NO_BYTE_WRAPPER")
    glyphlines.append("NO_HIDDEN_PAYLOAD")
    glyphlines.append(f"NOTEBOOKS {len(records)}")
    glyphlines.append(f"DICT {len(dictionary)}")

    for tok, line in dictionary.items():
        glyphlines.append(f"D {tok} {escape_literal(line)}")

    for idx, rec in enumerate(records):
        glyphlines.append(f"N {idx} {escape_literal(rec['path'])} {rec['code_sha256']} {rec['code_bytes']} {rec['line_count']}")
        for line in rec["lines"]:
            if line in reverse:
                glyphlines.append(f"T {reverse[line]}")
            else:
                glyphlines.append(f"L {escape_literal(line)}")
        glyphlines.append("ENDN")

    glyph_text = "\n".join(glyphlines) + "\n"
    glyph_sha = sha_text(glyph_text)

    raw_code_bytes = sum(r["code_bytes"] for r in records)
    glyph_bytes = len(glyph_text.encode("utf-8"))
    ratio = raw_code_bytes / glyph_bytes if glyph_bytes else 0.0
    reduction = 100.0 - ((glyph_bytes / raw_code_bytes) * 100.0) if raw_code_bytes else 0.0

    manifest = {
        "format": "GlyphMatics ARCAGI3 Notebook Structural Glyphpack",
        "version": "0.1.0",
        "glyphstring": "pARC3Zvhz",
        "mode": "functional_code_only",
        "notebook_count": len(records),
        "dictionary_entries": len(dictionary),
        "raw_code_bytes": raw_code_bytes,
        "glyphline_bytes": glyph_bytes,
        "semantic_compression_ratio": ratio,
        "semantic_reduction_percent": reduction,
        "glyphline_sha256": glyph_sha,
        "records": [
            {
                "path": r["path"],
                "code_sha256": r["code_sha256"],
                "code_bytes": r["code_bytes"],
                "line_count": r["line_count"],
            }
            for r in records
        ],
    }

    (out_dir / "arcagi3_top_notebooks.glyphlines.txt").write_text(glyph_text, encoding="utf-8")
    (out_dir / "arcagi3_top_notebooks_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "manifest": manifest,
        "glyph_text": glyph_text,
        "dictionary": dictionary,
        "records": records,
    }

def decode_glyphlines(glyph_text: str, out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    dictionary = {}
    current = None
    outputs = []

    for raw in glyph_text.splitlines():
        line = raw.rstrip("\n")
        if not line:
            continue

        if line.startswith("D "):
            _, tok, literal = line.split(" ", 2)
            dictionary[tok] = unescape_literal(literal)
            continue

        if line.startswith("N "):
            parts = line.split(" ", 5)
            idx = parts[1]
            path_literal = parts[2]
            expected_sha = parts[3]
            expected_bytes = int(parts[4])
            expected_lines = int(parts[5])
            current = {
                "idx": idx,
                "path": unescape_literal(path_literal),
                "expected_sha": expected_sha,
                "expected_bytes": expected_bytes,
                "expected_lines": expected_lines,
                "lines": [],
            }
            continue

        if line == "ENDN":
            if current is None:
                continue

            code = "\n".join(current["lines"]) + "\n"
            actual_sha = sha_text(code)
            ok = actual_sha == current["expected_sha"]

            name = Path(current["path"]).name
            if name.endswith(".ipynb"):
                name = name[:-6]
            out_path = out_dir / f"{current['idx']}_{name}.py"
            out_path.write_text(code, encoding="utf-8")

            outputs.append({
                "path": current["path"],
                "out": str(out_path),
                "expected_sha": current["expected_sha"],
                "actual_sha": actual_sha,
                "ok": ok,
                "bytes": len(code.encode("utf-8")),
            })
            current = None
            continue

        if current is not None:
            if line.startswith("T "):
                tok = line.split(" ", 1)[1]
                current["lines"].append(dictionary[tok])
            elif line.startswith("L "):
                current["lines"].append(unescape_literal(line[2:]))

    return {
        "restored": outputs,
        "ok": all(x["ok"] for x in outputs),
        "count": len(outputs),
    }

def choose_font(font_size: int):
    for fp in FONT_CANDIDATES:
        p = Path(fp)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), font_size)
            except Exception:
                pass
    return ImageFont.load_default()

def render_text_image(text: str, out_path: Path, font_size: int = 16, margin: int = 28, line_gap: int = 4, max_width_px: int = 1800) -> Dict:
    font = choose_font(font_size)

    # Wrap long visible lines so image remains readable/OCR-able.
    wrapped = []
    for line in text.splitlines():
        if len(line) <= 180:
            wrapped.append(line)
        else:
            # Use continuation prefix. Decoder from OCR is not implemented for wrapped lines,
            # but the text file remains exact. Image is transport/inspection artifact.
            for i in range(0, len(line), 180):
                prefix = "" if i == 0 else "↪"
                wrapped.append(prefix + line[i:i+180])

    dummy = Image.new("L", (10, 10), 255)
    d = ImageDraw.Draw(dummy)

    line_h = math.ceil(font_size * 1.28)
    width = min(max_width_px, max(600, max(int(d.textlength(x, font=font)) for x in wrapped) + margin * 2))
    height = max(200, len(wrapped) * (line_h + line_gap) + margin * 2)

    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)

    y = margin
    for line in wrapped:
        draw.text((margin, y), line, fill=0, font=font)
        y += line_h + line_gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

    return {
        "image": str(out_path),
        "width": width,
        "height": height,
        "rendered_lines": len(wrapped),
    }

def main():
    ap = argparse.ArgumentParser(description="Encode top ARCAGI3 notebooks into GlyphMatics structural glyphlines and a single image.")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default=str(HOME / "arcagi3_glyph_encoded"))
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--decode-test", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser()
    notebooks = find_notebooks(DEFAULT_ROOTS)
    ranked = sorted(notebooks, key=score_notebook, reverse=True)
    selected = ranked[:args.top]

    if not selected:
        raise SystemExit("No .ipynb files found.")

    result = encode_notebooks(selected, out_dir, max_tokens=args.max_tokens)
    manifest = result["manifest"]

    image_info = None
    if args.render:
        image_info = render_text_image(
            result["glyph_text"],
            out_dir / "arcagi3_top_notebooks_glyphline_image.png",
            font_size=16,
        )

    decode_info = None
    if args.decode_test:
        decode_info = decode_glyphlines(
            result["glyph_text"],
            out_dir / "restored_code",
        )

    print(json.dumps({
        "status": "encoded",
        "out_dir": str(out_dir),
        "glyphlines": str(out_dir / "arcagi3_top_notebooks.glyphlines.txt"),
        "manifest": str(out_dir / "arcagi3_top_notebooks_manifest.json"),
        "image": image_info,
        "decode_test": decode_info,
        "notebooks": manifest["notebook_count"],
        "raw_code_bytes": manifest["raw_code_bytes"],
        "glyphline_bytes": manifest["glyphline_bytes"],
        "semantic_compression_ratio": manifest["semantic_compression_ratio"],
        "semantic_reduction_percent": manifest["semantic_reduction_percent"],
        "selected": manifest["records"],
    }, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
