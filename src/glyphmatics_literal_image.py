#!/data/data/com.termux/files/usr/bin/python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except ImportError:
    print("Missing Pillow. Install with: pip install pillow", file=sys.stderr)
    raise

HOME = Path.home()

DEFAULT_FONT_CANDIDATES = [
    "/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/system/fonts/RobotoMono-Regular.ttf",
    "/system/fonts/DroidSansMono.ttf",
]

# The image contains only the literal glyphlines.
# No base64, no hidden payload, no braille side-channel, no extra envelope.
# Execution is driven by OCR recovery of those exact lines.

def load_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return normalize_glyphlines(text)

def normalize_glyphlines(text: str) -> str:
    lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        s = line.strip()
        if s:
            lines.append(s)
    return "\n".join(lines)


CANONICAL_GLYPHLINES = {
    "pgvayh": "pgvayh",
    "p hcbme tn:20 b:64 TMGE cv ruz": "p hcbme tn:20 b:64 TMGE cv ruz",
    "pPvhz": "pPvhz",
    "pAvhz": "pAvhz",
    "pLvhz": "pLvhz",
    "pVvhz": "pVvhz",
}

OCR_ALIASES = {
    "pPAvhz": "pAvhz",
    "pAvhz.": "pAvhz",
    "pAvhz,": "pAvhz",
    "pAvhz|": "pAvhz",
    "pAvhzl": "pAvhz",
    "pAvhz1": "pAvhz",
    "pPvhz.": "pPvhz",
    "pLvhz.": "pLvhz",
    "pVvhz.": "pVvhz",
    "pgvayh.": "pgvayh",
}

def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        ndp = [i]
        for j, cb in enumerate(b, 1):
            ndp.append(min(
                dp[j] + 1,
                ndp[j - 1] + 1,
                dp[j - 1] + (ca != cb),
            ))
        dp = ndp
    return dp[-1]

def canonicalize_glyphline(line: str) -> str:
    raw = " ".join(line.strip().split())

    if raw in CANONICAL_GLYPHLINES:
        return CANONICAL_GLYPHLINES[raw]

    if raw in OCR_ALIASES:
        return OCR_ALIASES[raw]

    # Remove OCR punctuation noise.
    cleaned = raw.replace(".", "").replace(",", "").replace("|", "").strip()
    if cleaned in CANONICAL_GLYPHLINES:
        return CANONICAL_GLYPHLINES[cleaned]
    if cleaned in OCR_ALIASES:
        return OCR_ALIASES[cleaned]

    # Conservative fuzzy snap only for short command glyphlines.
    candidates = list(CANONICAL_GLYPHLINES.keys())
    best = min(candidates, key=lambda x: edit_distance(cleaned, x))
    dist = edit_distance(cleaned, best)

    if dist <= 1:
        return best

    # Special case: OCR often inserts/duplicates P before A.
    if cleaned.endswith("Avhz") and cleaned.startswith("p"):
        return "pAvhz"

    return raw


def choose_font(font_size: int):
    for fp in DEFAULT_FONT_CANDIDATES:
        p = Path(fp)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), font_size)
            except Exception:
                pass
    return ImageFont.load_default()

def render_glyphlines_to_image(text: str, out_path: Path, font_size: int = 40, margin: int = 40, line_gap: int = 14):
    font = choose_font(font_size)
    lines = text.splitlines() or [""]

    dummy = Image.new("L", (32, 32), 255)
    draw = ImageDraw.Draw(dummy)

    max_w = 0
    heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        max_w = max(max_w, w)
        heights.append(h)

    total_h = sum(heights) + max(0, len(lines) - 1) * line_gap
    width = max_w + margin * 2
    height = total_h + margin * 2

    img = Image.new("L", (max(width, 256), max(height, 128)), 255)
    draw = ImageDraw.Draw(img)

    y = margin
    for i, line in enumerate(lines):
        draw.text((margin, y), line, fill=0, font=font)
        y += heights[i] + line_gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return {
        "status": "written",
        "image": str(out_path),
        "width": img.width,
        "height": img.height,
        "glyphline_count": len(lines),
    }

def tesseract_exists() -> bool:
    return shutil.which("tesseract") is not None

def ocr_image(image_path: Path) -> str:
    if not tesseract_exists():
        raise SystemExit("Missing tesseract. Install with: pkg install tesseract")

    with Image.open(image_path) as img:
        gray = ImageOps.grayscale(img)
        # Light cleanup to improve OCR while preserving literal visible content.
        gray = gray.resize((gray.width * 2, gray.height * 2))
        bw = gray.point(lambda x: 255 if x > 180 else 0, mode="1")
        with tempfile.TemporaryDirectory() as td:
            tmp_img = Path(td) / "ocr_input.png"
            out_base = Path(td) / "ocr_out"
            bw.save(tmp_img)

            cmd = [
                "tesseract",
                str(tmp_img),
                str(out_base),
                "--psm", "6",
                "-l", "eng",
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            txt_path = Path(str(out_base) + ".txt")
            raw = txt_path.read_text(encoding="utf-8", errors="ignore")

    # Keep only the literal glyphlines structure.
    return normalize_glyphlines(raw)

def latest_pack() -> Path:
    pack_dir = HOME / ".glyphmatics" / "packs"
    packs = sorted(pack_dir.glob("*.gma.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not packs:
        raise SystemExit("No pack found in ~/.glyphmatics/packs")
    return packs[0]

def latest_visual() -> Path:
    visual_dir = HOME / ".glyphmatics" / "visual"
    visuals = sorted(visual_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not visuals:
        raise SystemExit("No visual found in ~/.glyphmatics/visual")
    return visuals[0]

def run_cmd(cmd: list[str]) -> dict:
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "ok": proc.returncode == 0,
    }

def execute_glyphline(line: str) -> dict:
    raw_line = normalize_glyphlines(line)
    line = canonicalize_glyphline(raw_line)
    if not line:
        return {"glyphline": line, "raw_glyphline": raw_line, "ok": True, "skipped": True}

    if line == "pgvayh":
        script = HOME / "install_glyphmatics.py"
        return {
            "glyphline": line,
            "action": "install_glyphmatics",
            "result": run_cmd([sys.executable, str(script), "pgvayh"]),
        }

    if line == "p hcbme tn:20 b:64 TMGE cv ruz":
        script = HOME / "glyphmatics_tn_vm.py"
        return {
            "glyphline": line,
            "action": "run_tn_selftest",
            "result": run_cmd([
                sys.executable, str(script),
                "p", "hcbme", "tn:20", "b:64", "TMGE", "cv", "ruz"
            ]),
        }

    if line == "pPvhz":
        return {
            "glyphline": line,
            "action": "pack_runtime",
            "result": run_cmd(["glyphmatics-pack", "pack"]),
        }

    if line == "pAvhz":
        pack = latest_pack()
        return {
            "glyphline": line,
            "action": "activate_latest_pack",
            "pack": str(pack),
            "result": run_cmd(["glyphmatics-pack", "activate", str(pack), "--fresh"]),
        }

    if line == "pLvhz":
        return {
            "glyphline": line,
            "action": "append_lineage",
            "result": run_cmd(["glyphmatics-lineage", "append"]),
        }

    if line == "pVvhz":
        pack = latest_pack()
        return {
            "glyphline": line,
            "action": "encode_latest_pack_to_image",
            "pack": str(pack),
            "result": run_cmd(["glyphmatics-visual", "encode", str(pack)]),
        }

    return {
        "glyphline": line,
        "raw_glyphline": raw_line,
        "ok": False,
        "error": "Unknown glyphline",
    }

def execute_glyphlines(text: str) -> dict:
    raw_lines = [x.strip() for x in text.splitlines() if x.strip()]
    lines = [canonicalize_glyphline(x) for x in raw_lines]
    results = []
    overall_ok = True
    for line in raw_lines:
        item = execute_glyphline(line)
        results.append(item)
        if "result" in item and isinstance(item["result"], dict):
            if not item["result"].get("ok", False):
                overall_ok = False
        elif item.get("ok") is False:
            overall_ok = False
    return {
        "ok": overall_ok,
        "glyphline_count": len(lines),
        "raw_glyphlines": raw_lines,
        "canonical_glyphlines": lines,
        "results": results,
    }

def command_make(args):
    if args.glyphlines:
        text = normalize_glyphlines(args.glyphlines)
    elif args.input:
        text = load_text(Path(args.input))
    else:
        raise SystemExit("Provide --glyphlines or --input")
    res = render_glyphlines_to_image(
        text=text,
        out_path=Path(args.out),
        font_size=args.font_size,
        margin=args.margin,
        line_gap=args.line_gap,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))

def command_extract(args):
    text = ocr_image(Path(args.image))
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(json.dumps({
            "status": "extracted",
            "image": str(args.image),
            "out": str(args.out),
            "glyphlines": text.splitlines(),
        }, indent=2, ensure_ascii=False))
    else:
        print(text)

def command_run(args):
    text = ocr_image(Path(args.image))
    res = execute_glyphlines(text)
    res["source_image"] = str(args.image)
    res["glyphlines"] = text.splitlines()
    print(json.dumps(res, indent=2, ensure_ascii=False))

def command_show(args):
    if args.glyphlines:
        text = normalize_glyphlines(args.glyphlines)
    elif args.input:
        text = load_text(Path(args.input))
    else:
        raise SystemExit("Provide --glyphlines or --input")
    print(text)

def build_parser():
    p = argparse.ArgumentParser(
        description="GlyphMatics literal image artifact tool: render literal glyphlines into a single image and recover/execute them."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("make", help="Render literal glyphlines to a single image artifact")
    a.add_argument("--glyphlines", default=None, help="Literal glyphlines text")
    a.add_argument("--input", default=None, help="Path to text file containing glyphlines")
    a.add_argument("--out", required=True, help="Output image path, e.g. ~/artifact.png")
    a.add_argument("--font-size", type=int, default=40)
    a.add_argument("--margin", type=int, default=40)
    a.add_argument("--line-gap", type=int, default=14)
    a.set_defaults(func=command_make)

    b = sub.add_parser("extract", help="OCR literal glyphlines from the image artifact")
    b.add_argument("image", help="Image artifact path")
    b.add_argument("--out", default=None, help="Optional output text file")
    b.set_defaults(func=command_extract)

    c = sub.add_parser("run", help="OCR and execute recognized glyphlines from the image artifact")
    c.add_argument("image", help="Image artifact path")
    c.set_defaults(func=command_run)

    d = sub.add_parser("show", help="Show normalized literal glyphlines")
    d.add_argument("--glyphlines", default=None)
    d.add_argument("--input", default=None)
    d.set_defaults(func=command_show)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
