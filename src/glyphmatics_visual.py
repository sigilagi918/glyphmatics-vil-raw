#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path
from typing import Any, Tuple

HOME = Path.home()
GM_HOME = HOME / ".glyphmatics"
PACKS = GM_HOME / "packs"
VISUAL = GM_HOME / "visual"
LOCKS = GM_HOME / "locks"

VISUAL.mkdir(parents=True, exist_ok=True)
LOCKS.mkdir(parents=True, exist_ok=True)

MAGIC = b"GMVIL001"
HEADER_SIZE = 8 + 8 + 32
# magic[8] + payload_len[8] + sha256[32]

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

def latest_pack() -> Path:
    packs = sorted(PACKS.glob("*.gma.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not packs:
        raise SystemExit("No .gma.json pack found. Run: glyphmatics-pack pack")
    return packs[0]

def encode_file(src: str | None, out: str | None, width: int = 512) -> Path:
    src_path = Path(src).expanduser() if src else latest_pack()
    if not src_path.exists() or not src_path.is_file():
        raise SystemExit(f"Input file missing: {src_path}")

    payload = src_path.read_bytes()
    digest = hashlib.sha256(payload).digest()
    header = MAGIC + struct.pack(">Q", len(payload)) + digest
    blob = header + payload

    height = math.ceil(len(blob) / width)
    padded_len = width * height
    raster = blob + bytes(padded_len - len(blob))

    if out is None:
        out_path = VISUAL / f"visual_{sha_bytes(payload)[:16]}.pgm"
    else:
        out_path = Path(out).expanduser()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    pgm_header = (
        b"P5\n"
        b"# GlyphMatics VIL-RAW pVvhz image artifact\n"
        + f"{width} {height}\n255\n".encode("ascii")
    )

    out_path.write_bytes(pgm_header + raster)

    seal = {
        "glyphstring": "pVvhz",
        "mode": "visual_encode",
        "source": str(src_path),
        "visual": str(out_path),
        "source_size": len(payload),
        "source_sha256": sha_bytes(payload),
        "width": width,
        "height": height,
        "pixel_bytes": padded_len,
        "created_at": int(time.time()),
    }
    seal["visual_digest"] = sha_obj(seal)

    seal_path = LOCKS / f"visual_seal_{seal['visual_digest'][:16]}.json"
    seal["seal"] = str(seal_path)
    seal_path.write_text(json.dumps(seal, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "status": "visual_encoded",
        "glyphstring": "pVvhz",
        "source": str(src_path),
        "visual": str(out_path),
        "source_sha256": seal["source_sha256"],
        "source_size": len(payload),
        "width": width,
        "height": height,
        "visual_seal": str(seal_path),
        "visual_digest": seal["visual_digest"],
    }, indent=2))

    return out_path

def parse_pgm(path: Path) -> Tuple[int, int, bytes]:
    data = path.read_bytes()
    pos = 0

    def read_token() -> bytes:
        nonlocal pos

        while pos < len(data) and data[pos] in b" \t\r\n":
            pos += 1

        if pos < len(data) and data[pos:pos+1] == b"#":
            while pos < len(data) and data[pos:pos+1] not in b"\n":
                pos += 1
            return read_token()

        start = pos
        while pos < len(data) and data[pos] not in b" \t\r\n":
            pos += 1

        return data[start:pos]

    magic = read_token()
    if magic != b"P5":
        raise ValueError("Not a binary PGM/P5 image.")

    width = int(read_token())
    height = int(read_token())
    maxval = int(read_token())

    if maxval != 255:
        raise ValueError("Only 8-bit PGM maxval=255 is supported.")

    # Skip one whitespace byte after maxval.
    if pos < len(data) and data[pos] in b" \t\r\n":
        pos += 1

    raster = data[pos:]
    expected = width * height

    if len(raster) < expected:
        raise ValueError(f"PGM raster too short: got {len(raster)}, expected {expected}")

    return width, height, raster[:expected]

def extract_payload(image: str | Path) -> dict:
    img = Path(image).expanduser()
    width, height, raster = parse_pgm(img)

    if len(raster) < HEADER_SIZE:
        raise ValueError("Raster too small for GlyphMatics visual header.")

    magic = raster[:8]
    if magic != MAGIC:
        raise ValueError(f"Bad GlyphMatics visual magic: {magic!r}")

    payload_len = struct.unpack(">Q", raster[8:16])[0]
    expected_digest = raster[16:48]

    start = HEADER_SIZE
    end = start + payload_len

    if end > len(raster):
        raise ValueError("Payload length exceeds raster size.")

    payload = raster[start:end]
    actual_digest = hashlib.sha256(payload).digest()

    return {
        "image": str(img),
        "width": width,
        "height": height,
        "payload_len": payload_len,
        "expected_sha256": expected_digest.hex(),
        "actual_sha256": actual_digest.hex(),
        "verified": expected_digest == actual_digest,
        "payload": payload,
    }

def verify_image(image: str | Path) -> dict:
    info = extract_payload(image)
    result = {
        "verified": info["verified"],
        "image": info["image"],
        "width": info["width"],
        "height": info["height"],
        "payload_len": info["payload_len"],
        "expected_sha256": info["expected_sha256"],
        "actual_sha256": info["actual_sha256"],
    }
    print(json.dumps(result, indent=2))
    return result

def restore_image(image: str | Path, out: str | None) -> Path:
    info = extract_payload(image)

    if not info["verified"]:
        raise SystemExit("Image payload verification failed. Refusing restore.")

    if out is None:
        out_path = PACKS / f"restored_from_visual_{info['actual_sha256'][:16]}.gma.json"
    else:
        out_path = Path(out).expanduser()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(info["payload"])

    result = {
        "status": "restored",
        "glyphstring": "pVvhz",
        "image": info["image"],
        "out": str(out_path),
        "sha256": info["actual_sha256"],
        "bytes": info["payload_len"],
    }

    print(json.dumps(result, indent=2))
    return out_path

def seal(image: str | Path) -> None:
    info = extract_payload(image)
    seal_obj = {
        "glyphstring": "pVvhz",
        "mode": "visual_verify_hash_seal",
        "image": str(Path(image).expanduser()),
        "payload_len": info["payload_len"],
        "payload_sha256": info["actual_sha256"],
        "verified": info["verified"],
        "created_at": int(time.time()),
    }
    seal_obj["seal_digest"] = sha_obj(seal_obj)

    seal_path = LOCKS / f"visual_lock_{seal_obj['seal_digest'][:16]}.json"
    seal_obj["seal"] = str(seal_path)

    seal_path.write_text(json.dumps(seal_obj, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": "sealed" if info["verified"] else "failed",
        "seal": str(seal_path),
        "seal_digest": seal_obj["seal_digest"],
        "verified": info["verified"],
    }, indent=2))

def main() -> None:
    parser = argparse.ArgumentParser(description="GlyphMatics VIL-RAW visual artifact codec")
    sub = parser.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode")
    e.add_argument("input", nargs="?", default=None)
    e.add_argument("--out", default=None)
    e.add_argument("--width", type=int, default=512)

    v = sub.add_parser("verify")
    v.add_argument("image")

    r = sub.add_parser("restore")
    r.add_argument("image")
    r.add_argument("--out", default=None)

    s = sub.add_parser("seal")
    s.add_argument("image")

    args = parser.parse_args()

    if args.cmd == "encode":
        encode_file(args.input, args.out, args.width)
    elif args.cmd == "verify":
        verify_image(args.image)
    elif args.cmd == "restore":
        restore_image(args.image, args.out)
    elif args.cmd == "seal":
        seal(args.image)

if __name__ == "__main__":
    main()
