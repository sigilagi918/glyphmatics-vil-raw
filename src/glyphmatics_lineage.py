#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HOME = Path.home()
GM_HOME = HOME / ".glyphmatics"
PACKS = GM_HOME / "packs"
LOCKS = GM_HOME / "locks"
LINEAGE = GM_HOME / "lineage"

LINEAGE.mkdir(parents=True, exist_ok=True)

CHAIN = LINEAGE / "lineage.jsonl"
HEAD = LINEAGE / "head.json"

def sha_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def latest_pack() -> Path:
    packs = sorted(PACKS.glob("*.gma.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not packs:
        raise SystemExit("No .gma.json packs found in ~/.glyphmatics/packs")
    return packs[0]

def latest_activation() -> Path:
    acts = sorted(LOCKS.glob("pack_activation_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not acts:
        raise SystemExit("No pack_activation_*.json seals found in ~/.glyphmatics/locks")
    return acts[0]

def load_chain() -> List[Dict[str, Any]]:
    if not CHAIN.exists():
        return []
    blocks = []
    for line in CHAIN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            blocks.append(json.loads(line))
    return blocks

def verify_block_digest(block: Dict[str, Any]) -> bool:
    expected = block.get("block_digest")
    clean = dict(block)
    clean.pop("block_digest", None)
    return expected == sha_obj(clean)

def verify_chain(quiet: bool = False) -> Dict[str, Any]:
    blocks = load_chain()
    failures = []

    prev_digest = None
    prev_pack_digest = None

    for idx, block in enumerate(blocks):
        digest_ok = verify_block_digest(block)
        parent_ok = block.get("parent_block_digest") == prev_digest
        index_ok = block.get("lineage_index") == idx

        if not digest_ok or not parent_ok or not index_ok:
            failures.append({
                "index": idx,
                "digest_ok": digest_ok,
                "parent_ok": parent_ok,
                "index_ok": index_ok,
                "block_digest": block.get("block_digest"),
            })

        prev_digest = block.get("block_digest")
        prev_pack_digest = block.get("pack_digest")

    head = load_json(HEAD) if HEAD.exists() else None
    head_ok = True
    if blocks:
        head_ok = bool(head and head.get("block_digest") == blocks[-1].get("block_digest"))

    result = {
        "verified": len(failures) == 0 and head_ok,
        "block_count": len(blocks),
        "head_ok": head_ok,
        "head": head,
        "failures": failures,
    }

    if not quiet:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    return result

def append(packfile: Optional[str], activation_file: Optional[str], allow_mismatch: bool = False) -> Dict[str, Any]:
    pack_path = Path(packfile).expanduser() if packfile else latest_pack()
    act_path = Path(activation_file).expanduser() if activation_file else latest_activation()

    if not pack_path.exists():
        raise SystemExit(f"Pack missing: {pack_path}")

    if not act_path.exists():
        raise SystemExit(f"Activation seal missing: {act_path}")

    pack = load_json(pack_path)
    activation = load_json(act_path)

    pack_digest = pack.get("pack_digest")
    activation_pack_digest = activation.get("pack_digest")

    if not allow_mismatch and activation_pack_digest and activation_pack_digest != pack_digest:
        raise SystemExit(
            "Pack digest mismatch between selected pack and activation seal.\n"
            f"pack:       {pack_digest}\n"
            f"activation: {activation_pack_digest}\n"
            "Use --allow-mismatch only if you intentionally want to bind them."
        )

    blocks = load_chain()
    parent = blocks[-1] if blocks else None

    block = {
        "format": "GlyphMatics Lineage Block",
        "version": "0.5.0",
        "lineage_glyphstring": "pLvhz",
        "lineage_index": len(blocks),
        "created_at": int(time.time()),

        "parent_block_digest": parent.get("block_digest") if parent else None,
        "parent_pack_digest": parent.get("pack_digest") if parent else None,

        "pack_path": str(pack_path),
        "pack_file_sha256": sha_file(pack_path),
        "pack_digest": pack_digest,
        "pack_manifest_digest": pack.get("manifest_digest"),
        "pack_file_count": pack.get("file_count"),
        "pack_version": pack.get("version"),

        "activation_seal_path": str(act_path),
        "activation_seal_sha256": sha_file(act_path),
        "activation_digest": activation.get("activation_digest"),
        "activation_status": activation.get("status"),
        "activation_glyphstring": activation.get("activation_glyphstring"),
        "tn_verified": activation.get("tn_verified"),
        "tn_sealed": activation.get("tn_sealed"),
        "tn_error_passed": activation.get("tn_error_passed"),
        "tn_seal_digest": activation.get("tn_seal_digest"),

        "rule": {
            "pLvhz": "Python → lineage append → verify chain → hash-lock head → seal lineage",
            "parent_lock": "current block binds previous block digest and previous pack digest",
            "current_lock": "current block binds pack digest, activation digest, TN seal digest",
        },
    }

    block["block_digest"] = sha_obj(block)

    with CHAIN.open("a", encoding="utf-8") as f:
        f.write(json.dumps(block, ensure_ascii=False) + "\n")

    HEAD.write_text(json.dumps(block, indent=2, ensure_ascii=False), encoding="utf-8")

    verify = verify_chain(quiet=True)

    result = {
        "status": "lineage_appended" if verify["verified"] else "lineage_failed",
        "lineage_glyphstring": "pLvhz",
        "chain": str(CHAIN),
        "head": str(HEAD),
        "lineage_index": block["lineage_index"],
        "block_digest": block["block_digest"],
        "pack_digest": block["pack_digest"],
        "activation_digest": block["activation_digest"],
        "tn_seal_digest": block["tn_seal_digest"],
        "chain_verified": verify["verified"],
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

def head() -> None:
    if not HEAD.exists():
        raise SystemExit("No lineage head found.")
    print(HEAD.read_text(encoding="utf-8"))

def list_blocks() -> None:
    blocks = load_chain()
    summary = [
        {
            "lineage_index": b.get("lineage_index"),
            "block_digest": b.get("block_digest"),
            "parent_block_digest": b.get("parent_block_digest"),
            "pack_digest": b.get("pack_digest"),
            "activation_digest": b.get("activation_digest"),
            "tn_seal_digest": b.get("tn_seal_digest"),
            "created_at": b.get("created_at"),
        }
        for b in blocks
    ]
    print(json.dumps(summary, indent=2, ensure_ascii=False))

def main() -> None:
    parser = argparse.ArgumentParser(description="GlyphMatics artifact lineage chain")
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("append")
    a.add_argument("--pack", default=None)
    a.add_argument("--activation", default=None)
    a.add_argument("--allow-mismatch", action="store_true")

    sub.add_parser("verify")
    sub.add_parser("head")
    sub.add_parser("list")

    args = parser.parse_args()

    if args.cmd == "append":
        append(args.pack, args.activation, args.allow_mismatch)
    elif args.cmd == "verify":
        verify_chain()
    elif args.cmd == "head":
        head()
    elif args.cmd == "list":
        list_blocks()

if __name__ == "__main__":
    main()
