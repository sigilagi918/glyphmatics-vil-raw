#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HOME = Path.home()
GM_HOME = HOME / ".glyphmatics"
PACKS = GM_HOME / "packs"
LOCKS = GM_HOME / "locks"
ACTIVATIONS = GM_HOME / "activations"

PACKS.mkdir(parents=True, exist_ok=True)
LOCKS.mkdir(parents=True, exist_ok=True)
ACTIVATIONS.mkdir(parents=True, exist_ok=True)

DEFAULT_INCLUDE = [
    GM_HOME,
    HOME / "install_glyphmatics.py",
    HOME / "glyphmatics_artifact_vm.py",
    HOME / "glyphmatics_tn_vm.py",
    HOME / "glyphmatics_pack.py",
]

EXCLUDE_PARTS = {
    ".cache",
    "__pycache__",
}

EXCLUDE_DIR_NAMES = {
    "packs",
    "activations",
}

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(HOME))
    except ValueError:
        return str(path)

def should_include(path: Path) -> bool:
    if path.is_dir():
        return False

    parts = set(path.parts)

    if parts & EXCLUDE_PARTS:
        return False

    # Prevent recursive packs-in-packs and activation-in-pack.
    try:
        rel = path.relative_to(GM_HOME)
        if rel.parts and rel.parts[0] in EXCLUDE_DIR_NAMES:
            return False
    except ValueError:
        pass

    if path.name.endswith(".tmp"):
        return False

    return True

def collect_files() -> List[Path]:
    files: List[Path] = []
    seen = set()

    for root in DEFAULT_INCLUDE:
        if not root.exists():
            continue

        if root.is_file():
            candidates = [root]
        else:
            candidates = [p for p in root.rglob("*") if p.is_file()]

        for p in candidates:
            rp = p.resolve()
            if str(rp) in seen:
                continue
            if should_include(rp):
                seen.add(str(rp))
                files.append(rp)

    return sorted(files, key=lambda x: safe_rel(x))

def pack(out: str | None = None) -> Path:
    files = collect_files()
    entries = []

    for p in files:
        data = p.read_bytes()
        entries.append({
            "path": safe_rel(p),
            "size": len(data),
            "sha256": sha_bytes(data),
            "mode": stat.S_IMODE(p.stat().st_mode),
            "data_b64": base64.b64encode(data).decode("ascii"),
        })

    manifest = [
        {
            "path": e["path"],
            "size": e["size"],
            "sha256": e["sha256"],
            "mode": e["mode"],
        }
        for e in entries
    ]

    envelope = {
        "format": "GlyphMatics Artifact Pack",
        "version": "0.4.0",
        "pack_glyphstring": "pPvhz",
        "activate_glyphstring": "pAvhz",
        "created_at": int(time.time()),
        "home_root": str(HOME),
        "file_count": len(entries),
        "manifest": manifest,
        "manifest_digest": sha_obj(manifest),
        "files": entries,
    }

    envelope["pack_digest"] = sha_obj(envelope)

    if out is None:
        out_path = PACKS / f"glyphmatics_pack_{envelope['pack_digest'][:16]}.gma.json"
    else:
        out_path = Path(out).expanduser()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "status": "packed",
        "path": str(out_path),
        "file_count": len(entries),
        "manifest_digest": envelope["manifest_digest"],
        "pack_digest": envelope["pack_digest"],
    }, indent=2))

    return out_path

def load_pack(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))

def verify(path: str | Path, quiet: bool = False) -> Dict[str, Any]:
    p = Path(path).expanduser()
    env = load_pack(p)

    checks = []
    for e in env.get("files", []):
        data = base64.b64decode(e["data_b64"].encode("ascii"))
        actual = sha_bytes(data)
        checks.append({
            "path": e["path"],
            "expected": e["sha256"],
            "actual": actual,
            "match": actual == e["sha256"],
        })

    manifest = [
        {
            "path": e["path"],
            "size": e["size"],
            "sha256": e["sha256"],
            "mode": e.get("mode", 0o644),
        }
        for e in env.get("files", [])
    ]

    manifest_digest = sha_obj(manifest)
    manifest_match = manifest_digest == env.get("manifest_digest")

    env_copy = dict(env)
    stored_pack_digest = env_copy.pop("pack_digest", None)
    computed_pack_digest = sha_obj(env_copy)
    pack_match = stored_pack_digest == computed_pack_digest

    result = {
        "verified": bool(all(c["match"] for c in checks) and manifest_match and pack_match),
        "path": str(p),
        "file_count": len(checks),
        "manifest_digest_expected": env.get("manifest_digest"),
        "manifest_digest_actual": manifest_digest,
        "manifest_match": manifest_match,
        "pack_digest_expected": stored_pack_digest,
        "pack_digest_actual": computed_pack_digest,
        "pack_match": pack_match,
        "failed": [c for c in checks if not c["match"]],
    }

    if not quiet:
        print(json.dumps(result, indent=2))

    return result

def inspect(path: str | Path) -> None:
    env = load_pack(path)
    print(json.dumps({
        "format": env.get("format"),
        "version": env.get("version"),
        "pack_glyphstring": env.get("pack_glyphstring"),
        "activate_glyphstring": env.get("activate_glyphstring"),
        "created_at": env.get("created_at"),
        "file_count": env.get("file_count"),
        "manifest_digest": env.get("manifest_digest"),
        "pack_digest": env.get("pack_digest"),
        "files": env.get("manifest", []),
    }, indent=2, ensure_ascii=False))

def safe_unpack_path(target: Path, rel: str) -> Path:
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError(f"Unsafe path in pack: {rel}")

    out = (target / rel_path).resolve()
    root = target.resolve()

    if root not in out.parents and out != root:
        raise ValueError(f"Path escape blocked: {rel}")

    return out

def unpack(path: str | Path, target: str | None = None, quiet: bool = False) -> Dict[str, Any]:
    verification = verify(path, quiet=quiet)
    if not verification["verified"]:
        raise SystemExit("Pack verification failed. Refusing unpack.")

    env = load_pack(path)
    target_dir = Path(target).expanduser() if target else HOME / "glyphmatics_restore"
    target_dir.mkdir(parents=True, exist_ok=True)

    restored = []

    for e in env.get("files", []):
        out = safe_unpack_path(target_dir, e["path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        data = base64.b64decode(e["data_b64"].encode("ascii"))
        out.write_bytes(data)
        try:
            out.chmod(e.get("mode", 0o644))
        except PermissionError:
            pass
        restored.append(str(out))

    result = {
        "status": "unpacked",
        "target": str(target_dir),
        "restored_count": len(restored),
    }

    if not quiet:
        print(json.dumps(result, indent=2))

    return result

def latest() -> None:
    packs = sorted(PACKS.glob("*.gma.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not packs:
        raise SystemExit("No packs found in ~/.glyphmatics/packs")

    p = packs[0]
    env = load_pack(p)

    print(json.dumps({
        "path": str(p),
        "pack_digest": env.get("pack_digest"),
        "file_count": env.get("file_count"),
        "version": env.get("version"),
    }, indent=2))

def run_logged(cmd: List[str], env: Dict[str, str], log_path: Path) -> Dict[str, Any]:
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(Path(env["HOME"])),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log_path.write_text(proc.stdout, encoding="utf-8")

    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "log": str(log_path),
        "ok": proc.returncode == 0,
    }

def activate(
    packfile: str | Path,
    work_dir: str | None = None,
    tn: int = 20,
    bond: int = 64,
    keep: bool = True,
) -> Dict[str, Any]:
    pack_path = Path(packfile).expanduser()
    verification = verify(pack_path, quiet=True)

    if not verification["verified"]:
        raise SystemExit("Pack verification failed. Refusing activation.")

    pack_digest = verification["pack_digest_actual"]
    work = Path(work_dir).expanduser() if work_dir else ACTIVATIONS / f"activation_{pack_digest[:16]}"

    if work.exists() and not keep:
        shutil.rmtree(work)

    work.mkdir(parents=True, exist_ok=True)
    (work / "usr" / "bin").mkdir(parents=True, exist_ok=True)

    unpack(pack_path, target=str(work), quiet=True)

    env = dict(os.environ)
    env["HOME"] = str(work)
    env["PREFIX"] = str(work / "usr")

    install_script = work / "install_glyphmatics.py"
    tn_script = work / "glyphmatics_tn_vm.py"

    if not install_script.exists():
        raise SystemExit(f"Restored install script missing: {install_script}")

    if not tn_script.exists():
        raise SystemExit(f"Restored TN VM missing: {tn_script}")

    logs = work / "activation_logs"
    logs.mkdir(parents=True, exist_ok=True)

    install_result = run_logged(
        [sys.executable, str(install_script), "pgvayh"],
        env,
        logs / "install.log",
    )

    tn_result = run_logged(
        [
            sys.executable,
            str(tn_script),
            "p",
            "hcbme",
            f"tn:{tn}",
            f"b:{bond}",
            "TMGE",
            "cv",
            "ruz",
        ],
        env,
        logs / "tn_selftest.log",
    )

    last_tn = work / ".glyphmatics" / "state" / "last_tn_activation.json"
    tn_state = {}
    if last_tn.exists():
        tn_state = json.loads(last_tn.read_text(encoding="utf-8"))

    tn_context = tn_state.get("context", {})
    error_check = tn_context.get("error_check", {})

    activation_ok = bool(
        install_result["ok"]
        and tn_result["ok"]
        and tn_context.get("verified") is True
        and tn_context.get("sealed") is True
        and error_check.get("passed") is True
    )

    report = {
        "status": "activated" if activation_ok else "failed",
        "activation_glyphstring": "pAvhz",
        "pack": str(pack_path),
        "pack_digest": pack_digest,
        "work_dir": str(work),
        "verify_pack": verification,
        "install_result": install_result,
        "tn_result": tn_result,
        "tn_verified": tn_context.get("verified"),
        "tn_sealed": tn_context.get("sealed"),
        "tn_error_passed": error_check.get("passed"),
        "tn_seal_digest": tn_context.get("seal_digest"),
        "created_at": int(time.time()),
    }

    report["activation_digest"] = sha_obj(report)

    live_seal = LOCKS / f"pack_activation_{report['activation_digest'][:16]}.json"
    live_seal.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["activation_seal"] = str(live_seal)

    work_seal = work / ".glyphmatics" / "locks" / f"pack_activation_{report['activation_digest'][:16]}.json"
    work_seal.parent.mkdir(parents=True, exist_ok=True)
    work_seal.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description="GlyphMatics single-file artifact packer/activator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack")
    p.add_argument("--out", default=None)

    v = sub.add_parser("verify")
    v.add_argument("packfile")

    i = sub.add_parser("inspect")
    i.add_argument("packfile")

    u = sub.add_parser("unpack")
    u.add_argument("packfile")
    u.add_argument("--to", default=None)

    a = sub.add_parser("activate")
    a.add_argument("packfile")
    a.add_argument("--work-dir", default=None)
    a.add_argument("--tn", type=int, default=20)
    a.add_argument("--bond", type=int, default=64)
    a.add_argument("--fresh", action="store_true", help="Delete prior activation workdir first.")

    sub.add_parser("latest")

    args = parser.parse_args()

    if args.cmd == "pack":
        pack(args.out)
    elif args.cmd == "verify":
        verify(args.packfile)
    elif args.cmd == "inspect":
        inspect(args.packfile)
    elif args.cmd == "unpack":
        unpack(args.packfile, args.to)
    elif args.cmd == "activate":
        activate(
            args.packfile,
            work_dir=args.work_dir,
            tn=args.tn,
            bond=args.bond,
            keep=not args.fresh,
        )
    elif args.cmd == "latest":
        latest()

if __name__ == "__main__":
    main()
