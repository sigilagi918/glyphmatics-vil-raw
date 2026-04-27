#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import runpy
import stat
import sys
import time
from pathlib import Path

HOME = Path.home()
GM_HOME = HOME / ".glyphmatics"
ARTIFACTS = GM_HOME / "artifacts"
MEMORY = GM_HOME / "memory"
LOCKS = GM_HOME / "locks"
REGISTRY = GM_HOME / "registry"
STATE = GM_HOME / "state"

for p in [GM_HOME, ARTIFACTS, MEMORY, LOCKS, REGISTRY, STATE]:
    p.mkdir(parents=True, exist_ok=True)

DIALECTS = {
    "p": "python",
    "b": "bash",
    "j": "javascript",
    "t": "typescript",
    "r": "rust",
    "g": "go",
    "c": "c",
    "C": "cpp",
    "k": "kotlin",
    "s": "swift",
    "q": "sql",
}

PYTHON_OPS = {
    "g": "generate_filesystem",
    "v": "validate_runtime",
    "a": "anchor_core_artifact",
    "y": "yield_cli",
    "h": "hash_lock",
    "c": "create_space_or_collapse",
    "b": "build_backend",
    "m": "mount_memory",
    "e": "execute_chain",
    "r": "register",
    "u": "update_registry",
    "z": "seal_definition",
}

G0_G15 = [
    {"index": "G0", "role": "Origin", "visible": "⊙", "braille": "⠀", "hanzi": "一"},
    {"index": "G1", "role": "Split", "visible": "⟂", "braille": "⠁", "hanzi": "二"},
    {"index": "G2", "role": "Bind", "visible": "⛓", "braille": "⠃", "hanzi": "三"},
    {"index": "G3", "role": "Flow", "visible": "→", "braille": "⠉", "hanzi": "四"},
    {"index": "G4", "role": "Gate", "visible": "⊢", "braille": "⠙", "hanzi": "五"},
    {"index": "G5", "role": "Memory", "visible": "▣", "braille": "⠑", "hanzi": "六"},
    {"index": "G6", "role": "Signal", "visible": "⌁", "braille": "⠋", "hanzi": "七"},
    {"index": "G7", "role": "Transform", "visible": "⟳", "braille": "⠛", "hanzi": "八"},
    {"index": "G8", "role": "Anchor", "visible": "⌖", "braille": "⠓", "hanzi": "九"},
    {"index": "G9", "role": "Cycle", "visible": "↻", "braille": "⠊", "hanzi": "十"},
    {"index": "G10", "role": "Collapse", "visible": "⇣", "braille": "⠚", "hanzi": "百"},
    {"index": "G11", "role": "Expand", "visible": "⇡", "braille": "⠅", "hanzi": "千"},
    {"index": "G12", "role": "Sync", "visible": "≋", "braille": "⠇", "hanzi": "万"},
    {"index": "G13", "role": "Drift", "visible": "≈", "braille": "⠍", "hanzi": "亿"},
    {"index": "G14", "role": "Lock", "visible": "■", "braille": "⠝", "hanzi": "兆"},
    {"index": "G15", "role": "Key", "visible": "◆", "braille": "⠕", "hanzi": "世"},
]

def sha(obj) -> str:
    if isinstance(obj, bytes):
        raw = obj
    elif isinstance(obj, str):
        raw = obj.encode("utf-8")
    else:
        raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def glyph_digest(g):
    return sha(g["visible"] + g["braille"] + g["hanzi"])

def write_runtime():
    runtime = GM_HOME / "glyphmatics_runtime.py"
    runtime.write_text(RUNTIME_CODE, encoding="utf-8")
    runtime.chmod(runtime.stat().st_mode | stat.S_IXUSR)
    return runtime

def write_artifact(glyphstring: str):
    basis = []
    for g in G0_G15:
        x = dict(g)
        x["digest"] = glyph_digest(x)
        basis.append(x)

    artifact = {
        "artifact": "GlyphMatics Core Artifact",
        "version": "0.2.0",
        "install_glyphstring": glyphstring,
        "dialect_rule": "first symbol = dialect; following symbols = dialect-local functions",
        "dialects": DIALECTS,
        "python_ops": PYTHON_OPS,
        "basis": basis,
        "created_at": int(time.time()),
        "axiom": "Human-readable glyphstrings are debugging projections. The artifact is the executable memory surface.",
    }
    artifact["artifact_digest"] = sha(artifact)

    path = ARTIFACTS / "core_artifact.json"
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

def choose_bin_dir():
    prefix = os.environ.get("PREFIX")
    if prefix:
        p = Path(prefix) / "bin"
        if p.exists() and os.access(p, os.W_OK):
            return p
    p = HOME / ".local" / "bin"
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_cli(runtime: Path):
    bin_dir = choose_bin_dir()
    cli = bin_dir / "glyphmatics"
    cli.write_text(
        f"""#!{sys.executable}
import runpy, sys
sys.argv[0] = "glyphmatics"
runpy.run_path({str(runtime)!r}, run_name="__main__")
""",
        encoding="utf-8",
    )
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR)
    return cli

def write_lock(runtime: Path, artifact: Path, cli: Path, glyphstring: str):
    lock = {
        "glyphstring": glyphstring,
        "runtime": str(runtime),
        "artifact": str(artifact),
        "cli": str(cli),
        "files": {
            str(runtime): sha(runtime.read_bytes()),
            str(artifact): sha(artifact.read_bytes()),
            str(cli): sha(cli.read_bytes()),
        },
        "locked_at": int(time.time()),
    }
    lock["install_digest"] = sha(lock)
    path = LOCKS / "install.lock.json"
    path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return path, lock

RUNTIME_CODE = r'''
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HOME = Path.home()
GM_HOME = HOME / ".glyphmatics"
ARTIFACTS = GM_HOME / "artifacts"
LOCKS = GM_HOME / "locks"
STATE = GM_HOME / "state"
REGISTRY = GM_HOME / "registry"
MEMORY = GM_HOME / "memory"

def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def load_artifact():
    p = ARTIFACTS / "core_artifact.json"
    if not p.exists():
        raise SystemExit("Missing core artifact. Run: python ~/install_glyphmatics.py pgvayh")
    return json.loads(p.read_text(encoding="utf-8"))

def status():
    artifact = load_artifact()
    lock_path = LOCKS / "install.lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    print(json.dumps({
        "status": "GlyphMatics installed",
        "home": str(GM_HOME),
        "artifact": str(ARTIFACTS / "core_artifact.json"),
        "version": artifact.get("version"),
        "install_glyphstring": artifact.get("install_glyphstring"),
        "artifact_digest": artifact.get("artifact_digest"),
        "install_digest": lock.get("install_digest"),
    }, indent=2, ensure_ascii=False))

def basis():
    artifact = load_artifact()
    print(json.dumps(artifact["basis"], indent=2, ensure_ascii=False))

def dialects():
    artifact = load_artifact()
    print(json.dumps(artifact["dialects"], indent=2, ensure_ascii=False))

def decode(glyphstring: str):
    artifact = load_artifact()
    if not glyphstring:
        raise SystemExit("Empty glyphstring.")

    dialect_symbol = glyphstring[0]
    ops = glyphstring[1:]

    dialect = artifact["dialects"].get(dialect_symbol, "unknown")
    decoded = {
        "glyphstring": glyphstring,
        "dialect_symbol": dialect_symbol,
        "dialect": dialect,
        "ops": [],
    }

    if dialect_symbol == "p":
        reg = artifact["python_ops"]
        for op in ops:
            decoded["ops"].append({
                "glyph": op,
                "operation": reg.get(op, "unknown_python_op"),
            })
    else:
        for op in ops:
            decoded["ops"].append({
                "glyph": op,
                "operation": "dialect_registered_no_runtime_yet",
            })

    decoded["digest"] = hashlib.sha256(
        json.dumps(decoded, sort_keys=True).encode()
    ).hexdigest()

    print(json.dumps(decoded, indent=2, ensure_ascii=False))

def verify():
    lock_path = LOCKS / "install.lock.json"
    if not lock_path.exists():
        raise SystemExit("Missing install lock.")

    lock = json.loads(lock_path.read_text())
    checks = {}

    for file, expected in lock["files"].items():
        p = Path(file)
        actual = sha_bytes(p.read_bytes()) if p.exists() else None
        checks[file] = {
            "expected": expected,
            "actual": actual,
            "match": actual == expected,
        }

    print(json.dumps({
        "verified": all(x["match"] for x in checks.values()),
        "install_digest": lock.get("install_digest"),
        "checks": checks,
    }, indent=2))

def main():
    parser = argparse.ArgumentParser(description="GlyphMatics Runtime")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")
    sub.add_parser("verify")
    sub.add_parser("basis")
    sub.add_parser("dialects")

    d = sub.add_parser("decode")
    d.add_argument("glyphstring")

    args = parser.parse_args()

    if args.cmd == "status":
        status()
    elif args.cmd == "verify":
        verify()
    elif args.cmd == "basis":
        basis()
    elif args.cmd == "dialects":
        dialects()
    elif args.cmd == "decode":
        decode(args.glyphstring)

if __name__ == "__main__":
    main()
'''

def install(glyphstring: str):
    if glyphstring != "pgvayh":
        print(f"[warn] installing with noncanonical glyphstring: {glyphstring}")

    runtime = write_runtime()
    artifact = write_artifact(glyphstring)
    cli = write_cli(runtime)
    lock_path, lock = write_lock(runtime, artifact, cli, glyphstring)

    report = {
        "status": "installed",
        "glyphstring": glyphstring,
        "glyphmatics_home": str(GM_HOME),
        "runtime": str(runtime),
        "artifact": str(artifact),
        "cli": str(cli),
        "lock": str(lock_path),
        "install_digest": lock["install_digest"],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

def main():
    glyphstring = sys.argv[1] if len(sys.argv) > 1 else "pgvayh"
    install(glyphstring)

if __name__ == "__main__":
    main()
