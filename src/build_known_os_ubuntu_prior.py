#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

HOME = Path.home()

DEFAULT_UBUNTU_ROOTS = [
    Path("/data/data/com.termux/files/usr/var/lib/proot-distro/installed-rootfs/ubuntu"),
    HOME / "ubuntu",
    HOME / "Ubuntu",
    HOME / "ubuntu-rootfs",
    HOME / "rootfs/ubuntu",
    HOME / "proot/ubuntu",
]

SAFE_READ_RELATIVE = {
    "etc/os-release",
    "etc/lsb-release",
    "etc/debian_version",
    "etc/apt/sources.list",
    "etc/apt/sources.list.d",
    "var/lib/dpkg/status",
    "var/lib/apt/lists",
}

SKIP_PARTS = {
    "home",
    "root",
    "proc",
    "sys",
    "dev",
    "run",
    "tmp",
    "mnt",
    "media",
    "lost+found",
    ".cache",
    ".ssh",
    "var/log",
    "var/cache",
    "var/tmp",
}

SENSITIVE_PATTERNS = [
    "password",
    "passwd",
    "shadow",
    "secret",
    "token",
    "credential",
    "private",
    "id_rsa",
    "id_ed25519",
    "authorized_keys",
    "known_hosts",
    "history",
]

OS_LAYERS = {
    "boot": ["boot", "vmlinuz", "initrd", "grub"],
    "package_manager": ["apt", "dpkg", "sources.list", "preferences", "trusted.gpg"],
    "base_config": ["etc", "os-release", "lsb-release", "debian_version", "hostname", "hosts"],
    "shell_runtime": ["bash", "zsh", "sh", "profile", "bashrc"],
    "coreutils": ["bin", "usr/bin", "coreutils", "ls", "cat", "grep", "sed", "awk"],
    "system_services": ["systemd", "service", "init.d", "unit", ".service"],
    "networking": ["network", "netplan", "resolv", "ssh", "ssl", "ca-certificates"],
    "libraries": ["lib", "usr/lib", ".so", "ld-linux"],
    "headers_devel": ["include", "headers", ".h", "pkgconfig", "cmake"],
    "python_runtime": ["python", "site-packages", "dist-packages", "pip"],
    "compiler_toolchain": ["gcc", "g++", "clang", "make", "cmake", "ld", "as"],
    "docs_manpages": ["share/doc", "share/man", "man1", "man8"],
    "locale_timezone": ["locale", "zoneinfo", "timezone", "localtime"],
    "security": ["ssl", "cert", "apparmor", "sudoers", "pam", "polkit"],
}

VERTICAL_OS_STAGES = {
    "identity": ["os-release", "lsb-release", "debian_version"],
    "package_sources": ["sources.list", "apt/sources", "trusted.gpg", "keyrings"],
    "package_database": ["dpkg/status", "apt/lists", "available"],
    "base_binaries": ["/bin/", "/usr/bin/", "/sbin/", "/usr/sbin/"],
    "libraries": ["/lib/", "/usr/lib/", ".so"],
    "configuration": ["/etc/"],
    "services": ["systemd", ".service", "init.d"],
    "development": ["include", "pkgconfig", "cmake", "gcc", "clang"],
    "documentation": ["share/doc", "share/man"],
    "runtime_state_excluded": ["/proc/", "/sys/", "/dev/", "/run/", "/tmp/"],
}

TEXT_EXTS = {
    ".conf", ".list", ".sources", ".service", ".timer", ".socket", ".target",
    ".sh", ".bash", ".profile", ".ini", ".cfg", ".json", ".yaml", ".yml",
    ".txt", ".md", ".desktop", ".pc", ".cmake",
}

MAX_READ_BYTES = 512_000


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha_file_sample(path: Path, max_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()


def is_sensitive(path: Path) -> bool:
    s = str(path).lower()
    return any(x in s for x in SENSITIVE_PATTERNS)


def rel_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def should_skip(root: Path, path: Path) -> bool:
    rel = rel_path(root, path)
    parts = set(rel.split("/"))

    if is_sensitive(path):
        return True

    for sp in SKIP_PARTS:
        if "/" in sp:
            if sp in rel:
                return True
        elif sp in parts:
            return True

    return False


def file_kind(path: Path) -> str:
    try:
        mode = path.lstat().st_mode
    except Exception:
        return "unknown"

    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "dir"
    if stat.S_ISREG(mode):
        if os.access(path, os.X_OK):
            return "executable"
        return "file"
    return "special"


def classify_layers(rel: str) -> List[str]:
    s = rel.lower()
    layers = []
    for layer, keys in OS_LAYERS.items():
        if any(k.lower() in s for k in keys):
            layers.append(layer)
    return layers or ["general_os"]


def classify_stages(rel: str) -> List[str]:
    s = "/" + rel.lower()
    stages = []
    for stage, keys in VERTICAL_OS_STAGES.items():
        if any(k.lower() in s for k in keys):
            stages.append(stage)
    return stages or ["filesystem"]


def read_small_text(path: Path) -> str:
    data = path.read_bytes()[:MAX_READ_BYTES]
    return data.decode("utf-8", errors="replace")


def parse_os_release(root: Path) -> Dict[str, str]:
    out = {}
    p = root / "etc/os-release"
    if not p.exists():
        return out

    for line in read_small_text(p).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k] = v.strip().strip('"')
    return out


def parse_dpkg_status(root: Path, max_packages: int = 20000) -> Dict[str, Any]:
    p = root / "var/lib/dpkg/status"
    if not p.exists() or is_sensitive(p):
        return {
            "present": False,
            "package_count": 0,
            "packages_sample": [],
            "section_counts": {},
            "priority_counts": {},
        }

    text = read_small_text(p)
    blocks = re.split(r"\n\s*\n", text)
    packages = []
    section_counts = Counter()
    priority_counts = Counter()
    arch_counts = Counter()

    for block in blocks[:max_packages]:
        item = {}
        for line in block.splitlines():
            if ": " not in line:
                continue
            k, v = line.split(": ", 1)
            if k in {"Package", "Version", "Architecture", "Section", "Priority", "Essential"}:
                item[k.lower()] = v.strip()

        if "package" in item:
            packages.append(item)
            section_counts[item.get("section", "unknown")] += 1
            priority_counts[item.get("priority", "unknown")] += 1
            arch_counts[item.get("architecture", "unknown")] += 1

    return {
        "present": True,
        "package_count": len(packages),
        "packages_sample": packages[:500],
        "truncated": len(packages) > 500,
        "section_counts": dict(section_counts),
        "priority_counts": dict(priority_counts),
        "architecture_counts": dict(arch_counts),
        "status_sha256": sha_text(text),
    }


def parse_apt_sources(root: Path) -> Dict[str, Any]:
    paths = []
    one = root / "etc/apt/sources.list"
    if one.exists() and one.is_file() and not is_sensitive(one):
        paths.append(one)

    d = root / "etc/apt/sources.list.d"
    if d.exists() and d.is_dir():
        for p in sorted(d.glob("*")):
            if p.is_file() and not is_sensitive(p):
                paths.append(p)

    entries = []
    domains = Counter()

    for p in paths:
        try:
            text = read_small_text(p)
        except Exception:
            continue

        for line in text.splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            entries.append({
                "file": rel_path(root, p),
                "line_sha256": sha_text(raw),
                "line_sample": raw[:240],
            })
            m = re.search(r"https?://([^/\s]+)", raw)
            if m:
                domains[m.group(1)] += 1

    return {
        "source_file_count": len(paths),
        "entry_count": len(entries),
        "entries": entries[:200],
        "truncated": len(entries) > 200,
        "domain_counts": dict(domains),
    }


def scan_filesystem(root: Path, max_files: int = 0) -> Dict[str, Any]:
    records = []
    dir_counts = Counter()
    ext_counts = Counter()
    kind_counts = Counter()
    layer_map = defaultdict(list)
    stage_map = defaultdict(list)
    executable_names = Counter()
    config_names = Counter()
    service_names = Counter()

    count = 0

    for path in root.rglob("*"):
        if max_files and count >= max_files:
            break

        if should_skip(root, path):
            continue

        rel = rel_path(root, path)
        kind = file_kind(path)
        kind_counts[kind] += 1

        top = rel.split("/", 1)[0] if rel else "."
        dir_counts[top] += 1

        ext = path.suffix.lower()
        if ext:
            ext_counts[ext] += 1

        layers = classify_layers(rel)
        stages = classify_stages(rel)

        for layer in layers:
            layer_map[layer].append(rel)
        for stage in stages:
            stage_map[stage].append(rel)

        size = 0
        sample_hash = None
        text_hash = None

        if path.is_file():
            try:
                size = path.stat().st_size
                sample_hash = sha_file_sample(path, 256 * 1024)
            except Exception:
                pass

            name = path.name
            if kind == "executable":
                executable_names[name] += 1
            if ext in TEXT_EXTS or "/etc/" in "/" + rel:
                config_names[name] += 1
            if name.endswith(".service"):
                service_names[name] += 1

            if size <= MAX_READ_BYTES and (ext in TEXT_EXTS or rel in SAFE_READ_RELATIVE):
                try:
                    text_hash = sha_text(read_small_text(path))
                except Exception:
                    text_hash = None

        records.append({
            "path": rel,
            "kind": kind,
            "extension": ext,
            "size_bytes": size,
            "sample_sha256_first_256kb": sample_hash,
            "text_sha256_if_safe_small": text_hash,
            "layers": layers,
            "vertical_stages": stages,
        })
        count += 1

    return {
        "records": records,
        "dir_counts": dict(dir_counts),
        "extension_counts": dict(ext_counts),
        "kind_counts": dict(kind_counts),
        "layer_map": {k: v[:1000] for k, v in sorted(layer_map.items())},
        "stage_map": {k: v[:1000] for k, v in sorted(stage_map.items())},
        "executable_names_top": dict(executable_names.most_common(300)),
        "config_names_top": dict(config_names.most_common(300)),
        "service_names_top": dict(service_names.most_common(300)),
    }


def glyph_token(i: int, prefix: str) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    a = alphabet[(i // 26) % 26]
    b = alphabet[i % 26]
    return f"{prefix}{a}{b}"


def build_dictionary(scan: Dict[str, Any], dpkg: Dict[str, Any], apt: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    i = 0

    for name, count in scan["executable_names_top"].items():
        items.append({"glyph": glyph_token(i, "U"), "kind": "ubuntu_executable", "value": name, "count": count})
        i += 1

    for name, count in scan["config_names_top"].items():
        items.append({"glyph": glyph_token(i, "U"), "kind": "ubuntu_config_name", "value": name, "count": count})
        i += 1

    for name, count in scan["service_names_top"].items():
        items.append({"glyph": glyph_token(i, "U"), "kind": "ubuntu_systemd_service", "value": name, "count": count})
        i += 1

    for section, count in dpkg.get("section_counts", {}).items():
        items.append({"glyph": glyph_token(i, "U"), "kind": "ubuntu_package_section", "value": section, "count": count})
        i += 1

    for domain, count in apt.get("domain_counts", {}).items():
        items.append({"glyph": glyph_token(i, "U"), "kind": "ubuntu_apt_domain", "value": domain, "count": count})
        i += 1

    return items


def find_ubuntu_roots(explicit: List[str]) -> List[Path]:
    roots = [Path(x).expanduser() for x in explicit] if explicit else DEFAULT_UBUNTU_ROOTS
    found = []

    for r in roots:
        if not r.exists() or not r.is_dir():
            continue

        if (r / "etc/os-release").exists():
            found.append(r)
            continue

        # Allow parent dirs with named Ubuntu rootfs inside.
        for child in r.glob("*"):
            if child.is_dir() and (child / "etc/os-release").exists():
                try:
                    text = (child / "etc/os-release").read_text(errors="replace").lower()
                except Exception:
                    text = ""
                if "ubuntu" in text:
                    found.append(child)

    uniq = []
    seen = set()
    for r in found:
        rp = str(r.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(r.resolve())

    return uniq


def build(root: Path, out_dir: Path, max_files: int = 0) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    os_release = parse_os_release(root)
    dpkg = parse_dpkg_status(root)
    apt = parse_apt_sources(root)
    fs = scan_filesystem(root, max_files=max_files)
    dictionary = build_dictionary(fs, dpkg, apt)

    total_indexed_bytes = sum(x.get("size_bytes", 0) for x in fs["records"])

    glyphlines = [
        "pUBUNTUZvhz",
        "MODE KNOWN_OS_UBUNTU_PRIOR",
        "RULE PUBLIC_OS_STRUCTURE_NOT_FULL_BYTE_COPY",
        "NO_HOME",
        "NO_ROOT_USER_DATA",
        "NO_SECRETS",
        "NO_LOGS",
        "NO_PROC_SYS_DEV_RUNTIME",
        "NO_RAW_PACKAGE_ARCHIVE_COPY",
        "NO_BASE64",
        "NO_HIDDEN_PAYLOAD",
        "NO_BYTE_TO_BRAILLE",
        "NO_BYTE_TO_HANZI",
        f"ROOT {root}",
        f"OS_ID {os_release.get('ID', 'unknown')}",
        f"VERSION_ID {os_release.get('VERSION_ID', 'unknown')}",
        f"FILES_INDEXED {len(fs['records'])}",
        f"PACKAGES {dpkg.get('package_count', 0)}",
        f"INDEXED_BYTES {total_indexed_bytes}",
        f"DICTIONARY {len(dictionary)}",
    ]

    for item in dictionary:
        glyphlines.append(json.dumps(item, ensure_ascii=False, sort_keys=True))

    glyphline_text = "\n".join(glyphlines) + "\n"
    glyphline_bytes = len(glyphline_text.encode("utf-8"))
    ratio = total_indexed_bytes / glyphline_bytes if glyphline_bytes else 0.0
    reduction = 100.0 - ((glyphline_bytes / total_indexed_bytes) * 100.0) if total_indexed_bytes else 0.0

    prior = {
        "format": "GlyphMatics Known OS Ubuntu Prior",
        "version": "0.9.2-dev",
        "root": str(root),
        "created_at": int(time.time()),
        "boundary": "Ubuntu public OS structure only. User data, secrets, logs, and runtime pseudo-filesystems excluded.",
        "os_release": os_release,
        "apt_sources": apt,
        "dpkg_status": dpkg,
        "filesystem": fs,
        "dictionary": dictionary,
    }

    summary = {
        "format": "GlyphMatics Known OS Ubuntu Prior Summary",
        "version": "0.9.2-dev",
        "created_at": int(time.time()),
        "root": str(root),
        "os_pretty_name": os_release.get("PRETTY_NAME"),
        "os_id": os_release.get("ID"),
        "version_id": os_release.get("VERSION_ID"),
        "files_indexed": len(fs["records"]),
        "packages_indexed": dpkg.get("package_count", 0),
        "indexed_bytes": total_indexed_bytes,
        "glyphline_bytes": glyphline_bytes,
        "semantic_structure_ratio_vs_indexed_bytes": ratio,
        "semantic_structure_reduction_percent": reduction,
        "dictionary_entries": len(dictionary),
        "boundary": "Known OS prior, structural only, no private/user/runtime data.",
    }

    digest_source = json.dumps({"summary": summary, "prior": prior}, sort_keys=True, ensure_ascii=False)
    summary["dataset_digest"] = sha_text(digest_source)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "known_os_ubuntu_prior.json").write_text(json.dumps(prior, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "known_os_ubuntu_prior.glyphlines.txt").write_text(glyphline_text, encoding="utf-8")

    return {"status": "built", "out_dir": str(out_dir), **summary}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GlyphMatics known OS Ubuntu structural prior.")
    ap.add_argument("--root", action="append", default=[], help="Ubuntu rootfs path. Can repeat.")
    ap.add_argument("--out", default=str(HOME / "glyphmatics_known_os_ubuntu_prior"))
    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()

    roots = find_ubuntu_roots(args.root)
    if not roots:
        raise SystemExit(
            "No Ubuntu rootfs found. Install one with proot-distro or pass --root /path/to/ubuntu-rootfs"
        )

    # Build first detected Ubuntu root.
    result = build(roots[0], Path(args.out).expanduser(), max_files=args.max_files)
    result["detected_roots"] = [str(x) for x in roots]
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
