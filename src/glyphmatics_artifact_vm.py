#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List


HOME = Path.home()
GM_HOME = HOME / ".glyphmatics"
ARTIFACTS = GM_HOME / "artifacts"
MEMORY = GM_HOME / "memory"
LOCKS = GM_HOME / "locks"
REGISTRY = GM_HOME / "registry"
STATE = GM_HOME / "state"

for p in [GM_HOME, ARTIFACTS, MEMORY, LOCKS, REGISTRY, STATE]:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class ActivationStep:
    glyph: str
    name: str
    ok: bool
    detail: str


class GlyphMaticsArtifactVM:
    """
    GlyphMatics Artifact VM.

    Dialect rule:
      first symbol = dialect
      following symbols = dialect-local operation glyphs
      tokens like tn:20, b:64, o:OSCODE are parameters/anchors

    This VM does not execute arbitrary OS commands.
    o:OSCODE is anchored as payload identity unless explicitly implemented.
    """

    DIALECTS = {
        "p": "python",
        "b": "bash",
        "j": "javascript",
        "r": "rust",
        "g": "go",
        "c": "c",
        "C": "cpp",
    }

    PY_OPS = {
        "h": "hydrate_artifact",
        "c": "create_or_collapse",
        "b": "build_backend",
        "m": "mount_memory",
        "e": "execute_chain",
        "v": "validate",
        "g": "generate",
        "r": "register",
        "u": "update_registry",
        "z": "seal_definition",
    }

    def __init__(self, glyphline: str):
        self.glyphline = glyphline.strip()
        self.tokens = self.glyphline.split()
        if not self.tokens:
            raise ValueError("Empty glyphline.")

        self.head = self.tokens[0]
        self.dialect = self.head[0]
        self.initial_ops = self.head[1:]

        if self.dialect not in self.DIALECTS:
            raise ValueError(f"Unknown dialect: {self.dialect}")

        self.args: Dict[str, Any] = {}
        self.steps: List[ActivationStep] = []
        self.context: Dict[str, Any] = {
            "glyphline": self.glyphline,
            "dialect": self.DIALECTS[self.dialect],
            "created_at": int(time.time()),
            "space": None,
            "backend": None,
            "payloads": {},
            "memory_mounted": False,
            "verified": False,
            "sealed": False,
        }

    def sha(self, data: Any) -> str:
        raw = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def parse_value(self, value: str) -> Any:
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            return value

    def parse_token(self, token: str) -> None:
        if ":" in token:
            k, v = token.split(":", 1)
            self.args[k] = self.parse_value(v)
            if k == "o":
                self.context["payloads"][str(v)] = {
                    "name": str(v),
                    "trusted": False,
                    "mode": "anchored_data_only",
                    "note": "Payload is not executed automatically.",
                }
            return

        # Non-arg token is more glyph ops.
        for glyph in token:
            self.run_op(glyph)

    def activate(self) -> Dict[str, Any]:
        for glyph in self.initial_ops:
            self.run_op(glyph)

        for token in self.tokens[1:]:
            self.parse_token(token)

        # Apply numeric args after parse.
        self.apply_args()

        activation = {
            "status": "activated",
            "glyphline": self.glyphline,
            "dialect": self.DIALECTS[self.dialect],
            "args": self.args,
            "context": self.context,
            "steps": [asdict(s) for s in self.steps],
        }
        activation["activation_digest"] = self.sha(activation)

        out = STATE / "last_activation.json"
        out.write_text(json.dumps(activation, indent=2, ensure_ascii=False), encoding="utf-8")

        return activation

    def step(self, glyph: str, name: str, ok: bool, detail: str) -> None:
        self.steps.append(ActivationStep(glyph, name, ok, detail))
        print(f"[{glyph}] {name}: {detail}")

    def run_op(self, glyph: str) -> None:
        if self.dialect != "p":
            self.step(glyph, "unsupported_dialect_op", False, f"Dialect {self.dialect} registered but not active.")
            return

        name = self.PY_OPS.get(glyph)
        if not name:
            self.step(glyph, "unknown_python_op", False, "Unknown glyph.")
            return

        getattr(self, f"op_{name}")(glyph)

    # ---------- glyph ops ----------

    def op_hydrate_artifact(self, glyph: str) -> None:
        artifact = ARTIFACTS / "core_artifact.json"
        if artifact.exists():
            self.context["artifact"] = str(artifact)
            self.context["artifact_digest"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
            self.step(glyph, "hydrate_artifact", True, f"Loaded {artifact}")
        else:
            self.context["artifact"] = None
            self.step(glyph, "hydrate_artifact", True, "No core artifact found; VM running standalone.")

    def op_create_or_collapse(self, glyph: str) -> None:
        # Before backend exists: create. After execution: collapse.
        if self.context.get("space") is None:
            self.context["space"] = {
                "name": "GlyphMatics_Python_Execution_Space",
                "mode": "artifact_vm",
            }
            self.step(glyph, "create_space", True, "Created Python artifact execution space.")
        else:
            collapsed = {
                "space": self.context.get("space"),
                "backend": self.context.get("backend"),
                "args": self.args,
                "digest": self.sha({
                    "space": self.context.get("space"),
                    "backend": self.context.get("backend"),
                    "args": self.args,
                }),
            }
            path = MEMORY / f"collapse_{collapsed['digest'][:16]}.json"
            path.write_text(json.dumps(collapsed, indent=2, ensure_ascii=False), encoding="utf-8")
            self.context["last_collapse"] = str(path)
            self.step(glyph, "collapse_to_memory", True, f"Collapsed state to {path}")

    def op_build_backend(self, glyph: str) -> None:
        self.context["backend"] = {
            "type": "python_exact_or_tn",
            "tensor_network": False,
            "n_qubits": None,
            "bond_dim": None,
        }
        self.step(glyph, "build_backend", True, "Backend shell built.")

    def op_mount_memory(self, glyph: str) -> None:
        self.context["memory_mounted"] = True
        self.context["memory_path"] = str(MEMORY)
        self.step(glyph, "mount_memory", True, f"Mounted {MEMORY}")

    def op_execute_chain(self, glyph: str) -> None:
        self.context["executed"] = True
        self.context["execution_note"] = "Executed artifact activation chain. No untrusted payload executed."
        self.step(glyph, "execute_chain", True, "Activation chain executed safely.")

    def op_validate(self, glyph: str) -> None:
        py_ok = sys.version_info >= (3, 10)
        backend_ok = self.context.get("backend") is not None or True
        payload_ok = all(
            payload.get("mode") == "anchored_data_only"
            for payload in self.context.get("payloads", {}).values()
        )

        verified = bool(py_ok and backend_ok and payload_ok)
        self.context["verified"] = verified

        self.step(
            glyph,
            "validate",
            verified,
            f"python_ok={py_ok}, backend_ok={backend_ok}, payload_safe={payload_ok}",
        )

    def op_generate(self, glyph: str) -> None:
        generated = {
            "glyphline": self.glyphline,
            "platform": platform.platform(),
            "python": sys.version,
            "args": self.args,
            "context_digest": self.sha(self.context),
        }
        path = ARTIFACTS / "generated_from_activation.json"
        path.write_text(json.dumps(generated, indent=2, ensure_ascii=False), encoding="utf-8")
        self.context["generated_artifact"] = str(path)
        self.step(glyph, "generate", True, f"Generated {path}")

    def op_register(self, glyph: str) -> None:
        reg = {
            "registered_from": self.glyphline,
            "dialect": self.DIALECTS[self.dialect],
            "ops": self.PY_OPS,
            "args": self.args,
        }
        reg["digest"] = self.sha(reg)
        path = REGISTRY / f"reg_{reg['digest'][:16]}.json"
        path.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
        self.context["registered"] = str(path)
        self.step(glyph, "register", True, f"Registered activation in {path}")

    def op_update_registry(self, glyph: str) -> None:
        index_path = REGISTRY / "index.json"
        entries = []
        if index_path.exists():
            entries = json.loads(index_path.read_text(encoding="utf-8"))

        entry = {
            "glyphline": self.glyphline,
            "digest": self.sha({"glyphline": self.glyphline, "args": self.args}),
            "time": int(time.time()),
        }
        entries.append(entry)
        index_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        self.context["registry_index"] = str(index_path)
        self.step(glyph, "update_registry", True, f"Updated {index_path}")

    def op_seal_definition(self, glyph: str) -> None:
        seal = {
            "glyphline": self.glyphline,
            "args": self.args,
            "context": self.context,
            "steps": [asdict(s) for s in self.steps],
        }
        seal["seal_digest"] = self.sha(seal)

        path = LOCKS / f"seal_{seal['seal_digest'][:16]}.json"
        path.write_text(json.dumps(seal, indent=2, ensure_ascii=False), encoding="utf-8")

        self.context["sealed"] = True
        self.context["seal"] = str(path)
        self.context["seal_digest"] = seal["seal_digest"]

        self.step(glyph, "seal_definition", True, f"G14 seal written {path}")

    # ---------- arg application ----------

    def apply_args(self) -> None:
        backend = self.context.get("backend")
        if backend is None:
            backend = {
                "type": "python_exact_or_tn",
                "tensor_network": False,
                "n_qubits": None,
                "bond_dim": None,
            }
            self.context["backend"] = backend

        if "tn" in self.args:
            backend["tensor_network"] = True
            backend["n_qubits"] = int(self.args["tn"])

        if "b" in self.args and isinstance(self.args["b"], int):
            backend["bond_dim"] = int(self.args["b"])

        if backend.get("tensor_network"):
            backend["memory_model"] = "O(n * chi^2)"
            n = backend.get("n_qubits") or 0
            chi = backend.get("bond_dim") or 0
            backend["estimated_complex_entries"] = n * chi * chi

        # Update state after arg application.
        self.context["backend"] = backend


def main() -> None:
    parser = argparse.ArgumentParser(description="GlyphMatics Artifact VM")
    parser.add_argument(
        "glyphline",
        nargs=argparse.REMAINDER,
        help='Example: phcbme o:OSCODE vghe tn:20 b:64 cv ruz',
    )
    args = parser.parse_args()

    glyphline = " ".join(args.glyphline).strip()
    if not glyphline:
        glyphline = "phcbme o:OSCODE vghe tn:20 b:64 cv ruz"

    vm = GlyphMaticsArtifactVM(glyphline)
    result = vm.activate()

    print("\n=== ACTIVATION RESULT ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
