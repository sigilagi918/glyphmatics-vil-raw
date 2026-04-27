#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


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


class MPS:
    """
    Actual Matrix Product State backend.

    Stores n-qubit state as tensors:
      A[i].shape = (left_bond, physical_dim=2, right_bond)

    This supports real tensor operations:
      - product |0...0> initialization
      - single-qubit gate
      - adjacent two-qubit gate with SVD compression
      - norm contraction
      - small-n exact vector extraction for verification
    """

    def __init__(self, n_qubits: int, bond_dim: int):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        if bond_dim < 1:
            raise ValueError("bond_dim must be >= 1")

        self.n = n_qubits
        self.chi = bond_dim
        self.tensors = [
            np.array([[[1.0 + 0j], [0.0 + 0j]]], dtype=np.complex128)
            for _ in range(n_qubits)
        ]

    def apply_1q(self, gate: np.ndarray, site: int) -> None:
        if site < 0 or site >= self.n:
            raise IndexError("site out of range")
        A = self.tensors[site]
        self.tensors[site] = np.einsum("oi,lir->lor", gate, A)

    def apply_2q_adjacent(self, gate: np.ndarray, site: int) -> None:
        """
        Apply adjacent two-qubit gate on (site, site+1).

        gate shape: (2, 2, 2, 2)
          gate[out0, out1, in0, in1]
        """
        if site < 0 or site >= self.n - 1:
            raise IndexError("site must be in [0, n-2]")

        A = self.tensors[site]
        B = self.tensors[site + 1]

        Dl, _, Dm = A.shape
        Dm2, _, Dr = B.shape
        if Dm != Dm2:
            raise RuntimeError("MPS bond mismatch")

        theta = np.einsum("aib,bjc->aijc", A, B)
        theta = np.einsum("klij,aijc->aklc", gate, theta)

        mat = theta.reshape(Dl * 2, 2 * Dr)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        r = min(len(S), self.chi)
        U = U[:, :r]
        S = S[:r]
        Vh = Vh[:r, :]

        self.tensors[site] = U.reshape(Dl, 2, r)
        self.tensors[site + 1] = (S[:, None] * Vh).reshape(r, 2, Dr)

    def norm(self) -> float:
        E = np.array([[1.0 + 0j]], dtype=np.complex128)
        for A in self.tensors:
            E = np.einsum("ab,asi,bsj->ij", E, np.conj(A), A)
        return float(np.real_if_close(E[0, 0]))

    def max_bond(self) -> int:
        return max(max(A.shape[0], A.shape[2]) for A in self.tensors)

    def bond_dims(self) -> List[int]:
        return [int(A.shape[2]) for A in self.tensors[:-1]]

    def state_vector(self) -> np.ndarray:
        """
        Exact vector extraction. Only use for small verification.
        """
        psi = self.tensors[0][0, :, :]
        for A in self.tensors[1:]:
            psi = np.einsum("...a,ajb->...jb", psi, A)
        psi = np.squeeze(psi, axis=-1)
        return psi.reshape(-1)

    def amplitude(self, bitstring: str) -> complex:
        if len(bitstring) != self.n:
            raise ValueError("bitstring length must equal n_qubits")

        v = np.array([1.0 + 0j])
        for bit, A in zip(bitstring, self.tensors):
            idx = 1 if bit == "1" else 0
            v = np.einsum("a,ab->b", v, A[:, idx, :])
        return complex(v[0])


class GlyphMaticsTNVM:
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
        "T": "enable_tensor_network",
        "M": "initialize_mps",
        "G": "run_tensor_graph",
        "E": "error_bound_check",
    }

    def __init__(self, glyphline: str):
        self.glyphline = glyphline.strip()
        self.tokens = self.glyphline.split()
        if not self.tokens:
            raise ValueError("Empty glyphline")

        self.dialect = self.tokens[0][0]
        self.initial_ops = self.tokens[0][1:]

        if self.dialect not in self.DIALECTS:
            raise ValueError(f"Unknown dialect: {self.dialect}")

        self.args: Dict[str, Any] = {}
        self.steps: List[ActivationStep] = []
        self.mps: Optional[MPS] = None

        self.context: Dict[str, Any] = {
            "glyphline": self.glyphline,
            "dialect": self.DIALECTS[self.dialect],
            "created_at": int(time.time()),
            "artifact": None,
            "artifact_digest": None,
            "space": None,
            "backend": None,
            "memory_mounted": False,
            "verified": False,
            "sealed": False,
            "tn_compute": None,
            "error_check": None,
        }

    def sha(self, obj: Any) -> str:
        raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def parse_value(self, value: str) -> Any:
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            return value

    def activate(self) -> Dict[str, Any]:
        for glyph in self.initial_ops:
            self.run_op(glyph)

        for token in self.tokens[1:]:
            if ":" in token:
                k, v = token.split(":", 1)
                self.args[k] = self.parse_value(v)
                self.context.setdefault("args_seen", {})[k] = self.args[k]
            else:
                for glyph in token:
                    self.run_op(glyph)

        result = {
            "status": "activated",
            "glyphline": self.glyphline,
            "dialect": self.DIALECTS[self.dialect],
            "args": self.args,
            "context": self.context,
            "steps": [asdict(s) for s in self.steps],
        }
        result["activation_digest"] = self.sha(result)

        out = STATE / "last_tn_activation.json"
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\n=== TN ACTIVATION RESULT ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    def step(self, glyph: str, name: str, ok: bool, detail: str) -> None:
        self.steps.append(ActivationStep(glyph, name, ok, detail))
        print(f"[{glyph}] {name}: {detail}")

    def run_op(self, glyph: str) -> None:
        if self.dialect != "p":
            self.step(glyph, "unsupported_dialect", False, f"{self.dialect} registered but not active")
            return

        op = self.PY_OPS.get(glyph)
        if not op:
            self.step(glyph, "unknown_op", False, f"Unknown Python glyph: {glyph}")
            return

        getattr(self, f"op_{op}")(glyph)

    # ---------- base artifact ops ----------

    def op_hydrate_artifact(self, glyph: str) -> None:
        artifact = ARTIFACTS / "core_artifact.json"
        if artifact.exists():
            self.context["artifact"] = str(artifact)
            self.context["artifact_digest"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
            self.step(glyph, "hydrate_artifact", True, f"Loaded {artifact}")
        else:
            self.step(glyph, "hydrate_artifact", True, "No core artifact found; standalone mode")

    def op_create_or_collapse(self, glyph: str) -> None:
        if self.context["space"] is None:
            self.context["space"] = {
                "name": "GlyphMatics_TN_Execution_Space",
                "mode": "mps_tensor_network",
            }
            self.step(glyph, "create_space", True, "Created TN execution space")
        else:
            collapse = {
                "glyphline": self.glyphline,
                "args": self.args,
                "backend": self.context["backend"],
                "tn_compute": self.context["tn_compute"],
                "error_check": self.context["error_check"],
            }
            collapse["digest"] = self.sha(collapse)
            path = MEMORY / f"tn_collapse_{collapse['digest'][:16]}.json"
            path.write_text(json.dumps(collapse, indent=2, ensure_ascii=False), encoding="utf-8")
            self.context["last_collapse"] = str(path)
            self.step(glyph, "collapse_to_memory", True, f"Collapsed TN state to {path}")

    def op_build_backend(self, glyph: str) -> None:
        self.context["backend"] = {
            "type": "tensor_network_mps",
            "enabled": False,
            "n_qubits": None,
            "bond_dim": None,
            "memory_model": "O(n * chi^2)",
        }
        self.step(glyph, "build_backend", True, "Built MPS backend shell")

    def op_mount_memory(self, glyph: str) -> None:
        self.context["memory_mounted"] = True
        self.context["memory_path"] = str(MEMORY)
        self.step(glyph, "mount_memory", True, f"Mounted {MEMORY}")

    def op_execute_chain(self, glyph: str) -> None:
        self.context["executed"] = True
        self.step(glyph, "execute_chain", True, "Safe activation chain executed")

    def op_generate(self, glyph: str) -> None:
        generated = {
            "glyphline": self.glyphline,
            "args": self.args,
            "context_digest": self.sha(self.context),
        }
        path = ARTIFACTS / "tn_generated_artifact.json"
        path.write_text(json.dumps(generated, indent=2, ensure_ascii=False), encoding="utf-8")
        self.context["generated_artifact"] = str(path)
        self.step(glyph, "generate", True, f"Generated {path}")

    def op_validate(self, glyph: str) -> None:
        py_ok = sys.version_info >= (3, 10)
        backend_ok = self.context["backend"] is not None
        tn_ok = True

        if self.context["error_check"] is not None:
            tn_ok = bool(self.context["error_check"].get("passed"))

        verified = bool(py_ok and backend_ok and tn_ok)
        self.context["verified"] = verified

        self.step(
            glyph,
            "validate",
            verified,
            f"python_ok={py_ok}, backend_ok={backend_ok}, tn_error_ok={tn_ok}",
        )

    # ---------- real TN ops ----------

    def op_enable_tensor_network(self, glyph: str) -> None:
        if self.context["backend"] is None:
            self.op_build_backend("b")

        n = int(self.args.get("tn", 20))
        chi = int(self.args.get("b", 64))

        self.context["backend"].update({
            "enabled": True,
            "n_qubits": n,
            "bond_dim": chi,
            "estimated_complex_entries": n * chi * chi,
        })

        self.step(
            glyph,
            "enable_tensor_network",
            True,
            f"Enabled MPS backend n={n}, chi={chi}, estimated_entries={n * chi * chi}",
        )

    def op_initialize_mps(self, glyph: str) -> None:
        if self.context["backend"] is None or not self.context["backend"].get("enabled"):
            self.op_enable_tensor_network("T")

        n = int(self.context["backend"]["n_qubits"])
        chi = int(self.context["backend"]["bond_dim"])

        self.mps = MPS(n, chi)

        self.context["tn_compute"] = {
            "initialized": True,
            "n_qubits": n,
            "bond_dim": chi,
            "initial_norm": self.mps.norm(),
            "initial_max_bond": self.mps.max_bond(),
        }

        self.step(glyph, "initialize_mps", True, f"Initialized |0...0> MPS n={n}, chi={chi}")

    def op_run_tensor_graph(self, glyph: str) -> None:
        if self.mps is None:
            self.op_initialize_mps("M")

        H = (1.0 / math.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)

        CNOT = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.complex128,
        ).reshape(2, 2, 2, 2)

        # Real low-entanglement TN circuit:
        # H on qubit 0, then adjacent CNOT ladder creates GHZ:
        # (|00...0> + |11...1>) / sqrt(2)
        self.mps.apply_1q(H, 0)
        for site in range(self.mps.n - 1):
            self.mps.apply_2q_adjacent(CNOT, site)

        zeros = "0" * self.mps.n
        ones = "1" * self.mps.n

        amp_zero = self.mps.amplitude(zeros)
        amp_one = self.mps.amplitude(ones)

        compute = self.context["tn_compute"] or {}
        compute.update({
            "circuit": "GHZ_MPS_TEST",
            "description": "H(0) followed by CNOT ladder creates low-entanglement GHZ state",
            "norm": self.mps.norm(),
            "max_bond": self.mps.max_bond(),
            "bond_dims": self.mps.bond_dims(),
            "prob_all_zero": float(abs(amp_zero) ** 2),
            "prob_all_one": float(abs(amp_one) ** 2),
            "nonzero_expected": ["0" * self.mps.n, "1" * self.mps.n],
        })
        self.context["tn_compute"] = compute

        self.step(
            glyph,
            "run_tensor_graph",
            True,
            f"Ran GHZ MPS circuit; norm={compute['norm']:.12f}, max_bond={compute['max_bond']}",
        )

    def op_error_bound_check(self, glyph: str) -> None:
        # Verify same circuit against exact closed-form GHZ for small n.
        test_n = min(8, int(self.args.get("tn", 20)))
        chi = int(self.args.get("b", 64))

        mps = MPS(test_n, chi)

        H = (1.0 / math.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
        CNOT = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.complex128,
        ).reshape(2, 2, 2, 2)

        mps.apply_1q(H, 0)
        for site in range(test_n - 1):
            mps.apply_2q_adjacent(CNOT, site)

        psi = mps.state_vector()
        exact = np.zeros(1 << test_n, dtype=np.complex128)
        exact[0] = 1.0 / math.sqrt(2.0)
        exact[-1] = 1.0 / math.sqrt(2.0)

        l2_error = float(np.linalg.norm(psi - exact))
        norm_error = abs(float(np.sum(np.abs(psi) ** 2)) - 1.0)

        check = {
            "test_n": test_n,
            "bond_dim": chi,
            "reference": "exact_GHZ_closed_form",
            "l2_error": l2_error,
            "norm_error": norm_error,
            "threshold": 1e-9,
            "passed": bool(l2_error < 1e-9 and norm_error < 1e-9),
        }

        self.context["error_check"] = check

        self.step(
            glyph,
            "error_bound_check",
            check["passed"],
            f"small-n exact check l2_error={l2_error:.3e}, norm_error={norm_error:.3e}",
        )

    # ---------- registry / seal ----------

    def op_register(self, glyph: str) -> None:
        reg = {
            "kind": "tn_activation",
            "glyphline": self.glyphline,
            "args": self.args,
            "backend": self.context["backend"],
            "tn_compute": self.context["tn_compute"],
            "error_check": self.context["error_check"],
        }
        reg["digest"] = self.sha(reg)

        path = REGISTRY / f"tn_reg_{reg['digest'][:16]}.json"
        path.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")

        self.context["registered"] = str(path)
        self.step(glyph, "register", True, f"Registered {path}")

    def op_update_registry(self, glyph: str) -> None:
        index_path = REGISTRY / "tn_index.json"
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
            "kind": "tn_g14_seal",
            "glyphline": self.glyphline,
            "args": self.args,
            "context": self.context,
            "steps": [asdict(s) for s in self.steps],
        }
        seal["seal_digest"] = self.sha(seal)

        path = LOCKS / f"tn_seal_{seal['seal_digest'][:16]}.json"
        path.write_text(json.dumps(seal, indent=2, ensure_ascii=False), encoding="utf-8")

        self.context["sealed"] = True
        self.context["seal"] = str(path)
        self.context["seal_digest"] = seal["seal_digest"]

        self.step(glyph, "seal_definition", True, f"G14 TN seal written {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GlyphMatics Tensor Network VM")
    parser.add_argument(
        "glyphline",
        nargs=argparse.REMAINDER,
        help="Example: p hcbme tn:20 b:64 TMGE cv ruz",
    )
    args = parser.parse_args()

    glyphline = " ".join(args.glyphline).strip()
    if not glyphline:
        glyphline = "p hcbme tn:20 b:64 TMGE cv ruz"

    vm = GlyphMaticsTNVM(glyphline)
    vm.activate()


if __name__ == "__main__":
    main()
