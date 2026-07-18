"""Compatibility writer that upgrades governed Gate 0 runs to receipt v2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from smii.pipelines.refinement_policy import canonical_hash

from .body_carrier_receipt import BodyCarrierReceipt as LegacyBodyCarrierReceipt
from .body_carrier_v2_builder import build_body_carrier_receipt_v2
from .body_carrier_v2_policy import BodyDecision, Severity


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_mesh(parent: Path, expected_hash: str, preferred_name: str) -> Path:
    candidates = [parent / preferred_name, *sorted(parent.glob("*.npz"))]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        if _sha256_path(candidate) == expected_hash:
            return candidate
    raise FileNotFoundError(f"Could not resolve mesh checkpoint for hash {expected_hash}")


def _mesh_evidence(path: Path) -> tuple[dict[str, int], bool, bool]:
    with np.load(path, allow_pickle=False) as payload:
        if "vertices" not in payload or "faces" not in payload:
            raise KeyError(f"Mesh checkpoint {path} must contain vertices and faces")
        vertices = np.asarray(payload["vertices"])
        faces = np.asarray(payload["faces"])

    geometry_finite = bool(
        vertices.ndim == 2
        and vertices.shape[1:] == (3,)
        and np.isfinite(vertices).all()
    )
    face_values_finite = bool(np.isfinite(faces).all()) if np.issubdtype(
        faces.dtype, np.number
    ) else False
    topology_valid = bool(
        geometry_finite
        and faces.ndim == 2
        and faces.shape[1:] == (3,)
        and faces.shape[0] > 0
        and vertices.shape[0] > 0
        and face_values_finite
        and np.equal(faces, np.floor(faces)).all()
        and int(np.min(faces)) >= 0
        and int(np.max(faces)) < vertices.shape[0]
    )
    topology = {
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
    }
    return topology, geometry_finite, topology_valid


def _fit_payload(parent: Path) -> Mapping[str, Any] | None:
    preferred = parent / "afflec_measurement_fit.json"
    candidates = [preferred, *sorted(parent.glob("*_measurement_fit.json"))]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping) and isinstance(
            payload.get("refinement_receipt"), Mapping
        ):
            return payload
    return None


def _severity(payload: Mapping[str, Any]) -> Severity:
    status = str(payload.get("consistency_status", "")).upper()
    flags = payload.get("consistency_flags", ())
    if status == "FAIL":
        return "fail"
    if status == "WARN" or bool(flags):
        return "warn"
    return "pass"


def _warnings(payload: Mapping[str, Any]) -> tuple[str, ...]:
    value = payload.get("consistency_flags", ())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("consistency_flags must be a sequence")
    return tuple(dict.fromkeys(str(item) for item in value))


class BodyCarrierReceipt(LegacyBodyCarrierReceipt):
    """Legacy constructor with governed v2 serialization when evidence is present."""

    def to_json(self, path: str | Path) -> Path:
        target = Path(path)
        fit_payload = _fit_payload(target.parent)
        if fit_payload is None:
            return super().to_json(target)

        refinement_payload = fit_payload.get("refinement_receipt")
        if not isinstance(refinement_payload, Mapping):
            raise TypeError("refinement_receipt must be an object")
        expected_hash = str(fit_payload.get("refinement_receipt_hash", ""))
        actual_hash = canonical_hash(refinement_payload)
        if expected_hash != actual_hash:
            raise ValueError(
                "Refinement receipt hash does not match the governed fit payload"
            )
        decision_value = str(refinement_payload.get("decision", ""))
        if decision_value not in {"promote", "abstain", "reject"}:
            raise ValueError("Invalid refinement decision in governed fit payload")
        decision = cast(BodyDecision, decision_value)

        raw_path = _find_mesh(
            target.parent,
            self.raw_reprojection_hash,
            "afflec_body_raw_reprojection.npz",
        )
        refined_path = _find_mesh(
            target.parent,
            self.refined_pre_repair_hash,
            "afflec_body_refined_pre_repair.npz",
        )
        final_path = _find_mesh(
            target.parent,
            self.repaired_export_hash,
            "afflec_body.npz",
        )
        raw_topology, _, _ = _mesh_evidence(raw_path)
        refined_topology, _, _ = _mesh_evidence(refined_path)
        final_topology, geometry_finite, topology_valid = _mesh_evidence(final_path)

        receipt = build_body_carrier_receipt_v2(
            source_hash=self.source_hash,
            raw_reprojection_hash=self.raw_reprojection_hash,
            refined_pre_repair_hash=self.refined_pre_repair_hash,
            repaired_export_hash=self.repaired_export_hash,
            refinement_receipt_hash=actual_hash,
            refinement_decision=decision,
            raw_topology=raw_topology,
            refined_pre_repair_topology=refined_topology,
            final_export_topology=final_topology,
            topology_label=self.topology_label,
            final_geometry_finite=geometry_finite,
            final_topology_valid=topology_valid,
            final_landmark_residuals=self.landmark_residuals,
            final_skull_rigidity_residual=self.skull_rigidity_residual,
            body_fit_confidence=self.body_fit_confidence,
            trust_level=str(fit_payload.get("trust_level", "")),
            severity=_severity(fit_payload),
            warnings=_warnings(fit_payload),
        )
        legacy_path = target.with_name(f"{target.stem}_v1{target.suffix}")
        super().to_json(legacy_path)
        return receipt.to_json(target)
