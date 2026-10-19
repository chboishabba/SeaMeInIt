"""Governed body-mesh export with final-artifact authorization."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from smii.meshing import build_body_carrier_receipt_v2, repair_body_mesh_for_export

from .fit_from_images import (
    create_body_mesh_from_regression as _create_body_mesh_from_regression,
)
from .fit_from_images import save_regression_mesh as _legacy_save_regression_mesh


def _save_checkpoint(path: Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
    )
    return path


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hash(result: Any) -> str:
    paths = [
        Path(frame.image_path)
        for frame in tuple(getattr(result, "frames", ()) or ())
        if getattr(frame, "image_path", None) is not None
    ]
    existing = sorted((path for path in paths if path.is_file()), key=lambda path: str(path))
    if existing:
        digest = hashlib.sha256()
        for path in existing:
            digest.update(str(path).encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        return digest.hexdigest()

    digest = hashlib.sha256()
    for name in ("betas", "body_pose", "global_orient", "transl"):
        values = np.asarray(getattr(result, name), dtype=np.float64)
        digest.update(name.encode("utf-8"))
        digest.update(values.shape.__repr__().encode("utf-8"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _face_components(faces: np.ndarray) -> int:
    if faces.shape[0] == 0:
        return 0
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, triangle in enumerate(faces):
        for offset in range(3):
            edge = tuple(
                sorted(
                    (
                        int(triangle[offset]),
                        int(triangle[(offset + 1) % 3]),
                    )
                )
            )
            edge_to_faces.setdefault(edge, []).append(face_index)

    adjacency: list[set[int]] = [set() for _ in range(faces.shape[0])]
    for incident in edge_to_faces.values():
        for face_index in incident:
            adjacency[face_index].update(other for other in incident if other != face_index)

    unseen = set(range(faces.shape[0]))
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            current = stack.pop()
            neighbors = adjacency[current] & unseen
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
    return components


def _mesh_evidence(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[dict[str, int], bool, bool]:
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    geometry_finite = bool(
        vertices.ndim == 2
        and vertices.shape[1:] == (3,)
        and vertices.shape[0] > 0
        and np.isfinite(vertices).all()
    )
    faces_finite = bool(
        np.issubdtype(faces.dtype, np.number) and np.isfinite(faces).all()
    )
    indexed = bool(
        geometry_finite
        and faces.ndim == 2
        and faces.shape[1:] == (3,)
        and faces.shape[0] > 0
        and faces_finite
        and np.equal(faces, np.floor(faces)).all()
        and int(np.min(faces)) >= 0
        and int(np.max(faces)) < vertices.shape[0]
    )

    closed = False
    connected = False
    if indexed:
        edge_counts: dict[tuple[int, int], int] = {}
        integer_faces = np.asarray(faces, dtype=np.int64)
        for triangle in integer_faces:
            for offset in range(3):
                edge = tuple(
                    sorted(
                        (
                            int(triangle[offset]),
                            int(triangle[(offset + 1) % 3]),
                        )
                    )
                )
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        closed = bool(edge_counts) and all(count == 2 for count in edge_counts.values())
        connected = _face_components(integer_faces) == 1

    topology = {
        "vertex_count": int(vertices.shape[0]) if vertices.ndim >= 1 else 0,
        "face_count": int(faces.shape[0]) if faces.ndim >= 1 else 0,
    }
    return topology, geometry_finite, bool(indexed and closed and connected)


def _candidate_regression(result: Any, refinement: Any) -> Any:
    receipt = refinement.refinement_receipt
    evidence = receipt.candidate_evidence
    candidate_betas = np.asarray(evidence["candidate_betas"], dtype=float).reshape(-1)
    context = evidence.get("context", {})
    if not isinstance(context, Mapping):
        raise TypeError("Refinement candidate context must be an object")
    candidate_scale = float(context.get("scale", refinement.scale))
    candidate_translation = np.asarray(
        context.get("translation", refinement.translation),
        dtype=float,
    ).reshape(-1)
    if candidate_translation.size != 3:
        raise ValueError("Refinement candidate translation must contain three values")
    candidate_fit = replace(
        refinement,
        betas=candidate_betas,
        scale=candidate_scale,
        translation=candidate_translation,
    )
    return replace(result, measurement_fit=candidate_fit)


def _severity(result: Any) -> str:
    status = str(getattr(result, "consistency_status", "")).upper()
    flags = tuple(getattr(result, "consistency_flags", ()) or ())
    if status == "FAIL":
        return "fail"
    if status == "WARN" or flags:
        return "warn"
    return "pass"


def _confidence(result: Any) -> float:
    frames = tuple(getattr(result, "frames", ()) or ())
    values = [float(getattr(frame, "confidence", 0.0)) for frame in frames]
    return float(np.mean(values)) if values else 0.0


def _crown_eccentricity_residual(vertices: np.ndarray) -> float:
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] < 3:
        return 0.0
    threshold = float(np.quantile(points[:, 1], 0.95))
    crown = points[points[:, 1] >= threshold]
    if crown.shape[0] < 3:
        return 0.0
    spread = np.ptp(crown[:, [0, 2]], axis=0)
    major = float(np.max(spread))
    minor = float(np.min(spread))
    return 0.0 if major <= 1e-9 else float((major - minor) / major)


def _final_residuals(
    refinement: Any,
    selected_vertices: np.ndarray,
    final_vertices: np.ndarray,
) -> dict[str, float]:
    residuals = {"measurement_fit_residual": float(refinement.residual)}
    selected = np.asarray(selected_vertices, dtype=float)
    final = np.asarray(final_vertices, dtype=float)
    if selected.shape == final.shape and selected.ndim == 2:
        displacement = np.linalg.norm(final - selected, axis=1)
        residuals["repair_rms_displacement"] = float(
            np.sqrt(np.mean(displacement**2))
        )
        residuals["repair_max_displacement"] = float(np.max(displacement))
    return residuals


def _checkpoint_paths(path: Path) -> tuple[Path, Path]:
    return (
        path.with_name(f"{path.stem}_raw_reprojection.npz"),
        path.with_name(f"{path.stem}_refined_pre_repair.npz"),
    )


def save_regression_mesh(
    result: Any,
    path: Path,
    *,
    model_path: Path | None = None,
    model_type: str = "smplx",
    gender: str = "neutral",
    use_measurement_refinement: bool = True,
) -> Path:
    """Export a governed image fit and emit its final body authorization receipt."""

    refinement = getattr(result, "measurement_fit", None)
    receipt = getattr(refinement, "refinement_receipt", None)
    if not use_measurement_refinement or receipt is None:
        return _legacy_save_regression_mesh(
            result,
            path,
            model_path=model_path,
            model_type=model_type,
            gender=gender,
            use_measurement_refinement=use_measurement_refinement,
        )

    path = Path(path)
    raw_vertices, raw_faces = _create_body_mesh_from_regression(
        result,
        model_path=model_path,
        model_type=model_type,
        gender=gender,
        use_measurement_refinement=False,
    )
    candidate_result = _candidate_regression(result, refinement)
    candidate_vertices, candidate_faces = _create_body_mesh_from_regression(
        candidate_result,
        model_path=model_path,
        model_type=model_type,
        gender=gender,
        use_measurement_refinement=True,
    )

    raw_path, candidate_path = _checkpoint_paths(path)
    _save_checkpoint(raw_path, raw_vertices, raw_faces)
    _save_checkpoint(candidate_path, candidate_vertices, candidate_faces)

    if receipt.decision == "promote":
        selected_vertices = np.asarray(candidate_vertices)
        selected_faces = np.asarray(candidate_faces)
    else:
        selected_vertices = np.asarray(raw_vertices)
        selected_faces = np.asarray(raw_faces)

    warnings = list(tuple(getattr(result, "consistency_flags", ()) or ()))
    repaired = repair_body_mesh_for_export(selected_vertices, selected_faces)
    if repaired is None:
        final_vertices = selected_vertices
        final_faces = selected_faces
        _, _, selected_valid = _mesh_evidence(selected_vertices, selected_faces)
        if not selected_valid:
            warnings.append("final_export_repair_unavailable")
    else:
        final_vertices, final_faces = repaired
        if (
            np.asarray(final_vertices).shape != selected_vertices.shape
            or np.asarray(final_faces).shape != selected_faces.shape
            or not np.array_equal(final_faces, selected_faces)
        ):
            warnings.append("final_export_repaired")

    _save_checkpoint(path, final_vertices, final_faces)
    raw_topology, _, _ = _mesh_evidence(raw_vertices, raw_faces)
    candidate_topology, _, _ = _mesh_evidence(candidate_vertices, candidate_faces)
    final_topology, geometry_finite, topology_valid = _mesh_evidence(
        final_vertices,
        final_faces,
    )

    body_receipt = build_body_carrier_receipt_v2(
        source_hash=_source_hash(result),
        raw_reprojection_hash=_sha256_path(raw_path),
        refined_pre_repair_hash=_sha256_path(candidate_path),
        repaired_export_hash=_sha256_path(path),
        refinement_receipt=receipt,
        raw_topology=raw_topology,
        refined_pre_repair_topology=candidate_topology,
        final_export_topology=final_topology,
        topology_label=f"A_v{final_topology['vertex_count']}",
        final_geometry_finite=geometry_finite,
        final_topology_valid=topology_valid,
        final_landmark_residuals=_final_residuals(
            refinement,
            selected_vertices,
            final_vertices,
        ),
        final_skull_rigidity_residual=_crown_eccentricity_residual(final_vertices),
        body_fit_confidence=_confidence(result),
        trust_level=str(getattr(result, "trust_level", "")),
        severity=_severity(result),  # type: ignore[arg-type]
        warnings=tuple(dict.fromkeys(warnings)),
    )
    body_receipt.to_json(path.parent / "body_carrier_receipt.json")
    return path


__all__ = ["save_regression_mesh"]
