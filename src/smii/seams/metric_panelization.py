"""Metric-aware panel correction proxies for seam topology proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

Edge = tuple[int, int]

CORRECTION_FAMILIES = (
    "dart",
    "relief_cut",
    "ease",
    "gusset",
    "stretch_zone",
    "variable_knit",
    "pleat",
    "bias_orientation",
)

COMPATIBLE_FAMILY_PAIRS = {
    frozenset(("bias_orientation", "stretch_zone")),
    frozenset(("bias_orientation", "variable_knit")),
    frozenset(("bias_orientation", "ease")),
    frozenset(("ease", "pleat")),
}


@dataclass(frozen=True, slots=True)
class MetricEnergyWeights:
    """Weights for the metric-aware variational objective."""

    residual: float = 1.0
    seam: float = 1.0
    correction: float = 1.0
    complexity: float = 1.0
    manufacture: float = 1.0


@dataclass(frozen=True, slots=True)
class CorrectionCandidate:
    """A first-pass physical proxy for a panel-local metric correction."""

    family: str
    panel_label: int
    defect_id: str
    seed_vertices: tuple[int, ...]
    support_edges: tuple[Edge, ...]
    raw_residual: float
    corrected_residual: float
    correction_cost: float
    complexity_penalty: float
    manufacture_penalty: float
    gain: float
    reason: str

    def to_dict(self, *, selected: bool, rejected_reason: str | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "family": self.family,
            "panel_label": int(self.panel_label),
            "defect_id": self.defect_id,
            "seed_vertices": [int(vertex) for vertex in self.seed_vertices],
            "support_edges": [[int(a), int(b)] for a, b in self.support_edges],
            "raw_residual": float(self.raw_residual),
            "corrected_residual": float(self.corrected_residual),
            "correction_cost": float(self.correction_cost),
            "complexity_penalty": float(self.complexity_penalty),
            "manufacture_penalty": float(self.manufacture_penalty),
            "gain": float(self.gain),
            "selected": bool(selected),
            "reason": self.reason,
        }
        if rejected_reason is not None:
            payload["rejected_reason"] = rejected_reason
        return payload


def normalize_families(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Return a validated correction family tuple."""

    if value is None:
        return CORRECTION_FAMILIES
    if isinstance(value, str):
        families = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        families = tuple(str(part).strip() for part in value if str(part).strip())
    invalid = [family for family in families if family not in CORRECTION_FAMILIES]
    if invalid:
        raise ValueError(
            "Unknown correction families: "
            f"{', '.join(invalid)}. Expected one of {', '.join(CORRECTION_FAMILIES)}."
        )
    return families


def _normalize_edge(a: int, b: int) -> Edge:
    aa = int(a)
    bb = int(b)
    return (aa, bb) if aa <= bb else (bb, aa)


def _face_edges(face: Sequence[int]) -> tuple[Edge, Edge, Edge]:
    a, b, c = (int(face[0]), int(face[1]), int(face[2]))
    return (_normalize_edge(a, b), _normalize_edge(b, c), _normalize_edge(c, a))


def _triangle_angles(points: np.ndarray) -> np.ndarray:
    angles = np.zeros(3, dtype=float)
    for idx in range(3):
        origin = points[idx]
        left = points[(idx + 1) % 3] - origin
        right = points[(idx + 2) % 3] - origin
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 1e-12:
            continue
        cosine = float(np.clip(np.dot(left, right) / denom, -1.0, 1.0))
        angles[idx] = float(np.arccos(cosine))
    return angles


def _unwrap_vertices(vertices: np.ndarray, vertex_ids: Sequence[int]) -> np.ndarray:
    coords = np.asarray(vertices[list(vertex_ids)], dtype=float)
    if len(coords) == 0:
        return np.empty((0, 2), dtype=float)
    centered = coords - coords.mean(axis=0)
    if len(coords) == 1:
        return np.zeros((1, 2), dtype=float)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt[:2]
    if axes.shape[0] < 2:
        axes = np.vstack([axes, np.array([[0.0, 1.0, 0.0]])])
    return centered @ axes.T


def _area(vertices: np.ndarray, faces: Sequence[Sequence[int]]) -> float:
    total = 0.0
    for face in faces:
        a, b, c = np.asarray(vertices[list(face)], dtype=float)
        total += 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
    return total


def _edge_distortion(
    vertices: np.ndarray, vertex_ids: tuple[int, ...], edges: tuple[Edge, ...]
) -> float:
    if not edges:
        return 0.0
    uv = _unwrap_vertices(vertices, vertex_ids)
    local = {vertex: idx for idx, vertex in enumerate(vertex_ids)}
    residuals: list[float] = []
    for a, b in edges:
        if a not in local or b not in local:
            continue
        length_3d = float(np.linalg.norm(vertices[a] - vertices[b]))
        length_2d = float(np.linalg.norm(uv[local[a]] - uv[local[b]]))
        if length_3d > 1e-12:
            residuals.append(abs(length_2d - length_3d) / length_3d)
    return float(np.mean(residuals)) if residuals else 0.0


def _panel_stats(
    vertices: np.ndarray,
    panel_faces: Sequence[Sequence[int]],
) -> dict[str, object]:
    face_tuples = tuple(tuple(int(vertex) for vertex in face) for face in panel_faces)
    vertex_ids = tuple(sorted({vertex for face in face_tuples for vertex in face}))
    edge_counts: dict[Edge, int] = {}
    angle_sums = {vertex: 0.0 for vertex in vertex_ids}
    for face in face_tuples:
        for edge in _face_edges(face):
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
        angles = _triangle_angles(np.asarray(vertices[list(face)], dtype=float))
        for offset, vertex in enumerate(face):
            angle_sums[int(vertex)] += float(angles[offset])

    edges = tuple(sorted(edge_counts))
    boundary_edges = tuple(sorted(edge for edge, count in edge_counts.items() if count == 1))
    boundary_vertices = {vertex for edge in boundary_edges for vertex in edge}
    defects: dict[int, float] = {}
    for vertex in vertex_ids:
        target = np.pi if vertex in boundary_vertices else 2.0 * np.pi
        defects[int(vertex)] = max(0.0, float(target - angle_sums[int(vertex)]))
    edge_lengths = [float(np.linalg.norm(vertices[a] - vertices[b])) for a, b in boundary_edges]
    boundary_length = float(sum(edge_lengths))
    mean_boundary_edge = float(np.mean(edge_lengths)) if edge_lengths else 0.0
    max_boundary_edge = float(max(edge_lengths)) if edge_lengths else 0.0
    panel_area = _area(vertices, face_tuples)
    compact_boundary = 4.0 * float(np.sqrt(max(panel_area, 1e-12)))
    boundary_mismatch = (
        abs(boundary_length - compact_boundary) / max(compact_boundary, 1e-9)
        if boundary_edges
        else 0.0
    )
    anisotropy = 1.0
    if len(vertex_ids) >= 3:
        coords = np.asarray(vertices[list(vertex_ids)], dtype=float)
        centered = coords - coords.mean(axis=0)
        _u, singular, _vt = np.linalg.svd(centered, full_matrices=False)
        if singular.size >= 2 and singular[1] > 1e-12:
            anisotropy = float(singular[0] / singular[1])
    raw_flattening = _edge_distortion(vertices, vertex_ids, edges)
    curvature = float(np.mean(list(defects.values()))) / np.pi if defects else 0.0
    raw_residual = float(raw_flattening + 0.25 * curvature)
    seed_vertices = tuple(
        int(vertex)
        for vertex, _defect in sorted(defects.items(), key=lambda item: (-item[1], item[0]))[:3]
    )
    return {
        "vertex_ids": vertex_ids,
        "edges": edges,
        "boundary_edges": boundary_edges,
        "boundary_mismatch": boundary_mismatch,
        "max_boundary_ratio": max_boundary_edge / max(mean_boundary_edge, 1e-9),
        "anisotropy": anisotropy,
        "raw_flattening": raw_flattening,
        "raw_residual": raw_residual,
        "max_defect": max(defects.values()) if defects else 0.0,
        "seed_vertices": seed_vertices,
        "area": panel_area,
    }


def _candidate(
    *,
    family: str,
    panel_label: int,
    defect_id: str,
    seed_vertices: tuple[int, ...],
    support_edges: tuple[Edge, ...],
    raw_residual: float,
    reduction_fraction: float,
    correction_cost: float,
    complexity_penalty: float,
    manufacture_penalty: float,
    weights: MetricEnergyWeights,
    reason: str,
) -> CorrectionCandidate:
    corrected_residual = max(0.0, raw_residual * (1.0 - max(0.0, reduction_fraction)))
    residual_gain = weights.residual * (raw_residual - corrected_residual)
    total_penalty = (
        weights.correction * correction_cost
        + weights.complexity * complexity_penalty
        + weights.manufacture * manufacture_penalty
    )
    return CorrectionCandidate(
        family=family,
        panel_label=int(panel_label),
        defect_id=defect_id,
        seed_vertices=seed_vertices,
        support_edges=support_edges,
        raw_residual=float(raw_residual),
        corrected_residual=float(corrected_residual),
        correction_cost=float(correction_cost),
        complexity_penalty=float(complexity_penalty),
        manufacture_penalty=float(manufacture_penalty),
        gain=float(residual_gain - total_penalty),
        reason=reason,
    )


def generate_correction_candidates(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    families: Sequence[str] = CORRECTION_FAMILIES,
    weights: MetricEnergyWeights | None = None,
) -> tuple[list[CorrectionCandidate], list[dict[str, object]]]:
    """Generate scored correction candidates for each face-region panel."""

    weights = weights or MetricEnergyWeights()
    allowed = set(normalize_families(families))
    candidates: list[CorrectionCandidate] = []
    panel_reports: list[dict[str, object]] = []
    for label in sorted({int(value) for value in labels}):
        panel_faces = [
            tuple(int(vertex) for vertex in faces[idx]) for idx in np.where(labels == label)[0]
        ]
        stats = _panel_stats(vertices, panel_faces)
        raw_residual = float(stats["raw_residual"])
        raw_flattening = float(stats["raw_flattening"])
        max_defect = float(stats["max_defect"])
        mismatch = float(stats["boundary_mismatch"])
        ratio = float(stats["max_boundary_ratio"])
        anisotropy = float(stats["anisotropy"])
        seed_vertices = tuple(int(vertex) for vertex in stats["seed_vertices"])
        support_edges = tuple(stats["boundary_edges"])[:4]
        panel_reports.append(
            {
                "panel_label": int(label),
                "face_count": len(panel_faces),
                "raw_flattening_residual": raw_flattening,
                "raw_metric_residual": raw_residual,
                "angle_deficit": max_defect,
                "boundary_mismatch": mismatch,
                "anisotropy": anisotropy,
            }
        )
        if raw_residual <= 1e-9:
            continue
        if "dart" in allowed and max_defect > 0.08:
            candidates.append(
                _candidate(
                    family="dart",
                    panel_label=label,
                    defect_id=f"panel:{label}:curvature",
                    seed_vertices=seed_vertices[:1],
                    support_edges=support_edges[:2],
                    raw_residual=raw_residual,
                    reduction_fraction=min(0.55, 0.20 + max_defect / (3.0 * np.pi)),
                    correction_cost=0.015,
                    complexity_penalty=0.025,
                    manufacture_penalty=0.015,
                    weights=weights,
                    reason="localized angle deficit supports wedge take-up",
                )
            )
        if "relief_cut" in allowed and max_defect > 0.12 and support_edges:
            candidates.append(
                _candidate(
                    family="relief_cut",
                    panel_label=label,
                    defect_id=f"panel:{label}:curvature",
                    seed_vertices=seed_vertices[:1],
                    support_edges=support_edges[:3],
                    raw_residual=raw_residual,
                    reduction_fraction=min(0.62, 0.25 + max_defect / (2.8 * np.pi)),
                    correction_cost=0.020,
                    complexity_penalty=0.035,
                    manufacture_penalty=0.030,
                    weights=weights,
                    reason="high-curvature defect has reachable panel boundary",
                )
            )
        if "ease" in allowed and mismatch > 0.08:
            candidates.append(
                _candidate(
                    family="ease",
                    panel_label=label,
                    defect_id=f"panel:{label}:boundary_mismatch",
                    seed_vertices=seed_vertices,
                    support_edges=support_edges,
                    raw_residual=raw_residual,
                    reduction_fraction=min(0.35, 0.12 + mismatch * 0.35),
                    correction_cost=0.010,
                    complexity_penalty=0.010,
                    manufacture_penalty=0.020,
                    weights=weights,
                    reason="distributed boundary length mismatch can be absorbed as ease",
                )
            )
        if "gusset" in allowed and (mismatch > 0.18 or ratio > 1.85):
            candidates.append(
                _candidate(
                    family="gusset",
                    panel_label=label,
                    defect_id=f"panel:{label}:boundary_mismatch",
                    seed_vertices=seed_vertices,
                    support_edges=support_edges,
                    raw_residual=raw_residual,
                    reduction_fraction=min(0.62, 0.28 + mismatch * 0.60),
                    correction_cost=0.025,
                    complexity_penalty=0.040,
                    manufacture_penalty=0.035,
                    weights=weights,
                    reason="inserted area/angle compensation beats excessive stretch",
                )
            )
        if "stretch_zone" in allowed and raw_flattening > 0.015:
            candidates.append(
                _candidate(
                    family="stretch_zone",
                    panel_label=label,
                    defect_id=f"panel:{label}:strain",
                    seed_vertices=seed_vertices,
                    support_edges=support_edges,
                    raw_residual=raw_residual,
                    reduction_fraction=0.30,
                    correction_cost=0.020,
                    complexity_penalty=0.015,
                    manufacture_penalty=0.015,
                    weights=weights,
                    reason="bounded local strain tensor is within material proxy limits",
                )
            )
        if "variable_knit" in allowed and raw_flattening > 0.030:
            candidates.append(
                _candidate(
                    family="variable_knit",
                    panel_label=label,
                    defect_id=f"panel:{label}:strain",
                    seed_vertices=seed_vertices,
                    support_edges=support_edges,
                    raw_residual=raw_residual,
                    reduction_fraction=0.42,
                    correction_cost=0.030,
                    complexity_penalty=0.060,
                    manufacture_penalty=0.035,
                    weights=weights,
                    reason="spatially varying stretch field reduces residual with smoothness cost",
                )
            )
        if "pleat" in allowed and mismatch > 0.10 and ratio > 1.25:
            candidates.append(
                _candidate(
                    family="pleat",
                    panel_label=label,
                    defect_id=f"panel:{label}:boundary_mismatch",
                    seed_vertices=seed_vertices,
                    support_edges=support_edges[:2],
                    raw_residual=raw_residual,
                    reduction_fraction=min(0.45, 0.16 + mismatch * 0.40),
                    correction_cost=0.018,
                    complexity_penalty=0.040,
                    manufacture_penalty=0.050,
                    weights=weights,
                    reason="concentrated take-up is better represented as a fold strip",
                )
            )
        if "bias_orientation" in allowed and anisotropy > 1.20 and raw_residual > 0.010:
            candidates.append(
                _candidate(
                    family="bias_orientation",
                    panel_label=label,
                    defect_id=f"panel:{label}:strain",
                    seed_vertices=seed_vertices,
                    support_edges=support_edges,
                    raw_residual=raw_residual,
                    reduction_fraction=min(0.25, 0.08 + (anisotropy - 1.0) * 0.04),
                    correction_cost=0.006,
                    complexity_penalty=0.010,
                    manufacture_penalty=0.018,
                    weights=weights,
                    reason="anisotropic grain choice changes allowed strain",
                )
            )
    return candidates, panel_reports


def select_compatible_corrections(
    candidates: Sequence[CorrectionCandidate],
    *,
    max_corrections_per_panel: int,
) -> tuple[list[CorrectionCandidate], list[dict[str, object]]]:
    """Greedily select positive-gain compatible corrections."""

    selected: list[CorrectionCandidate] = []
    rejected: list[dict[str, object]] = []
    panel_counts: dict[int, int] = {}
    ordered = sorted(
        candidates, key=lambda candidate: (-candidate.gain, candidate.panel_label, candidate.family)
    )
    for candidate in ordered:
        if candidate.gain <= 0.0:
            rejected.append(candidate.to_dict(selected=False, rejected_reason="non_positive_gain"))
            continue
        if panel_counts.get(candidate.panel_label, 0) >= max(0, int(max_corrections_per_panel)):
            rejected.append(
                candidate.to_dict(selected=False, rejected_reason="panel_correction_limit")
            )
            continue
        incompatible = False
        for existing in selected:
            if existing.panel_label != candidate.panel_label:
                continue
            if existing.defect_id != candidate.defect_id:
                continue
            pair = frozenset((existing.family, candidate.family))
            if pair not in COMPATIBLE_FAMILY_PAIRS:
                incompatible = True
                break
        if incompatible:
            rejected.append(
                candidate.to_dict(
                    selected=False, rejected_reason="overlapping_incompatible_operator"
                )
            )
            continue
        selected.append(candidate)
        panel_counts[candidate.panel_label] = panel_counts.get(candidate.panel_label, 0) + 1
    return selected, rejected


def build_metric_panelization_payload(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    seam_edges: Sequence[Edge],
    families: Sequence[str] = CORRECTION_FAMILIES,
    max_corrections_per_panel: int = 3,
    weights: MetricEnergyWeights | None = None,
) -> dict[str, object]:
    """Evaluate and select metric corrections for a cut-graph topology proposal."""

    weights = weights or MetricEnergyWeights()
    candidates, panel_reports = generate_correction_candidates(
        vertices=vertices,
        faces=faces,
        labels=labels,
        families=families,
        weights=weights,
    )
    selected, rejected = select_compatible_corrections(
        candidates,
        max_corrections_per_panel=max_corrections_per_panel,
    )
    raw_by_panel = {
        int(report["panel_label"]): float(report["raw_metric_residual"]) for report in panel_reports
    }
    corrected_by_panel = dict(raw_by_panel)
    for correction in selected:
        current = corrected_by_panel.get(correction.panel_label, correction.raw_residual)
        improvement = correction.raw_residual - correction.corrected_residual
        corrected_by_panel[correction.panel_label] = max(0.0, current - improvement)
    for report in panel_reports:
        label = int(report["panel_label"])
        report["corrected_metric_residual"] = float(corrected_by_panel.get(label, 0.0))
        report["selected_correction_count"] = sum(
            1 for correction in selected if correction.panel_label == label
        )

    raw_values = list(raw_by_panel.values())
    corrected_values = list(corrected_by_panel.values())
    raw_total = float(sum(raw_values))
    corrected_total = float(sum(corrected_values))
    seam_complexity = float(len(seam_edges))
    energy = {
        "raw_residual_total": raw_total,
        "corrected_residual_total": corrected_total,
        "residual_reduction": float(raw_total - corrected_total),
        "seam_term": float(weights.seam * seam_complexity),
        "correction_term": float(weights.correction * sum(c.correction_cost for c in selected)),
        "complexity_term": float(weights.complexity * sum(c.complexity_penalty for c in selected)),
        "manufacture_term": float(
            weights.manufacture * sum(c.manufacture_penalty for c in selected)
        ),
    }
    energy["total"] = float(
        weights.residual * corrected_total
        + energy["seam_term"]
        + energy["correction_term"]
        + energy["complexity_term"]
        + energy["manufacture_term"]
    )
    return {
        "variational_object": "(M,g_M,fields,constraints)->(P_i,u_i,Delta_g_i,seams,corrections)",
        "residual_definition": "u_i^* g_flat - (g_body + Delta_g_i)",
        "families": list(normalize_families(families)),
        "max_corrections_per_panel": int(max_corrections_per_panel),
        "weights": {
            "residual": float(weights.residual),
            "seam": float(weights.seam),
            "correction": float(weights.correction),
            "complexity": float(weights.complexity),
            "manufacture": float(weights.manufacture),
        },
        "panels": panel_reports,
        "selected_corrections": [candidate.to_dict(selected=True) for candidate in selected],
        "rejected_corrections": rejected,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "energy": energy,
    }


__all__ = [
    "CORRECTION_FAMILIES",
    "COMPATIBLE_FAMILY_PAIRS",
    "CorrectionCandidate",
    "Edge",
    "MetricEnergyWeights",
    "build_metric_panelization_payload",
    "generate_correction_candidates",
    "normalize_families",
    "select_compatible_corrections",
]
