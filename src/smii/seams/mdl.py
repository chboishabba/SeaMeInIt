"""Minimum Description Length (MDL) priors and evaluation for seam solutions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import numpy as np

from smii.seams.solver import SeamSolution

__all__ = ["MDLPrior", "mdl_cost", "mdl_terms", "mdl_prior_preset", "validate_mdl_prior"]


@dataclass(frozen=True, slots=True)
class MDLPrior:
    """Weights for seam complexity penalties."""

    seam_count: float = 1.0
    seam_length: float = 1.0
    panel_count: float = 0.5
    boundary_roughness: float = 0.25
    symmetry_violation: float = 0.5


_MDL_PRESETS: Mapping[str, MDLPrior] = {
    "balanced": MDLPrior(
        seam_count=0.35,
        seam_length=0.20,
        panel_count=0.30,
        boundary_roughness=0.15,
        symmetry_violation=0.40,
    ),
    "simple": MDLPrior(
        seam_count=0.75,
        seam_length=0.35,
        panel_count=0.60,
        boundary_roughness=0.20,
        symmetry_violation=0.35,
    ),
    "symmetric": MDLPrior(
        seam_count=0.30,
        seam_length=0.20,
        panel_count=0.30,
        boundary_roughness=0.15,
        symmetry_violation=1.00,
    ),
}


def validate_mdl_prior(prior: MDLPrior) -> MDLPrior:
    """Fail closed on invalid MDL coefficients."""

    values = {
        "seam_count": prior.seam_count,
        "seam_length": prior.seam_length,
        "panel_count": prior.panel_count,
        "boundary_roughness": prior.boundary_roughness,
        "symmetry_violation": prior.symmetry_violation,
    }
    invalid = [name for name, value in values.items() if not np.isfinite(value) or value < 0.0]
    if invalid:
        raise ValueError(f"MDL weights must be finite and non-negative: {', '.join(sorted(invalid))}")
    return prior


def mdl_prior_preset(name: str = "balanced") -> MDLPrior:
    """Return a deterministic tuned MDL preset."""

    key = str(name).strip().lower()
    try:
        return _MDL_PRESETS[key]
    except KeyError as exc:
        available = ", ".join(sorted(_MDL_PRESETS))
        raise ValueError(f"unknown MDL preset {name!r}; choose one of: {available}") from exc


def _edge_length(edge: tuple[int, int], vertices: np.ndarray) -> float:
    a, b = edge
    if a >= len(vertices) or b >= len(vertices):
        return 0.0
    return float(np.linalg.norm(vertices[a] - vertices[b]))


def mdl_terms(
    solution: SeamSolution,
    *,
    vertices: np.ndarray,
    symmetry_penalty: float | None = None,
    boundary_roughness: float | None = None,
) -> Mapping[str, float]:
    """Compute MDL-relevant metrics for a seam solution."""

    all_edges: set[tuple[int, int]] = set()
    for panel_solution in solution.panel_solutions.values():
        all_edges.update(tuple(sorted(edge)) for edge in panel_solution.edges)

    lengths = np.array([_edge_length(edge, vertices) for edge in all_edges], dtype=float) if all_edges else np.array([], dtype=float)

    roughness = boundary_roughness
    if roughness is None:
        roughness = float(np.nanstd(lengths)) if lengths.size else 0.0

    symmetry_term = symmetry_penalty
    if symmetry_term is None:
        symmetry_term = float(solution.metadata.get("symmetry_penalty", 0.0)) if solution.metadata else 0.0

    terms: MutableMapping[str, float] = {
        "seam_count": float(len(all_edges)),
        "seam_length": float(lengths.sum()) if lengths.size else 0.0,
        "panel_count": float(len(solution.panel_solutions)),
        "boundary_roughness": float(roughness),
        "symmetry_violation": float(symmetry_term),
    }
    return terms


def mdl_cost(
    solution: SeamSolution,
    prior: MDLPrior,
    *,
    vertices: np.ndarray,
    symmetry_penalty: float | None = None,
    boundary_roughness: float | None = None,
) -> tuple[float, Mapping[str, float]]:
    """Compute MDL cost and a breakdown for reporting."""

    prior = validate_mdl_prior(prior)
    terms = mdl_terms(
        solution,
        vertices=vertices,
        symmetry_penalty=symmetry_penalty,
        boundary_roughness=boundary_roughness,
    )
    breakdown = {
        "seam_count": prior.seam_count * terms["seam_count"],
        "seam_length": prior.seam_length * terms["seam_length"],
        "panel_count": prior.panel_count * terms["panel_count"],
        "boundary_roughness": prior.boundary_roughness * terms["boundary_roughness"],
        "symmetry_violation": prior.symmetry_violation * terms["symmetry_violation"],
    }
    total = float(sum(breakdown.values()))
    return total, breakdown
