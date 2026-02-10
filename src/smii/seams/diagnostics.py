"""Diagnostics and reporting for seam solvers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import numpy as np

from smii.rom.seam_costs import SeamCostField
from smii.seams.kernels import EdgeKernel, KernelWeights, edge_energy
from smii.seams.mdl import MDLPrior, mdl_cost
from smii.seams.solver import PanelSolution, SeamSolution

try:  # Optional heavy import
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]

try:  # Optional heavy import
    from suit.seam_generator import SeamGraph
except Exception:  # pragma: no cover
    SeamGraph = None  # type: ignore[assignment]

__all__ = ["build_diagnostics_report", "save_report", "render_overlay"]


def _edge_breakdown(edge: tuple[int, int], kernel: EdgeKernel, weights: KernelWeights) -> Mapping[str, float]:
    return {
        "rom_mean": weights.rom_mean * kernel.rom_mean,
        "rom_max": weights.rom_max * kernel.rom_max,
        "rom_grad": weights.rom_grad * kernel.rom_grad,
        "curvature": weights.curvature * kernel.curvature,
        "fabric": weights.fabric * kernel.fabric_misalignment,
        "total": edge_energy(kernel, weights),
    }


def _panel_entries(
    panel_solution: PanelSolution,
    kernels: Mapping[tuple[int, int], EdgeKernel],
    weights: KernelWeights,
) -> list[Mapping[str, float | tuple[int, int]]]:
    entries: list[Mapping[str, float | tuple[int, int]]] = []
    for edge in panel_solution.edges:
        kernel = kernels.get(edge)
        if kernel is None:
            continue
        breakdown = _edge_breakdown(edge, kernel, weights)
        entries.append({"edge": edge, **breakdown})
    return entries


def build_diagnostics_report(
    seam_graph: "SeamGraph",
    solution: SeamSolution,
    kernels: Mapping[tuple[int, int], EdgeKernel],
    weights: KernelWeights,
    mdl_prior: MDLPrior,
    *,
    vertices: np.ndarray,
    cost_field: SeamCostField | None = None,
) -> Mapping[str, object]:
    """Generate a JSON-friendly diagnostics report."""

    mdl_value, mdl_breakdown = mdl_cost(solution, mdl_prior, vertices=vertices)
    report: MutableMapping[str, object] = {
        "solver": solution.solver,
        "total_cost": solution.total_cost,
        "mdl_cost": mdl_value,
        "mdl_breakdown": dict(mdl_breakdown),
        "warnings": list(solution.warnings),
        "panels": {},
    }

    for panel in seam_graph.panels:
        panel_solution = solution.panel_solutions.get(panel.name)
        if panel_solution is None:
            continue
        entries = _panel_entries(panel_solution, kernels, weights)
        vertex_costs = None
        if cost_field is not None:
            vertex_costs = {
                "mean": float(np.nanmean(cost_field.vertex_costs[list(panel.seam_vertices)]))
                if panel.seam_vertices
                else 0.0,
                "max": float(np.nanmax(cost_field.vertex_costs[list(panel.seam_vertices)]))
                if panel.seam_vertices
                else 0.0,
            }
        report["panels"][panel.name] = {
            "edges": entries,
            "cost_breakdown": dict(panel_solution.cost_breakdown),
            "warnings": list(panel_solution.warnings),
            "vertex_costs": vertex_costs,
            "fabric": {
                "fabric_id": getattr(panel_solution, "fabric_id", None),
                "grain_rotation_deg": getattr(panel_solution, "grain_rotation_deg", 0.0),
            },
        }
    return report


def save_report(report: Mapping[str, object], path: Path) -> None:
    """Persist diagnostics report to JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)


def render_overlay(
    seam_graph: "SeamGraph",
    solution: SeamSolution,
    *,
    vertices: np.ndarray,
    cost_field: SeamCostField | None = None,
    output: Path | None = None,
    highlight_threshold: float = 0.8,
) -> Path | None:
    """Render a simple overlay of seams on the body. Returns output path or None."""

    if plt is None or SeamGraph is None:
        return None

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c="lightgray", s=1, alpha=0.2)

    if cost_field is not None:
        verts = np.asarray(vertices, dtype=float)
        costs = np.asarray(cost_field.vertex_costs, dtype=float)
        norm_costs = (costs - np.nanmin(costs)) / max(np.nanmax(costs) - np.nanmin(costs), 1e-6)
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c=norm_costs, cmap="magma", s=3, alpha=0.4)

    edge_cost_lookup: Mapping[tuple[int, int], float] = {}
    if cost_field is not None:
        edge_cost_lookup = cost_field.to_edge_weights()

    for panel in seam_graph.panels:
        panel_solution = solution.panel_solutions.get(panel.name)
        if panel_solution is None:
            continue
        for edge in panel_solution.edges:
            a, b = edge
            xs = [vertices[a, 0], vertices[b, 0]]
            ys = [vertices[a, 1], vertices[b, 1]]
            zs = [vertices[a, 2], vertices[b, 2]]
            base_color = "cyan"
            if edge_cost_lookup:
                edge_val = float(edge_cost_lookup.get(edge, edge_cost_lookup.get((edge[1], edge[0]), 0.0)))
                color_scale = "red" if edge_val >= highlight_threshold * max(edge_cost_lookup.values(), default=1.0) else "cyan"
                base_color = color_scale
            ax.plot(xs, ys, zs, color=base_color, linewidth=1.5, alpha=0.9)

    ax.set_axis_off()
    if output is None:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return output
