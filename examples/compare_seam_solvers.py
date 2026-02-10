"""Compare seam solvers (MST, PDA, shortest-path, min-cut) and emit a summary table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from smii.rom.constraints import ConstraintRegistry, load_constraints
from smii.rom.seam_costs import load_seam_cost_field
from smii.seams.kernels import KernelWeights, build_edge_kernels
from smii.seams.mdl import MDLPrior
from smii.seams.pda import solve_seams_pda
from smii.seams.solver import solve_seams
from smii.seams.solvers_mincut import solve_seams_mincut
from smii.seams.solvers_sp import solve_seams_shortest_path
from suit.seam_generator import SeamGenerator


def _load_body(body_path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(body_path)
    return np.asarray(payload["vertices"], dtype=float), np.asarray(payload["faces"], dtype=int)


def _load_yaml_like(path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        import json
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _weights_from_path(path: Path | None) -> KernelWeights:
    if path is None:
        return KernelWeights()
    payload = _load_yaml_like(path)
    return KernelWeights(
        rom_mean=float(payload.get("rom_mean", 1.0)),
        rom_max=float(payload.get("rom_max", 1.0)),
        rom_grad=float(payload.get("rom_grad", 1.0)),
        curvature=float(payload.get("curvature", 1.0)),
        fabric=float(payload.get("fabric", 1.0)),
    )


def _mdl_from_path(path: Path | None) -> MDLPrior:
    if path is None:
        return MDLPrior()
    payload = _load_yaml_like(path)
    return MDLPrior(
        seam_count=float(payload.get("seam_count", 1.0)),
        seam_length=float(payload.get("seam_length", 1.0)),
        panel_count=float(payload.get("panel_count", 0.5)),
        boundary_roughness=float(payload.get("boundary_roughness", 0.25)),
        symmetry_violation=float(payload.get("symmetry_violation", 0.5)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ROM-driven seam solvers.")
    parser.add_argument("--body", type=Path, required=True, help="NPZ with vertices, faces.")
    parser.add_argument("--rom-costs", type=Path, required=True, help="NPZ with seam costs.")
    parser.add_argument("--constraints", type=Path, help="Optional constraint manifest directory.")
    parser.add_argument("--weights", type=Path, default=Path("configs/kernel_weights.yaml"), help="Kernel weights YAML/JSON.")
    parser.add_argument("--mdl", type=Path, default=Path("configs/mdl_prior.yaml"), help="MDL prior YAML/JSON.")
    parser.add_argument("--budget", type=int, default=8, help="PDA iteration budget.")
    args = parser.parse_args()

    vertices, faces = _load_body(args.body)
    cost_field = load_seam_cost_field(args.rom_costs)
    constraints: ConstraintRegistry | None = load_constraints(args.constraints) if args.constraints else None
    weights = _weights_from_path(args.weights)
    mdl_prior = _mdl_from_path(args.mdl)

    axis_map = {
        "longitudinal": np.array([0.0, 0.0, 1.0], dtype=float),
        "lateral": np.array([1.0, 0.0, 0.0], dtype=float),
    }
    seam_graph = SeamGenerator().generate(vertices, faces, measurement_loops=None, axis_map=axis_map)
    kernels = build_edge_kernels(cost_field, seam_graph, vertices=vertices, constraints=constraints)

    runs = {}
    runs["mst"] = solve_seams(seam_graph, cost_field, constraints=constraints)
    runs["pda"] = solve_seams_pda(
        seam_graph,
        kernels,
        weights,
        mdl_prior,
        constraints,
        budget=args.budget,
        cost_field=cost_field,
        vertices=vertices,
    )
    runs["shortest_path"] = solve_seams_shortest_path(
        seam_graph,
        kernels,
        weights,
        mdl_prior,
        constraints,
        cost_field=cost_field,
        vertices=vertices,
    )
    runs["mincut"] = solve_seams_mincut(
        seam_graph,
        kernels,
        weights,
        mdl_prior,
        constraints,
        cost_field=cost_field,
        vertices=vertices,
    )

    print("solver,total_cost,panel_count,seam_count")
    for name, solution in runs.items():
        edges = {
            edge for panel in solution.panel_solutions.values() for edge in panel.edges
        }
        print(f"{name},{solution.total_cost:.4f},{len(solution.panel_solutions)},{len(edges)}")


if __name__ == "__main__":
    main()
