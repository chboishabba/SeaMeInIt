"""Task- and fabric-aware seam solve demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from smii.rom import KernelBasis, KernelProjector, RomSample, build_seam_cost_field, load_basis, load_coupling_manifest
from smii.rom.constraints import ConstraintRegistry, load_constraints
from smii.rom.gates import build_gate_from_manifest
from smii.seams.diagnostics import build_diagnostics_report, render_overlay, save_report
from smii.seams.fabric_kernels import FabricProfile, load_fabrics_from_dir
from smii.seams.kernels import KernelWeights, build_edge_kernels
from smii.seams.mdl import MDLPrior
from smii.seams.pda import solve_seams_pda
from smii.seams.task_profiles import aggregate_rom_for_task, load_task_profile
from suit.seam_generator import SeamGenerator


def _load_body(body_path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(body_path)
    vertices = np.asarray(payload["vertices"], dtype=float)
    faces = np.asarray(payload["faces"], dtype=int)
    return vertices, faces


def _load_yaml_like(path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
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


def _load_samples(path: Path) -> tuple[list[RomSample], Sequence[tuple[int, int]] | None, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("samples", payload)
    if not isinstance(entries, Sequence):
        raise TypeError("Samples payload must be a list or contain a 'samples' list.")

    edges = None
    if isinstance(payload, Mapping) and "edges" in payload and isinstance(payload["edges"], Sequence):
        edges = [tuple(map(int, edge)) for edge in payload["edges"]]  # type: ignore[list-item]

    samples: list[RomSample] = []
    fields: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("Each sample entry must be an object.")
        pose_id = str(entry.get("pose_id", f"sample_{len(samples)}"))
        coeffs_raw = entry.get("coeffs")
        if not isinstance(coeffs_raw, Mapping):
            raise KeyError(f"Sample '{pose_id}' missing 'coeffs' mapping.")
        coeffs: dict[str, np.ndarray] = {}
        for name, values in coeffs_raw.items():
            arr = np.asarray(values, dtype=float)
            if arr.ndim != 1:
                raise ValueError(f"Coeff field '{name}' on sample '{pose_id}' must be 1D.")
            coeffs[str(name)] = arr
            if str(name) not in fields:
                fields.append(str(name))
        observations = entry.get("observations") if isinstance(entry.get("observations"), Mapping) else None
        samples.append(RomSample(pose_id=pose_id, coeffs=coeffs, observations=observations))

    return samples, edges, fields


def _resolve_basis(path: Path | None, component_count: int) -> KernelBasis:
    if path is not None and path.exists():
        basis = load_basis(path)
        if basis.metadata.component_count != component_count:
            raise ValueError(
                f"Basis component count ({basis.metadata.component_count}) does not match sample coeffs ({component_count})."
            )
        return basis
    matrix = np.eye(component_count)
    return KernelBasis.from_arrays(matrix)


def _axis_map() -> Mapping[str, np.ndarray]:
    return {
        "longitudinal": np.array([0.0, 0.0, 1.0], dtype=float),
        "lateral": np.array([1.0, 0.0, 0.0], dtype=float),
    }


def _fabric_selection(fabric_id: str | None, catalog: Mapping[str, FabricProfile]) -> FabricProfile:
    if fabric_id and fabric_id in catalog:
        return catalog[fabric_id]
    return next(iter(catalog.values()))


def _parse_vector(text: str | None) -> np.ndarray | None:
    if text is None:
        return None
    parts = [float(p) for p in text.split(",") if p]
    if len(parts) != 3:
        raise ValueError("Fabric grain must be three comma-separated values.")
    return np.array(parts, dtype=float)


def _maybe_gate(manifest_path: Path | None):
    if manifest_path is None:
        return None
    manifest = load_coupling_manifest(manifest_path)
    return build_gate_from_manifest(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", type=Path, required=True, help="NPZ containing vertices (N,3) and faces (M,3).")
    parser.add_argument("--samples", type=Path, default=Path("examples/data/rom_samples_demo.json"), help="ROM samples JSON.")
    parser.add_argument("--basis", type=Path, default=None, help="Optional canonical basis NPZ path.")
    parser.add_argument("--task-profile", type=Path, default=Path("configs/tasks/reach_overhead_v1.yaml"), help="Task profile YAML.")
    parser.add_argument("--fabric-id", type=str, default="woven_2way", help="Fabric id to assign to all panels.")
    parser.add_argument("--fabric-dir", type=Path, default=Path("configs/fabrics"), help="Directory of fabric YAMLs.")
    parser.add_argument("--fabric-grain", type=str, default="1,0,0", help="Fabric grain vector as comma-separated values.")
    parser.add_argument("--weights", type=Path, default=Path("configs/kernel_weights.yaml"), help="Kernel weights YAML.")
    parser.add_argument("--mdl", type=Path, default=Path("configs/mdl_prior.yaml"), help="MDL prior YAML.")
    parser.add_argument("--field", type=str, default=None, help="Field key to build seam costs from.")
    parser.add_argument("--variance-weight", type=float, default=1.0, help="Variance weight for seam costs.")
    parser.add_argument("--maximum-weight", type=float, default=0.25, help="Maximum weight for seam costs.")
    parser.add_argument("--out", type=Path, default=Path("outputs/task_fabric_demo"), help="Output directory.")
    parser.add_argument("--constraints", type=Path, default=None, help="Optional constraint manifest directory.")
    parser.add_argument("--gate-manifest", type=Path, default=None, help="Optional coupling gate manifest for samples.")
    parser.add_argument("--budget", type=int, default=8, help="Iteration budget for PDA solver.")
    parser.add_argument("--vertex-weight", type=float, default=0.0, help="Vertex weight contribution for solvers.")
    parser.add_argument("--grain-step", type=float, default=10.0, help="Grain rotation step in degrees.")
    args = parser.parse_args()

    vertices, faces = _load_body(args.body)
    samples, edges, fields = _load_samples(args.samples)
    if not samples:
        raise ValueError("At least one ROM sample is required.")
    component_count = len(next(iter(samples[0].coeffs.values())))
    basis = _resolve_basis(args.basis, component_count)
    projector = KernelProjector(basis)
    gate = _maybe_gate(args.gate_manifest)

    task_profile = load_task_profile(args.task_profile)
    fabrics = load_fabrics_from_dir(args.fabric_dir)
    fabric_profile = _fabric_selection(args.fabric_id, fabrics)
    grain_vector = _parse_vector(args.fabric_grain)

    aggregation = aggregate_rom_for_task(
        samples,
        projector,
        task_profile,
        field_keys=fields,
        optional_fields=None,
        edges=edges,
        gate=gate,
        diagnostics_top_k=3,
    )
    target_field = args.field or (fields[0] if fields else list(aggregation.per_field.keys())[0])
    cost_field = build_seam_cost_field(
        aggregation,
        field=target_field,
        variance_weight=args.variance_weight,
        maximum_weight=args.maximum_weight,
    )

    generator = SeamGenerator()
    seam_graph = generator.generate(vertices, faces, measurement_loops=None, axis_map=_axis_map())
    constraints: ConstraintRegistry | None = None
    if args.constraints:
        constraints = load_constraints(args.constraints)

    weights = _weights_from_path(args.weights)
    mdl_prior = _mdl_from_path(args.mdl)

    kernels = build_edge_kernels(
        cost_field,
        seam_graph,
        vertices=vertices,
        fabric_grain=grain_vector,
        fabric_profile=fabric_profile,
    )
    panel_fabrics = {panel.name: fabric_profile.fabric_id for panel in seam_graph.panels}

    solution = solve_seams_pda(
        seam_graph,
        kernels,
        weights,
        mdl_prior,
        constraints,
        budget=args.budget,
        cost_field=cost_field,
        vertices=vertices,
        vertex_weight=args.vertex_weight,
        fabric_catalog=fabrics,
        fabric_grain=grain_vector,
        panel_fabrics=panel_fabrics,
        grain_rotation_step=args.grain_step,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    report = build_diagnostics_report(
        seam_graph,
        solution,
        kernels,
        weights,
        mdl_prior,
        vertices=vertices,
        cost_field=cost_field,
    )
    save_report(report, args.out / "task_fabric_report.json")
    overlay_path = render_overlay(seam_graph, solution, vertices=vertices, cost_field=cost_field, output=args.out / "overlay.png")

    print(f"Task profile: {task_profile.task_id}")
    print(f"Fabric: {fabric_profile.fabric_id}")
    print(f"Solver: {solution.solver}")
    print(f"Total cost: {solution.total_cost:.4f}")
    print(f"Report: {args.out / 'task_fabric_report.json'}")
    if overlay_path:
        print(f"Overlay: {overlay_path}")


if __name__ == "__main__":
    main()
