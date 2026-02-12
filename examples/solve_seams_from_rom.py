"""Example CLI to run ROM-driven seam solvers with diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:  # Optional: only used for flex heatmap rendering
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - headless environments
    plt = None

from smii.rom.constraints import ConstraintRegistry, load_constraints
from smii.rom.flex_heatmap import accumulate_flex_stats, nearest_joint_vertices, project_flex_to_vertices
from smii.rom.seam_costs import load_seam_cost_field
from smii.seams.diagnostics import build_diagnostics_report, render_overlay, save_report
from smii.seams.fabric_kernels import FabricProfile, load_fabrics_from_dir
from smii.seams.kernels import KernelWeights, build_edge_kernels
from smii.seams.mdl import MDLPrior
from smii.seams.pda import solve_seams_pda
from smii.seams.solver import solve_seams
from smii.seams.solvers_mincut import solve_seams_mincut
from smii.seams.solvers_sp import solve_seams_shortest_path
from suit.seam_generator import SeamGenerator


def _load_body(body_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    payload = np.load(body_path)
    vertices = np.asarray(payload["vertices"], dtype=float)
    faces = np.asarray(payload["faces"], dtype=int)
    joints = np.asarray(payload["joints"], dtype=float) if "joints" in payload else None
    return vertices, faces, joints


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


def _resolve_fabric_profile(
    fabric_catalog: Mapping[str, FabricProfile] | None,
    fabric_id: str | None,
) -> FabricProfile | None:
    if not fabric_catalog:
        return None
    if fabric_id:
        profile = fabric_catalog.get(str(fabric_id))
        if profile is None:
            available = ", ".join(sorted(fabric_catalog.keys()))
            raise KeyError(f"Unknown fabric '{fabric_id}'. Available: {available}")
        return profile
    return next(iter(fabric_catalog.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ROM-driven seam solvers with diagnostics.")
    parser.add_argument("--body", type=Path, required=True, help="NPZ containing vertices (N,3) and faces (M,3).")
    parser.add_argument("--rom-costs", type=Path, required=True, help="NPZ produced by save_seam_cost_field.")
    parser.add_argument("--rom-meta", type=Path, default=None, help="Optional sampler metadata JSON (defaults to rom-costs with .json).")
    parser.add_argument(
        "--constraints",
        type=Path,
        help="Optional constraint manifest directory (forbidden_vertices.json, etc.).",
    )
    parser.add_argument(
        "--solver",
        choices=["pda", "mst", "shortest_path", "mincut"],
        default="pda",
        help="Solver to run.",
    )
    parser.add_argument("--weights", type=Path, default=Path("configs/kernel_weights.yaml"), help="YAML/JSON for kernel weights.")
    parser.add_argument("--mdl", type=Path, default=Path("configs/mdl_prior.yaml"), help="YAML/JSON for MDL prior.")
    parser.add_argument(
        "--fabric-grain",
        type=str,
        help="Optional fabric grain direction as comma-separated vector (e.g., 1,0,0).",
    )
    parser.add_argument(
        "--fabric-dir",
        type=Path,
        default=None,
        help="Optional directory with fabric YAML/JSON profiles.",
    )
    parser.add_argument(
        "--fabric-id",
        type=str,
        default=None,
        help="Fabric id to use when --fabric-dir is provided (defaults to first profile).",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/seams"), help="Output directory.")
    parser.add_argument("--vertex-weight", type=float, default=0.0, help="Vertex weight contribution for solvers.")
    parser.add_argument("--budget", type=int, default=8, help="Iteration budget for PDA solver.")
    parser.add_argument(
        "--flex-heatmap",
        action="store_true",
        help="Emit joint flex heatmap (PNG) and CSV using ROM metadata (requires sampler metadata JSON alongside NPZ).",
    )
    parser.add_argument(
        "--joint-centers",
        type=Path,
        default=None,
        help="Optional joint centers file (JSON with name->[x,y,z] or NPY/NPZ) to map flex heatmap to vertices; falls back to body npz 'joints' entry or midpoint.",
    )
    args = parser.parse_args()

    vertices, faces, joints = _load_body(args.body)
    cost_field = load_seam_cost_field(args.rom_costs)
    generator = SeamGenerator()
    axis_map = {
        "longitudinal": np.array([0.0, 0.0, 1.0], dtype=float),
        "lateral": np.array([1.0, 0.0, 0.0], dtype=float),
    }
    seam_graph = generator.generate(vertices, faces, measurement_loops=None, axis_map=axis_map)

    constraints: ConstraintRegistry | None = None
    if args.constraints:
        constraints = load_constraints(args.constraints)

    fabric_grain = None
    if args.fabric_grain:
        parts = [float(p) for p in args.fabric_grain.split(",")]
        if len(parts) == 3:
            fabric_grain = np.array(parts, dtype=float)

    fabric_catalog = load_fabrics_from_dir(args.fabric_dir) if args.fabric_dir else None
    fabric_profile = _resolve_fabric_profile(fabric_catalog, args.fabric_id)

    weights = _weights_from_path(args.weights)
    mdl_prior = _mdl_from_path(args.mdl)
    kernels = build_edge_kernels(
        cost_field,
        seam_graph,
        vertices=vertices,
        fabric_grain=fabric_grain,
        fabric_profile=fabric_profile,
        constraints=constraints,
    )
    panel_fabrics = (
        {panel.name: fabric_profile.fabric_id for panel in seam_graph.panels}
        if fabric_profile is not None
        else None
    )

    if args.solver == "pda":
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
            fabric_catalog=fabric_catalog,
            fabric_grain=fabric_grain,
            panel_fabrics=panel_fabrics,
        )
    elif args.solver == "mst":
        solution = solve_seams(
            seam_graph,
            cost_field,
            constraints=constraints,
            kernels=kernels,
            kernel_weights=weights,
            vertex_weight=args.vertex_weight,
        )
    elif args.solver == "shortest_path":
        solution = solve_seams_shortest_path(
            seam_graph,
            kernels,
            weights,
            mdl_prior,
            constraints,
            cost_field=cost_field,
            vertices=vertices,
            vertex_weight=args.vertex_weight,
        )
    else:
        solution = solve_seams_mincut(
            seam_graph,
            kernels,
            weights,
            mdl_prior,
            constraints,
            cost_field=cost_field,
            vertices=vertices,
            vertex_weight=args.vertex_weight,
        )

    report = build_diagnostics_report(
        seam_graph,
        solution,
        kernels,
        weights,
        mdl_prior,
        vertices=vertices,
        cost_field=cost_field,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    save_report(report, args.out / "seam_report.json")
    overlay_path = render_overlay(seam_graph, solution, vertices=vertices, cost_field=cost_field, output=args.out / "overlay.png")
    if args.flex_heatmap:
        meta_path = args.rom_meta if args.rom_meta else args.rom_costs.with_suffix(".json")
        if not meta_path.exists():
            print("Flex heatmap requested but sampler metadata JSON not found next to rom-costs; skipping.")
        else:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            pose_metadata = meta.get("pose_metadata", [])
            pose_ids = meta.get("pose_ids", [])
            flex_samples = [
                type("FlexSample", (), {"pose_id": pid, "weight": 1.0, "metadata": m})()
                for pid, m in zip(pose_ids, pose_metadata)
            ]
            stats = accumulate_flex_stats(flex_samples)
            joint_centers = None
            if args.joint_centers and args.joint_centers.exists():
                if args.joint_centers.suffix in {".json", ".yaml", ".yml"}:
                    joint_centers = json.loads(args.joint_centers.read_text(encoding="utf-8"))
                else:
                    payload = np.load(args.joint_centers)
                    joint_centers = payload.get("joints") if hasattr(payload, "get") else payload
            elif joints is not None:
                joint_centers = joints

            arr_centers: dict[str, np.ndarray] = {}
            if isinstance(joint_centers, dict):
                arr_centers = {k: np.asarray(v, dtype=float).reshape(3) for k, v in joint_centers.items()}
            elif joint_centers is not None and hasattr(joint_centers, "__len__"):
                arr_centers = {str(idx): np.asarray(vec, dtype=float).reshape(3) for idx, vec in enumerate(joint_centers)}

            joint_vertices: dict[str, int] = {}
            if arr_centers:
                joint_vertices = nearest_joint_vertices(arr_centers, vertices)
            else:
                mid = len(vertices) // 2 if len(vertices) else 0
                for name in stats.joint_max_abs:
                    joint_vertices[name.split(":")[0]] = mid
            flex_field = project_flex_to_vertices(stats, joint_vertices=joint_vertices, vertex_count=len(vertices), mode="max")
            flex_dir = args.out / "flex_heatmap"
            flex_dir.mkdir(parents=True, exist_ok=True)
            csv_path = flex_dir / "flex_stats.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["joint_axis", "max_abs_deg", "mean_abs_deg"])
                for joint_axis, val in sorted(stats.joint_max_abs.items()):
                    writer.writerow([joint_axis, val, stats.joint_mean_abs.get(joint_axis, 0.0)])
            if plt is not None:
                flex_norm = flex_field / max(np.max(flex_field), 1e-6)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=flex_norm, cmap="magma", s=3, alpha=0.6)
                ax.set_axis_off()
                png_path = flex_dir / "flex_heatmap.png"
                plt.savefig(png_path, bbox_inches="tight", dpi=200)
                plt.close(fig)
                # Overlay with seams
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=flex_norm, cmap="magma", s=2, alpha=0.35)
                seam_points: list[int] = []
                for panel in seam_graph.panels:
                    seam_points.extend(panel.seam_vertices)
                if seam_points:
                    seam_points = sorted(set(seam_points))
                    ax.scatter(
                        vertices[seam_points, 0],
                        vertices[seam_points, 1],
                        vertices[seam_points, 2],
                        color="cyan",
                        s=6,
                        alpha=0.8,
                        label="seam vertices",
                    )
                ax.set_axis_off()
                overlay_png = flex_dir / "flex_heatmap_with_seams.png"
                plt.savefig(overlay_png, bbox_inches="tight", dpi=200)
                plt.close(fig)
                print(f"Flex heatmap: {png_path}")
                print(f"Flex heatmap with seams: {overlay_png}")
            print(f"Flex stats CSV: {csv_path}")
    print(f"Solver: {solution.solver}")
    if fabric_profile is not None:
        print(f"Fabric: {fabric_profile.fabric_id}")
    print(f"Total cost: {solution.total_cost:.4f}")
    print(f"Report: {args.out / 'seam_report.json'}")
    if overlay_path:
        print(f"Overlay: {overlay_path}")


if __name__ == "__main__":
    main()
