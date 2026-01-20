"""Aggregate ROM samples (real or demo) and emit hotspot diagnostics.

Examples:
    # Run against the bundled demo payload with an identity basis
    python examples/rom_aggregate_from_samples.py

    # Use a generated canonical basis and your sampler output
    python examples/rom_aggregate_from_samples.py \\
        --samples /path/to/rom_samples.json \\
        --basis outputs/rom/canonical_basis.npz \\
        --gate-manifest data/constraints/coupling_manifest.json

Outputs are written to `outputs/rom/` (ignored by git).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from smii.rom import (
    KernelBasis,
    KernelProjector,
    RomGate,
    RomSample,
    aggregate_fields,
    annotate_seam_graph_with_costs,
    build_gate_from_manifest,
    build_seam_cost_field,
    load_basis,
    load_coupling_manifest,
    save_seam_cost_field,
)


def _load_samples(path: Path) -> tuple[list[RomSample], Sequence[tuple[int, int]] | None, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and "samples" in payload:
        entries = payload["samples"]
    else:
        entries = payload
    if not isinstance(entries, Sequence):
        raise TypeError("Samples payload must be a list or contain a 'samples' list.")

    edges = None
    if isinstance(payload, Mapping) and "edges" in payload:
        raw_edges = payload["edges"]
        if isinstance(raw_edges, Sequence):
            edges = [tuple(map(int, edge)) for edge in raw_edges]  # type: ignore[list-item]

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
    # Fall back to an identity basis so demo payloads run without assets
    matrix = np.eye(component_count)
    return KernelBasis.from_arrays(matrix)


def _maybe_gate(manifest_path: Path | None) -> RomGate | None:
    if manifest_path is None:
        return None
    manifest = load_coupling_manifest(manifest_path)
    return build_gate_from_manifest(manifest)


def _save_diagnostics(aggregation_path: Path, aggregation) -> None:
    aggregation_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_count": aggregation.sample_count,
        "total_samples": aggregation.total_samples,
        "rejection_report": {
            "accepted": aggregation.rejection_report.accepted_samples,
            "rejected": aggregation.rejection_report.rejected_samples,
            "rejection_rate": aggregation.rejection_report.rejection_rate,
            "reasons": [reason.__dict__ for reason in aggregation.rejection_report.reasons],
        },
        "fields": {
            name: {
                "mean": stats.mean.tolist(),
                "max": stats.maximum.tolist(),
                "variance": stats.variance.tolist(),
                "sample_count": stats.sample_count,
            }
            for name, stats in aggregation.per_field.items()
        },
    }
    aggregation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote aggregation summary to {aggregation_path}")


def _maybe_plot_hotspots(output_dir: Path, aggregation) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional diagnostic dependency
        print("matplotlib not installed; skipping hotspot plot.")
        return

    for field, diagnostics in aggregation.diagnostics.items():
        indices = [hotspot.index for hotspot in diagnostics.vertex_hotspots]
        variances = [hotspot.variance for hotspot in diagnostics.vertex_hotspots]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(indices, variances)
        ax.set_title(f"Top hotspots for {field}")
        ax.set_xlabel("Vertex index")
        ax.set_ylabel("Variance")
        ax.grid(True, alpha=0.3)
        output = output_dir / f"hotspots_{field}.png"
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        print(f"Saved hotspot plot to {output}")


def _main(args: argparse.Namespace) -> None:
    samples, edges, fields = _load_samples(args.samples)
    component_count = len(next(iter(samples[0].coeffs.values())))
    basis = _resolve_basis(args.basis, component_count)
    projector = KernelProjector(basis)
    gate = _maybe_gate(args.gate_manifest)

    aggregation = aggregate_fields(
        samples,
        projector,
        field_keys=fields,
        edges=edges,
        gate=gate,
        diagnostics_top_k=args.top_k,
    )
    _save_diagnostics(args.output_dir / "aggregation_summary.json", aggregation)
    _maybe_plot_hotspots(args.output_dir, aggregation)

    target_field = args.field or fields[0]
    cost_field = build_seam_cost_field(
        aggregation,
        field=target_field,
        variance_weight=args.variance_weight,
        maximum_weight=args.maximum_weight,
    )
    if args.save_costs is not None:
        save_seam_cost_field(cost_field, args.save_costs)
        print(f"Saved seam cost field to {args.save_costs}")

    if args.seam_loops is not None:
        # Optional seam cost mapping if a seam graph payload is provided (expects seam_vertices per panel)
        seam_payload = json.loads(Path(args.seam_loops).read_text(encoding="utf-8"))
        from suit.seam_generator import SeamGraph, SeamPanel

        panels: list[SeamPanel] = []
        for panel_entry in seam_payload.get("panels", []):
            panels.append(
                SeamPanel(
                    name=panel_entry["name"],
                    anchor_loops=tuple(panel_entry.get("anchor_loops", ("lower", "upper"))),  # type: ignore[arg-type]
                    side=panel_entry.get("side", "unknown"),
                    vertices=np.asarray(panel_entry["vertices"], dtype=float),
                    faces=np.asarray(panel_entry["faces"], dtype=int),
                    global_indices=tuple(panel_entry["global_indices"]),
                    seam_vertices=tuple(panel_entry["seam_vertices"]),
                    loop_vertex_indices=tuple(panel_entry.get("loop_vertex_indices", ())),
                    metadata=panel_entry.get("metadata", {}),
                )
            )
        seam_graph = SeamGraph(
            panels=tuple(panels),
            measurement_loops=tuple(),
            seam_metadata=seam_payload.get("seam_metadata", {}),
        )
        enriched = annotate_seam_graph_with_costs(cost_field, seam_graph)
        output = args.output_dir / "seam_costs.json"
        output.write_text(json.dumps(enriched.seam_costs, indent=2, default=float), encoding="utf-8")
        print(f"Wrote seam cost mapping to {output}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("examples/data/rom_samples_demo.json"),
        help="Path to sampler output (JSON with samples[]).",
    )
    parser.add_argument(
        "--basis",
        type=Path,
        default=None,
        help="Path to a canonical basis NPZ. Defaults to identity basis sized to the samples.",
    )
    parser.add_argument(
        "--field",
        type=str,
        default=None,
        help="Field name to derive seam costs from (default: first field in samples).",
    )
    parser.add_argument(
        "--gate-manifest",
        type=Path,
        default=None,
        help="Optional coupling manifest JSON used to gate samples.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Hotspots per field to surface.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rom"),
        help="Directory for aggregation summaries and plots (default: outputs/rom).",
    )
    parser.add_argument(
        "--variance-weight",
        type=float,
        default=1.0,
        help="Weight applied to variance term when deriving seam costs.",
    )
    parser.add_argument(
        "--maximum-weight",
        type=float,
        default=0.25,
        help="Weight applied to maximum term when deriving seam costs.",
    )
    parser.add_argument(
        "--save-costs",
        type=Path,
        default=Path("outputs/rom/seam_costs.npz"),
        help="Optional path to save seam cost field NPZ (default: outputs/rom/seam_costs.npz).",
    )
    parser.add_argument(
        "--seam-loops",
        type=str,
        default=None,
        help="Optional seam graph JSON to map costs onto (panels with seam_vertices/global_indices).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    _main(parse_args())
