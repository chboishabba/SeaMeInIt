"""Aggregate ROM samples (real or demo) and emit hotspot diagnostics.

Examples:
    # Run against the bundled demo payload with an identity basis
    python examples/rom_aggregate_from_samples.py

    # Use a generated canonical basis and your sampler output
    python examples/rom_aggregate_from_samples.py \\
        --samples /path/to/rom_samples.json \\
        --basis outputs/rom/canonical_basis.npz \\
        --basis-receipt outputs/rom/basis_receipt.json \\
        --out-rom-field-receipt outputs/rom/rom_field_receipt.json \\
        --gate-manifest data/constraints/coupling_manifest.json

Running without `--basis-receipt` remains diagnostic-only for the receipt
orchestrator.

Outputs are written to `outputs/rom/` (ignored by git).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from smii.rom import (
    KernelBasis,
    KernelProjector,
    ROMFieldReceipt,
    RomGate,
    RomSample,
    aggregate_fields,
    annotate_seam_graph_with_costs,
    build_gate_from_manifest,
    build_seam_cost_field,
    can_consume_basis_receipt,
    load_basis,
    load_basis_receipt,
    load_coupling_manifest,
    save_seam_cost_field,
)


def _load_samples(
    path: Path,
) -> tuple[list[RomSample], Sequence[tuple[int, int]] | None, list[str], Mapping[str, Any]]:
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

    meta: Mapping[str, Any] = {}
    if isinstance(payload, Mapping) and isinstance(payload.get("meta"), Mapping):
        meta = payload["meta"]

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

    return samples, edges, fields, meta


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


def _save_rom_fields(fields_path: Path, aggregation) -> None:
    fields_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for name, stats in aggregation.per_field.items():
        payload[f"{name}_mean"] = stats.mean
        payload[f"{name}_peak"] = stats.maximum
        payload[f"{name}_variance"] = stats.variance
    np.savez_compressed(fields_path, **payload)
    print(f"Wrote ROM fields to {fields_path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field_uniformity(field: np.ndarray) -> float:
    finite = np.asarray(field, dtype=float)
    if finite.ndim != 1:
        raise ValueError("Uniformity field must be one-dimensional.")
    if not np.isfinite(finite).all():
        raise ValueError("Uniformity field must contain only finite values.")
    spread = float(np.std(finite))
    magnitude = float(np.max(np.abs(finite)))
    if magnitude <= 1e-12:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - (spread / (magnitude + 1e-8)))))


def _pose_source(sample_meta: Mapping[str, Any], *, synthetic: bool) -> str:
    raw_source = sample_meta.get("pose_source") or sample_meta.get("source")
    if isinstance(raw_source, str) and raw_source:
        return raw_source
    if synthetic:
        return "synthetic_smplx_sweep"
    return "rom_corpus_aggregated"


def _emit_rom_field_receipt(
    *,
    basis_receipt_path: Path,
    samples_path: Path,
    aggregation_summary_path: Path,
    fields_path: Path,
    receipt_path: Path,
    aggregation,
    basis,
    target_field: str,
    sample_meta: Mapping[str, Any],
    allow_synthetic_promotion: bool,
) -> ROMFieldReceipt:
    basis_receipt = load_basis_receipt(basis_receipt_path)
    if not can_consume_basis_receipt(basis_receipt, "rom_field_aggregation"):
        raise ValueError(
            "BasisReceipt not promoted "
            f"(status={basis_receipt.promotion}). "
            f"Blocked: {basis_receipt.blocked_consumers}"
        )
    if basis.metadata.vertex_count != basis_receipt.basis_vertex_count:
        raise ValueError(
            "Basis vertex count mismatch with BasisReceipt: "
            f"basis={basis.metadata.vertex_count}, receipt={basis_receipt.basis_vertex_count}."
        )
    if basis.metadata.component_count != basis_receipt.basis_dimension:
        raise ValueError(
            "Basis dimension mismatch with BasisReceipt: "
            f"basis={basis.metadata.component_count}, receipt={basis_receipt.basis_dimension}."
        )

    field_names = sorted(aggregation.per_field.keys())
    uniformity_field_name = "pressure" if "pressure" in aggregation.per_field else target_field
    pressure_stats = aggregation.per_field.get(uniformity_field_name)
    if pressure_stats is None:
        raise KeyError(f"Aggregation missing target field '{uniformity_field_name}'.")
    peak_field = pressure_stats.maximum
    field_uniformity = _field_uniformity(peak_field)

    synthetic = bool(sample_meta.get("synthetic", False))
    promotes = field_uniformity < 0.95 and (not synthetic or allow_synthetic_promotion)
    receipt = ROMFieldReceipt(
        basis_receipt_hash=_sha256_file(basis_receipt_path),
        samples_hash=_sha256_file(samples_path),
        aggregation_summary_hash=_sha256_file(aggregation_summary_path),
        fields_hash=_sha256_file(fields_path),
        pose_count=int(aggregation.sample_count),
        total_samples=int(aggregation.total_samples),
        pose_source=_pose_source(sample_meta, synthetic=synthetic),
        fields_computed=field_names,
        vertex_count=int(basis_receipt.basis_vertex_count),
        peak_pressure_max=float(np.max(peak_field)),
        peak_pressure_percentile95=float(np.percentile(peak_field, 95)),
        field_uniformity=field_uniformity,
        synthetic=synthetic,
        promotion=1 if promotes else 0,
        blocked_consumers=[] if promotes else [],
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt.to_json(receipt_path)
    print(f"Wrote ROM field receipt to {receipt_path}")
    return receipt


def _validate_basis_receipt_for_aggregation(
    *,
    basis_receipt_path: Path,
    basis,
) -> None:
    basis_receipt = load_basis_receipt(basis_receipt_path)
    if not can_consume_basis_receipt(basis_receipt, "rom_field_aggregation"):
        raise ValueError(
            "BasisReceipt not promoted "
            f"(status={basis_receipt.promotion}). "
            f"Blocked: {basis_receipt.blocked_consumers}"
        )
    if basis.metadata.vertex_count != basis_receipt.basis_vertex_count:
        raise ValueError(
            "Basis vertex count mismatch with BasisReceipt: "
            f"basis={basis.metadata.vertex_count}, receipt={basis_receipt.basis_vertex_count}."
        )
    if basis.metadata.component_count != basis_receipt.basis_dimension:
        raise ValueError(
            "Basis dimension mismatch with BasisReceipt: "
            f"basis={basis.metadata.component_count}, receipt={basis_receipt.basis_dimension}."
        )


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
    samples, edges, fields, sample_meta = _load_samples(args.samples)
    component_count = len(next(iter(samples[0].coeffs.values())))
    if args.out_rom_field_receipt is not None and args.basis_receipt is None:
        raise ValueError("--out-rom-field-receipt requires --basis-receipt.")
    if args.basis_receipt is not None and args.basis is None:
        raise ValueError("--basis-receipt requires --basis.")
    basis = _resolve_basis(args.basis, component_count)
    if args.basis_receipt is not None:
        _validate_basis_receipt_for_aggregation(
            basis_receipt_path=args.basis_receipt,
            basis=basis,
        )
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
    aggregation_summary_path = args.output_dir / "aggregation_summary.json"
    _save_diagnostics(aggregation_summary_path, aggregation)
    fields_path = args.out_rom_fields or (args.output_dir / "rom_fields.npz")
    _save_rom_fields(fields_path, aggregation)
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

    if args.basis_receipt is not None:
        receipt_path = args.out_rom_field_receipt or (args.output_dir / "rom_field_receipt.json")
        _emit_rom_field_receipt(
            basis_receipt_path=args.basis_receipt,
            samples_path=args.samples,
            aggregation_summary_path=aggregation_summary_path,
            fields_path=fields_path,
            receipt_path=receipt_path,
            aggregation=aggregation,
            basis=basis,
            target_field=target_field,
            sample_meta=sample_meta,
            allow_synthetic_promotion=args.allow_synthetic_promotion,
        )

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
        "--basis-receipt",
        type=Path,
        default=None,
        help="Promoted BasisReceipt JSON required to emit a ROMFieldReceipt.",
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
        "--out-rom-fields",
        type=Path,
        default=None,
        help="Output path for aggregated ROM field NPZ (default: <output-dir>/rom_fields.npz).",
    )
    parser.add_argument(
        "--out-rom-field-receipt",
        type=Path,
        default=None,
        help=(
            "Output path for rom_field_receipt.json "
            "(default: <output-dir>/rom_field_receipt.json)."
        ),
    )
    parser.add_argument(
        "--allow-synthetic-promotion",
        action="store_true",
        help="Allow synthetic sample payloads to promote if field uniformity passes.",
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
