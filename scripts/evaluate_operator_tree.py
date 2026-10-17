#!/usr/bin/env python3
"""Sequentially materialize a retained operator tree and rerun unwrap backends.

This is the executable bridge between `smii.operator_basis_search.v1` and a
measured Gate-6 candidate.  The basis-search receipt only composes measured
single-operator metric deltas for depth > 1.  This runner instead:

1. reconstructs the exact source panel from mesh + seam topology;
2. reuses `unwrap_panels.py`'s already-materialized single-operator chart
   domains as original-face partitions;
3. computes their exact sequential/common refinement;
4. rebuilds connected chart domains from that partition;
5. reruns the same panel serialization backends on every resulting chart;
6. evaluates the unchanged parent score/distortion acceptance rule plus the
   stricter operator-search foldover/exportability guard.

A tree is never promoted from composed deltas.  Promotion here requires newly
measured backend results from the materialized chart domains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np

from smii.seams import (
    PANEL_SERIALIZATION_BACKENDS,
    build_panel_serialization_competition_receipt,
    select_serialization_candidate,
    serialize_panel,
)
from scripts import unwrap_panels as unwrap_mod

SCHEMA_VERSION = "smii.operator_tree_backend_evaluation.v1"
CLAIM_BOUNDARY = (
    "depth_n_tree_is_measured_on_materialized_charts_not_promoted_from_delta_composition"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return dict(payload)


def _panel_index(panel_id: str) -> int:
    if not panel_id.startswith("P") or not panel_id[1:].isdigit():
        raise ValueError("panel id must have the form P<integer>")
    return int(panel_id[1:])


def _materialized_domains(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    fabric = receipt.get("fabric_metric_receipt")
    if not isinstance(fabric, Mapping):
        raise ValueError("panel unwrap receipt has no fabric_metric_receipt")
    domains = fabric.get("materialized_chart_domains")
    if not isinstance(domains, Mapping):
        raise ValueError("panel unwrap receipt has no materialized_chart_domains")
    return domains


def _basis_receipt_for_panel(domains: Mapping[str, Any], panel_id: str) -> Mapping[str, Any]:
    receipts = domains.get("operator_basis_search_receipts", [])
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes, bytearray)):
        raise TypeError("operator_basis_search_receipts must be a list")
    for item in receipts:
        if isinstance(item, Mapping) and str(item.get("panel_id")) == panel_id:
            return item
    raise ValueError(f"no operator basis search receipt for {panel_id}")


def _choose_tree(basis_receipt: Mapping[str, Any], tree_id: str | None) -> Mapping[str, Any]:
    candidates_raw = basis_receipt.get("candidates", [])
    if not isinstance(candidates_raw, Sequence) or isinstance(
        candidates_raw, (str, bytes, bytearray)
    ):
        raise TypeError("operator basis candidates must be a list")
    candidates = [item for item in candidates_raw if isinstance(item, Mapping)]
    if tree_id is not None:
        for candidate in candidates:
            if str(candidate.get("tree_id")) == tree_id:
                return candidate
        raise ValueError(f"tree id {tree_id!r} is not retained in the basis-search receipt")
    depth_n = [candidate for candidate in candidates if int(candidate.get("operator_count", 0)) > 1]
    if not depth_n:
        raise ValueError("no retained depth>1 operator tree is available")

    def rank(candidate: Mapping[str, Any]) -> tuple[float, float, str]:
        profile_scores = candidate.get("profile_scores", {})
        default_score = (
            float(profile_scores.get("default", math.inf))
            if isinstance(profile_scores, Mapping)
            else math.inf
        )
        metrics = candidate.get("metrics", {})
        scalar = float(metrics.get("score", math.inf)) if isinstance(metrics, Mapping) else math.inf
        return default_score, scalar, str(candidate.get("tree_id", ""))

    return min(depth_n, key=rank)


def _reconstruct_source_panels(
    mesh_path: Path,
    seam_edges_path: Path,
) -> tuple[np.ndarray, list[object]]:
    vertices, faces = unwrap_mod._load_mesh(mesh_path)
    seam_payload = unwrap_mod._load_seam_payload(seam_edges_path)
    if seam_payload.face_labels is not None:
        panels = unwrap_mod._extract_panels_from_face_labels(
            faces=faces,
            face_labels=seam_payload.face_labels,
        )
    else:
        panels = unwrap_mod._extract_panels(
            vertex_count=int(vertices.shape[0]),
            faces=faces,
            seam_edges=seam_payload.edges,
        )
    if not panels:
        raise ValueError("no source panels could be reconstructed")
    return vertices, list(panels)


def _face_partition_from_variant(
    *,
    original_panel: object,
    child_panels: Sequence[object],
    variant_id: str,
) -> tuple[tuple[int, ...], ...]:
    original_faces = [
        (int(face[0]), int(face[1]), int(face[2])) for face in getattr(original_panel, "faces")
    ]
    face_to_index: dict[tuple[int, int, int], int] = {}
    for idx, face in enumerate(original_faces):
        if face in face_to_index:
            raise ValueError("source panel contains duplicate oriented triangle faces")
        face_to_index[face] = idx

    groups: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for child in child_panels:
        indices: list[int] = []
        for face_raw in getattr(child, "faces"):
            face = (int(face_raw[0]), int(face_raw[1]), int(face_raw[2]))
            try:
                idx = face_to_index[face]
            except KeyError as exc:
                raise ValueError(
                    f"variant {variant_id!r} contains a face outside the original parent"
                ) from exc
            if idx in seen:
                raise ValueError(f"variant {variant_id!r} overlaps source face {idx}")
            seen.add(idx)
            indices.append(idx)
        if indices:
            groups.append(tuple(sorted(indices)))
    expected = set(range(len(original_faces)))
    if seen != expected:
        missing = sorted(expected - seen)
        if variant_id.startswith("cutout_") and seen:
            groups.append(tuple(missing))
        else:
            raise ValueError(
                f"variant {variant_id!r} is not a full parent partition; missing faces {missing[:16]}"
            )
    return tuple(sorted(groups, key=lambda group: (group[0], len(group), group)))


def _common_refinement(
    partitions: Sequence[tuple[tuple[int, ...], ...]],
    *,
    face_count: int,
) -> tuple[tuple[tuple[int, ...], ...], list[dict[str, object]]]:
    current: tuple[frozenset[int], ...] = (frozenset(range(face_count)),)
    steps: list[dict[str, object]] = []
    for depth, partition in enumerate(partitions, start=1):
        operator_parts = tuple(frozenset(group) for group in partition)
        next_parts: list[frozenset[int]] = []
        for chart in current:
            for operator_part in operator_parts:
                intersection = chart & operator_part
                if intersection:
                    next_parts.append(intersection)
        normalized = tuple(
            sorted(
                set(next_parts),
                key=lambda group: (min(group), len(group), tuple(sorted(group))),
            )
        )
        if set().union(*normalized) != set(range(face_count)):
            raise ValueError(f"operator partition at depth {depth} loses source faces")
        flattened = [idx for group in normalized for idx in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError(f"operator partition at depth {depth} overlaps source faces")
        steps.append(
            {
                "depth": depth,
                "charts_before": len(current),
                "charts_after": len(normalized),
                "effective_refinement": len(normalized) > len(current),
                "partition_after": [sorted(group) for group in normalized],
            }
        )
        current = normalized
    final_partition = tuple(tuple(sorted(group)) for group in current)
    return final_partition, steps


def _connected_chart_panels(
    original_panel: object,
    partition: Sequence[Sequence[int]],
) -> list[unwrap_mod.PanelPatch]:
    source_faces = tuple(getattr(original_panel, "faces"))
    charts: list[unwrap_mod.PanelPatch] = []
    for group in partition:
        face_subset = tuple(source_faces[int(idx)] for idx in group)
        charts.extend(
            unwrap_mod._panel_from_faces(
                cast(unwrap_mod.PanelPatch, original_panel),
                face_subset,
            )
        )
    return charts


def _run_backend_competition(
    *,
    vertices: np.ndarray,
    charts: Sequence[unwrap_mod.PanelPatch],
    distortion_threshold: float,
    panel_id: str,
) -> tuple[
    list[dict[str, object]],
    list[tuple[unwrap_mod.PanelPatch, np.ndarray, float, list[object], str, str]],
]:
    competitions: list[dict[str, object]] = []
    group_results: list[
        tuple[unwrap_mod.PanelPatch, np.ndarray, float, list[object], str, str]
    ] = []
    for chart_idx, chart in enumerate(charts):
        candidates = []
        uv_by_backend: dict[str, np.ndarray] = {}
        for backend in PANEL_SERIALIZATION_BACKENDS:
            candidate, uv = serialize_panel(
                vertices=vertices,
                panel=chart,
                correction_tree=None,
                materialization_constraints={
                    "source_panel_id": panel_id,
                    "chart_materialization_kind": "sequential_operator_tree",
                },
                backend=backend,
                distortion_threshold=distortion_threshold,
            )
            candidates.append(candidate)
            if uv is not None:
                uv_by_backend[backend] = uv
        selected = select_serialization_candidate(candidates)
        uv = uv_by_backend.get(selected.backend)
        competition = build_panel_serialization_competition_receipt(
            panel_id=f"{panel_id}.tree.{chart_idx}",
            correction_tree_hash=None,
            correction_tree=None,
            candidates=candidates,
            selected_backend=selected.backend,
        )
        competitions.append(competition)
        if uv is None or selected.distortion is None:
            raise ValueError(
                f"no measured UV result for materialized chart {chart_idx}; selected {selected.backend}"
            )
        group_results.append(
            (
                chart,
                uv,
                float(selected.distortion),
                cast(list[object], candidates),
                selected.backend,
                "sequential_operator_tree",
            )
        )
    return competitions, group_results


def evaluate(
    *,
    panel_unwrap_receipt_path: Path,
    mesh_path: Path,
    seam_edges_path: Path,
    panel_id: str,
    tree_id: str | None,
) -> dict[str, object]:
    receipt = _load_object(panel_unwrap_receipt_path)
    domains = _materialized_domains(receipt)
    basis_receipt = _basis_receipt_for_panel(domains, panel_id)
    tree = _choose_tree(basis_receipt, tree_id)
    operators_raw = tree.get("operators", [])
    if not isinstance(operators_raw, Sequence) or isinstance(
        operators_raw, (str, bytes, bytearray)
    ):
        raise TypeError("tree operators must be a list")
    operators = [str(operator) for operator in operators_raw]
    if len(operators) < 2:
        raise ValueError("backend tree evaluation is intended for depth>1 candidates")

    vertices, source_panels = _reconstruct_source_panels(mesh_path, seam_edges_path)
    parent_idx = _panel_index(panel_id)
    if parent_idx >= len(source_panels):
        raise ValueError(f"source panel {panel_id} does not exist")
    original_panel = source_panels[parent_idx]

    failure_fields_raw = domains.get("parent_serialization_failure_fields", [])
    if not isinstance(failure_fields_raw, Sequence) or isinstance(
        failure_fields_raw, (str, bytes, bytearray)
    ):
        raise TypeError("parent_serialization_failure_fields must be a list")
    failure_fields = [item if isinstance(item, Mapping) else None for item in failure_fields_raw]

    materialization_receipt = receipt.get("correction_tree_materialization_receipt")
    if not isinstance(materialization_receipt, Mapping):
        raise ValueError("panel unwrap receipt has no correction_tree_materialization_receipt")
    (
        materialized_vertices,
        chart_panels,
        chart_parent_indices,
        chart_kinds,
        chart_variant_ids,
    ) = unwrap_mod._materialized_chart_panels(
        vertices=vertices,
        panels=cast(Sequence[unwrap_mod.PanelPatch], source_panels),
        materialization_receipt=materialization_receipt,
        failure_fields=cast(Sequence[Mapping[str, object] | None], failure_fields),
    )

    partitions: list[tuple[tuple[int, ...], ...]] = []
    operator_partition_receipts: list[dict[str, object]] = []
    inserted_gusset_patches: list[unwrap_mod.PanelPatch] = []
    seen_inserted_kinds: set[str] = set()
    for variant_id in operators:
        children: list[unwrap_mod.PanelPatch] = []
        for idx, (source_idx, candidate_variant, chart_kind) in enumerate(
            zip(chart_parent_indices, chart_variant_ids, chart_kinds, strict=False)
        ):
            if source_idx != parent_idx or candidate_variant != variant_id:
                continue
            kind_str = str(chart_kind)
            if kind_str.startswith("inserted_gusset_patch:"):
                if kind_str not in seen_inserted_kinds:
                    seen_inserted_kinds.add(kind_str)
                    inserted_gusset_patches.append(chart_panels[idx])
                continue
            children.append(chart_panels[idx])
        if not children:
            raise ValueError(
                f"retained operator {variant_id!r} has no materialized single-operator charts"
            )
        partition = _face_partition_from_variant(
            original_panel=original_panel,
            child_panels=children,
            variant_id=variant_id,
        )
        partitions.append(partition)
        operator_partition_receipts.append(
            {
                "variant_id": variant_id,
                "chart_count": len(partition),
                "partition": [list(group) for group in partition],
            }
        )

    face_count = len(tuple(getattr(original_panel, "faces")))
    final_partition, refinement_steps = _common_refinement(
        partitions,
        face_count=face_count,
    )
    final_charts = (
        _connected_chart_panels(original_panel, final_partition) + inserted_gusset_patches
    )
    if len(final_charts) < 2:
        raise ValueError("sequential operator tree did not produce a chart split")

    distortion_threshold = float(receipt.get("distortion_threshold", 0.05))
    competitions, group_results = _run_backend_competition(
        vertices=materialized_vertices,
        charts=final_charts,
        distortion_threshold=distortion_threshold,
        panel_id=panel_id,
    )
    metrics = unwrap_mod._variant_metrics_from_group_results(group_results)
    original_metrics_raw = basis_receipt.get("initial_metrics", {})
    if not isinstance(original_metrics_raw, Mapping):
        raise TypeError("operator basis initial_metrics must be an object")
    original_metrics = dict(original_metrics_raw)
    comparison = unwrap_mod._metric_comparison_receipt(original_metrics, metrics)

    split_valid = unwrap_mod._chart_group_backend_serializable(
        materialized_vertices,
        final_charts,
    )
    split_score = float(cast(Any, metrics.get("score", math.inf)))
    split_worst_distortion = float(cast(Any, metrics.get("worst_distortion", math.inf)))
    original_score = float(cast(Any, original_metrics.get("score", math.inf)))
    original_distortion = float(cast(Any, original_metrics.get("worst_distortion", math.inf)))
    measured_foldovers = int(cast(Any, metrics.get("foldovers", 0)))
    original_foldovers = int(cast(Any, original_metrics.get("foldovers", 0)))
    existing_parent_acceptance = bool(
        split_valid
        and split_score < original_score
        and split_worst_distortion <= original_distortion + 1e-12
    )
    operator_hard_gate = bool(
        existing_parent_acceptance
        and measured_foldovers <= original_foldovers
        and bool(metrics.get("exportable", False))
    )

    blockers: list[str] = []
    if not split_valid:
        blockers.append("materialized_tree_backend_domain_invalid")
    if split_score >= original_score:
        blockers.append("score_not_improved")
    if split_worst_distortion > original_distortion + 1e-12:
        blockers.append("distortion_regresses")
    if measured_foldovers > original_foldovers:
        blockers.append("foldovers_regress")
    if not bool(metrics.get("exportable", False)):
        blockers.append("not_exportable")

    return {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_panel_unwrap_receipt_hash": _sha256(panel_unwrap_receipt_path),
        "mesh_hash": _sha256(mesh_path),
        "seam_edges_hash": _sha256(seam_edges_path),
        "panel_id": panel_id,
        "source_tree_id": str(tree.get("tree_id", "")),
        "operators": operators,
        "source_composition_mode": str(tree.get("composition_mode", "")),
        "source_tree_metrics_were_diagnostic": not bool(tree.get("materialized", False)),
        "operator_partitions": operator_partition_receipts,
        "refinement_steps": refinement_steps,
        "final_face_partition": [list(group) for group in final_partition],
        "final_connected_chart_count": len(final_charts),
        "backend_competitions": competitions,
        "original_metrics": original_metrics,
        "measured_metrics": metrics,
        "metric_comparison": comparison,
        "existing_parent_acceptance": existing_parent_acceptance,
        "operator_hard_gate_passes": operator_hard_gate,
        "promotion": 1 if operator_hard_gate else 0,
        "blockers": blockers if blockers else [],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-unwrap-receipt", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--seam-edges", type=Path, required=True)
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--tree-id", default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = evaluate(
        panel_unwrap_receipt_path=args.panel_unwrap_receipt,
        mesh_path=args.mesh,
        seam_edges_path=args.seam_edges,
        panel_id=args.panel_id,
        tree_id=args.tree_id,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Wrote {SCHEMA_VERSION} to {args.out} "
        f"(promotion={result['promotion']}, charts={result['final_connected_chart_count']})"
    )


if __name__ == "__main__":
    main()
