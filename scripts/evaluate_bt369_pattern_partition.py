#!/usr/bin/env python3
"""Evaluate BT369-native pattern partitions through the existing Gate-6 backends.

This is intentionally a panelization experiment, not a new UV solver. It
builds a bounded family of BT369 face partitions from the already measured
serializer failure field, deduplicates identical topologies, rebuilds exact
face-backed charts, and reruns the same bootstrap/LSCM/xatlas competition used
by the current materialization-aware unwrap path.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from scripts import evaluate_operator_tree as tree_eval
from scripts import unwrap_panels as unwrap_mod
from smii.unwrap.bt369_manufacturing import (
    ManufacturingLimits,
    chart_diagnostics,
    regularize_partition,
)
from smii.unwrap.bt369_seam_tracing import (
    flow_trace_partition,
    morphological_cleanup,
    sensitivity,
)
from smii.unwrap.bt369_pattern_serializer import (
    BT369PatternPartitionReceipt,
    build_bt369_pattern_partition,
    default_bt369_pattern_policy_grid,
)

SCHEMA_VERSION = "smii.bt369_pattern_backend_evaluation.v1"
CLAIM_BOUNDARY = "bt369_partition_competes_under_existing_gate6_not_native_promotion_authority"
DEFAULT_PANEL_IDS = ("P0", "P2")


def _gate_receipt(
    *,
    original_metrics: Mapping[str, object],
    measured_metrics: Mapping[str, object],
    split_valid: bool,
    parent_chart_count: int,
) -> dict[str, object]:
    """Apply the unchanged parent-surgery gate plus the no-identity requirement."""

    original_score = float(cast(Any, original_metrics.get("score", math.inf)))
    original_distortion = float(cast(Any, original_metrics.get("worst_distortion", math.inf)))
    original_foldovers = int(cast(Any, original_metrics.get("foldovers", 0)))
    score = float(cast(Any, measured_metrics.get("score", math.inf)))
    distortion = float(cast(Any, measured_metrics.get("worst_distortion", math.inf)))
    foldovers = int(cast(Any, measured_metrics.get("foldovers", 0)))
    exportable = bool(measured_metrics.get("exportable", False))

    blockers: list[str] = []
    if not split_valid:
        blockers.append("bt369_chart_domain_invalid")
    if int(parent_chart_count) <= 1:
        blockers.append("bt369_partition_did_not_split_parent")
    if score >= original_score:
        blockers.append("score_not_improved")
    if distortion > original_distortion + 1e-12:
        blockers.append("distortion_regresses")
    if foldovers > original_foldovers:
        blockers.append("foldovers_regress")
    if not exportable:
        blockers.append("not_exportable")

    return {
        "promotion": 1 if not blockers else 0,
        "blockers": blockers,
        "parent_acceptance": bool(
            split_valid
            and parent_chart_count > 1
            and score < original_score
            and distortion <= original_distortion + 1e-12
        ),
        "foldover_guard": foldovers <= original_foldovers,
        "exportable_guard": exportable,
    }


def _unique_partitions(
    receipts: Sequence[BT369PatternPartitionReceipt],
) -> list[dict[str, object]]:
    """Deduplicate a policy sweep by exact face partition, preserving aliases."""

    by_hash: dict[str, dict[str, object]] = {}
    for receipt in receipts:
        item = by_hash.get(receipt.topology_hash)
        if item is None:
            by_hash[receipt.topology_hash] = {
                "receipt": receipt,
                "policy_aliases": [receipt.policy.policy_key],
            }
            continue
        aliases = cast(list[str], item["policy_aliases"])
        if receipt.policy.policy_key not in aliases:
            aliases.append(receipt.policy.policy_key)
    return [
        {
            "receipt": item["receipt"],
            "policy_aliases": sorted(cast(list[str], item["policy_aliases"])),
        }
        for _topology_hash, item in sorted(by_hash.items())
    ]


def _failure_field_for_panel(
    domains: Mapping[str, Any],
    panel_index: int,
) -> Mapping[str, Any]:
    raw = domains.get("parent_serialization_failure_fields", [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise TypeError("parent_serialization_failure_fields must be a list")
    if panel_index >= len(raw) or not isinstance(raw[panel_index], Mapping):
        raise ValueError(f"missing measured serializer failure field for P{panel_index}")
    return cast(Mapping[str, Any], raw[panel_index])


def _materialized_support(
    *,
    receipt: Mapping[str, Any],
    vertices: Any,
    source_panels: Sequence[unwrap_mod.PanelPatch],
    domains: Mapping[str, Any],
    parent_index: int,
) -> tuple[Any, list[unwrap_mod.PanelPatch], list[str]]:
    """Recover authorized synthetic gusset patches without choosing a local surgery."""

    materialization = receipt.get("correction_tree_materialization_receipt")
    if not isinstance(materialization, Mapping):
        return vertices, [], []
    failure_fields_raw = domains.get("parent_serialization_failure_fields", [])
    failure_fields = [
        item if isinstance(item, Mapping) else None
        for item in cast(Sequence[object], failure_fields_raw)
    ]
    (
        materialized_vertices,
        chart_panels,
        chart_parent_indices,
        chart_kinds,
        _chart_variant_ids,
    ) = unwrap_mod._materialized_chart_panels(
        vertices=vertices,
        panels=source_panels,
        materialization_receipt=materialization,
        failure_fields=cast(Sequence[Mapping[str, object] | None], failure_fields),
    )
    patches: list[unwrap_mod.PanelPatch] = []
    patch_kinds: list[str] = []
    seen: set[str] = set()
    for idx, (source_idx, kind_raw) in enumerate(
        zip(chart_parent_indices, chart_kinds, strict=False)
    ):
        kind = str(kind_raw)
        if source_idx != parent_index or not kind.startswith("inserted_gusset_patch:"):
            continue
        if kind in seen:
            continue
        seen.add(kind)
        patches.append(chart_panels[idx])
        patch_kinds.append(kind)
    return materialized_vertices, patches, patch_kinds


def _evaluate_panel(
    *,
    panel_id: str,
    receipt: Mapping[str, Any],
    domains: Mapping[str, Any],
    vertices: Any,
    source_panels: Sequence[unwrap_mod.PanelPatch],
) -> dict[str, object]:
    parent_index = tree_eval._panel_index(panel_id)
    if parent_index >= len(source_panels):
        raise ValueError(f"source panel {panel_id} does not exist")
    panel = source_panels[parent_index]
    failure_field = _failure_field_for_panel(domains, parent_index)
    basis_receipt = tree_eval._basis_receipt_for_panel(domains, panel_id)
    initial_raw = basis_receipt.get("initial_metrics", {})
    if not isinstance(initial_raw, Mapping):
        raise TypeError("operator basis initial_metrics must be an object")
    original_metrics = dict(initial_raw)
    distortion_threshold = float(receipt.get("distortion_threshold", 0.05))

    sweep_receipts = [
        build_bt369_pattern_partition(
            vertices=vertices,
            panel=panel,
            failure_field=failure_field,
            distortion_threshold=distortion_threshold,
            policy=policy,
            panel_id=panel_id,
        )
        for policy in default_bt369_pattern_policy_grid()
    ]
    unique = _unique_partitions(sweep_receipts)
    materialized_vertices, auxiliary_patches, auxiliary_patch_kinds = _materialized_support(
        receipt=receipt,
        vertices=vertices,
        source_panels=source_panels,
        domains=domains,
        parent_index=parent_index,
    )

    limits = ManufacturingLimits()
    evaluated: list[dict[str, object]] = []
    seen_family_topologies: set[tuple[tuple[int, ...], ...]] = set()
    for candidate_index, item in enumerate(unique):
        partition_receipt = cast(BT369PatternPartitionReceipt, item["receipt"])
        source_partition = partition_receipt.partition
        source_manufacturing = chart_diagnostics(
            vertices=vertices,
            faces=panel.faces,
            partition=source_partition,
            limits=limits,
        )
        regularized_partition = regularize_partition(
            vertices=vertices,
            faces=panel.faces,
            partition=source_partition,
            cells=partition_receipt.cells,
            limits=limits,
        )
        families = (
            ("merge_back", regularized_partition),
            (
                "morphological",
                morphological_cleanup(
                    vertices=vertices,
                    faces=panel.faces,
                    partition=source_partition,
                    cells=partition_receipt.cells,
                    limits=limits,
                ),
            ),
            (
                "flow_trace",
                flow_trace_partition(
                    vertices=vertices,
                    faces=panel.faces,
                    cells=partition_receipt.cells,
                    limits=limits,
                    target_charts=24,
                ),
            ),
        )
        for family, candidate_partition in families:
            if candidate_partition in seen_family_topologies:
                continue
            seen_family_topologies.add(candidate_partition)
            manufacturing = chart_diagnostics(
                vertices=vertices,
                faces=panel.faces,
                partition=candidate_partition,
                limits=limits,
            )
            candidate_id = f"{panel_id}_BT369_{candidate_index:03d}_{family}"
            base = {
                "candidate_id": candidate_id,
                "variant_id": f"bt369:{partition_receipt.policy.policy_key}:{family}",
                "candidate_family": family,
                "topology_hash": partition_receipt.topology_hash,
                "policy": partition_receipt.policy.to_dict(),
                "policy_aliases": item["policy_aliases"],
                "partition_receipt": partition_receipt.to_dict(),
                "source_manufacturing": source_manufacturing,
                "manufacturing": manufacturing,
                "manufacturing_blockers": list(manufacturing["manufacturing_blockers"]),
                "sensitivity": sensitivity(manufacturing),
                "regularized_partition": [list(c) for c in candidate_partition],
            }
            if manufacturing["manufacturing_blockers"]:
                evaluated.append(
                    {
                        **base,
                        "parent_chart_count": len(candidate_partition),
                        "auxiliary_patch_count": len(auxiliary_patches),
                        "chart_count": len(candidate_partition) + len(auxiliary_patches),
                        "screened_before_backend": True,
                        "geometric_promotion": None,
                        "promotion": 0,
                        "blockers": list(manufacturing["manufacturing_blockers"]),
                        "metrics": None,
                        "backend_competitions": [],
                    }
                )
                print(
                    f"[{panel_id}] {candidate_id} screened: "
                    f"{manufacturing['manufacturing_blockers']}",
                    flush=True,
                )
                continue
            parent_charts = tree_eval._connected_chart_panels(panel, candidate_partition)
            charts = parent_charts + auxiliary_patches
            print(f"[{panel_id}] {candidate_id}: {len(charts)} charts...", flush=True)
            competitions, group_results = tree_eval._run_backend_competition(
                vertices=materialized_vertices,
                charts=charts,
                distortion_threshold=distortion_threshold,
                panel_id=panel_id,
            )
            metrics = unwrap_mod._variant_metrics_from_group_results(group_results)
            split_valid = unwrap_mod._chart_group_backend_serializable(
                materialized_vertices,
                charts,
            )
            gate = _gate_receipt(
                original_metrics=original_metrics,
                measured_metrics=metrics,
                split_valid=split_valid,
                parent_chart_count=len(parent_charts),
            )
            evaluated.append(
                {
                    **base,
                    "parent_chart_count": len(parent_charts),
                    "auxiliary_patch_count": len(auxiliary_patches),
                    "auxiliary_patch_kinds": auxiliary_patch_kinds,
                    "chart_count": len(charts),
                    "split_valid": bool(split_valid),
                    "screened_before_backend": False,
                    "geometric_promotion": gate["promotion"],
                    "metrics": metrics,
                    "metric_comparison": unwrap_mod._metric_comparison_receipt(
                        original_metrics,
                        metrics,
                    ),
                    "backend_competitions": competitions,
                    **gate,
                }
            )

    pareto = unwrap_mod._variant_pareto_receipt(
        original_candidate_id=f"{panel_id}.original",
        original_metrics=original_metrics,
        variant_candidates=[
            {
                "candidate_id": candidate["candidate_id"],
                "variant_id": candidate["variant_id"],
                "metrics": candidate["metrics"],
            }
            for candidate in evaluated
            if candidate["metrics"] is not None
        ],
    )
    pareto_candidates = cast(Sequence[Mapping[str, object]], pareto.get("candidates", []))
    pareto_by_id = {
        str(candidate.get("candidate_id", "")): candidate for candidate in pareto_candidates
    }
    for candidate in evaluated:
        p = pareto_by_id.get(str(candidate["candidate_id"]), {})
        candidate["pareto_frontier_member"] = bool(p.get("pareto_frontier_member", False))
        candidate["pareto_status"] = str(p.get("pareto_status", "unknown"))
        candidate["profile_scores"] = dict(cast(Mapping[str, object], p.get("profile_scores", {})))

    promoted = [candidate for candidate in evaluated if int(candidate["promotion"]) == 1]
    default_scores = [
        (
            float(
                cast(Mapping[str, object], candidate.get("profile_scores", {})).get(
                    "default", math.inf
                )
            ),
            str(candidate["candidate_id"]),
        )
        for candidate in evaluated
        if candidate["metrics"] is not None
    ]
    best_bt369_candidate_id = min(default_scores)[1] if default_scores else None
    return {
        "panel_id": panel_id,
        "initial_metrics": original_metrics,
        "policy_count": len(sweep_receipts),
        "unique_topology_count": len(unique),
        "screened_candidate_count": sum(bool(c["screened_before_backend"]) for c in evaluated),
        "auxiliary_patch_kinds": auxiliary_patch_kinds,
        "candidates": evaluated,
        "pareto_receipt": pareto,
        "best_bt369_candidate_id": best_bt369_candidate_id,
        "promoted_candidate_ids": [str(candidate["candidate_id"]) for candidate in promoted],
        "promotion": 1 if promoted else 0,
        "blockers": [] if promoted else ["bt369_policy_sweep_no_promoted_partition"],
    }


def evaluate(
    *,
    panel_unwrap_receipt_path: Path,
    mesh_path: Path,
    seam_edges_path: Path,
    panel_ids: Sequence[str],
) -> dict[str, object]:
    receipt = tree_eval._load_object(panel_unwrap_receipt_path)
    domains = tree_eval._materialized_domains(receipt)
    vertices, panels_raw = tree_eval._reconstruct_source_panels(mesh_path, seam_edges_path)
    source_panels = cast(Sequence[unwrap_mod.PanelPatch], panels_raw)
    panels = [
        _evaluate_panel(
            panel_id=panel_id,
            receipt=receipt,
            domains=domains,
            vertices=vertices,
            source_panels=source_panels,
        )
        for panel_id in panel_ids
    ]
    promoted_panels = [str(panel["panel_id"]) for panel in panels if int(panel["promotion"]) == 1]
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "panel_unwrap_receipt_hash": tree_eval._sha256(panel_unwrap_receipt_path),
        "mesh_hash": tree_eval._sha256(mesh_path),
        "seam_edges_hash": tree_eval._sha256(seam_edges_path),
        "backend_set": list(unwrap_mod.PANEL_SERIALIZATION_BACKENDS),
        "panel_ids": list(panel_ids),
        "panels": panels,
        "promoted_panel_ids": promoted_panels,
        "all_requested_panels_promoted": len(promoted_panels) == len(panel_ids),
        "promotion": 1 if len(promoted_panels) == len(panel_ids) else 0,
        "blockers": []
        if len(promoted_panels) == len(panel_ids)
        else ["one_or_more_bt369_panel_sweeps_unpromoted"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-unwrap-receipt", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--seam-edges", type=Path, required=True)
    parser.add_argument(
        "--panel-id",
        action="append",
        default=[],
        help="Panel id to evaluate; repeat for multiple panels. Defaults to P0 and P2.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    panel_ids = tuple(args.panel_id) if args.panel_id else DEFAULT_PANEL_IDS
    result = evaluate(
        panel_unwrap_receipt_path=args.panel_unwrap_receipt,
        mesh_path=args.mesh,
        seam_edges_path=args.seam_edges,
        panel_ids=panel_ids,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
