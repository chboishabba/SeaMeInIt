"""Sequentially materialize measured panel-surgery operator supports.

The existing Gate-6 operator-basis search can rank depth-2 trees using measured
single-operator metric deltas, but those composed trees are explicitly
non-promoting until their topology is actually materialized.  This module
closes that *topological* gap without pretending that numerical unwrap metrics
compose linearly.

Operators are applied to a partition of the original panel face-index set.  A
support refines every current chart by exact intersection/difference, so face
identity is retained across every depth.  The result is therefore a true
sequential chart partition.  It still requires a numerical unwrap/backend pass
before any garment/panel promotion: materialized topology is not distortion
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "smii.operator_tree_materialization.v1"
CLAIM_BOUNDARY = (
    "exact_sequential_face_partition_only_backend_metrics_must_be_recomputed_before_promotion"
)

FAILURE_FIELD_VARIANT_FAMILIES = {
    "failure_relief_path": ("candidate_relief_paths", "failure_face_indices"),
    "failure_relief_tree": ("candidate_relief_paths", "failure_face_indices"),
    "failure_drain_path": ("selected_drain_paths", "drain_face_indices"),
    "failure_drain_tree": ("selected_drain_paths", "drain_face_indices"),
    "failure_wedge_relief": ("selected_wedge_reliefs", "wedge_face_indices"),
    "failure_wedge_tree": ("selected_wedge_reliefs", "wedge_face_indices"),
    "pareto_guided_dart_wedge": (
        "selected_guided_dart_wedges",
        "wedge_face_indices",
    ),
    "pareto_guided_dart_wedge_tree": (
        "selected_guided_dart_wedges",
        "wedge_face_indices",
    ),
    "failure_lens_patch": ("selected_lens_patches", "patch_face_indices"),
}

__all__ = [
    "CLAIM_BOUNDARY",
    "FAILURE_FIELD_VARIANT_FAMILIES",
    "SCHEMA_VERSION",
    "FacePartitionOperator",
    "OperatorTreeMaterializationReceipt",
    "OperatorTreeMaterializationStep",
    "materialize_operator_tree",
    "operator_from_failure_field",
]


def _face_index(value: object, *, face_count: int, key: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{key} must contain integers")
    try:
        index = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{key} must contain integers") from exc
    if index < 0 or index >= face_count:
        raise ValueError(f"{key} face index {index} is outside [0, {face_count - 1}]")
    return index


def _index_list(value: object, *, face_count: int, key: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{key} must be a sequence of face indices")
    return tuple(sorted({_face_index(item, face_count=face_count, key=key) for item in value}))


@dataclass(frozen=True, slots=True)
class FacePartitionOperator:
    """One measured operator support in original panel face-index space."""

    variant_id: str
    support_faces: tuple[int, ...]
    source_receipt: str

    def __post_init__(self) -> None:
        if not self.variant_id:
            raise ValueError("variant_id must be non-empty")
        if not self.source_receipt:
            raise ValueError("source_receipt must be non-empty")
        if not self.support_faces:
            raise ValueError("support_faces must be non-empty")
        normalized = tuple(sorted(set(int(face) for face in self.support_faces)))
        if any(face < 0 for face in normalized):
            raise ValueError("support_faces must be non-negative")
        object.__setattr__(self, "support_faces", normalized)

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "support_faces": list(self.support_faces),
            "source_receipt": self.source_receipt,
        }


@dataclass(frozen=True, slots=True)
class OperatorTreeMaterializationStep:
    depth: int
    variant_id: str
    support_faces: tuple[int, ...]
    charts_before: int
    charts_after: int
    effective_splits: int
    partition_after: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "variant_id": self.variant_id,
            "support_faces": list(self.support_faces),
            "charts_before": self.charts_before,
            "charts_after": self.charts_after,
            "effective_splits": self.effective_splits,
            "partition_after": [list(chart) for chart in self.partition_after],
        }


@dataclass(frozen=True, slots=True)
class OperatorTreeMaterializationReceipt:
    panel_id: str
    face_count: int
    operators: tuple[FacePartitionOperator, ...]
    steps: tuple[OperatorTreeMaterializationStep, ...]
    final_partition: tuple[tuple[int, ...], ...]
    face_coverage_complete: bool
    charts_pairwise_disjoint: bool
    sequential_materialization_complete: bool
    backend_recheck_required: bool = True
    promotion: int = 0
    claim_boundary: str = CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if not self.panel_id:
            raise ValueError("panel_id must be non-empty")
        if self.face_count <= 0:
            raise ValueError("face_count must be positive")
        if self.promotion != 0:
            raise ValueError(
                "operator-tree topology materialization is diagnostic until backend metrics are recomputed"
            )
        if not self.backend_recheck_required:
            raise ValueError("backend_recheck_required must remain true at this receipt boundary")

    @property
    def blockers(self) -> list[str]:
        blockers: list[str] = []
        if not self.face_coverage_complete:
            blockers.append("materialized_partition_loses_faces")
        if not self.charts_pairwise_disjoint:
            blockers.append("materialized_partition_overlaps_faces")
        if not self.sequential_materialization_complete:
            blockers.append("operator_tree_not_sequentially_materialized")
        blockers.append("backend_metrics_not_recomputed")
        return blockers

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "panel_id": self.panel_id,
            "face_count": self.face_count,
            "operators": [operator.to_dict() for operator in self.operators],
            "steps": [step.to_dict() for step in self.steps],
            "final_partition": [list(chart) for chart in self.final_partition],
            "final_chart_count": len(self.final_partition),
            "face_coverage_complete": self.face_coverage_complete,
            "charts_pairwise_disjoint": self.charts_pairwise_disjoint,
            "sequential_materialization_complete": self.sequential_materialization_complete,
            "backend_recheck_required": self.backend_recheck_required,
            "promotion": self.promotion,
            "blockers": self.blockers,
            "claim_boundary": self.claim_boundary,
        }


def _variant_matches(candidate_variant: str, requested_variant: str) -> bool:
    if candidate_variant == requested_variant:
        return True
    if (
        requested_variant.endswith("_tree")
        and candidate_variant == requested_variant[:-5] + "_path"
    ):
        return True
    return False


def operator_from_failure_field(
    failure_field: Mapping[str, Any],
    *,
    variant_id: str,
    face_count: int,
    source_receipt: str = "serialization_failure_field",
) -> FacePartitionOperator:
    """Recover an original-face support for a measured failure-field variant."""

    try:
        collection_key, support_key = FAILURE_FIELD_VARIANT_FAMILIES[variant_id]
    except KeyError as exc:
        raise ValueError(f"unsupported failure-field variant '{variant_id}'") from exc
    collection = failure_field.get(collection_key, [])
    if not isinstance(collection, Sequence) or isinstance(collection, (str, bytes, bytearray)):
        raise TypeError(f"failure field '{collection_key}' must be a sequence")

    support: set[int] = set()
    for candidate in collection:
        if not isinstance(candidate, Mapping):
            continue
        if collection_key == "candidate_relief_paths" and not bool(
            candidate.get("separates_bad_region", False)
        ):
            continue
        raw_indices = candidate.get(support_key, [])
        support.update(
            _index_list(
                raw_indices,
                face_count=face_count,
                key=f"{collection_key}.{support_key}",
            )
        )
    if not support:
        raise ValueError(
            f"failure field contains no materializable support for variant '{variant_id}'"
        )
    return FacePartitionOperator(
        variant_id=variant_id,
        support_faces=tuple(sorted(support)),
        source_receipt=source_receipt,
    )


def _partition_is_disjoint(partition: Sequence[Sequence[int]]) -> bool:
    seen: set[int] = set()
    for chart in partition:
        chart_set = set(chart)
        if seen & chart_set:
            return False
        seen.update(chart_set)
    return True


def materialize_operator_tree(
    *,
    panel_id: str,
    face_count: int,
    operators: Sequence[FacePartitionOperator],
) -> OperatorTreeMaterializationReceipt:
    """Apply supports in order, refining the actual chart partition at each depth."""

    if face_count <= 0:
        raise ValueError("face_count must be positive")
    if not operators:
        raise ValueError("operator tree must contain at least one operator")

    universe = set(range(face_count))
    partition: tuple[tuple[int, ...], ...] = (tuple(range(face_count)),)
    steps: list[OperatorTreeMaterializationStep] = []
    complete = True

    for depth, operator in enumerate(operators, start=1):
        support = set(operator.support_faces)
        if not support <= universe:
            raise ValueError(
                f"operator '{operator.variant_id}' references faces outside the panel universe"
            )
        next_partition: list[tuple[int, ...]] = []
        effective_splits = 0
        for chart in partition:
            chart_set = set(chart)
            inside = tuple(sorted(chart_set & support))
            outside = tuple(sorted(chart_set - support))
            if inside and outside:
                next_partition.extend((inside, outside))
                effective_splits += 1
            else:
                next_partition.append(tuple(sorted(chart_set)))
        partition = tuple(
            sorted(
                (chart for chart in next_partition if chart),
                key=lambda chart: (chart[0], len(chart), chart),
            )
        )
        if effective_splits == 0:
            complete = False
        steps.append(
            OperatorTreeMaterializationStep(
                depth=depth,
                variant_id=operator.variant_id,
                support_faces=operator.support_faces,
                charts_before=steps[-1].charts_after if steps else 1,
                charts_after=len(partition),
                effective_splits=effective_splits,
                partition_after=partition,
            )
        )

    flattened = [face for chart in partition for face in chart]
    coverage_complete = set(flattened) == universe and len(flattened) == face_count
    pairwise_disjoint = _partition_is_disjoint(partition)
    return OperatorTreeMaterializationReceipt(
        panel_id=panel_id,
        face_count=face_count,
        operators=tuple(operators),
        steps=tuple(steps),
        final_partition=partition,
        face_coverage_complete=coverage_complete,
        charts_pairwise_disjoint=pairwise_disjoint,
        sequential_materialization_complete=complete and coverage_complete and pairwise_disjoint,
    )
