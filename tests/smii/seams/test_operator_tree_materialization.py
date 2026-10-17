from __future__ import annotations

import pytest

from smii.seams.operator_tree_materialization import (
    FacePartitionOperator,
    materialize_operator_tree,
    operator_from_failure_field,
)


def test_sequential_materialization_refines_actual_partition() -> None:
    receipt = materialize_operator_tree(
        panel_id="P2",
        face_count=8,
        operators=[
            FacePartitionOperator(
                variant_id="failure_relief_path",
                support_faces=(0, 1, 2, 3),
                source_receipt="failure-field-sha256",
            ),
            FacePartitionOperator(
                variant_id="pareto_guided_dart_wedge",
                support_faces=(2, 3, 4),
                source_receipt="failure-field-sha256",
            ),
        ],
    )

    assert receipt.sequential_materialization_complete is True
    assert receipt.face_coverage_complete is True
    assert receipt.charts_pairwise_disjoint is True
    assert receipt.steps[0].partition_after == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert receipt.final_partition == ((0, 1), (2, 3), (4,), (5, 6, 7))
    assert receipt.promotion == 0
    assert receipt.blockers == ["backend_metrics_not_recomputed"]


def test_operator_that_does_not_refine_current_partition_stays_diagnostic() -> None:
    receipt = materialize_operator_tree(
        panel_id="P0",
        face_count=6,
        operators=[
            FacePartitionOperator(
                variant_id="failure_relief_path",
                support_faces=(0, 1, 2),
                source_receipt="failure-field-sha256",
            ),
            FacePartitionOperator(
                variant_id="failure_lens_patch",
                support_faces=(0, 1, 2),
                source_receipt="failure-field-sha256",
            ),
        ],
    )

    assert receipt.sequential_materialization_complete is False
    assert receipt.steps[1].effective_splits == 0
    assert "operator_tree_not_sequentially_materialized" in receipt.blockers
    assert "backend_metrics_not_recomputed" in receipt.blockers


def test_failure_field_supports_are_recovered_by_variant_family() -> None:
    failure_field = {
        "candidate_relief_paths": [
            {
                "failure_face_indices": [0, 1],
                "separates_bad_region": True,
            },
            {
                "failure_face_indices": [5],
                "separates_bad_region": False,
            },
        ],
        "selected_drain_paths": [
            {"drain_face_indices": [0, 1, 2]},
        ],
        "selected_guided_dart_wedges": [
            {"wedge_face_indices": [1, 2, 3]},
        ],
        "selected_lens_patches": [
            {"patch_face_indices": [2, 3, 4]},
        ],
    }

    relief = operator_from_failure_field(
        failure_field,
        variant_id="failure_relief_path",
        face_count=6,
    )
    drain = operator_from_failure_field(
        failure_field,
        variant_id="failure_drain_path",
        face_count=6,
    )
    dart = operator_from_failure_field(
        failure_field,
        variant_id="pareto_guided_dart_wedge",
        face_count=6,
    )
    lens = operator_from_failure_field(
        failure_field,
        variant_id="failure_lens_patch",
        face_count=6,
    )

    assert relief.support_faces == (0, 1)
    assert drain.support_faces == (0, 1, 2)
    assert dart.support_faces == (1, 2, 3)
    assert lens.support_faces == (2, 3, 4)


def test_materializer_rejects_unknown_or_out_of_range_supports() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        operator_from_failure_field({}, variant_id="cutout_r1", face_count=4)

    with pytest.raises(ValueError, match="outside"):
        operator_from_failure_field(
            {
                "selected_lens_patches": [
                    {"patch_face_indices": [0, 4]},
                ]
            },
            variant_id="failure_lens_patch",
            face_count=4,
        )
