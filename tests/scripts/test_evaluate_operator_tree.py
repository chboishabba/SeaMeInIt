from __future__ import annotations

from dataclasses import dataclass

import pytest

from scripts.evaluate_operator_tree import (
    _choose_tree,
    _common_refinement,
    _face_partition_from_variant,
)


@dataclass(frozen=True)
class _Panel:
    faces: tuple[tuple[int, int, int], ...]


def test_common_refinement_composes_actual_partitions() -> None:
    final, steps = _common_refinement(
        [
            ((0, 1, 2, 3), (4, 5, 6, 7)),
            ((0, 1, 4), (2, 3, 5, 6, 7)),
        ],
        face_count=8,
    )

    assert final == ((0, 1), (2, 3), (4,), (5, 6, 7))
    assert steps[0]["charts_after"] == 2
    assert steps[1]["charts_after"] == 4
    assert steps[1]["effective_refinement"] is True


def test_face_partition_recovers_original_face_identity() -> None:
    original = _Panel(
        faces=(
            (0, 1, 2),
            (2, 3, 0),
            (2, 4, 3),
            (4, 5, 3),
        )
    )
    children = [
        _Panel(faces=((0, 1, 2), (2, 3, 0))),
        _Panel(faces=((2, 4, 3), (4, 5, 3))),
    ]

    partition = _face_partition_from_variant(
        original_panel=original,
        child_panels=children,
        variant_id="failure_relief_path",
    )

    assert partition == ((0, 1), (2, 3))


def test_face_partition_rejects_overlap_or_missing_faces() -> None:
    original = _Panel(faces=((0, 1, 2), (2, 3, 0)))

    with pytest.raises(ValueError, match="overlaps"):
        _face_partition_from_variant(
            original_panel=original,
            child_panels=[
                _Panel(faces=((0, 1, 2),)),
                _Panel(faces=((0, 1, 2), (2, 3, 0))),
            ],
            variant_id="bad",
        )

    with pytest.raises(ValueError, match="missing faces"):
        _face_partition_from_variant(
            original_panel=original,
            child_panels=[_Panel(faces=((0, 1, 2),))],
            variant_id="bad",
        )


def test_choose_tree_prefers_best_retained_depth_n_profile() -> None:
    receipt = {
        "candidates": [
            {
                "tree_id": "P2_T001",
                "operator_count": 1,
                "profile_scores": {"default": 1.0},
                "metrics": {"score": 1.0},
            },
            {
                "tree_id": "P2_T002",
                "operator_count": 2,
                "profile_scores": {"default": 7.0},
                "metrics": {"score": 6.0},
            },
            {
                "tree_id": "P2_T003",
                "operator_count": 2,
                "profile_scores": {"default": 5.0},
                "metrics": {"score": 8.0},
            },
        ]
    }

    assert _choose_tree(receipt, None)["tree_id"] == "P2_T003"
    assert _choose_tree(receipt, "P2_T002")["tree_id"] == "P2_T002"


def test_choose_tree_does_not_fabricate_missing_depth_n_candidate() -> None:
    with pytest.raises(ValueError, match="depth>1"):
        _choose_tree(
            {
                "candidates": [
                    {
                        "tree_id": "P0_T000",
                        "operator_count": 1,
                        "profile_scores": {"default": 1.0},
                        "metrics": {"score": 1.0},
                    }
                ]
            },
            None,
        )
