from __future__ import annotations

from dataclasses import replace

from scripts.evaluate_bt369_pattern_partition import (
    _gate_receipt,
    _unique_partitions,
)
from smii.unwrap.bt369_pattern_serializer import (
    BT369PatternPolicy,
    build_bt369_pattern_partition,
)

import numpy as np
from types import SimpleNamespace


def _panel() -> tuple[np.ndarray, SimpleNamespace]:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    return vertices, SimpleNamespace(
        vertices=(0, 1, 2, 3),
        faces=((0, 1, 2), (0, 2, 3)),
    )


def test_gate_requires_score_distortion_foldover_exportability_and_validity() -> None:
    original = {
        "score": 100.0,
        "worst_distortion": 0.5,
        "foldovers": 10,
        "exportable": True,
    }
    promoted = _gate_receipt(
        original_metrics=original,
        measured_metrics={
            "score": 90.0,
            "worst_distortion": 0.4,
            "foldovers": 8,
            "exportable": True,
        },
        split_valid=True,
        chart_count=2,
    )
    assert promoted["promotion"] == 1
    assert promoted["blockers"] == []

    regression = _gate_receipt(
        original_metrics=original,
        measured_metrics={
            "score": 90.0,
            "worst_distortion": 0.6,
            "foldovers": 8,
            "exportable": True,
        },
        split_valid=True,
        chart_count=2,
    )
    assert regression["promotion"] == 0
    assert regression["blockers"] == ["distortion_regresses"]


def test_gate_does_not_promote_unsplit_identity_partition() -> None:
    gate = _gate_receipt(
        original_metrics={
            "score": 100.0,
            "worst_distortion": 0.5,
            "foldovers": 10,
            "exportable": True,
        },
        measured_metrics={
            "score": 90.0,
            "worst_distortion": 0.4,
            "foldovers": 8,
            "exportable": True,
        },
        split_valid=True,
        chart_count=1,
    )

    assert gate["promotion"] == 0
    assert "bt369_partition_did_not_split_parent" in gate["blockers"]


def test_policy_sweep_deduplicates_identical_topologies_but_keeps_aliases() -> None:
    vertices, panel = _panel()
    failure_field = {
        "face_distortions": [0.8, 0.01],
        "foldover_faces": [0],
        "high_distortion_faces": [0],
    }
    first_policy = BT369PatternPolicy(
        orientation_jump_limit=0,
        refinement_depth_jump_limit=0,
        shared_prefix_depth=0,
        isolate_foldovers=True,
        cut_residual_trit_boundaries=True,
    )
    second_policy = replace(first_policy, orientation_jump_limit=2)
    first = build_bt369_pattern_partition(
        vertices=vertices,
        panel=panel,
        failure_field=failure_field,
        distortion_threshold=0.05,
        policy=first_policy,
        panel_id="P0",
    )
    second = build_bt369_pattern_partition(
        vertices=vertices,
        panel=panel,
        failure_field=failure_field,
        distortion_threshold=0.05,
        policy=second_policy,
        panel_id="P0",
    )

    unique = _unique_partitions([first, second])

    assert len(unique) == 1
    assert unique[0]["receipt"].topology_hash == first.topology_hash
    assert set(unique[0]["policy_aliases"]) == {
        first_policy.policy_key,
        second_policy.policy_key,
    }
