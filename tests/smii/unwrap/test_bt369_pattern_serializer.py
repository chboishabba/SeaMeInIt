from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from smii.unwrap.bt369_pattern_serializer import (
    BT369PatternPolicy,
    build_bt369_pattern_partition,
    default_bt369_pattern_policy_grid,
)


def _two_triangle_panel() -> tuple[np.ndarray, SimpleNamespace]:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    panel = SimpleNamespace(
        vertices=(0, 1, 2, 3),
        faces=((0, 1, 2), (0, 2, 3)),
    )
    return vertices, panel


def test_bt369_partition_isolates_measured_foldover_and_preserves_faces() -> None:
    vertices, panel = _two_triangle_panel()
    failure_field = {
        "face_distortions": [0.8, 0.01],
        "foldover_faces": [0],
        "high_distortion_faces": [0],
    }

    receipt = build_bt369_pattern_partition(
        vertices=vertices,
        panel=panel,
        failure_field=failure_field,
        distortion_threshold=0.05,
        policy=BT369PatternPolicy(
            orientation_jump_limit=5,
            refinement_depth_jump_limit=8,
            shared_prefix_depth=0,
            isolate_foldovers=True,
            cut_residual_trit_boundaries=False,
        ),
        panel_id="P0",
    )

    assert receipt.face_count == 2
    assert receipt.face_coverage_complete is True
    assert receipt.faces_pairwise_disjoint is True
    assert receipt.partition == ((0,), (1,))
    assert [cell.residual_trit for cell in receipt.cells] == [-1, 0]
    assert receipt.cut_reason_histogram["foldover_boundary"] == 1
    assert (
        receipt.claim_boundary
        == "bt369_partition_is_candidate_panelization_not_manufacturing_authority"
    )


def test_bt369_partition_keeps_compatible_low_residual_faces_together() -> None:
    vertices, panel = _two_triangle_panel()
    failure_field = {
        "face_distortions": [0.01, 0.02],
        "foldover_faces": [],
        "high_distortion_faces": [],
    }

    receipt = build_bt369_pattern_partition(
        vertices=vertices,
        panel=panel,
        failure_field=failure_field,
        distortion_threshold=0.05,
        policy=BT369PatternPolicy(
            orientation_jump_limit=5,
            refinement_depth_jump_limit=8,
            shared_prefix_depth=0,
            isolate_foldovers=True,
            cut_residual_trit_boundaries=True,
        ),
        panel_id="P0",
    )

    assert receipt.partition == ((0, 1),)
    assert receipt.cut_edges == ()
    assert {cell.residual_trit for cell in receipt.cells} == {0}


def test_bt369_partition_is_deterministic_and_certificate_bounded() -> None:
    vertices, panel = _two_triangle_panel()
    failure_field = {
        "face_distortions": [0.4, 0.2],
        "foldover_faces": [],
        "high_distortion_faces": [0, 1],
    }
    policy = BT369PatternPolicy(
        orientation_jump_limit=1,
        refinement_depth_jump_limit=1,
        shared_prefix_depth=1,
        isolate_foldovers=True,
        cut_residual_trit_boundaries=True,
        max_depth=4,
    )

    first = build_bt369_pattern_partition(
        vertices=vertices,
        panel=panel,
        failure_field=failure_field,
        distortion_threshold=0.05,
        policy=policy,
        panel_id="P0",
    )
    second = build_bt369_pattern_partition(
        vertices=vertices,
        panel=panel,
        failure_field=failure_field,
        distortion_threshold=0.05,
        policy=policy,
        panel_id="P0",
    )

    assert first.topology_hash == second.topology_hash
    assert first.to_dict() == second.to_dict()
    assert all(0 <= cell.orientation_bin < 6 for cell in first.cells)
    assert all(0 <= cell.depth <= 4 for cell in first.cells)
    assert all(len(cell.triadic_prefix) == cell.depth for cell in first.cells)
    assert abs(sum(cell.area_weight for cell in first.cells) - 1.0) < 1e-12


def test_default_bt369_policy_grid_is_bounded_and_unique() -> None:
    policies = default_bt369_pattern_policy_grid()

    assert len(policies) == 18
    keys = {policy.policy_key for policy in policies}
    assert len(keys) == len(policies)
    assert {policy.orientation_jump_limit for policy in policies} == {0, 1, 2}
    assert {policy.refinement_depth_jump_limit for policy in policies} == {0, 1, 2}
    assert {policy.shared_prefix_depth for policy in policies} == {0, 1}
    assert all(policy.isolate_foldovers for policy in policies)
    assert all(policy.cut_residual_trit_boundaries for policy in policies)
