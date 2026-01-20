"""Tests for boundary splitting helper."""

from __future__ import annotations

from suit.panel_boundary_regularization import PanelIssue, split_boundary
from suit.seam_metadata import normalize_seam_metadata


def test_split_boundary_noop_without_suggestion() -> None:
    boundary = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]

    splits = split_boundary(boundary, [])

    assert splits == [boundary]


def test_split_boundary_returns_two_closed_loops() -> None:
    boundary = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    issues = [PanelIssue.from_code("SUGGEST_SPLIT", index=0)]

    splits = split_boundary(boundary, issues)

    assert len(splits) == 2
    for loop in splits:
        assert loop[0] == loop[-1]
        assert len(loop) >= 4


def test_split_boundary_avoids_forbidden_ranges() -> None:
    boundary = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 0.0),
        (0.0, 0.0),
    ]
    issues = [PanelIssue.from_code("SUGGEST_SPLIT", index=1)]
    avoid_ranges = [(1, 1)]

    splits = split_boundary(boundary, issues, avoid_ranges=avoid_ranges)

    original = boundary[:-1]
    cut_indices = []
    for loop in splits:
        if not loop:
            continue
        point = loop[0]
        if point in original:
            cut_indices.append(original.index(point))
    assert all(idx != 1 for idx in cut_indices)


def test_split_boundary_avoids_seam_partner_edges() -> None:
    boundary = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 0.0),
        (0.0, 0.0),
    ]
    issues = [PanelIssue.from_code("SUGGEST_SPLIT", index=1)]
    seams = {
        "panel_a": {
            "seam_partners": [
                {
                    "edge": (1, 2),
                    "partner_panel": "panel_b",
                    "partner_edge": (4, 5),
                    "role": "primary",
                }
            ]
        }
    }

    normalized = normalize_seam_metadata(seams)
    assert normalized
    avoid_ranges = normalized["panel_a"]["seam_avoid_ranges"]
    assert avoid_ranges == [(1, 2)]

    splits = split_boundary(boundary, issues, avoid_ranges=avoid_ranges)

    original = boundary[:-1]
    cut_indices = []
    for loop in splits:
        if not loop:
            continue
        point = loop[0]
        if point in original:
            cut_indices.append(original.index(point))
    assert all(idx not in {1, 2} for idx in cut_indices)
