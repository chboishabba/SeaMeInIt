from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.export_bt369_p2_manufacturing import (
    ALLOWANCE_BY_METHOD,
    _generate_p2_svg,
    export_p2_manufacturing,
)
from scripts.unwrap_panels import PanelPatch


def test_svg_generation_contains_notches_and_units() -> None:
    patch = PanelPatch(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=((0, 1, 2),),
        edges=((0, 1), (1, 2), (0, 2)),
    )
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    svg = _generate_p2_svg(
        charts_uv=[uv],
        chart_patches=[patch],
        foldovers_by_chart=[0],
        allowance_meters=0.015,
        manufacturing_method="home_sewing",
    )

    assert "<svg" in svg
    assert 'data-units="meters"' in svg
    assert 'data-notch="true"' in svg
    assert 'data-grain="warp"' in svg
    assert "<text" in svg
    assert "P2.chart.00" in svg


def test_svg_generation_highlights_foldovers() -> None:
    patch = PanelPatch(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=((0, 1, 2),),
        edges=((0, 1), (1, 2), (0, 2)),
    )
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    svg = _generate_p2_svg(
        charts_uv=[uv],
        chart_patches=[patch],
        foldovers_by_chart=[5],
        allowance_meters=0.010,
        manufacturing_method="overlock",
    )

    assert 'data-foldover-warning="true"' in svg
    assert "5 foldovers" in svg


def test_export_rejects_unknown_method(tmp_path: Path) -> None:
    dummy_frozen = tmp_path / "frozen.json"
    dummy_frozen.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown manufacturing method"):
        export_p2_manufacturing(
            frozen_winner_path=dummy_frozen,
            mesh_path=tmp_path / "mesh.npz",
            seam_edges_path=tmp_path / "seam.npz",
            out_dir=tmp_path / "out",
            manufacturing_method="magic_glue",
        )


def test_export_p2_manufacturing_integration(tmp_path: Path) -> None:
    frozen_path = Path(
        "outputs/demo/afflec_receipted_curated_20260605_143059/panels_fabric_operator_basis_search/frozen_p2_winner.json"
    )
    mesh_path = Path("outputs/demo/afflec_receipted_curated_20260605_143059/body/afflec_body.npz")
    seam_edges_path = Path(
        "outputs/demo/afflec_receipted_curated_20260605_143059/solver/seam_edges.npz"
    )

    if not frozen_path.exists() or not mesh_path.exists() or not seam_edges_path.exists():
        pytest.skip("Afflec test artifacts not present in outputs/")

    out_dir = tmp_path / "mfg_out"
    receipt = export_p2_manufacturing(
        frozen_winner_path=frozen_path,
        mesh_path=mesh_path,
        seam_edges_path=seam_edges_path,
        out_dir=out_dir,
        manufacturing_method="home_sewing",
    )

    assert receipt["schema_version"] == "smii.bt369_p2_manufacturing.v1"
    assert receipt["promotion"] == 1
    assert receipt["chart_count"] == 23
    assert receipt["manufacturing_method"] == "home_sewing"
    assert receipt["allowance_width_meters"] == 0.015
    assert (out_dir / "p2_cutting_layout.svg").exists()
    assert (out_dir / "p2_seam_allowance.npz").exists()
    assert (out_dir / "p2_chart_uvs.npz").exists()
    assert (out_dir / "p2_manufacturing_receipt.json").exists()
