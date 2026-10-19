"""Tests for seam allowance polygon offsetting and geometric validation."""

from __future__ import annotations

import numpy as np
import pytest

from smii.manufacturing.allowances import (
    ALLOWANCE_BY_METHOD,
    compute_seam_allowance_offset,
)


def test_square_allowance_offset() -> None:
    # 100mm x 100mm square
    square = np.array(
        [
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 100.0],
            [0.0, 100.0],
        ],
        dtype=np.float64,
    )
    allowance = 10.0  # 10 mm
    offset = compute_seam_allowance_offset(square, allowance_mm=allowance)

    assert len(offset) >= 4
    # Bounding box should expand by 10mm in every direction: [-10, 110]
    assert offset[:, 0].min() == pytest.approx(-10.0, abs=0.5)
    assert offset[:, 0].max() == pytest.approx(110.0, abs=0.5)
    assert offset[:, 1].min() == pytest.approx(-10.0, abs=0.5)
    assert offset[:, 1].max() == pytest.approx(110.0, abs=0.5)


def test_zero_allowance_returns_boundary() -> None:
    pts = np.array([[0.0, 0.0], [50.0, 0.0], [25.0, 50.0]], dtype=np.float64)
    res = compute_seam_allowance_offset(pts, allowance_mm=0.0)
    assert len(res) == 3


def test_allowance_by_method_lookup() -> None:
    assert ALLOWANCE_BY_METHOD["home_sewing"] == 15.0
    assert ALLOWANCE_BY_METHOD["overlock"] == 10.0
    assert ALLOWANCE_BY_METHOD["flatlock"] == 8.0
