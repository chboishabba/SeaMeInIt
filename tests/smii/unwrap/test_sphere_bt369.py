from __future__ import annotations

import numpy as np

from smii.unwrap import unwrap_sphere_bt369


def test_unwrap_sphere_bt369_exports_equal_area_cells_and_certificate() -> None:
    result = unwrap_sphere_bt369(lambda xyz: xyz[1], width=6, height=4)

    assert result.image.shape == (4, 6)
    assert len(result.cells) == 24
    assert {round(cell.area_weight, 12) for cell in result.cells} == {round(4.0 * np.pi / 24.0, 12)}
    assert result.certificate["projection"] == "cylindrical_equal_area_inverse_pullback"
    assert result.certificate["surface_basis"] == "triadic_spherical_bt369"
    assert result.certificate["coverage"] == {"missing": 0, "multi_hit": 0}
    assert (
        result.certificate["claim_boundary"] == "benchmark_gated_approximation_not_global_isometry"
    )


def test_unwrap_sphere_bt369_records_bt369_cell_state() -> None:
    result = unwrap_sphere_bt369(
        lambda xyz: np.asarray([xyz[0], xyz[1], xyz[2]]),
        width=8,
        height=4,
        residual_tol=1e-8,
        max_depth=3,
        mdl_lambda=0.0,
    )

    cells = result.cells
    assert result.image.shape == (4, 8, 3)
    assert {cell.orientation_bin for cell in cells}.issubset(set(range(6)))
    assert any(cell.depth > 0 for cell in cells)
    assert any(cell.parent_prefix for cell in cells)
    assert any(cell.seam_crossing_token and "wrap_EW" in cell.seam_crossing_token for cell in cells)
    assert set(result.certificate["trit_histogram"]) == {"-1", "0", "1"}
    assert sum(result.certificate["trit_histogram"].values()) == len(cells)


def test_unwrap_sphere_bt369_validates_projection_and_dimensions() -> None:
    try:
        unwrap_sphere_bt369(lambda xyz: 1.0, width=0, height=4)
    except ValueError as exc:
        assert "width and height" in str(exc)
    else:
        raise AssertionError("expected invalid dimensions to fail")

    try:
        unwrap_sphere_bt369(lambda xyz: 1.0, width=4, height=4, projection="equirect")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "equal_area" in str(exc)
    else:
        raise AssertionError("expected unsupported projection to fail")
