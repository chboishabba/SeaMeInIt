from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smii.pipelines.authorized_image_fit import regress_smplx_from_images


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "afflec"
AFFLEC_PHOTOS = (
    FIXTURE_DIR / "afflec1.jpg",
    FIXTURE_DIR / "afflec2.jpg",
    FIXTURE_DIR / "afflec3.avif",
)


def test_bundled_afflec_smoke_abstains_and_keeps_image_anchor() -> None:
    missing = [path for path in AFFLEC_PHOTOS if not path.exists()]
    if missing:
        pytest.skip(f"Afflec fixture images missing: {missing}")

    result = regress_smplx_from_images(
        AFFLEC_PHOTOS,
        detector="bbox",
        fit_mode="heuristic",
        refine_with_measurements=True,
    )

    assert result.measurement_fit is not None
    receipt = result.measurement_fit.refinement_receipt
    assert receipt.decision == "abstain"
    assert "reference_quality_insufficient_for_refinement" in receipt.blockers
    assert "WARN:low_view_diversity" in receipt.warnings
    assert "WARN:long_lens_flattening_risk" in receipt.warnings
    np.testing.assert_allclose(result.measurement_fit.betas, result.betas)
    assert receipt.selected_output_hash == receipt.input_hash
    assert receipt.selected_output_hash != receipt.candidate_hash
