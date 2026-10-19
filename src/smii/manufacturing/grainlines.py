"""Material grainline orientation vectors for garment pattern layout."""

from __future__ import annotations

import numpy as np

from smii.garment.pattern_model import PatternChart


def compute_grainline_vector(
    chart: PatternChart,
    arrow_length_mm: float = 80.0,
) -> tuple[float, float, float, float]:
    """Compute the start and end coordinates (x1, y1, x2, y2) of the fabric grainline arrow."""
    uv = chart.uv_mm
    if len(uv) == 0:
        return (0.0, 0.0, 0.0, 0.0)

    # Place near chart centroid
    centroid_x = float(np.mean(uv[:, 0]))
    centroid_y = float(np.mean(uv[:, 1]))

    # Longitudinal warp direction is aligned with vertical Y axis
    half_len = arrow_length_mm * 0.5
    x1 = centroid_x
    y1 = centroid_y - half_len
    x2 = centroid_x
    y2 = centroid_y + half_len

    return (x1, y1, x2, y2)


__all__ = ["compute_grainline_vector"]
