"""Registration notch placement along seam boundaries for garment assembly."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from smii.garment.pattern_model import PatternChart, SeamConnection, SeamSide


def generate_seam_notches(
    seam: SeamConnection,
    chart: PatternChart,
    notch_fractions: Sequence[float] = (0.25, 0.5, 0.75),
    notch_depth_mm: float = 5.0,
) -> list[tuple[float, float, float, float]]:
    """Generate registration notch tick lines (x1, y1, x2, y2) along the seam boundary."""
    side: SeamSide = seam.left if seam.left.chart_id == chart.chart_id else seam.right
    if side.chart_id != chart.chart_id:
        return []

    # Map source vertex ID to local 2D UV index
    vmap = {int(v): i for i, v in enumerate(chart.source_vertex_ids)}
    edges = side.source_edge_ids
    if len(edges) < 2:
        return []

    # Collect valid 2D seam points
    seam_pts: list[np.ndarray] = []
    for i in range(0, len(edges) - 1, 2):
        u, v = edges[i], edges[i + 1]
        if u in vmap and v in vmap:
            p1 = chart.uv_mm[vmap[u]]
            p2 = chart.uv_mm[vmap[v]]
            if not seam_pts:
                seam_pts.append(p1)
            seam_pts.append(p2)

    if len(seam_pts) < 2:
        return []

    pts = np.asarray(seam_pts, dtype=np.float64)
    # Compute cumulative arc-length
    diffs = pts[1:] - pts[:-1]
    seg_lens = np.linalg.norm(diffs, axis=1)
    total_len = float(np.sum(seg_lens))
    if total_len <= 1e-6:
        return []

    cum_lens = np.insert(np.cumsum(seg_lens), 0, 0.0)
    notches: list[tuple[float, float, float, float]] = []

    for frac in notch_fractions:
        target_s = frac * total_len
        idx = int(np.searchsorted(cum_lens, target_s, side="right") - 1)
        idx = max(0, min(idx, len(seg_lens) - 1))

        # Interpolate point on segment
        seg_len = max(seg_lens[idx], 1e-6)
        t = (target_s - cum_lens[idx]) / seg_len
        p = pts[idx] + t * diffs[idx]

        # Normal vector perpendicular to tangent
        tangent = diffs[idx] / seg_len
        normal = np.array([-tangent[1], tangent[0]])

        # Tick mark
        p_out = p + normal * notch_depth_mm
        notches.append((float(p[0]), float(p[1]), float(p_out[0]), float(p_out[1])))

    return notches


__all__ = ["generate_seam_notches"]
