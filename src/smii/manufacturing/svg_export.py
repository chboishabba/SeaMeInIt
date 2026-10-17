"""Production vector SVG pattern cutting layout generator."""

from __future__ import annotations

from smii.garment.pattern_model import FullSuitPattern, PatternChart
from .grainlines import compute_grainline_vector
from .nesting import NestingResult


def export_full_suit_svg(
    pattern: FullSuitPattern,
    nesting: NestingResult,
    title: str = "SMII Full-Suit Manufacturing Pattern Layout",
) -> str:
    """Render a nested full-suit vector SVG cutting pattern with allowances, grainlines, and scale bar."""
    total_w = nesting.fabric_width_mm
    total_h = nesting.fabric_length_mm

    chart_map: dict[str, PatternChart] = {c.chart_id: c for c in pattern.charts}

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.1f}mm" '
            f'height="{total_h:.1f}mm" viewBox="0 0 {total_w:.1f} {total_h:.1f}" '
            f'data-units="mm" data-fabric-width="{total_w:.1f}" data-fabric-length="{total_h:.1f}">'
        ),
        f"  <title>{title}</title>",
        "  <defs>",
        '    <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '      <path d="M 0 0 L 10 5 L 0 10 z" fill="#27ae60" />',
        "    </marker>",
        "  </defs>",
        "",
        "  <!-- 100mm Physical Calibration Scale Bar -->",
        '  <g id="calibration_scale" transform="translate(20, 20)">',
        '    <rect x="0" y="0" width="100" height="8" fill="#333" />',
        '    <rect x="0" y="0" width="50" height="8" fill="#fff" />',
        '    <rect x="0" y="0" width="10" height="8" fill="#333" />',
        '    <rect x="20" y="0" width="10" height="8" fill="#333" />',
        '    <rect x="40" y="0" width="10" height="8" fill="#333" />',
        '    <text x="110" y="7" font-family="sans-serif" font-size="8" fill="#333" font-weight="bold">100 mm</text>',
        "  </g>",
        "",
    ]

    # Render each chart piece
    for placement in nesting.placements:
        cid = placement.chart_id
        chart = chart_map.get(cid)
        if not chart:
            continue

        pts_cut = placement.offset_contour_mm
        poly_str = " ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in pts_cut)

        # Grainline in local placed space
        gl = compute_grainline_vector(chart)
        # Shift grainline to placement origin
        min_uv_x = float(chart.uv_mm[:, 0].min())
        min_uv_y = float(chart.uv_mm[:, 1].min())
        gl_x1 = gl[0] - min_uv_x + placement.x_mm
        gl_y1 = gl[1] - min_uv_y + placement.y_mm
        gl_x2 = gl[2] - min_uv_x + placement.x_mm
        gl_y2 = gl[3] - min_uv_y + placement.y_mm

        lines.extend(
            [
                f"  <!-- Pattern Piece: {cid} -->",
                f'  <g id="chart_{cid}" data-region="{chart.region_id}">',
                "    <!-- Outward Cut Line with Seam Allowance -->",
                (
                    f'    <polygon points="{poly_str}" fill="rgba(52, 152, 219, 0.08)" '
                    'stroke="#2c3e50" stroke-width="1.5" />'
                ),
                "    <!-- Grainline Vector -->",
                (
                    f'    <line x1="{gl_x1:.2f}" y1="{gl_y1:.2f}" x2="{gl_x2:.2f}" y2="{gl_y2:.2f}" '
                    'stroke="#27ae60" stroke-width="1.5" marker-end="url(#arrow)" data-grainline="true" />'
                ),
                "    <!-- Label -->",
                (
                    f'    <text x="{placement.x_mm + 15:.1f}" y="{placement.y_mm + 25:.1f}" '
                    f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#2c3e50">{cid}</text>'
                ),
                (
                    f'    <text x="{placement.x_mm + 15:.1f}" y="{placement.y_mm + 40:.1f}" '
                    f'font-family="sans-serif" font-size="9" fill="#7f8c8d">{len(chart.source_face_ids)} faces</text>'
                ),
                "  </g>",
                "",
            ]
        )

    lines.append("</svg>")
    return "\n".join(lines)


__all__ = ["export_full_suit_svg"]
