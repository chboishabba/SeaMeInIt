#!/usr/bin/env python3
"""Render mesh-only (and optional correspondence-line) orbit artifacts for a vertex map.

This is a debugging tool: it visualizes whether a nearest-neighbour correspondence
between two mesh topologies is geometrically plausible.

Inputs:
- source mesh (vertices/faces)
- target mesh (vertices/faces)
- correspondence NPZ containing `target_to_source_indices` + `target_to_source_distances`
  (or `source_to_target_*` if you choose the opposite direction)

Outputs (in --out-dir):
- `map_orbit.webm` + timestamped copy
- `map_front_3q.png` + timestamped copy
- `map_manifest.json` with provenance + stats
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
AXIS_NAME = {0: "x", 1: "y", 2: "z"}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rotate_xyz(vertices: np.ndarray, rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Rigid XYZ rotation helper (degrees)."""
    rx, ry, rz = map(math.radians, (rx_deg, ry_deg, rz_deg))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return vertices @ (rot_z @ rot_y @ rot_x).T


def _magma_like(value: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, float(value)))
    if t < 0.25:
        u = t / 0.25
        return (int(30 + 70 * u), int(0 + 10 * u), int(50 + 70 * u))
    if t < 0.5:
        u = (t - 0.25) / 0.25
        return (int(100 + 80 * u), int(10 + 40 * u), int(120 + 40 * u))
    if t < 0.75:
        u = (t - 0.5) / 0.25
        return (int(180 + 50 * u), int(50 + 80 * u), int(160 - 30 * u))
    u = (t - 0.75) / 0.25
    return (int(230 + 20 * u), int(130 + 90 * u), int(130 - 70 * u))


def _canonicalize_vertices(
    vertices: np.ndarray, *, enabled: bool, up_axis: str, width_axis: str | None
) -> tuple[np.ndarray, dict[str, object]]:
    original = np.asarray(vertices, dtype=float)
    spans = np.ptp(original, axis=0)
    meta: dict[str, object] = {
        "enabled": bool(enabled),
        "original_axis_spans": [float(v) for v in spans],
    }
    if not enabled:
        centered = original - np.mean(original, axis=0, keepdims=True)
        meta.update(
            {
                "axis_up": None,
                "axis_width": None,
                "axis_depth": None,
                "axis_order": ["x", "y", "z"],
            }
        )
        return centered, meta

    up_axis = str(up_axis).lower()
    if up_axis not in AXIS_INDEX:
        raise ValueError(f"Unknown up axis '{up_axis}'. Use one of: x,y,z.")
    up = AXIS_INDEX[up_axis]
    remaining = [idx for idx in range(3) if idx != up]

    width: int
    if width_axis is not None:
        width_axis = str(width_axis).lower()
        if width_axis not in AXIS_INDEX:
            raise ValueError(f"Unknown width axis '{width_axis}'. Use one of: x,y,z.")
        width = AXIS_INDEX[width_axis]
        if width == up or width not in remaining:
            raise ValueError("width axis must differ from up axis.")
    else:
        width = max(remaining, key=lambda idx: float(spans[idx]))

    depth = next(idx for idx in remaining if idx != width)
    canonical = np.stack([original[:, width], original[:, depth], original[:, up]], axis=1).astype(
        float
    )
    canonical -= np.mean(canonical, axis=0, keepdims=True)
    meta.update(
        {
            "axis_up": AXIS_NAME[up],
            "axis_width": AXIS_NAME[width],
            "axis_depth": AXIS_NAME[depth],
            "axis_order": [AXIS_NAME[width], AXIS_NAME[depth], AXIS_NAME[up]],
        }
    )
    return canonical, meta


def _project_vertices(
    vertices: np.ndarray, yaw_deg: float, elev_deg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = math.radians(yaw_deg)
    elev = math.radians(elev_deg)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    cos_elev, sin_elev = math.cos(elev), math.sin(elev)
    rot_z = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_elev, -sin_elev],
            [0.0, sin_elev, cos_elev],
        ],
        dtype=float,
    )
    projected = (vertices @ rot_z.T) @ rot_x.T
    return projected[:, 0], projected[:, 2], projected[:, 1]


def _to_screen(u: np.ndarray, v: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    span = max(float(u.max() - u.min()), float(v.max() - v.min()), 1e-6)
    scale = 0.84 * min(width, height) / span
    sx = (u - (u.min() + u.max()) * 0.5) * scale + (width / 2)
    sy = (v - (v.min() + v.max()) * 0.5) * scale + (height / 2)
    return sx, sy


def _encode_orbit(frames_dir: Path, webm_path: Path) -> Path:
    webm_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "24",
            "-i",
            str(frames_dir / "frame_%03d.png"),
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            "0",
            "-crf",
            "34",
            str(webm_path),
        ],
        check=True,
    )
    return webm_path


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    payload = np.load(path)
    vertices = np.asarray(payload["vertices"], dtype=float)
    faces = np.asarray(payload["faces"], dtype=int) if "faces" in payload else None
    return vertices, faces


def _load_correspondence(
    map_path: Path,
    *,
    direction: Literal["target_to_source", "source_to_target"],
) -> tuple[np.ndarray, np.ndarray, dict[str, object] | None]:
    payload = np.load(map_path, allow_pickle=True)
    if direction == "target_to_source":
        indices = np.asarray(payload["target_to_source_indices"], dtype=np.int64)
        distances = np.asarray(payload["target_to_source_distances"], dtype=np.float64)
    else:
        indices = np.asarray(payload["source_to_target_indices"], dtype=np.int64)
        distances = np.asarray(payload["source_to_target_distances"], dtype=np.float64)
    meta = None
    if "meta" in payload:
        raw = payload["meta"]
        if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
            try:
                meta = dict(raw.item())
            except Exception:
                meta = None
    return indices, distances, meta


def _distance_colors(distances: np.ndarray, *, clip: float) -> tuple[list[tuple[int, int, int]], dict[str, float]]:
    arr = np.asarray(distances, dtype=float)
    clip = float(max(clip, 1e-9))
    norm = np.clip(arr / clip, 0.0, 1.0)
    colors = [_magma_like(val) for val in norm]
    stats = {
        "min": float(np.min(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "std": float(np.std(arr)) if arr.size else 0.0,
        "clip": float(clip),
    }
    return colors, stats


def _render_frame(
    *,
    target_vertices: np.ndarray,
    target_colors: list[tuple[int, int, int]],
    source_vertices: np.ndarray | None,
    pairs: list[tuple[int, int]] | None,
    yaw_deg: float,
    elev_deg: float,
    width: int,
    height: int,
    point_size: int,
    target_alpha: int,
    source_alpha: int,
    line_alpha: int,
    line_width: int,
    output: Path,
) -> None:
    u, v, depth = _project_vertices(target_vertices, yaw_deg=yaw_deg, elev_deg=elev_deg)
    sx, sy = _to_screen(u, v, width, height)

    # Ensure source is projected in the same camera frame (same yaw/elev).
    if source_vertices is not None:
        su, sv, sdepth = _project_vertices(source_vertices, yaw_deg=yaw_deg, elev_deg=elev_deg)
        ssx, ssy = _to_screen(su, sv, width, height)
    else:
        ssx = ssy = None

    image = Image.new("RGB", (width, height), (250, 248, 245))
    draw = ImageDraw.Draw(image, "RGBA")

    # Draw source first (behind) as neutral gray.
    if source_vertices is not None and ssx is not None and ssy is not None:
        order = np.argsort(sdepth)
        step = max(1, len(order) // 9000)
        radius = max(1, int(round(point_size / 2)))
        for idx in order[::step]:
            x = int(round(float(ssx[idx])))
            y = int(round(float(ssy[idx])))
            if not (0 <= x < width and 0 <= y < height):
                continue
            y_screen = height - 1 - y
            draw.ellipse(
                (x - radius, y_screen - radius, x + radius, y_screen + radius),
                fill=(80, 80, 80, int(max(0, min(255, source_alpha)))),
            )

    # Optionally draw correspondence lines.
    if pairs and source_vertices is not None and ssx is not None and ssy is not None:
        for t_idx, s_idx in pairs:
            if t_idx >= len(target_vertices) or s_idx >= len(source_vertices):
                continue
            x1 = int(round(float(sx[t_idx])))
            y1 = int(round(float(sy[t_idx])))
            x2 = int(round(float(ssx[s_idx])))
            y2 = int(round(float(ssy[s_idx])))
            draw.line(
                (x1, height - 1 - y1, x2, height - 1 - y2),
                fill=(30, 170, 190, int(max(0, min(255, line_alpha)))),
                width=max(1, int(line_width)),
            )

    # Draw target colored by distance.
    order = np.argsort(depth)
    step = max(1, len(order) // 9000)
    radius = max(1, int(round(point_size / 2)))
    for idx in order[::step]:
        x = int(round(float(sx[idx])))
        y = int(round(float(sy[idx])))
        if not (0 <= x < width and 0 <= y < height):
            continue
        r, g, b = target_colors[idx]
        y_screen = height - 1 - y
        draw.ellipse(
            (x - radius, y_screen - radius, x + radius, y_screen + radius),
            fill=(r, g, b, int(max(0, min(255, target_alpha)))),
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mesh", type=Path, required=True)
    parser.add_argument("--target-mesh", type=Path, required=True)
    parser.add_argument("--vertex-map", type=Path, required=True, help="Correspondence NPZ.")
    parser.add_argument(
        "--direction",
        type=str,
        default="target_to_source",
        choices=("target_to_source", "source_to_target"),
        help="Which mapping arrays in the NPZ to visualize.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--elev", type=float, default=14.0)
    parser.add_argument("--yaw-start", type=float, default=-120.0)
    parser.add_argument(
        "--yaw-offset",
        type=float,
        default=0.0,
        help="Additional yaw offset (degrees) applied to all frames (use to align 'front').",
    )
    parser.add_argument(
        "--source-rotate-x",
        type=float,
        default=0.0,
        help="Rigid X rotation applied to source mesh (degrees).",
    )
    parser.add_argument(
        "--source-rotate-y",
        type=float,
        default=0.0,
        help="Rigid Y rotation applied to source mesh (degrees).",
    )
    parser.add_argument(
        "--source-rotate-z",
        type=float,
        default=0.0,
        help="Rigid Z rotation applied to source mesh (degrees).",
    )
    parser.add_argument(
        "--target-rotate-x",
        type=float,
        default=0.0,
        help="Rigid X rotation applied to target mesh (degrees).",
    )
    parser.add_argument(
        "--target-rotate-y",
        type=float,
        default=0.0,
        help="Rigid Y rotation applied to target mesh (degrees).",
    )
    parser.add_argument(
        "--target-rotate-z",
        type=float,
        default=0.0,
        help="Rigid Z rotation applied to target mesh (degrees).",
    )
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--point-size", type=int, default=6)
    parser.add_argument("--target-alpha", type=int, default=180)
    parser.add_argument("--source-alpha", type=int, default=70)
    parser.add_argument("--line-alpha", type=int, default=120)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument(
        "--show-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render the source mesh behind the target mesh.",
    )
    parser.add_argument(
        "--show-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render correspondence lines (subsampled) between target and source vertices.",
    )
    parser.add_argument(
        "--line-count",
        type=int,
        default=800,
        help="Number of correspondence lines to draw (uniform random sample).",
    )
    parser.add_argument(
        "--distance-clip",
        type=float,
        default=0.15,
        help="Distance value (meters) that maps to max colormap intensity.",
    )
    parser.add_argument(
        "--canonicalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply stable axis canonicalization before projection.",
    )
    parser.add_argument(
        "--axis-up",
        type=str,
        default="y",
        choices=("x", "y", "z"),
        help="Axis treated as up when --canonicalize is enabled.",
    )
    parser.add_argument(
        "--axis-width",
        type=str,
        default=None,
        choices=("x", "y", "z"),
        help="Optional width axis override when --canonicalize is enabled.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )
    args = parser.parse_args()

    src_vertices, _ = _load_mesh(args.source_mesh)
    tgt_vertices, _ = _load_mesh(args.target_mesh)
    indices, distances, map_meta = _load_correspondence(
        args.vertex_map, direction=args.direction  # type: ignore[arg-type]
    )

    # Canonicalize both in the same axis convention to keep visuals comparable.
    src_canon, axis_meta = _canonicalize_vertices(
        src_vertices,
        enabled=bool(args.canonicalize),
        up_axis=str(args.axis_up),
        width_axis=args.axis_width,
    )
    tgt_canon, _ = _canonicalize_vertices(
        tgt_vertices,
        enabled=bool(args.canonicalize),
        up_axis=str(args.axis_up),
        width_axis=args.axis_width,
    )

    if any(abs(v) > 1e-9 for v in (args.source_rotate_x, args.source_rotate_y, args.source_rotate_z)):
        src_canon = _rotate_xyz(src_canon, args.source_rotate_x, args.source_rotate_y, args.source_rotate_z)
    if any(abs(v) > 1e-9 for v in (args.target_rotate_x, args.target_rotate_y, args.target_rotate_z)):
        tgt_canon = _rotate_xyz(tgt_canon, args.target_rotate_x, args.target_rotate_y, args.target_rotate_z)

    colors, dist_stats = _distance_colors(distances, clip=float(args.distance_clip))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sample correspondence lines in the direction being shown.
    pairs: list[tuple[int, int]] | None = None
    if args.show_lines:
        rng = random.Random(0)
        count = max(0, int(args.line_count))
        if count > 0 and len(indices) > 0:
            sample = rng.sample(range(len(indices)), k=min(count, len(indices)))
            pairs = [(int(i), int(indices[int(i)])) for i in sample]

    frames_dir = out_dir / "map_orbit_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(int(args.frames)):
        yaw = float(args.yaw_start) + float(args.yaw_offset) + (360.0 * idx / max(int(args.frames), 1))
        frame_path = frames_dir / f"frame_{idx:03d}.png"
        _render_frame(
            target_vertices=tgt_canon,
            target_colors=colors,
            source_vertices=src_canon if args.show_source else None,
            pairs=pairs if args.show_source else None,
            yaw_deg=yaw,
            elev_deg=float(args.elev),
            width=int(args.width),
            height=int(args.height),
            point_size=int(args.point_size),
            target_alpha=int(args.target_alpha),
            source_alpha=int(args.source_alpha),
            line_alpha=int(args.line_alpha),
            line_width=int(args.line_width),
            output=frame_path,
        )

    front_png = out_dir / "map_front_3q.png"
    shutil.copy2(frames_dir / "frame_000.png", front_png)
    front_png_ts = out_dir / f"map_front_3q_{args.timestamp}.png"
    shutil.copy2(front_png, front_png_ts)

    webm = out_dir / "map_orbit.webm"
    webm_ts = out_dir / f"map_orbit_{args.timestamp}.webm"
    _encode_orbit(frames_dir, webm)
    shutil.copy2(webm, webm_ts)
    shutil.rmtree(frames_dir)

    manifest = {
        "timestamp": str(args.timestamp),
        "source_mesh": {
            "path": str(args.source_mesh),
            "sha256": _sha256_path(args.source_mesh),
            "vertex_count": int(src_vertices.shape[0]),
        },
        "target_mesh": {
            "path": str(args.target_mesh),
            "sha256": _sha256_path(args.target_mesh),
            "vertex_count": int(tgt_vertices.shape[0]),
        },
        "vertex_map": {
            "path": str(args.vertex_map),
            "sha256": _sha256_path(args.vertex_map),
            "direction": str(args.direction),
            "meta": map_meta,
            "distance_stats": dist_stats,
        },
        "render_axis": dict(axis_meta),
        "render_params": {
            "canonicalize": bool(args.canonicalize),
            "axis_up": str(args.axis_up),
            "axis_width": str(args.axis_width) if args.axis_width is not None else None,
            "width": int(args.width),
            "height": int(args.height),
            "elev": float(args.elev),
            "yaw_start": float(args.yaw_start),
            "yaw_offset": float(args.yaw_offset),
            "frames": int(args.frames),
            "point_size": int(args.point_size),
            "target_alpha": int(args.target_alpha),
            "source_alpha": int(args.source_alpha),
            "line_alpha": int(args.line_alpha),
            "line_width": int(args.line_width),
            "show_source": bool(args.show_source),
            "show_lines": bool(args.show_lines),
            "line_count": int(args.line_count),
            "distance_clip": float(args.distance_clip),
            "source_rotate_deg": {
                "x": float(args.source_rotate_x),
                "y": float(args.source_rotate_y),
                "z": float(args.source_rotate_z),
            },
            "target_rotate_deg": {
                "x": float(args.target_rotate_x),
                "y": float(args.target_rotate_y),
                "z": float(args.target_rotate_z),
            },
        },
        "outputs": {
            "front_png": str(front_png),
            "front_png_timestamped": str(front_png_ts),
            "orbit_webm": str(webm),
            "orbit_webm_timestamped": str(webm_ts),
        },
    }
    (out_dir / "map_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
