#!/usr/bin/env python3
"""Render a single mesh + optional seam report into orbit/still artifacts.

This is the "single-run" sibling of scripts/render_variant_orbits.py.
It writes into an explicit --out-dir so we can generate timestamped bundles
without mutating/overwriting the original run directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_AXIS_NAME = {0: "x", 1: "y", 2: "z"}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    vertices: np.ndarray,
    *,
    enabled: bool,
    up_axis: str,
    width_axis: str | None,
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
    if up_axis not in _AXIS_INDEX:
        raise ValueError(f"Unknown up axis '{up_axis}'. Use one of: x,y,z.")
    up = _AXIS_INDEX[up_axis]
    remaining = [idx for idx in range(3) if idx != up]

    if width_axis is not None:
        width_axis = str(width_axis).lower()
        if width_axis not in _AXIS_INDEX:
            raise ValueError(f"Unknown width axis '{width_axis}'. Use one of: x,y,z.")
        width = _AXIS_INDEX[width_axis]
        if width == up or width not in remaining:
            raise ValueError("width axis must differ from up axis and be one of the non-up axes.")
    else:
        width = max(remaining, key=lambda idx: float(spans[idx]))

    depth = next(idx for idx in remaining if idx != width)
    canonical = np.stack([original[:, width], original[:, depth], original[:, up]], axis=1).astype(
        float
    )
    canonical -= np.mean(canonical, axis=0, keepdims=True)
    meta.update(
        {
            "axis_up": _AXIS_NAME[up],
            "axis_width": _AXIS_NAME[width],
            "axis_depth": _AXIS_NAME[depth],
            "axis_order": [_AXIS_NAME[width], _AXIS_NAME[depth], _AXIS_NAME[up]],
        }
    )
    return canonical, meta


def _rotate_xyz(vertices: np.ndarray, rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = map(math.radians, (rx_deg, ry_deg, rz_deg))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return vertices @ (rot_z @ rot_y @ rot_x).T


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


def _to_screen(
    u: np.ndarray, v: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    span = max(float(u.max() - u.min()), float(v.max() - v.min()), 1e-6)
    scale = 0.84 * min(width, height) / span
    sx = (u - (u.min() + u.max()) * 0.5) * scale + (width / 2)
    sy = (v - (v.min() + v.max()) * 0.5) * scale + (height / 2)
    return sx, sy


def _mesh_edge_set(faces: np.ndarray) -> set[tuple[int, int]]:
    edge_set: set[tuple[int, int]] = set()
    for a, b, c in np.asarray(faces, dtype=int):
        for u, v in ((a, b), (b, c), (c, a)):
            uu, vv = int(u), int(v)
            if uu == vv:
                continue
            edge_set.add((uu, vv) if uu < vv else (vv, uu))
    return edge_set


def _collect_seam_edges(report: dict[str, Any]) -> list[tuple[int, int]]:
    edge_set: set[tuple[int, int]] = set()
    panels = report.get("panels", {})
    if not isinstance(panels, dict):
        return []
    for panel in panels.values():
        if not isinstance(panel, dict):
            continue
        entries = panel.get("edges", [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            edge = entry.get("edge")
            if not isinstance(edge, list) or len(edge) != 2:
                continue
            a = int(edge[0])
            b = int(edge[1])
            if a == b:
                continue
            edge_set.add((a, b) if a < b else (b, a))
    return sorted(edge_set)


def _edge_conformance_metrics(
    seam_edges: list[tuple[int, int]], mesh_edges: set[tuple[int, int]]
) -> dict[str, object]:
    unique = {(a, b) if a < b else (b, a) for a, b in seam_edges}
    if not unique:
        return {
            "seam_unique_edge_count": 0,
            "mesh_edge_valid_count": 0,
            "mesh_edge_invalid_count": 0,
            "mesh_edge_valid_ratio": 1.0,
        }
    valid = sum(1 for edge in unique if edge in mesh_edges)
    invalid = int(len(unique) - valid)
    return {
        "seam_unique_edge_count": int(len(unique)),
        "mesh_edge_valid_count": int(valid),
        "mesh_edge_invalid_count": int(invalid),
        "mesh_edge_valid_ratio": float(valid / max(1, len(unique))),
    }


def _vertex_colors(costs_npz: Path | None, vertex_count: int) -> list[tuple[int, int, int]]:
    if costs_npz is None:
        return [(90, 90, 90) for _ in range(vertex_count)]

    payload = np.load(costs_npz, allow_pickle=True)
    if "vertex_costs" in payload:
        costs = np.asarray(payload["vertex_costs"], dtype=float).reshape(-1)
    elif "costs" in payload:
        costs = np.asarray(payload["costs"], dtype=float).reshape(-1)
    else:
        return [(90, 90, 90) for _ in range(vertex_count)]

    if len(costs) < vertex_count:
        costs = np.pad(costs, (0, vertex_count - len(costs)), constant_values=float(costs[-1]) if len(costs) else 0.0)
    costs = costs[:vertex_count]
    lo, hi = float(np.min(costs)), float(np.max(costs))
    span = max(hi - lo, 1e-12)
    normalized = (costs - lo) / span
    return [_magma_like(float(v)) for v in normalized]


def _render_frame(
    vertices: np.ndarray,
    vertex_colors: list[tuple[int, int, int]],
    seam_edges: list[tuple[int, int]],
    yaw_deg: float,
    *,
    width: int,
    height: int,
    elev_deg: float,
    point_size: int,
    body_alpha: int,
    seam_width: int,
    seam_only: bool,
    output: Path,
) -> None:
    u, v, depth = _project_vertices(vertices, yaw_deg=yaw_deg, elev_deg=elev_deg)
    sx, sy = _to_screen(u, v, width, height)
    image = Image.new("RGB", (width, height), (250, 248, 245))
    draw = ImageDraw.Draw(image, "RGBA")

    if not seam_only:
        order = np.argsort(depth)
        step = max(1, len(order) // 9000)
        radius = max(1, int(round(point_size / 2)))
        for idx in order[::step]:
            x = int(round(float(sx[idx])))
            y = int(round(float(sy[idx])))
            if not (0 <= x < width and 0 <= y < height):
                continue
            r, g, b = vertex_colors[idx]
            y_screen = height - 1 - y
            draw.ellipse(
                (x - radius, y_screen - radius, x + radius, y_screen + radius),
                fill=(r, g, b, int(max(0, min(255, body_alpha)))),
            )

    for a, b in seam_edges:
        if a >= len(vertices) or b >= len(vertices):
            continue
        x1 = int(round(float(sx[a])))
        y1 = int(round(float(sy[a])))
        x2 = int(round(float(sx[b])))
        y2 = int(round(float(sy[b])))
        draw.line(
            (x1, height - 1 - y1, x2, height - 1 - y2),
            fill=(20, 220, 220, 230),
            width=max(1, int(seam_width)),
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _encode_orbit(frames_dir: Path, gif_path: Path, webm_path: Path) -> None:
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    webm_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "18",
            "-i",
            str(frames_dir / "frame_%03d.png"),
            "-vf",
            "scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            str(gif_path),
        ],
        check=True,
    )
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="NPZ with vertices/faces.")
    parser.add_argument("--seam-report", type=Path, default=None, help="Optional seam_report.json.")
    parser.add_argument("--costs", type=Path, default=None, help="Optional NPZ with vertex_costs for colormap.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--stem",
        type=str,
        required=True,
        help="Output stem (filenames become <stem>__front_3q.png, <stem>__orbit.webm, etc).",
    )
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
    parser.add_argument("--rotate-x", type=float, default=0.0, help="Rigid X rotation applied to vertices (degrees).")
    parser.add_argument("--rotate-y", type=float, default=0.0, help="Rigid Y rotation applied to vertices (degrees).")
    parser.add_argument("--rotate-z", type=float, default=0.0, help="Rigid Z rotation applied to vertices (degrees).")
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--point-size", type=int, default=6)
    parser.add_argument("--body-alpha", type=int, default=160)
    parser.add_argument("--seam-width", type=int, default=6)
    parser.add_argument(
        "--seam-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render seam lines only (no pointcloud).",
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
        help="Optional width axis override when --canonicalize is enabled (depth inferred).",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )
    args = parser.parse_args()

    mesh_payload = np.load(args.mesh)
    vertices_raw = np.asarray(mesh_payload["vertices"], dtype=float)
    faces = np.asarray(mesh_payload["faces"], dtype=int) if "faces" in mesh_payload else None
    vertices, axis_meta = _canonicalize_vertices(
        vertices_raw,
        enabled=bool(args.canonicalize),
        up_axis=str(args.axis_up),
        width_axis=args.axis_width,
    )
    if any(abs(v) > 1e-9 for v in (args.rotate_x, args.rotate_y, args.rotate_z)):
        vertices = _rotate_xyz(vertices, args.rotate_x, args.rotate_y, args.rotate_z)

    seam_report: dict[str, Any] | None = None
    seam_edges: list[tuple[int, int]] = []
    if args.seam_report is not None:
        seam_report = json.loads(args.seam_report.read_text(encoding="utf-8"))
        seam_edges = _collect_seam_edges(seam_report)
        if seam_edges:
            max_idx = max(max(a, b) for a, b in seam_edges)
            if max_idx >= len(vertices):
                raise ValueError(
                    f"Seam indices out of range: max={max_idx} but mesh has {len(vertices)} vertices. "
                    f"mesh={args.mesh} seam_report={args.seam_report}"
                )

    mesh_edges = _mesh_edge_set(faces) if faces is not None else set()
    conformance = (
        _edge_conformance_metrics(seam_edges, mesh_edges) if seam_edges and mesh_edges else None
    )

    colors = _vertex_colors(args.costs, len(vertices))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / f"{args.stem}__orbit_frames__{args.timestamp}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(int(args.frames)):
        yaw = float(args.yaw_start) + float(args.yaw_offset) + (360.0 * idx / max(1, int(args.frames)))
        _render_frame(
            vertices,
            colors,
            seam_edges,
            yaw,
            width=int(args.width),
            height=int(args.height),
            elev_deg=float(args.elev),
            point_size=int(args.point_size),
            body_alpha=int(args.body_alpha),
            seam_width=int(args.seam_width),
            seam_only=bool(args.seam_only),
            output=frames_dir / f"frame_{idx:03d}.png",
        )

    still_path = out_dir / f"{args.stem}__front_3q__{args.timestamp}.png"
    _render_frame(
        vertices,
        colors,
        seam_edges,
        yaw_deg=-35.0 + float(args.yaw_offset),
        width=int(args.width),
        height=int(args.height),
        elev_deg=float(args.elev),
        point_size=int(args.point_size),
        body_alpha=int(args.body_alpha),
        seam_width=int(args.seam_width),
        seam_only=bool(args.seam_only),
        output=still_path,
    )

    gif_path = out_dir / f"{args.stem}__orbit__{args.timestamp}.gif"
    webm_path = out_dir / f"{args.stem}__orbit__{args.timestamp}.webm"
    _encode_orbit(frames_dir, gif_path, webm_path)

    manifest = {
        "timestamp_utc": str(args.timestamp),
        "mesh": {
            "path": str(args.mesh),
            "sha256": _sha256_path(args.mesh),
            "vertex_count": int(vertices_raw.shape[0]),
            "face_count": int(faces.shape[0]) if faces is not None else None,
        },
        "seam_report": {
            "path": str(args.seam_report) if args.seam_report is not None else None,
            "sha256": _sha256_path(args.seam_report) if args.seam_report is not None else None,
            "unique_edge_count": int(len({(a, b) if a < b else (b, a) for a, b in seam_edges})),
            "edge_conformance": conformance,
        },
        "costs": {
            "path": str(args.costs) if args.costs is not None else None,
            "sha256": _sha256_path(args.costs) if args.costs is not None else None,
        },
        "render": {
            "width": int(args.width),
            "height": int(args.height),
            "frames": int(args.frames),
            "elev": float(args.elev),
            "yaw_start": float(args.yaw_start),
            "yaw_offset": float(args.yaw_offset),
            "point_size": int(args.point_size),
            "body_alpha": int(args.body_alpha),
            "seam_width": int(args.seam_width),
            "seam_only": bool(args.seam_only),
            "canonicalize": bool(args.canonicalize),
            "rotate_deg": {
                "x": float(args.rotate_x),
                "y": float(args.rotate_y),
                "z": float(args.rotate_z),
            },
            "render_axis": dict(axis_meta),
        },
        "outputs": {
            "still_png": str(still_path),
            "orbit_gif": str(gif_path),
            "orbit_webm": str(webm_path),
            "frames_dir": str(frames_dir),
        },
    }
    (out_dir / f"{args.stem}__manifest__{args.timestamp}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(webm_path)


if __name__ == "__main__":
    main()
