#!/usr/bin/env python3
"""Render seam variant still/orbit artifacts with configurable point size."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


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


def _canonicalize_vertices(vertices: np.ndarray) -> np.ndarray:
    spans = np.ptp(vertices, axis=0)
    vertical_axis = int(np.argmax(spans))
    horizontal_axes = [idx for idx in range(3) if idx != vertical_axis]
    horizontal_axes.sort(key=lambda idx: spans[idx], reverse=True)
    width_axis, depth_axis = horizontal_axes
    canonical = np.stack(
        [vertices[:, width_axis], vertices[:, depth_axis], vertices[:, vertical_axis]],
        axis=1,
    ).astype(float)
    canonical -= np.mean(canonical, axis=0, keepdims=True)
    return canonical


def _project_vertices(vertices: np.ndarray, yaw_deg: float, elev_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _collect_edges(report: dict[str, object]) -> list[tuple[int, int]]:
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
    output: Path,
) -> None:
    u, v, depth = _project_vertices(vertices, yaw_deg=yaw_deg, elev_deg=elev_deg)
    sx, sy = _to_screen(u, v, width, height)
    image = Image.new("RGB", (width, height), (250, 248, 245))
    draw = ImageDraw.Draw(image, "RGBA")

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
            fill=(r, g, b, 110),
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
            width=max(2, radius),
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _encode_orbit(frames_dir: Path, gif_path: Path, webm_path: Path) -> tuple[Path, Path]:
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
    return gif_path, webm_path


def _archive_existing_artifacts(run_dir: Path, timestamp: str) -> None:
    targets = (
        "overlay_front_3q.png",
        "overlay_orbit.gif",
        "overlay_orbit.webm",
        "seam_report.json",
    )
    existing = [run_dir / name for name in targets if (run_dir / name).exists()]
    if not existing:
        return
    archive_dir = run_dir / "artifact_history" / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    for file_path in existing:
        shutil.copy2(file_path, archive_dir / file_path.name)


def _vertex_colors(cost_path: Path, vertex_count: int) -> list[tuple[int, int, int]]:
    payload = np.load(cost_path)
    costs = np.asarray(payload["vertex_costs"], dtype=float)
    if len(costs) != vertex_count:
        raise ValueError(
            f"Vertex count mismatch: cost field has {len(costs)} entries while body has {vertex_count} vertices. "
            "Use a matching ROM cost file for this body mesh."
        )
    lo = float(np.nanmin(costs))
    hi = float(np.nanmax(costs))
    norm = (costs[:vertex_count] - lo) / max(hi - lo, 1e-6)
    return [_magma_like(value) for value in norm]


def _resolve_runs(run_root: Path, include: Iterable[str]) -> list[Path]:
    selected = [name for name in include if name]
    if selected:
        return [run_root / name for name in selected]
    return sorted(path for path in run_root.iterdir() if path.is_dir() and "grain100" in path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("outputs/seams_run/variant_matrix_20260212"))
    parser.add_argument("--body", type=Path, default=Path("outputs/afflec_demo/afflec_body.npz"))
    parser.add_argument("--cost-default", type=Path, default=Path("outputs/rom/seam_costs_afflec_realshape_edges.npz"))
    parser.add_argument("--cost-smallrom", type=Path, default=Path("outputs/rom/seam_costs_afflec.npz"))
    parser.add_argument("--run", action="append", default=[], help="Specific run name to render; repeatable.")
    parser.add_argument("--point-size", type=int, default=5, help="Point diameter in pixels.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--elev", type=float, default=14.0)
    parser.add_argument("--yaw-start", type=float, default=-120.0)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument(
        "--timestamp",
        type=str,
        default=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        help="Timestamp suffix for timestamped artifact outputs.",
    )
    parser.add_argument(
        "--archive-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Archive existing canonical artifacts into artifact_history/<timestamp> before overwrite.",
    )
    args = parser.parse_args()

    body = np.load(args.body)
    vertices = _canonicalize_vertices(np.asarray(body["vertices"], dtype=float))
    runs = _resolve_runs(args.run_root, args.run)
    summary: list[dict[str, object]] = []

    for run_dir in runs:
        report_path = run_dir / "seam_report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        edges = _collect_edges(report)
        cost_path = args.cost_smallrom if "smallrom" in run_dir.name else args.cost_default
        colors = _vertex_colors(cost_path, len(vertices))
        if args.archive_existing:
            _archive_existing_artifacts(run_dir, args.timestamp)

        still_path = run_dir / "overlay_front_3q.png"
        still_timestamped = run_dir / f"overlay_front_3q_{args.timestamp}.png"
        _render_frame(
            vertices,
            colors,
            edges,
            args.yaw_start,
            width=args.width,
            height=args.height,
            elev_deg=args.elev,
            point_size=args.point_size,
            output=still_path,
        )
        shutil.copy2(still_path, still_timestamped)

        frames_dir = run_dir / "orbit_frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(args.frames):
            yaw = args.yaw_start + (360.0 * idx / max(args.frames, 1))
            frame_path = frames_dir / f"frame_{idx:03d}.png"
            _render_frame(
                vertices,
                colors,
                edges,
                yaw,
                width=args.width,
                height=args.height,
                elev_deg=args.elev,
                point_size=args.point_size,
                output=frame_path,
            )

        gif_path = run_dir / "overlay_orbit.gif"
        webm_path = run_dir / "overlay_orbit.webm"
        gif_timestamped = run_dir / f"overlay_orbit_{args.timestamp}.gif"
        webm_timestamped = run_dir / f"overlay_orbit_{args.timestamp}.webm"
        gif_path, webm_path = _encode_orbit(frames_dir, gif_path, webm_path)
        shutil.copy2(gif_path, gif_timestamped)
        shutil.copy2(webm_path, webm_timestamped)
        shutil.rmtree(frames_dir)
        summary.append(
            {
                "run": run_dir.name,
                "solver": report.get("solver"),
                "total_cost": report.get("total_cost"),
                "warnings": len(report.get("warnings", [])) if isinstance(report.get("warnings"), list) else 0,
                "point_size": args.point_size,
                "timestamp": args.timestamp,
                "png": str(still_path),
                "png_timestamped": str(still_timestamped),
                "gif": str(gif_path),
                "gif_timestamped": str(gif_timestamped),
                "webm": str(webm_path),
                "webm_timestamped": str(webm_timestamped),
            }
        )
        print(f"rendered {run_dir.name}")

    summary_path = args.run_root / "artifact_summary_all.json"
    summary_timestamped = args.run_root / f"artifact_summary_{args.timestamp}.json"
    summary_csv = args.run_root / "artifact_summary_all.csv"
    summary_csv_timestamped = args.run_root / f"artifact_summary_{args.timestamp}.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_timestamped.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    header = (
        "run,solver,total_cost,warnings,point_size,timestamp,"
        "png,png_timestamped,gif,gif_timestamped,webm,webm_timestamped\n"
    )
    with summary_csv.open("w", encoding="utf-8") as stream:
        stream.write(header)
        for row in summary:
            stream.write(
                ",".join(
                    [
                        str(row.get("run", "")),
                        str(row.get("solver", "")),
                        str(row.get("total_cost", "")),
                        str(row.get("warnings", "")),
                        str(row.get("point_size", "")),
                        str(row.get("timestamp", "")),
                        str(row.get("png", "")),
                        str(row.get("png_timestamped", "")),
                        str(row.get("gif", "")),
                        str(row.get("gif_timestamped", "")),
                        str(row.get("webm", "")),
                        str(row.get("webm_timestamped", "")),
                    ]
                )
                + "\n"
            )
    shutil.copy2(summary_csv, summary_csv_timestamped)
    print(f"wrote {summary_path}")
    print(f"wrote {summary_timestamped}")
    print(f"wrote {summary_csv}")
    print(f"wrote {summary_csv_timestamped}")


if __name__ == "__main__":
    main()
