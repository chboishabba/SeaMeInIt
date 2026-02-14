#!/usr/bin/env python3
"""Render seam variant still/orbit artifacts with configurable point size."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

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


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_AXIS_NAME = {0: "x", 1: "y", 2: "z"}


def _canonicalize_vertices(
    vertices: np.ndarray,
    *,
    enabled: bool,
    up_axis: str,
    width_axis: str | None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Reorder vertices into a stable render frame.

    Historical renders inferred up-axis from max span which is unstable on T-poses
    (arm span can exceed height). Protocol now: default to an explicit up-axis.
    """

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

    width: int
    if width_axis is not None:
        width_axis = str(width_axis).lower()
        if width_axis not in _AXIS_INDEX:
            raise ValueError(f"Unknown width axis '{width_axis}'. Use one of: x,y,z.")
        width = _AXIS_INDEX[width_axis]
        if width == up:
            raise ValueError("width axis must differ from up axis.")
        if width not in remaining:
            raise ValueError("width axis must be one of the non-up axes.")
    else:
        # Choose width among the remaining axes by span for a reasonable default.
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


def _validate_edge_indices(
    seam_edges: list[tuple[int, int]], vertex_count: int, run_dir: Path
) -> None:
    invalid = [edge for edge in seam_edges if edge[0] >= vertex_count or edge[1] >= vertex_count]
    if not invalid:
        return
    sample = ", ".join(f"({a},{b})" for a, b in invalid[:5])
    raise ValueError(
        "Seam/body vertex index mismatch for "
        f"'{run_dir.name}': body has {vertex_count} vertices but {len(invalid)} seam edges reference out-of-range "
        f"indices (examples: {sample}). Use a matching body mesh for this seam report."
    )


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


def _slug(value: str, *, max_len: int = 80) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    token = token.strip("_.-") or "na"
    return token[:max_len]


def _descriptive_name(
    *,
    kind: str,
    run_name: str,
    body_path: Path,
    cost_path: Path,
    seam_only: bool,
    point_size: int,
    seam_width: int,
    timestamp: str,
    ext: str,
) -> str:
    return (
        f"{_slug(kind, max_len=24)}__run-{_slug(run_name)}__body-{_slug(body_path.stem)}__"
        f"cost-{_slug(cost_path.stem)}__so-{int(bool(seam_only))}__ps-{int(point_size)}__"
        f"sw-{int(seam_width)}__ts-{_slug(timestamp, max_len=20)}.{ext}"
    )


def _body_stats(
    original_vertices: np.ndarray, canonical_vertices: np.ndarray, faces: np.ndarray | None
) -> dict[str, object]:
    original = np.asarray(original_vertices, dtype=float)
    canonical = np.asarray(canonical_vertices, dtype=float)
    original_min = np.min(original, axis=0)
    original_max = np.max(original, axis=0)
    canonical_min = np.min(canonical, axis=0)
    canonical_max = np.max(canonical, axis=0)
    return {
        "vertex_count": int(original.shape[0]),
        "face_count": int(faces.shape[0]) if faces is not None else None,
        "bbox_min": [float(v) for v in original_min],
        "bbox_max": [float(v) for v in original_max],
        "bbox_spans": [float(v) for v in (original_max - original_min)],
        "canonical_bbox_spans": [float(v) for v in (canonical_max - canonical_min)],
        "centroid": [float(v) for v in np.mean(original, axis=0)],
    }


def _vertex_colors(
    cost_path: Path, vertex_count: int
) -> tuple[list[tuple[int, int, int]], dict[str, object]]:
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
    colors = [_magma_like(value) for value in norm]
    stats = {
        "vertex_cost_count": int(len(costs)),
        "min": float(lo),
        "max": float(hi),
        "mean": float(np.nanmean(costs)),
        "std": float(np.nanstd(costs)),
    }
    return colors, stats


def _write_manifest(
    run_dir: Path,
    *,
    timestamp: str,
    report_path: Path,
    report: dict[str, object],
    body_path: Path,
    body_sha256: str,
    body_stats: Mapping[str, object],
    cost_path: Path,
    cost_sha256: str,
    cost_stats: Mapping[str, object],
    args: argparse.Namespace,
    edge_count: int,
    outputs: Mapping[str, str],
) -> tuple[Path, Path]:
    payload: dict[str, object] = {
        "timestamp": timestamp,
        "run": run_dir.name,
        "seam_report": {
            "path": str(report_path),
            "solver": report.get("solver"),
            "warnings_count": len(report.get("warnings", []))
            if isinstance(report.get("warnings"), list)
            else 0,
            "provenance": report.get("provenance", {}),
        },
        "body": {
            "path": str(body_path),
            "sha256": body_sha256,
            **dict(body_stats),
        },
        "cost_field": {
            "path": str(cost_path),
            "sha256": cost_sha256,
            **dict(cost_stats),
        },
        "render_params": {
            "width": int(args.width),
            "height": int(args.height),
            "elev": float(args.elev),
            "yaw_start": float(args.yaw_start),
            "frames": int(args.frames),
            "point_size": int(args.point_size),
            "body_alpha": int(args.body_alpha),
            "seam_width": int(args.seam_width),
            "seam_only": bool(args.seam_only),
            "canonicalize": bool(args.canonicalize),
            "axis_up": str(args.axis_up),
            "axis_width": str(args.axis_width) if args.axis_width is not None else None,
        },
        "seam": {
            "unique_edge_count": int(edge_count),
        },
        "outputs": dict(outputs),
    }
    latest = run_dir / "render_input_manifest.json"
    stamped = run_dir / f"render_input_manifest_{timestamp}.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    stamped.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return latest, stamped


def _resolve_runs(run_root: Path, include: Iterable[str]) -> list[Path]:
    selected = [name for name in include if name]
    if selected:
        return [run_root / name for name in selected]
    return sorted(path for path in run_root.iterdir() if path.is_dir() and "grain100" in path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, default=Path("outputs/seams_run/variant_matrix_20260212")
    )
    parser.add_argument("--body", type=Path, default=Path("outputs/afflec_demo/afflec_body.npz"))
    parser.add_argument(
        "--cost-default",
        type=Path,
        default=Path("outputs/rom/seam_costs_afflec_realshape_edges.npz"),
    )
    parser.add_argument(
        "--cost-smallrom", type=Path, default=Path("outputs/rom/seam_costs_afflec.npz")
    )
    parser.add_argument(
        "--run", action="append", default=[], help="Specific run name to render; repeatable."
    )
    parser.add_argument("--point-size", type=int, default=5, help="Point diameter in pixels.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--elev", type=float, default=14.0)
    parser.add_argument("--yaw-start", type=float, default=-120.0)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument(
        "--canonicalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reorder axes into a stable render frame before projection.",
    )
    parser.add_argument(
        "--axis-up",
        type=str,
        default="y",
        choices=("x", "y", "z"),
        help="Axis treated as 'up' when --canonicalize is enabled.",
    )
    parser.add_argument(
        "--axis-width",
        type=str,
        default=None,
        choices=("x", "y", "z"),
        help="Optional width axis override when --canonicalize is enabled (depth inferred).",
    )
    parser.add_argument(
        "--body-alpha", type=int, default=110, help="Alpha for body points (0-255)."
    )
    parser.add_argument("--seam-width", type=int, default=6, help="Pixel width for seam lines.")
    parser.add_argument(
        "--seam-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render seam lines only.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        help="Timestamp suffix for timestamped artifact outputs.",
    )
    parser.add_argument(
        "--archive-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Archive existing canonical artifacts into artifact_history/<timestamp> before overwrite.",
    )
    parser.add_argument(
        "--descriptive-alias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write additional descriptive artifact filenames alongside canonical overlay_* outputs.",
    )
    args = parser.parse_args()
    selected_runs_explicit = bool([name for name in args.run if name])

    body = np.load(args.body)
    body_vertices = np.asarray(body["vertices"], dtype=float)
    body_faces = np.asarray(body["faces"], dtype=int) if "faces" in body else None
    vertices, axis_meta = _canonicalize_vertices(
        body_vertices,
        enabled=bool(args.canonicalize),
        up_axis=str(args.axis_up),
        width_axis=args.axis_width,
    )
    body_sha256 = _sha256_path(args.body)
    body_meta = _body_stats(body_vertices, vertices, body_faces)
    body_meta["render_axis"] = dict(axis_meta)
    runs = _resolve_runs(args.run_root, args.run)
    summary: list[dict[str, object]] = []

    for run_dir in runs:
        report_path = run_dir / "seam_report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        edges = _collect_edges(report)
        _validate_edge_indices(edges, len(vertices), run_dir)
        cost_path = args.cost_smallrom if "smallrom" in run_dir.name else args.cost_default
        colors, cost_stats = _vertex_colors(cost_path, len(vertices))
        cost_sha256 = _sha256_path(cost_path)
        if args.archive_existing:
            _archive_existing_artifacts(run_dir, args.timestamp)

        still_path = run_dir / "overlay_front_3q.png"
        still_timestamped = run_dir / f"overlay_front_3q_{args.timestamp}.png"
        still_descriptive = run_dir / _descriptive_name(
            kind="front_3q",
            run_name=run_dir.name,
            body_path=args.body,
            cost_path=cost_path,
            seam_only=args.seam_only,
            point_size=args.point_size,
            seam_width=args.seam_width,
            timestamp=args.timestamp,
            ext="png",
        )
        _render_frame(
            vertices,
            colors,
            edges,
            args.yaw_start,
            width=args.width,
            height=args.height,
            elev_deg=args.elev,
            point_size=args.point_size,
            body_alpha=args.body_alpha,
            seam_width=args.seam_width,
            seam_only=args.seam_only,
            output=still_path,
        )
        shutil.copy2(still_path, still_timestamped)
        if args.descriptive_alias:
            shutil.copy2(still_path, still_descriptive)

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
                body_alpha=args.body_alpha,
                seam_width=args.seam_width,
                seam_only=args.seam_only,
                output=frame_path,
            )

        gif_path = run_dir / "overlay_orbit.gif"
        webm_path = run_dir / "overlay_orbit.webm"
        gif_timestamped = run_dir / f"overlay_orbit_{args.timestamp}.gif"
        webm_timestamped = run_dir / f"overlay_orbit_{args.timestamp}.webm"
        gif_descriptive = run_dir / _descriptive_name(
            kind="orbit",
            run_name=run_dir.name,
            body_path=args.body,
            cost_path=cost_path,
            seam_only=args.seam_only,
            point_size=args.point_size,
            seam_width=args.seam_width,
            timestamp=args.timestamp,
            ext="gif",
        )
        webm_descriptive = run_dir / _descriptive_name(
            kind="orbit",
            run_name=run_dir.name,
            body_path=args.body,
            cost_path=cost_path,
            seam_only=args.seam_only,
            point_size=args.point_size,
            seam_width=args.seam_width,
            timestamp=args.timestamp,
            ext="webm",
        )
        gif_path, webm_path = _encode_orbit(frames_dir, gif_path, webm_path)
        shutil.copy2(gif_path, gif_timestamped)
        shutil.copy2(webm_path, webm_timestamped)
        if args.descriptive_alias:
            shutil.copy2(gif_path, gif_descriptive)
            shutil.copy2(webm_path, webm_descriptive)
        shutil.rmtree(frames_dir)
        manifest_latest, manifest_timestamped = _write_manifest(
            run_dir,
            timestamp=args.timestamp,
            report_path=report_path,
            report=report,
            body_path=args.body,
            body_sha256=body_sha256,
            body_stats=body_meta,
            cost_path=cost_path,
            cost_sha256=cost_sha256,
            cost_stats=cost_stats,
            args=args,
            edge_count=len(edges),
            outputs={
                "png": str(still_path),
                "png_timestamped": str(still_timestamped),
                "png_descriptive": str(still_descriptive) if args.descriptive_alias else "",
                "gif": str(gif_path),
                "gif_timestamped": str(gif_timestamped),
                "gif_descriptive": str(gif_descriptive) if args.descriptive_alias else "",
                "webm": str(webm_path),
                "webm_timestamped": str(webm_timestamped),
                "webm_descriptive": str(webm_descriptive) if args.descriptive_alias else "",
            },
        )
        summary.append(
            {
                "run": run_dir.name,
                "solver": report.get("solver"),
                "total_cost": report.get("total_cost"),
                "warnings": len(report.get("warnings", []))
                if isinstance(report.get("warnings"), list)
                else 0,
                "edge_count": len(edges),
                "point_size": args.point_size,
                "timestamp": args.timestamp,
                "png": str(still_path),
                "png_timestamped": str(still_timestamped),
                "png_descriptive": str(still_descriptive) if args.descriptive_alias else None,
                "gif": str(gif_path),
                "gif_timestamped": str(gif_timestamped),
                "gif_descriptive": str(gif_descriptive) if args.descriptive_alias else None,
                "webm": str(webm_path),
                "webm_timestamped": str(webm_timestamped),
                "webm_descriptive": str(webm_descriptive) if args.descriptive_alias else None,
                "manifest": str(manifest_latest),
                "manifest_timestamped": str(manifest_timestamped),
            }
        )
        print(f"rendered {run_dir.name}")

    if selected_runs_explicit:
        summary_path = args.run_root / f"artifact_summary_selected_{args.timestamp}.json"
        summary_timestamped = summary_path
        summary_csv = args.run_root / f"artifact_summary_selected_{args.timestamp}.csv"
        summary_csv_timestamped = summary_csv
    else:
        summary_path = args.run_root / "artifact_summary_all.json"
        summary_timestamped = args.run_root / f"artifact_summary_{args.timestamp}.json"
        summary_csv = args.run_root / "artifact_summary_all.csv"
        summary_csv_timestamped = args.run_root / f"artifact_summary_{args.timestamp}.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_timestamped.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    header = (
        "run,solver,total_cost,warnings,edge_count,point_size,timestamp,"
        "png,png_timestamped,png_descriptive,gif,gif_timestamped,gif_descriptive,"
        "webm,webm_timestamped,webm_descriptive,manifest,manifest_timestamped\n"
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
                        str(row.get("edge_count", "")),
                        str(row.get("point_size", "")),
                        str(row.get("timestamp", "")),
                        str(row.get("png", "")),
                        str(row.get("png_timestamped", "")),
                        str(row.get("png_descriptive", "")),
                        str(row.get("gif", "")),
                        str(row.get("gif_timestamped", "")),
                        str(row.get("gif_descriptive", "")),
                        str(row.get("webm", "")),
                        str(row.get("webm_timestamped", "")),
                        str(row.get("webm_descriptive", "")),
                        str(row.get("manifest", "")),
                        str(row.get("manifest_timestamped", "")),
                    ]
                )
                + "\n"
            )
    if summary_csv != summary_csv_timestamped:
        shutil.copy2(summary_csv, summary_csv_timestamped)
    print(f"wrote {summary_path}")
    if summary_timestamped != summary_path:
        print(f"wrote {summary_timestamped}")
    print(f"wrote {summary_csv}")
    if summary_csv_timestamped != summary_csv:
        print(f"wrote {summary_csv_timestamped}")


if __name__ == "__main__":
    main()
