#!/usr/bin/env python3
"""Render side-by-side diagnostics for the real ROM kernel fields."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from smii.meshing import load_body_record
from smii.rom.sampler_real import (
    ParameterLayout,
    PoseSample,
    SmplxPoseBackend,
    _build_joint_map,
    _central_difference,
    _find_params_path,
    _load_pose_sweep,
    _load_smplx_parameter_payload,
    _load_weights,
    _pose_defaults_from_payload,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]


def _field_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "p95": float(np.percentile(arr, 95.0)),
    }


def compute_pose_kernel_fields(
    backend: Any,
    *,
    neutral_pose: Mapping[str, np.ndarray],
    sample: PoseSample,
    layout: ParameterLayout,
    weights: np.ndarray,
    fd_step: float,
    epsilon: float,
) -> dict[str, np.ndarray]:
    """Compute the three most useful scalar fields for one ROM sample.

    Returns:
        displacement_magnitude:
            ||V(theta) - V0|| per vertex
        derivative_magnitude:
            sum_j w_j ||dV/dtheta_j||^2 per vertex
        seam_sensitivity:
            sum_j w_j |disp_hat dot dV/dtheta_j|^2 per vertex
    """

    v0 = np.asarray(backend.evaluate(neutral_pose), dtype=float)
    vt = np.asarray(backend.evaluate(sample.parameters), dtype=float)
    disp = vt - v0
    displacement_magnitude = np.linalg.norm(disp, axis=1)
    disp_norm = disp / (displacement_magnitude[:, None] + float(epsilon))

    derivative_magnitude = np.zeros(v0.shape[0], dtype=float)
    seam_sensitivity = np.zeros(v0.shape[0], dtype=float)

    coordinate_indices = [idx for idx, w in enumerate(np.asarray(weights, dtype=float)) if float(w) != 0.0]
    for coord_idx in coordinate_indices:
        block, local = layout.coordinate(coord_idx)
        deriv = np.asarray(
            _central_difference(
                backend,
                sample.parameters,
                block=block,
                index=local,
                step=fd_step,
            ),
            dtype=float,
        )
        weight = float(weights[coord_idx])
        derivative_magnitude += weight * np.sum(deriv * deriv, axis=1)
        seam_sensitivity += weight * np.abs(np.sum(disp_norm * deriv, axis=1)) ** 2

    return {
        "displacement_magnitude": displacement_magnitude,
        "derivative_magnitude": derivative_magnitude,
        "seam_sensitivity": seam_sensitivity,
    }


def _render_field_png(
    vertices: np.ndarray,
    faces: np.ndarray | None,
    field: np.ndarray,
    *,
    title: str,
    output: Path,
) -> Path | None:
    if plt is None:
        return None

    verts = np.asarray(vertices, dtype=float)
    values = np.asarray(field, dtype=float).reshape(-1)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6.5, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        c=values,
        cmap="magma",
        s=3,
        alpha=0.95,
    )
    ax.view_init(elev=12, azim=112)
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def _html_image(path: str | None, label: str) -> str:
    if path is None:
        return f"<div class='missing'>Missing {html.escape(label)}</div>"
    return (
        f"<figure class='card'><img src='{html.escape(path)}' alt='{html.escape(label)}' />"
        f"<figcaption><code>{html.escape(label)}</code></figcaption></figure>"
    )


def render_kernel_diagnostic(
    *,
    body_path: Path,
    poses_path: Path,
    weights_path: Path,
    out_dir: Path,
    params_path: Path | None = None,
    fd_step: float = 1e-3,
    epsilon: float = 1e-6,
    pose_limit: int | None = None,
) -> Path:
    body_record = load_body_record(body_path)
    vertices = np.asarray(body_record["vertices"], dtype=float)
    faces_raw = body_record.get("faces")
    faces = np.asarray(faces_raw, dtype=int) if faces_raw is not None else None
    resolved_params = _find_params_path(body_path, params_path)
    parameter_payload = json.loads(resolved_params.read_text(encoding="utf-8"))
    decoded_params, _, _, _ = _load_smplx_parameter_payload(parameter_payload)
    backend = SmplxPoseBackend(parameter_payload=parameter_payload, params_path=resolved_params)
    layout = ParameterLayout.from_shapes(backend.parameter_shapes())
    base_pose = _pose_defaults_from_payload(decoded_params, layout)
    neutral_pose, samples, pose_meta = _load_pose_sweep(
        poses_path,
        layout=layout,
        pose_limit=pose_limit,
        base_pose=base_pose,
    )
    weights = _load_weights(weights_path, layout=layout, joint_map=_build_joint_map())

    out_dir.mkdir(parents=True, exist_ok=True)
    pose_rows: list[dict[str, Any]] = []
    aggregate = {
        "displacement_magnitude": np.zeros(vertices.shape[0], dtype=float),
        "derivative_magnitude": np.zeros(vertices.shape[0], dtype=float),
        "seam_sensitivity": np.zeros(vertices.shape[0], dtype=float),
    }

    for sample in samples:
        fields = compute_pose_kernel_fields(
            backend,
            neutral_pose=neutral_pose,
            sample=sample,
            layout=layout,
            weights=weights,
            fd_step=fd_step,
            epsilon=epsilon,
        )
        for name, values in fields.items():
            aggregate[name] += float(sample.weight) * np.asarray(values, dtype=float)

        pose_dir = out_dir / sample.pose_id
        pose_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pose_dir / "fields.npz", **fields)
        images: dict[str, str | None] = {}
        for field_name, values in fields.items():
            output = pose_dir / f"{field_name}.png"
            rendered = _render_field_png(
                vertices,
                faces,
                values,
                title=f"{sample.pose_id}: {field_name}",
                output=output,
            )
            images[field_name] = output.name if rendered is not None else None
        pose_rows.append(
            {
                "pose_id": sample.pose_id,
                "weight": float(sample.weight),
                "stats": {name: _field_stats(values) for name, values in fields.items()},
                "images": images,
            }
        )

    aggregate_dir = out_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(aggregate_dir / "fields.npz", **aggregate)
    aggregate_images: dict[str, str | None] = {}
    for field_name, values in aggregate.items():
        output = aggregate_dir / f"{field_name}.png"
        rendered = _render_field_png(
            vertices,
            faces,
            values,
            title=f"aggregate: {field_name}",
            output=output,
        )
        aggregate_images[field_name] = output.name if rendered is not None else None

    summary = {
        "body_path": str(body_path),
        "params_path": str(resolved_params),
        "poses_path": str(poses_path),
        "weights_path": str(weights_path),
        "pose_count": len(samples),
        "vertex_count": int(vertices.shape[0]),
        "fd_step": float(fd_step),
        "epsilon": float(epsilon),
        "pose_meta": pose_meta,
        "aggregate_stats": {name: _field_stats(values) for name, values in aggregate.items()},
        "poses": pose_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sections: list[str] = []
    sections.append(
        "<section><h2>Aggregate</h2><div class='grid'>"
        + "".join(
            _html_image(
                f"aggregate/{aggregate_images[name]}" if aggregate_images[name] else None,
                f"aggregate/{name}",
            )
            for name in ("displacement_magnitude", "derivative_magnitude", "seam_sensitivity")
        )
        + "</div></section>"
    )
    for row in pose_rows:
        pose_id = str(row["pose_id"])
        stats = row["stats"]
        sections.append(
            "<section>"
            f"<h2>{html.escape(pose_id)}</h2>"
            f"<p>weight={row['weight']:.3f} | displacement p95={stats['displacement_magnitude']['p95']:.6f} | "
            f"derivative p95={stats['derivative_magnitude']['p95']:.6f} | "
            f"seam_sensitivity p95={stats['seam_sensitivity']['p95']:.6f}</p>"
            "<div class='grid'>"
            + "".join(
                _html_image(
                    f"{pose_id}/{row['images'][name]}" if row["images"][name] else None,
                    f"{pose_id}/{name}",
                )
                for name in ("displacement_magnitude", "derivative_magnitude", "seam_sensitivity")
            )
            + "</div></section>"
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ROM Kernel Diagnostic</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; color: #222; background: #f4efe8; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }}
    .card {{ margin: 0; padding: 0.75rem; background: #fffaf2; border: 1px solid #d7cec1; border-radius: 14px; }}
    img {{ width: 100%; border-radius: 10px; background: #fff; }}
    code {{ background: #f4f4f4; padding: 0.08rem 0.25rem; }}
    .callout {{ border-left: 4px solid #2f6db3; padding: 0.75rem 1rem; background: #f6fbff; margin-bottom: 1.5rem; }}
    .missing {{ padding: 2rem 1rem; background: #fff6f6; border: 1px solid #e2b5b5; border-radius: 10px; }}
  </style>
</head>
<body>
  <h1>ROM Kernel Diagnostic</h1>
  <div class="callout">
    <p>This page renders three scalar fields on the same body and same pose sweep:</p>
    <p><code>displacement_magnitude</code> = ||V(theta) - V0||</p>
    <p><code>derivative_magnitude</code> = sum_j w_j ||dV/dtheta_j||^2</p>
    <p><code>seam_sensitivity</code> = sum_j w_j |disp_hat dot dV/dtheta_j|^2</p>
  </div>
  {''.join(sections)}
</body>
</html>
"""
    index_path = out_dir / "index.html"
    index_path.write_text(html_text, encoding="utf-8")
    return index_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--params", type=Path, default=None)
    parser.add_argument("--fd-step", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--pose-limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    index_path = render_kernel_diagnostic(
        body_path=args.body,
        poses_path=args.poses,
        weights_path=args.weights,
        out_dir=args.out_dir,
        params_path=args.params,
        fd_step=float(args.fd_step),
        epsilon=float(args.epsilon),
        pose_limit=args.pose_limit,
    )
    print(index_path)


if __name__ == "__main__":
    main()
