"""Manufacturing-aware, deterministic BT369 region regularization.

Area and seam lengths are measured on the source 3D mesh. These are
screening proxies, not physical fabrication or assembly certification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections import defaultdict
from typing import Sequence

import numpy as np

from smii.unwrap.bt369_pattern_serializer import _circular_orientation_distance


@dataclass(frozen=True)
class ManufacturingLimits:
    max_charts: int = 64
    min_area_fraction: float = 0.005
    min_faces: int = 10
    max_seam_length_ratio: float = 12.0

    def __post_init__(self) -> None:
        if self.max_charts < 1 or self.min_faces < 1:
            raise ValueError("chart and face limits must be positive")
        if not 0 <= self.min_area_fraction < 1:
            raise ValueError("min_area_fraction must be in [0, 1)")
        if not math.isfinite(self.max_seam_length_ratio) or self.max_seam_length_ratio <= 0:
            raise ValueError("max_seam_length_ratio must be finite and positive")


def _geometry(vertices: np.ndarray, faces: Sequence[tuple[int, int, int]]):
    arr = np.asarray(faces, dtype=int)
    v = np.asarray(vertices, dtype=float)
    areas = (
        np.linalg.norm(np.cross(v[arr[:, 1]] - v[arr[:, 0]], v[arr[:, 2]] - v[arr[:, 0]]), axis=1)
        / 2
    )
    if not np.isfinite(areas).all() or float(areas.sum()) <= 0:
        raise ValueError("panel areas must be finite and positive")
    edge_faces = defaultdict(list)
    for i, (a, b, c) in enumerate(faces):
        for x, y in ((a, b), (b, c), (c, a)):
            edge_faces[tuple(sorted((x, y)))].append(i)
    return v, areas, edge_faces


def _partition(charts: Sequence[Sequence[int]], face_count: int):
    result = tuple(
        sorted(
            (tuple(sorted(int(f) for f in chart)) for chart in charts if chart), key=lambda c: c[0]
        )
    )
    flat = [f for c in result for f in c]
    if sorted(flat) != list(range(face_count)):
        raise ValueError("partition must cover every face exactly once")
    return result


def chart_diagnostics(
    *,
    vertices: np.ndarray,
    faces: Sequence[tuple[int, int, int]],
    partition: Sequence[Sequence[int]],
    limits: ManufacturingLimits,
) -> dict:
    v, areas, edge_faces = _geometry(vertices, faces)
    charts = _partition(partition, len(faces))
    owner = {f: i for i, chart in enumerate(charts) for f in chart}
    fractions = [float(areas[list(c)].sum() / areas.sum()) for c in charts]
    seam_length = 0.0
    boundary_length = 0.0
    for (a, b), incident in edge_faces.items():
        length = float(np.linalg.norm(v[a] - v[b]))
        if len(incident) == 1:
            boundary_length += length
        elif len(incident) == 2 and owner[incident[0]] != owner[incident[1]]:
            seam_length += length
    scale = math.sqrt(float(areas.sum()))
    ratio = seam_length / scale
    blockers = []
    if len(charts) > limits.max_charts:
        blockers.append("excessive_fragmentation")
    if any(
        len(c) < limits.min_faces or fraction < limits.min_area_fraction
        for c, fraction in zip(charts, fractions)
    ):
        blockers.append("subminimum_chart_area")
    if ratio > limits.max_seam_length_ratio:
        blockers.append("excessive_seam_complexity")
    return {
        "chart_count": len(charts),
        "minimum_face_count": min(map(len, charts)),
        "minimum_area_fraction": min(fractions),
        "total_internal_seam_length": seam_length,
        "external_boundary_length": boundary_length,
        "seam_length_over_sqrt_area": ratio,
        "limits": {
            "max_charts": limits.max_charts,
            "min_faces": limits.min_faces,
            "min_area_fraction": limits.min_area_fraction,
            "max_seam_length_ratio": limits.max_seam_length_ratio,
        },
        "manufacturing_admissible": not blockers,
        "manufacturing_blockers": blockers,
        "claim_boundary": "geometric_manufacturing_proxy_not_fabrication_authority",
    }


def regularize_partition(
    *,
    vertices: np.ndarray,
    faces: Sequence[tuple[int, int, int]],
    partition: Sequence[Sequence[int]],
    cells: Sequence[object],
    limits: ManufacturingLimits,
) -> tuple[tuple[int, ...], ...]:
    """Merge the smallest deficient connected chart into a neighboring chart.

    Shared-edge length, residual mismatch and circular orientation distance
    rank merge targets. A merge never disconnects a chart; unattached islands
    remain blocked by the manufacturing gate.
    """
    v, areas, edge_faces = _geometry(vertices, faces)
    charts = _partition(partition, len(faces))
    if len(cells) != len(faces):
        raise ValueError("cell count must equal face count")

    face_adj: list[list[tuple[int, float]]] = [[] for _ in range(len(faces))]
    for (a, b), incident in edge_faces.items():
        if len(incident) == 2:
            f1, f2 = incident
            length = float(np.linalg.norm(v[a] - v[b]))
            face_adj[f1].append((f2, length))
            face_adj[f2].append((f1, length))

    regions = {i: set(c) for i, c in enumerate(charts)}
    owner = {f: i for i, chart in regions.items() for f in chart}
    total = float(areas.sum())

    res_arr = np.asarray(
        [float(getattr(cells[f], "residual", 0.0)) for f in range(len(faces))], dtype=float
    )
    or_arr = np.asarray(
        [int(getattr(cells[f], "orientation_bin", 0)) for f in range(len(faces))], dtype=int
    )

    reg_res_sum = {i: float(res_arr[list(c)].sum()) for i, c in regions.items()}
    reg_area_sum = {i: float(areas[list(c)].sum()) for i, c in regions.items()}

    while len(regions) > 1:
        deficient = sorted(
            (
                i
                for i, c in regions.items()
                if len(c) < limits.min_faces or reg_area_sum[i] / total < limits.min_area_fraction
            ),
            key=lambda i: (reg_area_sum[i], i),
        )
        if not deficient:
            break
        moved = False
        for source in deficient:
            src = regions[source]
            neighbors: dict[int, float] = defaultdict(float)
            for f in src:
                for n_f, length in face_adj[f]:
                    target_cand = owner[n_f]
                    if target_cand != source:
                        neighbors[target_cand] += length
            if not neighbors:
                continue
            src_res = reg_res_sum[source] / len(src)
            src_or = int(or_arr[min(src)])

            def rank(target: int):
                dest = regions[target]
                dst_res = reg_res_sum[target] / len(dest)
                dst_or = int(or_arr[min(dest)])
                return (
                    abs(src_res - dst_res) + 0.1 * _circular_orientation_distance(src_or, dst_or),
                    -neighbors[target],
                    target,
                )

            target = min(neighbors, key=rank)
            regions[target].update(src)
            reg_res_sum[target] += reg_res_sum[source]
            reg_area_sum[target] += reg_area_sum[source]
            for face in src:
                owner[face] = target
            del regions[source]
            del reg_res_sum[source]
            del reg_area_sum[source]
            moved = True
            break
        if not moved:
            break
    return _partition(tuple(regions.values()), len(faces))
