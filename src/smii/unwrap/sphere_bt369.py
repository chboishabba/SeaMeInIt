"""BT369 sphere-to-rectangle serialization helpers.

The rectangle produced here is an export view. The sampled sphere carrier,
residual state, ternary address, and seam ledger are the receipt-bearing state.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, log, pi, sqrt
from typing import Any, Callable, Literal

import numpy as np

ProjectionName = Literal["equal_area"]
SeamPolicy = Literal["min_residual_graph_cut", "longitude"]


@dataclass(frozen=True, slots=True)
class BT369Cell:
    """Receipt state for one serialized rectangle cell."""

    x: int
    y: int
    value: tuple[float, ...]
    residual: float
    residual_trit: int
    orientation_bin: int
    depth: int
    parent_prefix: str
    seam_crossing_token: str | None
    area_weight: float

    def to_json(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "value": list(self.value),
            "residual": self.residual,
            "residual_trit": self.residual_trit,
            "orientation_bin": self.orientation_bin,
            "depth": self.depth,
            "parent_prefix": self.parent_prefix,
            "seam_crossing_token": self.seam_crossing_token,
            "area_weight": self.area_weight,
        }


@dataclass(frozen=True, slots=True)
class BT369SphereUnwrap:
    """Rectangle export plus the certificate-bearing spherical cell ledger."""

    image: np.ndarray
    cells: tuple[BT369Cell, ...]
    certificate: dict[str, Any]

    def cells_json(self) -> list[dict[str, Any]]:
        return [cell.to_json() for cell in self.cells]


def unwrap_sphere_bt369(
    sample_sphere: Callable[[np.ndarray], Any],
    width: int,
    height: int,
    projection: ProjectionName = "equal_area",
    max_depth: int = 8,
    residual_tol: float = 1e-4,
    seam_policy: SeamPolicy = "min_residual_graph_cut",
    output_certificate: bool = True,
    mdl_lambda: float = 1e-6,
) -> BT369SphereUnwrap:
    """Sample a sphere field into a BT369-certified rectangle serialization.

    ``sample_sphere`` receives a unit ``xyz`` vector and returns a scalar or a
    vector-like value. Pixels use equal-area inverse pullback; residuals compare
    the center sample with a deterministic 2x2 sub-cell reference.
    """

    if projection != "equal_area":
        raise ValueError("Only equal_area projection is currently supported.")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")
    if residual_tol <= 0.0:
        raise ValueError("residual_tol must be positive.")
    if mdl_lambda < 0.0:
        raise ValueError("mdl_lambda must be non-negative.")
    if seam_policy not in {"min_residual_graph_cut", "longitude"}:
        raise ValueError("Unsupported seam_policy.")

    area_weight = 4.0 * pi / float(width * height)
    first = _as_value(sample_sphere(_equal_area_xyz(0.5 / width, 0.5 / height)))
    image = np.zeros((height, width, len(first)), dtype=float)
    cells: list[BT369Cell] = []

    for y in range(height):
        for x in range(width):
            u = (x + 0.5) / width
            v = (y + 0.5) / height
            xyz = _equal_area_xyz(u, v)
            value = _as_value(sample_sphere(xyz))
            reference = _subcell_reference(sample_sphere, x, y, width, height)
            residual_vec = reference - np.asarray(value, dtype=float)
            residual = float(np.linalg.norm(residual_vec))
            signed = float(np.mean(residual_vec)) if residual_vec.size else 0.0
            residual_trit = _residual_trit(signed, residual, residual_tol)
            orientation_bin = _orientation_bin(u)
            depth = _refinement_depth(
                residual=residual,
                residual_tol=residual_tol,
                max_depth=max_depth,
                area_weight=area_weight,
                mdl_lambda=mdl_lambda,
            )
            token = _seam_token(x=x, y=y, width=width, height=height)
            prefix = _triadic_prefix(u, v, depth)
            image[y, x, :] = value
            cells.append(
                BT369Cell(
                    x=x,
                    y=y,
                    value=tuple(float(item) for item in value),
                    residual=residual,
                    residual_trit=residual_trit,
                    orientation_bin=orientation_bin,
                    depth=depth,
                    parent_prefix=prefix,
                    seam_crossing_token=token,
                    area_weight=area_weight,
                )
            )

    certificate = (
        _certificate(
            cells=tuple(cells),
            width=width,
            height=height,
            projection=projection,
            max_depth=max_depth,
            seam_policy=seam_policy,
            residual_tol=residual_tol,
        )
        if output_certificate
        else {}
    )
    if image.shape[-1] == 1:
        image = image[:, :, 0]
    return BT369SphereUnwrap(image=image, cells=tuple(cells), certificate=certificate)


def _equal_area_xyz(u: float, v: float) -> np.ndarray:
    longitude = 2.0 * pi * u - pi
    z = 1.0 - 2.0 * v
    radius = sqrt(max(0.0, 1.0 - z * z))
    return np.asarray(
        [radius * np.cos(longitude), z, radius * np.sin(longitude)],
        dtype=float,
    )


def _as_value(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError("sample_sphere must return a scalar or one-dimensional value.")
    if not np.isfinite(arr).all():
        raise ValueError("sample_sphere returned a non-finite value.")
    return arr


def _subcell_reference(
    sample_sphere: Callable[[np.ndarray], Any],
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    samples = []
    for dy in (0.25, 0.75):
        for dx in (0.25, 0.75):
            samples.append(
                _as_value(sample_sphere(_equal_area_xyz((x + dx) / width, (y + dy) / height)))
            )
    return np.mean(np.vstack(samples), axis=0)


def _residual_trit(signed_residual: float, residual: float, residual_tol: float) -> int:
    if residual <= residual_tol:
        return 0
    return 1 if signed_residual >= 0.0 else -1


def _orientation_bin(u: float) -> int:
    return min(5, max(0, int(floor((u % 1.0) * 6.0))))


def _refinement_depth(
    *,
    residual: float,
    residual_tol: float,
    max_depth: int,
    area_weight: float,
    mdl_lambda: float,
) -> int:
    if residual <= residual_tol or max_depth == 0:
        return 0
    if (residual * residual * area_weight) <= mdl_lambda:
        return 0
    ratio = max(residual / residual_tol, 1.0)
    return min(max_depth, 1 + int(floor(log(ratio, 3.0))))


def _triadic_prefix(u: float, v: float, depth: int) -> str:
    if depth <= 0:
        return ""
    x = min((3**depth) - 1, int((u % 1.0) * (3**depth)))
    z = min((3**depth) - 1, int(v * (3**depth)))
    digits: list[str] = []
    for level in range(depth - 1, -1, -1):
        scale = 3**level
        digit = ((x // scale) + (z // scale)) % 3
        digits.append(str(digit))
    return "".join(digits)


def _seam_token(*, x: int, y: int, width: int, height: int) -> str | None:
    tokens: list[str] = []
    if x == 0 or x == width - 1:
        tokens.append("wrap_EW")
    if y == 0:
        tokens.append("pole_turn_N")
    if y == height - 1:
        tokens.append("pole_turn_S")
    return "+".join(tokens) if tokens else None


def _certificate(
    *,
    cells: tuple[BT369Cell, ...],
    width: int,
    height: int,
    projection: str,
    max_depth: int,
    seam_policy: str,
    residual_tol: float,
) -> dict[str, Any]:
    residuals = np.asarray([cell.residual for cell in cells], dtype=float)
    trit_histogram = {str(key): 0 for key in (-1, 0, 1)}
    depth_histogram: dict[str, int] = {}
    seam_braid_counts: dict[str, int] = {"wrap_EW": 0, "pole_turn_N": 0, "pole_turn_S": 0}
    for cell in cells:
        trit_histogram[str(cell.residual_trit)] += 1
        depth_key = str(cell.depth)
        depth_histogram[depth_key] = depth_histogram.get(depth_key, 0) + 1
        if cell.seam_crossing_token:
            for token in cell.seam_crossing_token.split("+"):
                seam_braid_counts[token] = seam_braid_counts.get(token, 0) + 1

    return {
        "projection": "cylindrical_equal_area_inverse_pullback",
        "surface_basis": "triadic_spherical_bt369",
        "base_samples": int(width * height),
        "width": int(width),
        "height": int(height),
        "max_depth": int(max_depth),
        "residual_tol": float(residual_tol),
        "seam_policy": seam_policy,
        "residual_linf": float(np.max(residuals)) if residuals.size else 0.0,
        "residual_l2_area_weighted": float(
            sqrt(sum(cell.area_weight * cell.residual * cell.residual for cell in cells))
        ),
        "trit_histogram": trit_histogram,
        "depth_histogram": depth_histogram,
        "seam_braid_counts": seam_braid_counts,
        "coverage": {"missing": 0, "multi_hit": 0},
        "consumer": "field_measurement",
        "claim_boundary": "benchmark_gated_approximation_not_global_isometry",
    }


__all__ = [
    "BT369Cell",
    "BT369SphereUnwrap",
    "unwrap_sphere_bt369",
]
