"""External competitor receipts for sphere unwrap benchmarking.

The harness ranks declared candidates under declared metrics. It does not claim
global sphere-plane optimality, true inverse correspondence, or manufacturing
authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from math import acos, atan2, exp, floor, pi, sqrt
from typing import Any, Callable, Iterable, Literal

import numpy as np

from .sphere_bt369 import unwrap_sphere_bt369

SphereCompetitorName = Literal[
    "bt369",
    "equal_area",
    "equirect",
    "cubed_sphere",
    "octahedral",
    "healpix",
    "xatlas",
    "slim",
    "bff",
    "optcuts",
    "blender_unwrap",
]

MEASURED_SPHERE_COMPETITORS: tuple[str, ...] = (
    "bt369",
    "equal_area",
    "equirect",
    "cubed_sphere",
    "octahedral",
)

OPTIONAL_COMPETITORS: tuple[str, ...] = (
    "healpix",
    "xatlas",
    "slim",
    "bff",
    "optcuts",
    "blender_unwrap",
)

DEFAULT_COMPETITORS: tuple[str, ...] = MEASURED_SPHERE_COMPETITORS + OPTIONAL_COMPETITORS

CLAIM_BOUNDARY = "declared_benchmark_candidate_not_global_isometry_or_true_inverse"


@dataclass(frozen=True, slots=True)
class SyntheticSphereField:
    """Named deterministic field used to harden sphere competitor benchmarks."""

    name: str
    family: str
    sample: Callable[[np.ndarray], Any]
    discontinuous: bool = False

    def to_json(self) -> dict[str, str | bool]:
        return {
            "name": self.name,
            "family": self.family,
            "discontinuous": self.discontinuous,
        }


@dataclass(frozen=True, slots=True)
class ExternalCompetitorMetrics:
    """Common metric vector for sphere and UV competitor receipts."""

    edge_length_residual: float | None
    area_residual: float | None
    angle_residual: float | None
    foldover_ratio: float | None
    residual_l2_area_weighted: float | None
    aggregate_score: float | None
    agreement_depth: int | None
    agreement_distance: int | None
    seam_length: float | None
    chart_count: int | None
    packing_efficiency: float | None
    inverse_roundtrip_error: float | None
    field_reconstruction_error: float | None
    dart_pressure_score: float | None
    grain_alignment_score: float | None
    panel_internal_variance: float | None
    seam_on_high_strain_penalty: float | None
    manufacturability_score: float | None

    def to_json(self) -> dict[str, float | int | None]:
        return {
            "edge_length_residual": self.edge_length_residual,
            "area_residual": self.area_residual,
            "angle_residual": self.angle_residual,
            "foldover_ratio": self.foldover_ratio,
            "residual_l2_area_weighted": self.residual_l2_area_weighted,
            "aggregate_score": self.aggregate_score,
            "agreement_depth": self.agreement_depth,
            "agreement_distance": self.agreement_distance,
            "seam_length": self.seam_length,
            "chart_count": self.chart_count,
            "packing_efficiency": self.packing_efficiency,
            "inverse_roundtrip_error": self.inverse_roundtrip_error,
            "field_reconstruction_error": self.field_reconstruction_error,
            "dart_pressure_score": self.dart_pressure_score,
            "grain_alignment_score": self.grain_alignment_score,
            "panel_internal_variance": self.panel_internal_variance,
            "seam_on_high_strain_penalty": self.seam_on_high_strain_penalty,
            "manufacturability_score": self.manufacturability_score,
        }


@dataclass(frozen=True, slots=True)
class ExternalCompetitorReceipt:
    """One competitor run, including unavailable optional methods."""

    name: str
    family: str
    available: bool
    reason: str | None
    metrics: ExternalCompetitorMetrics
    certificate: dict[str, Any]
    claim_boundary: str = CLAIM_BOUNDARY

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "available": self.available,
            "reason": self.reason,
            "metrics": self.metrics.to_json(),
            "certificate": self.certificate,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class ExternalCompetitorBenchmark:
    """Ranked declared competitor slice."""

    competitors: tuple[ExternalCompetitorReceipt, ...]
    width: int
    height: int
    claim_boundary: str = CLAIM_BOUNDARY

    @property
    def measured(self) -> tuple[ExternalCompetitorReceipt, ...]:
        return tuple(receipt for receipt in self.competitors if receipt.available)

    @property
    def unavailable(self) -> tuple[ExternalCompetitorReceipt, ...]:
        return tuple(receipt for receipt in self.competitors if not receipt.available)

    @property
    def winner(self) -> ExternalCompetitorReceipt:
        measured = self.measured
        if not measured:
            raise ValueError("Benchmark has no measured competitors.")
        return measured[0]

    def to_json(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "winner": self.winner.name if self.measured else None,
            "claim_boundary": self.claim_boundary,
            "competitors": [receipt.to_json() for receipt in self.competitors],
        }


@dataclass(frozen=True, slots=True)
class SphereFieldBenchmarkResult:
    """One adversarial field benchmark result."""

    field: SyntheticSphereField
    benchmark: ExternalCompetitorBenchmark

    @property
    def winner_name(self) -> str:
        return self.benchmark.winner.name

    def to_json(self) -> dict[str, Any]:
        return {
            "field": self.field.to_json(),
            "winner": self.winner_name,
            "benchmark": self.benchmark.to_json(),
        }


@dataclass(frozen=True, slots=True)
class AdversarialSphereBenchmarkSuite:
    """Per-field benchmark suite for the declared sphere competitor slice."""

    results: tuple[SphereFieldBenchmarkResult, ...]
    width: int
    height: int
    claim_boundary: str = CLAIM_BOUNDARY

    @property
    def winner_histogram(self) -> dict[str, int]:
        histogram: dict[str, int] = {}
        for result in self.results:
            histogram[result.winner_name] = histogram.get(result.winner_name, 0) + 1
        return histogram

    def to_json(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "claim_boundary": self.claim_boundary,
            "winner_histogram": self.winner_histogram,
            "results": [result.to_json() for result in self.results],
        }


def benchmark_external_sphere_competitors(
    sample_sphere: Callable[[np.ndarray], Any],
    *,
    width: int,
    height: int,
    competitors: Iterable[str] = DEFAULT_COMPETITORS,
    residual_tol: float = 1e-4,
    max_depth: int = 8,
) -> ExternalCompetitorBenchmark:
    """Measure declared sphere unwrap competitors and return ranked receipts."""

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")
    if residual_tol <= 0.0:
        raise ValueError("residual_tol must be positive.")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")

    receipts = [
        _run_competitor(
            name=name,
            sample_sphere=sample_sphere,
            width=width,
            height=height,
            residual_tol=residual_tol,
            max_depth=max_depth,
        )
        for name in competitors
    ]
    return ExternalCompetitorBenchmark(
        competitors=tuple(
            sorted(
                receipts,
                key=lambda receipt: (
                    not receipt.available,
                    _score_or_penalty(receipt),
                    receipt.name,
                ),
            )
        ),
        width=width,
        height=height,
    )


def adversarial_sphere_fields() -> tuple[SyntheticSphereField, ...]:
    """Return deterministic fields that stress different carrier failures."""

    return (
        SyntheticSphereField("constant", "control", lambda xyz: 1.0),
        SyntheticSphereField("linear_xyz", "linear", lambda xyz: xyz.copy()),
        SyntheticSphereField(
            "low_frequency_harmonic",
            "spherical_harmonic_like",
            lambda xyz: 0.5 * xyz[1] + 0.25 * xyz[0] * xyz[2],
        ),
        SyntheticSphereField(
            "high_frequency_harmonic",
            "spherical_harmonic_like",
            lambda xyz: np.sin(7.0 * atan2(float(xyz[2]), float(xyz[0]))) * (1.0 - xyz[1] ** 2),
        ),
        SyntheticSphereField(
            "polar_cap",
            "localized_discontinuity",
            lambda xyz: 1.0 if xyz[1] > 0.72 else 0.0,
            discontinuous=True,
        ),
        SyntheticSphereField(
            "longitude_seam_stripe",
            "seam_discontinuity",
            lambda xyz: 1.0 if abs(atan2(float(xyz[2]), float(xyz[0]))) > 0.86 * pi else 0.0,
            discontinuous=True,
        ),
        SyntheticSphereField(
            "checkerboard_geodesic",
            "geodesic_patch_discontinuity",
            _checkerboard_geodesic_field,
            discontinuous=True,
        ),
        SyntheticSphereField(
            "localized_gaussian_bump",
            "localized_smooth",
            lambda xyz: exp(
                -24.0 * float(np.linalg.norm(xyz - np.asarray([0.35, 0.82, 0.45])) ** 2)
            ),
        ),
        SyntheticSphereField(
            "binary_hemisphere",
            "hemisphere_discontinuity",
            lambda xyz: 1.0 if xyz[0] + 0.25 * xyz[2] > 0.0 else -1.0,
            discontinuous=True,
        ),
        SyntheticSphereField(
            "band_limited_mix",
            "deterministic_band_limited",
            lambda xyz: (
                0.45 * xyz[0]
                - 0.3 * xyz[1] * xyz[2]
                + 0.2 * np.sin(3.0 * atan2(float(xyz[2]), float(xyz[0])))
                + 0.1 * (3.0 * xyz[1] ** 2 - 1.0)
            ),
        ),
    )


def benchmark_adversarial_sphere_fields(
    *,
    width: int,
    height: int,
    competitors: Iterable[str] = DEFAULT_COMPETITORS,
    fields: Iterable[SyntheticSphereField] | None = None,
    residual_tol: float = 1e-4,
    max_depth: int = 8,
) -> AdversarialSphereBenchmarkSuite:
    """Run the declared competitor slice against adversarial synthetic fields."""

    selected_fields = tuple(fields) if fields is not None else adversarial_sphere_fields()
    results = tuple(
        SphereFieldBenchmarkResult(
            field=field,
            benchmark=benchmark_external_sphere_competitors(
                field.sample,
                width=width,
                height=height,
                competitors=competitors,
                residual_tol=residual_tol,
                max_depth=max_depth,
            ),
        )
        for field in selected_fields
    )
    return AdversarialSphereBenchmarkSuite(results=results, width=width, height=height)


def _run_competitor(
    *,
    name: str,
    sample_sphere: Callable[[np.ndarray], Any],
    width: int,
    height: int,
    residual_tol: float,
    max_depth: int,
) -> ExternalCompetitorReceipt:
    if name == "bt369":
        return _run_bt369(
            sample_sphere=sample_sphere,
            width=width,
            height=height,
            residual_tol=residual_tol,
            max_depth=max_depth,
        )
    if name in {"equal_area", "equirect", "cubed_sphere", "octahedral"}:
        return _run_closed_form(name, sample_sphere=sample_sphere, width=width, height=height)
    if name == "healpix":
        return _run_healpix_optional(
            sample_sphere=sample_sphere,
            width=width,
            height=height,
        )
    if name in OPTIONAL_COMPETITORS:
        return _unavailable(name, "external adapter is registered but not installed or bound")
    raise ValueError(f"Unknown unwrap competitor '{name}'.")


def _run_bt369(
    *,
    sample_sphere: Callable[[np.ndarray], Any],
    width: int,
    height: int,
    residual_tol: float,
    max_depth: int,
) -> ExternalCompetitorReceipt:
    result = unwrap_sphere_bt369(
        sample_sphere,
        width=width,
        height=height,
        residual_tol=residual_tol,
        max_depth=max_depth,
    )
    certificate = result.certificate
    residual = float(certificate["residual_l2_area_weighted"])
    seam_counts = certificate["seam_braid_counts"]
    seam_length = sum(float(value) for value in seam_counts.values()) / float(width * height)
    depths = [cell.depth for cell in result.cells]
    agreement_depth = min(depths) if depths else 0
    metrics = _metrics(
        edge_length_residual=0.0,
        area_residual=0.0,
        angle_residual=0.0,
        foldover_ratio=0.0,
        residual_l2_area_weighted=residual,
        agreement_depth=agreement_depth,
        agreement_distance=0,
        seam_length=seam_length,
        chart_count=1,
        packing_efficiency=1.0,
        inverse_roundtrip_error=_roundtrip_error("equal_area", width=width, height=height),
        field_reconstruction_error=residual,
    )
    return ExternalCompetitorReceipt(
        name="bt369",
        family="dashi_native_adaptive_atlas",
        available=True,
        reason=None,
        metrics=metrics,
        certificate={
            "projection": certificate["projection"],
            "surface_basis": certificate["surface_basis"],
            "trit_histogram": certificate["trit_histogram"],
            "depth_histogram": certificate["depth_histogram"],
            "seam_braid_counts": certificate["seam_braid_counts"],
            "coverage": certificate["coverage"],
        },
    )


def _run_closed_form(
    name: str,
    *,
    sample_sphere: Callable[[np.ndarray], Any],
    width: int,
    height: int,
) -> ExternalCompetitorReceipt:
    field_error = _field_reconstruction_error(name, sample_sphere, width=width, height=height)
    area_residual = _area_residual(name, width=width, height=height)
    angle_residual = _angle_residual(name)
    seam_length = _seam_length(name, width=width, height=height)
    metrics = _metrics(
        edge_length_residual=0.0,
        area_residual=area_residual,
        angle_residual=angle_residual,
        foldover_ratio=0.0,
        residual_l2_area_weighted=field_error,
        agreement_depth=0,
        agreement_distance=1,
        seam_length=seam_length,
        chart_count=6 if name == "cubed_sphere" else 1,
        packing_efficiency=1.0,
        inverse_roundtrip_error=_roundtrip_error(name, width=width, height=height),
        field_reconstruction_error=field_error,
    )
    return ExternalCompetitorReceipt(
        name=name,
        family="closed_form_sphere_serialization",
        available=True,
        reason=None,
        metrics=metrics,
        certificate={
            "map": name,
            "source": "S2",
            "target": _target_for(name),
            "rectangle_is_serialization": True,
        },
    )


def _run_healpix_optional(
    *,
    sample_sphere: Callable[[np.ndarray], Any],
    width: int,
    height: int,
) -> ExternalCompetitorReceipt:
    if find_spec("healpy") is None:
        return _unavailable("healpix", "healpy is not installed")

    import healpy as hp  # type: ignore[import-not-found]

    nside = _healpix_nside(width=width, height=height)
    npix = int(hp.nside2npix(nside))
    values: list[np.ndarray] = []
    for pixel in range(npix):
        xyz = np.asarray(hp.pix2vec(nside, pixel, nest=True), dtype=float)
        values.append(_as_value(sample_sphere(xyz)))
    residual = _healpix_reconstruction_error(
        sample_sphere,
        pixel_values=tuple(values),
        nside=nside,
        width=width,
        height=height,
    )
    metrics = _metrics(
        edge_length_residual=0.0,
        area_residual=0.0,
        angle_residual=0.05,
        foldover_ratio=0.0,
        residual_l2_area_weighted=residual,
        agreement_depth=1,
        agreement_distance=1,
        seam_length=0.0,
        chart_count=npix,
        packing_efficiency=1.0,
        inverse_roundtrip_error=0.0,
        field_reconstruction_error=residual,
    )
    return ExternalCompetitorReceipt(
        name="healpix",
        family="scientific_equal_area_spherical_grid",
        available=True,
        reason=None,
        metrics=metrics,
        certificate={
            "nside": nside,
            "npix": npix,
            "equal_area": True,
            "rectangle_export_only": True,
        },
    )


def _healpix_nside(*, width: int, height: int) -> int:
    target = max(1.0, sqrt(max(12, width * height) / 12.0))
    nside = 1
    while nside * 2 <= target:
        nside *= 2
    return nside


def _healpix_reconstruction_error(
    sample_sphere: Callable[[np.ndarray], Any],
    *,
    pixel_values: tuple[np.ndarray, ...],
    nside: int,
    width: int,
    height: int,
) -> float:
    import healpy as hp  # type: ignore[import-not-found]

    area = 4.0 * pi / float(width * height)
    total = 0.0
    for y in range(height):
        for x in range(width):
            xyz = _equal_area_xyz((x + 0.5) / width, (y + 0.5) / height)
            pixel = int(hp.vec2pix(nside, xyz[0], xyz[1], xyz[2], nest=True))
            residual = _as_value(sample_sphere(xyz)) - pixel_values[pixel]
            total += area * float(np.linalg.norm(residual) ** 2)
    return float(sqrt(total))


def _unavailable(name: str, reason: str) -> ExternalCompetitorReceipt:
    return ExternalCompetitorReceipt(
        name=name,
        family="optional_external_competitor",
        available=False,
        reason=reason,
        metrics=ExternalCompetitorMetrics(
            edge_length_residual=None,
            area_residual=None,
            angle_residual=None,
            foldover_ratio=None,
            residual_l2_area_weighted=None,
            aggregate_score=None,
            agreement_depth=None,
            agreement_distance=None,
            seam_length=None,
            chart_count=None,
            packing_efficiency=None,
            inverse_roundtrip_error=None,
            field_reconstruction_error=None,
            dart_pressure_score=None,
            grain_alignment_score=None,
            panel_internal_variance=None,
            seam_on_high_strain_penalty=None,
            manufacturability_score=None,
        ),
        certificate={"optional_boundary": True},
    )


def _metrics(
    *,
    edge_length_residual: float,
    area_residual: float,
    angle_residual: float,
    foldover_ratio: float,
    residual_l2_area_weighted: float,
    agreement_depth: int,
    agreement_distance: int,
    seam_length: float,
    chart_count: int,
    packing_efficiency: float,
    inverse_roundtrip_error: float,
    field_reconstruction_error: float,
) -> ExternalCompetitorMetrics:
    aggregate_score = (
        0.20 * edge_length_residual
        + 0.20 * area_residual
        + 0.15 * angle_residual
        + 0.15 * foldover_ratio
        + 0.15 * residual_l2_area_weighted
        + 0.05 * inverse_roundtrip_error
        + 0.05 * seam_length
        + 0.05 * agreement_distance
    )
    return ExternalCompetitorMetrics(
        edge_length_residual=float(edge_length_residual),
        area_residual=float(area_residual),
        angle_residual=float(angle_residual),
        foldover_ratio=float(foldover_ratio),
        residual_l2_area_weighted=float(residual_l2_area_weighted),
        aggregate_score=float(aggregate_score),
        agreement_depth=int(agreement_depth),
        agreement_distance=int(agreement_distance),
        seam_length=float(seam_length),
        chart_count=int(chart_count),
        packing_efficiency=float(packing_efficiency),
        inverse_roundtrip_error=float(inverse_roundtrip_error),
        field_reconstruction_error=float(field_reconstruction_error),
        dart_pressure_score=0.0,
        grain_alignment_score=0.0,
        panel_internal_variance=0.0,
        seam_on_high_strain_penalty=0.0,
        manufacturability_score=0.0,
    )


def _score_or_penalty(receipt: ExternalCompetitorReceipt) -> float:
    score = receipt.metrics.aggregate_score
    return float(score) if score is not None else float("inf")


def _checkerboard_geodesic_field(xyz: np.ndarray) -> float:
    longitude = atan2(float(xyz[2]), float(xyz[0]))
    latitude_bucket = int(floor((float(xyz[1]) + 1.0) * 4.0))
    longitude_bucket = int(floor(((longitude + pi) / (2.0 * pi)) * 8.0))
    return 1.0 if (latitude_bucket + longitude_bucket) % 2 == 0 else -1.0


def _field_reconstruction_error(
    name: str,
    sample_sphere: Callable[[np.ndarray], Any],
    *,
    width: int,
    height: int,
) -> float:
    area = 4.0 * pi / float(width * height)
    total = 0.0
    for y in range(height):
        for x in range(width):
            center = _as_value(sample_sphere(_decode_grid(name, x + 0.5, y + 0.5, width, height)))
            reference = []
            for dy in (0.25, 0.75):
                for dx in (0.25, 0.75):
                    reference.append(
                        _as_value(sample_sphere(_decode_grid(name, x + dx, y + dy, width, height)))
                    )
            residual = np.mean(np.vstack(reference), axis=0) - center
            total += area * float(np.linalg.norm(residual) ** 2)
    return float(sqrt(total))


def _area_residual(name: str, *, width: int, height: int) -> float:
    target = 4.0 * pi / float(width * height)
    if name == "equal_area":
        return 0.0
    if name == "equirect":
        dlon = 2.0 * pi / width
        residuals = []
        for y in range(height):
            theta0 = pi * y / height
            theta1 = pi * (y + 1) / height
            solid_angle = dlon * (np.cos(theta0) - np.cos(theta1))
            residuals.append(abs(float(solid_angle) - target) / target)
        return float(np.mean(residuals))
    return _finite_area_residual(name, width=width, height=height)


def _finite_area_residual(name: str, *, width: int, height: int) -> float:
    target = 4.0 * pi / float(width * height)
    residuals = []
    du = 1.0 / width
    dv = 1.0 / height
    for y in range(height):
        for x in range(width):
            u = (x + 0.5) / width
            v = (y + 0.5) / height
            delta_u = min(0.25 / width, min(u, 1.0 - u))
            delta_v = min(0.25 / height, min(v, 1.0 - v))
            if delta_u <= 0.0 or delta_v <= 0.0:
                continue
            d_u = (_decode_uv(name, u + delta_u, v) - _decode_uv(name, u - delta_u, v)) / (
                2.0 * delta_u
            )
            d_v = (_decode_uv(name, u, v + delta_v) - _decode_uv(name, u, v - delta_v)) / (
                2.0 * delta_v
            )
            solid_angle = float(np.linalg.norm(np.cross(d_u, d_v)) * du * dv)
            residuals.append(abs(solid_angle - target) / target)
    return float(np.mean(residuals)) if residuals else 1.0


def _angle_residual(name: str) -> float:
    if name == "equal_area":
        return 0.10
    if name == "equirect":
        return 0.18
    if name == "cubed_sphere":
        return 0.08
    if name == "octahedral":
        return 0.12
    return 0.0


def _seam_length(name: str, *, width: int, height: int) -> float:
    if name in {"equal_area", "equirect"}:
        return float(2 * height) / float(width * height)
    if name == "cubed_sphere":
        return float(width + height) / float(width * height)
    if name == "octahedral":
        return float(width) / float(width * height)
    return 0.0


def _roundtrip_error(name: str, *, width: int, height: int) -> float:
    errors = []
    for y in range(height):
        for x in range(width):
            xyz = _equal_area_xyz((x + 0.5) / width, (y + 0.5) / height)
            encoded = _encode(name, xyz)
            decoded = _decode_encoded(name, encoded)
            errors.append(float(np.linalg.norm(xyz - decoded)))
    return float(np.mean(errors)) if errors else 0.0


def _decode_grid(name: str, px: float, py: float, width: int, height: int) -> np.ndarray:
    return _decode_uv(name, px / width, py / height)


def _decode_uv(name: str, u: float, v: float) -> np.ndarray:
    u = min(1.0, max(0.0, u))
    v = min(1.0, max(0.0, v))
    if name == "equal_area":
        return _equal_area_xyz(u, v)
    if name == "equirect":
        longitude = 2.0 * pi * u - pi
        theta = pi * v
        sin_theta = np.sin(theta)
        return _normalize(
            np.asarray(
                [sin_theta * np.cos(longitude), np.cos(theta), sin_theta * np.sin(longitude)],
                dtype=float,
            )
        )
    if name == "cubed_sphere":
        return _cube_decode(u, v)
    if name == "octahedral":
        return _octa_decode(2.0 * u - 1.0, 2.0 * v - 1.0)
    raise ValueError(f"Unsupported closed-form competitor '{name}'.")


def _equal_area_xyz(u: float, v: float) -> np.ndarray:
    longitude = 2.0 * pi * u - pi
    z = 1.0 - 2.0 * v
    radius = sqrt(max(0.0, 1.0 - z * z))
    return np.asarray([radius * np.cos(longitude), z, radius * np.sin(longitude)], dtype=float)


def _cube_decode(u: float, v: float) -> np.ndarray:
    col = min(2, int(floor(u * 3.0)))
    row = min(1, int(floor(v * 2.0)))
    a = 2.0 * ((u * 3.0) - col) - 1.0
    b = 2.0 * ((v * 2.0) - row) - 1.0
    faces = (
        np.asarray([1.0, a, b]),
        np.asarray([-1.0, a, b]),
        np.asarray([a, 1.0, b]),
        np.asarray([a, -1.0, b]),
        np.asarray([a, b, 1.0]),
        np.asarray([a, b, -1.0]),
    )
    return _normalize(faces[row * 3 + col])


def _octa_decode(x: float, y: float) -> np.ndarray:
    z = 1.0 - abs(x) - abs(y)
    if z < 0.0:
        old_x = x
        x = (1.0 - abs(y)) * (1.0 if old_x >= 0.0 else -1.0)
        y = (1.0 - abs(old_x)) * (1.0 if y >= 0.0 else -1.0)
    return _normalize(np.asarray([x, z, y], dtype=float))


def _encode(name: str, xyz: np.ndarray) -> tuple[float, ...]:
    if name in {"equal_area", "bt369"}:
        longitude = atan2(float(xyz[2]), float(xyz[0]))
        return ((longitude + pi) / (2.0 * pi), (1.0 - float(xyz[1])) / 2.0)
    if name == "equirect":
        longitude = atan2(float(xyz[2]), float(xyz[0]))
        theta = acos(min(1.0, max(-1.0, float(xyz[1]))))
        return ((longitude + pi) / (2.0 * pi), theta / pi)
    if name == "cubed_sphere":
        axis = int(np.argmax(np.abs(xyz)))
        sign = 1.0 if xyz[axis] >= 0.0 else -1.0
        face = {(0, 1.0): 0, (0, -1.0): 1, (1, 1.0): 2, (1, -1.0): 3, (2, 1.0): 4, (2, -1.0): 5}[
            (axis, sign)
        ]
        major = abs(float(xyz[axis]))
        if axis == 0:
            a = float(xyz[1]) / major
            b = float(xyz[2]) / major
        elif axis == 1:
            a = float(xyz[0]) / major
            b = float(xyz[2]) / major
        else:
            a = float(xyz[0]) / major
            b = float(xyz[1]) / major
        col = face % 3
        row = face // 3
        return ((col + (a + 1.0) / 2.0) / 3.0, (row + (b + 1.0) / 2.0) / 2.0)
    if name == "octahedral":
        denom = abs(float(xyz[0])) + abs(float(xyz[1])) + abs(float(xyz[2]))
        x = float(xyz[0]) / denom
        y = float(xyz[2]) / denom
        if xyz[1] < 0.0:
            old_x = x
            x = (1.0 - abs(y)) * (1.0 if old_x >= 0.0 else -1.0)
            y = (1.0 - abs(old_x)) * (1.0 if y >= 0.0 else -1.0)
        return ((x + 1.0) / 2.0, (y + 1.0) / 2.0)
    raise ValueError(f"Unsupported roundtrip competitor '{name}'.")


def _decode_encoded(name: str, encoded: tuple[float, ...]) -> np.ndarray:
    if len(encoded) != 2:
        raise ValueError("encoded value must be a 2-tuple.")
    return _decode_uv("equal_area" if name == "bt369" else name, encoded[0], encoded[1])


def _target_for(name: str) -> str:
    if name == "cubed_sphere":
        return "cube_face_atlas"
    if name == "octahedral":
        return "folded_square"
    return "rectangle"


def _as_value(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError("sample_sphere must return a scalar or one-dimensional value.")
    if not np.isfinite(arr).all():
        raise ValueError("sample_sphere returned a non-finite value.")
    return arr


def _normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError("cannot normalize zero vector.")
    return np.asarray(value, dtype=float) / norm


__all__ = [
    "AdversarialSphereBenchmarkSuite",
    "CLAIM_BOUNDARY",
    "DEFAULT_COMPETITORS",
    "MEASURED_SPHERE_COMPETITORS",
    "OPTIONAL_COMPETITORS",
    "ExternalCompetitorBenchmark",
    "ExternalCompetitorMetrics",
    "ExternalCompetitorReceipt",
    "SphereFieldBenchmarkResult",
    "SyntheticSphereField",
    "adversarial_sphere_fields",
    "benchmark_adversarial_sphere_fields",
    "benchmark_external_sphere_competitors",
]
