"""ROM aggregation helpers for Sprint 2."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from .basis import ArrayLike, KernelProjector
from .gates import GateReason, RomGate


@dataclass(frozen=True, slots=True)
class RomSample:
    """ROM sample containing coefficient vectors for one pose/derivation."""

    pose_id: str
    coeffs: Mapping[str, ArrayLike]
    observations: Mapping[str, float | bool] | None = None


@dataclass(frozen=True, slots=True)
class FieldStats:
    """Aggregated statistics for a projected field."""

    mean: np.ndarray
    maximum: np.ndarray
    variance: np.ndarray
    sample_count: int


@dataclass(frozen=True, slots=True)
class VertexHotspot:
    """Top-variance vertex with max value captured for diagnostics."""

    index: int
    variance: float
    maximum: float


@dataclass(frozen=True, slots=True)
class EdgeHotspot:
    """Top-variance edge with max value captured for diagnostics."""

    edge: tuple[int, int]
    variance: float
    maximum: float


@dataclass(frozen=True, slots=True)
class AggregationDiagnostics:
    """Hotspot diagnostics per field."""

    vertex_hotspots: tuple[VertexHotspot, ...]
    edge_hotspots: tuple[EdgeHotspot, ...]


@dataclass(frozen=True, slots=True)
class RejectionReport:
    """Summary of gate rejections encountered during aggregation."""

    total_samples: int
    accepted_samples: int
    rejected_samples: int
    rejection_rate: float
    reasons: tuple["RejectionReason", ...]


@dataclass(frozen=True, slots=True)
class RejectionReason:
    """Specific rejection reason emitted by a PDA gate."""

    id: str
    description: str
    severity: str
    count: int


@dataclass(frozen=True, slots=True)
class RomAggregation:
    """Aggregation result across one or more fields."""

    per_field: Mapping[str, FieldStats]
    per_edge_field: Mapping[str, FieldStats]
    diagnostics: Mapping[str, AggregationDiagnostics]
    sample_count: int
    total_samples: int
    rejection_report: RejectionReport
    edges: Sequence[tuple[int, int]] | None


def _init_accumulators(count: int, field_names: Sequence[str]) -> MutableMapping[str, dict[str, np.ndarray]]:
    accumulators: MutableMapping[str, dict[str, np.ndarray]] = {}
    for name in field_names:
        accumulators[name] = {
            "sum": np.zeros(count, dtype=float),
            "sumsq": np.zeros(count, dtype=float),
            "max": np.full(count, -np.inf, dtype=float),
            "count": np.zeros(1, dtype=int),
        }
    return accumulators


def _finalize_stats(accumulators: Mapping[str, dict[str, np.ndarray]]) -> Mapping[str, FieldStats]:
    finalized: MutableMapping[str, FieldStats] = {}
    for name, acc in accumulators.items():
        present_count = int(acc["count"][0])
        if present_count == 0:
            shape = acc["sum"].shape
            nan_field = np.full(shape, np.nan, dtype=float)
            finalized[name] = FieldStats(
                mean=nan_field,
                maximum=nan_field.copy(),
                variance=nan_field.copy(),
                sample_count=present_count,
            )
            continue
        mean = acc["sum"] / present_count
        variance = np.maximum(acc["sumsq"] / present_count - mean**2, 0.0)
        finalized[name] = FieldStats(
            mean=mean,
            maximum=acc["max"],
            variance=variance,
            sample_count=present_count,
        )
    return finalized


def _compute_hotspots(
    vertex_stats: Mapping[str, FieldStats],
    edge_stats: Mapping[str, FieldStats],
    *,
    edges: Sequence[tuple[int, int]] | None,
    top_k: int,
) -> Mapping[str, AggregationDiagnostics]:
    diagnostics: MutableMapping[str, AggregationDiagnostics] = {}
    for name, stats in vertex_stats.items():
        if stats.sample_count == 0:
            diagnostics[name] = AggregationDiagnostics(vertex_hotspots=tuple(), edge_hotspots=tuple())
            continue
        vertex_order = np.argsort(stats.variance)[::-1]
        vertex_hotspots = tuple(
            VertexHotspot(
                index=int(idx),
                variance=float(stats.variance[idx]),
                maximum=float(stats.maximum[idx]),
            )
            for idx in vertex_order[:top_k]
        )

        edge_hotspots: tuple[EdgeHotspot, ...] = tuple()
        if edges and name in edge_stats and edge_stats[name].sample_count > 0:
            edge_stat = edge_stats[name]
            edge_order = np.argsort(edge_stat.variance)[::-1]
            edge_hotspots = tuple(
                EdgeHotspot(
                    edge=edges[int(idx)],
                    variance=float(edge_stat.variance[idx]),
                    maximum=float(edge_stat.maximum[idx]),
                )
                for idx in edge_order[:top_k]
                if idx < len(edges)
            )

        diagnostics[name] = AggregationDiagnostics(
            vertex_hotspots=vertex_hotspots,
            edge_hotspots=edge_hotspots,
        )
    return diagnostics


def aggregate_fields(
    samples: Iterable[RomSample],
    projector: KernelProjector,
    *,
    field_keys: Sequence[str] | None = None,
    optional_fields: Sequence[str] | None = None,
    include_observed_fields: bool = True,
    edges: Sequence[tuple[int, int]] | None = None,
    gate: RomGate | None = None,
    diagnostics_top_k: int = 5,
) -> RomAggregation:
    """Aggregate per-vertex and per-edge statistics for ROM samples.

    Args:
        samples: Iterable of ROM samples; gating applied if a RomGate is provided.
        projector: Kernel projector built on the canonical basis.
        field_keys: Fields that must exist on every accepted sample (required). If omitted,
            fields are discovered from accepted samples.
        optional_fields: Fields that should be aggregated when present but may be missing
            on accepted samples without triggering an error.
        include_observed_fields: If True (default), any additional fields observed on
            accepted samples are aggregated as observed-only diagnostics.
        edges: Optional sequence of edges defined by vertex id pairs; when provided,
            edge-wise magnitudes are aggregated alongside vertices.
        gate: Optional PDA gate to accept or reject samples.
        diagnostics_top_k: Number of hotspots to surface per field.
    """

    iterator = list(samples)
    if not iterator:
        raise ValueError("At least one ROM sample is required for aggregation.")

    required_fields = set(field_keys or [])
    optional = set(optional_fields or [])

    vertex_acc = _init_accumulators(projector.vertex_count, sorted(required_fields | optional))
    edge_count = len(edges) if edges else 0
    edge_acc = _init_accumulators(edge_count, sorted(required_fields | optional)) if edge_count else {}
    edge_array = np.asarray(edges, dtype=int) if edges else None

    def ensure_field(name: str) -> None:
        if name in vertex_acc:
            return
        vertex_acc[name] = {
            "sum": np.zeros(projector.vertex_count, dtype=float),
            "sumsq": np.zeros(projector.vertex_count, dtype=float),
            "max": np.full(projector.vertex_count, -np.inf, dtype=float),
            "count": np.zeros(1, dtype=int),
        }
        if edges:
            assert edge_array is not None
            edge_acc[name] = {
                "sum": np.zeros(edge_count, dtype=float),
                "sumsq": np.zeros(edge_count, dtype=float),
                "max": np.full(edge_count, -np.inf, dtype=float),
                "count": np.zeros(1, dtype=int),
            }

    total_samples = len(iterator)
    accepted_samples = 0
    rejection_reasons: Counter[str] = Counter()
    rejection_details: dict[str, GateReason] = {}

    for sample in iterator:
        if gate is not None:
            decision = gate.evaluate(sample.observations or {})
            if not decision.accepted:
                for reason in decision.reasons:
                    rejection_reasons.update([reason.id])
                    rejection_details.setdefault(reason.id, reason)
                continue

        missing_required = [field for field in required_fields if field not in sample.coeffs]
        if missing_required:
            raise KeyError(f"Sample '{sample.pose_id}' missing required fields: {missing_required}.")

        field_items = sample.coeffs.items()
        if not include_observed_fields:
            field_items = ((k, v) for k, v in sample.coeffs.items() if k in required_fields or k in optional)

        for field_name, coeffs in field_items:
            if field_name not in required_fields and field_name not in optional and not include_observed_fields:
                continue
            ensure_field(field_name)
            field_values = projector.project(coeffs)
            acc = vertex_acc[field_name]
            acc["sum"] += field_values
            acc["sumsq"] += field_values**2
            acc["max"] = np.maximum(acc["max"], field_values)
            acc["count"][0] += 1

            if edges:
                assert edge_array is not None
                edge_values = np.abs(field_values[edge_array[:, 0]] - field_values[edge_array[:, 1]])
                edge_store = edge_acc[field_name]
                edge_store["sum"] += edge_values
                edge_store["sumsq"] += edge_values**2
                edge_store["max"] = np.maximum(edge_store["max"], edge_values)
                edge_store["count"][0] += 1

        accepted_samples += 1

    for optional_field in optional:
        ensure_field(optional_field)

    if accepted_samples == 0:
        raise ValueError("No ROM samples were accepted after gating.")

    per_field = _finalize_stats(vertex_acc)
    per_edge_field = _finalize_stats(edge_acc) if edges else {}
    diagnostics = _compute_hotspots(per_field, per_edge_field, edges=edges, top_k=diagnostics_top_k)

    rejection_reason_summaries = tuple(
        RejectionReason(
            id=reason_id,
            description=rejection_details.get(reason_id, GateReason(reason_id, "", "unknown", True)).description,
            severity=rejection_details.get(reason_id, GateReason(reason_id, "", "unknown", True)).severity,
            count=count,
        )
        for reason_id, count in sorted(rejection_reasons.items())
    )

    rejection_report = RejectionReport(
        total_samples=total_samples,
        accepted_samples=accepted_samples,
        rejected_samples=total_samples - accepted_samples,
        rejection_rate=float(total_samples - accepted_samples) / float(total_samples),
        reasons=rejection_reason_summaries,
    )

    return RomAggregation(
        per_field=per_field,
        per_edge_field=per_edge_field,
        diagnostics=diagnostics,
        sample_count=accepted_samples,
        total_samples=total_samples,
        rejection_report=rejection_report,
        edges=tuple(edges) if edges else None,
    )
