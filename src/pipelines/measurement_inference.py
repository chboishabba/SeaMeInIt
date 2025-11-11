"""Measurement inference utilities for completing manual measurement sets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import csv
import math

import numpy as np

__all__ = [
    "DEFAULT_DATASET_PATH",
    "MeasurementEstimate",
    "MeasurementReport",
    "GaussianMeasurementModel",
    "load_measurement_samples",
    "load_default_model",
]

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "templates" / "manual_measurement_samples.csv"


@dataclass(frozen=True)
class MeasurementEstimate:
    """Represents a single measurement value and its provenance."""

    name: str
    value: float
    source: str
    confidence: float
    variance: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "value": float(self.value),
            "source": self.source,
            "confidence": float(self.confidence),
            "variance": float(self.variance),
        }

    @property
    def is_inferred(self) -> bool:
        return self.source != "measured"


@dataclass(frozen=True)
class MeasurementReport:
    """Holds the completed measurement vector with metadata."""

    estimates: tuple[MeasurementEstimate, ...]
    coverage: float

    def values(self) -> dict[str, float]:
        return {estimate.name: float(estimate.value) for estimate in self.estimates}

    def measured(self) -> tuple[MeasurementEstimate, ...]:
        return tuple(estimate for estimate in self.estimates if not estimate.is_inferred)

    def inferred(self) -> tuple[MeasurementEstimate, ...]:
        return tuple(estimate for estimate in self.estimates if estimate.is_inferred)

    def visualization_payload(self) -> list[dict[str, float | str]]:
        """Return a serialisable view highlighting measured vs inferred values."""

        return [estimate.as_dict() for estimate in self.estimates]


@dataclass(slots=True)
class GaussianMeasurementModel:
    """Multivariate Gaussian model over manual measurements.

    The model supports conditioning on a subset of observed measurements to infer
    missing values together with confidence intervals derived from the
    conditional covariance.
    """

    names: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray
    regularization: float = 1e-6

    @classmethod
    def from_samples(
        cls,
        names: Sequence[str],
        samples: np.ndarray,
        *,
        regularization: float = 1e-6,
    ) -> "GaussianMeasurementModel":
        if samples.ndim != 2:
            raise ValueError("Samples must be a 2D matrix of shape (n_samples, n_measurements).")
        if len(names) != samples.shape[1]:
            raise ValueError("Number of measurement names must match sample columns.")

        mean = np.mean(samples, axis=0)
        covariance = np.cov(samples, rowvar=False)
        covariance = _ensure_positive_semidefinite(covariance, regularization)
        return cls(tuple(names), mean, covariance, regularization)

    def infer(self, provided: Mapping[str, float]) -> MeasurementReport:
        """Infer missing measurements given a partial observation mapping."""

        provided = {name: float(value) for name, value in provided.items()}
        observed_idx = [index for index, name in enumerate(self.names) if name in provided]
        missing_idx = [index for index, name in enumerate(self.names) if name not in provided]

        if not observed_idx:
            cond_mean = self.mean[missing_idx]
            cond_cov = self.covariance[np.ix_(missing_idx, missing_idx)]
        else:
            sigma_oo = self.covariance[np.ix_(observed_idx, observed_idx)].copy()
            sigma_mo = self.covariance[np.ix_(missing_idx, observed_idx)].copy()
            sigma_om = self.covariance[np.ix_(observed_idx, missing_idx)].copy()

            sigma_oo += np.eye(len(observed_idx)) * self.regularization
            delta = np.array([provided[self.names[index]] for index in observed_idx]) - self.mean[observed_idx]
            solve = np.linalg.solve(sigma_oo, delta)
            cond_mean = self.mean[missing_idx] + sigma_mo @ solve
            cond_cov = self.covariance[np.ix_(missing_idx, missing_idx)] - sigma_mo @ np.linalg.solve(
                sigma_oo, sigma_om
            )
            cond_cov = _symmetrise(cond_cov)

        baseline_var = np.diag(self.covariance)
        estimates: list[MeasurementEstimate] = []

        missing_lookup = {index: pos for pos, index in enumerate(missing_idx)}
        for index, name in enumerate(self.names):
            if name in provided:
                estimates.append(
                    MeasurementEstimate(
                        name=name,
                        value=provided[name],
                        source="measured",
                        confidence=1.0,
                        variance=0.0,
                    )
                )
            else:
                pos = missing_lookup[index]
                value = float(cond_mean[pos])
                variance = float(max(cond_cov[pos, pos], 0.0)) if cond_cov.size else float(baseline_var[index])
                confidence = _confidence_from_variance(variance, float(baseline_var[index]))
                estimates.append(
                    MeasurementEstimate(
                        name=name,
                        value=value,
                        source="inferred",
                        confidence=confidence,
                        variance=variance,
                    )
                )

        coverage = len(observed_idx) / len(self.names) if self.names else 0.0
        return MeasurementReport(tuple(estimates), coverage)


def load_measurement_samples(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Load manual measurement samples from a CSV dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Measurement dataset not found at {path}")

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV dataset must define header rows.")
        names = tuple(field for field in reader.fieldnames if field != "sample_id")
        rows: list[list[float]] = []
        for row in reader:
            rows.append([float(row[name]) for name in names])

    if not rows:
        raise ValueError("Measurement dataset is empty.")

    samples = np.asarray(rows, dtype=float)
    return names, samples


def load_default_model() -> GaussianMeasurementModel:
    """Load the repository's canonical measurement inference model."""

    names, samples = load_measurement_samples(DEFAULT_DATASET_PATH)
    return GaussianMeasurementModel.from_samples(names, samples)


def _ensure_positive_semidefinite(matrix: np.ndarray, regularization: float) -> np.ndarray:
    sym = _symmetrise(matrix)
    eigenvalues = np.linalg.eigvalsh(sym)
    min_eig = np.min(eigenvalues)
    if min_eig < 0:
        sym += np.eye(sym.shape[0]) * (abs(min_eig) + regularization)
    else:
        sym += np.eye(sym.shape[0]) * regularization
    return sym


def _symmetrise(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2.0


def _confidence_from_variance(variance: float, baseline: float) -> float:
    if not math.isfinite(variance) or variance < 0:
        return 0.0
    baseline = max(baseline, 1e-9)
    std_ratio = math.sqrt(variance) / math.sqrt(baseline)
    confidence = 1.0 - min(1.0, std_ratio)
    return max(0.0, min(1.0, confidence))
