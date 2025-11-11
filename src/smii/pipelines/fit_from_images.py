"""Utilities for deriving manual measurements from Afflec imagery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

HEADER_PREFIX = "# measurement:"


class MeasurementExtractionError(RuntimeError):
    """Raised when an Afflec image payload cannot be parsed."""


@dataclass(frozen=True)
class AfflecImageMeasurementExtractor:
    """Extract manual measurement values embedded in Afflec image headers.

    The Afflec capture pipeline produces calibrated grayscale PGM images. Each
    file stores measurement metadata within PNM comment lines prefixed with
    ``"# measurement:"`` before the raster payload. The metadata syntax follows
    ``"# measurement:<name>=<value>"`` and supports floating point values.
    """

    header_prefix: str = HEADER_PREFIX

    def parse_measurements(self, image_path: Path) -> dict[str, float]:
        """Return measurement values encoded in ``image_path``.

        Parameters
        ----------
        image_path:
            ASCII PGM (``P2``) image path written by the Afflec preprocessing
            stage.
        """

        try:
            text = image_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - defensive guard
            raise MeasurementExtractionError(
                f"Afflec image {image_path} is not UTF-8 encoded"
            ) from exc

        lines = text.splitlines()
        if not lines or not lines[0].strip().startswith("P"):
            raise MeasurementExtractionError(
                f"Afflec image {image_path} does not look like a portable graymap"
            )

        measurements: dict[str, float] = {}
        for line in lines:
            line = line.strip()
            if not line or not line.startswith(self.header_prefix):
                if line.startswith("P"):
                    # Magic numbers (e.g. P2) precede metadata and raster payloads.
                    continue
                # Comments appear before raster data. Stop parsing once headers end.
                if line and not line.startswith("#"):
                    break
                continue
            _, _, payload = line.partition(self.header_prefix)
            name, sep, value = payload.partition("=")
            if sep == "":
                raise MeasurementExtractionError(
                    f"Malformed measurement declaration in {image_path}: {line}"
                )
            measurement_name = name.strip()
            try:
                measurement_value = float(value.strip())
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise MeasurementExtractionError(
                    f"Measurement value for {measurement_name!r} is not numeric"
                ) from exc
            measurements[measurement_name] = measurement_value

        if not measurements:
            raise MeasurementExtractionError(
                f"Afflec image {image_path} does not declare any measurements"
            )
        return measurements

    def batch_extract(self, image_paths: Iterable[Path]) -> dict[str, float]:
        """Aggregate measurements from a set of Afflec images."""

        aggregated: dict[str, float] = {}
        for path in image_paths:
            parsed = self.parse_measurements(path)
            for key, value in parsed.items():
                if key in aggregated and aggregated[key] != value:
                    raise MeasurementExtractionError(
                        f"Conflicting values for {key!r}: {aggregated[key]} vs {value}"
                    )
                aggregated[key] = value
        return aggregated


def extract_measurements_from_afflec_images(image_paths: Iterable[Path]) -> dict[str, float]:
    """Convenience wrapper used by the CLI pipeline."""

    extractor = AfflecImageMeasurementExtractor()
    return extractor.batch_extract(image_paths)


__all__ = [
    "AfflecImageMeasurementExtractor",
    "MeasurementExtractionError",
    "extract_measurements_from_afflec_images",
]
