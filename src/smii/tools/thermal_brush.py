"""Command line thermal brush for undersuit zone weighting."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping

from suit.thermal_zones import DEFAULT_THERMAL_ZONE_SPEC, ThermalZoneSpec

_BRUSH_VERSION = 1


@dataclass(slots=True)
class ThermalBrushSession:
    """Manage thermal priority weights for an undersuit specification."""

    spec: ThermalZoneSpec = DEFAULT_THERMAL_ZONE_SPEC
    weights: MutableMapping[str, float] = field(default_factory=dict)
    _zone_ids: set[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._zone_ids = {zone.identifier for zone in self.spec.zones}
        self.weights = {
            zone_id: float(value)
            for zone_id, value in self.spec.normalise_weights(dict(self.weights)).items()
        }

    def apply_brush(self, zone_id: str, weight: float) -> None:
        """Assign a new priority weight to ``zone_id``."""

        if zone_id not in self._zone_ids:
            available = ", ".join(zone.identifier for zone in self.spec.zones)
            msg = f"Unknown zone '{zone_id}'. Available zones: {available}"
            raise KeyError(msg)
        self.weights[zone_id] = max(float(weight), 0.0)

    def apply_brushes(self, entries: Mapping[str, float]) -> None:
        """Bulk apply a set of weight updates."""

        for zone_id, weight in entries.items():
            self.apply_brush(zone_id, weight)

    def normalise(self) -> None:
        """Scale weights so the maximum priority equals 1.0."""

        if not self.weights:
            return
        max_value = max(self.weights.values())
        if max_value <= 0.0:
            self.weights = {
                zone.identifier: 1.0 for zone in self.spec.zones if not zone.fallback
            }
            return
        self.weights = {zone_id: value / max_value for zone_id, value in self.weights.items()}

    def to_payload(self) -> Dict[str, object]:
        """Return a serialisable payload describing the brush weights."""

        return {
            "version": _BRUSH_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "spec": {
                "name": self.spec.name,
                "zones": [
                    {
                        "id": zone.identifier,
                        "label": zone.label,
                        "default_heat_load": zone.default_heat_load,
                    }
                    for zone in self.spec.zones
                ],
            },
            "weights": dict(self.weights),
        }

    def save(self, output_path: Path | str) -> None:
        """Persist the session payload to disk."""

        payload = self.to_payload()
        save_weights(output_path, payload)


def load_brush_payload(source: Path | str) -> Dict[str, object]:
    """Load a brush payload from disk."""

    path = Path(source)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):  # pragma: no cover - schema guard
        raise ValueError("Thermal brush payloads must be JSON objects")
    return data


def load_weights(source: Path | str) -> Dict[str, float]:
    """Load thermal weights from a persisted brush payload."""

    payload = load_brush_payload(source)
    weights = payload.get("weights", {})
    if not isinstance(weights, dict):  # pragma: no cover - schema guard
        raise ValueError("The 'weights' field must be an object of zone weights")
    return {str(key): float(value) for key, value in weights.items()}


def save_weights(destination: Path | str, payload: Mapping[str, object]) -> None:
    """Write a payload to disk as formatted JSON."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_assignments(pairs: Iterable[str]) -> Dict[str, float]:
    assignments: Dict[str, float] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Assignments must be of the form zone=weight, received: {pair}")
        zone_id, weight_text = pair.split("=", 1)
        assignments[zone_id.strip()] = float(weight_text)
    return assignments


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Path to write the updated weights")
    parser.add_argument(
        "--weights",
        type=Path,
        help="Existing weights payload to load before applying new brushes",
    )
    parser.add_argument(
        "--set",
        metavar="ZONE=WEIGHT",
        nargs="*",
        default=(),
        help="Apply one or more weight assignments (e.g. core=1.5)",
    )
    parser.add_argument(
        "--normalise",
        action="store_true",
        help="Normalise weights after applying changes so the max priority is 1.0",
    )
    parser.add_argument(
        "--list-zones",
        action="store_true",
        help="List available zones instead of writing a payload",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Entry point for the command line thermal brush."""

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    spec = DEFAULT_THERMAL_ZONE_SPEC
    initial_weights: Dict[str, float] = {}
    if args.weights:
        initial_weights = load_weights(args.weights)

    session = ThermalBrushSession(spec=spec, weights=initial_weights)

    if args.list_zones:
        for zone in spec.zones:
            print(f"{zone.identifier}: {zone.label} (target {zone.default_heat_load}W)")
        return 0

    if args.set:
        assignments = _parse_assignments(args.set)
        session.apply_brushes(assignments)

    if args.normalise:
        session.normalise()

    session.save(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
