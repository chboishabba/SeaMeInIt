"""Export tools for generating rigid shell fabrication bundles."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Protocol, Sequence

__all__ = [
    "HardLayerExportError",
    "HardLayerCADBackend",
    "ShellPanel",
    "HardLayerExporter",
]


class HardLayerExportError(RuntimeError):
    """Raised when hard layer export steps fail."""


class HardLayerCADBackend(Protocol):
    """Protocol describing the required CAD backend behaviour."""

    def load_panel(self, panel: "ShellPanel") -> Any:  # pragma: no cover - interface definition
        """Create an intermediate CAD model from a panel description."""

    def embed_label(self, model: Any, label: str) -> None:  # pragma: no cover - interface definition
        """Embed a fabrication label inside the CAD representation."""

    def embed_sizing_marker(self, model: Any, marker: str) -> None:  # pragma: no cover - interface
        """Add a sizing or alignment marker to the CAD representation."""

    def embed_fit_test_geometry(
        self, model: Any, geometry: Mapping[str, Any] | None
    ) -> None:  # pragma: no cover - interface
        """Add fit-test geometry primitives used for QA."""

    def export(self, model: Any, destination: Path, fmt: str) -> Path:  # pragma: no cover - interface
        """Serialize the CAD model to the provided destination with the requested format."""


@dataclass(slots=True)
class ShellPanel:
    """Description of a rigid shell panel ready for export."""

    name: str
    label: str
    mesh: Path | None = None
    size_marker: str | None = None
    fit_test_geometry: Mapping[str, Any] | None = None

    def metadata(self) -> Mapping[str, Any]:
        """Return a serialisable view of this panel."""

        payload: MutableMapping[str, Any] = {
            "name": self.name,
            "label": self.label,
        }
        if self.size_marker:
            payload["size_marker"] = self.size_marker
        if self.fit_test_geometry:
            payload["fit_test_geometry"] = self.fit_test_geometry
        if self.mesh is not None:
            payload["mesh"] = str(self.mesh)
        return payload


class _TextualCADBackend:
    """Fallback backend that stores metadata as JSON for testing and documentation."""

    def load_panel(self, panel: ShellPanel) -> MutableMapping[str, Any]:
        model: MutableMapping[str, Any] = {
            "panel": panel.name,
            "mesh": str(panel.mesh) if panel.mesh else None,
        }
        return model

    def embed_label(self, model: MutableMapping[str, Any], label: str) -> None:
        model["label"] = label

    def embed_sizing_marker(self, model: MutableMapping[str, Any], marker: str) -> None:
        if marker:
            model["size_marker"] = marker

    def embed_fit_test_geometry(
        self, model: MutableMapping[str, Any], geometry: Mapping[str, Any] | None
    ) -> None:
        if geometry:
            model["fit_test_geometry"] = dict(geometry)

    def export(self, model: MutableMapping[str, Any], destination: Path, fmt: str) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        serialised = dict(model)
        serialised["format"] = fmt.lower()
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(serialised, stream, indent=2, sort_keys=True)
        return destination


class HardLayerExporter:
    """Coordinate exports of shell panels into CAD formats."""

    def __init__(self, backend: HardLayerCADBackend | None = None) -> None:
        self.backend = backend or _TextualCADBackend()

    def export(
        self,
        panels: Sequence[ShellPanel],
        output_dir: Path | str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Export panels to STL and STEP formats and generate bundle metadata."""

        if not panels:
            msg = "At least one shell panel must be provided for export."
            raise ValueError(msg)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        seen_slugs: set[str] = set()
        bundle_metadata: MutableMapping[str, Any] = {"panels": []}
        if metadata:
            for key, value in metadata.items():
                if key == "panels":
                    msg = "Custom metadata must not define the 'panels' key."
                    raise ValueError(msg)
                bundle_metadata[key] = value

        created_paths: list[Path] = []

        for panel in panels:
            if not panel.label:
                msg = f"Panel '{panel.name}' must define a label for fabrication tracking."
                raise ValueError(msg)
            slug = _slugify(panel.name)
            if slug in seen_slugs:
                msg = f"Duplicate panel name detected after slug conversion: {panel.name!r}"
                raise ValueError(msg)
            seen_slugs.add(slug)

            try:
                model = self.backend.load_panel(panel)
                self.backend.embed_label(model, panel.label)
                if panel.size_marker:
                    self.backend.embed_sizing_marker(model, panel.size_marker)
                self.backend.embed_fit_test_geometry(model, panel.fit_test_geometry)

                stl_path = out_dir / f"{slug}.stl"
                step_path = out_dir / f"{slug}.step"

                self.backend.export(model, stl_path, "stl")
                created_paths.append(stl_path)
                self.backend.export(model, step_path, "step")
                created_paths.append(step_path)

            except Exception as exc:  # pragma: no cover - re-raise with context
                for path in created_paths:
                    if path.exists():
                        path.unlink()
                msg = f"Failed to export panel '{panel.name}'"
                raise HardLayerExportError(msg) from exc

            panel_entry: MutableMapping[str, Any] = {
                "name": panel.name,
                "label": panel.label,
                "stl": str(stl_path.relative_to(out_dir)),
                "step": str(step_path.relative_to(out_dir)),
            }
            if panel.size_marker:
                panel_entry["size_marker"] = panel.size_marker
            if panel.fit_test_geometry:
                panel_entry["fit_test_geometry"] = panel.fit_test_geometry
            if panel.mesh is not None:
                panel_entry["mesh"] = str(panel.mesh)
            bundle_metadata["panels"].append(panel_entry)

        metadata_path = out_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(bundle_metadata, stream, indent=2, sort_keys=True)

        return {
            "output_dir": out_dir,
            "panels": bundle_metadata["panels"],
            "metadata_path": metadata_path,
            "metadata": bundle_metadata,
        }


def _slugify(value: str) -> str:
    """Convert a panel name into a filesystem-friendly slug."""

    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "panel"
