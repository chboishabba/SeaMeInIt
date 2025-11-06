"""CLI for exporting hard layer shells into fabrication bundles."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from exporters.hard_layer_export import HardLayerExporter, ShellPanel


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):  # pragma: no cover - defensive validation
        msg = f"Expected JSON object at {path}"
        raise TypeError(msg)
    return payload


def _load_panel_list(path: Path) -> Sequence[ShellPanel]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Sequence):
        msg = "Panel description file must contain a JSON array."
        raise TypeError(msg)

    panels: list[ShellPanel] = []
    for entry in payload:
        if not isinstance(entry, Mapping):
            msg = "Panel entries must be JSON objects."
            raise TypeError(msg)
        name = str(entry.get("name"))
        label = str(entry.get("label", "")).strip()
        mesh = entry.get("mesh")
        size_marker = entry.get("size_marker")
        fit_test_geometry = entry.get("fit_test_geometry")
        panel = ShellPanel(
            name=name,
            label=label,
            mesh=Path(mesh) if mesh else None,
            size_marker=str(size_marker) if size_marker is not None else None,
            fit_test_geometry=fit_test_geometry if isinstance(fit_test_geometry, Mapping) else None,
        )
        panels.append(panel)
    return panels


def _copy_attachments(attachments: Iterable[Path], destination: Path) -> list[Mapping[str, str]]:
    manifest: list[Mapping[str, str]] = []
    destination.mkdir(parents=True, exist_ok=True)
    for path in attachments:
        if not path.exists():
            msg = f"Attachment {path} does not exist"
            raise FileNotFoundError(msg)
        target = destination / path.name
        if path.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target)
            manifest.append({"name": path.name, "type": "directory", "path": str(target.relative_to(destination.parent))})
        else:
            shutil.copy2(path, target)
            manifest.append({"name": path.name, "type": "file", "path": str(target.relative_to(destination.parent))})
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export segmented shell panels to a fabrication bundle.")
    parser.add_argument("--panels", type=Path, required=True, help="Path to JSON array describing shell panels.")
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional JSON metadata file merged into the export bundle metadata.",
    )
    parser.add_argument(
        "--attachments",
        type=Path,
        nargs="*",
        default=[],
        help="Optional supplementary attachment files or directories to copy alongside CAD outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exports/hard_layer"),
        help="Directory where the fabrication bundle will be written.",
    )
    args = parser.parse_args(argv)

    panels = _load_panel_list(args.panels)

    bundle_metadata: MutableMapping[str, Any] = {}
    if args.metadata:
        bundle_metadata.update(_load_json(args.metadata))

    exporter = HardLayerExporter()
    result = exporter.export(panels, output_dir=args.output, metadata=bundle_metadata)

    attachments_manifest: list[Mapping[str, str]] = []
    if args.attachments:
        attachments_manifest = _copy_attachments(args.attachments, args.output / "attachments")

    if attachments_manifest:
        metadata = result["metadata"]
        metadata["attachments"] = attachments_manifest
        with result["metadata_path"].open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)

    for panel in result["panels"]:
        print(
            "Created {label} panel with STL {stl} and STEP {step}".format(
                label=panel["label"], stl=panel["stl"], step=panel["step"]
            )
        )

    if attachments_manifest:
        print(f"Copied {len(attachments_manifest)} attachments into bundle.")

    print(f"Metadata stored at {result['metadata_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
