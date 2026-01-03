"""CLI for exporting undersuit patterns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from exporters.patterns import PatternExporter


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):  # pragma: no cover - defensive validation
        msg = "Pattern inputs must be JSON objects."
        raise TypeError(msg)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export undersuit panels as 2D pattern files.")
    parser.add_argument(
        "--mesh", type=Path, required=True, help="Path to undersuit panel mesh JSON"
    )
    parser.add_argument("--seams", type=Path, required=True, help="Path to seam annotation JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exports/patterns"),
        help="Directory where exported pattern files will be stored.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "svg", "dxf"],
        help="One or more formats to export (default: pdf svg dxf).",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Scale factor applied during flattening."
    )
    parser.add_argument(
        "--seam-allowance",
        type=float,
        default=0.01,
        help="Default seam allowance applied to panels (meters).",
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Optional identifier stored inside exported metadata.",
    )
    parser.add_argument(
        "--backend",
        choices=("simple", "lscm"),
        default="simple",
        help="Flattening backend used when generating 2D panels.",
    )
    parser.add_argument(
        "--annotate-level",
        choices=("off", "summary", "full"),
        default="summary",
        help="SVG annotation level for issue overlays (default: summary).",
    )
    args = parser.parse_args(argv)

    mesh_payload = _load_json(args.mesh)
    seams_payload = _load_json(args.seams)

    exporter = PatternExporter(
        backend=args.backend,
        scale=args.scale,
        seam_allowance=args.seam_allowance,
    )
    metadata: dict[str, Any] = {}
    if args.label:
        metadata["label"] = args.label

    created = exporter.export(
        mesh_payload,
        seams_payload,
        output_dir=args.output,
        formats=args.formats,
        metadata=metadata,
        annotate_level=args.annotate_level,
    )

    for fmt, path in created.items():
        print(f"Wrote {fmt.upper()} pattern to {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
