#!/usr/bin/env python3
"""Emit a timestamped mesh registry for provenance and labeling.

This script does not rename meshes. It records stable identifiers (hashes,
topology counts, bbox) so other pipeline stages can refer to meshes without
guessing from filenames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(arr).view(np.uint8))
    return digest.hexdigest()


def _guess_units(bbox_spans: np.ndarray) -> str:
    spans = np.asarray(bbox_spans, dtype=float).reshape(3)
    height = float(np.max(spans))
    if height < 0.1:
        return "sub-decimeter (likely cm/mm scaled)"
    if height < 0.8:
        return "sub-meter (verify scale)"
    if height < 3.0:
        return "meter-scale (likely OK)"
    return "large (>3m) (verify scale)"


def _load_mesh(path: Path) -> dict[str, Any]:
    payload = np.load(path)
    vertices = np.asarray(payload["vertices"], dtype=float)
    faces = np.asarray(payload["faces"], dtype=int) if "faces" in payload else None
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    spans = bbox_max - bbox_min
    return {
        "path": str(path),
        "sha256": _sha256_path(path),
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]) if faces is not None else None,
        "vertices_sha256": _sha256_array(vertices),
        "bbox_min": [float(v) for v in bbox_min],
        "bbox_max": [float(v) for v in bbox_max],
        "bbox_spans": [float(v) for v in spans],
        "units_guess": _guess_units(spans),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/mesh_registry"))
    parser.add_argument(
        "--alias-dir",
        type=Path,
        default=None,
        help="Optional directory to write descriptive symlinks pointing at the registered meshes.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--stage", type=str, default=None)
    parser.add_argument("meshes", nargs="+", type=Path, help="NPZ meshes to register.")
    args = parser.parse_args()

    records = []
    for mesh in args.meshes:
        record = _load_mesh(mesh)
        if args.subject is not None:
            record["subject"] = str(args.subject)
        if args.stage is not None:
            record["stage"] = str(args.stage)
        records.append(record)

    out_dir = args.out_dir / str(args.timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mesh_registry.json"
    out_path.write_text(
        json.dumps({"timestamp": str(args.timestamp), "meshes": records}, indent=2),
        encoding="utf-8",
    )
    if args.alias_dir is not None:
        alias_dir = Path(args.alias_dir)
        alias_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            src_path = Path(record["path"])
            subject = str(record.get("subject") or "na")
            stage = str(record.get("stage") or "na")
            topo = f"v{record['vertex_count']}_f{record.get('face_count') or 'na'}"
            short = str(record.get("vertices_sha256") or record.get("sha256") or "")[:10] or "na"
            alias_name = f"{subject}__{stage}__topo-{topo}__h-{short}.npz"
            link_path = alias_dir / alias_name
            if link_path.exists():
                continue
            rel = os.path.relpath(str(src_path), start=str(alias_dir))
            link_path.symlink_to(rel)
    print(out_path)


if __name__ == "__main__":
    main()
