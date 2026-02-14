#!/usr/bin/env python3
"""Create timestamped, gitignored asset bundles under outputs/.

We don't commit binaries in this repo. This tool exists so we can
freeze artifacts between experiments without overwriting prior outputs.

Usage examples:
  python scripts/new_asset_bundle.py create --label ogre_debug
  python scripts/new_asset_bundle.py snapshot --bundle outputs/assets_bundles/20260213_120613 \
    --copy outputs/seams_run/variant_matrix_20260213_035227/shortest_path_knit_4way_light_grain100_ogre/overlay_orbit.webm

"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil


BUNDLE_SUBDIRS = [
    "notes",
    "inputs",
    "renders",
    "seams",
    "rom",
    "maps",
    "manifests",
    "snapshots",
]


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_name(s: str) -> str:
    # Make filenames stable across shells and easy to eyeball.
    keep = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def create_bundle(label: str | None) -> Path:
    ts = _utc_ts()
    name = ts if not label else f"{ts}__{_safe_name(label)}"
    root = Path("outputs") / "assets_bundles" / name
    root.mkdir(parents=True, exist_ok=False)
    for d in BUNDLE_SUBDIRS:
        (root / d).mkdir(parents=True, exist_ok=True)

    (root / "README.md").write_text(
        """# Local Asset Bundle (Gitignored)

This directory is intentionally gitignored.

## Purpose
Timestamped scratch space for artifacts (PNGs/WEBMs/NPZ/JSON) produced during seam/ROM/debug runs so we don't overwrite earlier outputs.

## Regeneration
Preferred: run the pipeline scripts and point `--out-dir` or `--bundle-root` at this folder.

## Contents
- `inputs/`: copied inputs used for the run (images, configs)
- `rom/`: ROM/native artifacts
- `seams/`: seam solve reports + serialized seam graphs
- `renders/`: orbit renders (mesh, seams, overlays)
- `maps/`: correspondence visualizations
- `snapshots/`: frozen copies of key artifacts pulled from elsewhere in `outputs/`
- `manifests/`: machine-readable run manifests
- `notes/`: human notes (what looked wrong/right, known failure modes)
""",
        encoding="utf-8",
    )

    manifest = {
        "created_utc": ts,
        "label": label,
        "bundle_root": str(root),
        "gitignore_expected": True,
        "snapshots": [],
    }
    (root / "manifests" / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


@dataclass
class SnapshotEntry:
    src: str
    dst: str
    bytes: int
    sha256: str


def snapshot_into_bundle(bundle: Path, copies: list[Path]) -> list[SnapshotEntry]:
    if not bundle.exists():
        raise FileNotFoundError(f"bundle not found: {bundle}")

    out_dir = bundle / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[SnapshotEntry] = []
    for src in copies:
        if not src.exists():
            raise FileNotFoundError(f"copy source not found: {src}")

        # Preserve a meaningful name by encoding the original relative path.
        rel = src
        try:
            rel = src.relative_to(Path.cwd())
        except ValueError:
            pass

        dst_name = _safe_name(str(rel))
        if src.is_dir():
            dst = out_dir / dst_name
            if dst.exists():
                raise FileExistsError(f"destination already exists: {dst}")
            shutil.copytree(src, dst)
            # We don't hash whole directories; store size=0 and sha256="".
            entries.append(SnapshotEntry(str(src), str(dst), 0, ""))
        else:
            # Keep extension.
            if src.suffix and not dst_name.endswith(src.suffix):
                dst_name += src.suffix
            dst = out_dir / dst_name
            shutil.copy2(src, dst)
            entries.append(
                SnapshotEntry(
                    str(src),
                    str(dst),
                    dst.stat().st_size,
                    _sha256(dst),
                )
            )

    manifest_path = bundle / "manifests" / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("snapshots", [])
    for e in entries:
        manifest["snapshots"].append(
            {"src": e.src, "dst": e.dst, "bytes": e.bytes, "sha256": e.sha256}
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return entries


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="create a new bundle")
    p_create.add_argument("--label", default=None)

    p_snap = sub.add_parser("snapshot", help="copy artifacts into an existing bundle")
    p_snap.add_argument("--bundle", required=True, type=Path)
    p_snap.add_argument("--copy", action="append", default=[], type=Path)

    args = p.parse_args()

    if args.cmd == "create":
        root = create_bundle(args.label)
        print(root)
        return 0

    if args.cmd == "snapshot":
        bundle: Path = args.bundle
        copies: list[Path] = args.copy
        if not copies:
            raise SystemExit("snapshot requires at least one --copy")
        snapshot_into_bundle(bundle, copies)
        print(bundle)
        return 0

    raise SystemExit(f"unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
