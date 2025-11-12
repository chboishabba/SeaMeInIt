"""Download and extract SeaMeInIt body model assets.

The helper understands both the licensed SMPL-X distribution (requiring an
authenticated download URL) and open alternatives such as SMPLer-X. It handles
fetching, checksum verification, extraction, and writes an asset manifest that
other tooling can use to discover the installed bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
import zipfile

import requests

CHUNK_SIZE = 1024 * 1024  # 1 MiB


@dataclass(frozen=True)
class ModelSpec:
    """Metadata describing a downloadable asset bundle."""

    name: str
    description: str
    license: str
    default_dest: Path
    model_type: str = "smplx"
    default_url: str | None = None
    default_sha256: str | None = None
    requires_authentication: bool = False


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "smplx": ModelSpec(
        name="smplx",
        description="Official SMPL-X body model release.",
        license=(
            "SMPL-X Model License — requires registration at https://smpl-x.is.tue.mpg.de/."
        ),
        default_dest=Path("assets/smplx"),
        requires_authentication=True,
    ),
    "smplerx": ModelSpec(
        name="smplerx",
        description="SMPLer-X pretrained checkpoint bundle (open download).",
        license="S-Lab License 1.0 — redistribution for non-commercial use only.",
        default_dest=Path("assets/smplerx"),
        default_url=(
            "https://huggingface.co/caizhongang/SMPLer-X/resolve/main/smplerx_models.zip?download=1"
        ),
    ),
}


class DownloadError(RuntimeError):
    """Raised when the SMPL-X archive cannot be downloaded."""


class ExtractionError(RuntimeError):
    """Raised when the SMPL-X archive cannot be extracted."""


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY),
        default="smplx",
        help=(
            "Asset bundle to install. Defaults to 'smplx'; pass 'smplerx' to fetch the "
            "open SMPLer-X release."
        ),
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("SMPLX_DOWNLOAD_URL"),
        help=(
            "Authenticated download URL. When omitted, the tool falls back to the "
            "registry default for the selected model if one is available."
        ),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("SMPLX_AUTH_TOKEN"),
        help="Optional session token (Cookie header) for licensed downloads.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Path to a pre-downloaded archive to extract instead of fetching.",
        default=None,
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory for extracted assets (defaults to assets/<model>).",
    )
    parser.add_argument(
        "--sha256",
        default=os.environ.get("SMPLX_ARCHIVE_SHA256"),
        help="Expected archive SHA-256 checksum for validation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination directory if it already exists.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    spec = MODEL_REGISTRY[args.model]

    if args.dest is None:
        dest = spec.default_dest
    else:
        dest = args.dest

    url = args.url or spec.default_url
    sha256 = args.sha256 or spec.default_sha256

    if args.archive is None and not url:
        raise DownloadError("A download URL or --archive path must be supplied.")

    if dest.exists():
        if not args.force and any(dest.iterdir()):
            raise ExtractionError(f"Destination {dest} already contains files. Use --force to overwrite.")
        if args.force:
            shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if args.archive:
        archive_path = args.archive
    else:
        archive_path = download_archive(url, args.token)

    if sha256:
        verify_checksum(archive_path, sha256)

    extract_archive(archive_path, dest)
    manifest_path = write_manifest(
        dest,
        spec=spec,
        source_url=url,
        archive_path=archive_path if args.archive else None,
        sha256=sha256,
    )
    print(f"SMPL-X assets extracted to {dest.resolve()}")
    print(f"Wrote asset manifest to {manifest_path}")
    return 0


def download_archive(url: str, token: str | None) -> Path:
    parsed = urlparse(url)
    filename = Path(parsed.path).name or "smplx_download.bin"
    tmp_dir = Path(tempfile.mkdtemp(prefix="smplx_"))
    archive_path = tmp_dir / filename

    headers = {}
    if token:
        headers["Cookie"] = token

    print(f"Downloading {url} -> {archive_path}")
    with requests.get(url, stream=True, headers=headers, timeout=60) as response:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - network error handling
            raise DownloadError(f"Failed to download SMPL-X archive: {exc}") from exc
        with open(archive_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)
    return archive_path


def verify_checksum(path: Path, expected_sha256: str) -> None:
    expected = expected_sha256.lower()
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise DownloadError(
            f"Checksum mismatch for {path}. Expected {expected}, computed {actual}."
        )
    print(f"Checksum verified for {path.name}")


def extract_archive(archive_path: Path, dest: Path) -> None:
    """Extract *archive_path* into *dest* regardless of misleading suffixes."""

    # Some third-party mirrors publish zip archives with ``.tar`` or ``.pth.tar``
    # suffixes. Prefer inspecting the payload over trusting the filename so that
    # we can gracefully handle these mislabeled bundles.
    if zipfile.is_zipfile(archive_path):
        extract_zip(archive_path, dest)
        return

    if tarfile.is_tarfile(archive_path):
        extract_tar(archive_path, dest)
        return

    suffixes = archive_path.suffixes
    if suffixes and suffixes[-1] == ".zip":
        extract_zip(archive_path, dest)
        return
    if any(suffix in {".tar", ".tgz", ".gz", ".xz"} for suffix in suffixes):
        extract_tar(archive_path, dest)
        return

    raise ExtractionError(f"Unsupported archive format for {archive_path}")


def extract_zip(archive_path: Path, dest: Path) -> None:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(dest)
    except zipfile.BadZipFile as exc:
        raise ExtractionError(f"Failed to extract zip archive {archive_path}") from exc


def extract_tar(archive_path: Path, dest: Path) -> None:
    try:
        with tarfile.open(archive_path) as archive:
            safe_extract(archive, dest)
    except tarfile.TarError as exc:
        raise ExtractionError(f"Failed to extract tar archive {archive_path}") from exc


def safe_extract(archive: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in archive.getmembers():
        member_path = dest / member.name
        if not str(member_path.resolve()).startswith(str(dest)):
            raise ExtractionError(f"Attempted path traversal in archive: {member.name}")
    archive.extractall(dest)


def write_manifest(
    dest: Path,
    *,
    spec: ModelSpec,
    source_url: str | None,
    archive_path: Path | None,
    sha256: str | None,
) -> Path:
    """Persist metadata describing the installed asset bundle."""

    entries: list[str] = []
    try:
        for child in dest.iterdir():
            if child.name == "manifest.json":
                continue
            entries.append(str(child.relative_to(dest)))
    except FileNotFoundError:
        entries = []

    manifest = {
        "model": spec.name,
        "model_type": spec.model_type,
        "description": spec.description,
        "license": spec.license,
        "source_url": source_url,
        "source_archive": str(archive_path) if archive_path else None,
        "sha256": sha256,
        "requires_authentication": spec.requires_authentication,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "tools/download_smplx.py",
        "contents": sorted(entries),
    }

    manifest_path = dest / "manifest.json"
    dest.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main(sys.argv[1:]))
