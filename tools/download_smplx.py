"""Download and extract licensed SMPL-X assets.

The official distribution of SMPL-X requires a signed license and authenticated
session.  This helper script streamlines fetching the archive once a user has a
valid download URL or authenticated cookie.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
import zipfile

import requests

DEFAULT_DEST = Path("assets/smplx")
CHUNK_SIZE = 1024 * 1024  # 1 MiB


class DownloadError(RuntimeError):
    """Raised when the SMPL-X archive cannot be downloaded."""


class ExtractionError(RuntimeError):
    """Raised when the SMPL-X archive cannot be extracted."""


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("SMPLX_DOWNLOAD_URL"), help="Authenticated download URL")
    parser.add_argument("--token", default=os.environ.get("SMPLX_AUTH_TOKEN"), help="Optional session token (Cookie header)")
    parser.add_argument("--archive", type=Path, help="Path to a pre-downloaded SMPL-X archive", default=None)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Destination directory for extracted assets")
    parser.add_argument("--sha256", default=os.environ.get("SMPLX_ARCHIVE_SHA256"), help="Expected archive SHA-256 checksum")
    parser.add_argument("--force", action="store_true", help="Overwrite the destination directory if it already exists")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    dest = args.dest

    if args.archive is None and not args.url:
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
        archive_path = download_archive(args.url, args.token)

    if args.sha256:
        verify_checksum(archive_path, args.sha256)

    extract_archive(archive_path, dest)
    print(f"SMPL-X assets extracted to {dest.resolve()}")
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
    suffixes = archive_path.suffixes
    if suffixes[-1] == ".zip":
        extract_zip(archive_path, dest)
    elif any(suffix in {".tar", ".tgz", ".gz", ".xz"} for suffix in suffixes):
        extract_tar(archive_path, dest)
    else:
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


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main(sys.argv[1:]))
