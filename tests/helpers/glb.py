"""Utility helpers to decode GLB payloads for tests."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Tuple


def parse_glb(path: Path) -> Tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    header = struct.unpack_from("<4sII", data, 0)
    magic, version, length = header
    if magic != b"glTF" or version != 2 or length != len(data):
        raise ValueError("Invalid GLB header")
    offset = 12
    json_dict: dict[str, Any] | None = None
    bin_payload = b""
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<I4s", data, offset)
        offset += 8
        chunk_data = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == b"JSON":
            json_dict = json.loads(chunk_data.decode("utf-8"))
        elif chunk_type.startswith(b"BIN"):
            bin_payload = chunk_data
    if json_dict is None:
        raise ValueError("GLB missing JSON chunk")
    return json_dict, bin_payload

