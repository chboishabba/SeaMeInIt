#!/usr/bin/env python3
"""Freeze a measured P2 flow-trace winner without fabricating artifact evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def freeze_winner(source: Path, candidate_id: str, out: Path) -> dict:
    raw = source.read_bytes()
    receipt = json.loads(raw)
    panels = [p for p in receipt["panels"] if p["panel_id"] == "P2"]
    if len(panels) != 1:
        raise ValueError("exactly one P2 panel required")
    candidates = [c for c in panels[0]["candidates"] if c["candidate_id"] == candidate_id]
    if len(candidates) != 1:
        raise ValueError("candidate_id must match exactly one P2 candidate")
    candidate = candidates[0]
    if candidate["candidate_family"] != "flow_trace" or candidate.get("metrics") is None:
        raise ValueError("winner must be a freshly evaluated flow_trace candidate")
    if int(candidate.get("geometric_promotion", 0)) != 1:
        raise ValueError("candidate does not have measured Gate-6 promotion")
    frozen = {
        "schema_version": "smii.bt369_frozen_p2_winner.v1",
        "source_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "source_artifact_hashes": {
            key: receipt[key] for key in
            ("panel_unwrap_receipt_hash", "mesh_hash", "seam_edges_hash")
        },
        "candidate_id": candidate_id,
        "topology": candidate["regularized_partition"],
        "topology_sha256": hashlib.sha256(
            json.dumps(candidate["regularized_partition"], separators=(",", ":")).encode()
        ).hexdigest(),
        "policy": candidate["policy"],
        "manufacturing": candidate["manufacturing"],
        "sensitivity": candidate["sensitivity"],
        "metrics": candidate["metrics"],
        "backend_competitions": candidate["backend_competitions"],
        "gate6": {
            key: candidate[key] for key in
            ("geometric_promotion", "parent_acceptance", "foldover_guard", "exportable_guard")
        },
        "claim_boundary": "geometric_gate6_only_manufacturing_and_fabrication_separate",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(frozen, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    frozen = freeze_winner(args.receipt, args.candidate_id, args.out)
    print(json.dumps({"candidate_id": frozen["candidate_id"],
                      "topology_sha256": frozen["topology_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
