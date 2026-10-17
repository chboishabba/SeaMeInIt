#!/usr/bin/env python3
"""Materialize a depth-N failure-field operator tree as exact face partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from smii.seams.operator_tree_materialization import (
    materialize_operator_tree,
    operator_from_failure_field,
)


def _load_object(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--face-count", type=int, required=True)
    parser.add_argument("--failure-field", type=Path, required=True)
    parser.add_argument(
        "--operator",
        action="append",
        required=True,
        help="operator variant id; repeat in the exact sequential order to materialize",
    )
    parser.add_argument("--source-receipt", default="serialization_failure_field")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    failure_field = _load_object(args.failure_field)
    operators = [
        operator_from_failure_field(
            failure_field,
            variant_id=variant_id,
            face_count=args.face_count,
            source_receipt=args.source_receipt,
        )
        for variant_id in args.operator
    ]
    receipt = materialize_operator_tree(
        panel_id=args.panel_id,
        face_count=args.face_count,
        operators=operators,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote operator-tree materialization receipt to {args.out} "
        f"(charts={len(receipt.final_partition)}, complete={receipt.sequential_materialization_complete})"
    )


if __name__ == "__main__":
    main()
