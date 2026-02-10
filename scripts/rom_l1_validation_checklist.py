#!/usr/bin/env python3
"""
ROM L1 validation checklist:
- Confirms runtime legality + chain weighting are present
- Computes compression signals: repeated violation signatures, worst-pair concentration
- Checks filtering/downweighting behavior
- Optionally summarizes flex_stats.csv if provided

Usage:
  python scripts/rom_l1_validation_checklist.py \
    --rom-meta outputs/rom/afflec_rom_run.json \
    --flex-stats outputs/seams_run/flex_heatmap/flex_stats.csv \
    --topk 15
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Tuple

import numpy as np


def _safe_get(d: Mapping[str, Any], keys: Iterable[str], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _try_load_csv(path: Path) -> Optional[list[dict[str, str]]]:
    import csv

    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


@dataclass(frozen=True)
class Signature:
    """A coarse 'hypervoxel signature' for L1: legality + active axes bucket."""

    worst_pair: str
    violations_key: Tuple[str, ...]
    active_axes_key: Tuple[str, ...]

    def key(self) -> tuple:
        return (self.worst_pair, self.violations_key, self.active_axes_key)


def _normalize_pair(pair: Any) -> str:
    # worst_pair may be None, list, tuple, "A|B", etc.
    if pair is None:
        return "none"
    if isinstance(pair, str):
        return pair
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        return f"{pair[0]}|{pair[1]}"
    return str(pair)


def _bucket_active_axes(active_axes: Any) -> Tuple[str, ...]:
    if not active_axes:
        return tuple()
    if isinstance(active_axes, str):
        return (active_axes,)
    if isinstance(active_axes, (list, tuple)):
        # keep order stable for signature: sort to ignore enumeration ordering
        return tuple(sorted(str(x) for x in active_axes))
    return (str(active_axes),)


def _bucket_violations(violations: Any) -> Tuple[str, ...]:
    # violations might be list of strings or dicts; keep stable coarse key
    if not violations:
        return tuple()
    if isinstance(violations, (list, tuple)):
        out = []
        for v in violations:
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, Mapping):
                # prefer a 'type' field; otherwise stringify
                out.append(str(v.get("type", v)))
            else:
                out.append(str(v))
        return tuple(sorted(out))
    if isinstance(violations, Mapping):
        return tuple(sorted(str(k) for k in violations.keys()))
    return (str(violations),)


def _extract_pose_records(rom_meta: Mapping[str, Any]) -> tuple[list[str], list[Mapping[str, Any]]]:
    """
    Tries to find pose_ids and pose_metadata in common layouts.
    """
    # Your sampler likely writes something like:
    # meta["pose_ids"] and meta["pose_metadata"]
    pose_ids = _safe_get(rom_meta, ["pose_ids"], None) or _safe_get(rom_meta, ["poses", "pose_ids"], None)

    # Fallbacks for pose metadata; keep going until we find a list
    pose_md = None
    for keys in (
        ["pose_metadata"],
        ["poses", "pose_metadata"],
        ["pose_meta"],
        ["pose_stats"],
    ):
        candidate = _safe_get(rom_meta, keys, None)
        if isinstance(candidate, list):
            pose_md = candidate
            break

    if not isinstance(pose_ids, list) or not isinstance(pose_md, list):
        raise ValueError(
            "Could not locate pose_ids/pose_metadata in rom meta JSON. "
            "Expected keys: pose_ids + pose_metadata (or poses.pose_ids / poses.pose_metadata)."
        )
    # Ensure equal length; if not, truncate to min but warn.
    n = min(len(pose_ids), len(pose_md))
    return [str(x) for x in pose_ids[:n]], [pose_md[i] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom-meta", type=Path, required=True, help="Sampler out-meta/rom-meta JSON")
    ap.add_argument("--flex-stats", type=Path, default=None, help="Optional flex_stats.csv from seam diagnostic run")
    ap.add_argument("--topk", type=int, default=12, help="Top-K items to print for counters")
    ap.add_argument(
        "--sig-mode",
        choices=["l1", "strict"],
        default="l1",
        help="Signature strictness. l1=worst_pair+violations+active_axes; strict=also includes score bucket.",
    )
    args = ap.parse_args()

    meta = _load_json(args.rom_meta)
    pose_ids, pose_md = _extract_pose_records(meta)

    # --- Checklist flags
    has_runtime_legality = 0
    has_chain_factor = 0
    has_active_axes = 0
    filtered_count = 0

    # --- Counters for compression signals
    worst_pair_ctr = Counter()
    violation_ctr = Counter()
    signature_ctr = Counter()
    score_vals = []

    # Track by pose id for quick debugging
    examples_by_pair: dict[str, list[str]] = defaultdict(list)

    for pid, md in zip(pose_ids, pose_md):
        if not isinstance(md, Mapping):
            continue

        active_axes = md.get("active_axes")
        if active_axes:
            has_active_axes += 1

        cf = md.get("chain_factor")
        if cf is not None:
            has_chain_factor += 1

        lr = md.get("legality_runtime")
        if isinstance(lr, Mapping):
            has_runtime_legality += 1
            score = lr.get("score", 0.0)
            try:
                score_f = float(score)
                score_vals.append(score_f)
            except Exception:
                pass

            wp = _normalize_pair(lr.get("worst_pair"))
            worst_pair_ctr[wp] += 1
            if len(examples_by_pair[wp]) < 5:
                examples_by_pair[wp].append(pid)

            viol_key = _bucket_violations(lr.get("violations"))
            for v in viol_key:
                violation_ctr[v] += 1

            sig = Signature(
                worst_pair=wp,
                violations_key=viol_key,
                active_axes_key=_bucket_active_axes(active_axes),
            ).key()

            if args.sig_mode == "strict":
                # Add a coarse score bucket to distinguish near-boundary vs hard collisions
                # (Buckets: 0, (0,1], (1,2], (2,5], >5)
                b = 0
                try:
                    s = float(score)
                    if s <= 0:
                        b = 0
                    elif s <= 1:
                        b = 1
                    elif s <= 2:
                        b = 2
                    elif s <= 5:
                        b = 3
                    else:
                        b = 4
                except Exception:
                    b = -1
                sig = (*sig, b)

            signature_ctr[sig] += 1

        # filtering happens in sampler by skipping poses; we can detect intent flags:
        if md.get("filter_illegal"):
            filtered_count += 1

    n = len(pose_ids)

    print("\n=== L1 VALIDATION CHECKLIST ===")
    print(f"ROM meta: {args.rom_meta}")
    print(f"Poses recorded: {n}")

    # Presence checks
    print("\n[Presence]")
    print(f"poses with active_axes metadata: {has_active_axes}/{n} ({has_active_axes/n:.1%})")
    print(f"poses with chain_factor recorded: {has_chain_factor}/{n} ({has_chain_factor/n:.1%})")
    print(f"poses with legality_runtime recorded: {has_runtime_legality}/{n} ({has_runtime_legality/n:.1%})")
    print(f"poses carrying filter_illegal flag (intent): {filtered_count}/{n} ({filtered_count/n:.1%})")

    # Chain factor sanity
    chain_factors = [
        float(md.get("chain_factor")) for md in pose_md if isinstance(md, Mapping) and md.get("chain_factor") is not None
    ]
    if chain_factors:
        cf_arr = np.asarray(chain_factors, dtype=float)
        print("\n[Chain factor]")
        print(f"chain_factor min/median/max: {cf_arr.min():.3f} / {np.median(cf_arr):.3f} / {cf_arr.max():.3f}")
        print("Expected: >1.0 for multi-axis samples; ~=1.0 for single-axis samples.")

    # Legality distribution
    if score_vals:
        s = np.asarray(score_vals, dtype=float)
        nz = float(np.mean(s > 0))
        print("\n[Legality score]")
        print(f"score min/median/max: {s.min():.3f} / {np.median(s):.3f} / {s.max():.3f}")
        print(f"nonzero score rate: {nz:.1%}")
        print("Expected: some nonzero scores once chain/collision activates; not necessarily huge at L1.")

    # Compression / repetition signals
    if signature_ctr:
        uniq = len(signature_ctr)
        top = signature_ctr.most_common(args.topk)
        top_mass = sum(c for _, c in top) / max(1, sum(signature_ctr.values()))
        print("\n[Compression signals]")
        print(f"unique signatures: {uniq} out of {n} poses ({uniq/n:.1%} of pose count)")
        print(f"top-{args.topk} signatures mass: {top_mass:.1%}")
        print("Healthy L1: unique signatures significantly smaller than pose count; top mass not tiny.")

    if worst_pair_ctr:
        print("\n[Worst collision pairs] (top)")
        for pair, c in worst_pair_ctr.most_common(args.topk):
            ex = ", ".join(examples_by_pair[pair][:3])
            print(f"  {pair:30s}  {c:6d}  e.g. {ex}")

    if violation_ctr:
        print("\n[Violation types] (top)")
        for v, c in violation_ctr.most_common(args.topk):
            print(f"  {v:30s}  {c:6d}")

    # Optional flex stats summary
    if args.flex_stats:
        rows = _try_load_csv(args.flex_stats)
        if rows:
            # Heuristic: look for columns that resemble "mean" or "max"
            cols = set(rows[0].keys())
            numeric_cols = [c for c in cols if any(k in c.lower() for k in ("mean", "max", "sum", "score", "cost"))]
            print("\n[Flex stats CSV]")
            print(f"rows: {len(rows)}, numeric-ish cols: {sorted(numeric_cols)[:8]}{'...' if len(numeric_cols)>8 else ''}")
            # If there's a 'vertex_cost_max' or similar, print top rows
            candidate = None
            for c in numeric_cols:
                if "max" in c.lower():
                    candidate = c
                    break
            if candidate:
                def _tofloat(x: str) -> float:
                    try:
                        return float(x)
                    except Exception:
                        return float("nan")
                rows_sorted = sorted(rows, key=lambda r: _tofloat(r.get(candidate, "")), reverse=True)
                print(f"Top {min(args.topk, len(rows_sorted))} by {candidate}:")
                for r in rows_sorted[: args.topk]:
                    # Try to print a pose id if present
                    pid = r.get("pose_id") or r.get("pose") or r.get("id") or ""
                    print(f"  {pid:20s} {candidate}={r.get(candidate)}")
        else:
            print("\n[Flex stats CSV] not found or empty:", args.flex_stats)

    print("\n=== INTERPRETATION QUICK GUIDE ===")
    print("- If legality_runtime is missing on most poses: you are not saving enriched metadata or not scoring at runtime.")
    print("- If unique signatures ~= pose count: your legality/chain fields aren’t producing repeated boundary patterns yet.")
    print("- If worst_pair is dominated by 'none': collisions aren’t activating; increase task amplitude or adjust radii/alpha.")
    print("- If chain_factor is always 1.0: active_axes metadata isn’t flowing or chain amplification isn’t applied.")


if __name__ == "__main__":
    main()
