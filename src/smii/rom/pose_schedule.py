"""Pose schedule loader for ROM sampling (L0/L1/L2 procedural passes)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from smii.pipelines.fit_from_images import SMPLX_BODY_JOINTS
from smii.rom.pose_legality import LegalityConfig, LegalityResult, score_legality

__all__ = ["ScheduledSample", "load_schedule_samples"]

AXIS_VECTORS = {
    "flexion": np.array([1.0, 0.0, 0.0], dtype=float),
    "abduction": np.array([0.0, 1.0, 0.0], dtype=float),
    "rotation": np.array([0.0, 0.0, 1.0], dtype=float),
    "bend": np.array([1.0, 0.0, 0.0], dtype=float),
    "twist": np.array([0.0, 1.0, 0.0], dtype=float),
    "pitch": np.array([1.0, 0.0, 0.0], dtype=float),
    "yaw": np.array([0.0, 1.0, 0.0], dtype=float),
}

DEFAULT_AXIS_RANGES_DEG = {
    "flexion": (-45.0, 135.0),
    "abduction": (-30.0, 120.0),
    "rotation": (-60.0, 60.0),
    "bend": (-30.0, 30.0),
    "twist": (-45.0, 45.0),
    "pitch": (-20.0, 20.0),
    "yaw": (-30.0, 30.0),
}

_BILATERAL = {"shoulder", "elbow", "hip", "knee", "ankle", "wrist"}
_SPINE_EXPANSION = ("spine1", "spine2", "spine3")


@dataclass(frozen=True, slots=True)
class ScheduledSample:
    """Lightweight pose sample for schedule expansion."""

    pose_id: str
    parameters: Mapping[str, np.ndarray]
    weight: float = 1.0
    metadata: Mapping[str, Any] | None = None


def _ensure_pose_blocks(layout_blocks: Iterable[str], base_pose: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    pose: dict[str, np.ndarray] = {}
    for name in layout_blocks:
        if name in base_pose:
            pose[name] = np.asarray(base_pose[name], dtype=float).reshape(-1).copy()
        else:
            raise KeyError(f"Missing parameter block '{name}' in base pose for schedule-driven sampling.")
    return pose


def _axis_entry(entry: Any) -> tuple[str, tuple[float, float]]:
    if isinstance(entry, str):
        axis = entry
        rng = DEFAULT_AXIS_RANGES_DEG.get(axis, (-30.0, 30.0))
        return axis, (float(rng[0]), float(rng[1]))
    if isinstance(entry, Mapping):
        axis = str(entry.get("name"))
        rng = entry.get("range_deg", DEFAULT_AXIS_RANGES_DEG.get(axis, (-30.0, 30.0)))
        if not isinstance(rng, Sequence) or len(rng) != 2:
            raise ValueError(f"Axis '{axis}' must provide range_deg [min,max].")
        return axis, (float(rng[0]), float(rng[1]))
    raise TypeError("Axis entry must be a string or mapping with 'name'.")


def _axis_vector(axis: str) -> np.ndarray:
    if axis not in AXIS_VECTORS:
        raise KeyError(f"Unknown axis '{axis}' in schedule.")
    return AXIS_VECTORS[axis]


def _resolve_joint_names(joint_id: str, side: str | None) -> tuple[str, ...]:
    if joint_id == "spine":
        return _SPINE_EXPANSION
    if joint_id == "neck":
        return ("neck",)
    if joint_id in _BILATERAL:
        if side is None:
            raise ValueError(f"Bilateral joint '{joint_id}' requires a side.")
        return (f"{side}_{joint_id}",)
    return (joint_id,)


def _set_joint(body_pose: np.ndarray, *, joint_map: Mapping[str, int], joint: str, axis: str, angle_rad: float, side: str | None) -> None:
    if joint not in joint_map:
        raise KeyError(f"Joint '{joint}' is not present in the SMPL-X layout.")
    axis_vec = _axis_vector(axis)
    start = joint_map[joint]
    sign = -1.0 if side == "right" else 1.0
    body_pose[start : start + 3] = axis_vec * angle_rad * sign


def _distribute_angle(joint_names: Sequence[str], angle_rad: float) -> list[float]:
    if not joint_names:
        return []
    share = float(angle_rad) / float(len(joint_names))
    return [share for _ in joint_names]


def _linspace_deg(min_max: tuple[float, float], steps: int) -> np.ndarray:
    return np.deg2rad(np.linspace(min_max[0], min_max[1], steps, dtype=float))


def _task_controller(controller: str, t: float, *, rng: np.random.Generator) -> dict[str, float]:
    """Simple procedural controllers; angles returned in degrees."""
    # Controller outputs are in degrees to align with ranges elsewhere.
    if controller == "neutral_breathing":
        return {
            ("spine", "bend"): 3.0 * np.sin(2 * np.pi * t),
            ("spine", "twist"): 2.0 * np.sin(2 * np.pi * t + np.pi / 4),
        }
    if controller == "reach_overhead":
        return {
            ("shoulder", "flexion"): 90.0 + 50.0 * t,
            ("elbow", "flexion"): 20.0 + 20.0 * (1 - t),
        }
    if controller == "reach_forward":
        return {
            ("shoulder", "flexion"): 50.0 + 40.0 * t,
            ("elbow", "flexion"): 10.0 + 15.0 * (1 - t),
        }
    if controller == "squat_step":
        hip = 60.0 + 40.0 * t
        knee = 60.0 + 40.0 * t
        return {("hip", "flexion"): hip, ("knee", "flexion"): knee}
    if controller == "twist_reach":
        return {
            ("spine", "twist"): 20.0 + 25.0 * t,
            ("shoulder", "flexion"): 60.0 + 30.0 * t,
        }
    if controller == "gait_proxy":
        phase = 2 * np.pi * t
        hip = 25.0 * np.sin(phase)
        knee = 30.0 * np.maximum(0.0, np.sin(phase))
        return {("hip", "flexion"): hip, ("knee", "flexion"): knee}
    # Fallback: small neutral jitter to avoid zero-only samples.
    return {("spine", "bend"): rng.uniform(-2.0, 2.0), ("spine", "twist"): rng.uniform(-2.0, 2.0)}


def _apply_joint_angles(
    body_pose: np.ndarray,
    *,
    joint_map: Mapping[str, int],
    angles_deg: Mapping[tuple[str, str], float],
    side: str | None,
) -> None:
    for (joint_id, axis), angle_deg in angles_deg.items():
        targets = _resolve_joint_names(joint_id, side if joint_id in _BILATERAL else None)
        angle_rad = np.deg2rad(angle_deg)
        if targets == _SPINE_EXPANSION:
            distributed = _distribute_angle(targets, angle_rad)
            for joint, chunk_angle in zip(targets, distributed):
                _set_joint(body_pose, joint_map=joint_map, joint=joint, axis=axis, angle_rad=chunk_angle, side=side if joint_id in _BILATERAL else None)
        else:
            for joint in targets:
                _set_joint(body_pose, joint_map=joint_map, joint=joint, axis=axis, angle_rad=angle_rad, side=side if joint_id in _BILATERAL else None)


def load_schedule_samples(
    schedule_path: Path,
    *,
    layout_blocks: Sequence[str],
    joint_map: Mapping[str, int],
    base_pose: Mapping[str, np.ndarray],
    seed_override: int | None = None,
    legality_config: LegalityConfig | None = None,
    filter_illegal: bool = False,
    legality_weight: float | None = None,
) -> tuple[Mapping[str, np.ndarray], list[ScheduledSample], Mapping[str, Any]]:
    """Expand a sweep schedule into PoseSample-like entries."""

    payload = yaml.safe_load(schedule_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Sweep schedule must be a mapping.")

    meta = payload.get("meta", {}) if isinstance(payload.get("meta"), Mapping) else {}
    seed = seed_override if seed_override is not None else meta.get("seed", 13)
    rng = np.random.default_rng(seed)

    neutral_pose = _ensure_pose_blocks(layout_blocks, base_pose)
    if "body_pose" not in neutral_pose:
        raise KeyError("Schedule requires 'body_pose' in the pose layout.")

    samples: list[ScheduledSample] = []

    def add_sample(pose_id: str, pose: Mapping[str, np.ndarray], weight: float = 1.0, metadata: Mapping[str, Any] | None = None) -> None:
        pose_copy = {k: np.array(v, copy=True) for k, v in pose.items()}
        merged_meta = dict(metadata) if metadata else {}
        if legality_config is not None:
            merged_meta["_legality_config"] = legality_config
        if filter_illegal:
            merged_meta["filter_illegal"] = True
        if legality_weight is not None:
            merged_meta["legality_weight"] = float(legality_weight)
        samples.append(
            ScheduledSample(
                pose_id=pose_id,
                parameters=pose_copy,
                weight=weight,
                metadata=merged_meta if merged_meta else None,
            )
        )

    # L0 single-DOF sweeps.
    l0 = payload.get("l0", {}) if isinstance(payload.get("l0"), Mapping) else {}
    for joint_entry in l0.get("joints", []):
        joint_id = joint_entry.get("id") if isinstance(joint_entry, Mapping) else None
        if not joint_id:
            continue
        sides = joint_entry.get("sides", ["neutral"]) if isinstance(joint_entry, Mapping) else ["neutral"]
        axis_entries = joint_entry.get("axes", []) if isinstance(joint_entry, Mapping) else []
        steps = int(joint_entry.get("steps", 5)) if isinstance(joint_entry, Mapping) else 5
        for axis_entry in axis_entries:
            axis, rng_deg = _axis_entry(axis_entry)
            values = _linspace_deg(rng_deg, steps)
            for side in sides:
                for idx, angle in enumerate(values):
                    pose = _ensure_pose_blocks(layout_blocks, neutral_pose)
                    targets = _resolve_joint_names(joint_id, side if joint_id in _BILATERAL else None)
                    if targets == _SPINE_EXPANSION:
                        for joint, chunk in zip(targets, _distribute_angle(targets, angle)):
                            _set_joint(pose["body_pose"], joint_map=joint_map, joint=joint, axis=axis, angle_rad=chunk, side=side if joint_id in _BILATERAL else None)
                    else:
                        for joint in targets:
                            _set_joint(pose["body_pose"], joint_map=joint_map, joint=joint, axis=axis, angle_rad=angle, side=side if joint_id in _BILATERAL else None)
                    side_label = side if side not in (None, "neutral") else "center"
                    pose_id = f"L0-{joint_id}-{axis}-{side_label}-{idx:03d}"
                    add_sample(
                        pose_id,
                        pose,
                        metadata={
                            "level": "L0",
                            "joint": joint_id,
                            "axis": axis,
                            "side": side,
                            "angle_deg": float(np.rad2deg(angle)),
                            "active_axes": [f"{joint_id}:{axis}"],
                        },
                    )

    # L1 pairs.
    l1 = payload.get("l1", {}) if isinstance(payload.get("l1"), Mapping) else {}
    grid_steps = int(l1.get("grid_steps", 5)) if isinstance(l1, Mapping) else 5
    for pair_entry in l1.get("pairs", []):
        if not isinstance(pair_entry, Mapping):
            continue
        pair_id = pair_entry.get("id", "pair")
        sides = pair_entry.get("sides", ["neutral"])
        joints = pair_entry.get("joints", [])
        if len(joints) != 2:
            continue
        (joint_a, axis_a_raw), (joint_b, axis_b_raw) = joints
        axis_a, rng_a = _axis_entry(axis_a_raw if isinstance(axis_a_raw, Mapping) else {"name": axis_a_raw})
        axis_b, rng_b = _axis_entry(axis_b_raw if isinstance(axis_b_raw, Mapping) else {"name": axis_b_raw})
        values_a = _linspace_deg(rng_a, grid_steps)
        values_b = _linspace_deg(rng_b, grid_steps)
        for side in sides:
            for idx_a, angle_a in enumerate(values_a):
                for idx_b, angle_b in enumerate(values_b):
                    pose = _ensure_pose_blocks(layout_blocks, neutral_pose)
                    _apply_joint_angles(
                        pose["body_pose"],
                        joint_map=joint_map,
                        angles_deg={(joint_a, axis_a): np.rad2deg(angle_a), (joint_b, axis_b): np.rad2deg(angle_b)},
                        side=side if joint_a in _BILATERAL or joint_b in _BILATERAL else None,
                    )
                    side_label = side if side not in (None, "neutral") else "center"
                    pose_id = f"L1-{pair_id}-{side_label}-{idx_a:02d}-{idx_b:02d}"
                    add_sample(
                        pose_id,
                        pose,
                        metadata={
                            "level": "L1",
                            "pair": pair_id,
                            "side": side,
                            "angle_deg": {axis_a: float(np.rad2deg(angle_a)), axis_b: float(np.rad2deg(angle_b))},
                            "active_axes": [f"{joint_a}:{axis_a}", f"{joint_b}:{axis_b}"],
                        },
                    )

    # L2 tasks (controllers).
    l2 = payload.get("l2", {}) if isinstance(payload.get("l2"), Mapping) else {}
    tasks = l2.get("tasks", [])
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        task_id = task.get("id", "task")
        controller = task.get("controller", "neutral_breathing")
        sample_count = int(task.get("samples", 16))
        t_vals = np.linspace(0.0, 1.0, sample_count, dtype=float)
        for idx, t in enumerate(t_vals):
            pose = _ensure_pose_blocks(layout_blocks, neutral_pose)
            angles = _task_controller(controller, float(t), rng=rng)
            # Apply to both sides for bilateral joints to keep symmetry.
            for side in ("left", "right"):
                _apply_joint_angles(pose["body_pose"], joint_map=joint_map, angles_deg=angles, side=side)
            pose_id = f"L2-{task_id}-{idx:03d}"
            add_sample(
                pose_id,
                pose,
                metadata={
                    "level": "L2",
                    "task": task_id,
                    "controller": controller,
                    "t": float(t),
                    "active_axes": [f"{joint}:{axis}" for joint, axis in angles.keys()],
                },
            )

    meta_out = {
        "meta": meta,
        "schedule_path": str(schedule_path),
        "counts": {
            "l0": sum(1 for s in samples if s.pose_id.startswith("L0-")),
            "l1": sum(1 for s in samples if s.pose_id.startswith("L1-")),
            "l2": sum(1 for s in samples if s.pose_id.startswith("L2-")),
        },
        "seed_used": int(seed),
    }
    return neutral_pose, samples, meta_out
