"""Finite-difference ROM sampler that streams seam costs without large tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, MutableSequence, Sequence

import numpy as np

from smii.meshing import load_body_record
from smii.pipelines.fit_from_images import SMPLX_BODY_JOINTS
from smii.rom.basis import KernelProjector, load_basis
from smii.rom.completeness import build_envelope, compare_envelopes, spearman_rank
from smii.rom.pose_legality import LegalityConfig
from smii.rom.seam_costs import SeamCostField, save_seam_cost_field


POSE_PARAMS_ORDER: tuple[str, ...] = (
    "global_orient",
    "body_pose",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "transl",
)


def _load_smplx_parameter_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], float, str, str]:
    """Decode serialized SMPL-X parameters without importing full fit pipeline deps."""

    scale = float(payload.get("scale", 1.0))
    model_type = str(payload.get("model_type", "smplx"))
    gender = str(payload.get("gender", "neutral"))

    param_keys_raw = payload.get("parameters")
    if param_keys_raw is None:
        param_keys = [
            key
            for key in payload.keys()
            if key not in {"model_type", "gender", "scale", "parameters"}
        ]
    elif isinstance(param_keys_raw, Sequence) and not isinstance(param_keys_raw, (str, bytes)):
        param_keys = [str(name) for name in param_keys_raw]
    else:
        raise TypeError("'parameters' field must be a sequence of parameter names if provided.")

    decoded: dict[str, np.ndarray] = {}
    for name in param_keys:
        if name not in payload:
            raise KeyError(f"Parameter '{name}' is listed but missing from the payload.")
        array = np.asarray(payload[name], dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        elif array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim > 2:
            array = array.reshape(array.shape[0], -1)
        decoded[name] = array
    return decoded, scale, model_type, gender


@dataclass(frozen=True, slots=True)
class PoseSample:
    """Single pose entry with measure weight."""

    pose_id: str
    parameters: Mapping[str, np.ndarray]
    weight: float
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ParameterBlock:
    name: str
    length: int
    offset: int


@dataclass(frozen=True, slots=True)
class ParameterLayout:
    """Flat view of pose parameters for deterministic indexing."""

    blocks: tuple[ParameterBlock, ...]
    total: int

    @classmethod
    def from_shapes(
        cls, shapes: Mapping[str, Sequence[int]], *, subset: Sequence[str] | None = None
    ) -> "ParameterLayout":
        ordered = []
        offset = 0
        allowed = set(subset) if subset else None
        for name in POSE_PARAMS_ORDER:
            if allowed is not None and name not in allowed:
                continue
            if name not in shapes:
                continue
            length = int(shapes[name][-1])
            ordered.append(ParameterBlock(name=name, length=length, offset=offset))
            offset += length
        return cls(blocks=tuple(ordered), total=offset)

    def coordinate(self, index: int) -> tuple[str, int]:
        if index < 0 or index >= self.total:
            raise IndexError(f"Coordinate {index} is out of range for total {self.total}.")
        for block in self.blocks:
            if block.offset <= index < block.offset + block.length:
                return block.name, index - block.offset
        raise IndexError(f"Coordinate {index} could not be resolved.")

    def as_slices(self) -> Mapping[str, slice]:
        return {
            block.name: slice(block.offset, block.offset + block.length) for block in self.blocks
        }

    def validate_pose(self, pose: Mapping[str, np.ndarray]) -> None:
        for block in self.blocks:
            if block.name not in pose:
                raise KeyError(f"Pose missing parameter block '{block.name}'.")
            values = np.asarray(pose[block.name], dtype=float)
            if values.shape not in {(block.length,), (1, block.length)}:
                raise ValueError(f"Parameter '{block.name}' must have length {block.length}.")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _find_params_path(body_path: Path, override: Path | None) -> Path:
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"Parameter payload '{override}' does not exist.")
        return override

    candidates = [
        body_path.with_name(body_path.stem.replace("_body", "") + "_smplx_params.json"),
        body_path.with_name(body_path.stem + "_smplx_params.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "SMPL-X parameter payload not found. Provide --params or place a '*_smplx_params.json' next to the body."
    )


def _coerce_pose_array(
    value: Any, *, length: int, joint_map: Mapping[str, int] | None = None
) -> np.ndarray:
    if isinstance(value, Mapping):
        if joint_map is None:
            raise TypeError("Joint overrides require a joint layout.")
        result = np.zeros(length, dtype=float)
        for name, vec in value.items():
            if name not in joint_map:
                raise KeyError(f"Unknown joint '{name}' in pose mapping.")
            start = joint_map[name]
            arr = np.asarray(vec, dtype=float).reshape(-1)
            if arr.size != 3:
                raise ValueError(f"Joint '{name}' must provide three axis-angle values.")
            result[start : start + 3] = arr
        return result

    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(length, dtype=float)
    if arr.size == length:
        return arr
    raise ValueError(f"Expected length {length} for pose array, got {arr.size}.")


def _build_joint_map() -> Mapping[str, int]:
    return {name: 3 * idx for idx, name in enumerate(SMPLX_BODY_JOINTS)}


def _merge_pose(
    entry: Mapping[str, Any],
    *,
    layout: ParameterLayout,
    joint_map: Mapping[str, int],
    base: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    merged: dict[str, np.ndarray] = {
        name: np.asarray(values, dtype=float).reshape(-1) for name, values in base.items()
    }
    length_map = {block.name: block.length for block in layout.blocks}
    for block in layout.blocks:
        if block.name not in merged:
            merged[block.name] = np.zeros(block.length, dtype=float)

    for key, raw in entry.items():
        if key not in length_map:
            continue
        if key == "body_pose":
            merged[key] = _coerce_pose_array(raw, length=length_map[key], joint_map=joint_map)
        else:
            merged[key] = _coerce_pose_array(raw, length=length_map[key], joint_map=None)
    return merged


def _load_pose_sweep(
    path: Path,
    *,
    layout: ParameterLayout,
    pose_limit: int | None,
    base_pose: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[PoseSample], Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Pose sweep payload must be a JSON object.")
    joint_map = _build_joint_map()

    neutral_raw = payload.get("neutral", {})
    if not isinstance(neutral_raw, Mapping):
        raise TypeError("'neutral' entry must be an object.")
    neutral_pose = _merge_pose(neutral_raw, layout=layout, joint_map=joint_map, base=base_pose)

    pose_entries = payload.get("poses")
    if not isinstance(pose_entries, Sequence):
        raise TypeError("'poses' must be a list.")
    meta = payload.get("meta", {})
    if not isinstance(meta, Mapping):
        meta = {}

    samples: list[PoseSample] = []
    for index, entry in enumerate(pose_entries):
        if not isinstance(entry, Mapping):
            raise TypeError("Each pose entry must be a JSON object.")
        pose_id = str(entry.get("id", f"pose_{index:03d}"))
        weight = float(entry.get("weight", 1.0))
        pose_params = _merge_pose(entry, layout=layout, joint_map=joint_map, base=neutral_pose)
        samples.append(PoseSample(pose_id=pose_id, parameters=pose_params, weight=weight))
        if pose_limit is not None and len(samples) >= pose_limit:
            break

    return neutral_pose, samples, meta


def _expand_weight_block(
    value: Any,
    *,
    length: int,
    joint_map: Mapping[str, int] | None = None,
) -> np.ndarray:
    if isinstance(value, Mapping):
        default = float(value.get("default", 0.0))
        overrides = value.get("overrides", {})
        weights = np.full(length, default, dtype=float)
        if isinstance(overrides, Mapping) and joint_map is not None:
            for name, override in overrides.items():
                if name not in joint_map:
                    raise KeyError(f"Unknown joint override '{name}' in weights.")
                start = joint_map[name]
                weights[start : start + 3] = float(override)
        return weights

    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full(length, float(arr[0]), dtype=float)
    if arr.size == length:
        return arr
    raise ValueError(f"Weight block expected length {length}, got {arr.size}.")


def _load_weights(
    path: Path, *, layout: ParameterLayout, joint_map: Mapping[str, int]
) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or "weights" not in payload:
        raise KeyError("Weights payload must include a 'weights' object.")
    weights_raw = payload["weights"]
    if not isinstance(weights_raw, Mapping):
        raise TypeError("'weights' must be a mapping of parameter names.")

    weights = np.zeros(layout.total, dtype=float)
    slices = layout.as_slices()
    for block in layout.blocks:
        if block.name not in weights_raw:
            block_weights = np.zeros(block.length, dtype=float)
        else:
            block_weights = _expand_weight_block(
                weights_raw[block.name],
                length=block.length,
                joint_map=joint_map if block.name == "body_pose" else None,
            )
        weights[slices[block.name]] = block_weights
    return weights


def _pose_defaults_from_payload(
    parameter_payload: Mapping[str, Any], layout: ParameterLayout
) -> dict[str, np.ndarray]:
    defaults: dict[str, np.ndarray] = {}
    for block in layout.blocks:
        if block.name in parameter_payload:
            defaults[block.name] = np.asarray(parameter_payload[block.name], dtype=float).reshape(
                -1
            )
        else:
            defaults[block.name] = np.zeros(block.length, dtype=float)
    return defaults


class SmplxPoseBackend:
    """Thin wrapper that exposes SMPL-X vertices (and optional joints) for pose dictionaries."""

    def __init__(
        self,
        *,
        parameter_payload: Mapping[str, Any],
        params_path: Path,
        model_path: Path | None = None,
    ) -> None:
        try:
            from avatar_model import (
                BodyModel,
            )  # Local import to avoid heavy dependency at module import.
        except ImportError as exc:  # pragma: no cover - heavy dependency path
            raise ImportError(
                "avatar_model (and torch) are required to run the real ROM sampler. Install the SMPL-X extras."
            ) from exc

        params, scale, model_type, gender = _load_smplx_parameter_payload(parameter_payload)
        self.scale = float(scale)
        betas = np.asarray(params.get("betas", np.zeros((1, 10), dtype=np.float32)))
        expr = np.asarray(params.get("expression", np.zeros((1, 10), dtype=np.float32)))
        self.model = BodyModel(
            model_path=model_path or Path("assets") / model_type,
            model_type=model_type,
            gender=gender,
            batch_size=1,
            num_betas=int(betas.shape[-1]),
            num_expression_coeffs=int(expr.shape[-1]),
        )
        self.model.set_parameters(params)
        self.params_path = params_path

    def parameter_shapes(self) -> Mapping[str, Sequence[int]]:
        return self.model.parameter_shapes()

    def evaluate(self, pose: Mapping[str, np.ndarray]) -> np.ndarray:
        updates: MutableMapping[str, np.ndarray] = {}
        for name, values in pose.items():
            arr = np.asarray(values, dtype=np.float32).reshape(1, -1)
            updates[name] = arr
        self.model.set_parameters(updates)
        vertices = self.model.vertices().detach().cpu().numpy()[0]
        return np.asarray(vertices * self.scale, dtype=float)

    def evaluate_with_joints(
        self, pose: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return vertices and joint centers for a pose."""

        updates: MutableMapping[str, np.ndarray] = {}
        for name, values in pose.items():
            arr = np.asarray(values, dtype=np.float32).reshape(1, -1)
            updates[name] = arr
        self.model.set_parameters(updates)
        vertices = self.model.vertices().detach().cpu().numpy()[0] * self.scale
        joints = self.model.joints().detach().cpu().numpy()[0] * self.scale
        joint_map: dict[str, np.ndarray] = {}
        for idx, name in enumerate(SMPLX_BODY_JOINTS):
            if idx < joints.shape[0]:
                joint_map[name] = np.asarray(joints[idx], dtype=float)
        return np.asarray(vertices, dtype=float), joint_map


def _central_difference(
    backend: SmplxPoseBackend,
    base_pose: Mapping[str, np.ndarray],
    *,
    block: str,
    index: int,
    step: float,
) -> np.ndarray:
    plus = {name: np.array(values, copy=True) for name, values in base_pose.items()}
    minus = {name: np.array(values, copy=True) for name, values in base_pose.items()}
    plus[block][index] += step
    minus[block][index] -= step
    v_plus = backend.evaluate(plus)
    v_minus = backend.evaluate(minus)
    return (v_plus - v_minus) / (2.0 * step)


def _stream_diagonal_costs(
    backend: SmplxPoseBackend,
    *,
    neutral_pose: Mapping[str, np.ndarray],
    samples: Sequence[PoseSample],
    layout: ParameterLayout,
    weights: np.ndarray,
    fd_step: float,
    epsilon: float,
    sample_field_sink: MutableSequence[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, float]], list[Mapping[str, Any]], np.ndarray]:
    v0 = backend.evaluate(neutral_pose)
    vertex_count = v0.shape[0]
    costs = np.zeros(vertex_count, dtype=float)
    call_counts = {"base": 1, "fd": 0}
    pose_stats: list[dict[str, float]] = []
    pose_metadata: list[Mapping[str, Any]] = []

    coordinate_indices = [idx for idx, w in enumerate(weights) if float(w) != 0.0]

    for sample in samples:
        layout.validate_pose(sample.parameters)
        vt, joints = backend.evaluate_with_joints(sample.parameters)
        call_counts["base"] += 1
        disp = vt - v0
        denom = np.linalg.norm(disp, axis=1) + float(epsilon)
        disp_norm = disp / denom[:, None]

        q = np.zeros(vertex_count, dtype=float)
        for coord_idx in coordinate_indices:
            block, local = layout.coordinate(coord_idx)
            deriv = _central_difference(
                backend,
                sample.parameters,
                block=block,
                index=local,
                step=fd_step,
            )
            call_counts["fd"] += 2
            sens = np.abs((disp_norm * deriv).sum(axis=1))
            weight = float(weights[coord_idx])
            q += weight * sens * sens
        chain_factor = 1.0
        meta = dict(sample.metadata) if sample.metadata else {}
        active_axes = meta.get("active_axes")
        if active_axes and isinstance(active_axes, (list, tuple)):
            chain_amp = float(meta.get("chain_amplification", 0.25))
            chain_factor = 1.0 + chain_amp * max(len(active_axes) - 1, 0)
            meta["chain_factor"] = chain_factor

        legality_meta: Mapping[str, Any] | None = None
        legality_cfg = meta.pop("_legality_config", None)  # populated upstream when desired
        if legality_cfg is not None:
            from smii.rom.pose_legality import LegalityConfig as _LC, score_legality as _score

            try:
                lc: _LC = legality_cfg if isinstance(legality_cfg, _LC) else _LC()
                legality = _score(joints, lc)
                legality_meta = {
                    "score": legality.score,
                    "violations": legality.violations,
                    "worst_pair": legality.worst_pair,
                    "worst_penetration": legality.worst_penetration,
                }
                meta["legality_runtime"] = legality_meta
                if meta.get("filter_illegal") and legality.score > 0:
                    continue
                if meta.get("legality_weight") and legality.score > 0:
                    chain_factor *= float(np.exp(-float(meta["legality_weight"]) * legality.score))
            except Exception as exc:  # pragma: no cover - defensive
                meta["legality_runtime_error"] = str(exc)

        weighted_field = chain_factor * q
        costs += float(sample.weight) * weighted_field
        pose_stats.append(
            {
                "pose_id": sample.pose_id,
                "weight": float(sample.weight),
                "chain_factor": float(chain_factor),
                "q_mean": float(np.mean(q)),
                "q_max": float(np.max(q)),
            }
        )
        pose_metadata.append(meta)
        if sample_field_sink is not None:
            sample_field_sink.append(
                {
                    "pose_id": sample.pose_id,
                    "weight": float(sample.weight),
                    "field": np.asarray(weighted_field, dtype=float),
                    "metadata": dict(meta),
                }
            )

    call_counts["total"] = call_counts["base"] + call_counts["fd"]
    return costs, call_counts, pose_stats, pose_metadata, v0


def _nearest_neighbor_map(
    source: np.ndarray, target: np.ndarray, *, batch_size: int = 256
) -> tuple[np.ndarray, float, float]:
    """Map each target vertex to the nearest source vertex."""

    mapping, min_dists = _nearest_neighbor_lookup(source, target, batch_size=batch_size)
    max_dist = float(np.max(min_dists)) if len(min_dists) else 0.0
    mean_dist = float(np.mean(min_dists)) if len(min_dists) else 0.0
    return mapping, max_dist, mean_dist


def _nearest_neighbor_lookup(
    source: np.ndarray, target: np.ndarray, *, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Map each target vertex to nearest source vertex and return per-target distances."""

    source_arr = np.asarray(source, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    if (
        source_arr.ndim != 2
        or target_arr.ndim != 2
        or source_arr.shape[1] != 3
        or target_arr.shape[1] != 3
    ):
        raise ValueError("Source and target vertices must be shaped (N, 3).")

    mapping = np.empty(target_arr.shape[0], dtype=np.int64)
    distances = np.empty(target_arr.shape[0], dtype=np.float64)
    step = max(1, int(batch_size))

    for start in range(0, target_arr.shape[0], step):
        end = min(start + step, target_arr.shape[0])
        chunk = target_arr[start:end]
        diff = chunk[:, None, :] - source_arr[None, :, :]
        dist2 = np.einsum("bij,bij->bi", diff, diff)
        nearest = np.argmin(dist2, axis=1)
        mapping[start:end] = nearest.astype(np.int64)
        distances[start:end] = np.sqrt(dist2[np.arange(len(nearest)), nearest])

    return mapping, distances


def _mapping_collision_ratio(indices: np.ndarray) -> tuple[int, float]:
    unique_targets = int(np.unique(np.asarray(indices, dtype=np.int64)).size)
    collisions = max(0, len(indices) - unique_targets)
    return unique_targets, float(collisions / max(1, len(indices)))


def _save_vertex_correspondence(
    path: Path,
    *,
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    source_label: str,
    target_label: str,
    target_to_source_indices: np.ndarray | None = None,
    target_to_source_distances: np.ndarray | None = None,
    policy: str,
    max_distance: float,
) -> Mapping[str, Any]:
    """Persist full bidirectional nearest-neighbor correspondence for this sampler run."""

    source_arr = np.asarray(source_vertices, dtype=float)
    target_arr = np.asarray(target_vertices, dtype=float)
    if source_arr.ndim != 2 or source_arr.shape[1] != 3:
        raise ValueError("source_vertices must be shaped (N, 3).")
    if target_arr.ndim != 2 or target_arr.shape[1] != 3:
        raise ValueError("target_vertices must be shaped (N, 3).")

    if source_arr.shape[0] == target_arr.shape[0]:
        identity = np.arange(source_arr.shape[0], dtype=np.int64)
        source_to_target = np.array(identity, copy=True)
        target_to_source = np.array(identity, copy=True)
        pair_dist = np.linalg.norm(source_arr - target_arr, axis=1)
        source_to_target_dist = np.array(pair_dist, copy=True)
        target_to_source_dist = np.array(pair_dist, copy=True)
    else:
        source_to_target, source_to_target_dist = _nearest_neighbor_lookup(target_arr, source_arr)
        if target_to_source_indices is None or target_to_source_distances is None:
            target_to_source, target_to_source_dist = _nearest_neighbor_lookup(
                source_arr, target_arr
            )
        else:
            target_to_source = np.asarray(target_to_source_indices, dtype=np.int64)
            target_to_source_dist = np.asarray(target_to_source_distances, dtype=np.float64)
            if target_to_source.shape[0] != target_arr.shape[0]:
                raise ValueError("target_to_source_indices length must equal target vertex count.")
            if target_to_source_dist.shape[0] != target_arr.shape[0]:
                raise ValueError(
                    "target_to_source_distances length must equal target vertex count."
                )

    src_unique, src_collision = _mapping_collision_ratio(source_to_target)
    tgt_unique, tgt_collision = _mapping_collision_ratio(target_to_source)
    meta: dict[str, Any] = {
        "source_label": source_label,
        "target_label": target_label,
        "source_vertex_count": int(source_arr.shape[0]),
        "target_vertex_count": int(target_arr.shape[0]),
        "policy": str(policy),
        "max_distance_threshold": float(max_distance),
        "source_to_target_max_distance": float(np.max(source_to_target_dist))
        if len(source_to_target_dist)
        else 0.0,
        "source_to_target_mean_distance": float(np.mean(source_to_target_dist))
        if len(source_to_target_dist)
        else 0.0,
        "source_to_target_unique_targets": int(src_unique),
        "source_to_target_collision_ratio": float(src_collision),
        "target_to_source_max_distance": float(np.max(target_to_source_dist))
        if len(target_to_source_dist)
        else 0.0,
        "target_to_source_mean_distance": float(np.mean(target_to_source_dist))
        if len(target_to_source_dist)
        else 0.0,
        "target_to_source_unique_sources": int(tgt_unique),
        "target_to_source_collision_ratio": float(tgt_collision),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        source_to_target_indices=np.asarray(source_to_target, dtype=np.int64),
        source_to_target_distances=np.asarray(source_to_target_dist, dtype=np.float64),
        target_to_source_indices=np.asarray(target_to_source, dtype=np.int64),
        target_to_source_distances=np.asarray(target_to_source_dist, dtype=np.float64),
        meta=np.array(meta, dtype=object),
    )
    return meta


def _remap_costs(
    costs: np.ndarray,
    neutral_vertices: np.ndarray,
    target_vertices: np.ndarray,
    *,
    policy: str,
    max_distance: float,
    target_to_source_mapping: np.ndarray | None = None,
    target_to_source_distances: np.ndarray | None = None,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Apply vertex-count remapping with configurable policy."""

    source_vertex_count = len(costs)
    target_vertex_count = target_vertices.shape[0]
    mapping_info: dict[str, Any] = {
        "policy": str(policy),
        "threshold": float(max_distance),
        "mode": "identity",
        "max_distance": 0.0,
        "mean_distance": 0.0,
        "source_vertex_count": source_vertex_count,
        "target_vertex_count": target_vertex_count,
    }

    if source_vertex_count == target_vertex_count:
        return costs, mapping_info

    if policy == "error":
        raise ValueError(
            f"Vertex count mismatch (source={source_vertex_count}, target={target_vertex_count}) with policy=error."
        )

    if target_to_source_mapping is None or target_to_source_distances is None:
        mapping, min_dists = _nearest_neighbor_lookup(neutral_vertices, target_vertices)
    else:
        mapping = np.asarray(target_to_source_mapping, dtype=np.int64)
        min_dists = np.asarray(target_to_source_distances, dtype=np.float64)
        if mapping.shape[0] != target_vertex_count:
            raise ValueError("target_to_source_mapping length must match target vertex count.")
        if min_dists.shape[0] != target_vertex_count:
            raise ValueError("target_to_source_distances length must match target vertex count.")

    remapped = costs[mapping]
    max_dist = float(np.max(min_dists)) if len(min_dists) else 0.0
    mean_dist = float(np.mean(min_dists)) if len(min_dists) else 0.0
    mapping_info["mode"] = "nearest_neighbor"
    mapping_info["max_distance"] = float(max_dist)
    mapping_info["mean_distance"] = float(mean_dist)

    if max_dist > max_distance:
        message = (
            f"Remap max distance {max_dist:.6f} exceeds threshold {max_distance:.6f} "
            f"(policy={policy}, source={source_vertex_count}, target={target_vertex_count})."
        )
        if policy == "error":
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    return remapped, mapping_info


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _save_meta(
    path: Path,
    *,
    pose_ids: Sequence[str],
    pose_metadata: Sequence[Mapping[str, Any] | None],
    pose_meta: Mapping[str, Any],
    sweep_hash: str | None,
    schedule_hash: str | None,
    weights_hash: str,
    body_hash: str,
    call_counts: Mapping[str, Any],
    fd_step: float,
    epsilon: float,
    vertex_count: int,
    source_vertex_count: int,
    weights: np.ndarray,
    joint_subset: Sequence[str] | None,
    pose_limit: int | None,
    costs_path: Path,
    params_path: Path,
    body_path: Path,
    sweep_path: Path | None,
    schedule_path: Path | None,
    pose_stats: Sequence[Mapping[str, float]],
    git_commit: str | None,
    mapping_info: Mapping[str, Any],
) -> None:
    payload = {
        "meta": {
            "synthetic": False,
            "mode": "diagonal",
            "fd_step": float(fd_step),
            "epsilon": float(epsilon),
            "vertex_count": vertex_count,
            "pose_count": len(pose_ids),
            "pose_limit": pose_limit,
            "joint_subset": list(joint_subset) if joint_subset else None,
            "weights_l1": float(np.sum(np.abs(weights))),
            "weights_max": float(np.max(np.abs(weights))) if weights.size else 0.0,
            "source_vertex_count": source_vertex_count,
            "mapping": dict(mapping_info),
            "git_commit": git_commit,
            "call_counts": dict(call_counts),
            "sweep_hash": sweep_hash,
            "schedule_hash": schedule_hash,
            "weights_hash": weights_hash,
            "body_hash": body_hash,
            "params_path": str(params_path),
            "body_path": str(body_path),
            "sweep_path": str(sweep_path) if sweep_path else None,
            "schedule_path": str(schedule_path) if schedule_path else None,
            "costs_path": str(costs_path),
        },
        "pose_ids": list(pose_ids),
        "pose_metadata": list(pose_metadata),
        "pose_meta": pose_meta,
        "pose_stats": list(pose_stats),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_coeff_samples(
    path: Path,
    *,
    projector: KernelProjector,
    sampled_fields: Sequence[Mapping[str, Any]],
    basis_path: Path,
    body_path: Path,
    body_hash: str,
    weights_hash: str,
    source_vertex_count: int,
    target_vertex_count: int,
    params_path: Path,
    sweep_path: Path | None,
    schedule_path: Path | None,
    fd_step: float,
    mapping_info: Mapping[str, Any],
    git_commit: str | None,
) -> None:
    samples_payload: list[dict[str, Any]] = []
    for entry in sampled_fields:
        field = np.asarray(entry["field"], dtype=float).reshape(-1)
        coeffs = projector.encode(field)
        samples_payload.append(
            {
                "pose_id": str(entry["pose_id"]),
                "weight": float(entry["weight"]),
                "coeffs": {"seam_sensitivity": coeffs.tolist()},
                "observations": None,
                "metadata": dict(entry.get("metadata") or {}),
            }
        )

    payload = {
        "meta": {
            "synthetic": False,
            "field_name": "seam_sensitivity",
            "basis_path": str(basis_path),
            "basis_hash": _sha256_path(basis_path),
            "basis_vertex_count": int(projector.vertex_count),
            "basis_component_count": int(projector.component_count),
            "body_path": str(body_path),
            "body_hash": body_hash,
            "params_path": str(params_path),
            "weights_hash": weights_hash,
            "source_vertex_count": int(source_vertex_count),
            "target_vertex_count": int(target_vertex_count),
            "fd_step": float(fd_step),
            "mapping": dict(mapping_info),
            "git_commit": git_commit,
            "sweep_path": str(sweep_path) if sweep_path else None,
            "schedule_path": str(schedule_path) if schedule_path else None,
        },
        "samples": samples_payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_seam_cost_field(
    costs: np.ndarray, *, samples_used: int, fd_step: float
) -> SeamCostField:
    return SeamCostField(
        field="rom_fd_sensitivity",
        vertex_costs=costs,
        edge_costs=np.zeros(0, dtype=float),
        edges=tuple(),
        samples_used=samples_used,
        metadata={
            "fd_step": float(fd_step),
            "mode": "diagonal",
            "samples_used": int(samples_used),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--body",
        required=True,
        type=Path,
        help="Body mesh npz/json containing vertices (faces optional).",
    )
    parser.add_argument(
        "--poses",
        required=False,
        type=Path,
        help="Pose sweep JSON (see data/rom/afflec_sweep.json).",
    )
    parser.add_argument(
        "--schedule",
        required=False,
        type=Path,
        help="YAML sweep schedule (data/rom/sweep_schedule.yaml). Overrides --poses when set.",
    )
    parser.add_argument(
        "--weights", required=True, type=Path, help="Joint weights JSON (diagonal W)."
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="Optional SMPL-X parameter payload (JSON). Auto-resolves next to body when omitted.",
    )
    parser.add_argument(
        "--fd-step", type=float, default=1e-3, help="Central difference step size (radians)."
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-6, help="Epsilon added to displacement norm."
    )
    parser.add_argument(
        "--out-costs",
        type=Path,
        default=Path("outputs/rom/seam_costs_afflec.npz"),
        help="Output NPZ for seam costs.",
    )
    parser.add_argument(
        "--out-meta",
        type=Path,
        default=Path("outputs/rom/afflec_rom_run.json"),
        help="Output JSON for sampler provenance.",
    )
    parser.add_argument(
        "--basis",
        type=Path,
        default=None,
        help="Optional canonical basis NPZ used for coefficient export/reporting.",
    )
    parser.add_argument(
        "--out-coeff-samples",
        type=Path,
        default=None,
        help="Optional JSON export of per-pose operator coefficients derived from the sampled sensitivity fields.",
    )
    parser.add_argument(
        "--vertex-map",
        type=str,
        choices=("nearest", "error"),
        default="nearest",
        help="Vertex mapping policy when vertex counts differ: 'nearest' remaps in neutral pose; 'error' fails fast.",
    )
    parser.add_argument(
        "--max-map-distance",
        type=float,
        default=0.03,
        help="Maximum allowed nearest-neighbour distance (meters) when remapping vertices.",
    )
    parser.add_argument(
        "--out-correspondence",
        type=Path,
        default=None,
        help=(
            "Optional NPZ export for full transform-native vertex correspondence "
            "(source<->target nearest-neighbour maps with distances)."
        ),
    )
    parser.add_argument(
        "--pose-limit", type=int, default=None, help="Optional cap on number of poses processed."
    )
    parser.add_argument(
        "--joint-subset",
        type=str,
        nargs="+",
        default=None,
        help="Restrict evaluation to specific parameter blocks (e.g., body_pose global_orient).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="diagonal",
        choices=("diagonal",),
        help="Contraction mode (diagonal only for this sprint).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed when expanding schedule-driven sweeps.",
    )
    parser.add_argument(
        "--out-samples",
        type=Path,
        default=None,
        help="Optional JSON manifest of expanded samples (schedule mode).",
    )
    parser.add_argument(
        "--out-envelope",
        type=Path,
        default=None,
        help="Optional JSON envelope report (L0/L1 angles) when using schedule.",
    )
    parser.add_argument(
        "--prev-envelope",
        type=Path,
        default=None,
        help="Optional previous envelope JSON to compare for completeness.",
    )
    parser.add_argument(
        "--prev-costs",
        type=Path,
        default=None,
        help="Optional previous seam costs NPZ to compute rank stability.",
    )
    parser.add_argument(
        "--out-certificate",
        type=Path,
        default=None,
        help="Optional L3-style completeness certificate JSON.",
    )
    parser.add_argument(
        "--filter-illegal",
        action="store_true",
        help="Filter schedule-expanded samples with legality score > 0.",
    )
    parser.add_argument(
        "--legality-weight",
        type=float,
        default=None,
        help="If set, downweight samples by exp(-w * legality_score).",
    )
    parser.add_argument(
        "--chain-amplification",
        type=float,
        default=0.25,
        help="Amplification factor applied when multiple axes/joints are active in a sample (chain-aware weighting).",
    )
    args = parser.parse_args(argv)
    if args.out_coeff_samples is not None and args.basis is None:
        raise ValueError("--basis is required when using --out-coeff-samples.")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode != "diagonal":
        raise ValueError("Only diagonal contraction is supported in this sprint.")

    body_path = Path(os.path.expandvars(str(args.body))).expanduser()
    sweep_path = Path(os.path.expandvars(str(args.poses))).expanduser() if args.poses else None
    schedule_path = (
        Path(os.path.expandvars(str(args.schedule))).expanduser() if args.schedule else None
    )
    weights_path = Path(os.path.expandvars(str(args.weights))).expanduser()
    params_path = _find_params_path(body_path, args.params)
    if sweep_path is None and schedule_path is None:
        raise ValueError("Provide either --poses or --schedule for ROM sampling.")
    if sweep_path is not None and schedule_path is not None:
        warnings.warn("--schedule supplied; --poses will be ignored.", RuntimeWarning, stacklevel=2)
        sweep_path = None

    body_record = load_body_record(body_path)
    vertices = np.asarray(body_record.get("vertices"))
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Body vertices must be shaped (N, 3).")

    parameter_payload = json.loads(params_path.read_text(encoding="utf-8"))
    backend = SmplxPoseBackend(parameter_payload=parameter_payload, params_path=params_path)
    layout = ParameterLayout.from_shapes(backend.parameter_shapes(), subset=args.joint_subset)
    if layout.total == 0:
        raise ValueError("No pose parameters selected; cannot compute finite differences.")

    joint_map = _build_joint_map()
    base_pose_defaults = _pose_defaults_from_payload(parameter_payload, layout)
    legality_cfg = LegalityConfig()
    if schedule_path is not None:
        from smii.rom.pose_schedule import load_schedule_samples

        neutral_pose, scheduled_samples, pose_meta = load_schedule_samples(
            schedule_path,
            layout_blocks=[block.name for block in layout.blocks],
            joint_map=joint_map,
            base_pose=base_pose_defaults,
            seed_override=args.seed,
            legality_config=legality_cfg,
            filter_illegal=args.filter_illegal,
            legality_weight=args.legality_weight,
        )
        samples = [
            PoseSample(
                pose_id=s.pose_id, parameters=s.parameters, weight=s.weight, metadata=s.metadata
            )
            for s in scheduled_samples
        ]
        pose_meta["schedule_path"] = str(schedule_path)
    else:
        neutral_pose, samples, pose_meta = _load_pose_sweep(
            sweep_path,
            layout=layout,
            pose_limit=args.pose_limit,
            base_pose=base_pose_defaults,
        )
    weights = _load_weights(weights_path, layout=layout, joint_map=joint_map)
    if weights.size != layout.total:
        raise ValueError("Weights vector length does not match pose parameter layout.")

    # Embed legality and chain settings into metadata for downstream diagnostics.
    enriched_samples: list[PoseSample] = []
    for sample in samples:
        md = dict(sample.metadata) if sample.metadata else {}
        md["_legality_config"] = legality_cfg
        if args.filter_illegal:
            md["filter_illegal"] = True
        if args.legality_weight is not None:
            md["legality_weight"] = float(args.legality_weight)
        md["chain_amplification"] = float(args.chain_amplification)
        enriched_samples.append(
            PoseSample(
                pose_id=sample.pose_id,
                parameters=sample.parameters,
                weight=sample.weight,
                metadata=md,
            )
        )

    sampled_fields: list[dict[str, Any]] = []
    costs, call_counts, pose_stats, pose_metadata, neutral_vertices = _stream_diagonal_costs(
        backend,
        neutral_pose=neutral_pose,
        samples=enriched_samples,
        layout=layout,
        weights=weights,
        fd_step=args.fd_step,
        epsilon=args.epsilon,
        sample_field_sink=sampled_fields if args.out_coeff_samples is not None else None,
    )

    source_vertex_count = len(costs)
    target_to_source_mapping: np.ndarray | None = None
    target_to_source_distances: np.ndarray | None = None
    if source_vertex_count != vertices.shape[0]:
        target_to_source_mapping, target_to_source_distances = _nearest_neighbor_lookup(
            neutral_vertices, vertices
        )
    costs, mapping_info = _remap_costs(
        costs,
        neutral_vertices,
        vertices,
        policy=args.vertex_map,
        max_distance=args.max_map_distance,
        target_to_source_mapping=target_to_source_mapping,
        target_to_source_distances=target_to_source_distances,
    )
    mapping_info = dict(mapping_info)
    target_vertex_count = len(costs)
    sweep_hash = _sha256_path(sweep_path) if sweep_path else None
    schedule_hash = _sha256_path(schedule_path) if schedule_path else None
    weights_hash = _sha256_path(weights_path)
    body_hash = _sha256_array(vertices)
    git_commit = _git_commit()
    if mapping_info.get("mode") != "identity":
        print(
            f"Mapped costs from source vertex count {source_vertex_count} to target {target_vertex_count} "
            f"(max distance {mapping_info['max_distance']:.6f}, mean {mapping_info['mean_distance']:.6f}, "
            f"threshold {args.max_map_distance:.6f}, policy={args.vertex_map})."
        )
    if args.out_correspondence is not None:
        correspondence_meta = _save_vertex_correspondence(
            args.out_correspondence,
            source_vertices=neutral_vertices,
            target_vertices=vertices,
            source_label="rom_neutral_vertices",
            target_label="sampler_body_vertices",
            target_to_source_indices=target_to_source_mapping,
            target_to_source_distances=target_to_source_distances,
            policy=args.vertex_map,
            max_distance=args.max_map_distance,
        )
        mapping_info["correspondence_artifact"] = str(args.out_correspondence)
        mapping_info["correspondence_meta"] = dict(correspondence_meta)
        print(f"Wrote transform-native correspondence to {args.out_correspondence}")

    if args.out_coeff_samples is not None:
        basis = load_basis(args.basis)
        projector = KernelProjector(basis)
        if projector.vertex_count != target_vertex_count:
            raise ValueError(
                f"Basis vertex count {projector.vertex_count} does not match sampled field vertex count "
                f"{target_vertex_count}."
            )
        remapped_fields: list[dict[str, Any]] = []
        for entry in sampled_fields:
            field = np.asarray(entry["field"], dtype=float)
            remapped_field, _ = _remap_costs(
                field,
                neutral_vertices,
                vertices,
                policy=args.vertex_map,
                max_distance=args.max_map_distance,
                target_to_source_mapping=target_to_source_mapping,
                target_to_source_distances=target_to_source_distances,
            )
            remapped_fields.append(
                {
                    "pose_id": entry["pose_id"],
                    "weight": entry["weight"],
                    "field": remapped_field,
                    "metadata": entry.get("metadata"),
                }
            )
        _save_coeff_samples(
            args.out_coeff_samples,
            projector=projector,
            sampled_fields=remapped_fields,
            basis_path=args.basis,
            body_path=body_path,
            body_hash=body_hash,
            weights_hash=weights_hash,
            source_vertex_count=source_vertex_count,
            target_vertex_count=target_vertex_count,
            params_path=params_path,
            sweep_path=sweep_path,
            schedule_path=schedule_path,
            fd_step=args.fd_step,
            mapping_info=mapping_info,
            git_commit=git_commit,
        )
        print(f"Wrote coefficient samples to {args.out_coeff_samples}")

    samples_used = len(pose_stats)
    cost_field = _build_seam_cost_field(costs, samples_used=samples_used, fd_step=args.fd_step)
    save_seam_cost_field(cost_field, args.out_costs)
    _save_meta(
        args.out_meta,
        pose_ids=[stat["pose_id"] for stat in pose_stats],
        pose_metadata=pose_metadata,
        pose_meta=pose_meta,
        sweep_hash=sweep_hash,
        schedule_hash=schedule_hash,
        weights_hash=weights_hash,
        body_hash=body_hash,
        call_counts=call_counts,
        fd_step=args.fd_step,
        epsilon=args.epsilon,
        vertex_count=target_vertex_count,
        source_vertex_count=source_vertex_count,
        weights=weights,
        joint_subset=args.joint_subset,
        pose_limit=args.pose_limit,
        costs_path=args.out_costs,
        params_path=params_path,
        body_path=body_path,
        sweep_path=sweep_path,
        schedule_path=schedule_path,
        pose_stats=pose_stats,
        git_commit=git_commit,
        mapping_info=mapping_info,
    )

    if args.out_samples and schedule_path is not None:
        manifest = {
            "meta": {
                "schedule_path": str(schedule_path),
                "seed": args.seed,
            },
            "samples": [
                {
                    "pose_id": sample.pose_id,
                    "weight": float(sample.weight),
                    "metadata": sample.metadata,
                }
                for sample in samples
            ],
        }
        args.out_samples.parent.mkdir(parents=True, exist_ok=True)
        args.out_samples.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote sample manifest to {args.out_samples} (count={len(samples)})")

    envelope = None
    if schedule_path is not None:
        envelope = build_envelope(samples)
        if args.out_envelope:
            args.out_envelope.parent.mkdir(parents=True, exist_ok=True)
            args.out_envelope.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
            print(f"Wrote envelope summary to {args.out_envelope}")

    if args.out_certificate and schedule_path is not None:
        prev_env = None
        if args.prev_envelope and args.prev_envelope.exists():
            prev_env = json.loads(args.prev_envelope.read_text(encoding="utf-8"))
        envelope_deltas = compare_envelopes(
            prev_env or {"L0": {}, "L1": {}}, envelope or {"L0": {}, "L1": {}}
        )

        rank_corr = None
        if args.prev_costs and args.prev_costs.exists():
            prev_npz = np.load(args.prev_costs)
            prev_costs = prev_npz.get("vertex_costs")
            if prev_costs is None:
                prev_costs = prev_npz.get("arr_0")
            if prev_costs is not None and len(prev_costs) == len(costs):
                rank_corr = spearman_rank(prev_costs, costs)

        status = "INIT" if prev_env is None else "PASS"
        flagged = []
        for entry in envelope_deltas.get("L0", []):
            if entry.get("changed"):
                flagged.append(entry["key"])
        if rank_corr is not None and rank_corr < 0.98:
            flagged.append("rank_stability")
        if flagged and status != "INIT":
            status = "WARN"

        certificate = {
            "status": status,
            "flags": flagged,
            "envelope": envelope,
            "envelope_deltas": envelope_deltas,
            "rank_correlation": rank_corr,
            "counts": envelope.get("counts") if envelope else {},
            "inputs": {
                "schedule_path": str(schedule_path),
                "seed": args.seed,
                "prev_envelope": str(args.prev_envelope) if args.prev_envelope else None,
                "prev_costs": str(args.prev_costs) if args.prev_costs else None,
            },
        }
        args.out_certificate.parent.mkdir(parents=True, exist_ok=True)
        args.out_certificate.write_text(json.dumps(certificate, indent=2), encoding="utf-8")
        print(f"Wrote completeness certificate to {args.out_certificate}")

    print(f"Wrote seam costs to {args.out_costs} (len={len(costs)})")
    print(f"Wrote provenance to {args.out_meta}")


if __name__ == "__main__":  # pragma: no cover
    main()
