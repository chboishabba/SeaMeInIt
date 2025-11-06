"""Utilities to export SMPL-X meshes for Unity and Unreal."""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class ExportFormat(str, Enum):
    """Supported export file formats."""

    FBX = "fbx"
    GLB = "glb"


@dataclass(slots=True)
class EngineConfig:
    """Configuration describing the target engine coordinate conventions."""

    name: str
    unit_scale: float
    up_axis: str
    forward_axis: str


@dataclass(slots=True)
class SMPLXTemplate:
    """Container holding the mesh and rig information required for export."""

    vertices: List[Tuple[float, float, float]]
    faces: List[Tuple[int, int, int]]
    joint_names: List[str]
    joint_parents: List[int]
    joint_positions: List[Tuple[float, float, float]]
    skin_weights: List[Tuple[float, float, float, float]]
    skin_joints: List[Tuple[int, int, int, int]]

    def validate(self) -> None:
        vertex_count = len(self.vertices)
        if any(len(vertex) != 3 for vertex in self.vertices):
            msg = "vertices must contain xyz coordinates"
            raise ValueError(msg)
        if any(len(face) != 3 for face in self.faces):
            msg = "faces must be triangles"
            raise ValueError(msg)
        if len(self.joint_names) != len(self.joint_parents):
            msg = "joint_names and joint_parents must have the same length"
            raise ValueError(msg)
        if len(self.joint_positions) != len(self.joint_names):
            msg = "joint_positions must match joint_names"
            raise ValueError(msg)
        if any(len(pos) != 3 for pos in self.joint_positions):
            msg = "joint_positions must have xyz coordinates"
            raise ValueError(msg)
        if len(self.skin_weights) != vertex_count:
            msg = "skin_weights must match vertex count"
            raise ValueError(msg)
        if len(self.skin_joints) != vertex_count:
            msg = "skin_joints must match vertex count"
            raise ValueError(msg)
        if any(len(weight) != 4 for weight in self.skin_weights):
            msg = "skin_weights must provide four influences"
            raise ValueError(msg)
        if any(len(joint) != 4 for joint in self.skin_joints):
            msg = "skin_joints must provide four joint indices"
            raise ValueError(msg)


DEFAULT_BONE_REMAP: Mapping[str, str] = {
    "pelvis": "Hips",
    "left_hip": "LeftUpLeg",
    "right_hip": "RightUpLeg",
    "spine1": "Spine",
    "spine2": "Spine1",
    "spine3": "Spine2",
    "neck": "Neck",
    "head": "Head",
    "left_shoulder": "LeftShoulder",
    "left_elbow": "LeftArm",
    "left_wrist": "LeftForeArm",
    "right_shoulder": "RightShoulder",
    "right_elbow": "RightArm",
    "right_wrist": "RightForeArm",
}


UNITY_CONFIG = EngineConfig(name="Unity", unit_scale=1.0, up_axis="Y", forward_axis="Z")
UNREAL_CONFIG = EngineConfig(name="Unreal", unit_scale=0.01, up_axis="Z", forward_axis="X")


@dataclass(slots=True)
class Bone:
    name: str
    parent_index: Optional[int]
    position: Tuple[float, float, float]


def load_smplx_template(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    joint_names: Sequence[str],
    joint_parents: Sequence[int],
    joint_positions: Sequence[Sequence[float]],
    skin_weights: Sequence[Sequence[float]],
    skin_joints: Sequence[Sequence[int]],
) -> SMPLXTemplate:
    """Create a :class:`SMPLXTemplate` from python sequences."""

    template = SMPLXTemplate(
        vertices=[tuple(map(float, vertex)) for vertex in vertices],
        faces=[tuple(int(idx) for idx in face) for face in faces],
        joint_names=list(joint_names),
        joint_parents=[int(parent) for parent in joint_parents],
        joint_positions=[tuple(map(float, pos)) for pos in joint_positions],
        skin_weights=[tuple(map(float, weight)) for weight in skin_weights],
        skin_joints=[tuple(int(idx) for idx in joint) for joint in skin_joints],
    )
    template.validate()
    return template


def build_neutral_pose(template: SMPLXTemplate, remap: Mapping[str, str]) -> List[Bone]:
    """Build a list of bones in neutral pose using the provided mapping."""

    bones: List[Bone] = []
    for idx, name in enumerate(template.joint_names):
        target_name = remap.get(name, name)
        parent = template.joint_parents[idx]
        parent_index = parent if parent >= 0 else None
        bones.append(Bone(target_name, parent_index, template.joint_positions[idx]))
    return bones


class UnityUnrealExporter:
    """Export SMPL-X meshes to Unity or Unreal friendly assets."""

    def __init__(self, config: EngineConfig, remap: Mapping[str, str] | None = None) -> None:
        self.config = config
        self.remap: Mapping[str, str] = remap or DEFAULT_BONE_REMAP

    def export(
        self,
        template: SMPLXTemplate,
        output: Path | str,
        fmt: ExportFormat | str,
    ) -> Path:
        """Export the given template to the requested format."""

        template.validate()
        bones = build_neutral_pose(template, self.remap)
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not isinstance(fmt, ExportFormat):
            try:
                fmt = ExportFormat(fmt)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"Unsupported format {fmt!r}") from exc

        if fmt is ExportFormat.GLB:
            data = self._build_glb_payload(template, bones)
            output_path.write_bytes(data)
        elif fmt is ExportFormat.FBX:
            output_path.write_text(self._build_fbx_ascii(template, bones), encoding="utf-8")
        else:
            msg = f"Unsupported format {fmt}"
            raise ValueError(msg)
        return output_path

    # ------------------------------------------------------------------
    def _build_glb_payload(self, template: SMPLXTemplate, bones: Sequence[Bone]) -> bytes:
        positions_scaled = [
            tuple(coord * self.config.unit_scale for coord in vertex)
            for vertex in template.vertices
        ]
        indices = [index for face in template.faces for index in face]

        bone_index_lookup: Dict[int, int] = {idx: idx for idx in range(len(bones))}
        joint_indices: List[Tuple[int, int, int, int]] = []
        for joint_tuple in template.skin_joints:
            mapped = tuple(bone_index_lookup.get(idx, 0) for idx in joint_tuple)
            joint_indices.append(mapped)

        weights = template.skin_weights
        inv_bind_mats = self._compute_inverse_bind_matrices(bones)

        buffer = bytearray()
        buffer_views: List[dict] = []
        accessors: List[dict] = []

        def add_buffer_view(
            data: bytes,
            *,
            target: Optional[int] = None,
            component_type: Optional[int] = None,
            count: Optional[int] = None,
            type_: Optional[str] = None,
            min_: Optional[Iterable[float]] = None,
            max_: Optional[Iterable[float]] = None,
        ) -> Tuple[int, int]:
            offset = self._pad_buffer(buffer)
            buffer.extend(data)
            view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
            if target is not None:
                view["target"] = target
            buffer_views.append(view)
            view_index = len(buffer_views) - 1
            accessor_index = -1
            if component_type is not None and count is not None and type_ is not None:
                accessor = {
                    "bufferView": view_index,
                    "componentType": component_type,
                    "count": count,
                    "type": type_,
                }
                if min_ is not None:
                    accessor["min"] = [float(x) for x in min_]
                if max_ is not None:
                    accessor["max"] = [float(x) for x in max_]
                accessors.append(accessor)
                accessor_index = len(accessors) - 1
            return view_index, accessor_index

        pos_bytes = b"".join(struct.pack("<3f", *vertex) for vertex in positions_scaled)
        min_pos = [min(coords) for coords in zip(*positions_scaled)] if positions_scaled else [0.0, 0.0, 0.0]
        max_pos = [max(coords) for coords in zip(*positions_scaled)] if positions_scaled else [0.0, 0.0, 0.0]
        _, pos_accessor = add_buffer_view(
            pos_bytes,
            target=34962,
            component_type=5126,
            count=len(positions_scaled),
            type_="VEC3",
            min_=min_pos,
            max_=max_pos,
        )

        idx_bytes = b"".join(struct.pack("<I", index) for index in indices)
        _, idx_accessor = add_buffer_view(
            idx_bytes,
            target=34963,
            component_type=5125,
            count=len(indices),
            type_="SCALAR",
            min_=[float(min(indices))] if indices else [0.0],
            max_=[float(max(indices))] if indices else [0.0],
        )

        joints_bytes = b"".join(struct.pack("<4H", *joint) for joint in joint_indices)
        _, joints_accessor = add_buffer_view(
            joints_bytes,
            target=34962,
            component_type=5123,
            count=len(joint_indices),
            type_="VEC4",
        )

        weights_bytes = b"".join(struct.pack("<4f", *weight) for weight in weights)
        _, weights_accessor = add_buffer_view(
            weights_bytes,
            target=34962,
            component_type=5126,
            count=len(weights),
            type_="VEC4",
        )

        bind_bytes = b"".join(struct.pack("<16f", *matrix) for matrix in inv_bind_mats)
        _, bind_accessor = add_buffer_view(
            bind_bytes,
            component_type=5126,
            count=len(inv_bind_mats),
            type_="MAT4",
        )

        nodes, joint_indices_lookup = self._build_nodes(bones)
        mesh_node_index = len(nodes)
        nodes.append({"name": "SMPLXMesh", "mesh": 0, "skin": 0})
        root_index = joint_indices_lookup[0]
        root_children = nodes[root_index].setdefault("children", [])
        if mesh_node_index not in root_children:
            root_children.append(mesh_node_index)

        gltf = {
            "asset": {"version": "2.0", "generator": "SeaMeInIt Exporter"},
            "scene": 0,
            "scenes": [{"nodes": [root_index]}],
            "nodes": nodes,
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": {
                                "POSITION": pos_accessor,
                                "JOINTS_0": joints_accessor,
                                "WEIGHTS_0": weights_accessor,
                            },
                            "indices": idx_accessor,
                        }
                    ]
                }
            ],
            "skins": [
                {
                    "joints": [joint_indices_lookup[i] for i in range(len(bones))],
                    "inverseBindMatrices": bind_accessor,
                    "skeleton": root_index,
                }
            ],
            "buffers": [{"byteLength": self._pad_buffer(buffer)}],
            "bufferViews": buffer_views,
            "accessors": accessors,
        }

        json_chunk = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        json_padded = self._pad_bytes(json_chunk, 4, pad=b" ")
        bin_chunk = self._pad_bytes(bytes(buffer), 4, pad=b"\x00")

        total_length = 12 + 8 + len(json_padded) + 8 + len(bin_chunk)
        header = struct.pack("<4sII", b"glTF", 2, total_length)
        json_header = struct.pack("<I4s", len(json_padded), b"JSON")
        bin_header = struct.pack("<I4s", len(bin_chunk), b"BIN\0")
        return b"".join([header, json_header, json_padded, bin_header, bin_chunk])

    def _compute_inverse_bind_matrices(self, bones: Sequence[Bone]) -> List[List[float]]:
        global_positions: List[Tuple[float, float, float]] = [
            (0.0, 0.0, 0.0) for _ in bones
        ]
        for idx, bone in enumerate(bones):
            local = tuple(coord * self.config.unit_scale for coord in bone.position)
            if bone.parent_index is None:
                global_positions[idx] = local
            else:
                parent = global_positions[bone.parent_index]
                global_positions[idx] = tuple(parent[i] + local[i] for i in range(3))
        matrices: List[List[float]] = []
        for pos in global_positions:
            tx, ty, tz = (-pos[0], -pos[1], -pos[2])
            matrices.append(
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    tx,
                    ty,
                    tz,
                    1.0,
                ]
            )
        return matrices

    def _build_nodes(self, bones: Sequence[Bone]) -> Tuple[List[dict], Dict[int, int]]:
        nodes: List[dict] = []
        index_lookup: Dict[int, int] = {}
        for idx, bone in enumerate(bones):
            translation = [coord * self.config.unit_scale for coord in bone.position]
            node = {
                "name": bone.name,
                "translation": translation,
                "rotation": [0, 0, 0, 1],
            }
            nodes.append(node)
            index_lookup[idx] = idx
        for idx, bone in enumerate(bones):
            if bone.parent_index is not None:
                parent_node = nodes[index_lookup[bone.parent_index]]
                parent_node.setdefault("children", []).append(index_lookup[idx])
        return nodes, index_lookup

    # ------------------------------------------------------------------
    def _build_fbx_ascii(self, template: SMPLXTemplate, bones: Sequence[Bone]) -> str:
        lines = [
            "; FBX 7.4.0 project file",
            f"; Exported for {self.config.name}",
            "",
            "Units:",
            f"  Scale: {self.config.unit_scale}",
            f"  UpAxis: {self.config.up_axis}",
            f"  FrontAxis: {self.config.forward_axis}",
            "Bones:",
        ]
        for idx, bone in enumerate(bones):
            parent_name = "None" if bone.parent_index is None else bones[bone.parent_index].name
            position = [coord * self.config.unit_scale for coord in bone.position]
            lines.append(
                f"  - {bone.name} (parent={parent_name}) pos=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f})"
            )
        lines.append("Mesh:")
        lines.append(f"  Vertices: {len(template.vertices)}")
        lines.append(f"  Faces: {len(template.faces)}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _pad_buffer(buffer: bytearray) -> int:
        padding = (-len(buffer)) % 4
        if padding:
            buffer.extend(b"\x00" * padding)
        return len(buffer)

    @staticmethod
    def _pad_bytes(data: bytes, alignment: int, pad: bytes) -> bytes:
        if len(pad) != 1:
            raise ValueError("pad must be a single byte")
        padding = (-len(data)) % alignment
        if padding:
            data += pad * padding
        return data

