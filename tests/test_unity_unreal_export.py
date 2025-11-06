from __future__ import annotations

from pathlib import Path

from src.exporters.unity_unreal_export import (
    ExportFormat,
    UNITY_CONFIG,
    UnityUnrealExporter,
    build_neutral_pose,
    load_smplx_template,
)
from tests.helpers.glb import parse_glb


def _create_template():
    vertices = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.1, 0.5, 0.0],
        [-0.1, 0.5, 0.0],
    ]
    faces = [[0, 1, 2], [0, 1, 3]]
    joint_names = ["pelvis", "spine1", "neck", "head"]
    joint_parents = [-1, 0, 1, 2]
    joint_positions = [
        [0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.2, 0.0],
        [0.0, 0.3, 0.0],
    ]
    skin_weights = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    skin_joints = [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [3, 0, 0, 0],
    ]
    return load_smplx_template(
        vertices=vertices,
        faces=faces,
        joint_names=joint_names,
        joint_parents=joint_parents,
        joint_positions=joint_positions,
        skin_weights=skin_weights,
        skin_joints=skin_joints,
    )


def test_neutral_pose_mapping():
    template = _create_template()
    bones = build_neutral_pose(template, {})
    assert [bone.name for bone in bones] == template.joint_names
    assert bones[1].parent_index == 0


def test_export_glb_contains_expected_bones(tmp_path: Path):
    template = _create_template()
    exporter = UnityUnrealExporter(UNITY_CONFIG)
    glb_path = exporter.export(template, tmp_path / "dummy.glb", ExportFormat.GLB)

    json_data, _ = parse_glb(glb_path)
    joints = json_data["skins"][0]["joints"]
    node_names = [json_data["nodes"][idx]["name"] for idx in joints]
    assert node_names == ["Hips", "Spine", "Neck", "Head"]

    root_children = json_data["nodes"][joints[0]]["children"]
    mesh_node_index = next(
        idx for idx in root_children if json_data["nodes"][idx].get("mesh") == 0
    )
    mesh_node = json_data["nodes"][mesh_node_index]
    assert mesh_node["name"] == "SMPLXMesh"

    translation = json_data["nodes"][joints[1]]["translation"]
    assert translation == [0.0, 0.1, 0.0]


def test_export_fbx_ascii_has_engine_metadata(tmp_path: Path):
    template = _create_template()
    exporter = UnityUnrealExporter(UNITY_CONFIG)
    fbx_path = exporter.export(template, tmp_path / "dummy.fbx", ExportFormat.FBX)

    content = fbx_path.read_text(encoding="utf-8")
    assert "Scale: 1.0" in content
    assert "UpAxis: Y" in content
    assert "- Spine (parent=Hips)" in content
    assert "Vertices: 4" in content


def test_export_rejects_unknown_format(tmp_path: Path):
    template = _create_template()
    exporter = UnityUnrealExporter(UNITY_CONFIG)
    try:
        exporter.export(template, tmp_path / "dummy.txt", "obj")  # type: ignore[arg-type]
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unsupported format")

