from __future__ import annotations

import numpy as np
import trimesh

from tools import repair_body_mesh as repair


def _build_sample_mesh() -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    # Remove one face to introduce a hole on the primary component.
    box.update_faces(np.arange(len(box.faces) - 1))

    satellite = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    satellite.apply_translation((2.0, 0.0, 0.0))

    vertices = np.vstack((box.vertices, satellite.vertices))
    faces = np.vstack((box.faces, satellite.faces + len(box.vertices)))
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def test_summarise_components_reports_stats() -> None:
    mesh = _build_sample_mesh()
    components = repair.split_components(mesh)
    summaries = repair.summarise_components(components)

    assert [summary.index for summary in summaries] == list(range(len(components)))
    for component, summary in zip(components, summaries, strict=True):
        assert summary.vertex_count == len(component.vertices)
        assert summary.face_count == len(component.faces)


def test_repair_body_component_fills_primary_hole() -> None:
    mesh = _build_sample_mesh()
    components = repair.split_components(mesh)

    primary_index = repair.select_body_component(components)
    assert not components[primary_index].is_watertight

    before, after = repair.repair_body_component(components, component_index=primary_index)

    assert before is False
    assert after is True
    assert components[primary_index].is_watertight

    repaired = repair.assemble_components(components)
    assert isinstance(repaired, trimesh.Trimesh)
    assert len(repaired.vertices) == sum(len(component.vertices) for component in components)
