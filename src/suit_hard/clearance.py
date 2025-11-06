"""Utilities for simulating hard-shell clearance across pose sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "Mesh",
    "ContactPoint",
    "PoseClearance",
    "ClearanceResult",
    "interpolate_poses",
    "analyze_clearance",
]


@dataclass(slots=True)
class Mesh:
    """Simple triangle mesh container used for clearance checks."""

    vertices: np.ndarray
    faces: np.ndarray
    _triangles: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        verts = np.asarray(self.vertices, dtype=float)
        faces = np.asarray(self.faces, dtype=int)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("Mesh vertices must be an (N, 3) array.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Mesh faces must be an (M, 3) index array.")
        if np.any(faces < 0) or np.any(faces >= len(verts)):
            raise ValueError("Mesh faces reference invalid vertex indices.")
        self.vertices = verts
        self.faces = faces
        self._triangles = verts[faces]

    @property
    def triangles(self) -> np.ndarray:
        return self._triangles

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    def signed_distance(self, point: np.ndarray) -> tuple[float, np.ndarray]:
        """Return the signed distance and closest point on the mesh to *point*."""

        closest = None
        min_distance = float("inf")
        for tri in self.triangles:
            distance, nearest = _point_to_triangle_distance(point, tri)
            if distance < min_distance:
                min_distance = distance
                closest = nearest
        assert closest is not None  # mesh has at least one triangle

        if _point_inside_mesh(point, self.triangles):
            return -min_distance, closest
        return float(min_distance), closest


@dataclass(slots=True)
class ContactPoint:
    """Information about an interpenetration between shell and target."""

    vertex_index: int
    position: np.ndarray
    nearest_point: np.ndarray
    penetration: float
    normal: np.ndarray


@dataclass(slots=True)
class PoseClearance:
    """Per-pose summary of clearance metrics."""

    index: int
    transform: np.ndarray
    min_clearance: float
    max_penetration: float
    contacts: list[ContactPoint] = field(default_factory=list)


@dataclass(slots=True)
class ClearanceResult:
    """Aggregate clearance metrics for a simulation run."""

    shell: Mesh
    target: Mesh
    poses: list[PoseClearance]

    @property
    def worst_penetration(self) -> float:
        if not self.poses:
            return 0.0
        return max(pose.max_penetration for pose in self.poses)

    @property
    def best_clearance(self) -> float:
        if not self.poses:
            return 0.0
        return min(pose.min_clearance for pose in self.poses)

    @property
    def recommended_offset(self) -> np.ndarray:
        accumulation = np.zeros(3, dtype=float)
        count = 0
        for pose in self.poses:
            for contact in pose.contacts:
                accumulation += contact.normal * contact.penetration
                count += 1
        if count == 0:
            return accumulation
        return accumulation / float(count)

    def to_dict(self) -> dict[str, object]:
        """Serialise the result into JSON-friendly structures."""

        return {
            "worst_penetration": float(self.worst_penetration),
            "best_clearance": float(self.best_clearance),
            "recommended_offset": self.recommended_offset.tolist(),
            "poses": [
                {
                    "index": pose.index,
                    "min_clearance": float(pose.min_clearance),
                    "max_penetration": float(pose.max_penetration),
                    "contact_count": len(pose.contacts),
                    "contacts": [
                        {
                            "vertex_index": contact.vertex_index,
                            "penetration": float(contact.penetration),
                            "position": contact.position.tolist(),
                            "nearest_point": contact.nearest_point.tolist(),
                            "normal": contact.normal.tolist(),
                        }
                        for contact in pose.contacts
                    ],
                }
                for pose in self.poses
            ],
        }


def analyze_clearance(
    shell: Mesh,
    target: Mesh,
    transforms: Sequence[np.ndarray],
) -> ClearanceResult:
    """Simulate shell clearance for the target over the provided pose transforms."""

    poses: list[PoseClearance] = []
    target_vertices = target.vertices
    for index, transform in enumerate(transforms):
        transformed_vertices = _apply_transform(target_vertices, transform)
        pose_contacts: list[ContactPoint] = []
        min_clearance = float("inf")
        max_penetration = 0.0
        for vertex_index, vertex in enumerate(transformed_vertices):
            signed_distance, nearest = shell.signed_distance(vertex)
            min_clearance = min(min_clearance, signed_distance)
            if signed_distance < 0.0:
                penetration = -signed_distance
                direction = nearest - vertex
                normal = _safe_normalise(direction)
                pose_contacts.append(
                    ContactPoint(
                        vertex_index=vertex_index,
                        position=vertex,
                        nearest_point=nearest,
                        penetration=penetration,
                        normal=normal,
                    )
                )
                max_penetration = max(max_penetration, penetration)
        if min_clearance is float("inf"):
            min_clearance = 0.0
        poses.append(
            PoseClearance(
                index=index,
                transform=transform,
                min_clearance=min_clearance,
                max_penetration=max_penetration,
                contacts=pose_contacts,
            )
        )
    return ClearanceResult(shell=shell, target=target, poses=poses)


def interpolate_poses(
    key_transforms: Sequence[np.ndarray],
    *,
    samples_per_segment: int = 0,
) -> list[np.ndarray]:
    """Linearly interpolate between successive transforms."""

    if not key_transforms:
        return []
    if samples_per_segment < 0:
        raise ValueError("samples_per_segment must be non-negative")
    result: list[np.ndarray] = []
    for index, current in enumerate(key_transforms[:-1]):
        next_transform = key_transforms[index + 1]
        result.append(current)
        for step in range(1, samples_per_segment + 1):
            alpha = step / float(samples_per_segment + 1)
            interpolated = (1.0 - alpha) * current + alpha * next_transform
            result.append(interpolated)
    result.append(key_transforms[-1])
    return result


def _apply_transform(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if transform.shape == (4, 4):
        homogenous = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
        transformed = homogenous @ transform.T
        return transformed[:, :3]
    if transform.shape == (3, 3):
        return vertices @ transform.T
    raise ValueError("Transforms must be 3x3 or 4x4 matrices.")


def _point_to_triangle_distance(
    point: np.ndarray, triangle: np.ndarray
) -> tuple[float, np.ndarray]:
    a, b, c = triangle
    ab = b - a
    ac = c - a
    ap = point - a

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap)), a

    bp = point - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp)), b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        projection = a + v * ab
        return float(np.linalg.norm(point - projection)), projection

    cp = point - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp)), c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        projection = a + w * ac
        return float(np.linalg.norm(point - projection)), projection

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        projection = b + w * (c - b)
        return float(np.linalg.norm(point - projection)), projection

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    projection = a + ab * v + ac * w
    return float(np.linalg.norm(point - projection)), projection


def _point_inside_mesh(point: np.ndarray, triangles: np.ndarray) -> bool:
    direction = np.array([1.0, 0.31415926, 0.15926536])
    offset_point = point + direction * 1e-9
    count = 0
    for tri in triangles:
        if _ray_intersects_triangle(offset_point, direction, tri):
            count += 1
    return (count % 2) == 1


def _ray_intersects_triangle(
    origin: np.ndarray, direction: np.ndarray, triangle: np.ndarray
) -> bool:
    epsilon = 1e-9
    v0, v1, v2 = triangle
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(direction, edge2)
    a = np.dot(edge1, h)
    if -epsilon < a < epsilon:
        return False
    f = 1.0 / a
    s = origin - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False
    q = np.cross(s, edge1)
    v = f * np.dot(direction, q)
    if v < 0.0 or u + v > 1.0:
        return False
    t = f * np.dot(edge2, q)
    return t > epsilon


def _safe_normalise(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return np.array([0.0, 0.0, 1.0])
    return vector / norm
