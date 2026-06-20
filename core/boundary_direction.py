from __future__ import annotations

import numpy as np

from core.sdf import Box, BoxFrame, CappedCone, Cone, Cylinder, PlacedSDF2D, Pyramid, Torus
from core.sdf.base import SDFNode

AXIS_ALIGNED_DIRECTIONS = (
    (-1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 0.0, 1.0),
)


def owner_outside_direction_from_normal(
    owner: SDFNode,
    normal: tuple[float, float, float],
) -> int | None:
    normal_array = np.asarray(normal, dtype=np.float64)
    length = np.linalg.norm(normal_array)
    if length <= 1.0e-12:
        return None
    normal_array /= length
    directions = owner_outside_direction_vectors(owner)
    if directions is None:
        return None
    scores = tuple(
        float(np.dot(normal_array, direction))
        for direction in directions
    )
    direction = int(np.argmax(scores))
    if scores[direction] < 0.95:
        return None
    return direction


def owner_outside_direction_vector(
    owner: SDFNode,
    outside_direction: int | None,
) -> np.ndarray | None:
    if outside_direction is None:
        return None
    directions = owner_outside_direction_vectors(owner)
    if directions is None or not 0 <= outside_direction < len(directions):
        return None
    return directions[outside_direction]


def owner_world_axis_direction(
    owner: SDFNode,
    outside_direction: int | None,
) -> int | None:
    direction = owner_outside_direction_vector(owner, outside_direction)
    if direction is None:
        return None
    return world_axis_direction_from_vector(direction)


def owner_outside_direction_vectors(owner: SDFNode) -> tuple[np.ndarray, ...] | None:
    if isinstance(owner, PlacedSDF2D):
        axis_u = np.asarray(owner.axis_u, dtype=np.float64)
        axis_v = np.asarray(owner.axis_v, dtype=np.float64)
        return (-axis_u, axis_u, -axis_v, axis_v)
    if isinstance(owner, (Box, BoxFrame, CappedCone, Cone, Cylinder, Pyramid, Torus)):
        axis_u = np.asarray(owner.axis_u, dtype=np.float64)
        axis_v = np.asarray(owner.axis_v, dtype=np.float64)
        axis_w = np.asarray(owner.axis_w, dtype=np.float64)
        return (-axis_u, axis_u, -axis_v, axis_v, -axis_w, axis_w)
    return tuple(
        np.asarray(direction, dtype=np.float64)
        for direction in AXIS_ALIGNED_DIRECTIONS
    )


def world_axis_direction_from_vector(vector: np.ndarray) -> int | None:
    axis = int(np.argmax(np.abs(vector)))
    if abs(float(vector[axis])) < 0.999:
        return None
    return 2 * axis + int(vector[axis] > 0.0)
