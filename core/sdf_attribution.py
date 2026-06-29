from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.sdf.base import FloatArray, SDFNode
from core.sdf.operators import Difference, Intersection, Union, Xor
from core.sdf.transforms import Rotate, Scale, Translate


def boundary_owner_ids(node: SDFNode) -> set[int]:
    """Return object IDs that attribution may assign to the final boundary."""
    if isinstance(node, (Translate, Rotate, Scale)):
        assert node.child is not None
        return boundary_owner_ids(node.child)
    if isinstance(node, (Union, Intersection, Difference, Xor)):
        assert node.left is not None and node.right is not None
        return boundary_owner_ids(node.left) | boundary_owner_ids(node.right)
    children = node.children()
    if children:
        return set().union(*(boundary_owner_ids(child) for child in children))
    return {node.object_id}


def pick_sdf_surface(
    root: SDFNode,
    ray_origin: NDArray[np.float64],
    ray_direction: NDArray[np.float64],
    *,
    hit_tolerance: float = 0.0008,
    maximum_travel: float = 100.0,
) -> NDArray[np.float64] | None:
    """Ray-march an SDF and return the first visible surface point."""
    travel = 0.0
    for _step in range(160):
        point = ray_origin + ray_direction * travel
        coordinates = tuple(
            np.asarray([point[index]], dtype=np.float64) for index in range(3)
        )
        value = float(root.to_numpy(*coordinates)[0])
        if abs(value) < hit_tolerance:
            return point
        travel += max(abs(value), 0.0002)
        if travel > maximum_travel:
            break
    return None


def pick_boundary_owner(
    root: SDFNode,
    ray_origin: NDArray[np.float64],
    ray_direction: NDArray[np.float64],
    *,
    hit_tolerance: float = 0.0008,
    maximum_travel: float = 100.0,
) -> tuple[NDArray[np.float64], int, NDArray[np.float64]] | None:
    """Ray-march the final SDF and return its controlling boundary owner."""
    point = pick_sdf_surface(
        root,
        ray_origin,
        ray_direction,
        hit_tolerance=hit_tolerance,
        maximum_travel=maximum_travel,
    )
    if point is None:
        return None
    coordinates = tuple(
        np.asarray([point[index]], dtype=np.float64) for index in range(3)
    )
    _distance, object_ids = evaluate_with_attribution(root, *coordinates)
    gradient = np.empty(3, dtype=np.float64)
    for axis in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[axis] = hit_tolerance
        positive = point + offset
        negative = point - offset
        positive_distance = root.to_numpy(
            *(np.asarray([positive[index]], dtype=np.float64) for index in range(3))
        )
        negative_distance = root.to_numpy(
            *(np.asarray([negative[index]], dtype=np.float64) for index in range(3))
        )
        gradient[axis] = positive_distance[0] - negative_distance[0]
    normal = gradient / max(np.linalg.norm(gradient), 1e-12)
    return point, int(object_ids[0]), normal


def _profile_boolean_sources(
    node: SDFNode,
) -> tuple[SDFNode, SDFNode, str] | None:
    profile = getattr(node, "profile", None)
    operation = getattr(profile, "operation", None)
    sources = node.children()
    if operation not in {"union", "intersection", "difference", "xor"}:
        return None
    if len(sources) != 2:
        return None
    left, right = sources
    return left, right, operation


def _boolean_sources(
    node: SDFNode,
) -> tuple[SDFNode, SDFNode, str] | None:
    profile_sources = _profile_boolean_sources(node)
    if profile_sources is not None:
        return profile_sources
    if not isinstance(node, (Union, Intersection, Difference, Xor)):
        return None
    assert node.left is not None and node.right is not None
    if isinstance(node, Union):
        operation = "union"
    elif isinstance(node, Intersection):
        operation = "intersection"
    elif isinstance(node, Xor):
        operation = "xor"
    else:
        operation = "difference"
    return node.left, node.right, operation


def evaluate_with_attribution(
    node: SDFNode,
    X: FloatArray,
    Y: FloatArray,
    Z: FloatArray,
) -> tuple[FloatArray, NDArray[np.uint16]]:
    """Evaluate SDF and identify the graph object controlling the value."""
    if isinstance(node, Translate):
        assert node.child is not None
        ox, oy, oz = node.offset
        return evaluate_with_attribution(node.child, X - ox, Y - oy, Z - oz)
    if isinstance(node, Scale):
        assert node.child is not None
        distance, object_ids = evaluate_with_attribution(
            node.child,
            X / node.factor,
            Y / node.factor,
            Z / node.factor,
        )
        return np.asarray(distance * node.factor, dtype=np.float64), object_ids
    if isinstance(node, Rotate):
        assert node.child is not None
        angle = np.deg2rad(node.angle_degrees)
        c, s = np.cos(angle), np.sin(angle)
        if node.axis == "x":
            local = (X, c * Y + s * Z, -s * Y + c * Z)
        elif node.axis == "y":
            local = (c * X - s * Z, Y, s * X + c * Z)
        else:
            local = (c * X + s * Y, -s * X + c * Y, Z)
        return evaluate_with_attribution(node.child, *local)
    boolean_sources = _boolean_sources(node)
    if boolean_sources is not None:
        left, right, operation = boolean_sources
        left_distance, left_ids = evaluate_with_attribution(left, X, Y, Z)
        right_distance, right_ids = evaluate_with_attribution(right, X, Y, Z)
        if operation == "union":
            choose_left = left_distance <= right_distance
            distance = np.minimum(left_distance, right_distance)
        elif operation == "intersection":
            choose_left = left_distance >= right_distance
            distance = np.maximum(left_distance, right_distance)
        elif operation == "difference":
            choose_left = left_distance >= -right_distance
            distance = np.maximum(left_distance, -right_distance)
        else:
            minimum = np.minimum(left_distance, right_distance)
            negative_maximum = -np.maximum(left_distance, right_distance)
            choose_minimum = minimum >= negative_maximum
            choose_left = np.where(
                choose_minimum,
                left_distance <= right_distance,
                left_distance >= right_distance,
            )
            distance = np.maximum(minimum, negative_maximum)
        return (
            np.asarray(distance, dtype=np.float64),
            np.where(choose_left, left_ids, right_ids).astype(np.uint16, copy=False),
        )
    distance = node.to_numpy(X, Y, Z)
    object_ids = np.full(distance.shape, node.object_id, dtype=np.uint16)
    return np.asarray(distance, dtype=np.float64), object_ids


def evaluate_volume_attribution(
    node: SDFNode,
    X: FloatArray,
    Y: FloatArray,
    Z: FloatArray,
) -> NDArray[np.uint16]:
    """Identify the constructive volume owner for interior SDF samples."""
    if isinstance(node, Translate):
        assert node.child is not None
        ox, oy, oz = node.offset
        return evaluate_volume_attribution(node.child, X - ox, Y - oy, Z - oz)
    if isinstance(node, Scale):
        assert node.child is not None
        return evaluate_volume_attribution(
            node.child,
            X / node.factor,
            Y / node.factor,
            Z / node.factor,
        )
    if isinstance(node, Rotate):
        assert node.child is not None
        angle = np.deg2rad(node.angle_degrees)
        c, s = np.cos(angle), np.sin(angle)
        if node.axis == "x":
            local = (X, c * Y + s * Z, -s * Y + c * Z)
        elif node.axis == "y":
            local = (c * X - s * Z, Y, s * X + c * Z)
        else:
            local = (c * X + s * Y, -s * X + c * Y, Z)
        return evaluate_volume_attribution(node.child, *local)
    boolean_sources = _boolean_sources(node)
    if boolean_sources is not None:
        left, right, operation = boolean_sources
        if operation == "difference":
            return evaluate_volume_attribution(left, X, Y, Z)
        left_distance = left.to_numpy(X, Y, Z)
        right_distance = right.to_numpy(X, Y, Z)
        left_ids = evaluate_volume_attribution(left, X, Y, Z)
        right_ids = evaluate_volume_attribution(right, X, Y, Z)
        if operation == "union":
            choose_left = left_distance <= right_distance
        elif operation == "intersection":
            choose_left = left_distance >= right_distance
        else:
            choose_left = (left_distance < 0.0) & (right_distance >= 0.0)
        return np.where(choose_left, left_ids, right_ids).astype(
            np.uint16,
            copy=False,
        )
    return np.full(
        np.broadcast_shapes(X.shape, Y.shape, Z.shape),
        node.object_id,
        dtype=np.uint16,
    )


__all__ = [
    "boundary_owner_ids",
    "evaluate_volume_attribution",
    "evaluate_with_attribution",
    "pick_boundary_owner",
    "pick_sdf_surface",
]
