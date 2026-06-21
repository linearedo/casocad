from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.sdf.base import FloatArray, SDFNode
from core.sdf.operators import Difference, Intersection, SmoothUnion, Union
from core.sdf.transforms import Rotate, Scale, Translate

NODE_INSIDE = np.uint8(0)
NODE_BOUNDARY = np.uint8(1)
BOUNDARY_OFFSETS = (
    (-1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 0.0, 1.0),
)
BoundaryOffsets = tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class BoundaryFaceSamples:
    """Refined SDF zero crossings for exposed retained-node directions."""

    boundary_faces: NDArray[np.uint8]
    node_indices: NDArray[np.intp]
    directions: NDArray[np.uint8]
    positions: NDArray[np.float64]
    normals: NDArray[np.float64]
    owner_object_ids: NDArray[np.uint16]
    approximation_errors: NDArray[np.float64]


def boundary_owner_ids(node: SDFNode) -> set[int]:
    """Return object IDs that attribution may assign to the final boundary."""
    if isinstance(node, (Translate, Rotate, Scale)):
        assert node.child is not None
        return boundary_owner_ids(node.child)
    if isinstance(node, (Union, Intersection, Difference, SmoothUnion)):
        assert node.left is not None and node.right is not None
        return boundary_owner_ids(node.left) | boundary_owner_ids(node.right)
    children = node.children()
    if children:
        return set().union(*(boundary_owner_ids(child) for child in children))
    return {node.object_id}


def retained_mask(sdf: NDArray[np.float64]) -> NDArray[np.bool_]:
    return np.asarray(sdf <= 0.0, dtype=np.bool_)


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
            *(
                np.asarray([positive[index]], dtype=np.float64)
                for index in range(3)
            )
        )
        negative_distance = root.to_numpy(
            *(
                np.asarray([negative[index]], dtype=np.float64)
                for index in range(3)
            )
        )
        gradient[axis] = positive_distance[0] - negative_distance[0]
    normal = gradient / max(np.linalg.norm(gradient), 1e-12)
    return point, int(object_ids[0]), normal


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


def classify_nodes(
    root: SDFNode,
    X: FloatArray,
    Y: FloatArray,
    Z: FloatArray,
    retained: NDArray[np.bool_],
    dx: float,
    offsets: BoundaryOffsets = BOUNDARY_OFFSETS,
) -> NDArray[np.uint8]:
    """Classify the single retained lattice layer adjacent to outside."""
    faces = classify_boundary_faces(root, X, Y, Z, retained, dx, offsets)
    return np.where(faces != 0, NODE_BOUNDARY, NODE_INSIDE).astype(np.uint8)


def classify_boundary_faces(
    root: SDFNode,
    X: FloatArray,
    Y: FloatArray,
    Z: FloatArray,
    retained: NDArray[np.bool_],
    dx: float,
    offsets: BoundaryOffsets = BOUNDARY_OFFSETS,
) -> NDArray[np.uint8]:
    """Return an outside-neighbor bit mask for each retained node."""
    faces = np.zeros(retained.shape, dtype=np.uint8)
    for bit, (offset_x, offset_y, offset_z) in enumerate(offsets):
        neighbor_sdf = root.to_numpy(
            X + offset_x * dx,
            Y + offset_y * dx,
            Z + offset_z * dx,
        )
        faces[retained & (neighbor_sdf > 0.0)] |= np.uint8(1 << bit)
    return faces[retained]


def sample_boundary_faces(
    root: SDFNode,
    X: FloatArray,
    Y: FloatArray,
    Z: FloatArray,
    retained: NDArray[np.bool_],
    dx: float,
    *,
    refinement_steps: int = 8,
    offsets: BoundaryOffsets = BOUNDARY_OFFSETS,
) -> BoundaryFaceSamples:
    """Refine every inside-to-outside lattice edge to an SDF zero crossing."""
    retained_indices = np.flatnonzero(retained)
    retained_lookup = np.full(retained.shape, -1, dtype=np.intp)
    retained_lookup[retained_indices] = np.arange(
        retained_indices.size,
        dtype=np.intp,
    )
    boundary_faces = np.zeros(retained_indices.size, dtype=np.uint8)
    sample_nodes: list[NDArray[np.intp]] = []
    sample_directions: list[NDArray[np.uint8]] = []
    sample_positions: list[NDArray[np.float64]] = []
    sample_normals: list[NDArray[np.float64]] = []
    sample_owners: list[NDArray[np.uint16]] = []
    sample_errors: list[NDArray[np.float64]] = []
    coordinates = np.column_stack((X, Y, Z))
    inside_sdf = root.to_numpy(X, Y, Z)

    for direction, offset in enumerate(offsets):
        offset_array = np.asarray(offset, dtype=np.float64)
        neighbor_coordinates = coordinates + offset_array * dx
        neighbor_sdf = root.to_numpy(
            neighbor_coordinates[:, 0],
            neighbor_coordinates[:, 1],
            neighbor_coordinates[:, 2],
        )
        exposed = retained & (neighbor_sdf > 0.0)
        if not exposed.any():
            continue
        full_indices = np.flatnonzero(exposed)
        local_indices = retained_lookup[full_indices]
        boundary_faces[local_indices] |= np.uint8(1 << direction)
        inside_points = coordinates[full_indices]
        outside_points = neighbor_coordinates[full_indices]
        low = np.zeros(full_indices.size, dtype=np.float64)
        high = np.ones(full_indices.size, dtype=np.float64)
        low_sdf = inside_sdf[full_indices]
        high_sdf = neighbor_sdf[full_indices]
        denominator = low_sdf - high_sdf
        initial = np.divide(
            low_sdf,
            denominator,
            out=np.full(low_sdf.shape, 0.5, dtype=np.float64),
            where=np.abs(denominator) > 1e-15,
        )
        initial = np.clip(initial, 0.0, 1.0)
        initial_points = inside_points + initial[:, None] * (
            outside_points - inside_points
        )
        initial_sdf = root.to_numpy(
            initial_points[:, 0],
            initial_points[:, 1],
            initial_points[:, 2],
        )
        initial_inside = initial_sdf <= 0.0
        exact_inside = np.abs(low_sdf) <= 1e-14
        exact_initial = np.abs(initial_sdf) <= 1e-14
        low = np.where(initial_inside, initial, low)
        high = np.where(initial_inside, high, initial)
        for _step in range(refinement_steps):
            middle = 0.5 * (low + high)
            middle_points = inside_points + middle[:, None] * (
                outside_points - inside_points
            )
            middle_sdf = root.to_numpy(
                middle_points[:, 0],
                middle_points[:, 1],
                middle_points[:, 2],
            )
            middle_inside = middle_sdf <= 0.0
            low = np.where(middle_inside, middle, low)
            high = np.where(middle_inside, high, middle)
        crossing = inside_points + (0.5 * (low + high))[:, None] * (
            outside_points - inside_points
        )
        crossing[exact_initial] = initial_points[exact_initial]
        crossing[exact_inside] = inside_points[exact_inside]
        # At intersections the final SDF can be zero along a finite edge
        # segment. Attribute the transition from its refined OUTSIDE bracket,
        # where the branch that actually makes the neighbor outside is clear.
        outside_bracket = inside_points + high[:, None] * (
            outside_points - inside_points
        )
        _distance, owners = evaluate_with_attribution(
            root,
            outside_bracket[:, 0],
            outside_bracket[:, 1],
            outside_bracket[:, 2],
        )
        gradient_step = max(dx * 1e-4, 1e-8)
        gradients = np.empty(crossing.shape, dtype=np.float64)
        for axis in range(3):
            axis_offset = np.zeros(3, dtype=np.float64)
            axis_offset[axis] = gradient_step
            positive = crossing + axis_offset
            negative = crossing - axis_offset
            gradients[:, axis] = root.to_numpy(
                positive[:, 0],
                positive[:, 1],
                positive[:, 2],
            ) - root.to_numpy(
                negative[:, 0],
                negative[:, 1],
                negative[:, 2],
            )
        lengths = np.linalg.norm(gradients, axis=1)
        normals = gradients / np.maximum(lengths[:, None], 1e-15)
        sample_nodes.append(local_indices)
        sample_directions.append(
            np.full(local_indices.shape, direction, dtype=np.uint8)
        )
        sample_positions.append(crossing)
        sample_normals.append(normals)
        sample_owners.append(owners)
        sample_errors.append(
            np.linalg.norm(crossing - inside_points, axis=1)
        )

    return BoundaryFaceSamples(
        boundary_faces=boundary_faces,
        node_indices=(
            np.concatenate(sample_nodes)
            if sample_nodes
            else np.empty(0, dtype=np.intp)
        ),
        directions=(
            np.concatenate(sample_directions)
            if sample_directions
            else np.empty(0, dtype=np.uint8)
        ),
        positions=(
            np.concatenate(sample_positions)
            if sample_positions
            else np.empty((0, 3), dtype=np.float64)
        ),
        normals=(
            np.concatenate(sample_normals)
            if sample_normals
            else np.empty((0, 3), dtype=np.float64)
        ),
        owner_object_ids=(
            np.concatenate(sample_owners).astype(np.uint16, copy=False)
            if sample_owners
            else np.empty(0, dtype=np.uint16)
        ),
        approximation_errors=(
            np.concatenate(sample_errors)
            if sample_errors
            else np.empty(0, dtype=np.float64)
        ),
    )


def nearest_tag_mask(
    tag: SDFNode,
    grid: object,
    i: NDArray[np.uint64],
    j: NDArray[np.uint64],
    k: NDArray[np.uint64],
    X: FloatArray,
    Y: FloatArray,
    Z: FloatArray,
) -> NDArray[np.bool_]:
    """Select one nearest lattice layer to a placed 2D object's plane."""
    from core.sdf.placed_2d import PlacedSDF2D

    if not isinstance(tag, PlacedSDF2D) or tag.profile is None:
        raise TypeError("nearest_tag_mask requires PlacedSDF2D")
    normal = np.asarray(tag.normal, dtype=np.float64)
    dominant_axis = int(np.argmax(np.abs(normal)))
    coordinates = (X, Y, Z)
    indices = (i, j, k)
    origins = (grid.x_min, grid.y_min, grid.z_min)
    tag_origin = np.asarray(tag.origin, dtype=np.float64)
    target_coordinate = np.full(X.shape, tag_origin[dominant_axis], dtype=np.float64)
    for axis in range(3):
        if axis == dominant_axis:
            continue
        target_coordinate -= (
            (coordinates[axis] - tag_origin[axis])
            * normal[axis]
            / normal[dominant_axis]
        )
    target_index = np.rint(
        (target_coordinate - origins[dominant_axis]) / grid.dx
    ).astype(np.int64)
    u, v, _plane = tag.project_numpy(X, Y, Z)
    inside_profile = tag.profile.to_numpy(u, v) <= 0.0
    return np.asarray(
        (indices[dominant_axis].astype(np.int64) == target_index)
        & inside_profile,
        dtype=np.bool_,
    )


def _profile_boolean_sources(
    node: SDFNode,
) -> tuple[SDFNode, SDFNode, str, float] | None:
    profile = getattr(node, "profile", None)
    operation = getattr(profile, "operation", None)
    sources = node.children()
    if operation not in {"union", "intersection", "difference", "smooth_union"}:
        return None
    if len(sources) != 2:
        return None
    left, right = sources
    smoothing = float(getattr(profile, "smoothing", 0.0))
    return left, right, operation, smoothing


def _boolean_sources(
    node: SDFNode,
) -> tuple[SDFNode, SDFNode, str, float] | None:
    profile_sources = _profile_boolean_sources(node)
    if profile_sources is not None:
        return profile_sources
    if not isinstance(node, (Union, Intersection, Difference, SmoothUnion)):
        return None
    assert node.left is not None and node.right is not None
    if isinstance(node, Union):
        operation = "union"
    elif isinstance(node, Intersection):
        operation = "intersection"
    elif isinstance(node, Difference):
        operation = "difference"
    else:
        operation = "smooth_union"
    return node.left, node.right, operation, getattr(node, "smoothing", 0.0)


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
        left, right, operation, smoothing = boolean_sources
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
            h = np.clip(
                0.5 + 0.5 * (right_distance - left_distance) / smoothing,
                0.0,
                1.0,
            )
            distance = (
                right_distance * (1.0 - h)
                + left_distance * h
                - smoothing * h * (1.0 - h)
            )
            choose_left = h >= 0.5
        return (
            np.asarray(distance, dtype=np.float64),
            np.where(choose_left, left_ids, right_ids).astype(
                np.uint16, copy=False
            ),
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
    """Identify the constructive volume owner for retained lattice nodes."""
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
        left, right, operation, _smoothing = boolean_sources
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
            choose_left = left_distance <= right_distance
        return np.where(choose_left, left_ids, right_ids).astype(
            np.uint16, copy=False
        )
    return np.full(
        np.broadcast_shapes(X.shape, Y.shape, Z.shape),
        node.object_id,
        dtype=np.uint16,
    )
