"""Boundary-region membership classifier (design_docs/boundary_region_v2.md §2).

The single source of truth for "which boundary points belong to this region",
shared by the viewport (hover highlight, selection) and the meshing API
(`MeshableBoundaryRegion.contains`). A world point belongs to a region iff:

1. it lies on the Domain boundary       (|f_root| <= tolerance)
2. the region's owner leaf is the active operand there (spec §4 provenance:
   operators are min/max selections, so the active leaf owns the surface;
   for Subtract the OBSTACLE owns the cut surface)
3. it is within the analytic patch scope, when the region has one
   (patch_id, or the legacy outside_direction normal-alignment test)
4. every cut in the chain is satisfied  (sign of the ghost volume vs side)

Only sign tests of known SDFs — exact, cheap, tessellation-independent.
Tolerances are scale-relative (owner extent), never absolute meters.
"""
from __future__ import annotations

from math import cos, radians, sin

import numpy as np
from numpy.typing import NDArray

from core.boundary import BoundaryCut, BoundaryRegion
from core.boundary_direction import owner_outside_direction_vector
from core.boundary_patches import (
    boundary_region_scope_mask,
    surface_selector_volume,
)
from core.sdf import Difference, Intersection, Rotate, Scale, Translate, Union, Xor
from core.sdf.base import SDFNode

# On-surface band as a fraction of the owner's bounding-box diagonal
# (~ the old absolute PATCH_TOLERANCE=1.5e-3 for the meter-scale default scene).
RELATIVE_SURFACE_TOLERANCE = 1.0e-3
# Matches boundary_direction.owner_outside_direction_from_normal.
DIRECTION_ALIGNMENT_MINIMUM = 0.95

FloatArrays = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


def find_node_by_object_id(root: SDFNode, object_id: int) -> SDFNode | None:
    if root.object_id == object_id:
        return root
    for child in root.children():
        found = find_node_by_object_id(child, object_id)
        if found is not None:
            return found
    return None


def _bounding_diagonal(node: SDFNode) -> float:
    box = node.bounding_box()
    return float(
        np.linalg.norm(
            (
                box.x_max - box.x_min,
                box.y_max - box.y_min,
                box.z_max - box.z_min,
            )
        )
    )


def region_tolerance(root: SDFNode, region: BoundaryRegion) -> float:
    """Scale-relative on-surface band: owner extent when resolvable, else the
    Domain extent."""
    owner = find_node_by_object_id(root, region.owner_object_id)
    reference = owner if owner is not None else root
    diagonal = _bounding_diagonal(reference)
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        diagonal = max(_bounding_diagonal(root), 1.0e-9)
    return RELATIVE_SURFACE_TOLERANCE * diagonal


def _split_points(points: NDArray[np.float64]) -> FloatArrays:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("classifier points must have shape (N, 3)")
    return array[:, 0], array[:, 1], array[:, 2]


def _walk_owner(
    node: SDFNode,
    owner_object_id: int,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    tie: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return (field values, owner-is-active mask) for this subtree.

    Operators attribute activity to whichever operand attains the min/max
    (ties count for both sides); transforms warp the query points exactly as
    their ``to_numpy`` does, so attribution reaches leaves below them.
    Any non-operator, non-transform node is a provenance leaf (matching
    ``_surface_patches_for_node``): generators like Extrude/Revolve and tubes
    own their whole surface.
    """
    if isinstance(node, Translate):
        assert node.child is not None
        ox, oy, oz = node.offset
        return _walk_owner(node.child, owner_object_id, x - ox, y - oy, z - oz, tie)
    if isinstance(node, Scale):
        assert node.child is not None
        factor = float(node.factor)
        values, mask = _walk_owner(
            node.child, owner_object_id, x / factor, y / factor, z / factor,
            tie / factor,
        )
        return values * factor, mask
    if isinstance(node, Rotate):
        assert node.child is not None
        c = cos(radians(node.angle_degrees))
        s = sin(radians(node.angle_degrees))
        if node.axis == "x":
            local = (x, c * y + s * z, -s * y + c * z)
        elif node.axis == "y":
            local = (c * x - s * z, y, s * x + c * z)
        else:
            local = (c * x + s * y, -s * x + c * y, z)
        return _walk_owner(node.child, owner_object_id, *local, tie)
    if isinstance(node, (Union, Intersection, Difference, Xor)):
        assert node.left is not None and node.right is not None
        lv, lm = _walk_owner(node.left, owner_object_id, x, y, z, tie)
        rv, rm = _walk_owner(node.right, owner_object_id, x, y, z, tie)
        if isinstance(node, Union):
            values = np.minimum(lv, rv)
            mask = (lm & (lv <= rv + tie)) | (rm & (rv <= lv + tie))
        elif isinstance(node, Intersection):
            values = np.maximum(lv, rv)
            mask = (lm & (lv >= rv - tie)) | (rm & (rv >= lv - tie))
        elif isinstance(node, Difference):
            values = np.maximum(lv, -rv)
            mask = (lm & (lv >= -rv - tie)) | (rm & (-rv >= lv - tie))
        else:  # Xor = max(min(l, r), -max(l, r))
            inner = np.minimum(lv, rv)
            outer = -np.maximum(lv, rv)
            values = np.maximum(inner, outer)
            inner_active = inner >= outer - tie
            outer_active = outer >= inner - tie
            inner_mask = (lm & (lv <= rv + tie)) | (rm & (rv <= lv + tie))
            outer_mask = (lm & (lv >= rv - tie)) | (rm & (rv >= lv - tie))
            mask = (inner_active & inner_mask) | (outer_active & outer_mask)
        return values, mask
    values = np.asarray(node.to_numpy(x, y, z), dtype=np.float64)
    mask = np.full(values.shape, node.object_id == owner_object_id, dtype=np.bool_)
    return values, mask


def owner_active_mask(
    root: SDFNode,
    owner_object_id: int,
    points: NDArray[np.float64],
    *,
    tie_tolerance: float,
) -> NDArray[np.bool_]:
    x, y, z = _split_points(points)
    _values, mask = _walk_owner(root, owner_object_id, x, y, z, tie_tolerance)
    return mask


def _root_normals(
    root: SDFNode,
    points: NDArray[np.float64],
    step: float,
) -> NDArray[np.float64]:
    x, y, z = _split_points(points)
    gradient = np.stack(
        (
            np.asarray(root.to_numpy(x + step, y, z) - root.to_numpy(x - step, y, z)),
            np.asarray(root.to_numpy(x, y + step, z) - root.to_numpy(x, y - step, z)),
            np.asarray(root.to_numpy(x, y, z + step) - root.to_numpy(x, y, z - step)),
        ),
        axis=1,
    ).astype(np.float64)
    lengths = np.linalg.norm(gradient, axis=1)
    lengths = np.where(lengths > 1.0e-12, lengths, 1.0)
    return gradient / lengths[:, None]


def cut_volume(root: SDFNode, cut: BoundaryCut) -> SDFNode:
    """The classification field of one cut: a 3D ghost as-is, a lower-dim
    ghost extruded through the scene."""
    if cut.ghost.dimension == 3:
        return cut.ghost
    volume = surface_selector_volume(root, cut.ghost)
    if volume is None:
        raise ValueError(
            f"boundary cut ghost {cut.ghost.name!r} cannot be converted to a "
            "classification volume"
        )
    return volume


def sample_boundary_points(
    root: SDFNode,
    *,
    resolution: int = 24,
) -> tuple[NDArray[np.float64], float]:
    """Coarse point cloud near the Domain boundary plus its band width.

    Grid-samples the root bounding box and keeps points within a band of the
    surface proportional to the cell size. Approximate by design — callers
    use it for diagnostics (empty-cut warnings), never for membership."""
    box = root.bounding_box()
    steps = max(int(resolution), 2)
    xs = np.linspace(box.x_min, box.x_max, steps)
    ys = np.linspace(box.y_min, box.y_max, steps)
    zs = np.linspace(box.z_min, box.z_max, steps)
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()), axis=1)
    values = np.asarray(
        root.to_numpy(points[:, 0], points[:, 1], points[:, 2]), dtype=np.float64
    )
    cell = max(
        (box.x_max - box.x_min) / (steps - 1),
        (box.y_max - box.y_min) / (steps - 1),
        (box.z_max - box.z_min) / (steps - 1),
        1.0e-12,
    )
    band = 0.75 * cell
    return points[np.abs(values) <= band], band


def boundary_region_mask(
    root: SDFNode,
    region: BoundaryRegion,
    points: NDArray[np.float64],
    *,
    tolerance: float | None = None,
) -> NDArray[np.bool_]:
    """Exact membership of ``points`` in ``region`` (see module docstring)."""
    array = np.asarray(points, dtype=np.float64)
    x, y, z = _split_points(array)
    tol = float(tolerance) if tolerance is not None else region_tolerance(root, region)

    # 1. on the Domain boundary
    root_values = np.asarray(root.to_numpy(x, y, z), dtype=np.float64)
    mask = np.abs(root_values) <= tol

    # 2. owner leaf active (provenance)
    mask &= owner_active_mask(
        root, region.owner_object_id, array, tie_tolerance=tol
    )

    # 3. analytic patch scope
    if region.patch_id is not None:
        mask &= boundary_region_scope_mask(root, region, array, tolerance=tol)
    elif region.outside_direction is not None:
        owner = find_node_by_object_id(root, region.owner_object_id)
        direction = (
            owner_outside_direction_vector(owner, region.outside_direction)
            if owner is not None
            else None
        )
        if direction is not None:
            normals = _root_normals(root, array, max(tol * 0.5, 1.0e-9))
            alignment = normals @ np.asarray(direction, dtype=np.float64)
            mask &= alignment >= DIRECTION_ALIGNMENT_MINIMUM

    # 4. the cut chain
    for cut in region.cuts:
        volume = cut_volume(root, cut)
        values = np.asarray(volume.to_numpy(x, y, z), dtype=np.float64)
        inside = values <= tol
        mask &= inside if cut.side == "inside" else ~inside

    return mask


__all__ = [
    "DIRECTION_ALIGNMENT_MINIMUM",
    "RELATIVE_SURFACE_TOLERANCE",
    "boundary_region_mask",
    "cut_volume",
    "find_node_by_object_id",
    "owner_active_mask",
    "region_tolerance",
    "sample_boundary_points",
]
