"""Shortest on-surface paths for the smooth-polyline boundary knife.

Given points the user clicked ON a Domain boundary, approximate the shortest
path between consecutive clicks that stays on the boundary (the |f| = 0 set):
discrete curve-shortening constrained to the surface — midpoint smoothing
pulls the polyline taut, Newton projection (p -= f * grad f / |grad f|^2)
pulls it back onto the zero set, endpoints stay pinned. The result plus its
surface normals feeds NormalCurtain, the knife's classification field.

Everything is scale-relative to the Domain bounding diagonal, never absolute
meters.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.sdf import NormalCurtain
from core.sdf.base import SDFNode

# Path density/effort relative to the Domain diagonal: one sample per ~2% of
# the diagonal keeps segments locally straight; the iteration budget lets the
# midpoint flow settle across edges and curved sheets.
RELATIVE_SAMPLE_SPACING = 0.02
MINIMUM_SEGMENTS = 8
MAXIMUM_SEGMENTS = 96
SMOOTHING_ITERATIONS = 48
PROJECTION_STEPS = 4

Point3D = tuple[float, float, float]


def _bounding_diagonal(root: SDFNode) -> float:
    box = root.bounding_box()
    diagonal = float(
        np.linalg.norm(
            (
                box.x_max - box.x_min,
                box.y_max - box.y_min,
                box.z_max - box.z_min,
            )
        )
    )
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        return 1.0
    return diagonal


def _evaluate(root: SDFNode, points: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(
        root.to_numpy(points[:, 0], points[:, 1], points[:, 2]),
        dtype=np.float64,
    )


def _gradients(
    root: SDFNode,
    points: NDArray[np.float64],
    step: float,
) -> NDArray[np.float64]:
    gradients = np.empty_like(points)
    for axis in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[axis] = step
        gradients[:, axis] = _evaluate(root, points + offset) - _evaluate(
            root, points - offset
        )
    return gradients / (2.0 * step)


def project_to_surface(
    root: SDFNode,
    points: NDArray[np.float64],
    *,
    step: float,
    iterations: int = PROJECTION_STEPS,
) -> NDArray[np.float64]:
    """Newton-project points onto the |f| = 0 boundary."""
    projected = np.array(points, dtype=np.float64)
    for _ in range(iterations):
        values = _evaluate(root, projected)
        gradients = _gradients(root, projected, step)
        squared = np.maximum(
            np.einsum("ij,ij->i", gradients, gradients), 1.0e-18
        )
        projected -= (values / squared)[:, None] * gradients
    return projected


def _resample_by_arclength(
    points: NDArray[np.float64],
    segment_count: int,
) -> NDArray[np.float64]:
    deltas = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cumulative = np.concatenate(((0.0,), np.cumsum(deltas)))
    total = cumulative[-1]
    if total <= 0.0:
        return points
    targets = np.linspace(0.0, total, segment_count + 1)
    return np.stack(
        [
            np.interp(targets, cumulative, points[:, axis])
            for axis in range(3)
        ],
        axis=1,
    )


def surface_shortest_path(
    root: SDFNode,
    start: Point3D,
    end: Point3D,
    *,
    iterations: int = SMOOTHING_ITERATIONS,
) -> NDArray[np.float64]:
    """Approximate shortest on-boundary path from ``start`` to ``end``.

    Both endpoints must already lie on the boundary (the cutter's clicks are
    ray-picked there); they are pinned exactly."""
    first = np.asarray(start, dtype=np.float64)
    last = np.asarray(end, dtype=np.float64)
    diagonal = _bounding_diagonal(root)
    chord = float(np.linalg.norm(last - first))
    if chord <= 1.0e-9 * diagonal:
        raise ValueError("smooth polyline points must be distinct")
    segment_count = int(
        np.clip(
            np.ceil(chord / (RELATIVE_SAMPLE_SPACING * diagonal)),
            MINIMUM_SEGMENTS,
            MAXIMUM_SEGMENTS,
        )
    )
    parameters = np.linspace(0.0, 1.0, segment_count + 1)
    path = first[None, :] * (1.0 - parameters)[:, None] + last[None, :] * parameters[:, None]
    step = max(diagonal * 1.0e-5, 1.0e-9)
    path = project_to_surface(root, path, step=step)
    path[0], path[-1] = first, last
    for _ in range(int(iterations)):
        path[1:-1] = 0.25 * path[:-2] + 0.5 * path[1:-1] + 0.25 * path[2:]
        path = project_to_surface(root, path, step=step, iterations=1)
        path[0], path[-1] = first, last
    path = _resample_by_arclength(path, segment_count)
    path = project_to_surface(root, path, step=step, iterations=2)
    path[0], path[-1] = first, last
    return path


def surface_path_normals(
    root: SDFNode,
    path: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Unit boundary normals along an on-surface path."""
    diagonal = _bounding_diagonal(root)
    step = max(diagonal * 1.0e-5, 1.0e-9)
    gradients = _gradients(root, np.asarray(path, dtype=np.float64), step)
    lengths = np.linalg.norm(gradients, axis=1)
    valid = lengths > 1.0e-12
    if not valid.any():
        raise ValueError("could not resolve surface normals along the path")
    gradients[valid] /= lengths[valid, None]
    if not valid.all():
        indices = np.arange(valid.size)
        good = indices[valid]
        nearest = good[np.argmin(np.abs(indices[:, None] - good[None, :]), axis=1)]
        gradients[~valid] = gradients[nearest[~valid]]
    return gradients


def smooth_polyline_knife(
    root: SDFNode,
    clicked_points: tuple[Point3D, ...],
    *,
    name: str = "smooth_polyline_knife",
) -> NormalCurtain:
    """Chain shortest on-boundary paths through the clicked points and wrap
    them in the NormalCurtain classification field."""
    distinct: list[Point3D] = []
    threshold = 1.0e-9 * _bounding_diagonal(root)
    for point in clicked_points:
        if not distinct or np.linalg.norm(
            np.subtract(point, distinct[-1])
        ) > threshold:
            distinct.append(point)
    if len(distinct) < 2:
        raise ValueError("smooth polyline needs at least two distinct points")
    legs = []
    for start, end in zip(distinct, distinct[1:]):
        leg = surface_shortest_path(root, start, end)
        legs.append(leg if not legs else leg[1:])
    path = np.concatenate(legs, axis=0)
    normals = surface_path_normals(root, path)
    box = root.bounding_box()
    extent = 4.0 * max(
        box.x_max - box.x_min,
        box.y_max - box.y_min,
        box.z_max - box.z_min,
        1.0e-6,
    )
    return NormalCurtain(
        name=name,
        object_id=0,
        points=tuple(tuple(float(v) for v in point) for point in path),
        normals=tuple(tuple(float(v) for v in normal) for normal in normals),
        extent=extent,
    )


__all__ = [
    "project_to_surface",
    "smooth_polyline_knife",
    "surface_path_normals",
    "surface_shortest_path",
]
