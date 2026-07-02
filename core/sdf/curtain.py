"""Side-of-curtain classification field for on-surface knife paths.

The smooth-polyline boundary knife stores a dense path that lies ON a Domain
boundary plus the unit surface normals there. Its classification field is the
signed distance to the ruled "curtain" swept from the path along the normals:
the zero set follows the drawn path through the Domain thickness, and the
sign is which side of the curtain a query point falls on (per-segment
binormals, cross(tangent, normal)). This generalizes the straight half-plane
segment knife to curved boundaries. Classification-only ghost geometry —
never a scene object, never rendered as SDF.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BoundingBox3D, FloatArray, SDFNode

Point3D = tuple[float, float, float]

# Bound the M x segments temporaries when classifying dense point sets.
_QUERY_CHUNK = 8192


@dataclass
class NormalCurtain(SDFNode):
    points: tuple[Point3D, ...] = ((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0))
    normals: tuple[Point3D, ...] = ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    extent: float = 4.0

    def __post_init__(self) -> None:
        points = _as_points(self.points)
        normals = _as_points(self.normals)
        if len(points) != len(normals):
            raise ValueError("curtain needs one normal per path point")
        points, normals = _drop_duplicate_path_points(points, normals)
        if len(points) < 2:
            raise ValueError("curtain requires at least two distinct path points")
        if not np.isfinite(self.extent) or self.extent <= 0.0:
            raise ValueError("curtain extent must be finite and positive")
        self.points = points
        self.normals = _unit_tuples(normals, "curtain normals must be nonzero")
        self._segment_binormals()  # validate the frame is well-defined

    @property
    def dimension(self) -> int:
        return 3

    def _segment_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(path points, segment vectors, squared lengths, unit binormals)."""
        cached = getattr(self, "_segment_cache", None)
        if cached is not None:
            return cached
        path = np.asarray(self.points, dtype=np.float64)
        tangents = path[1:] - path[:-1]
        squared_lengths = np.einsum("ij,ij->i", tangents, tangents)
        binormals = self._segment_binormals()
        cache = (path, tangents, squared_lengths, binormals)
        self._segment_cache = cache
        return cache

    def _segment_binormals(self) -> np.ndarray:
        path = np.asarray(self.points, dtype=np.float64)
        normals = np.asarray(self.normals, dtype=np.float64)
        tangents = path[1:] - path[:-1]
        binormals = np.cross(tangents, 0.5 * (normals[:-1] + normals[1:]))
        lengths = np.linalg.norm(binormals, axis=1)
        valid = lengths > 1.0e-12
        if not valid.any():
            raise ValueError("curtain path is parallel to its surface normals")
        binormals[valid] /= lengths[valid, None]
        if not valid.all():
            # A path leg momentarily parallel to its normal has no side of its
            # own; borrow the nearest well-defined segment's.
            indices = np.arange(valid.size)
            good = indices[valid]
            nearest = good[
                np.argmin(np.abs(indices[:, None] - good[None, :]), axis=1)
            ]
            binormals[~valid] = binormals[nearest[~valid]]
        return binormals

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        x, y, z = np.broadcast_arrays(
            np.asarray(X, dtype=np.float64),
            np.asarray(Y, dtype=np.float64),
            np.asarray(Z, dtype=np.float64),
        )
        shape = x.shape
        queries = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
        path, tangents, squared_lengths, binormals = self._segment_arrays()
        values = np.empty(queries.shape[0], dtype=np.float64)
        for start in range(0, queries.shape[0], _QUERY_CHUNK):
            block = queries[start:start + _QUERY_CHUNK]
            offsets = block[:, None, :] - path[None, :-1, :]
            along = np.clip(
                np.einsum("msj,sj->ms", offsets, tangents)
                / squared_lengths[None, :],
                0.0,
                1.0,
            )
            rejections = offsets - along[:, :, None] * tangents[None, :, :]
            distances_sq = np.einsum("msj,msj->ms", rejections, rejections)
            nearest = np.argmin(distances_sq, axis=1)
            rows = np.arange(block.shape[0])
            side = np.einsum(
                "mj,mj->m", rejections[rows, nearest], binormals[nearest]
            )
            values[start:start + _QUERY_CHUNK] = np.where(
                side < 0.0, -1.0, 1.0
            ) * np.sqrt(distances_sq[rows, nearest])
        return values.reshape(shape)

    def bounding_box(self) -> BoundingBox3D:
        path = np.asarray(self.points, dtype=np.float64)
        minimum = path.min(axis=0) - self.extent
        maximum = path.max(axis=0) + self.extent
        return BoundingBox3D(
            float(minimum[0]),
            float(maximum[0]),
            float(minimum[1]),
            float(maximum[1]),
            float(minimum[2]),
            float(maximum[2]),
        )


def _as_points(points) -> tuple[Point3D, ...]:
    return tuple(
        (float(point[0]), float(point[1]), float(point[2])) for point in points
    )


def _drop_duplicate_path_points(
    points: tuple[Point3D, ...],
    normals: tuple[Point3D, ...],
) -> tuple[tuple[Point3D, ...], tuple[Point3D, ...]]:
    if not points:
        return points, normals
    kept_points = [points[0]]
    kept_normals = [normals[0]]
    for point, normal in zip(points[1:], normals[1:]):
        if np.linalg.norm(np.subtract(point, kept_points[-1])) > 1.0e-12:
            kept_points.append(point)
            kept_normals.append(normal)
    return tuple(kept_points), tuple(kept_normals)


def _unit_tuples(vectors: tuple[Point3D, ...], message: str) -> tuple[Point3D, ...]:
    result = []
    for vector in vectors:
        length = float(np.linalg.norm(vector))
        if length <= 1.0e-12:
            raise ValueError(message)
        result.append(tuple(float(component) / length for component in vector))
    return tuple(result)
