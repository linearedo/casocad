from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .base import BoundingBox3D, FloatArray, SDFNode
from .primitives_2d import BezierCurveProfile, PolylineProfile, Profile2D

CurveProfile2D = BezierCurveProfile | PolylineProfile


def _normalized(vector: tuple[float, float, float]) -> NDArray[np.float64]:
    array = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(array)
    if length <= 1e-12:
        raise ValueError("workplane axes must be nonzero")
    return array / length


def _project_to_workplane(
    origin: tuple[float, float, float],
    axis_u: tuple[float, float, float],
    axis_v: tuple[float, float, float],
    normal: tuple[float, float, float],
    X: FloatArray,
    Y: FloatArray,
    Z: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    ox, oy, oz = origin
    rx, ry, rz = X - ox, Y - oy, Z - oz
    u = rx * axis_u[0] + ry * axis_u[1] + rz * axis_u[2]
    v = rx * axis_v[0] + ry * axis_v[1] + rz * axis_v[2]
    plane = rx * normal[0] + ry * normal[1] + rz * normal[2]
    return (
        np.asarray(u, dtype=np.float64),
        np.asarray(v, dtype=np.float64),
        np.asarray(plane, dtype=np.float64),
    )


def _workplane_normal(
    axis_u: tuple[float, float, float],
    axis_v: tuple[float, float, float],
) -> tuple[float, float, float]:
    normal = np.cross(axis_u, axis_v)
    normal /= np.linalg.norm(normal)
    return tuple(float(value) for value in normal)


def _workplane_corners(
    origin: tuple[float, float, float],
    axis_u: tuple[float, float, float],
    axis_v: tuple[float, float, float],
    bounds: tuple[float, float, float, float],
) -> NDArray[np.float64]:
    u_min, u_max, v_min, v_max = bounds
    origin_array = np.asarray(origin)
    axis_u_array = np.asarray(axis_u)
    axis_v_array = np.asarray(axis_v)
    return np.asarray(
        [
            origin_array + u * axis_u_array + v * axis_v_array
            for u in (u_min, u_max)
            for v in (v_min, v_max)
        ]
    )


def _lies_in_plane_of(
    origin: tuple[float, float, float],
    normal: tuple[float, float, float],
    plane: object,
    tolerance: float = 1e-6,
) -> bool:
    plane_normal = np.asarray(getattr(plane, "normal"), dtype=np.float64)
    plane_origin = np.asarray(getattr(plane, "origin"), dtype=np.float64)
    origin_delta = np.asarray(origin) - plane_origin
    own_normal = np.asarray(normal, dtype=np.float64)
    normal_alignment = abs(float(np.dot(own_normal, plane_normal)))
    return (
        abs(float(np.dot(origin_delta, plane_normal))) <= tolerance
        and abs(1.0 - normal_alignment) <= tolerance
    )


@dataclass
class PlacedSDF2D(SDFNode):
    profile: Profile2D | None = None
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis_u: tuple[float, float, float] = (1.0, 0.0, 0.0)
    axis_v: tuple[float, float, float] = (0.0, 1.0, 0.0)
    sources: tuple[SDFNode, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.profile is None:
            raise ValueError("PlacedSDF2D requires a filled profile")
        u = _normalized(self.axis_u)
        v = _normalized(self.axis_v)
        if abs(float(np.dot(u, v))) > 1e-6:
            raise ValueError("workplane axes must be orthogonal")
        self.axis_u = tuple(float(value) for value in u)
        self.axis_v = tuple(float(value) for value in v)

    @property
    def dimension(self) -> int:
        return 2

    @property
    def kind(self) -> str:
        return "placed_sdf_2d"

    @property
    def normal(self) -> tuple[float, float, float]:
        return _workplane_normal(self.axis_u, self.axis_v)

    def children(self) -> tuple[SDFNode, ...]:
        return self.sources

    def lies_in_plane_of(
        self,
        plane: object,
        tolerance: float = 1e-6,
    ) -> bool:
        return _lies_in_plane_of(self.origin, self.normal, plane, tolerance)

    def is_coplanar_with(self, other: PlacedSDF2D, tolerance: float = 1e-6) -> bool:
        same_axes = (
            np.allclose(self.axis_u, other.axis_u, atol=tolerance)
            and np.allclose(self.axis_v, other.axis_v, atol=tolerance)
        )
        if not same_axes:
            return False
        delta = np.asarray(other.origin) - np.asarray(self.origin)
        return abs(float(np.dot(delta, self.normal))) <= tolerance

    def shares_plane_with(
        self,
        other: PlacedSDF2D,
        tolerance: float = 1e-6,
    ) -> bool:
        normal_alignment = abs(float(np.dot(self.normal, other.normal)))
        if abs(1.0 - normal_alignment) > tolerance:
            return False
        delta = np.asarray(other.origin) - np.asarray(self.origin)
        return abs(float(np.dot(delta, self.normal))) <= tolerance

    def project_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        return _project_to_workplane(
            self.origin,
            self.axis_u,
            self.axis_v,
            self.normal,
            X,
            Y,
            Z,
        )

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        assert self.profile is not None
        u, v, _plane = self.project_numpy(X, Y, Z)
        return self.profile.to_numpy(u, v)

    def bounding_box(self) -> BoundingBox3D:
        assert self.profile is not None
        corners = _workplane_corners(
            self.origin,
            self.axis_u,
            self.axis_v,
            self.profile.bounds(),
        )
        minimum = corners.min(axis=0)
        maximum = corners.max(axis=0)
        padding = 0.002
        return BoundingBox3D(
            minimum[0] - padding,
            maximum[0] + padding,
            minimum[1] - padding,
            maximum[1] + padding,
            minimum[2] - padding,
            maximum[2] + padding,
        )


@dataclass
class PlacedPolyline2D(SDFNode):
    profile: CurveProfile2D | None = None
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis_u: tuple[float, float, float] = (1.0, 0.0, 0.0)
    axis_v: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def __post_init__(self) -> None:
        if self.profile is None:
            raise ValueError("PlacedPolyline2D requires a curve profile")
        u = _normalized(self.axis_u)
        v = _normalized(self.axis_v)
        if abs(float(np.dot(u, v))) > 1e-6:
            raise ValueError("workplane axes must be orthogonal")
        self.axis_u = tuple(float(value) for value in u)
        self.axis_v = tuple(float(value) for value in v)

    @property
    def dimension(self) -> int:
        return 1

    @property
    def kind(self) -> str:
        if isinstance(self.profile, BezierCurveProfile):
            if len(self.profile.points) > 3:
                return "placed_bezier_polycurve_2d"
            return "placed_bezier_curve_2d"
        return "placed_polyline_2d"

    @property
    def normal(self) -> tuple[float, float, float]:
        return _workplane_normal(self.axis_u, self.axis_v)

    def lies_in_plane_of(
        self,
        plane: object,
        tolerance: float = 1e-6,
    ) -> bool:
        return _lies_in_plane_of(self.origin, self.normal, plane, tolerance)

    def project_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        return _project_to_workplane(
            self.origin,
            self.axis_u,
            self.axis_v,
            self.normal,
            X,
            Y,
            Z,
        )

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        assert self.profile is not None
        u, v, _plane = self.project_numpy(X, Y, Z)
        return self.profile.to_numpy(u, v)

    def contains_points(
        self,
        positions: NDArray[np.float64],
        tolerance: float,
    ) -> NDArray[np.bool_]:
        assert self.profile is not None
        u, v, plane = self.project_numpy(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
        )
        return np.asarray(
            (np.abs(plane) <= tolerance)
            & (self.profile.to_numpy(u, v) <= tolerance),
            dtype=np.bool_,
        )

    def bounding_box(self) -> BoundingBox3D:
        assert self.profile is not None
        corners = _workplane_corners(
            self.origin,
            self.axis_u,
            self.axis_v,
            self.profile.bounds(),
        )
        minimum = corners.min(axis=0)
        maximum = corners.max(axis=0)
        padding = 0.004
        return BoundingBox3D(
            minimum[0] - padding,
            maximum[0] + padding,
            minimum[1] - padding,
            maximum[1] + padding,
            minimum[2] - padding,
            maximum[2] + padding,
        )
