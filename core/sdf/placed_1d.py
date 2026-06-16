from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .base import BoundingBox3D, FloatArray, SDFNode, glsl_float, glsl_vec3
from .primitives_1d import Profile1D


def _normalized(vector: tuple[float, float, float]) -> NDArray[np.float64]:
    array = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(array)
    if length <= 1e-12:
        raise ValueError("line axis must be nonzero")
    return array / length


@dataclass
class PlacedSDF1D(SDFNode):
    profile: Profile1D | None = None
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis_u: tuple[float, float, float] = (1.0, 0.0, 0.0)
    sources: tuple[SDFNode, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.profile is None:
            raise ValueError("PlacedSDF1D requires a filled profile")
        axis = _normalized(self.axis_u)
        self.axis_u = tuple(float(value) for value in axis)

    @property
    def dimension(self) -> int:
        return 1

    @property
    def kind(self) -> str:
        return "placed_sdf_1d"

    def children(self) -> tuple[SDFNode, ...]:
        return self.sources

    def is_collinear_with(
        self,
        other: PlacedSDF1D,
        tolerance: float = 1e-6,
    ) -> bool:
        if not np.allclose(self.axis_u, other.axis_u, atol=tolerance):
            return False
        delta = np.asarray(other.origin) - np.asarray(self.origin)
        axis = np.asarray(self.axis_u)
        perpendicular = delta - np.dot(delta, axis) * axis
        return float(np.linalg.norm(perpendicular)) <= tolerance

    def lies_in_plane_of(
        self,
        plane: object,
        tolerance: float = 1e-6,
    ) -> bool:
        normal = np.asarray(getattr(plane, "normal"), dtype=np.float64)
        plane_origin = np.asarray(getattr(plane, "origin"), dtype=np.float64)
        origin_delta = np.asarray(self.origin) - plane_origin
        return (
            abs(float(np.dot(origin_delta, normal))) <= tolerance
            and abs(float(np.dot(self.axis_u, normal))) <= tolerance
        )

    def project_numpy(
        self,
        X: FloatArray,
        Y: FloatArray,
        Z: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        axis = np.asarray(self.axis_u, dtype=np.float64)
        rx = X - self.origin[0]
        ry = Y - self.origin[1]
        rz = Z - self.origin[2]
        coordinate = rx * axis[0] + ry * axis[1] + rz * axis[2]
        perpendicular_x = rx - coordinate * axis[0]
        perpendicular_y = ry - coordinate * axis[1]
        perpendicular_z = rz - coordinate * axis[2]
        return (
            np.asarray(coordinate, dtype=np.float64),
            np.asarray(
                np.sqrt(
                    perpendicular_x**2
                    + perpendicular_y**2
                    + perpendicular_z**2
                ),
                dtype=np.float64,
            ),
        )

    def _project_glsl(self, p_var: str) -> tuple[str, str]:
        local = f"({p_var} - {glsl_vec3(self.origin)})"
        coordinate = f"dot({local}, {glsl_vec3(self.axis_u)})"
        radial = (
            f"length({local} - {coordinate} * {glsl_vec3(self.axis_u)})"
        )
        return coordinate, radial

    def to_glsl(self, p_var: str = "p") -> str:
        assert self.profile is not None
        coordinate, radial = self._project_glsl(p_var)
        profile = self.profile.to_glsl(coordinate)
        thickness = glsl_float(0.004)
        return f"max({profile}, {radial} - {thickness})"

    def to_numpy(
        self,
        X: FloatArray,
        Y: FloatArray,
        Z: FloatArray,
    ) -> FloatArray:
        assert self.profile is not None
        coordinate, _radial = self.project_numpy(X, Y, Z)
        return self.profile.to_numpy(coordinate)

    def contains_points(
        self,
        positions: NDArray[np.float64],
        tolerance: float,
    ) -> NDArray[np.bool_]:
        assert self.profile is not None
        coordinate, radial = self.project_numpy(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
        )
        return np.asarray(
            (radial <= tolerance)
            & (self.profile.to_numpy(coordinate) <= tolerance),
            dtype=np.bool_,
        )

    def bounding_box(self) -> BoundingBox3D:
        assert self.profile is not None
        minimum, maximum = self.profile.bounds()
        origin = np.asarray(self.origin, dtype=np.float64)
        axis = np.asarray(self.axis_u, dtype=np.float64)
        endpoints = np.asarray(
            (origin + minimum * axis, origin + maximum * axis)
        )
        lower = endpoints.min(axis=0)
        upper = endpoints.max(axis=0)
        padding = 0.004
        return BoundingBox3D(
            lower[0] - padding,
            upper[0] + padding,
            lower[1] - padding,
            upper[1] + padding,
            lower[2] - padding,
            upper[2] + padding,
        )
