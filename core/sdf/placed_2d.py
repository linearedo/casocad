from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .base import BoundingBox3D, FloatArray, SDFNode, glsl_float, glsl_vec3
from .primitives_2d import Profile2D


def _normalized(vector: tuple[float, float, float]) -> NDArray[np.float64]:
    array = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(array)
    if length <= 1e-12:
        raise ValueError("workplane axes must be nonzero")
    return array / length


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
        normal = np.cross(self.axis_u, self.axis_v)
        normal /= np.linalg.norm(normal)
        return tuple(float(value) for value in normal)

    def children(self) -> tuple[SDFNode, ...]:
        return self.sources

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
        ox, oy, oz = self.origin
        rx, ry, rz = X - ox, Y - oy, Z - oz
        u = rx * self.axis_u[0] + ry * self.axis_u[1] + rz * self.axis_u[2]
        v = rx * self.axis_v[0] + ry * self.axis_v[1] + rz * self.axis_v[2]
        normal = self.normal
        plane = rx * normal[0] + ry * normal[1] + rz * normal[2]
        return (
            np.asarray(u, dtype=np.float64),
            np.asarray(v, dtype=np.float64),
            np.asarray(plane, dtype=np.float64),
        )

    def _project_glsl(self, p_var: str) -> tuple[str, str, str]:
        local = f"({p_var} - {glsl_vec3(self.origin)})"
        u = f"dot({local}, {glsl_vec3(self.axis_u)})"
        v = f"dot({local}, {glsl_vec3(self.axis_v)})"
        plane = f"dot({local}, {glsl_vec3(self.normal)})"
        return u, v, plane

    def to_glsl(self, p_var: str = "p") -> str:
        assert self.profile is not None
        u, v, plane = self._project_glsl(p_var)
        profile = self.profile.to_glsl(f"vec2({u}, {v})")
        # Visualization-only thin sheet. Tagging uses the exact zero-thickness plane.
        thickness = glsl_float(0.002)
        return f"max({profile}, abs({plane}) - {thickness})"

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        assert self.profile is not None
        u, v, _plane = self.project_numpy(X, Y, Z)
        return self.profile.to_numpy(u, v)

    def bounding_box(self) -> BoundingBox3D:
        assert self.profile is not None
        u_min, u_max, v_min, v_max = self.profile.bounds()
        origin = np.asarray(self.origin)
        axis_u = np.asarray(self.axis_u)
        axis_v = np.asarray(self.axis_v)
        corners = np.asarray(
            [
                origin + u * axis_u + v * axis_v
                for u in (u_min, u_max)
                for v in (v_min, v_max)
            ]
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
