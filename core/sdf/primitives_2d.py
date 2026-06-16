from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np
from numpy.typing import NDArray

from .base import glsl_float

FloatArray = NDArray[np.float64]


class Profile2D(ABC):
    @property
    def kind(self) -> str:
        return type(self).__name__.lower()

    @abstractmethod
    def to_glsl(self, p_var: str = "q") -> str:
        """Return a GLSL expression for a local vec2."""

    @abstractmethod
    def to_numpy(self, U: FloatArray, V: FloatArray) -> FloatArray:
        """Evaluate a local filled-region signed distance."""

    @abstractmethod
    def bounds(self) -> tuple[float, float, float, float]:
        """Internal finite local bounds: u_min, u_max, v_min, v_max."""


def _vec2(values: tuple[float, float]) -> str:
    return f"vec2({glsl_float(values[0])}, {glsl_float(values[1])})"


@dataclass(frozen=True)
class CircleProfile(Profile2D):
    center: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.5

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("circle radius must be positive")

    def to_glsl(self, p_var: str = "q") -> str:
        return (
            f"(length({p_var} - {_vec2(self.center)})"
            f" - {glsl_float(self.radius)})"
        )

    def to_numpy(self, U: FloatArray, V: FloatArray) -> FloatArray:
        cu, cv = self.center
        return np.asarray(
            np.sqrt((U - cu) ** 2 + (V - cv) ** 2) - self.radius,
            dtype=np.float64,
        )

    def bounds(self) -> tuple[float, float, float, float]:
        cu, cv = self.center
        return (
            cu - self.radius,
            cu + self.radius,
            cv - self.radius,
            cv + self.radius,
        )


@dataclass(frozen=True)
class RectangleProfile(Profile2D):
    center: tuple[float, float] = (0.0, 0.0)
    half_size: tuple[float, float] = (0.5, 0.35)

    def __post_init__(self) -> None:
        if any(value <= 0.0 for value in self.half_size):
            raise ValueError("rectangle half sizes must be positive")

    def to_glsl(self, p_var: str = "q") -> str:
        q = (
            f"(abs({p_var} - {_vec2(self.center)})"
            f" - {_vec2(self.half_size)})"
        )
        return f"(length(max({q}, vec2(0.0))) + min(max({q}.x, {q}.y), 0.0))"

    def to_numpy(self, U: FloatArray, V: FloatArray) -> FloatArray:
        cu, cv = self.center
        hu, hv = self.half_size
        qu = np.abs(U - cu) - hu
        qv = np.abs(V - cv) - hv
        outside = np.sqrt(np.maximum(qu, 0.0) ** 2 + np.maximum(qv, 0.0) ** 2)
        inside = np.minimum(np.maximum(qu, qv), 0.0)
        return np.asarray(outside + inside, dtype=np.float64)

    def bounds(self) -> tuple[float, float, float, float]:
        cu, cv = self.center
        hu, hv = self.half_size
        return cu - hu, cu + hu, cv - hv, cv + hv


@dataclass(frozen=True)
class SquareProfile(RectangleProfile):
    half_size: float = 0.5

    def __post_init__(self) -> None:
        if self.half_size <= 0.0:
            raise ValueError("square half size must be positive")

    def _rectangle(self) -> RectangleProfile:
        return RectangleProfile(self.center, (self.half_size, self.half_size))

    def to_glsl(self, p_var: str = "q") -> str:
        return self._rectangle().to_glsl(p_var)

    def to_numpy(self, U: FloatArray, V: FloatArray) -> FloatArray:
        return self._rectangle().to_numpy(U, V)

    def bounds(self) -> tuple[float, float, float, float]:
        return self._rectangle().bounds()


@dataclass(frozen=True)
class RoundedRectangleProfile(RectangleProfile):
    corner_radius: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.corner_radius <= 0.0:
            raise ValueError("corner radius must be positive")
        if self.corner_radius > min(self.half_size):
            raise ValueError("corner radius exceeds rectangle half size")

    def to_glsl(self, p_var: str = "q") -> str:
        inner = (
            self.half_size[0] - self.corner_radius,
            self.half_size[1] - self.corner_radius,
        )
        q = f"(abs({p_var} - {_vec2(self.center)}) - {_vec2(inner)})"
        return (
            f"(length(max({q}, vec2(0.0)))"
            f" + min(max({q}.x, {q}.y), 0.0)"
            f" - {glsl_float(self.corner_radius)})"
        )

    def to_numpy(self, U: FloatArray, V: FloatArray) -> FloatArray:
        cu, cv = self.center
        inner_u = self.half_size[0] - self.corner_radius
        inner_v = self.half_size[1] - self.corner_radius
        qu = np.abs(U - cu) - inner_u
        qv = np.abs(V - cv) - inner_v
        outside = np.sqrt(np.maximum(qu, 0.0) ** 2 + np.maximum(qv, 0.0) ** 2)
        inside = np.minimum(np.maximum(qu, qv), 0.0)
        return np.asarray(outside + inside - self.corner_radius, dtype=np.float64)


@dataclass(frozen=True)
class EllipseProfile(Profile2D):
    center: tuple[float, float] = (0.0, 0.0)
    semi_axes: tuple[float, float] = (0.6, 0.35)

    def __post_init__(self) -> None:
        if any(value <= 0.0 for value in self.semi_axes):
            raise ValueError("ellipse semi-axes must be positive")

    def to_glsl(self, p_var: str = "q") -> str:
        # Stable implicit approximation; sign is exact for the ellipse.
        axes = _vec2(self.semi_axes)
        minimum = glsl_float(min(self.semi_axes))
        return f"((length(({p_var} - {_vec2(self.center)}) / {axes}) - 1.0) * {minimum})"

    def to_numpy(self, U: FloatArray, V: FloatArray) -> FloatArray:
        cu, cv = self.center
        au, av = self.semi_axes
        normalized = np.sqrt(((U - cu) / au) ** 2 + ((V - cv) / av) ** 2)
        return np.asarray((normalized - 1.0) * min(au, av), dtype=np.float64)

    def bounds(self) -> tuple[float, float, float, float]:
        cu, cv = self.center
        au, av = self.semi_axes
        return cu - au, cu + au, cv - av, cv + av


@dataclass(frozen=True)
class RegularPolygonProfile(Profile2D):
    center: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.5
    side_count: int = 6
    rotation: float = 0.0

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("polygon radius must be positive")
        if self.side_count < 3:
            raise ValueError("polygon requires at least three sides")

    def _vertices(self) -> NDArray[np.float64]:
        angles = self.rotation + np.arange(self.side_count) * 2.0 * pi / self.side_count
        return np.column_stack(
            (
                self.center[0] + self.radius * np.cos(angles),
                self.center[1] + self.radius * np.sin(angles),
            )
        )

    def to_glsl(self, p_var: str = "q") -> str:
        # Convex polygon as intersection of oriented half-planes.
        vertices = self._vertices()
        expressions: list[str] = []
        for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
            edge = second - first
            normal = np.asarray((edge[1], -edge[0]))
            normal /= np.linalg.norm(normal)
            expressions.append(
                f"dot({p_var} - {_vec2(tuple(first))}, {_vec2(tuple(normal))})"
            )
        expression = expressions[0]
        for item in expressions[1:]:
            expression = f"max({expression}, {item})"
        return expression

    def to_numpy(self, U: FloatArray, V: FloatArray) -> FloatArray:
        vertices = self._vertices()
        distances = []
        for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
            edge = second - first
            normal = np.asarray((edge[1], -edge[0]))
            normal /= np.linalg.norm(normal)
            distances.append((U - first[0]) * normal[0] + (V - first[1]) * normal[1])
        return np.asarray(np.maximum.reduce(distances), dtype=np.float64)

    def bounds(self) -> tuple[float, float, float, float]:
        vertices = self._vertices()
        return (
            float(vertices[:, 0].min()),
            float(vertices[:, 0].max()),
            float(vertices[:, 1].min()),
            float(vertices[:, 1].max()),
        )


@dataclass(frozen=True)
class OffsetProfile(Profile2D):
    child: Profile2D
    offset: tuple[float, float] = (0.0, 0.0)

    def to_glsl(self, p_var: str = "q") -> str:
        return self.child.to_glsl(f"({p_var} - {_vec2(self.offset)})")

    def to_numpy(self, U: FloatArray, V: FloatArray) -> FloatArray:
        return self.child.to_numpy(U - self.offset[0], V - self.offset[1])

    def bounds(self) -> tuple[float, float, float, float]:
        u_min, u_max, v_min, v_max = self.child.bounds()
        return (
            u_min + self.offset[0],
            u_max + self.offset[0],
            v_min + self.offset[1],
            v_max + self.offset[1],
        )


@dataclass(frozen=True)
class BinaryProfile(Profile2D):
    left: Profile2D
    right: Profile2D
    operation: str = "union"
    smoothing: float = 0.1

    def __post_init__(self) -> None:
        if self.operation not in {"union", "intersection", "difference", "smooth_union"}:
            raise ValueError(f"unsupported 2D boolean operation: {self.operation}")
        if self.operation == "smooth_union" and self.smoothing <= 0.0:
            raise ValueError("smooth union radius must be positive")

    def to_glsl(self, p_var: str = "q") -> str:
        left = self.left.to_glsl(p_var)
        right = self.right.to_glsl(p_var)
        if self.operation == "union":
            return f"min({left}, {right})"
        if self.operation == "intersection":
            return f"max({left}, {right})"
        if self.operation == "difference":
            return f"max({left}, -({right}))"
        k = glsl_float(self.smoothing)
        h = f"clamp(0.5 + 0.5 * (({right}) - ({left})) / {k}, 0.0, 1.0)"
        return f"(mix(({right}), ({left}), {h}) - {k} * {h} * (1.0 - {h}))"

    def to_numpy(self, U: FloatArray, V: FloatArray) -> FloatArray:
        left = self.left.to_numpy(U, V)
        right = self.right.to_numpy(U, V)
        if self.operation == "union":
            return np.minimum(left, right)
        if self.operation == "intersection":
            return np.maximum(left, right)
        if self.operation == "difference":
            return np.maximum(left, -right)
        h = np.clip(0.5 + 0.5 * (right - left) / self.smoothing, 0.0, 1.0)
        return np.asarray(
            right * (1.0 - h)
            + left * h
            - self.smoothing * h * (1.0 - h),
            dtype=np.float64,
        )

    def bounds(self) -> tuple[float, float, float, float]:
        left = self.left.bounds()
        right = self.right.bounds()
        if self.operation == "difference":
            return left
        return (
            min(left[0], right[0]),
            max(left[1], right[1]),
            min(left[2], right[2]),
            max(left[3], right[3]),
        )
