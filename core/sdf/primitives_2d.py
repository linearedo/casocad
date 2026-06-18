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


def _as_points(
    points: tuple[tuple[float, float], ...] | list[tuple[float, float]] | list[list[float]],
) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in points)


def _segment_distance_numpy(
    U: FloatArray,
    V: FloatArray,
    first: tuple[float, float],
    second: tuple[float, float],
) -> FloatArray:
    ax, ay = first
    bx, by = second
    bax = bx - ax
    bay = by - ay
    denominator = bax * bax + bay * bay
    if denominator <= 1e-24:
        return np.asarray(np.sqrt((U - ax) ** 2 + (V - ay) ** 2), dtype=np.float64)
    h = np.clip(((U - ax) * bax + (V - ay) * bay) / denominator, 0.0, 1.0)
    dx = U - ax - h * bax
    dy = V - ay - h * bay
    return np.asarray(np.sqrt(dx * dx + dy * dy), dtype=np.float64)


def _segment_distance_glsl(
    p_var: str,
    first: tuple[float, float],
    second: tuple[float, float],
) -> str:
    a = _vec2(first)
    b = _vec2(second)
    pa = f"({p_var} - {a})"
    ba = f"({b} - {a})"
    h = f"clamp(dot({pa}, {ba}) / max(dot({ba}, {ba}), 1.0e-12), 0.0, 1.0)"
    return f"length({pa} - {ba} * {h})"


def _polyline_distance_numpy(
    U: FloatArray,
    V: FloatArray,
    points: tuple[tuple[float, float], ...],
    closed: bool,
) -> FloatArray:
    pairs = zip(points, points[1:])
    distances = [
        _segment_distance_numpy(U, V, first, second)
        for first, second in pairs
    ]
    if closed:
        distances.append(_segment_distance_numpy(U, V, points[-1], points[0]))
    return np.asarray(np.minimum.reduce(distances), dtype=np.float64)


def _polyline_distance_glsl(
    p_var: str,
    points: tuple[tuple[float, float], ...],
    closed: bool,
) -> str:
    pairs = list(zip(points, points[1:]))
    if closed:
        pairs.append((points[-1], points[0]))
    expression = _segment_distance_glsl(p_var, pairs[0][0], pairs[0][1])
    for first, second in pairs[1:]:
        expression = f"min({expression}, {_segment_distance_glsl(p_var, first, second)})"
    return expression


def _quadratic_bezier_spans(
    points: tuple[tuple[float, float], ...],
) -> tuple[
    tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    ...,
]:
    return tuple(
        (points[index], points[index + 1], points[index + 2])
        for index in range(0, len(points) - 2, 2)
    )


def _quadratic_bezier_distance_numpy(
    U: FloatArray,
    V: FloatArray,
    start: tuple[float, float],
    control: tuple[float, float],
    end: tuple[float, float],
) -> FloatArray:
    ax, ay = start
    bx, by = control
    cx, cy = end
    a_x = bx - ax
    a_y = by - ay
    b_x = ax - 2.0 * bx + cx
    b_y = ay - 2.0 * by + cy
    c_x = 2.0 * a_x
    c_y = 2.0 * a_y
    b_dot_b = b_x * b_x + b_y * b_y
    if b_dot_b <= 1.0e-24:
        return _segment_distance_numpy(U, V, start, end)

    d_x = ax - U
    d_y = ay - V
    kk = 1.0 / b_dot_b
    kx = kk * (a_x * b_x + a_y * b_y)
    ky = kk * (2.0 * (a_x * a_x + a_y * a_y) + d_x * b_x + d_y * b_y) / 3.0
    kz = kk * (d_x * a_x + d_y * a_y)
    p = ky - kx * kx
    q = kx * (2.0 * kx * kx - 3.0 * ky) + kz
    h = q * q + 4.0 * p * p * p
    result = np.full(np.shape(U), np.inf, dtype=np.float64)

    single_root = h >= 0.0
    if np.any(single_root):
        h_root = np.sqrt(np.maximum(h[single_root], 0.0))
        x_0 = 0.5 * (h_root - q[single_root])
        x_1 = 0.5 * (-h_root - q[single_root])
        t = np.clip(np.cbrt(x_0) + np.cbrt(x_1) - kx, 0.0, 1.0)
        w_x = d_x[single_root] + (c_x + b_x * t) * t
        w_y = d_y[single_root] + (c_y + b_y * t) * t
        result[single_root] = w_x * w_x + w_y * w_y

    three_roots = ~single_root
    if np.any(three_roots):
        p_values = p[three_roots]
        q_values = q[three_roots]
        z = np.sqrt(np.maximum(-p_values, 0.0))
        denominator = 2.0 * p_values * z
        angle_argument = np.divide(
            q_values,
            denominator,
            out=np.zeros_like(q_values),
            where=np.abs(denominator) > 1.0e-24,
        )
        angle = np.arccos(np.clip(angle_argument, -1.0, 1.0)) / 3.0
        m = np.cos(angle)
        n = np.sin(angle) * 1.732050808
        t_0 = np.clip((m + m) * z - kx, 0.0, 1.0)
        t_1 = np.clip((-n - m) * z - kx, 0.0, 1.0)
        d_x_values = d_x[three_roots]
        d_y_values = d_y[three_roots]
        w_0_x = d_x_values + (c_x + b_x * t_0) * t_0
        w_0_y = d_y_values + (c_y + b_y * t_0) * t_0
        w_1_x = d_x_values + (c_x + b_x * t_1) * t_1
        w_1_y = d_y_values + (c_y + b_y * t_1) * t_1
        result[three_roots] = np.minimum(
            w_0_x * w_0_x + w_0_y * w_0_y,
            w_1_x * w_1_x + w_1_y * w_1_y,
        )

    return np.asarray(np.sqrt(np.maximum(result, 0.0)), dtype=np.float64)


def _quadratic_bezier_distance_glsl(
    p_var: str,
    start: tuple[float, float],
    control: tuple[float, float],
    end: tuple[float, float],
) -> str:
    return (
        f"quadraticBezierDistance({p_var}, {_vec2(start)}, "
        f"{_vec2(control)}, {_vec2(end)})"
    )


@dataclass(frozen=True)
class PolylineProfile(Profile2D):
    points: tuple[tuple[float, float], ...] = (
        (-0.6, -0.4),
        (0.6, -0.4),
        (0.35, 0.4),
        (-0.35, 0.4),
    )

    def __post_init__(self) -> None:
        normalized = _as_points(self.points)
        if len(normalized) < 2:
            raise ValueError("polyline requires at least two points")
        if all(
            np.linalg.norm(np.asarray(second) - np.asarray(first)) <= 1e-12
            for first, second in zip(normalized, normalized[1:])
        ):
            raise ValueError("polyline requires at least one nonzero segment")
        object.__setattr__(self, "points", normalized)

    @property
    def kind(self) -> str:
        return "polyline"

    def to_glsl(self, p_var: str = "q") -> str:
        return _polyline_distance_glsl(p_var, self.points, closed=False)

    def to_numpy(self, U: FloatArray, V: FloatArray) -> FloatArray:
        return _polyline_distance_numpy(U, V, self.points, closed=False)

    def bounds(self) -> tuple[float, float, float, float]:
        points = np.asarray(self.points, dtype=np.float64)
        return (
            float(points[:, 0].min()),
            float(points[:, 0].max()),
            float(points[:, 1].min()),
            float(points[:, 1].max()),
        )


@dataclass(frozen=True)
class BezierCurveProfile(Profile2D):
    points: tuple[tuple[float, float], ...] = (
        (-0.6, -0.35),
        (0.0, 0.55),
        (0.6, -0.35),
    )

    def __post_init__(self) -> None:
        normalized = _as_points(self.points)
        if len(normalized) < 3:
            raise ValueError("bezier curve requires at least three points")
        if len(normalized) % 2 == 0:
            raise ValueError(
                "bezier curve requires an odd point count: anchor, control, anchor"
            )
        if all(
            np.linalg.norm(np.asarray(control) - np.asarray(start)) <= 1e-12
            and np.linalg.norm(np.asarray(end) - np.asarray(start)) <= 1e-12
            for start, control, end in _quadratic_bezier_spans(normalized)
        ):
            raise ValueError("bezier curve requires at least one nonzero span")
        object.__setattr__(self, "points", normalized)

    @property
    def kind(self) -> str:
        return "bezier_polycurve" if len(self.points) > 3 else "bezier_curve"

    def to_glsl(self, p_var: str = "q") -> str:
        spans = _quadratic_bezier_spans(self.points)
        expression = _quadratic_bezier_distance_glsl(p_var, *spans[0])
        for span in spans[1:]:
            expression = (
                f"min({expression}, {_quadratic_bezier_distance_glsl(p_var, *span)})"
            )
        return expression

    def to_numpy(self, U: FloatArray, V: FloatArray) -> FloatArray:
        distances = [
            _quadratic_bezier_distance_numpy(U, V, start, control, end)
            for start, control, end in _quadratic_bezier_spans(self.points)
        ]
        return np.asarray(np.minimum.reduce(distances), dtype=np.float64)

    def bounds(self) -> tuple[float, float, float, float]:
        points = np.asarray(self.points, dtype=np.float64)
        return (
            float(points[:, 0].min()),
            float(points[:, 0].max()),
            float(points[:, 1].min()),
            float(points[:, 1].max()),
        )


@dataclass(frozen=True)
class PolygonProfile(Profile2D):
    points: tuple[tuple[float, float], ...] = (
        (-0.6, -0.4),
        (0.6, -0.4),
        (0.35, 0.4),
        (-0.35, 0.4),
    )

    def __post_init__(self) -> None:
        normalized = _as_points(self.points)
        if len(normalized) >= 2 and normalized[0] == normalized[-1]:
            normalized = normalized[:-1]
        if len(normalized) < 3:
            raise ValueError("polygon requires at least three points")
        object.__setattr__(self, "points", normalized)

    @property
    def kind(self) -> str:
        return "polygon"

    def to_glsl(self, p_var: str = "q") -> str:
        distance = _polyline_distance_glsl(p_var, self.points, closed=True)
        inside = "false"
        for first, second in zip(
            self.points,
            (*self.points[1:], self.points[0]),
            strict=True,
        ):
            ax, ay = first
            bx, by = second
            condition = (
                f"(({glsl_float(ay)} > {p_var}.y) != ({glsl_float(by)} > {p_var}.y))"
                f" && ({p_var}.x < ({glsl_float(bx)} - {glsl_float(ax)})"
                f" * ({p_var}.y - {glsl_float(ay)})"
                f" / ({glsl_float(by)} - {glsl_float(ay)}) + {glsl_float(ax)})"
            )
            inside = f"(({inside}) != ({condition}))"
        return f"(({inside}) ? -({distance}) : ({distance}))"

    def to_numpy(self, U: FloatArray, V: FloatArray) -> FloatArray:
        distance = _polyline_distance_numpy(U, V, self.points, closed=True)
        inside = np.full(np.shape(distance), False, dtype=np.bool_)
        for first, second in zip(
            self.points,
            (*self.points[1:], self.points[0]),
            strict=True,
        ):
            ax, ay = first
            bx, by = second
            active = (ay > V) != (by > V)
            intersection = np.full(np.shape(distance), ax, dtype=np.float64)
            np.divide(
                (bx - ax) * (V - ay),
                by - ay,
                out=intersection,
                where=active,
            )
            intersection += ax
            crosses = active & (U < intersection)
            inside ^= crosses
        return np.asarray(np.where(inside, -distance, distance), dtype=np.float64)

    def bounds(self) -> tuple[float, float, float, float]:
        points = np.asarray(self.points, dtype=np.float64)
        return (
            float(points[:, 0].min()),
            float(points[:, 0].max()),
            float(points[:, 1].min()),
            float(points[:, 1].max()),
        )


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
