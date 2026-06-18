from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .base import glsl_float

FloatArray = NDArray[np.float64]


class Profile1D(ABC):
    @property
    def kind(self) -> str:
        return type(self).__name__.lower()

    @abstractmethod
    def to_glsl(self, p_var: str = "t") -> str:
        """Return a GLSL expression for one intrinsic coordinate."""

    @abstractmethod
    def to_numpy(self, T: FloatArray) -> FloatArray:
        """Evaluate a local filled-segment signed distance."""

    @abstractmethod
    def bounds(self) -> tuple[float, float]:
        """Internal finite local bounds: t_min, t_max."""


@dataclass(frozen=True)
class SegmentProfile(Profile1D):
    center: float = 0.0
    half_length: float = 0.5

    def __post_init__(self) -> None:
        if self.half_length <= 0.0:
            raise ValueError("segment half length must be positive")

    @property
    def kind(self) -> str:
        return "segment"

    def to_glsl(self, p_var: str = "t") -> str:
        return (
            f"(abs({p_var} - {glsl_float(self.center)})"
            f" - {glsl_float(self.half_length)})"
        )

    def to_numpy(self, T: FloatArray) -> FloatArray:
        return np.asarray(
            np.abs(T - self.center) - self.half_length,
            dtype=np.float64,
        )

    def bounds(self) -> tuple[float, float]:
        return self.center - self.half_length, self.center + self.half_length


@dataclass(frozen=True)
class OffsetProfile1D(Profile1D):
    child: Profile1D
    offset: float = 0.0

    def to_glsl(self, p_var: str = "t") -> str:
        return self.child.to_glsl(
            f"({p_var} - {glsl_float(self.offset)})"
        )

    def to_numpy(self, T: FloatArray) -> FloatArray:
        return self.child.to_numpy(T - self.offset)

    def bounds(self) -> tuple[float, float]:
        minimum, maximum = self.child.bounds()
        return minimum + self.offset, maximum + self.offset


@dataclass(frozen=True)
class BinaryProfile1D(Profile1D):
    left: Profile1D
    right: Profile1D
    operation: str = "union"
    smoothing: float = 0.1

    def __post_init__(self) -> None:
        if self.operation not in {
            "union",
            "intersection",
            "difference",
            "smooth_union",
        }:
            raise ValueError(
                f"unsupported 1D boolean operation: {self.operation}"
            )
        if self.operation == "smooth_union" and self.smoothing <= 0.0:
            raise ValueError("smooth union radius must be positive")

    def to_glsl(self, p_var: str = "t") -> str:
        left = self.left.to_glsl(p_var)
        right = self.right.to_glsl(p_var)
        if self.operation == "union":
            return f"min({left}, {right})"
        if self.operation == "intersection":
            return f"max({left}, {right})"
        if self.operation == "difference":
            return f"max({left}, -({right}))"
        smoothing = glsl_float(self.smoothing)
        blend = (
            f"clamp(0.5 + 0.5 * (({right}) - ({left}))"
            f" / {smoothing}, 0.0, 1.0)"
        )
        return (
            f"(mix(({right}), ({left}), {blend})"
            f" - {smoothing} * {blend} * (1.0 - {blend}))"
        )

    def to_numpy(self, T: FloatArray) -> FloatArray:
        left = self.left.to_numpy(T)
        right = self.right.to_numpy(T)
        if self.operation == "union":
            return np.minimum(left, right)
        if self.operation == "intersection":
            return np.maximum(left, right)
        if self.operation == "difference":
            return np.maximum(left, -right)
        blend = np.clip(
            0.5 + 0.5 * (right - left) / self.smoothing,
            0.0,
            1.0,
        )
        return np.asarray(
            right * (1.0 - blend)
            + left * blend
            - self.smoothing * blend * (1.0 - blend),
            dtype=np.float64,
        )

    def bounds(self) -> tuple[float, float]:
        left = self.left.bounds()
        right = self.right.bounds()
        if self.operation == "difference":
            return left
        return min(left[0], right[0]), max(left[1], right[1])
