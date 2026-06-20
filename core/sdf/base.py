from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BoundingBox3D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def __post_init__(self) -> None:
        if not (
            self.x_min <= self.x_max
            and self.y_min <= self.y_max
            and self.z_min <= self.z_max
        ):
            raise ValueError("bounding box minima must not exceed maxima")

    def union(self, other: BoundingBox3D) -> BoundingBox3D:
        return BoundingBox3D(
            min(self.x_min, other.x_min),
            max(self.x_max, other.x_max),
            min(self.y_min, other.y_min),
            max(self.y_max, other.y_max),
            min(self.z_min, other.z_min),
            max(self.z_max, other.z_max),
        )

    def intersection(self, other: BoundingBox3D) -> BoundingBox3D:
        values = (
            max(self.x_min, other.x_min),
            min(self.x_max, other.x_max),
            max(self.y_min, other.y_min),
            min(self.y_max, other.y_max),
            max(self.z_min, other.z_min),
            min(self.z_max, other.z_max),
        )
        if values[0] > values[1] or values[2] > values[3] or values[4] > values[5]:
            raise ValueError("intersection has an empty bounding box")
        return BoundingBox3D(*values)


@dataclass
class SDFNode(ABC):
    name: str
    object_id: int = 0

    @property
    @abstractmethod
    def dimension(self) -> Literal[1, 2, 3]:
        """Coordinate dimension of this visible SDF object."""

    @property
    def kind(self) -> str:
        return type(self).__name__.lower()

    def children(self) -> tuple[SDFNode, ...]:
        return ()

    @abstractmethod
    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        """Evaluate signed distance for broadcastable float64 arrays."""

    def bounding_box(self) -> BoundingBox3D:
        """Internal provisional traversal bound, not part of SDF semantics."""
        raise NotImplementedError(f"{type(self).__name__} has no traversal bound")

    def leaves(self) -> tuple[SDFNode, ...]:
        children = self.children()
        if not children:
            return (self,)
        return tuple(leaf for child in children for leaf in child.leaves())
