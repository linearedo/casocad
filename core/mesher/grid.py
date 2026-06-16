from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from core.sdf.base import BoundingBox3D, SDFNode
from core.sdf.placed_2d import PlacedSDF2D


@dataclass(frozen=True)
class GridSpec:
    x_min: float
    y_min: float
    z_min: float
    dx: float
    nx: int
    ny: int
    nz: int
    dimension: int = 3
    lattice_origin: tuple[float, float, float] | None = None
    axis_i: tuple[float, float, float] = (1.0, 0.0, 0.0)
    axis_j: tuple[float, float, float] = (0.0, 1.0, 0.0)
    axis_k: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if self.dimension not in {2, 3}:
            raise ValueError("grid dimension must be 2 or 3")
        if self.dimension == 2 and self.nz != 1:
            raise ValueError("2D grids must contain exactly one k layer")

    @property
    def node_count(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def envelope(self) -> BoundingBox3D:
        if self.dimension == 2:
            origin = np.asarray(self.lattice_origin, dtype=np.float64)
            axis_i = np.asarray(self.axis_i, dtype=np.float64)
            axis_j = np.asarray(self.axis_j, dtype=np.float64)
            corners = np.asarray(
                [
                    origin + i * axis_i + j * axis_j
                    for i in (0.0, (self.nx - 1) * self.dx)
                    for j in (0.0, (self.ny - 1) * self.dx)
                ]
            )
            minimum = corners.min(axis=0)
            maximum = corners.max(axis=0)
            return BoundingBox3D(
                minimum[0],
                maximum[0],
                minimum[1],
                maximum[1],
                minimum[2],
                maximum[2],
            )
        return BoundingBox3D(
            self.x_min,
            self.x_min + (self.nx - 1) * self.dx,
            self.y_min,
            self.y_min + (self.ny - 1) * self.dx,
            self.z_min,
            self.z_min + (self.nz - 1) * self.dx,
        )


@dataclass(frozen=True)
class GridChunk:
    i: NDArray[np.uint64]
    j: NDArray[np.uint64]
    k: NDArray[np.uint64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]


def derive_grid(box: BoundingBox3D, dx: float) -> GridSpec:
    return GridSpec(
        x_min=box.x_min,
        y_min=box.y_min,
        z_min=box.z_min,
        dx=dx,
        nx=ceil((box.x_max - box.x_min) / dx) + 1,
        ny=ceil((box.y_max - box.y_min) / dx) + 1,
        nz=ceil((box.z_max - box.z_min) / dx) + 1,
    )


def derive_lattice_grid(root: SDFNode, dx: float) -> GridSpec:
    if root.dimension == 3:
        return derive_grid(root.bounding_box(), dx)
    if not isinstance(root, PlacedSDF2D):
        raise ValueError("2D lattice roots must be PlacedSDF2D objects")
    assert root.profile is not None
    u_min, u_max, v_min, v_max = root.profile.bounds()
    origin = np.asarray(root.origin, dtype=np.float64)
    axis_u = np.asarray(root.axis_u, dtype=np.float64)
    axis_v = np.asarray(root.axis_v, dtype=np.float64)
    lattice_origin = origin + u_min * axis_u + v_min * axis_v
    box = root.bounding_box()
    return GridSpec(
        x_min=box.x_min,
        y_min=box.y_min,
        z_min=box.z_min,
        dx=dx,
        nx=ceil((u_max - u_min) / dx) + 1,
        ny=ceil((v_max - v_min) / dx) + 1,
        nz=1,
        dimension=2,
        lattice_origin=tuple(float(value) for value in lattice_origin),
        axis_i=root.axis_u,
        axis_j=root.axis_v,
        axis_k=root.normal,
    )


def generate_chunks(grid: GridSpec, chunk_size: int) -> Iterator[GridChunk]:
    for start in range(0, grid.node_count, chunk_size):
        stop = min(start + chunk_size, grid.node_count)
        linear = np.arange(start, stop, dtype=np.uint64)
        yz = np.uint64(grid.ny * grid.nz)
        nz = np.uint64(grid.nz)
        i = linear // yz
        remainder = linear % yz
        j = remainder // nz
        k = remainder % nz
        origin = np.asarray(
            grid.lattice_origin
            if grid.lattice_origin is not None
            else (grid.x_min, grid.y_min, grid.z_min),
            dtype=np.float64,
        )
        positions = (
            origin
            + i[:, None] * grid.dx * np.asarray(grid.axis_i)
            + j[:, None] * grid.dx * np.asarray(grid.axis_j)
            + k[:, None] * grid.dx * np.asarray(grid.axis_k)
        )
        yield GridChunk(
            i=i,
            j=j,
            k=k,
            x=np.asarray(positions[:, 0], dtype=np.float64),
            y=np.asarray(positions[:, 1], dtype=np.float64),
            z=np.asarray(positions[:, 2], dtype=np.float64),
        )
