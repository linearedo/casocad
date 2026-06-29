"""Viewport-surface value types and the empty/failed/colour fallbacks.

The disposable render artifacts (`ViewportSurface`, its key and scene) plus the
non-misleading fallbacks. A leaf module imported by everything else; it imports
nothing from the viewport package.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from core.sdf import SDFNode


SurfaceStatus = Literal["ready", "outline", "empty", "failed"]


_DEFAULT_RESOLUTION = 12


@dataclass(frozen=True)
class ViewportSurfaceKey:
    object_id: int
    scene_revision: int
    resolution: int = _DEFAULT_RESOLUTION


@dataclass(frozen=True)
class ViewportSurface:
    key: ViewportSurfaceKey
    object_kind: str
    status: SurfaceStatus
    vertices: NDArray[np.float32]
    normals: NDArray[np.float32]
    indices: NDArray[np.uint32]
    wire_indices: NDArray[np.uint32]
    color: tuple[float, float, float]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    message: str = ""

    @property
    def has_geometry(self) -> bool:
        return bool(self.vertices.size and (self.indices.size or self.wire_indices.size))

    @property
    def triangle_count(self) -> int:
        return int(self.indices.size // 3)

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])


@dataclass(frozen=True)
class ViewportSurfaceScene:
    revision: int
    surfaces: tuple[ViewportSurface, ...]
    build_ms: float

    @property
    def has_geometry(self) -> bool:
        return any(surface.has_geometry for surface in self.surfaces)

    @property
    def vertex_count(self) -> int:
        return sum(surface.vertex_count for surface in self.surfaces)

    @property
    def triangle_count(self) -> int:
        return sum(surface.triangle_count for surface in self.surfaces)

    @property
    def failed_messages(self) -> tuple[str, ...]:
        return tuple(
            surface.message
            for surface in self.surfaces
            if surface.status == "failed" and surface.message
        )


def _object_color(object_id: int) -> tuple[float, float, float]:
    value = (int(object_id) * 2_654_435_761) & 0xFFFFFFFF
    hue = (value % 360) / 360.0
    saturation = 0.48
    value_luma = 0.92
    return _hsv_to_rgb(hue, saturation, value_luma)


def _hsv_to_rgb(hue: float, saturation: float, value: float) -> tuple[float, float, float]:
    h = (hue % 1.0) * 6.0
    c = value * saturation
    x = c * (1.0 - abs(h % 2.0 - 1.0))
    m = value - c
    if h < 1.0:
        rgb = (c, x, 0.0)
    elif h < 2.0:
        rgb = (x, c, 0.0)
    elif h < 3.0:
        rgb = (0.0, c, x)
    elif h < 4.0:
        rgb = (0.0, x, c)
    elif h < 5.0:
        rgb = (x, 0.0, c)
    else:
        rgb = (c, 0.0, x)
    return tuple(float(component + m) for component in rgb)


def _empty_surface(
    node: SDFNode,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
    message: str,
) -> ViewportSurface:
    mins, maxs = _safe_node_bounds(node)
    return ViewportSurface(
        key=key,
        object_kind=type(node).__name__,
        status="empty",
        vertices=np.zeros((0, 3), dtype=np.float32),
        normals=np.zeros((0, 3), dtype=np.float32),
        indices=np.zeros(0, dtype=np.uint32),
        wire_indices=np.zeros(0, dtype=np.uint32),
        color=color,
        bounds_min=mins,
        bounds_max=maxs,
        message=message,
    )


def _failed_surface(
    node: SDFNode,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
    message: str,
) -> ViewportSurface:
    mins, maxs = _safe_node_bounds(node)
    return ViewportSurface(
        key=key,
        object_kind=type(node).__name__,
        status="failed",
        vertices=np.zeros((0, 3), dtype=np.float32),
        normals=np.zeros((0, 3), dtype=np.float32),
        indices=np.zeros(0, dtype=np.uint32),
        wire_indices=np.zeros(0, dtype=np.uint32),
        color=color,
        bounds_min=mins,
        bounds_max=maxs,
        message=message,
    )


def _safe_node_bounds(node: SDFNode) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    try:
        box = node.bounding_box()
    except Exception:  # noqa: BLE001
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    return (
        (float(box.x_min), float(box.y_min), float(box.z_min)),
        (float(box.x_max), float(box.y_max), float(box.z_max)),
    )
