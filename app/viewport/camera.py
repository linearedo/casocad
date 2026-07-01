"""Orbit camera for the viewport.

Owns the orbit state (target/distance/yaw/pitch/focal), the working-scale
envelope, framing, and screen rays — every place the camera math or its
scale floors live. The widget owns policy (when to reframe, animation
timing); the camera owns mechanism. `view_scale` is meters per working
unit: floors and ceilings scale with it so mm and km work stay reachable
while the grid remains a truthful world-space snap reference.
"""
from __future__ import annotations

import math

import numpy as np

# The familiar startup framing: 6 m away from a 1 m grid. Working-unit
# switches reproduce this view scaled by the unit factor.
DEFAULT_VIEW_DISTANCE = 6.0
_WORLD_UP = np.array([0.0, 0.0, 1.0])
_FALLBACK_UP = np.array([0.0, 1.0, 0.0])


class ViewportCamera:
    def __init__(self) -> None:
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.distance = DEFAULT_VIEW_DISTANCE
        self.yaw = math.radians(35.0)
        self.pitch = math.radians(28.0)
        self.focal = 1.5
        self.view_scale = 1.0  # meters per working unit

    # -- basis ---------------------------------------------------------------

    def position(self) -> np.ndarray:
        cos_pitch = math.cos(self.pitch)
        offset = np.array([
            cos_pitch * math.cos(self.yaw),
            cos_pitch * math.sin(self.yaw),
            math.sin(self.pitch),
        ])
        return self.target + self.distance * offset

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(position, forward, right, up), matching the shader convention."""
        position = self.position()
        forward = self.target - position
        forward = forward / max(np.linalg.norm(forward), 1e-9)
        world_up = _WORLD_UP
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = _FALLBACK_UP
        right = np.cross(forward, world_up)
        right /= max(np.linalg.norm(right), 1e-9)
        up = np.cross(right, forward)
        return position, forward, right, up

    # -- interaction -----------------------------------------------------------

    def orbit(self, delta_x: float, delta_y: float) -> None:
        self.yaw -= delta_x * 0.01
        self.pitch = max(-1.5, min(1.5, self.pitch + delta_y * 0.01))

    def zoom_limits(self) -> tuple[float, float]:
        """Distance envelope, widened (never shrunk) so the working scale and
        the meter-scale defaults both stay reachable."""
        return (
            0.5 * min(1.0, self.view_scale),
            200.0 * max(1.0, self.view_scale),
        )

    def zoom_by(self, wheel_delta: float) -> None:
        minimum, maximum = self.zoom_limits()
        self.distance = max(
            minimum,
            min(maximum, self.distance * math.exp(-wheel_delta * 0.0012)),
        )

    def fly_step(self) -> float:
        return max(self.distance * 0.06, 0.05 * self.view_scale)

    # -- framing ---------------------------------------------------------------

    def frame_target(
        self,
        target: tuple[float, float, float],
        distance: float,
    ) -> None:
        self.target = np.array(target, dtype=np.float64)
        self.distance = float(distance)

    def frame_box(self, box) -> None:
        cx = (box.x_min + box.x_max) * 0.5
        cy = (box.y_min + box.y_max) * 0.5
        cz = (box.z_min + box.z_max) * 0.5
        extent = max(box.x_max - box.x_min, box.y_max - box.y_min,
                     box.z_max - box.z_min, 1e-3)
        self.target = np.array([cx, cy, cz], dtype=np.float64)
        self.distance = max(extent * 1.6, min(1.0, self.view_scale))

    def reframe_to_working_scale(self) -> None:
        """Reproduce the startup framing at the working scale, keeping the
        target (full workspace rescale on unit switch)."""
        self.distance = DEFAULT_VIEW_DISTANCE * self.view_scale

    # -- rays ------------------------------------------------------------------

    def screen_ray(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(origin, direction) of the camera ray through screen (x, y),
        matching the QRhi surface/grid camera convention."""
        position, forward, right, up = self.basis()
        w, h = max(width, 1), max(height, 1)
        suvx = (x - 0.5 * w) / h
        suvy = -((y - 0.5 * h) / h)
        direction = 2.0 * suvx * right + 2.0 * suvy * up + self.focal * forward
        direction /= max(np.linalg.norm(direction), 1e-9)
        return position, direction
