from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np
from numpy.typing import NDArray

from core.sdf import BoundingBox3D


@dataclass
class OrbitCamera:
    yaw_degrees: float = 215.0
    pitch_degrees: float = 22.0
    distance: float = 3.2
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    field_of_view_degrees: float = 45.0

    @property
    def position(self) -> tuple[float, float, float]:
        yaw = radians(self.yaw_degrees)
        pitch = radians(self.pitch_degrees)
        horizontal = self.distance * cos(pitch)
        tx, ty, tz = self.target
        return (
            tx + horizontal * sin(yaw),
            ty + horizontal * cos(yaw),
            tz + self.distance * sin(pitch),
        )

    @property
    def focal_length(self) -> float:
        return float(1.0 / np.tan(radians(self.field_of_view_degrees) * 0.5))

    @staticmethod
    def standard_view_angles() -> tuple[float, float]:
        return 215.0, 22.0

    @staticmethod
    def plane_view_angles(plane: str) -> tuple[float, float]:
        if plane == "xy":
            return 180.0, 90.0
        if plane == "xz":
            return 180.0, 0.0
        if plane == "yz":
            return 90.0, 0.0
        raise ValueError(f"unknown reference plane: {plane}")

    def set_standard_view(self) -> None:
        self.yaw_degrees, self.pitch_degrees = self.standard_view_angles()

    def set_plane_view(self, plane: str) -> None:
        self.yaw_degrees, self.pitch_degrees = self.plane_view_angles(plane)

    def _basis(self) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        eye = np.asarray(self.position, dtype=np.float64)
        target = np.asarray(self.target, dtype=np.float64)
        forward = target - eye
        forward /= np.linalg.norm(forward)
        yaw = radians(self.yaw_degrees)
        right = np.asarray((-cos(yaw), sin(yaw), 0.0), dtype=np.float64)
        up = np.cross(right, forward)
        up /= max(np.linalg.norm(up), 1e-12)
        return forward, right, up

    def orbit(self, delta_x: float, delta_y: float) -> None:
        self.yaw_degrees -= delta_x * 0.35
        self.pitch_degrees = max(
            -89.5, min(89.5, self.pitch_degrees + delta_y * 0.35)
        )

    def zoom(self, wheel_steps: float) -> None:
        self.distance = max(0.35, min(50.0, self.distance * (0.88**wheel_steps)))

    def pan(self, delta_x: float, delta_y: float) -> None:
        target = np.asarray(self.target, dtype=np.float64)
        _forward, right, up = self._basis()
        scale = self.distance * 0.0015
        target += (-right * delta_x + up * delta_y) * scale
        self.target = tuple(float(value) for value in target)

    def frame(self, box: BoundingBox3D) -> None:
        self.target = (
            (box.x_min + box.x_max) * 0.5,
            (box.y_min + box.y_max) * 0.5,
            (box.z_min + box.z_max) * 0.5,
        )
        diagonal = (
            (box.x_max - box.x_min) ** 2
            + (box.y_max - box.y_min) ** 2
            + (box.z_max - box.z_min) ** 2
        ) ** 0.5
        self.distance = max(0.5, diagonal * 1.35)

    def view_projection(
        self, aspect_ratio: float
    ) -> NDArray[np.float32]:
        eye = np.asarray(self.position, dtype=np.float64)
        forward, right, up = self._basis()
        view = np.eye(4, dtype=np.float64)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -view[:3, :3] @ eye

        near = max(0.001, self.distance * 0.01)
        far = max(100.0, self.distance * 20.0)
        f = self.focal_length
        projection = np.zeros((4, 4), dtype=np.float64)
        projection[0, 0] = f / max(aspect_ratio, 1e-6)
        projection[1, 1] = f
        projection[2, 2] = (far + near) / (near - far)
        projection[2, 3] = (2.0 * far * near) / (near - far)
        projection[3, 2] = -1.0
        return np.asarray(projection @ view, dtype=np.float32)

    def view_rotation(self) -> NDArray[np.float32]:
        forward, right, up = self._basis()
        return np.asarray((right, up, -forward), dtype=np.float32)

    def screen_ray(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        eye = np.asarray(self.position, dtype=np.float64)
        forward, right, up = self._basis()
        ndc_x = 2.0 * x / max(width, 1.0) - 1.0
        ndc_y = 1.0 - 2.0 * y / max(height, 1.0)
        aspect = width / max(height, 1.0)
        direction = (
            forward * self.focal_length
            + right * ndc_x * aspect
            + up * ndc_y
        )
        direction /= np.linalg.norm(direction)
        return eye, direction

    def screen_to_ground(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> tuple[float, float, float] | None:
        return self.screen_to_plane("xy", x, y, width, height)

    def screen_to_plane(
        self,
        plane: str,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> tuple[float, float, float] | None:
        origin, direction = self.screen_ray(x, y, width, height)
        axis = {"yz": 0, "xz": 1, "xy": 2}.get(plane)
        if axis is None:
            raise ValueError(f"unknown reference plane: {plane}")
        if abs(direction[axis]) <= 1e-9:
            return None
        distance = -origin[axis] / direction[axis]
        if distance <= 0.0:
            return None
        point = origin + direction * distance
        return tuple(float(value) for value in point)
