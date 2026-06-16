from __future__ import annotations

from core.mesher import FluidDomain
from core.scene import SceneDocument
from core.sdf import Box, SmoothUnion, Sphere


def build_scene() -> SceneDocument:
    sphere = Sphere(
        name="benchmark_sphere",
        object_id=1,
        center=(-0.35, 0.0, 0.0),
        radius=0.65,
    )
    box = Box(
        name="benchmark_box",
        object_id=2,
        center=(0.4, 0.0, 0.0),
        half_size=(0.55, 0.45, 0.45),
    )
    root = SmoothUnion(
        name="smooth_benchmark",
        object_id=3,
        left=sphere,
        right=box,
        smoothing=0.18,
    )
    return SceneDocument([root], FluidDomain(root))
