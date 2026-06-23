from __future__ import annotations

from core.mesher import FluidDomain
from core.scene import SceneDocument
from core.sdf import Box, Sphere, Union


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
    root = Union(
        name="union_benchmark",
        object_id=3,
        left=sphere,
        right=box,
    )
    return SceneDocument([root], FluidDomain(root))
