from __future__ import annotations

from core.domain import FluidDomain
from core.scene import SceneDocument
from core.sdf import Box, Difference, Intersection, Sphere, Union


def build_scene() -> SceneDocument:
    union = Union(
        name="union_example",
        object_id=3,
        left=Sphere(
            name="union_sphere",
            object_id=1,
            center=(-1.4, 0.0, 0.0),
            radius=0.65,
        ),
        right=Box(
            name="union_box",
            object_id=2,
            center=(-0.9, 0.0, 0.0),
            half_size=(0.55, 0.55, 0.55),
        ),
    )
    intersection = Intersection(
        name="intersection_example",
        object_id=6,
        left=Sphere(
            name="intersection_sphere",
            object_id=4,
            center=(0.0, 0.0, 0.0),
            radius=0.72,
        ),
        right=Box(
            name="intersection_box",
            object_id=5,
            center=(0.25, 0.0, 0.0),
            half_size=(0.55, 0.55, 0.55),
        ),
    )
    difference = Difference(
        name="difference_example",
        object_id=9,
        left=Box(
            name="difference_box",
            object_id=7,
            center=(1.4, 0.0, 0.0),
            half_size=(0.7, 0.6, 0.6),
        ),
        right=Sphere(
            name="difference_sphere",
            object_id=8,
            center=(1.55, 0.0, 0.0),
            radius=0.5,
        ),
    )
    root = Union(
        name="boolean_demo_domain",
        object_id=10,
        left=Union(
            name="union_and_intersection",
            object_id=11,
            left=union,
            right=intersection,
        ),
        right=difference,
    )
    return SceneDocument([root], FluidDomain(root))
