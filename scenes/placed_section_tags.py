from __future__ import annotations

from core.boundary import BoundaryRegion
from core.domain import FluidDomain
from core.scene import SceneDocument
from core.sdf import Box, RectangleProfile, PlacedSDF2D


def build_scene() -> SceneDocument:
    fluid = Box(
        name="tagging_volume",
        object_id=1,
        half_size=(1.2, 0.6, 0.5),
    )
    inlet = BoundaryRegion(
        name="inlet",
        object_id=2,
        owner_object_id=fluid.object_id,
        outside_direction=0,
    )
    outlet = BoundaryRegion(
        name="outlet",
        object_id=3,
        owner_object_id=fluid.object_id,
        outside_direction=1,
    )
    midplane = PlacedSDF2D(
        name="internal_midplane",
        object_id=4,
        profile=RectangleProfile(half_size=(0.6, 0.5)),
        axis_u=(0.0, 1.0, 0.0),
        axis_v=(0.0, 0.0, 1.0),
    )
    return SceneDocument(
        [fluid, midplane],
        FluidDomain(fluid, (inlet, outlet, midplane)),
        [inlet, outlet],
    )
