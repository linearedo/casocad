from __future__ import annotations

from core.boundary import BoundaryRegion
from core.domain import FluidDomain
from core.scene import SceneDocument
from core.sdf import Cylinder


def build_scene() -> SceneDocument:
    fluid = Cylinder(
        name="pipe_fluid",
        object_id=1,
        radius=0.45,
        half_height=1.5,
    )
    inlet = BoundaryRegion(
        name="inlet",
        object_id=2,
        owner_object_id=fluid.object_id,
        outside_direction=4,
    )
    outlet = BoundaryRegion(
        name="outlet",
        object_id=3,
        owner_object_id=fluid.object_id,
        outside_direction=5,
    )
    return SceneDocument(
        [fluid],
        FluidDomain(fluid, (inlet, outlet)),
        [inlet, outlet],
    )
