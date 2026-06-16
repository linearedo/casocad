from __future__ import annotations

from core.boundary import BoundaryRegion
from core.scene import SceneDocument
from core.sdf import PlacedSDF2D
from scenes.pipe_3d import build_scene as build_pipe_scene
from scenes.placed_section_tags import build_scene as build_tagging_scene


def test_3d_examples_use_boundary_regions_for_inlet_and_outlet() -> None:
    for document in (
        SceneDocument.default(),
        build_pipe_scene(),
        build_tagging_scene(),
    ):
        assert document.fluid_domain is not None
        tags = {tag.name: tag for tag in document.fluid_domain.tag_objects}

        assert isinstance(tags["inlet"], BoundaryRegion)
        assert isinstance(tags["outlet"], BoundaryRegion)
        assert not isinstance(tags["inlet"], PlacedSDF2D)
        assert not isinstance(tags["outlet"], PlacedSDF2D)


def test_placed_section_example_keeps_internal_midplane_as_placed_sdf() -> None:
    document = build_tagging_scene()
    assert document.fluid_domain is not None

    tags = {tag.name: tag for tag in document.fluid_domain.tag_objects}

    assert isinstance(tags["internal_midplane"], PlacedSDF2D)
