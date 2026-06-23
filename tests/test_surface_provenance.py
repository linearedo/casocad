from __future__ import annotations

from core.sdf import Box, Difference, Sphere, Union
from core.sdf.roles import Domain, DomainKind
from core.surface_provenance import (
    PatchTag,
    domain_surface_provenance,
)


def _fluid(name: str, region) -> Domain:
    return Domain(name=name, kind=DomainKind.FLUID, region=region)


def _carve() -> Domain:
    # Fluid = Box minus a spherical obstacle (the canonical carve).
    box = Box(name="box", object_id=1, half_size=(1.0, 1.0, 1.0))
    hole = Sphere(name="hole", object_id=2, radius=0.3)
    return _fluid("fluid", Difference(name="f", object_id=3, left=box, right=hole))


def test_provenance_attributes_each_surface_to_its_owner() -> None:
    prov = domain_surface_provenance(_carve())
    owners = {p.owner_object_id for p in prov}
    # Box (1) outer faces + the subtracted sphere (2) cut surface.
    assert owners == {1, 2}


def test_subtracted_obstacle_surface_is_a_cut_surface() -> None:
    prov = domain_surface_provenance(_carve())
    cut = [p for p in prov if p.is_cut_surface]
    assert len(cut) == 1
    assert cut[0].owner_object_id == 2  # the obstacle owns the cut surface
    assert cut[0].owner_kind == "sphere"


def test_outer_domain_faces_are_not_cut_surfaces() -> None:
    prov = domain_surface_provenance(_carve())
    box_surfaces = [p for p in prov if p.owner_object_id == 1]
    assert box_surfaces
    assert all(not p.is_cut_surface for p in box_surfaces)


def test_every_surface_defaults_to_wall() -> None:
    prov = domain_surface_provenance(_carve())
    assert all(p.tag is PatchTag.WALL for p in prov)


def test_tag_override_applies_per_owner() -> None:
    # Mark the outer box (owner 1) as the inlet; the cut surface stays a wall.
    prov = domain_surface_provenance(
        _carve(), tag_overrides={1: PatchTag.INLET}
    )
    by_owner = {p.owner_object_id: p.tag for p in prov}
    assert by_owner[1] is PatchTag.INLET
    assert by_owner[2] is PatchTag.WALL


def test_union_obstacle_surfaces_are_not_cut() -> None:
    # A union of two obstacles: both contribute normal (non-cut) surfaces.
    a = Sphere(name="a", object_id=1, radius=0.5)
    b = Sphere(name="b", object_id=2, center=(0.4, 0, 0), radius=0.5)
    dom = _fluid("blob", Union(name="u", object_id=3, left=a, right=b))
    prov = domain_surface_provenance(dom)
    assert {p.owner_object_id for p in prov} == {1, 2}
    assert all(not p.is_cut_surface for p in prov)
