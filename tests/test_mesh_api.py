from __future__ import annotations

from pathlib import Path

import numpy as np

import pytest

from core.meshing import (
    MeshableDomain,
    MeshableDomains,
    load_meshable_domains,
    meshable_domains_from_model,
)
from core.model import Model, ModelCompileError
from core.scene import SceneDocument
from core.serialization import save_scene
from core.sdf.base import BoundingBox3D
from core.sdf import Sphere
from core.sdf.roles import Domain, DomainKind


def test_load_meshable_domains_from_scene_json() -> None:
    domains = load_meshable_domains("scene.json")

    assert len(domains) == 1
    domain = domains[0]
    assert domain.name == "von_karman_fluid"
    assert domain.kind == ("fluid",)
    assert domains["von_karman_fluid"] is domain
    assert domains["fluid"] is domain
    assert domains.keys() == ("von_karman_fluid", "fluid")
    assert domain.dimension == 3
    assert len(domain.boundary_tags) == 2

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=np.float64,
    )
    values = domain.domain_sdf(points)
    assert values.shape == (3,)
    assert values[0] > 0.0   # cylinder obstacle is carved out of the fluid
    assert values[1] < 0.0
    assert values[2] > 0.0


def test_meshable_domain_rejects_bad_point_shape() -> None:
    domain = load_meshable_domains("scene.json")[0]

    try:
        domain.domain_sdf(np.array([0.0, 0.0, 0.0]))
    except ValueError as exc:
        assert "shape (N, 3)" in str(exc)
    else:
        raise AssertionError("expected bad point shape to be rejected")


def test_meshable_domains_reports_ambiguous_kind() -> None:
    def query(points: np.ndarray) -> np.ndarray:
        return np.zeros(points.shape[0], dtype=np.float64)

    bounds = BoundingBox3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    domains = MeshableDomains(
        (
            MeshableDomain("water", ("fluid",), 3, bounds, query),
            MeshableDomain("air", ("fluid",), 3, bounds, query),
        )
    )

    assert domains["water"].name == "water"
    assert domains.by_kind("fluid")[1].name == "air"
    try:
        domains["fluid"]
    except KeyError as exc:
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("expected ambiguous kind lookup to fail")


def test_meshable_domains_from_exact_model() -> None:
    model = Model(
        domains=(
            Domain(
                name="water",
                kind=DomainKind.FLUID,
                region=Sphere(name="water", center=(0.0, 0.0, 0.0), radius=0.5),
            ),
            Domain(
                name="pipe",
                kind=DomainKind.SOLID,
                region=Sphere(name="pipe", center=(2.0, 0.0, 0.0), radius=0.25),
            ),
        )
    )

    domains = meshable_domains_from_model(model)

    assert len(domains) == 2
    assert domains["water"].kind == ("fluid",)
    assert domains["pipe"].kind == ("solid",)
    values = domains["water"].domain_sdf(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    )
    assert values[0] < 0.0
    assert values[1] > 0.0


def test_meshable_domains_from_model_compiles_before_meshing() -> None:
    model = Model(
        domains=(
            Domain(
                name="a",
                kind=DomainKind.FLUID,
                region=Sphere(name="a", center=(0.0, 0.0, 0.0), radius=0.5),
            ),
            Domain(
                name="b",
                kind=DomainKind.SOLID,
                region=Sphere(name="b", center=(0.2, 0.0, 0.0), radius=0.5),
            ),
        )
    )

    with pytest.raises(ModelCompileError):
        meshable_domains_from_model(model)


def test_load_meshable_domains_includes_saved_solid_domain(tmp_path: Path) -> None:
    document = SceneDocument()
    handle = document.add_primitive("box")
    box = document.node(handle)
    document.set_domain_root(handle, DomainKind.SOLID)
    path = tmp_path / "solid.json"
    save_scene(document, path)

    domains = load_meshable_domains(path)

    assert len(domains) == 1
    assert domains[0].name == box.name
    assert domains[0].kind == ("solid",)
    assert domains["solid"] is domains[0]


def test_boundary_regions_are_callable_from_mesher_scripts(tmp_path) -> None:
    """boundary_region_v2 §6: EVERY region is addressable and classifiable —
    including direction-only ones the old contract silently dropped."""
    from core.boundary import BoundaryRegion
    from core.scene import SceneDocument
    from core.sdf import Box, Sphere
    from core.serialization import save_scene

    document = SceneDocument.default()
    box = next(n for _h, n, _p in document.walk() if isinstance(n, Box))
    whole = document.node(document.add_boundary_region(box.object_id))
    assert isinstance(whole, BoundaryRegion)
    ghost = Sphere(name="knife", object_id=0, center=(-1.6, 0.0, 0.0), radius=0.5)
    inside_handle, _ = document.split_boundary_region(whole, ghost)
    document.node(inside_handle).tag = "inlet"
    path = tmp_path / "scene.json"
    save_scene(document, path)

    domain = load_meshable_domains(path)["von_karman_fluid"]
    names = domain.boundary_regions.keys()
    assert any("inside" in name for name in names)
    assert "inlet" in names and "outlet" in names  # direction-only regions callable

    jet = next(r for r in domain.boundary_regions if r.tag == "inlet")
    face_points = np.array(
        [
            [-1.6, 0.0, 0.0],   # centre of the -X face: inside the knife
            [-1.6, 0.6, 0.3],   # far corner of the -X face: outside the knife
            [1.6, 0.0, 0.0],    # +X face: wrong side of the box entirely
        ]
    )
    mask = jet.contains(face_points)
    assert mask.tolist() == [True, False, False]
    # owner_sdf is the exact field of the generating surface
    assert abs(float(jet.owner_sdf(np.array([[-1.6, 0.0, 0.0]]))[0])) < 1e-9
    # selector_sdf is negative inside the kept knife-half
    assert float(jet.selector_sdf(np.array([[-1.6, 0.0, 0.0]]))[0]) < 0.0
    assert float(jet.selector_sdf(np.array([[-1.6, 0.6, 0.3]]))[0]) > 0.0

    legacy_inlet = domain.boundary_regions["inlet"]
    assert legacy_inlet.selector_sdf is None
    assert legacy_inlet.contains(face_points).tolist() == [True, True, False]
