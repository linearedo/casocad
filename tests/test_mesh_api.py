from __future__ import annotations

import numpy as np

from core.meshing import MeshableDomain, MeshableDomains, load_meshable_domains
from core.sdf.base import BoundingBox3D


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
