from __future__ import annotations

import json

import numpy as np
import pytest

from core.scene import SceneDocument
from core.serialization import load_scene, save_scene
from core.sdf import (
    BezierCurveProfile,
    PlacedPolyline2D,
    PlacedSDF2D,
    PolygonProfile,
    PolylineProfile,
)

TRAPEZOID_POINTS = (
    (-0.6, -0.4),
    (0.6, -0.4),
    (0.35, 0.4),
    (-0.35, 0.4),
)


def test_polyline_and_polygon_defaults_are_trapezoid_point_shapes() -> None:
    assert PolylineProfile().points == TRAPEZOID_POINTS
    assert PolygonProfile().points == TRAPEZOID_POINTS
    assert BezierCurveProfile().points == (
        (-0.6, -0.35),
        (0.0, 0.55),
        (0.6, -0.35),
    )


def test_polyline_profile_measures_distance_to_nearest_segment() -> None:
    profile = PolylineProfile(points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)))
    values = profile.to_numpy(
        np.asarray((0.5, 1.0, 1.3), dtype=np.float64),
        np.asarray((0.25, 0.5, 1.0), dtype=np.float64),
    )

    np.testing.assert_allclose(values, (0.25, 0.0, 0.3))
    assert profile.bounds() == (0.0, 1.0, 0.0, 1.0)


def test_bezier_curve_profile_measures_exact_quadratic_distance() -> None:
    profile = BezierCurveProfile(
        points=((-1.0, 0.0), (0.0, 1.0), (1.0, 0.0))
    )
    values = profile.to_numpy(
        np.asarray((0.0, 0.0, -1.0, 1.0), dtype=np.float64),
        np.asarray((0.5, 1.0, 0.0, 0.0), dtype=np.float64),
    )

    np.testing.assert_allclose(values, (0.0, 0.5, 0.0, 0.0), atol=1e-12)
    assert "quadraticBezierDistance" in profile.to_glsl("q")


def test_bezier_curve_profile_reduces_distance_across_multiple_spans() -> None:
    profile = BezierCurveProfile(
        points=(
            (-1.0, 0.0),
            (-0.5, 1.0),
            (0.0, 0.0),
            (0.5, -1.0),
            (1.0, 0.0),
        )
    )
    values = profile.to_numpy(
        np.asarray((-1.0, 0.0, 1.0), dtype=np.float64),
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
    )

    np.testing.assert_allclose(values, (0.0, 0.0, 0.0), atol=1e-12)
    assert profile.kind == "bezier_polycurve"


def test_bezier_curve_profile_requires_odd_anchor_control_point_order() -> None:
    with pytest.raises(ValueError, match="odd point count"):
        BezierCurveProfile(points=((0.0, 0.0), (0.5, 1.0), (1.0, 0.0), (2.0, 0.0)))


def test_polyline_closure_is_only_an_explicit_repeated_last_point() -> None:
    open_profile = PolylineProfile(
        points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    )
    closed_profile = PolylineProfile(
        points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0))
    )
    values_open = open_profile.to_numpy(
        np.asarray((0.0,), dtype=np.float64),
        np.asarray((0.5,), dtype=np.float64),
    )
    values_closed = closed_profile.to_numpy(
        np.asarray((0.0,), dtype=np.float64),
        np.asarray((0.5,), dtype=np.float64),
    )

    assert closed_profile.points[-1] == closed_profile.points[0]
    assert values_open[0] == 0.5
    assert values_closed[0] == 0.0


def test_polygon_profile_has_signed_distance() -> None:
    profile = PolygonProfile(
        points=((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    )
    values = profile.to_numpy(
        np.asarray((0.0, 1.5, 1.0), dtype=np.float64),
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
    )

    assert values[0] < 0.0
    assert values[1] > 0.0
    assert abs(values[2]) <= 1e-12
    assert profile.bounds() == (-1.0, 1.0, -1.0, 1.0)


def test_polygon_closure_is_implicit_and_repeated_last_point_is_optional() -> None:
    open_points = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    explicitly_closed_points = (*open_points, open_points[0])

    assert PolygonProfile(points=open_points).points == open_points
    assert PolygonProfile(points=explicitly_closed_points).points == open_points


def test_add_sdf_menu_polyline_and_polygon_use_trapezoid_defaults() -> None:
    document = SceneDocument()
    polyline = document.node(document.add_primitive("polyline"))
    bezier = document.node(document.add_primitive("bezier_curve"))
    bezier_polycurve = document.node(document.add_primitive("bezier_polycurve"))
    polygon = document.node(document.add_primitive("polygon"))

    assert isinstance(polyline, PlacedPolyline2D)
    assert isinstance(polyline.profile, PolylineProfile)
    assert polyline.profile.points == TRAPEZOID_POINTS
    assert isinstance(bezier, PlacedPolyline2D)
    assert isinstance(bezier.profile, BezierCurveProfile)
    assert bezier.kind == "placed_bezier_curve_2d"
    assert bezier.profile.kind == "bezier_curve"
    assert isinstance(bezier_polycurve, PlacedPolyline2D)
    assert isinstance(bezier_polycurve.profile, BezierCurveProfile)
    assert bezier_polycurve.kind == "placed_bezier_polycurve_2d"
    assert bezier_polycurve.profile.kind == "bezier_polycurve"
    assert isinstance(polygon, PlacedSDF2D)
    assert isinstance(polygon.profile, PolygonProfile)
    assert polygon.profile.points == TRAPEZOID_POINTS


def test_document_creates_polyline_and_polygon_from_points() -> None:
    document = SceneDocument()
    polyline_handle = document.add_polyline(
        ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0))
    )
    polygon_handle = document.add_polygon(
        ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    )
    polyline = document.node(polyline_handle)
    polygon = document.node(polygon_handle)

    assert isinstance(polyline, PlacedPolyline2D)
    assert polyline.dimension == 1
    assert isinstance(polygon, PlacedSDF2D)
    assert polygon.dimension == 2


def test_document_creates_point_shapes_from_world_grid_points() -> None:
    document = SceneDocument()

    polyline_handle = document.add_point_shape_from_world_points(
        "polyline",
        ((1.0, 0.0, 2.0), (2.0, 0.0, 2.0), (2.0, 0.0, 3.0)),
        "xz",
    )
    bezier_handle = document.add_point_shape_from_world_points(
        "bezier_curve",
        ((1.0, 0.0, 2.0), (1.5, 0.0, 3.0), (2.0, 0.0, 2.0)),
        "xz",
    )
    bezier_polycurve_handle = document.add_point_shape_from_world_points(
        "bezier_polycurve",
        (
            (1.0, 0.0, 2.0),
            (1.25, 0.0, 3.0),
            (1.5, 0.0, 2.0),
            (1.75, 0.0, 1.0),
            (2.0, 0.0, 2.0),
        ),
        "xz",
    )
    polygon_handle = document.add_point_shape_from_world_points(
        "polygon",
        ((0.0, 1.0, 2.0), (0.0, 2.0, 2.0), (0.0, 2.0, 3.0)),
        "yz",
    )

    polyline = document.node(polyline_handle)
    bezier = document.node(bezier_handle)
    bezier_polycurve = document.node(bezier_polycurve_handle)
    polygon = document.node(polygon_handle)

    assert isinstance(polyline, PlacedPolyline2D)
    assert polyline.origin == (1.0, 0.0, 2.0)
    assert polyline.profile is not None
    assert polyline.profile.points == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    assert isinstance(bezier, PlacedPolyline2D)
    assert isinstance(bezier.profile, BezierCurveProfile)
    assert bezier.profile.points == ((0.0, 0.0), (0.5, 1.0), (1.0, 0.0))
    assert isinstance(bezier_polycurve, PlacedPolyline2D)
    assert isinstance(bezier_polycurve.profile, BezierCurveProfile)
    assert bezier_polycurve.profile.points == (
        (0.0, 0.0),
        (0.25, 1.0),
        (0.5, 0.0),
        (0.75, -1.0),
        (1.0, 0.0),
    )
    assert isinstance(polygon, PlacedSDF2D)
    assert polygon.origin == (0.0, 1.0, 2.0)
    assert isinstance(polygon.profile, PolygonProfile)
    assert polygon.profile.points == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))


def test_document_creates_polygon_from_polyline() -> None:
    document = SceneDocument()
    handle = document.add_primitive("polyline")
    polyline = document.node(handle)
    assert isinstance(polyline, PlacedPolyline2D)
    polyline.profile = PolylineProfile(
        points=((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    )

    polygon_handle = document.create_polygon_from_polyline(handle)
    polygon = document.node(polygon_handle)

    assert isinstance(polygon, PlacedSDF2D)
    assert isinstance(polygon.profile, PolygonProfile)
    assert polygon.profile.points == polyline.profile.points
    assert polygon.origin == polyline.origin


def test_polyline_and_polygon_scene_roundtrip(tmp_path) -> None:
    document = SceneDocument()
    polyline_handle = document.add_primitive("polyline")
    document.add_primitive("bezier_curve")
    document.add_primitive("bezier_polycurve")
    document.create_polygon_from_polyline(polyline_handle)
    path = tmp_path / "polyline-polygon.casocad.json"

    save_scene(document, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile_types = [item["profile"]["type"] for item in payload["objects"]]
    restored = load_scene(path)

    assert "PolylineProfile" in profile_types
    assert "BezierCurveProfile" in profile_types
    assert "PolygonProfile" in profile_types
    assert any(
        isinstance(node, PlacedPolyline2D)
        and isinstance(node.profile, PolylineProfile)
        for node in restored.objects
    )
    assert any(
        isinstance(node, PlacedPolyline2D)
        and isinstance(node.profile, BezierCurveProfile)
        for node in restored.objects
    )
    assert any(
        isinstance(node, PlacedSDF2D) and isinstance(node.profile, PolygonProfile)
        for node in restored.objects
    )
