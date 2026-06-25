from __future__ import annotations

from app.panels.display import display_kind
from core.boundary import BoundaryRegion
from core.sdf import (
    QuadraticBezierCurveProfile,
    Box,
    CircleProfile,
    PlacedPolyline1D,
    PlacedSDF1D,
    PlacedSDF2D,
    RectangleProfile,
    SegmentProfile,
)


def test_display_kind_uses_profile_for_placed_sdf_nodes() -> None:
    assert display_kind(
        PlacedSDF2D(name="floor", profile=RectangleProfile())
    ) == "rectangle"
    assert display_kind(
        PlacedSDF2D(name="inlet", profile=CircleProfile())
    ) == "circle"
    assert display_kind(
        PlacedSDF1D(name="edge", profile=SegmentProfile())
    ) == "segment"


def test_display_kind_uses_profile_for_placed_curves() -> None:
    curve = PlacedPolyline1D(name="curve", profile=QuadraticBezierCurveProfile())
    assert curve.kind == "placed_quadratic_bezier_curve_1d"
    assert display_kind(curve) == "quadratic_bezier_curve"


def test_display_kind_keeps_normal_sdf_and_boundary_kinds() -> None:
    assert display_kind(Box(name="box")) == "box"
    assert (
        display_kind(BoundaryRegion(name="wall", object_id=3, owner_object_id=1))
        == "boundary_region"
    )
