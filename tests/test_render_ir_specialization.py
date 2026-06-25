from __future__ import annotations

from core.render_ir import build_render_ir
from core.sdf import (
    QuadraticBezierCurveProfile,
    QuadraticBezierSurfaceProfile,
    PlacedPolyline1D,
    PlacedSDF2D,
    PolygonProfile,
    SDFTree,
)


def test_polygon_profile_builds_direct_render_leaf() -> None:
    section = PlacedSDF2D(
        name="poly",
        profile=PolygonProfile(
            points=((-0.5, -0.3), (0.5, -0.3), (0.35, 0.4), (-0.35, 0.4))
        ),
    )
    render_ir = build_render_ir(SDFTree(root=section))
    assert render_ir.supported
    assert render_ir.nodes[render_ir.root_indices[0]].kind == "placed_polygon_2d"


def test_quadratic_bezier_surface_profile_builds_direct_render_leaf() -> None:
    section = PlacedSDF2D(
        name="quadratic_bezier",
        profile=QuadraticBezierSurfaceProfile(
            points=((-0.5, -0.3), (0.0, 0.5), (0.5, -0.3))
        ),
    )
    render_ir = build_render_ir(SDFTree(root=section))
    assert render_ir.supported
    assert render_ir.nodes[render_ir.root_indices[0]].kind == (
        "placed_quadratic_bezier_surface_2d"
    )


def test_quadratic_bezier_curve_profile_builds_single_quadratic_render_leaf() -> None:
    curve = PlacedPolyline1D(
        name="curve",
        profile=QuadraticBezierCurveProfile(
            points=((-0.5, -0.3), (0.0, 0.5), (0.5, -0.3))
        ),
    )
    render_ir = build_render_ir(SDFTree(root=curve))
    assert render_ir.supported
    assert render_ir.nodes[render_ir.root_indices[0]].kind == (
        "placed_quadratic_bezier_curve_1d"
    )


def test_quadratic_bezier_polycurve_profile_uses_loop_render_leaf() -> None:
    curve = PlacedPolyline1D(
        name="polycurve",
        profile=QuadraticBezierCurveProfile(
            points=(
                (-0.5, -0.3),
                (-0.25, 0.5),
                (0.0, -0.2),
                (0.25, 0.5),
                (0.5, -0.3),
            )
        ),
    )
    render_ir = build_render_ir(SDFTree(root=curve))
    assert render_ir.supported
    assert render_ir.nodes[render_ir.root_indices[0]].kind == (
        "placed_quadratic_bezier_polycurve_1d"
    )


def test_multi_segment_quadratic_bezier_surface_profile_uses_loop_render_leaf() -> None:
    section = PlacedSDF2D(
        name="quadratic_bezier",
        profile=QuadraticBezierSurfaceProfile(
            points=(
                (-0.5, -0.3),
                (-0.25, 0.5),
                (0.0, -0.2),
                (0.25, 0.5),
                (0.5, -0.3),
            )
        ),
    )
    render_ir = build_render_ir(SDFTree(root=section))
    assert render_ir.supported
    assert render_ir.nodes[render_ir.root_indices[0]].kind == (
        "placed_quadratic_bezier_surface_2d"
    )
