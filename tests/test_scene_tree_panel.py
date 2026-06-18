from __future__ import annotations

from app.panels.scene_tree import SDF_MENU_ITEMS, SDF_MENU_SECTIONS, sdf_icon_path


def test_sdf_menu_items_have_svg_icons() -> None:
    assert SDF_MENU_ITEMS == (
        ("Segment 1D", "segment"),
        ("Polyline 1D", "polyline"),
        ("Bezier Curve 1D", "bezier_curve"),
        ("Bezier Polycurve 1D", "bezier_polycurve"),
        ("Circle 2D", "circle"),
        ("Rectangle 2D", "rectangle"),
        ("Square 2D", "square"),
        ("Rounded Rectangle 2D", "rounded_rectangle"),
        ("Ellipse 2D", "ellipse"),
        ("Regular Polygon 2D", "regular_polygon"),
        ("Polygon 2D", "polygon"),
        ("Sphere", "sphere"),
        ("Box", "box"),
        ("Cylinder", "cylinder"),
        ("Torus", "torus"),
    )
    for _label, kind in SDF_MENU_ITEMS:
        path = sdf_icon_path(kind)
        assert path.exists(), f"missing SDF icon for {kind}"
        assert path.suffix == ".svg"


def test_sdf_menu_items_are_grouped_by_dimension() -> None:
    assert SDF_MENU_SECTIONS == (
        (
            "1D",
            (
                ("Segment 1D", "segment"),
                ("Polyline 1D", "polyline"),
                ("Bezier Curve 1D", "bezier_curve"),
                ("Bezier Polycurve 1D", "bezier_polycurve"),
            ),
        ),
        (
            "2D",
            (
                ("Circle 2D", "circle"),
                ("Rectangle 2D", "rectangle"),
                ("Square 2D", "square"),
                ("Rounded Rectangle 2D", "rounded_rectangle"),
                ("Ellipse 2D", "ellipse"),
                ("Regular Polygon 2D", "regular_polygon"),
                ("Polygon 2D", "polygon"),
            ),
        ),
        (
            "3D",
            (
                ("Sphere", "sphere"),
                ("Box", "box"),
                ("Cylinder", "cylinder"),
                ("Torus", "torus"),
            ),
        ),
    )
