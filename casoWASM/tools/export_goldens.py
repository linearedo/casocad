#!/usr/bin/env python3
"""Export golden SDF samples from the Python casoCAD kernel.

Read-only use of the existing casoCAD code: builds one fixture per node kind
(and several composed scenes), samples each on a deterministic grid inflated
around its bounding box, and writes a plain-text golden file consumed by the
Rust parity tests in casoWASM/kernel/tests/parity.rs.

Run from the casoCAD repo root:

    .venv/bin/python casoWASM/tools/export_goldens.py

Format (full-precision repr floats):

    fixture <name>
    p <x> <y> <z> <value>
    ...
    end
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from core.sdf.curtain import NormalCurtain
from core.sdf.operators import Difference, Intersection, Union, Xor
from core.sdf.placed_1d import PlacedSDF1D
from core.sdf.placed_2d import PlacedPolyline1D, PlacedSDF2D
from core.sdf.primitives_1d import BinaryProfile1D, OffsetProfile1D, SegmentProfile
from core.sdf.primitives_2d import (
    BinaryProfile,
    CircleProfile,
    DistanceOffsetProfile,
    EllipseProfile,
    OffsetProfile,
    PolygonProfile,
    PolylineProfile,
    QuadraticBezierCurveProfile,
    QuadraticBezierSurfaceProfile,
    RectangleProfile,
    RegularPolygonProfile,
    RoundedRectangleProfile,
    SquareProfile,
)
from core.sdf.primitives_3d import (
    Box,
    BoxFrame,
    CappedCone,
    Cone,
    Cylinder,
    Pyramid,
    Sphere,
    Torus,
)
from core.sdf.solid_from_2d import Extrude, Revolve
from core.sdf.transforms import Rotate, Scale, Translate
from core.sdf.tubes import PolylineTube, QuadraticBezierTube

GRID = 6
MARGIN = 0.6

ORIENT = {
    "axis_u": (0.6, 0.8, 0.0),
    "axis_v": (-0.8, 0.6, 0.0),
    "axis_w": (0.0, 0.0, 1.0),
}


def _section(profile, origin=(0.15, -0.1, 0.2), axis_u=(1.0, 0.0, 0.0), axis_v=(0.0, 0.6, 0.8)):
    return PlacedSDF2D("section", profile=profile, origin=origin, axis_u=axis_u, axis_v=axis_v)


def build_fixtures() -> dict[str, object]:
    fixtures: dict[str, object] = {}

    fixtures["sphere"] = Sphere("sphere", center=(0.1, -0.2, 0.3), radius=0.7)
    fixtures["box_axis"] = Box("box_axis", center=(0.2, 0.1, -0.1), half_size=(0.5, 0.3, 0.4))
    fixtures["box_oriented"] = Box(
        "box_oriented", center=(0.0, 0.2, 0.1), half_size=(0.4, 0.6, 0.25), **ORIENT
    )
    fixtures["cylinder"] = Cylinder("cylinder", center=(-0.1, 0.0, 0.2), radius=0.35, half_height=0.5)
    fixtures["cylinder_oriented"] = Cylinder(
        "cylinder_oriented", center=(0.1, 0.1, 0.0), radius=0.3, half_height=0.45, **ORIENT
    )
    fixtures["cone"] = Cone("cone", center=(0.0, 0.1, -0.1), radius=0.4, half_height=0.5)
    fixtures["cappedcone"] = CappedCone(
        "cappedcone", center=(0.05, -0.05, 0.1), radius_a=0.45, radius_b=0.2, half_height=0.5
    )
    fixtures["pyramid"] = Pyramid(
        "pyramid", center=(0.0, 0.0, 0.05), base_half_size=0.45, half_height=0.4
    )
    fixtures["boxframe"] = BoxFrame(
        "boxframe", center=(0.0, 0.1, 0.0), half_size=(0.5, 0.4, 0.3), thickness=0.06
    )
    fixtures["torus"] = Torus(
        "torus", center=(0.1, 0.0, -0.05), major_radius=0.5, minor_radius=0.14, **ORIENT
    )
    fixtures["polyline_tube_round"] = PolylineTube(
        "polyline_tube_round",
        points=((-0.7, -0.1, 0.0), (0.0, 0.5, 0.2), (0.7, 0.0, -0.2)),
        radius=0.12,
    )
    fixtures["polyline_tube_flat_inner"] = PolylineTube(
        "polyline_tube_flat_inner",
        points=((-0.7, -0.1, 0.0), (0.0, 0.5, 0.2), (0.7, 0.0, -0.2)),
        radius=0.15,
        inner_radius=0.05,
        caps="flat",
    )
    fixtures["bezier_tube_round"] = QuadraticBezierTube(
        "bezier_tube_round",
        points=((-0.75, 0.0, 0.0), (0.0, 0.55, 0.3), (0.75, 0.0, 0.0)),
        radius=0.12,
    )
    fixtures["bezier_polycurve_tube_flat"] = QuadraticBezierTube(
        "bezier_polycurve_tube_flat",
        points=(
            (-0.8, 0.0, 0.0),
            (-0.4, 0.5, 0.1),
            (0.0, 0.1, 0.2),
            (0.4, -0.4, 0.1),
            (0.8, 0.1, 0.0),
        ),
        radius=0.1,
        inner_radius=0.03,
        caps="flat",
    )

    profiles = {
        "circle": CircleProfile(center=(0.1, -0.05), radius=0.45),
        "rectangle": RectangleProfile(center=(0.0, 0.1), half_size=(0.5, 0.3)),
        "square": SquareProfile(center=(-0.1, 0.0), half_size=0.4),
        "rounded_rectangle": RoundedRectangleProfile(
            center=(0.0, 0.0), half_size=(0.5, 0.35), corner_radius=0.12
        ),
        "ellipse": EllipseProfile(center=(0.05, -0.1), semi_axes=(0.6, 0.3)),
        "ellipse_circular": EllipseProfile(center=(0.0, 0.0), semi_axes=(0.4, 0.4)),
        "regular_polygon": RegularPolygonProfile(
            center=(0.0, 0.05), radius=0.5, side_count=5, rotation=0.3
        ),
        "polygon": PolygonProfile(
            points=((-0.6, -0.4), (0.6, -0.4), (0.2, 0.1), (0.5, 0.5), (-0.4, 0.4))
        ),
        "polyline": PolylineProfile(points=((-0.6, -0.4), (0.6, -0.4), (0.35, 0.4), (-0.35, 0.4))),
        "bezier_curve": QuadraticBezierCurveProfile(
            points=((-0.6, -0.35), (0.0, 0.55), (0.6, -0.35))
        ),
        "bezier_polycurve": QuadraticBezierCurveProfile(
            points=((-0.65, -0.35), (-0.25, 0.55), (0.1, 0.25), (0.45, -0.05), (0.55, -0.45))
        ),
        "bezier_surface_open": QuadraticBezierSurfaceProfile(
            points=((-0.65, -0.35), (-0.25, 0.55), (0.1, 0.25), (0.45, -0.05), (0.55, -0.45))
        ),
        "bezier_surface_closed": QuadraticBezierSurfaceProfile(
            points=(
                (-0.5, -0.3),
                (0.0, 0.6),
                (0.5, -0.3),
                (0.0, -0.7),
                (-0.5, -0.3),
            )
        ),
        "offset": OffsetProfile(child=CircleProfile(center=(0.0, 0.0), radius=0.3), offset=(0.2, -0.15)),
        "distance_offset": DistanceOffsetProfile(
            child=RectangleProfile(center=(0.0, 0.0), half_size=(0.4, 0.25)), offset=0.08
        ),
        "binary_union": BinaryProfile(
            left=CircleProfile(center=(-0.2, 0.0), radius=0.3),
            right=RectangleProfile(center=(0.2, 0.0), half_size=(0.3, 0.2)),
            operation="union",
        ),
        "binary_difference": BinaryProfile(
            left=RectangleProfile(center=(0.0, 0.0), half_size=(0.5, 0.35)),
            right=CircleProfile(center=(0.15, 0.05), radius=0.2),
            operation="difference",
        ),
    }
    for key, profile in profiles.items():
        fixtures[f"placed2d_{key}"] = _section(profile)

    fixtures["placed1d_segment"] = PlacedSDF1D(
        "placed1d_segment",
        profile=SegmentProfile(center=0.1, half_length=0.6),
        origin=(0.0, 0.1, -0.05),
        axis_u=(0.6, 0.8, 0.0),
    )
    fixtures["placed1d_binary"] = PlacedSDF1D(
        "placed1d_binary",
        profile=BinaryProfile1D(
            left=SegmentProfile(center=-0.2, half_length=0.4),
            right=OffsetProfile1D(child=SegmentProfile(center=0.0, half_length=0.3), offset=0.35),
            operation="difference",
        ),
        origin=(0.1, 0.0, 0.0),
        axis_u=(0.0, 0.0, 1.0),
    )
    fixtures["placed_polyline_1d"] = PlacedPolyline1D(
        "placed_polyline_1d",
        profile=PolylineProfile(points=((-0.5, -0.2), (0.0, 0.3), (0.5, -0.1))),
        origin=(0.0, 0.0, 0.1),
        axis_u=(1.0, 0.0, 0.0),
        axis_v=(0.0, 0.6, 0.8),
    )
    fixtures["placed_bezier_1d"] = PlacedPolyline1D(
        "placed_bezier_1d",
        profile=QuadraticBezierCurveProfile(points=((-0.5, -0.2), (0.0, 0.5), (0.5, -0.2))),
        origin=(0.05, -0.05, 0.0),
        axis_u=(1.0, 0.0, 0.0),
        axis_v=(0.0, 1.0, 0.0),
    )

    fixtures["extrude"] = Extrude(
        "extrude",
        section=_section(PolygonProfile(points=((-0.4, -0.3), (0.5, -0.2), (0.3, 0.4), (-0.35, 0.3)))),
        height=0.8,
        center_offset=0.15,
    )
    fixtures["revolve_full"] = Revolve(
        "revolve_full",
        section=_section(CircleProfile(center=(0.45, 0.0), radius=0.15)),
        axis="v",
    )
    fixtures["revolve_partial"] = Revolve(
        "revolve_partial",
        section=_section(CircleProfile(center=(0.45, 0.0), radius=0.15)),
        axis="v",
        angle_degrees=120.0,
    )
    fixtures["revolve_negative"] = Revolve(
        "revolve_negative",
        section=_section(RectangleProfile(center=(0.4, 0.1), half_size=(0.12, 0.2))),
        axis="u",
        angle_degrees=-90.0,
    )

    fixtures["normalcurtain"] = NormalCurtain(
        "normalcurtain",
        points=((-0.5, -0.1, 0.0), (0.0, 0.2, 0.05), (0.5, 0.1, -0.05)),
        normals=((0.0, 0.1, 1.0), (0.1, 0.0, 1.0), (0.0, -0.1, 1.0)),
        extent=2.0,
    )

    sphere = Sphere("op_sphere", center=(0.2, 0.0, 0.0), radius=0.5)
    box = Box("op_box", center=(-0.1, 0.0, 0.0), half_size=(0.45, 0.35, 0.3))
    cyl = Cylinder("op_cyl", center=(0.0, 0.2, 0.0), radius=0.25, half_height=0.6)
    torus = Torus("op_torus", center=(0.0, 0.0, 0.1), major_radius=0.45, minor_radius=0.12)
    fixtures["op_union"] = Union("op_union", left=sphere, right=box)
    fixtures["op_intersection"] = Intersection("op_intersection", left=sphere, right=box)
    fixtures["op_difference"] = Difference("op_difference", left=box, right=sphere)
    fixtures["op_xor"] = Xor("op_xor", left=sphere, right=box)
    fixtures["op_nested"] = Difference(
        "op_nested",
        left=Intersection("nested_i", left=box, right=sphere),
        right=Union("nested_u", left=cyl, right=torus),
    )

    fixtures["transform_translate"] = Translate(
        "transform_translate",
        child=Cylinder("t_cyl", center=(0.0, 0.0, 0.0), radius=0.3, half_height=0.4),
        offset=(0.3, -0.2, 0.1),
    )
    fixtures["transform_scale"] = Scale(
        "transform_scale",
        child=Box("s_box", center=(0.1, 0.0, 0.0), half_size=(0.3, 0.2, 0.25)),
        factor=1.7,
    )
    fixtures["transform_rotate_x"] = Rotate(
        "transform_rotate_x",
        child=Box("r_box_x", center=(0.0, 0.1, 0.0), half_size=(0.4, 0.2, 0.3)),
        axis="x",
        angle_degrees=35.0,
    )
    fixtures["transform_rotate_y"] = Rotate(
        "transform_rotate_y",
        child=Box("r_box_y", center=(0.0, 0.1, 0.0), half_size=(0.4, 0.2, 0.3)),
        axis="y",
        angle_degrees=-50.0,
    )
    fixtures["transform_rotate_z"] = Rotate(
        "transform_rotate_z",
        child=Box("r_box_z", center=(0.0, 0.1, 0.0), half_size=(0.4, 0.2, 0.3)),
        axis="z",
        angle_degrees=120.0,
    )
    fixtures["transform_stack"] = Translate(
        "transform_stack",
        child=Rotate(
            "stack_rot",
            child=Scale(
                "stack_scale",
                child=Torus("stack_torus", major_radius=0.4, minor_radius=0.1),
                factor=1.3,
            ),
            axis="y",
            angle_degrees=40.0,
        ),
        offset=(0.2, 0.1, -0.15),
    )

    fixtures["von_karman"] = Difference(
        "von_karman",
        left=Box("flow_volume", center=(0.0, 0.0, 0.0), half_size=(1.6, 0.7, 0.45)),
        right=Cylinder("cylinder_obstacle", center=(0.0, 0.0, 0.0), radius=0.24, half_height=0.55),
    )

    return fixtures


def sample_points(node) -> np.ndarray:
    box = node.bounding_box()
    spans = (
        (box.x_min, box.x_max),
        (box.y_min, box.y_max),
        (box.z_min, box.z_max),
    )
    axes = []
    for minimum, maximum in spans:
        pad = MARGIN * max(maximum - minimum, 0.25)
        axes.append(np.linspace(minimum - pad, maximum + pad, GRID))
    grid_x, grid_y, grid_z = np.meshgrid(*axes, indexing="ij")
    points = np.stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()), axis=1)
    extras = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (-1.0, 0.5, -0.25),
            (0.123456789, -0.987654321, 0.5),
            (10.0, -10.0, 10.0),
        ],
        dtype=np.float64,
    )
    return np.concatenate((points, extras), axis=0)


def main() -> int:
    fixtures = build_fixtures()
    output = Path(__file__).resolve().parents[1] / "kernel" / "tests" / "goldens" / "kernel_goldens.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for name in sorted(fixtures):
        node = fixtures[name]
        points = sample_points(node)
        values = node.to_numpy(points[:, 0], points[:, 1], points[:, 2])
        lines.append(f"fixture {name}")
        for point, value in zip(points, values):
            lines.append(
                "p "
                f"{float(point[0])!r} {float(point[1])!r} "
                f"{float(point[2])!r} {float(value)!r}"
            )
        lines.append("end")
    output.write_text("\n".join(lines) + "\n")
    total = sum(len(sample_points(fixtures[name])) for name in fixtures)
    print(f"wrote {len(fixtures)} fixtures / {total} samples to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
