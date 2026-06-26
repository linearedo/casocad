from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.boundary import BoundaryRegion
from core.boundary_patches import (
    boundary_patch_preview_node,
    boundary_region_preview_node,
    boundary_selector_from_node,
    pick_boundary_patch,
)
from core.domain import FluidDomain
from core.boundary_selection import (
    boundary_interval_mask,
    surface_split_selector_mask,
)
from core.render_ir import build_render_ir
from core.scene import SceneDocument
from core.serialization import load_scene, save_scene
from core.sdf import (
    QuadraticBezierCurveProfile,
    QuadraticBezierSurfaceProfile,
    Box,
    CircleProfile,
    Cylinder,
    Difference,
    EllipseProfile,
    Intersection,
    PlacedPolyline1D,
    PlacedSDF1D,
    PlacedSDF2D,
    PolygonProfile,
    PolylineProfile,
    RectangleProfile,
    SDFTree,
    SegmentProfile,
    Sphere,
    Torus,
)


def test_pick_box_face_returns_explicit_surface_patch() -> None:
    box = Box(name="volume", object_id=1)

    hit = pick_boundary_patch(
        box,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
    )

    assert hit is not None
    assert hit.owner_object_id == box.object_id
    assert hit.patch_id == "+X"
    assert hit.patch_type == "face"
    assert hit.outside_direction == 1
    assert hit.normal == (1.0, 0.0, 0.0)


def test_pick_cylinder_side_and_cap_patches() -> None:
    cylinder = Cylinder(name="pipe", object_id=2)

    side = pick_boundary_patch(
        cylinder,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
    )
    cap = pick_boundary_patch(
        cylinder,
        np.asarray((0.0, 0.0, 2.0), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    )

    assert side is not None
    assert side.patch_id == "side_wall"
    assert side.patch_type == "side_wall"
    assert cap is not None
    assert cap.patch_id == "+Z_cap"
    assert cap.patch_type == "cap"
    assert cap.outside_direction == 5


def test_pick_sphere_surface_patch() -> None:
    sphere = Sphere(name="obstacle", object_id=2)

    hit = pick_boundary_patch(
        sphere,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
    )

    assert hit is not None
    assert hit.owner_object_id == sphere.object_id
    assert hit.patch_id == "surface"
    assert hit.patch_type == "surface"
    assert hit.normal == pytest.approx((1.0, 0.0, 0.0))


def test_pick_torus_surface_patch() -> None:
    torus = Torus(name="ring", object_id=3)

    hit = pick_boundary_patch(
        torus,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
    )

    assert hit is not None
    assert hit.owner_object_id == torus.object_id
    assert hit.patch_id == "surface"
    assert hit.patch_type == "surface"
    assert hit.normal == pytest.approx((1.0, 0.0, 0.0))


def test_pick_difference_cut_surface_is_attributed_to_obstacle_patch() -> None:
    volume = Box(name="volume", object_id=1, half_size=(1.0, 1.0, 1.0))
    obstacle = Cylinder(name="obstacle", object_id=2, radius=0.25, half_height=0.8)
    fluid = Difference(name="fluid", object_id=3, left=volume, right=obstacle)

    hit = pick_boundary_patch(
        fluid,
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
    )

    assert hit is not None
    assert hit.owner_object_id == obstacle.object_id
    assert hit.patch_id == "cut_surface.side_wall"
    assert hit.patch_type == "cut_surface"
    assert hit.normal == (-1.0, -0.0, -0.0)


def test_default_von_karman_obstacle_cut_surface_is_pickable_near_opening() -> None:
    document = SceneDocument.default()
    assert document.fluid_domain is not None
    root = document.fluid_domain.root
    ray_origin = np.asarray((0.7, -1.0, 1.2), dtype=np.float64)
    target = np.asarray((0.24, 0.0, 0.0), dtype=np.float64)
    ray_direction = target - ray_origin
    ray_direction /= np.linalg.norm(ray_direction)

    hit = pick_boundary_patch(root, ray_origin, ray_direction)

    assert hit is not None
    assert hit.owner_object_id == 2
    assert hit.patch_id == "cut_surface.side_wall"
    assert hit.patch_type == "cut_surface"


def test_pick_2d_rectangle_edge_returns_curve_patch() -> None:
    rectangle = PlacedSDF2D(
        name="section",
        object_id=4,
        profile=RectangleProfile(half_size=(0.5, 0.35)),
    )

    hit = pick_boundary_patch(
        rectangle,
        np.asarray((0.5, 0.0, 2.0), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    )

    assert hit is not None
    assert hit.owner_object_id == rectangle.object_id
    assert hit.patch_id == "+U"
    assert hit.patch_type == "edge"
    assert hit.outside_direction == 1
    assert hit.normal == (1.0, 0.0, 0.0)


def test_pick_2d_rectangle_edge_allows_small_screen_miss() -> None:
    rectangle = PlacedSDF2D(
        name="section",
        object_id=4,
        profile=RectangleProfile(half_size=(0.5, 0.35)),
    )

    hit = pick_boundary_patch(
        rectangle,
        np.asarray((0.535, 0.0, 2.0), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    )

    assert hit is not None
    assert hit.patch_id == "+U"
    assert hit.patch_type == "edge"


def test_pick_2d_circle_returns_stable_curve_patch() -> None:
    circle = PlacedSDF2D(
        name="section",
        object_id=4,
        profile=CircleProfile(radius=0.5),
    )

    hit = pick_boundary_patch(
        circle,
        np.asarray((0.5, 0.0, 2.0), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    )

    assert hit is not None
    assert hit.owner_object_id == circle.object_id
    assert hit.patch_id == "curve"
    assert hit.patch_type == "curve"
    assert hit.normal == pytest.approx((1.0, 0.0, 0.0))


def test_scene_2d_boundary_region_from_hit_stores_curve_patch_metadata() -> None:
    rectangle = PlacedSDF2D(
        name="section",
        object_id=4,
        profile=RectangleProfile(half_size=(0.5, 0.35)),
    )
    document = SceneDocument(objects=[rectangle])
    document.fluid_domain = FluidDomain(rectangle)
    hit = pick_boundary_patch(
        rectangle,
        np.asarray((0.5, 0.0, 2.0), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    )
    assert hit is not None

    handle = document.add_boundary_region_from_hit(hit)
    region = document.node(handle)

    assert isinstance(region, BoundaryRegion)
    assert region.owner_object_id == rectangle.object_id
    assert region.patch_id == "+U"
    assert region.patch_type == "edge"
    assert region.outside_direction == 1
    assert region in document.fluid_domain.tag_objects


def test_2d_fluid_domain_accepts_boundary_region_metadata_tag() -> None:
    rectangle = PlacedSDF2D(
        name="section",
        object_id=4,
        profile=RectangleProfile(half_size=(0.5, 0.35)),
    )
    region = BoundaryRegion(
        name="outlet",
        object_id=5,
        owner_object_id=rectangle.object_id,
        outside_direction=1,
        patch_id="+U",
        patch_type="edge",
    )

    domain = FluidDomain(rectangle, (region,))

    assert domain.tag_objects == (region,)


def test_scene_2d_segment_selector_creates_boundary_interval_region() -> None:
    rectangle = PlacedSDF2D(
        name="section",
        object_id=4,
        profile=RectangleProfile(half_size=(0.5, 0.35)),
    )
    selector = PlacedSDF1D(
        name="outlet_interval",
        object_id=5,
        profile=SegmentProfile(half_length=0.15),
        origin=(0.5, -0.05, 0.0),
        axis_u=(0.0, 1.0, 0.0),
    )
    document = SceneDocument(objects=[rectangle, selector])
    document.fluid_domain = FluidDomain(rectangle)
    hit = pick_boundary_patch(
        rectangle,
        np.asarray((0.5, 0.0, 2.0), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    )
    assert hit is not None
    base_handle = document.add_boundary_region_from_hit(hit)
    base_region = document.node(base_handle)
    before = rectangle.to_numpy(
        np.asarray([0.25], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
    )

    interval_handle = document.add_boundary_selector_region(base_region, selector)
    interval = document.node(interval_handle)
    after = rectangle.to_numpy(
        np.asarray([0.25], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
    )

    assert interval.patch_id == "+U"
    assert interval.selector_id == f"selector:{selector.object_id}"
    assert interval.selector_type == "boundary_curve_interval"
    assert interval.selector_start == pytest.approx(((-0.2) + 0.35) / 0.7)
    assert interval.selector_end == pytest.approx((0.1 + 0.35) / 0.7)
    assert np.array_equal(before, after)


def test_scene_planar_cutter_split_creates_inside_and_outside_regions() -> None:
    rectangle = PlacedSDF2D(
        name="section",
        object_id=4,
        profile=RectangleProfile(half_size=(0.5, 0.35)),
    )
    selector = PlacedSDF1D(
        name="outlet_interval",
        object_id=5,
        profile=SegmentProfile(half_length=0.15),
        origin=(0.5, -0.05, 0.0),
        axis_u=(0.0, 1.0, 0.0),
    )
    document = SceneDocument(objects=[rectangle, selector])
    document.fluid_domain = FluidDomain(rectangle)
    hit = pick_boundary_patch(
        rectangle,
        np.asarray((0.5, 0.0, 2.0), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    )
    assert hit is not None
    base_handle = document.add_boundary_region_from_hit(hit)

    split_handles = document.add_boundary_selector_split_regions(
        document.node(base_handle),
        selector,
    )
    inside = document.node(split_handles[0])
    outside = document.node(split_handles[1])

    assert inside.selector_side == "inside"
    assert outside.selector_side == "outside"
    assert inside.selector_id == f"selector:{selector.object_id}"
    assert outside.selector_id == f"selector:{selector.object_id}"
    assert inside.selector_type == "boundary_curve_interval"
    assert outside.selector_type == "boundary_curve_interval"
    assert inside.patch_id == "+U"
    assert outside.patch_id == "+U"
    assert selector in document.fluid_domain.selector_objects


def test_2dboundary_interval_mask_restricts_mesher_samples() -> None:
    rectangle = PlacedSDF2D(
        name="section",
        object_id=4,
        profile=RectangleProfile(half_size=(0.5, 0.35)),
    )
    region = BoundaryRegion(
        name="middle_outlet",
        object_id=5,
        owner_object_id=rectangle.object_id,
        outside_direction=1,
        patch_id="+U",
        patch_type="edge",
        selector_id="selector:6",
        selector_type="boundary_curve_interval",
        selector_start=0.2,
        selector_end=0.6,
    )
    positions = np.asarray(
        (
            (0.5, -0.3, 0.0),
            (0.5, 0.0, 0.0),
            (0.5, 0.3, 0.0),
        ),
        dtype=np.float64,
    )

    mask = boundary_interval_mask(region, rectangle, positions)

    assert mask.tolist() == [False, True, False]


def test_scene_2d_circle_selector_creates_curve_interval_region() -> None:
    circle = PlacedSDF2D(
        name="section",
        object_id=4,
        profile=CircleProfile(radius=0.5),
    )
    selector = PlacedSDF1D(
        name="arc_selector",
        object_id=5,
        profile=SegmentProfile(half_length=float(np.sqrt(0.5**2 + 0.5**2) * 0.5)),
        origin=(0.25, 0.25, 0.0),
        axis_u=tuple(
            value / np.sqrt(0.5**2 + 0.5**2)
            for value in (-0.5, 0.5, 0.0)
        ),
    )
    document = SceneDocument(objects=[circle, selector])
    document.fluid_domain = FluidDomain(circle)
    hit = pick_boundary_patch(
        circle,
        np.asarray((0.5, 0.0, 2.0), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    )
    assert hit is not None
    base_handle = document.add_boundary_region_from_hit(hit)

    interval_handle = document.add_boundary_selector_region(
        document.node(base_handle),
        selector,
    )
    interval = document.node(interval_handle)

    assert interval.patch_id == "curve"
    assert interval.selector_type == "boundary_curve_interval"
    assert interval.selector_start == pytest.approx(0.0)
    assert interval.selector_end == pytest.approx(0.25)


def test_2d_circle_interval_preview_is_render_ir_supported() -> None:
    circle = PlacedSDF2D(
        name="section",
        object_id=4,
        profile=CircleProfile(radius=0.5),
    )
    region = BoundaryRegion(
        name="arc",
        object_id=5,
        owner_object_id=circle.object_id,
        patch_id="curve",
        patch_type="curve",
        selector_id="selector:6",
        selector_type="boundary_curve_interval",
        selector_start=0.0,
        selector_end=0.25,
    )

    preview = boundary_region_preview_node(circle, region)
    assert isinstance(preview, PlacedPolyline1D)
    render_ir = build_render_ir(SDFTree(preview, components=(preview,)))

    assert render_ir.supported
    assert render_ir.root_indices


def test_2d_circle_interval_mask_restricts_mesher_samples() -> None:
    circle = PlacedSDF2D(
        name="section",
        object_id=4,
        profile=CircleProfile(radius=0.5),
    )
    region = BoundaryRegion(
        name="arc",
        object_id=5,
        owner_object_id=circle.object_id,
        patch_id="curve",
        patch_type="curve",
        selector_id="selector:6",
        selector_type="boundary_curve_interval",
        selector_start=0.0,
        selector_end=0.25,
    )
    positions = np.asarray(
        (
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (-0.5, 0.0, 0.0),
        ),
        dtype=np.float64,
    )

    mask = boundary_interval_mask(region, circle, positions)

    assert mask.tolist() == [True, True, False]


def test_2d_ellipse_interval_mask_restricts_mesher_samples() -> None:
    ellipse = PlacedSDF2D(
        name="section",
        object_id=4,
        profile=EllipseProfile(semi_axes=(0.6, 0.35)),
    )
    region = BoundaryRegion(
        name="arc",
        object_id=5,
        owner_object_id=ellipse.object_id,
        patch_id="curve",
        patch_type="curve",
        selector_id="selector:6",
        selector_type="boundary_curve_interval",
        selector_start=0.0,
        selector_end=0.25,
    )
    positions = np.asarray(
        (
            (0.6, 0.0, 0.0),
            (0.0, 0.35, 0.0),
            (-0.6, 0.0, 0.0),
        ),
        dtype=np.float64,
    )

    mask = boundary_interval_mask(region, ellipse, positions)

    assert mask.tolist() == [True, True, False]


def test_1d_objects_create_boundary_selectors_without_cutting_geometry() -> None:
    segment = PlacedSDF1D(
        name="split",
        object_id=5,
        profile=SegmentProfile(),
    )
    polyline = PlacedPolyline1D(
        name="curve_selector",
        object_id=6,
        profile=PolylineProfile(),
    )
    quadratic_bezier_curve = PlacedPolyline1D(
        name="quadratic_bezier_selector",
        object_id=7,
        profile=QuadraticBezierCurveProfile(
            points=((-0.5, 0.0), (0.0, 0.4), (0.5, 0.0)),
        ),
    )
    quadratic_bezier_polycurve = PlacedPolyline1D(
        name="quadratic_bezier_polycurve_selector",
        object_id=8,
        profile=QuadraticBezierCurveProfile(
            points=(
                (-0.5, 0.0),
                (-0.25, 0.4),
                (0.0, 0.0),
                (0.25, -0.4),
                (0.5, 0.0),
            ),
        ),
    )

    surface_selector = boundary_selector_from_node(segment, domain_dimension=3)
    curve_selector = boundary_selector_from_node(polyline, domain_dimension=2)
    quadratic_bezier_selector = boundary_selector_from_node(quadratic_bezier_curve, domain_dimension=2)
    polycurve_selector = boundary_selector_from_node(quadratic_bezier_polycurve, domain_dimension=3)

    assert surface_selector is not None
    assert surface_selector.selector_type == "surface_split_curve"
    assert surface_selector.object_id == segment.object_id
    assert curve_selector is not None
    assert curve_selector.selector_type == "boundary_curve_selector"
    assert curve_selector.object_id == polyline.object_id
    assert quadratic_bezier_selector is not None
    assert quadratic_bezier_selector.selector_type == "boundary_curve_selector"
    assert quadratic_bezier_selector.object_id == quadratic_bezier_curve.object_id
    assert polycurve_selector is not None
    assert polycurve_selector.selector_type == "surface_split_curve"
    assert polycurve_selector.object_id == quadratic_bezier_polycurve.object_id


def test_scene_boundary_region_from_hit_stores_patch_identity() -> None:
    volume = Box(name="volume", object_id=1)
    document = SceneDocument(objects=[volume])
    document.fluid_domain = FluidDomain(volume)
    hit = pick_boundary_patch(
        volume,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
    )
    assert hit is not None

    handle = document.add_boundary_region_from_hit(hit)
    region = document.node(handle)

    assert region.owner_object_id == volume.object_id
    assert region.patch_id == "+X"
    assert region.patch_type == "face"
    assert region.outside_direction == 1


def test_scene_boundary_selector_region_stores_selector_without_cutting_geometry() -> None:
    volume = Box(name="volume", object_id=1)
    document = SceneDocument(objects=[volume])
    document.fluid_domain = FluidDomain(volume)
    hit = pick_boundary_patch(
        volume,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
    )
    assert hit is not None
    base_handle = document.add_boundary_region_from_hit(hit)
    selector_handle = document.add_primitive("segment")
    base_region = document.node(base_handle)
    selector = document.node(selector_handle)
    before = volume.to_numpy(
        np.asarray([0.25], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
    )

    selector_region_handle = document.add_boundary_selector_region(
        base_region,
        selector,
    )
    selector_region = document.node(selector_region_handle)
    after = volume.to_numpy(
        np.asarray([0.25], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
    )

    assert selector_region.owner_object_id == base_region.owner_object_id
    assert selector_region.patch_id == base_region.patch_id
    assert selector_region.patch_type == base_region.patch_type
    assert selector_region.selector_id == f"selector:{selector.object_id}"
    assert selector_region.selector_type == "surface_split_curve"
    assert selector in document.fluid_domain.selector_objects
    assert np.array_equal(before, after)


def test_scene_3d_solid_selector_region_stores_sdf_subregion() -> None:
    volume = Box(name="volume", object_id=1)
    cutter = Sphere(
        name="inlet_half",
        object_id=2,
        center=(0.5, 0.0, 0.0),
        radius=0.3,
    )
    document = SceneDocument(objects=[volume, cutter])
    document.fluid_domain = FluidDomain(volume)
    hit = pick_boundary_patch(
        volume,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
    )
    assert hit is not None
    base_handle = document.add_boundary_region_from_hit(hit)
    before = volume.to_numpy(
        np.asarray([0.25], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
    )

    selector_region_handle = document.add_boundary_selector_region(
        document.node(base_handle),
        cutter,
    )
    selector_region = document.node(selector_region_handle)
    after = volume.to_numpy(
        np.asarray([0.25], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
    )

    assert selector_region.selector_id == f"selector:{cutter.object_id}"
    assert selector_region.selector_type == "surface_sdf_subregion"
    assert cutter in document.fluid_domain.selector_objects
    assert np.array_equal(before, after)


def test_scene_surface_cutter_split_creates_inside_and_outside_regions() -> None:
    volume = Box(name="volume", object_id=1)
    cutter = Sphere(
        name="inlet_half",
        object_id=2,
        center=(0.5, 0.0, 0.0),
        radius=0.3,
    )
    document = SceneDocument(objects=[volume, cutter])
    document.fluid_domain = FluidDomain(volume)
    hit = pick_boundary_patch(
        volume,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
    )
    assert hit is not None
    base_handle = document.add_boundary_region_from_hit(hit)

    split_handles = document.add_boundary_selector_split_regions(
        document.node(base_handle),
        cutter,
    )
    inside = document.node(split_handles[0])
    outside = document.node(split_handles[1])

    assert inside.selector_side == "inside"
    assert outside.selector_side == "outside"
    assert inside.selector_id == f"selector:{cutter.object_id}"
    assert outside.selector_id == f"selector:{cutter.object_id}"
    assert inside.selector_type == "surface_sdf_subregion"
    assert outside.selector_type == "surface_sdf_subregion"
    assert inside.patch_id == "+X"
    assert outside.patch_id == "+X"
    assert cutter in document.fluid_domain.selector_objects


def test_pick_boundary_patch_returns_3d_sdf_selector_hit() -> None:
    volume = Box(name="volume", object_id=1)
    cutter = Sphere(
        name="inlet_half",
        object_id=2,
        center=(0.5, 0.0, 0.0),
        radius=0.3,
    )

    hit = pick_boundary_patch(
        volume,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
        selector_objects=(cutter,),
    )

    assert hit is not None
    assert hit.patch_id == "+X"
    assert hit.selector is not None
    assert hit.selector.selector_id == f"selector:{cutter.object_id}"
    assert hit.selector.selector_type == "surface_sdf_subregion"
    assert hit.selector.side == "inside"


def test_pick_boundary_patch_returns_outside_selector_hit() -> None:
    volume = Box(name="volume", object_id=1)
    cutter = Sphere(
        name="small_inlet",
        object_id=2,
        center=(0.5, 0.0, 0.0),
        radius=0.1,
    )

    hit = pick_boundary_patch(
        volume,
        np.asarray((2.0, 0.3, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
        selector_objects=(cutter,),
    )

    assert hit is not None
    assert hit.patch_id == "+X"
    assert hit.selector is not None
    assert hit.selector.selector_id == f"selector:{cutter.object_id}"
    assert hit.selector.selector_type == "surface_sdf_subregion"
    assert hit.selector.side == "outside"


def test_scene_boundary_region_from_selector_hit_stores_selector_metadata() -> None:
    volume = Box(name="volume", object_id=1)
    cutter = Sphere(
        name="inlet_half",
        object_id=2,
        center=(0.5, 0.0, 0.0),
        radius=0.3,
    )
    document = SceneDocument(objects=[volume, cutter])
    document.fluid_domain = FluidDomain(volume, selector_objects=(cutter,))
    hit = pick_boundary_patch(
        volume,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
        selector_objects=document.fluid_domain.selector_objects,
    )
    assert hit is not None
    assert hit.selector is not None

    handle = document.add_boundary_region_from_hit(hit)
    region = document.node(handle)

    assert isinstance(region, BoundaryRegion)
    assert region.patch_id == "+X"
    assert region.selector_id == f"selector:{cutter.object_id}"
    assert region.selector_type == "surface_sdf_subregion"
    assert region.selector_side == "inside"


def test_boundary_patch_preview_node_for_selector_hit_is_render_ir_supported() -> None:
    volume = Box(name="volume", object_id=1)
    cutter = Sphere(
        name="inlet_half",
        object_id=2,
        center=(0.5, 0.0, 0.0),
        radius=0.3,
    )
    hit = pick_boundary_patch(
        volume,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
        selector_objects=(cutter,),
    )
    assert hit is not None
    assert hit.selector is not None

    preview = boundary_patch_preview_node(
        volume,
        hit,
        selector_objects=(cutter,),
    )
    assert isinstance(preview, Intersection)
    render_ir = build_render_ir(SDFTree(preview, components=(preview,)))

    assert render_ir.supported
    assert render_ir.root_indices


def test_boundary_patch_preview_node_for_outside_selector_hit_is_render_ir_supported() -> None:
    volume = Box(name="volume", object_id=1)
    cutter = Sphere(
        name="small_inlet",
        object_id=2,
        center=(0.5, 0.0, 0.0),
        radius=0.1,
    )
    hit = pick_boundary_patch(
        volume,
        np.asarray((2.0, 0.3, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
        selector_objects=(cutter,),
    )
    assert hit is not None
    assert hit.selector is not None
    assert hit.selector.side == "outside"

    preview = boundary_patch_preview_node(
        volume,
        hit,
        selector_objects=(cutter,),
    )
    assert isinstance(preview, Difference)
    render_ir = build_render_ir(SDFTree(preview, components=(preview,)))

    assert render_ir.supported
    assert render_ir.root_indices


def test_boundary_patch_preview_node_for_1d_selector_hit_is_render_ir_supported() -> None:
    volume = Box(name="volume", object_id=1)
    selector = PlacedSDF1D(
        name="split",
        object_id=2,
        profile=SegmentProfile(half_length=0.25),
        origin=(0.5, 0.0, 0.0),
        axis_u=(0.0, 1.0, 0.0),
    )
    hit = pick_boundary_patch(
        volume,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
        selector_objects=(selector,),
    )
    assert hit is not None
    assert hit.selector is not None
    assert hit.selector.selector_type == "surface_split_curve"

    preview = boundary_patch_preview_node(
        volume,
        hit,
        selector_objects=(selector,),
    )
    assert isinstance(preview, Intersection)
    render_ir = build_render_ir(SDFTree(preview, components=(preview,)))

    assert render_ir.supported
    assert render_ir.root_indices


def test_boundary_patch_preview_node_for_polyline_selector_hit_is_render_ir_supported() -> None:
    volume = Box(name="volume", object_id=1)
    selector = PlacedPolyline1D(
        name="split_curve",
        object_id=2,
        profile=PolylineProfile(points=((0.5, -0.3), (0.5, 0.3))),
        origin=(0.0, 0.0, 0.0),
        axis_u=(1.0, 0.0, 0.0),
        axis_v=(0.0, 1.0, 0.0),
    )
    hit = pick_boundary_patch(
        volume,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
        selector_objects=(selector,),
    )
    assert hit is not None
    assert hit.selector is not None
    assert hit.selector.selector_id == f"selector:{selector.object_id}"
    assert hit.selector.selector_type == "surface_split_curve"

    preview = boundary_patch_preview_node(
        volume,
        hit,
        selector_objects=(selector,),
    )
    assert isinstance(preview, Intersection)
    render_ir = build_render_ir(SDFTree(preview, components=(preview,)))

    assert render_ir.supported
    assert render_ir.root_indices


def test_boundary_patch_preview_node_for_quadratic_bezier_selector_hit_is_render_ir_supported() -> None:
    volume = Box(name="volume", object_id=1)
    selector = PlacedPolyline1D(
        name="split_bezier",
        object_id=2,
        profile=QuadraticBezierCurveProfile(points=((0.5, -0.3), (0.5, 0.0), (0.5, 0.3))),
        origin=(0.0, 0.0, 0.0),
        axis_u=(1.0, 0.0, 0.0),
        axis_v=(0.0, 1.0, 0.0),
    )
    hit = pick_boundary_patch(
        volume,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
        selector_objects=(selector,),
    )
    assert hit is not None
    assert hit.selector is not None
    assert hit.selector.selector_id == f"selector:{selector.object_id}"
    assert hit.selector.selector_type == "surface_split_curve"

    preview = boundary_patch_preview_node(
        volume,
        hit,
        selector_objects=(selector,),
    )
    assert isinstance(preview, Intersection)
    render_ir = build_render_ir(SDFTree(preview, components=(preview,)))

    assert render_ir.supported
    assert render_ir.root_indices


def test_boundary_patch_preview_node_for_planar_profile_selector_is_render_ir_supported() -> None:
    volume = Box(name="volume", object_id=1)
    selector = PlacedSDF2D(
        name="planar_cut",
        object_id=2,
        profile=PolygonProfile(
            points=(
                (-0.25, -0.25),
                (0.25, -0.25),
                (0.25, 0.25),
                (-0.25, 0.25),
            )
        ),
        origin=(0.5, 0.0, 0.0),
        axis_u=(0.0, 1.0, 0.0),
        axis_v=(0.0, 0.0, 1.0),
    )
    hit = pick_boundary_patch(
        volume,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
        selector_objects=(selector,),
    )
    assert hit is not None
    assert hit.selector is not None
    assert hit.selector.selector_id == f"selector:{selector.object_id}"
    assert hit.selector.selector_type == "surface_split_profile"

    preview = boundary_patch_preview_node(
        volume,
        hit,
        selector_objects=(selector,),
    )
    assert isinstance(preview, Intersection)
    render_ir = build_render_ir(SDFTree(preview, components=(preview,)))

    assert render_ir.supported
    assert render_ir.root_indices


def test_selector_backed_boundary_region_serializes_selector_objects(
    tmp_path: Path,
) -> None:
    volume = Box(name="volume", object_id=1)
    selector = PlacedSDF1D(
        name="split",
        object_id=2,
        profile=SegmentProfile(half_length=0.25),
        origin=(0.5, 0.0, 0.0),
        axis_u=(0.0, 1.0, 0.0),
    )
    document = SceneDocument(objects=[volume, selector])
    document.fluid_domain = FluidDomain(volume)
    hit = pick_boundary_patch(
        volume,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
    )
    assert hit is not None
    base_handle = document.add_boundary_region_from_hit(hit)
    document.add_boundary_selector_region(document.node(base_handle), selector)
    path = tmp_path / "selector_region.json"

    save_scene(document, path)
    loaded = load_scene(path)

    assert loaded.fluid_domain is not None
    assert [item.object_id for item in loaded.fluid_domain.selector_objects] == [2]
    selector_regions = [
        region for region in loaded.boundary_regions if region.selector_id is not None
    ]
    assert len(selector_regions) == 1
    assert selector_regions[0].selector_id == "selector:2"
    assert selector_regions[0].selector_type == "surface_split_curve"
    assert selector_regions[0].selector_side == "inside"


def test_outside_selector_boundary_region_serializes_selector_side(
    tmp_path: Path,
) -> None:
    volume = Box(name="volume", object_id=1)
    cutter = Sphere(
        name="small_inlet",
        object_id=2,
        center=(0.5, 0.0, 0.0),
        radius=0.1,
    )
    document = SceneDocument(objects=[volume, cutter])
    document.fluid_domain = FluidDomain(volume, selector_objects=(cutter,))
    hit = pick_boundary_patch(
        volume,
        np.asarray((2.0, 0.3, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
        selector_objects=document.fluid_domain.selector_objects,
    )
    assert hit is not None
    assert hit.selector is not None
    assert hit.selector.side == "outside"
    document.add_boundary_region_from_hit(hit)
    path = tmp_path / "outside_selector_region.json"

    save_scene(document, path)
    loaded = load_scene(path)

    selector_regions = [
        region for region in loaded.boundary_regions if region.selector_id is not None
    ]
    assert len(selector_regions) == 1
    assert selector_regions[0].selector_id == "selector:2"
    assert selector_regions[0].selector_side == "outside"


def testsurface_split_selector_mask_restricts_3d_boundary_samples() -> None:
    selector = PlacedSDF1D(
        name="split",
        object_id=7,
        profile=SegmentProfile(half_length=0.25),
        origin=(0.5, 0.0, 0.0),
        axis_u=(0.0, 1.0, 0.0),
    )
    positions = np.asarray(
        (
            (0.5, 0.0, 0.0),
            (0.5, 0.35, 0.0),
            (0.5, 0.0, 0.2),
        ),
        dtype=np.float64,
    )

    mask = surface_split_selector_mask(
        f"selector:{selector.object_id}",
        {selector.object_id: selector},
        Box(name="volume", object_id=1),
        positions,
        tolerance=0.01,
    )

    assert mask.tolist() == [True, False, True]


def test_planar_profile_selector_mask_splits_surface_area_not_curve() -> None:
    selector = PlacedSDF2D(
        name="planar_cut",
        object_id=7,
        profile=PolygonProfile(
            points=(
                (-0.25, -0.25),
                (0.25, -0.25),
                (0.25, 0.25),
                (-0.25, 0.25),
            )
        ),
        origin=(0.5, 0.0, 0.0),
        axis_u=(0.0, 1.0, 0.0),
        axis_v=(0.0, 0.0, 1.0),
    )
    positions = np.asarray(
        (
            (0.5, 0.0, 0.0),
            (0.5, 0.24, 0.24),
            (0.5, 0.35, 0.0),
            (0.5, 0.0, 0.35),
        ),
        dtype=np.float64,
    )

    mask = surface_split_selector_mask(
        f"selector:{selector.object_id}",
        {selector.object_id: selector},
        Box(name="volume", object_id=1),
        positions,
        tolerance=0.0,
    )

    assert mask.tolist() == [True, True, False, False]


def test_planar_selector_mask_is_limited_to_selected_boundary_region() -> None:
    root = Box(name="volume", object_id=1)
    region = BoundaryRegion(
        name="volume +X / planar_cut inside",
        object_id=8,
        owner_object_id=root.object_id,
        outside_direction=1,
        patch_id="+X",
        patch_type="face",
        selector_id="selector:7",
        selector_type="surface_split_profile",
        selector_side="inside",
    )
    positions = np.asarray(
        (
            (0.5, 0.0, 0.0),
            (-0.5, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    selectors = (
        PlacedSDF2D(
            name="__boundary_selector_planar_segment_cutter",
            object_id=7,
            profile=PolygonProfile(
                points=(
                    (-2.0, 0.0),
                    (2.0, 0.0),
                    (2.0, 2.0),
                    (-2.0, 2.0),
                )
            ),
            origin=(0.5, 0.0, 0.0),
            axis_u=(0.0, 1.0, 0.0),
            axis_v=(0.0, 0.0, 1.0),
        ),
        PlacedSDF2D(
            name="planar_polygon_cut",
            object_id=7,
            profile=PolygonProfile(
                points=(
                    (-0.25, -0.25),
                    (0.25, -0.25),
                    (0.25, 0.25),
                    (-0.25, 0.25),
                )
            ),
            origin=(0.5, 0.0, 0.0),
            axis_u=(0.0, 1.0, 0.0),
            axis_v=(0.0, 0.0, 1.0),
        ),
        PlacedSDF2D(
            name="planar_bezier_cut",
            object_id=7,
            profile=QuadraticBezierSurfaceProfile(
                points=(
                    (-0.25, -0.25),
                    (0.0, 0.35),
                    (0.25, -0.25),
                )
            ),
            origin=(0.5, 0.0, 0.0),
            axis_u=(0.0, 1.0, 0.0),
            axis_v=(0.0, 0.0, 1.0),
        ),
    )

    for selector in selectors:
        mask = surface_split_selector_mask(
            f"selector:{selector.object_id}",
            {selector.object_id: selector},
            root,
            positions,
            region=region,
            tolerance=0.0,
        )

        assert mask.tolist() == [True, False]


def test_surface_sdf_selector_mask_uses_universal_cutter_formula() -> None:
    cutter = Sphere(
        name="cut",
        object_id=7,
        center=(0.5, 0.0, 0.0),
        radius=0.25,
    )
    positions = np.asarray(
        (
            (0.5, 0.0, 0.0),
            (0.5, 0.35, 0.0),
            (0.5, 0.0, 0.2),
        ),
        dtype=np.float64,
    )

    mask = surface_split_selector_mask(
        f"selector:{cutter.object_id}",
        {cutter.object_id: cutter},
        Box(name="volume", object_id=1),
        positions,
        tolerance=0.0,
    )

    assert mask.tolist() == [True, False, True]


def test_surface_sdf_selector_mask_supports_outside_subregion() -> None:
    cutter = Sphere(
        name="cut",
        object_id=7,
        center=(0.5, 0.0, 0.0),
        radius=0.25,
    )
    positions = np.asarray(
        (
            (0.5, 0.0, 0.0),
            (0.5, 0.35, 0.0),
            (0.5, 0.0, 0.2),
        ),
        dtype=np.float64,
    )

    mask = surface_split_selector_mask(
        f"selector:{cutter.object_id}",
        {cutter.object_id: cutter},
        Box(name="volume", object_id=1),
        positions,
        side="outside",
        tolerance=0.0,
    )

    assert mask.tolist() == [False, True, False]


def test_surface_sdf_selector_mask_is_limited_to_selected_boundary_region() -> None:
    root = Box(name="volume", object_id=1)
    cutter = Sphere(
        name="cut",
        object_id=7,
        center=(0.5, 0.0, 0.0),
        radius=0.25,
    )
    region = BoundaryRegion(
        name="volume +X / surface_cut inside",
        object_id=8,
        owner_object_id=root.object_id,
        outside_direction=1,
        patch_id="+X",
        patch_type="face",
        selector_id=f"selector:{cutter.object_id}",
        selector_type="surface_sdf_subregion",
        selector_side="inside",
    )
    inside_positions = np.asarray(
        (
            (0.5, 0.0, 0.0),
            (-0.5, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    outside_positions = np.asarray(
        (
            (0.5, 0.35, 0.0),
            (-0.5, 0.35, 0.0),
        ),
        dtype=np.float64,
    )

    inside_mask = surface_split_selector_mask(
        f"selector:{cutter.object_id}",
        {cutter.object_id: cutter},
        root,
        inside_positions,
        region=region,
        tolerance=0.0,
    )
    outside_mask = surface_split_selector_mask(
        f"selector:{cutter.object_id}",
        {cutter.object_id: cutter},
        root,
        outside_positions,
        region=region,
        side="outside",
        tolerance=0.0,
    )

    assert inside_mask.tolist() == [True, False]
    assert outside_mask.tolist() == [True, False]


def test_boundary_patch_preview_node_for_box_face_is_render_ir_supported() -> None:
    box = Box(name="volume", object_id=1)
    hit = pick_boundary_patch(
        box,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
    )
    assert hit is not None

    preview = boundary_patch_preview_node(box, hit)
    assert preview is not None
    render_ir = build_render_ir(SDFTree(preview, components=(preview,)))

    assert render_ir.supported
    assert render_ir.root_indices


def test_boundary_patch_preview_node_for_cylinder_side_uses_unique_transient_ids() -> None:
    cylinder = Cylinder(name="pipe", object_id=2)
    hit = pick_boundary_patch(
        cylinder,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
    )
    assert hit is not None

    preview = boundary_patch_preview_node(cylinder, hit)
    assert isinstance(preview, Difference)
    render_ir = build_render_ir(SDFTree(preview, components=(preview,)))

    assert render_ir.supported
    assert render_ir.root_indices


def test_boundary_patch_preview_node_for_2d_edge_is_visible_strip() -> None:
    rectangle = PlacedSDF2D(
        name="section",
        object_id=4,
        profile=RectangleProfile(half_size=(0.5, 0.35)),
    )
    hit = pick_boundary_patch(
        rectangle,
        np.asarray((0.5, 0.0, 2.0), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    )
    assert hit is not None

    preview = boundary_patch_preview_node(rectangle, hit)
    assert isinstance(preview, PlacedSDF2D)
    assert isinstance(preview.profile, RectangleProfile)
    assert preview.profile.half_size[0] == pytest.approx(0.006)
    render_ir = build_render_ir(SDFTree(preview, components=(preview,)))

    assert render_ir.supported
    assert render_ir.root_indices


def test_boundary_patch_preview_node_for_2d_circle_curve_is_render_ir_supported() -> None:
    circle = PlacedSDF2D(
        name="section",
        object_id=4,
        profile=CircleProfile(radius=0.5),
    )
    hit = pick_boundary_patch(
        circle,
        np.asarray((0.5, 0.0, 2.0), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    )
    assert hit is not None

    preview = boundary_patch_preview_node(circle, hit)
    assert isinstance(preview, PlacedSDF2D)
    render_ir = build_render_ir(SDFTree(preview, components=(preview,)))

    assert render_ir.supported
    assert render_ir.root_indices


def test_boundary_region_preview_node_for_selected_box_face_is_render_ir_supported() -> None:
    box = Box(name="volume", object_id=1)
    region = BoundaryRegion(
        name="outlet",
        object_id=2,
        owner_object_id=box.object_id,
        outside_direction=1,
        patch_id="+X",
        patch_type="face",
    )

    preview = boundary_region_preview_node(box, region)
    assert preview is not None
    render_ir = build_render_ir(SDFTree(preview, components=(preview,)))

    assert render_ir.supported
    assert render_ir.root_indices


def test_boundary_region_preview_node_for_3d_sdf_selector_is_render_ir_supported() -> None:
    box = Box(name="volume", object_id=1)
    cutter = Sphere(
        name="face_subregion",
        object_id=2,
        center=(0.5, 0.0, 0.0),
        radius=0.25,
    )
    region = BoundaryRegion(
        name="partial_outlet",
        object_id=3,
        owner_object_id=box.object_id,
        outside_direction=1,
        patch_id="+X",
        patch_type="face",
        selector_id=f"selector:{cutter.object_id}",
        selector_type="surface_sdf_subregion",
    )

    preview = boundary_region_preview_node(
        box,
        region,
        selector_objects=(cutter,),
    )
    assert isinstance(preview, Intersection)
    render_ir = build_render_ir(SDFTree(preview, components=(preview,)))

    assert render_ir.supported
    assert render_ir.root_indices


def test_boundary_region_preview_node_for_outside_3d_sdf_selector_is_render_ir_supported() -> None:
    box = Box(name="volume", object_id=1)
    cutter = Sphere(
        name="face_subregion",
        object_id=2,
        center=(0.5, 0.0, 0.0),
        radius=0.25,
    )
    region = BoundaryRegion(
        name="base_outlet",
        object_id=3,
        owner_object_id=box.object_id,
        outside_direction=1,
        patch_id="+X",
        patch_type="face",
        selector_id=f"selector:{cutter.object_id}",
        selector_type="surface_sdf_subregion",
        selector_side="outside",
    )

    preview = boundary_region_preview_node(
        box,
        region,
        selector_objects=(cutter,),
    )
    assert isinstance(preview, Difference)
    render_ir = build_render_ir(SDFTree(preview, components=(preview,)))

    assert render_ir.supported
    assert render_ir.root_indices


def test_boundary_region_preview_node_shows_both_planar_segment_split_sides() -> None:
    box = Box(name="volume", object_id=1)
    selector = PlacedSDF2D(
        name="__boundary_selector_planar_segment_cutter",
        object_id=2,
        profile=PolygonProfile(
            points=(
                (0.0, 0.0),
                (2.0, 0.0),
                (2.0, 2.0),
                (0.0, 2.0),
            )
        ),
        origin=(0.5, -1.0, 0.0),
        axis_u=(0.0, 1.0, 0.0),
        axis_v=(0.0, 0.0, 1.0),
    )
    inside = BoundaryRegion(
        name="volume +X / segment inside",
        object_id=3,
        owner_object_id=box.object_id,
        outside_direction=1,
        patch_id="+X",
        patch_type="face",
        selector_id=f"selector:{selector.object_id}",
        selector_type="surface_split_profile",
        selector_side="inside",
    )
    outside = BoundaryRegion(
        name="volume +X / segment outside",
        object_id=4,
        owner_object_id=box.object_id,
        outside_direction=1,
        patch_id="+X",
        patch_type="face",
        selector_id=f"selector:{selector.object_id}",
        selector_type="surface_split_profile",
        selector_side="outside",
    )

    inside_preview = boundary_region_preview_node(
        box,
        inside,
        selector_objects=(selector,),
    )
    outside_preview = boundary_region_preview_node(
        box,
        outside,
        selector_objects=(selector,),
    )
    assert inside_preview is not None
    assert outside_preview is not None
    render_ir = build_render_ir(
        SDFTree(
            inside_preview,
            components=(inside_preview, outside_preview),
        )
    )

    assert render_ir.supported
    assert inside_preview.to_numpy(
        np.asarray([0.512], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.25], dtype=np.float64),
    )[0] <= 0.0
    assert outside_preview.to_numpy(
        np.asarray([0.512], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([-0.25], dtype=np.float64),
    )[0] <= 0.0


def test_boundary_region_preview_node_for_2d_interval_is_visible_strip() -> None:
    rectangle = PlacedSDF2D(
        name="section",
        object_id=4,
        profile=RectangleProfile(half_size=(0.5, 0.35)),
    )
    region = BoundaryRegion(
        name="middle_outlet",
        object_id=5,
        owner_object_id=rectangle.object_id,
        outside_direction=1,
        patch_id="+U",
        patch_type="edge",
        selector_id="selector:6",
        selector_type="boundary_curve_interval",
        selector_start=0.25,
        selector_end=0.75,
    )

    preview = boundary_region_preview_node(rectangle, region)
    assert isinstance(preview, PlacedSDF2D)
    assert isinstance(preview.profile, RectangleProfile)
    assert preview.origin == pytest.approx((0.5, 0.0, 0.0))
    assert preview.profile.half_size == pytest.approx((0.006, 0.175))
    render_ir = build_render_ir(SDFTree(preview, components=(preview,)))

    assert render_ir.supported
    assert render_ir.root_indices


def test_deleting_selector_removes_selector_backed_boundary_region() -> None:
    volume = Box(name="volume", object_id=1)
    document = SceneDocument(objects=[volume])
    document.fluid_domain = FluidDomain(volume)
    hit = pick_boundary_patch(
        volume,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
    )
    assert hit is not None
    base_handle = document.add_boundary_region_from_hit(hit)
    selector_handle = document.add_primitive("segment")
    selector_region_handle = document.add_boundary_selector_region(
        document.node(base_handle),
        document.node(selector_handle),
    )

    document.delete(selector_handle)

    assert document.node(base_handle) in document.boundary_regions
    try:
        document.node(selector_region_handle)
    except KeyError:
        pass
    else:
        raise AssertionError("selector-backed region should be removed")
