from __future__ import annotations

import logging
from pathlib import Path

import pytest

from core.boundary import BoundaryRegion
from core.scene import SceneDocument
from core.serialization import load_scene, save_scene
from core.sdf import (
    BezierCurveProfile,
    BezierSurfaceProfile,
    Box,
    BoxFrame,
    CappedCone,
    CircleProfile,
    Cone,
    Cylinder,
    EllipseProfile,
    Intersection,
    PlacedPolyline2D,
    PlacedSDF1D,
    PlacedSDF2D,
    PolygonProfile,
    PolylineProfile,
    Pyramid,
    RectangleProfile,
    RegularPolygonProfile,
    Rotate,
    RoundedRectangleProfile,
    SegmentProfile,
    SmoothUnion,
    Sphere,
    SquareProfile,
    Torus,
    Union,
)

from ._benchmark import benchmark_scene_step
from ._benchmark import RenderUploadProbe

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def render_upload_probe() -> RenderUploadProbe | None:
    probe = RenderUploadProbe.create_optional()
    try:
        yield probe
    finally:
        if probe is not None:
            probe.close()


def test_all_3d_primitive_creation_timing() -> None:
    document = SceneDocument()
    expected_types = {
        "sphere": Sphere,
        "box": Box,
        "cylinder": Cylinder,
        "cone": Cone,
        "capped_cone": CappedCone,
        "pyramid": Pyramid,
        "box_frame": BoxFrame,
        "torus": Torus,
    }

    handles: dict[str, int] = {}
    for kind in expected_types:
        handle, timing = benchmark_scene_step(
            document,
            f"create_3d_{kind}",
            lambda scene, primitive=kind: scene.add_primitive(primitive),
            None,
        )
        handles[kind] = handle
        assert timing.tree_node_count >= 1
        assert isinstance(document.node(handle), expected_types[kind])

    for kind, handle in handles.items():
        _, _ = benchmark_scene_step(
            document,
            f"move_3d_{kind}",
            lambda scene, target=handle: scene.move_object(
                target,
                (0.05, -0.03, 0.02),
            ),
            None,
        )
        _, _ = benchmark_scene_step(
            document,
            f"rotate_3d_{kind}",
            lambda scene, target=handle: scene.rotate_object(target, "y", 12.0),
            None,
        )


def test_all_2d_and_1d_primitive_creation_timing() -> None:
    document = SceneDocument()
    expected_profiles = {
        "segment": SegmentProfile,
        "polyline": PolylineProfile,
        "bezier_curve": BezierCurveProfile,
        "bezier_polycurve": BezierCurveProfile,
        "circle": CircleProfile,
        "rectangle": RectangleProfile,
        "square": SquareProfile,
        "rounded_rectangle": RoundedRectangleProfile,
        "ellipse": EllipseProfile,
        "regular_polygon": RegularPolygonProfile,
        "polygon": PolygonProfile,
        "bezier_surface": BezierSurfaceProfile,
    }

    for kind, profile_type in expected_profiles.items():
        handle, timing = benchmark_scene_step(
            document,
            f"create_lower_dim_{kind}",
            lambda scene, primitive=kind: scene.add_primitive(primitive),
            None,
        )
        node = document.node(handle)
        assert timing.tree_node_count >= 1
        assert isinstance(node, (PlacedSDF1D, PlacedPolyline2D, PlacedSDF2D))
        assert isinstance(node.profile, profile_type)
        _, _ = benchmark_scene_step(
            document,
            f"move_lower_dim_{kind}",
            lambda scene, target=handle: scene.move_object(
                target,
                (0.03, 0.02, 0.0),
            ),
            None,
        )
        _, _ = benchmark_scene_step(
            document,
            f"rotate_lower_dim_{kind}",
            lambda scene, target=handle: scene.rotate_object(target, "z", 7.5),
            None,
        )


def test_lower_dimensional_render_ir_upload_timing(
    render_upload_probe: RenderUploadProbe | None,
) -> None:
    if render_upload_probe is None:
        pytest.skip("render upload probe unavailable")

    lower_dimensional_kinds = (
        "segment",
        "polyline",
        "bezier_curve",
        "bezier_polycurve",
        "rounded_rectangle",
        "ellipse",
        "regular_polygon",
        "polygon",
        "bezier_surface",
    )
    for kind in lower_dimensional_kinds:
        document = SceneDocument()
        handle, create_timing = benchmark_scene_step(
            document,
            f"render_ir_create_{kind}",
            lambda scene, primitive=kind: scene.add_primitive(primitive),
            render_upload_probe,
        )
        _, move_timing = benchmark_scene_step(
            document,
            f"render_ir_move_{kind}",
            lambda scene, target=handle: scene.move_object(
                target,
                (0.04, -0.02, 0.0),
            ),
            render_upload_probe,
        )

        assert create_timing.upload is not None
        assert move_timing.upload is not None
        assert create_timing.render_ir_supported
        assert move_timing.render_ir_supported
        assert create_timing.upload.program_compile_ms < 500.0
        assert move_timing.upload.reused_program
        assert move_timing.upload.program_compile_ms == 0.0


def test_drag_creation_timing_for_supported_tools() -> None:
    document = SceneDocument()
    drag_kinds = (
        "segment",
        "polyline",
        "bezier_curve",
        "bezier_polycurve",
        "polyline_tube",
        "bezier_tube",
        "circle",
        "rectangle",
        "square",
        "rounded_rectangle",
        "ellipse",
        "regular_polygon",
        "polygon",
        "sphere",
        "box",
        "cylinder",
        "cone",
        "capped_cone",
        "pyramid",
        "box_frame",
        "torus",
    )

    for index, kind in enumerate(drag_kinds):
        parameters = (
            {"top_diameter": 0.30}
            if kind == "capped_cone"
            else {"minor_diameter": 0.20}
            if kind == "torus"
            else None
        )
        start = (-0.75, -0.35 + index * 0.02, 0.0)
        end = (0.45, 0.55 + index * 0.02, 0.60)
        handle, timing = benchmark_scene_step(
            document,
            f"drag_create_{kind}",
            lambda scene, primitive=kind, params=parameters: (
                scene.add_primitive_from_drag(
                    primitive,
                    start,
                    end,
                    params,
                )
            ),
            None,
        )
        assert timing.tree_node_count >= 1
        assert document.node(handle).object_id > 0


def test_3d_boolean_variant_timing() -> None:
    for operation, expected_type in (
        ("union", Union),
        ("intersection", Intersection),
        ("difference", object),
        ("smooth_union", SmoothUnion),
    ):
        document = SceneDocument()
        first = document.add_primitive("sphere")
        second = document.add_primitive("box")
        _, _ = benchmark_scene_step(
            document,
            f"boolean_3d_offset_operand_{operation}",
            lambda scene: scene.move_object(second, (0.35, 0.0, 0.0)),
            None,
        )
        combined, timing = benchmark_scene_step(
            document,
            f"boolean_3d_{operation}",
            lambda scene, op=operation: scene.combine(first, second, op),
            None,
        )
        assert timing.tree_node_count >= 3
        if operation != "difference":
            assert isinstance(document.node(combined), expected_type)


def test_2d_and_1d_boolean_variant_timing() -> None:
    for operation in ("union", "intersection", "difference", "smooth_union"):
        document = SceneDocument()
        first = document.add_primitive("rectangle")
        second = document.add_primitive("circle")
        _, _ = benchmark_scene_step(
            document,
            f"boolean_2d_offset_operand_{operation}",
            lambda scene: scene.move_object(second, (0.25, 0.0, 0.0)),
            None,
        )
        combined, timing = benchmark_scene_step(
            document,
            f"boolean_2d_{operation}",
            lambda scene, op=operation: scene.combine(first, second, op),
            None,
        )
        assert timing.tree_node_count >= 2
        assert isinstance(document.node(combined), PlacedSDF2D)

    for operation in ("union", "intersection", "difference"):
        document = SceneDocument()
        first = document.add_primitive("segment")
        second = document.add_primitive("segment")
        _, _ = benchmark_scene_step(
            document,
            f"boolean_1d_offset_operand_{operation}",
            lambda scene: scene.move_object(second, (0.35, 0.0, 0.0)),
            None,
        )
        combined, timing = benchmark_scene_step(
            document,
            f"boolean_1d_{operation}",
            lambda scene, op=operation: scene.combine(first, second, op),
            None,
        )
        assert timing.tree_node_count >= 1
        assert isinstance(document.node(combined), PlacedSDF1D)


def test_curve_polygon_and_point_shape_workflow_timing() -> None:
    document = SceneDocument()
    polyline_handle, _ = benchmark_scene_step(
        document,
        "add_polyline_custom_points",
        lambda scene: scene.add_polyline(
            ((-0.5, -0.2), (0.0, 0.35), (0.55, -0.1))
        ),
        None,
    )
    polygon_handle, _ = benchmark_scene_step(
        document,
        "create_polygon_from_polyline",
        lambda scene: scene.create_polygon_from_polyline(polyline_handle),
        None,
    )
    _, _ = benchmark_scene_step(
        document,
        "add_polygon_custom_points",
        lambda scene: scene.add_polygon(
            ((-0.4, -0.3), (0.5, -0.2), (0.3, 0.45), (-0.35, 0.35))
        ),
        None,
    )
    _, _ = benchmark_scene_step(
        document,
        "add_bezier_curve_custom_points",
        lambda scene: scene.add_bezier_curve(
            ((-0.4, 0.0), (0.0, 0.6), (0.5, -0.1))
        ),
        None,
    )

    for kind in (
        "polyline",
        "bezier_curve",
        "bezier_polycurve",
        "polygon",
        "bezier_surface",
        "polyline_tube",
        "bezier_tube",
    ):
        handle, timing = benchmark_scene_step(
            document,
            f"add_point_shape_{kind}",
            lambda scene, primitive=kind: scene.add_point_shape_from_world_points(
                primitive,
                _world_points_for_shape(primitive),
                "xz",
            ),
            None,
        )
        assert timing.tree_node_count >= 1
        assert document.node(handle).object_id > 0

    assert isinstance(document.node(polygon_handle), PlacedSDF2D)


def test_copy_paste_delete_and_transform_wrappers_timing() -> None:
    document = SceneDocument()
    sphere = document.add_primitive("sphere")
    box = document.add_primitive("box")
    combined = document.combine(sphere, box, "union")

    copied, _ = benchmark_scene_step(
        document,
        "copy_combined_nodes",
        lambda scene: scene.copy_nodes([combined]),
        None,
    )
    pasted, _ = benchmark_scene_step(
        document,
        "paste_combined_nodes",
        lambda scene: scene.paste_nodes(copied, offset=(0.8, 0.1, 0.0)),
        None,
    )
    rotated, _ = benchmark_scene_step(
        document,
        "wrap_rotate_pasted_node",
        lambda scene: scene.wrap_transform(pasted[0], "rotate"),
        None,
    )
    _, _ = benchmark_scene_step(
        document,
        "update_wrapped_rotation",
        lambda scene: _set_rotation_angle(scene, rotated, 33.0),
        None,
    )
    _, _ = benchmark_scene_step(
        document,
        "delete_wrapped_rotation",
        lambda scene: scene.delete(rotated),
        None,
    )

    assert isinstance(copied, list)
    assert pasted


def test_delete_workflows_timing(
    render_upload_probe: RenderUploadProbe | None,
) -> None:
    document = SceneDocument()
    sphere = document.add_primitive("sphere")
    box = document.add_primitive("box")
    cylinder = document.add_primitive("cylinder")
    union = document.combine(sphere, box, "union")
    wrapped_union, _ = benchmark_scene_step(
        document,
        "delete_wrap_translate_union",
        lambda scene: scene.wrap_transform(union, "translate"),
        render_upload_probe,
    )
    _, _ = benchmark_scene_step(
        document,
        "delete_nested_operand_from_wrapped_union",
        lambda scene: scene.delete_many([sphere]),
        render_upload_probe,
    )
    _, _ = benchmark_scene_step(
        document,
        "delete_remaining_wrapped_union_root",
        lambda scene: scene.delete_many([wrapped_union]),
        render_upload_probe,
    )
    _, _ = benchmark_scene_step(
        document,
        "delete_standalone_cylinder",
        lambda scene: scene.delete_many([cylinder]),
        render_upload_probe,
    )

    section_document = SceneDocument()
    rectangle = section_document.add_primitive("rectangle")
    circle = section_document.add_primitive("circle")
    polygon = section_document.add_polygon(
        ((-0.4, -0.3), (0.45, -0.2), (0.35, 0.35), (-0.35, 0.30))
    )
    _, _ = benchmark_scene_step(
        section_document,
        "delete_multiple_2d_roots",
        lambda scene: scene.delete_many([rectangle, polygon]),
        render_upload_probe,
    )
    _, _ = benchmark_scene_step(
        section_document,
        "delete_last_2d_root",
        lambda scene: scene.delete_many([circle]),
        render_upload_probe,
    )


def test_domain_tags_and_boundary_region_timing() -> None:
    document = SceneDocument.default()
    root_handle = document.handle_for(document.fluid_domain.root)
    owner_object_id = document.fluid_domain.tag_objects[0].owner_object_id

    _, _ = benchmark_scene_step(
        document,
        "set_default_fluid_root",
        lambda scene: scene.set_fluid_root(root_handle),
        None,
    )
    region_handle, _ = benchmark_scene_step(
        document,
        "add_3d_boundary_region",
        lambda scene: scene.add_boundary_region(owner_object_id, 0),
        None,
    )
    _, _ = benchmark_scene_step(
        document,
        "disable_3d_boundary_region_tag",
        lambda scene: scene.set_tag_enabled(region_handle, False),
        None,
    )
    _, _ = benchmark_scene_step(
        document,
        "enable_3d_boundary_region_tag",
        lambda scene: scene.set_tag_enabled(region_handle, True),
        None,
    )

    section_document = SceneDocument()
    root_2d = section_document.add_primitive("rectangle")
    root_node = section_document.node(root_2d)
    assert isinstance(root_node, PlacedSDF2D)
    _, _ = benchmark_scene_step(
        section_document,
        "set_2d_fluid_root",
        lambda scene: scene.set_fluid_root(root_2d),
        None,
    )
    boundary_handle, _ = benchmark_scene_step(
        section_document,
        "add_2d_boundary_region",
        lambda scene: scene.add_boundary_region(root_node.object_id, 1),
        None,
    )

    assert isinstance(document.node(region_handle), BoundaryRegion)
    assert isinstance(section_document.node(boundary_handle), PlacedSDF1D)


def test_scene_serialization_current_format_timing(tmp_path: Path) -> None:
    document = SceneDocument.default()
    default_path = tmp_path / "default_scene.casocad.json"
    loaded_default, _ = benchmark_scene_step(
        document,
        "serialize_load_default_3d_scene",
        lambda scene: _save_and_load_scene(scene, default_path),
        None,
    )
    assert loaded_default.fluid_domain is not None
    assert loaded_default.fluid_domain.root.dimension == 3
    assert all(
        isinstance(tag, BoundaryRegion)
        for tag in loaded_default.fluid_domain.tag_objects
    )

    section_document = SceneDocument()
    root_2d = section_document.add_primitive("rectangle")
    root_node = section_document.node(root_2d)
    assert isinstance(root_node, PlacedSDF2D)
    section_document.set_fluid_root(root_2d)
    section_document.add_boundary_region(root_node.object_id, 1)
    section_path = tmp_path / "section_scene.casocad.json"
    loaded_section, _ = benchmark_scene_step(
        section_document,
        "serialize_load_2d_boundary_scene",
        lambda scene: _save_and_load_scene(scene, section_path),
        None,
    )
    assert loaded_section.fluid_domain is not None
    assert loaded_section.fluid_domain.root.dimension == 2
    assert all(
        isinstance(tag, (PlacedSDF1D, PlacedPolyline2D))
        for tag in loaded_section.fluid_domain.tag_objects
    )


def _save_and_load_scene(document: SceneDocument, path: Path) -> SceneDocument:
    save_scene(document, path)
    return load_scene(path)


def _set_rotation_angle(
    document: SceneDocument,
    handle: int,
    angle_degrees: float,
) -> int:
    node = document.node(handle)
    assert isinstance(node, Rotate)
    node.angle_degrees = float(angle_degrees)
    document.mark_changed()
    return handle


def _world_points_for_shape(
    kind: str,
) -> tuple[tuple[float, float, float], ...]:
    if kind in {"polyline", "polyline_tube"}:
        return (
            (-0.5, 0.0, -0.2),
            (0.0, 0.0, 0.4),
            (0.55, 0.0, -0.1),
        )
    if kind in {"bezier_curve", "bezier_tube"}:
        return (
            (-0.5, 0.0, -0.2),
            (0.0, 0.0, 0.55),
            (0.55, 0.0, -0.1),
        )
    if kind in {"bezier_polycurve", "bezier_surface"}:
        return (
            (-0.5, 0.0, -0.2),
            (-0.2, 0.0, 0.55),
            (0.1, 0.0, 0.0),
            (0.35, 0.0, -0.45),
            (0.65, 0.0, 0.25),
        )
    return (
        (-0.45, 0.0, -0.25),
        (0.45, 0.0, -0.20),
        (0.35, 0.0, 0.35),
        (-0.35, 0.0, 0.30),
    )
