from __future__ import annotations

import logging

import numpy as np
import pytest

from core.scene import SceneDocument
from core.sdf import QuadraticBezierTube, Extrude, PolylineTube, Revolve, Scale, Translate

from ._benchmark import RenderUploadProbe, benchmark_scene_step

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def render_upload_probe() -> RenderUploadProbe | None:
    probe = RenderUploadProbe.create_optional()
    try:
        yield probe
    finally:
        if probe is not None:
            probe.close()


def test_default_startup_scene_timing(
    render_upload_probe: RenderUploadProbe | None,
) -> None:
    document = SceneDocument.default()

    _, timing = benchmark_scene_step(
        document,
        "default_startup_scene",
        lambda scene: scene.version,
        render_upload_probe,
    )

    assert timing.tree_node_count >= 3
    assert timing.render_ir_supported


def test_core_3d_operations_timing(
    render_upload_probe: RenderUploadProbe | None,
) -> None:
    document = SceneDocument()

    sphere_handle, sphere_timing = benchmark_scene_step(
        document,
        "create_sphere",
        lambda scene: scene.add_primitive("sphere"),
        render_upload_probe,
    )
    box_handle, _ = benchmark_scene_step(
        document,
        "create_box",
        lambda scene: scene.add_primitive("box"),
        render_upload_probe,
    )
    moved_handle, _ = benchmark_scene_step(
        document,
        "move_sphere",
        lambda scene: scene.move_object(sphere_handle, (0.55, 0.15, -0.10)),
        render_upload_probe,
    )
    combined_handle, _ = benchmark_scene_step(
        document,
        "boolean_difference",
        lambda scene: scene.combine(moved_handle, box_handle, "difference"),
        render_upload_probe,
    )
    translated_handle, _ = benchmark_scene_step(
        document,
        "wrap_translate",
        lambda scene: scene.wrap_transform(combined_handle, "translate"),
        render_upload_probe,
    )
    _, _ = benchmark_scene_step(
        document,
        "translate_wrapped_boolean",
        lambda scene: scene.move_object(translated_handle, (0.20, -0.10, 0.05)),
        render_upload_probe,
    )
    rotated_handle, _ = benchmark_scene_step(
        document,
        "rotate_wrapped_boolean",
        lambda scene: scene.rotate_object(translated_handle, "z", 22.5),
        render_upload_probe,
    )
    scaled_handle, _ = benchmark_scene_step(
        document,
        "wrap_scale",
        lambda scene: scene.wrap_transform(rotated_handle, "scale"),
        render_upload_probe,
    )
    _, _ = benchmark_scene_step(
        document,
        "update_scale_factor",
        lambda scene: _set_scale_factor(scene, scaled_handle, 1.18),
        render_upload_probe,
    )

    assert sphere_timing.render_ir_supported
    assert moved_handle == sphere_handle
    assert rotated_handle == translated_handle
    assert isinstance(document.node(translated_handle), Translate)
    assert isinstance(document.node(scaled_handle), Scale)


def test_core_2d_boolean_extrude_revolve_timing(
    render_upload_probe: RenderUploadProbe | None,
) -> None:
    document = SceneDocument()

    rectangle_handle, rectangle_timing = benchmark_scene_step(
        document,
        "create_rectangle",
        lambda scene: scene.add_primitive("rectangle"),
        render_upload_probe,
    )
    circle_handle, _ = benchmark_scene_step(
        document,
        "create_circle",
        lambda scene: scene.add_primitive("circle"),
        render_upload_probe,
    )
    _, _ = benchmark_scene_step(
        document,
        "move_circle",
        lambda scene: scene.move_object(circle_handle, (0.35, 0.0, 0.0)),
        render_upload_probe,
    )
    combined_handle, _ = benchmark_scene_step(
        document,
        "boolean_union_2d",
        lambda scene: scene.combine(rectangle_handle, circle_handle, "union"),
        render_upload_probe,
    )
    extrude_handle, _ = benchmark_scene_step(
        document,
        "extrude_combined_2d",
        lambda scene: scene.solid_from_2d(
            [combined_handle],
            "extrude",
            signed_height=1.4,
        ),
        render_upload_probe,
    )
    revolve_source_handle, _ = benchmark_scene_step(
        document,
        "create_revolve_source",
        lambda scene: scene.add_primitive("rectangle"),
        render_upload_probe,
    )
    _, _ = benchmark_scene_step(
        document,
        "rotate_revolve_source",
        lambda scene: scene.rotate_object(revolve_source_handle, "z", 18.0),
        render_upload_probe,
    )
    _, _ = benchmark_scene_step(
        document,
        "offset_revolve_source",
        lambda scene: scene.move_object(revolve_source_handle, (1.10, 0.0, 0.0)),
        render_upload_probe,
    )
    revolve_handle, _ = benchmark_scene_step(
        document,
        "revolve_rectangle",
        lambda scene: scene.solid_from_2d(
            [revolve_source_handle],
            "revolve",
            revolve_axis_origin=(0.0, 0.0, 0.0),
            revolve_axis_direction=(0.0, 0.0, 1.0),
            revolve_radial_direction=(1.0, 0.0, 0.0),
            revolve_angle_degrees=180.0,
        ),
        render_upload_probe,
    )

    assert rectangle_timing.render_ir_supported
    assert isinstance(document.node(extrude_handle), Extrude)
    assert isinstance(document.node(revolve_handle), Revolve)


def test_moving_revolve_preserves_shape() -> None:
    document = SceneDocument()
    source_handle = document.add_primitive_from_drag(
        "rectangle",
        (0.0, -0.5, 0.0),
        (0.5, 0.5, 0.0),
    )
    revolve_handle = document.solid_from_2d(
        [source_handle],
        "revolve",
        revolve_axis_origin=(0.0, 0.0, 0.0),
        revolve_axis_direction=(0.0, 1.0, 0.0),
        revolve_radial_direction=(1.0, 0.0, 0.0),
    )
    revolve = document.node(revolve_handle)
    assert isinstance(revolve, Revolve)

    x = np.asarray((0.0, 0.15, 0.3, 0.55), dtype=np.float64)
    y = np.asarray((-0.25, 0.0, 0.3, 0.45), dtype=np.float64)
    z = np.asarray((0.0, 0.1, -0.2, 0.35), dtype=np.float64)
    before = revolve.to_numpy(x, y, z)

    delta = (1.25, -0.35, 0.2)
    moved_handle = document.move_object(revolve_handle, delta)
    moved = document.node(moved_handle)
    assert moved_handle == revolve_handle
    assert isinstance(moved, Revolve)
    assert moved.axis_origin == delta

    after = moved.to_numpy(
        x + delta[0],
        y + delta[1],
        z + delta[2],
    )
    np.testing.assert_allclose(after, before, rtol=0.0, atol=1.0e-12)


def test_path_based_solids_timing(
    render_upload_probe: RenderUploadProbe | None,
) -> None:
    document = SceneDocument()

    polyline_handle, polyline_timing = benchmark_scene_step(
        document,
        "create_polyline_tube",
        lambda scene: scene.add_polyline_tube(
            (
                (-0.8, -0.2, 0.0),
                (-0.1, 0.3, 0.2),
                (0.6, 0.1, 0.4),
            ),
            radius=0.14,
        ),
        render_upload_probe,
    )
    _, _ = benchmark_scene_step(
        document,
        "move_polyline_tube",
        lambda scene: scene.move_object(polyline_handle, (0.15, -0.05, 0.10)),
        render_upload_probe,
    )
    _, _ = benchmark_scene_step(
        document,
        "rotate_polyline_tube",
        lambda scene: scene.rotate_object(polyline_handle, "y", 35.0),
        render_upload_probe,
    )
    quadratic_bezier_handle, _ = benchmark_scene_step(
        document,
        "create_quadratic_bezier_tube",
        lambda scene: scene.add_quadratic_bezier_tube(
            (
                (-0.6, 0.0, -0.1),
                (0.0, 0.7, 0.35),
                (0.7, -0.1, 0.2),
            ),
            radius=0.12,
        ),
        render_upload_probe,
    )
    _, _ = benchmark_scene_step(
        document,
        "move_quadratic_bezier_tube",
        lambda scene: scene.move_object(quadratic_bezier_handle, (-0.10, 0.05, 0.0)),
        render_upload_probe,
    )
    _, _ = benchmark_scene_step(
        document,
        "rotate_quadratic_bezier_tube",
        lambda scene: scene.rotate_object(quadratic_bezier_handle, "x", -20.0),
        render_upload_probe,
    )

    assert polyline_timing.render_ir_supported
    assert isinstance(document.node(polyline_handle), PolylineTube)
    assert isinstance(document.node(quadratic_bezier_handle), QuadraticBezierTube)


def test_delete_reuses_previous_render_topology_timing(
    render_upload_probe: RenderUploadProbe | None,
) -> None:
    if render_upload_probe is None:
        pytest.skip("render upload probe unavailable")

    document = SceneDocument()

    sphere_handle, first_timing = benchmark_scene_step(
        document,
        "delete_cache_create_sphere",
        lambda scene: scene.add_primitive("sphere"),
        render_upload_probe,
    )
    box_handle, _ = benchmark_scene_step(
        document,
        "delete_cache_create_box",
        lambda scene: scene.add_primitive("box"),
        render_upload_probe,
    )
    _, delete_timing = benchmark_scene_step(
        document,
        "delete_cache_remove_box_to_sphere_topology",
        lambda scene: scene.delete_many([box_handle]),
        render_upload_probe,
    )

    assert first_timing.upload is not None
    assert delete_timing.upload is not None
    assert delete_timing.upload.reused_program
    assert delete_timing.upload.program_compile_ms == 0.0
    assert document.node(sphere_handle).object_id > 0


def test_recreated_topology_reuses_program_across_object_id_changes(
    render_upload_probe: RenderUploadProbe | None,
) -> None:
    if render_upload_probe is None:
        pytest.skip("render upload probe unavailable")

    document = SceneDocument()

    first_handle, first_timing = benchmark_scene_step(
        document,
        "create_sphere_first_id",
        lambda scene: scene.add_primitive("sphere"),
        render_upload_probe,
    )
    first_object_id = document.node(first_handle).object_id
    _, _ = benchmark_scene_step(
        document,
        "delete_first_sphere",
        lambda scene: scene.delete_many([first_handle]),
        render_upload_probe,
    )
    second_handle, second_timing = benchmark_scene_step(
        document,
        "create_sphere_second_id",
        lambda scene: scene.add_primitive("sphere"),
        render_upload_probe,
    )
    second_object_id = document.node(second_handle).object_id

    assert first_timing.render_ir_supported
    assert first_timing.upload is not None
    assert second_timing.upload is not None
    assert first_object_id != second_object_id
    assert second_timing.upload.reused_program


def _set_scale_factor(
    document: SceneDocument,
    handle: int,
    factor: float,
) -> int:
    node = document.node(handle)
    assert isinstance(node, Scale)
    node.factor = float(factor)
    document.mark_changed()
    return handle
