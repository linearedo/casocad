from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from app.artifacts import (
    COARSE_VIEWPORT_SURFACE_RESOLUTION,
    REFINED_VIEWPORT_SURFACE_RESOLUTION,
    REVOLVE_VIEWPORT_SURFACE_RESOLUTION,
    _next_surface_resolution,
)
from app.main_window import (
    parse_solid_from_2d_method,
    viewport_render_resolution_for_tree,
)
from core.scene import SceneDocument
from app.viewport.renderers.qrhi.surface_renderer import _GRID_FRAG
from app.viewport.renderers.qrhi.viewport import (
    QRhiViewportWidget,
    _orientation_axis_projection,
    _revolve_signal_frame,
)


def _projection(label: str, yaw: float, pitch: float) -> tuple[float, float, float]:
    projections = {
        axis: (dx, dy, depth)
        for axis, dx, dy, depth in _orientation_axis_projection(yaw, pitch)
    }
    return projections[label]


def test_orientation_projection_keeps_axes_normalized() -> None:
    for yaw, pitch in (
        (math.radians(35.0), math.radians(28.0)),
        (math.radians(-90.0), math.radians(90.0)),
        (math.radians(-90.0), 0.0),
        (0.0, 0.0),
    ):
        for label in ("X", "Y", "Z"):
            dx, dy, depth = _projection(label, yaw, pitch)
            assert abs(dx * dx + dy * dy + depth * depth - 1.0) < 1e-9


def test_top_view_projects_z_into_depth() -> None:
    dx, dy, depth = _projection("Z", math.radians(-90.0), math.radians(90.0))

    assert abs(dx) < 1e-9
    assert abs(dy) < 1e-9
    assert depth < -0.99


def test_empty_scene_grid_shader_uses_fragment_grid_math() -> None:
    assert "vec3 gridA" in _GRID_FRAG
    assert "gl_FragCoord.xy" in _GRID_FRAG
    assert "fwidth(g)" in _GRID_FRAG
    assert "u_grid_plane==1?p.xz" in _GRID_FRAG
    assert "u_grid_plane==2?p.yz" in _GRID_FRAG
    assert "u_camera_position, rd, u_max_ray_distance" in _GRID_FRAG


def test_negative_revolve_drag_keeps_radial_axis_and_emits_signed_sector() -> None:
    frame = _revolve_signal_frame(
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        -90.0,
    )

    assert frame is not None
    radial, angle = frame
    assert angle == -90.0
    assert abs(radial[0] - 1.0) < 1e-9
    assert abs(radial[1]) < 1e-9
    assert abs(radial[2]) < 1e-9


def test_revolve_command_defaults_to_custom_axis_vector_workflow() -> None:
    assert parse_solid_from_2d_method("revolve") == ("revolve", "custom")
    assert parse_solid_from_2d_method("revolve:u") == ("revolve", "u")


def test_2d_viewport_tree_renders_directly_at_refined_resolution() -> None:
    tree = SimpleNamespace(
        components=(
            SimpleNamespace(dimension=2),
            SimpleNamespace(dimension=2),
        )
    )

    resolution, refine_after = viewport_render_resolution_for_tree(tree)

    assert resolution == REFINED_VIEWPORT_SURFACE_RESOLUTION
    assert not refine_after


def test_3d_viewport_tree_starts_coarse_then_refines() -> None:
    tree = SimpleNamespace(components=(SimpleNamespace(dimension=3),))

    resolution, refine_after = viewport_render_resolution_for_tree(tree)

    # 3D objects render coarse first for responsiveness, then climb the ladder
    # to full precision off-thread.
    assert resolution == COARSE_VIEWPORT_SURFACE_RESOLUTION
    assert refine_after


def test_pure_revolve_viewport_tree_uses_single_interactive_pass() -> None:
    document = SceneDocument()
    source = document.add_primitive_from_drag(
        "circle",
        (-0.25, 0.0, 0.0),
        (0.25, 0.0, 0.0),
    )
    document.solid_from_2d([source], "revolve")
    _version, tree = document.visual_snapshot()

    resolution, refine_after = viewport_render_resolution_for_tree(tree)

    assert resolution == REVOLVE_VIEWPORT_SURFACE_RESOLUTION
    assert not refine_after


def test_boolean_viewport_tree_uses_quality_single_pass() -> None:
    document = SceneDocument()
    sphere = document.add_primitive_from_drag(
        "sphere",
        (-0.25, -0.25, 0.0),
        (0.25, 0.25, 0.0),
    )
    box = document.add_primitive_from_drag(
        "box",
        (-0.2, -0.2, 0.0),
        (0.2, 0.2, 0.0),
    )
    document.combine(sphere, box, "intersection")
    _version, tree = document.visual_snapshot()

    resolution, refine_after = viewport_render_resolution_for_tree(tree)

    # Booleans draw at a higher interactive floor than primitives (dual contour,
    # not a cheap analytic mesh), then climb the ladder to full precision.
    from app.artifacts import BOOLEAN_DRAW_RESOLUTION

    assert resolution == BOOLEAN_DRAW_RESOLUTION
    assert refine_after
    ladder = [resolution]
    while (nxt := _next_surface_resolution(ladder[-1])) is not None:
        ladder.append(nxt)
    assert ladder[-1] == REFINED_VIEWPORT_SURFACE_RESOLUTION
    assert ladder == [BOOLEAN_DRAW_RESOLUTION, 64, 96, 128]


def test_drawn_revolve_axis_vector_builds_world_axis_frame() -> None:
    viewport = SimpleNamespace(
        _grid_spacing=1.0,
        _revolve_section_normal=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        _revolve_section_center=np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
        _revolve_origin=None,
        _revolve_axis=None,
        _revolve_radial=None,
        _revolve_axis_label="",
    )

    ok = QRhiViewportWidget._set_revolve_axis_vector(
        viewport,
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
    )

    assert ok
    np.testing.assert_allclose(viewport._revolve_origin, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(viewport._revolve_axis, (1.0, 0.0, 0.0))
    np.testing.assert_allclose(viewport._revolve_radial, (0.0, 1.0, 0.0))
    assert viewport._revolve_axis_label == "X axis"
