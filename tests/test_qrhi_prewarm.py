from __future__ import annotations

import numpy as np

from app.viewport.renderers.qrhi.surface_renderer import (
    _LINE_STRIDE,
    _SURFACE_STRIDE,
    QRhiSurfaceRenderer,
    _SurfaceChunk,
    _chunk_from_surface,
    _dynamic_line_payload,
    _safe_normal_array,
)
from app.viewport.surface_builder import (
    ViewportSurfaceCache,
    build_viewport_surface_scene,
)
from core.scene import SceneDocument


def _surface_scene(document: SceneDocument):
    version, tree = document.visual_snapshot()
    return build_viewport_surface_scene(
        tree,
        version,
        cache=ViewportSurfaceCache(resolution=14),
    )


def test_surface_renderer_has_no_tool_shader_prewarm_contract() -> None:
    renderer = QRhiSurfaceRenderer()

    assert not renderer.should_prewarm_tool_pipeline()
    assert renderer.prewarm_for_tool(None, "quadratic_bezier_surface") is None
    assert renderer.save_pipeline_cache() is None


def test_indexed_surface_chunk_uses_stable_vertex_and_index_payloads() -> None:
    document = SceneDocument()
    document.add_primitive("sphere")
    scene = _surface_scene(document)

    assert scene is not None
    chunk = _chunk_from_surface(scene.surfaces[0])

    assert chunk is not None
    assert chunk.vertex_count == scene.surfaces[0].vertex_count
    assert len(chunk.vertex_bytes) == chunk.vertex_count * _SURFACE_STRIDE
    assert chunk.index_count == scene.surfaces[0].indices.size
    assert chunk.index_count > 0
    assert chunk.wire_index_count == scene.surfaces[0].wire_indices.size
    assert chunk.thick_line_vertex_count == 0
    assert chunk.thick_line_bytes == b""


def test_outline_surface_chunk_uses_dynamic_thick_line_payload() -> None:
    document = SceneDocument()
    document.add_primitive("segment")
    scene = _surface_scene(document)

    assert scene is not None
    chunk = _chunk_from_surface(scene.surfaces[0])
    assert chunk is not None

    payload, vertex_count = _dynamic_line_payload([chunk], None)

    assert chunk.index_count == 0
    assert chunk.thick_line_vertex_count >= 6
    assert len(chunk.thick_line_bytes) == chunk.thick_line_vertex_count * _LINE_STRIDE
    assert payload == chunk.thick_line_bytes
    assert vertex_count == chunk.thick_line_vertex_count


def test_surface_upload_sanitizes_invalid_normals() -> None:
    normals = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (np.nan, 0.0, 0.0),
            (np.inf, 0.0, 0.0),
            (0.0, 3.0, 4.0),
        ),
        dtype=np.float32,
    )

    sanitized = _safe_normal_array(normals)

    np.testing.assert_allclose(sanitized[0], (0.0, 0.0, 1.0))
    np.testing.assert_allclose(sanitized[1], (0.0, 0.0, 1.0))
    np.testing.assert_allclose(sanitized[2], (0.0, 0.0, 1.0))
    np.testing.assert_allclose(sanitized[3], (0.0, 0.6, 0.8), atol=1.0e-7)


def test_surface_renderer_set_scene_replaces_cpu_chunks_before_qrhi_init() -> None:
    first_document = SceneDocument()
    first_document.add_primitive("sphere")
    first_scene = _surface_scene(first_document)
    second_document = SceneDocument()
    second_document.add_primitive("segment")
    second_scene = _surface_scene(second_document)
    renderer = QRhiSurfaceRenderer()

    renderer.set_surface_scene(first_scene)
    first_chunks = list(renderer._chunks)
    renderer.set_surface_scene(second_scene)

    assert len(first_chunks) == 1
    assert len(renderer._chunks) == 1
    assert renderer._chunks[0] is not first_chunks[0]
    assert renderer._chunks[0].index_count == 0
    assert renderer._chunks[0].thick_line_vertex_count > 0


def test_surface_renderer_retires_qrhi_chunks_before_destroying_buffers() -> None:
    class FakeBuffer:
        def __init__(self) -> None:
            self.destroy_count = 0

        def destroy(self) -> None:
            self.destroy_count += 1

    renderer = QRhiSurfaceRenderer()
    renderer._rhi = object()
    vertex_buffer = FakeBuffer()
    renderer._chunks = [
        _SurfaceChunk(
            vertex_bytes=b"",
            vertex_count=1,
            index_bytes=b"",
            index_count=0,
            wire_index_bytes=b"",
            wire_index_count=0,
            thick_line_bytes=b"",
            thick_line_vertex_count=0,
            object_id=1,
            vertex_buffer=vertex_buffer,
        )
    ]

    renderer._retire_current_chunks()

    assert renderer._chunks == []
    assert len(renderer._retired_chunks) == 1
    assert vertex_buffer.destroy_count == 0

    renderer._collect_retired_chunks()
    renderer._collect_retired_chunks()
    assert vertex_buffer.destroy_count == 0

    renderer._collect_retired_chunks()
    assert vertex_buffer.destroy_count == 1
    assert renderer._retired_chunks == []


def test_surface_camera_matrix_matches_viewport_focal_camera() -> None:
    renderer = QRhiSurfaceRenderer()
    renderer._clip_y_sign = -1.0
    renderer._depth_zero_to_one = False
    camera = {
        "u_camera_position": (0.0, 0.0, 6.0),
        "u_camera_target": (0.0, 0.0, 0.0),
        "u_camera_right": (1.0, 0.0, 0.0),
        "u_camera_up": (0.0, 1.0, 0.0),
        "u_focal_length": 1.5,
    }

    matrix = renderer._camera_matrix(800, 400, camera)

    center = _project(matrix, (0.0, 0.0, 0.0))
    right = _project(matrix, (1.0, 0.0, 0.0))
    up = _project(matrix, (0.0, 1.0, 0.0))

    np.testing.assert_allclose(center[:2], (0.0, 0.0), atol=1.0e-7)
    np.testing.assert_allclose(right[0], 1.5 * 0.5 / 6.0, atol=1.0e-7)
    np.testing.assert_allclose(up[1], 1.5 / 6.0, atol=1.0e-7)

    camera["u_focal_length"] = 3.0
    zoomed = _project(renderer._camera_matrix(800, 400, camera), (1.0, 0.0, 0.0))

    np.testing.assert_allclose(zoomed[0], 2.0 * right[0], atol=1.0e-7)


def test_surface_camera_matrix_respects_clip_y_sign() -> None:
    renderer = QRhiSurfaceRenderer()
    renderer._depth_zero_to_one = False
    camera = {
        "u_camera_position": (0.0, 0.0, 6.0),
        "u_camera_target": (0.0, 0.0, 0.0),
        "u_camera_right": (1.0, 0.0, 0.0),
        "u_camera_up": (0.0, 1.0, 0.0),
        "u_focal_length": 1.5,
    }

    renderer._clip_y_sign = -1.0
    y_up = _project(renderer._camera_matrix(800, 400, camera), (0.0, 1.0, 0.0))
    renderer._clip_y_sign = 1.0
    y_down = _project(renderer._camera_matrix(800, 400, camera), (0.0, 1.0, 0.0))

    np.testing.assert_allclose(y_up[1], -y_down[1], atol=1.0e-7)


def _project(matrix: np.ndarray, point: tuple[float, float, float]) -> np.ndarray:
    clip = matrix @ np.asarray((*point, 1.0), dtype=np.float32)
    return clip[:3] / clip[3]
