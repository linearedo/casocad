from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QRhiWidget

from .loader import MeshArtifactLoader, MeshPreviewChunk, MeshPreviewSummary
from .renderer import QRhiMeshRenderer


_BACKENDS = {
    "vulkan": QRhiWidget.Api.Vulkan,
    "opengl": QRhiWidget.Api.OpenGL,
    "metal": getattr(QRhiWidget.Api, "Metal", None),
    "d3d11": getattr(QRhiWidget.Api, "Direct3D11", None),
}


def _choose_api() -> "QRhiWidget.Api | None":
    import os

    wanted = os.environ.get("QRHI_BACKEND", "").lower()
    if wanted in _BACKENDS and _BACKENDS[wanted] is not None:
        return _BACKENDS[wanted]
    return None


class QRhiMeshViewerWidget(QRhiWidget):
    status_changed = Signal(str)
    summary_changed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        api = _choose_api()
        if api is not None:
            self.setApi(api)
        self.setMinimumSize(520, 360)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._renderer = QRhiMeshRenderer()
        self._renderer_ready = False
        self._loader = MeshArtifactLoader()
        self._loader.chunk_loaded.connect(self._on_chunk_loaded)
        self._loader.finished.connect(self._on_load_finished)
        self._loader.failed.connect(self._on_load_failed)
        self._vertices: list[np.ndarray] = []
        self._colors: list[np.ndarray] = []
        self._wire_vertices: list[np.ndarray] = []
        self._wire_colors: list[np.ndarray] = []
        self._loaded_render_triangles = 0
        self._loaded_wire_edges = 0
        self._loaded_chunk_count = 0
        self._target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._distance = 6.0
        self._yaw = math.radians(35.0)
        self._pitch = math.radians(28.0)
        self._last_pos = None
        self._background = (0.055, 0.060, 0.068)
        self._preview_vertex_limit = 300_000
        self._filled_visible = False
        self._wireframe_visible = True

    def load_artifact(self, path: str | Path) -> None:
        self.clear_mesh()
        self._loader.set_preview_limits(max_preview_vertices=self._preview_vertex_limit)
        self.status_changed.emit(
            f"Loading mesh artifact: {Path(path)} "
            f"(max render triangles={self.preview_render_triangle_limit():,})"
        )
        self._loader.load(path)

    def set_preview_vertex_limit(self, limit: int) -> None:
        self._preview_vertex_limit = max(3, int(limit))

    def preview_vertex_limit(self) -> int:
        return self._preview_vertex_limit

    def set_preview_render_triangle_limit(self, limit: int) -> None:
        self.set_preview_vertex_limit(max(1, int(limit)) * 3)

    def preview_render_triangle_limit(self) -> int:
        return self._preview_vertex_limit // 3

    def set_filled_visible(self, visible: bool) -> None:
        self._filled_visible = bool(visible)
        self.update()

    def filled_visible(self) -> bool:
        return self._filled_visible

    def set_wireframe_visible(self, visible: bool) -> None:
        self._wireframe_visible = bool(visible)
        self.update()

    def wireframe_visible(self) -> bool:
        return self._wireframe_visible

    def clear_mesh(self) -> None:
        self._vertices.clear()
        self._colors.clear()
        self._wire_vertices.clear()
        self._wire_colors.clear()
        self._loaded_render_triangles = 0
        self._loaded_wire_edges = 0
        self._loaded_chunk_count = 0
        self._renderer.clear()
        self.update()

    def initialize(self, cb) -> None:
        if self._renderer_ready:
            return
        self._renderer.initialize(self.rhi(), self.renderTarget())
        self._renderer_ready = True
        if self._vertices:
            self._upload_current_mesh()

    def render(self, cb) -> None:
        self._renderer.render(cb, self.renderTarget(), self._camera_values())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_pos = event.position()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_pos is None:
            return
        delta = event.position() - self._last_pos
        self._last_pos = event.position()
        self._yaw -= delta.x() * 0.01
        self._pitch = max(-1.5, min(1.5, self._pitch + delta.y() * 0.01))
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._last_pos = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta == 0:
            return
        self._distance = max(0.01, min(1.0e6, self._distance * math.exp(-delta * 0.0012)))
        self.update()

    def _on_chunk_loaded(self, chunk: MeshPreviewChunk) -> None:
        self._vertices.append(chunk.vertices)
        self._colors.append(chunk.colors)
        if chunk.wire_vertices.size:
            self._wire_vertices.append(chunk.wire_vertices)
            self._wire_colors.append(chunk.wire_colors)
        self._loaded_render_triangles += chunk.triangle_count
        self._loaded_wire_edges += chunk.edge_count
        self._loaded_chunk_count += 1
        if self._loaded_chunk_count == 1 or self._loaded_chunk_count % 25 == 0:
            self.status_changed.emit(
                "Loaded preview chunks: "
                f"{self._loaded_chunk_count} chunk(s), "
                f"{self._loaded_render_triangles} render triangle(s), "
                f"{self._loaded_wire_edges} wire edge(s)"
            )

    def _on_load_finished(self, summary: MeshPreviewSummary) -> None:
        if self._vertices:
            self.status_changed.emit(
                "Uploading preview to GPU: "
                f"{summary.preview_vertex_count} fill vertex/vertices, "
                f"{summary.preview_edge_count} wire edge(s)"
            )
        self._upload_current_mesh()
        self._frame_bounds(summary.bounds_min, summary.bounds_max)
        suffix = " (preview truncated)" if summary.truncated else ""
        self.status_changed.emit(
            "Mesh preview ready: "
            f"{summary.element_count} mesh element(s), "
            f"{summary.preview_triangle_count} render triangle(s), "
            f"{summary.preview_edge_count} wire edge(s){suffix}"
        )
        self.summary_changed.emit(summary)
        self.update()

    def _on_load_failed(self, message: str) -> None:
        self.status_changed.emit(f"Mesh preview failed: {message}")

    def _upload_current_mesh(self) -> None:
        if not self._vertices:
            self._renderer.clear()
            return
        vertices = np.vstack(self._vertices).astype(np.float32, copy=False)
        colors = np.vstack(self._colors).astype(np.float32, copy=False)
        wire_vertices = (
            np.vstack(self._wire_vertices).astype(np.float32, copy=False)
            if self._wire_vertices
            else np.zeros((0, 3), dtype=np.float32)
        )
        wire_colors = (
            np.vstack(self._wire_colors).astype(np.float32, copy=False)
            if self._wire_colors
            else np.zeros((0, 3), dtype=np.float32)
        )
        self._renderer.set_mesh(vertices, colors, wire_vertices, wire_colors)

    def _frame_bounds(
        self,
        bounds_min: tuple[float, float, float],
        bounds_max: tuple[float, float, float],
    ) -> None:
        lo = np.asarray(bounds_min, dtype=np.float64)
        hi = np.asarray(bounds_max, dtype=np.float64)
        center = (lo + hi) * 0.5
        diagonal = float(np.linalg.norm(hi - lo))
        self._target = center
        self._distance = max(diagonal * 1.8, 2.0)

    def _camera_values(self) -> dict[str, object]:
        cos_pitch = math.cos(self._pitch)
        direction = np.array(
            [
                math.cos(self._yaw) * cos_pitch,
                math.sin(self._yaw) * cos_pitch,
                math.sin(self._pitch),
            ],
            dtype=np.float64,
        )
        position = self._target + direction * self._distance
        return {
            "position": tuple(float(value) for value in position),
            "target": tuple(float(value) for value in self._target),
            "up": (0.0, 0.0, 1.0),
            "distance": self._distance,
            "background_color": self._background,
            "filled_visible": self._filled_visible,
            "wireframe_visible": self._wireframe_visible,
        }


__all__ = ["QRhiMeshViewerWidget"]
