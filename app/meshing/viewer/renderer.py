from __future__ import annotations

import logging
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QByteArray
from PySide6.QtGui import (
    QColor,
    QRhiBuffer,
    QRhiDepthStencilClearValue,
    QRhiGraphicsPipeline,
    QRhiShaderResourceBinding,
    QRhiShaderStage,
    QRhiVertexInputAttribute,
    QRhiVertexInputBinding,
    QRhiVertexInputLayout,
    QRhiViewport,
    QShader,
)


log = logging.getLogger(__name__)

_VERTEX_STRIDE = 24
_VERT = """\
#version 450
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 0) out vec3 v_color;
layout(std140, binding = 0) uniform MeshUBO {
    mat4 mvp;
};
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_color = in_color;
}
"""
_FRAG = """\
#version 450
layout(location = 0) in vec3 v_color;
layout(location = 0) out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}
"""


def _bake(ext: str, glsl: str) -> QShader:
    temp_dir = tempfile.mkdtemp(prefix="casocad_mesh_qrhi_")
    source = os.path.join(temp_dir, f"shader.{ext}")
    output = source + ".qsb"
    with open(source, "w", encoding="utf-8") as stream:
        stream.write(glsl)
    qsb = os.path.join(os.path.dirname(sys.executable), "pyside6-qsb")
    subprocess.run([qsb, "--glsl", "430", "-o", output, source], check=True)
    with open(output, "rb") as stream:
        return QShader.fromSerialized(QByteArray(stream.read()))


def _pack_mat4(matrix: NDArray[np.float32]) -> bytes:
    return struct.pack("<16f", *matrix.T.reshape(-1))


def _normalize(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return vector
    return vector / norm


def _look_at(
    eye: NDArray[np.float64],
    target: NDArray[np.float64],
    up: NDArray[np.float64],
) -> NDArray[np.float64]:
    forward = _normalize(target - eye)
    right = _normalize(np.cross(forward, up))
    if float(np.linalg.norm(right)) <= 1.0e-12:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    camera_up = np.cross(right, forward)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, :3] = right
    matrix[1, :3] = camera_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -float(np.dot(right, eye))
    matrix[1, 3] = -float(np.dot(camera_up, eye))
    matrix[2, 3] = float(np.dot(forward, eye))
    return matrix


def _perspective(
    fov_y_radians: float,
    aspect: float,
    near: float,
    far: float,
    *,
    depth_zero_to_one: bool,
) -> NDArray[np.float64]:
    focal = 1.0 / np.tan(fov_y_radians * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = focal / max(aspect, 1.0e-6)
    matrix[1, 1] = focal
    if depth_zero_to_one:
        matrix[2, 2] = far / (near - far)
        matrix[2, 3] = (far * near) / (near - far)
    else:
        matrix[2, 2] = (far + near) / (near - far)
        matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


class QRhiMeshRenderer:
    def __init__(self) -> None:
        self._rhi = None
        self._render_pass_descriptor = None
        self._vertex_shader = None
        self._fragment_shader = None
        self._uniform_buffer = None
        self._vertex_buffer = None
        self._wire_vertex_buffer = None
        self._shader_resources = None
        self._pipeline = None
        self._wire_pipeline = None
        self._vertex_bytes = b""
        self._wire_vertex_bytes = b""
        self._vertex_count = 0
        self._wire_vertex_count = 0
        self._uploaded = False
        self._wire_uploaded = False
        self._clip_y_sign = 1.0
        self._depth_zero_to_one = True

    def initialize(self, rhi, render_target) -> None:
        self._rhi = rhi
        self._render_pass_descriptor = render_target.renderPassDescriptor()
        self._clip_y_sign = -1.0 if rhi.isYUpInNDC() else 1.0
        if hasattr(rhi, "isClipDepthZeroToOne"):
            self._depth_zero_to_one = bool(rhi.isClipDepthZeroToOne())
        backend = rhi.backendName() if hasattr(rhi, "backendName") else "?"
        log.info(
            "mesh qrhi: initialize backend=%s clip_y_sign=%+.0f depth_zero_to_one=%s",
            backend,
            self._clip_y_sign,
            self._depth_zero_to_one,
        )
        self._vertex_shader = _bake("vert", _VERT)
        self._fragment_shader = _bake("frag", _FRAG)
        self._uniform_buffer = rhi.newBuffer(
            QRhiBuffer.Type.Dynamic,
            QRhiBuffer.UsageFlag.UniformBuffer,
            64,
        )
        self._uniform_buffer.create()
        self._build_pipeline()
        if self._vertex_bytes:
            self._build_vertex_buffer()
        if self._wire_vertex_bytes:
            self._build_wire_vertex_buffer()

    def set_mesh(
        self,
        vertices: NDArray[np.float32],
        colors: NDArray[np.float32],
        wire_vertices: NDArray[np.float32] | None = None,
        wire_colors: NDArray[np.float32] | None = None,
    ) -> None:
        if vertices.shape != colors.shape:
            raise ValueError("mesh vertices and colors must have the same shape")
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("mesh vertices must have shape (N, 3)")
        interleaved = np.column_stack((vertices, colors)).astype(np.float32, copy=False)
        self._vertex_bytes = interleaved.tobytes()
        self._vertex_count = int(vertices.shape[0])
        self._uploaded = False
        if wire_vertices is not None and wire_colors is not None:
            if wire_vertices.shape != wire_colors.shape:
                raise ValueError("wire vertices and colors must have the same shape")
            if wire_vertices.ndim != 2 or wire_vertices.shape[1] != 3:
                raise ValueError("wire vertices must have shape (N, 3)")
            wire = np.column_stack((wire_vertices, wire_colors)).astype(
                np.float32,
                copy=False,
            )
            self._wire_vertex_bytes = wire.tobytes()
            self._wire_vertex_count = int(wire_vertices.shape[0])
        else:
            self._wire_vertex_bytes = b""
            self._wire_vertex_count = 0
        self._wire_uploaded = False
        if self._rhi is not None:
            self._build_vertex_buffer()
            self._build_wire_vertex_buffer()

    def clear(self) -> None:
        self._vertex_bytes = b""
        self._wire_vertex_bytes = b""
        self._vertex_count = 0
        self._wire_vertex_count = 0
        self._uploaded = False
        self._wire_uploaded = False
        self._vertex_buffer = None
        self._wire_vertex_buffer = None

    def _build_vertex_buffer(self) -> None:
        assert self._rhi is not None
        size = max(len(self._vertex_bytes), _VERTEX_STRIDE)
        self._vertex_buffer = self._rhi.newBuffer(
            QRhiBuffer.Type.Static,
            QRhiBuffer.UsageFlag.VertexBuffer,
            size,
        )
        self._vertex_buffer.create()
        self._uploaded = False

    def _build_wire_vertex_buffer(self) -> None:
        assert self._rhi is not None
        if not self._wire_vertex_bytes:
            self._wire_vertex_buffer = None
            return
        size = max(len(self._wire_vertex_bytes), _VERTEX_STRIDE)
        self._wire_vertex_buffer = self._rhi.newBuffer(
            QRhiBuffer.Type.Static,
            QRhiBuffer.UsageFlag.VertexBuffer,
            size,
        )
        self._wire_vertex_buffer.create()
        self._wire_uploaded = False

    def _build_pipeline(self) -> None:
        assert self._rhi is not None
        vertex_stage = QRhiShaderResourceBinding.StageFlag.VertexStage
        self._shader_resources = self._rhi.newShaderResourceBindings()
        self._shader_resources.setBindings(
            [QRhiShaderResourceBinding.uniformBuffer(0, vertex_stage, self._uniform_buffer)]
        )
        self._shader_resources.create()
        float3 = QRhiVertexInputAttribute.Format.Float3
        layout = QRhiVertexInputLayout()
        layout.setBindings([QRhiVertexInputBinding(_VERTEX_STRIDE)])
        layout.setAttributes(
            [
                QRhiVertexInputAttribute(0, 0, float3, 0),
                QRhiVertexInputAttribute(0, 1, float3, 12),
            ]
        )
        pipeline = self._rhi.newGraphicsPipeline()
        pipeline.setTopology(QRhiGraphicsPipeline.Topology.Triangles)
        pipeline.setCullMode(QRhiGraphicsPipeline.CullMode.None_)
        pipeline.setDepthTest(True)
        pipeline.setDepthWrite(True)
        pipeline.setShaderStages(
            [
                QRhiShaderStage(QRhiShaderStage.Type.Vertex, self._vertex_shader),
                QRhiShaderStage(QRhiShaderStage.Type.Fragment, self._fragment_shader),
            ]
        )
        pipeline.setVertexInputLayout(layout)
        pipeline.setShaderResourceBindings(self._shader_resources)
        pipeline.setRenderPassDescriptor(self._render_pass_descriptor)
        if not pipeline.create():
            log.warning("mesh qrhi: pipeline create() failed")
        self._pipeline = pipeline
        wire_pipeline = self._rhi.newGraphicsPipeline()
        wire_pipeline.setTopology(QRhiGraphicsPipeline.Topology.Lines)
        wire_pipeline.setCullMode(QRhiGraphicsPipeline.CullMode.None_)
        wire_pipeline.setDepthTest(False)
        wire_pipeline.setDepthWrite(False)
        wire_pipeline.setShaderStages(
            [
                QRhiShaderStage(QRhiShaderStage.Type.Vertex, self._vertex_shader),
                QRhiShaderStage(QRhiShaderStage.Type.Fragment, self._fragment_shader),
            ]
        )
        wire_pipeline.setVertexInputLayout(layout)
        wire_pipeline.setShaderResourceBindings(self._shader_resources)
        wire_pipeline.setRenderPassDescriptor(self._render_pass_descriptor)
        if not wire_pipeline.create():
            log.warning("mesh qrhi: wire pipeline create() failed")
        self._wire_pipeline = wire_pipeline

    def render(self, cb, render_target, camera: dict[str, object]) -> None:
        assert self._rhi is not None
        size = render_target.pixelSize()
        width = max(size.width(), 1)
        height = max(size.height(), 1)
        background = camera.get("background_color", (0.07, 0.08, 0.10))
        rub = self._rhi.nextResourceUpdateBatch()
        if self._vertex_buffer is not None and self._vertex_bytes and not self._uploaded:
            rub.uploadStaticBuffer(self._vertex_buffer, self._vertex_bytes)
            self._uploaded = True
        if (
            self._wire_vertex_buffer is not None
            and self._wire_vertex_bytes
            and not self._wire_uploaded
        ):
            rub.uploadStaticBuffer(self._wire_vertex_buffer, self._wire_vertex_bytes)
            self._wire_uploaded = True
        matrix = self._camera_matrix(width, height, camera)
        rub.updateDynamicBuffer(self._uniform_buffer, 0, _pack_mat4(matrix))
        cb.beginPass(
            render_target,
            QColor.fromRgbF(*background, 1.0),
            QRhiDepthStencilClearValue(1.0, 0),
            rub,
        )
        if (
            camera.get("filled_visible", True)
            and self._pipeline is not None
            and self._shader_resources is not None
            and self._vertex_buffer is not None
            and self._vertex_count > 0
        ):
            cb.setGraphicsPipeline(self._pipeline)
            cb.setViewport(QRhiViewport(0, 0, width, height))
            cb.setShaderResources(self._shader_resources)
            cb.setVertexInput(0, [(self._vertex_buffer, 0)])
            cb.draw(self._vertex_count)
        if (
            camera.get("wireframe_visible", True)
            and self._wire_pipeline is not None
            and self._shader_resources is not None
            and self._wire_vertex_buffer is not None
            and self._wire_vertex_count > 0
        ):
            cb.setGraphicsPipeline(self._wire_pipeline)
            cb.setViewport(QRhiViewport(0, 0, width, height))
            cb.setShaderResources(self._shader_resources)
            cb.setVertexInput(0, [(self._wire_vertex_buffer, 0)])
            cb.draw(self._wire_vertex_count)
        cb.endPass()

    def _camera_matrix(
        self,
        width: int,
        height: int,
        camera: dict[str, object],
    ) -> NDArray[np.float32]:
        eye = np.asarray(camera["position"], dtype=np.float64)
        target = np.asarray(camera["target"], dtype=np.float64)
        up = np.asarray(camera["up"], dtype=np.float64)
        distance = max(float(camera.get("distance", 6.0)), 0.1)
        near = max(distance / 1000.0, 0.001)
        far = max(distance * 100.0, 100.0)
        view = _look_at(eye, target, up)
        projection = _perspective(
            np.radians(45.0),
            float(width) / float(height),
            near,
            far,
            depth_zero_to_one=self._depth_zero_to_one,
        )
        projection[1, :] *= self._clip_y_sign
        return np.asarray(projection @ view, dtype=np.float32)


__all__ = ["QRhiMeshRenderer"]
