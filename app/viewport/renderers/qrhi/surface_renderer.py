from __future__ import annotations

import logging
import os
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QByteArray
from PySide6.QtGui import (
    QColor,
    QRhiBuffer,
    QRhiCommandBuffer,
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

from app.viewport.surface_builder import ViewportSurface, ViewportSurfaceScene

from .vulkanize import UBO_BINDING, uniform_block_members, vulkanize


log = logging.getLogger(__name__)

_SURFACE_STRIDE = 36
_LINE_MAX_VERTS = 65536
_LINE_STRIDE = 44
_LINE_HALF_PX = 3.0
_MAX_UPLOAD_BYTES_PER_FRAME = 64 * 1024 * 1024
_RESOURCE_RETIRE_FRAMES = 3
_MAX_RETIRED_SURFACE_BYTES = 256 * 1024 * 1024
_LINE_VERTEX_PATTERN = (
    (0.0, -1.0),
    (1.0, -1.0),
    (1.0, 1.0),
    (0.0, -1.0),
    (1.0, 1.0),
    (0.0, 1.0),
)

_FULLSCREEN_VERT = """\
#version 450
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_SURFACE_VERT = """\
#version 450
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;
layout(location = 0) out vec3 v_color;
layout(std140, binding = 0) uniform SurfaceUBO {
    mat4 mvp;
    float opacity;
};
vec3 safeNormal(vec3 value) {
    float len2 = dot(value, value);
    if (!(len2 > 1.0e-12) || !(len2 < 1.0e12)) {
        return vec3(0.0, 0.0, 1.0);
    }
    return value * inversesqrt(len2);
}
void main() {
    vec3 n = safeNormal(in_normal);
    vec3 light = normalize(vec3(0.35, 0.45, 0.82));
    float diffuse = abs(dot(n, light)) * 0.45 + 0.55;
    v_color = clamp(in_color * diffuse, 0.0, 1.0);
    gl_Position = mvp * vec4(in_position, 1.0);
}
"""

_SURFACE_FRAG = """\
#version 450
layout(location = 0) in vec3 v_color;
layout(location = 0) out vec4 frag_color;
layout(std140, binding = 0) uniform SurfaceUBO {
    mat4 mvp;
    float opacity;
};
void main() {
    frag_color = vec4(v_color, clamp(opacity, 0.0, 1.0));
}
"""

_GRID_FRAG = """\
#version 460
layout(location = 0) out vec4 frag_color;
uniform vec2 u_resolution;
uniform vec3 u_camera_position;
uniform vec3 u_camera_target;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform float u_focal_length;
uniform float u_max_ray_distance;
uniform vec3 u_background_color;
uniform int u_show_grid;
uniform float u_grid_spacing;
uniform int u_grid_plane;
uniform int u_fb_y_up;
vec3 gridA(vec3 ro, vec3 rd, float mt, vec3 col, float s) {
    if(u_show_grid==0) return col;
    vec3 n=u_grid_plane==1?vec3(0.,1.,0.):
             (u_grid_plane==2?vec3(1.,0.,0.):vec3(0.,0.,1.));
    float den=dot(rd,n);
    if(abs(den)<1e-6) return col;
    float tt=-dot(ro,n)/den;
    if(tt<=0. || tt>=mt) return col;
    vec3 p=ro+rd*tt;
    vec2 g=u_grid_plane==1?p.xz:(u_grid_plane==2?p.yz:p.xy);
    vec2 w=fwidth(g);
    vec2 a=abs(fract(g/u_grid_spacing+.5)-.5)*u_grid_spacing;
    float line=1.-smoothstep(0.,max(max(w.x,w.y),1e-5)*1.5,min(a.x,a.y));
    // Fade over a distance proportional to the cell size (1 m baseline), so
    // coarse grids (km work) stay visible across their own cells.
    float ft=tt/max(u_grid_spacing,1.);
    float fade=clamp(1./(1.+ft*ft*.002),0.,1.);
    return mix(col,vec3(.62,.75,.92),line*s*fade);
}
// One world axis (line through the origin along `axis`) drawn with a
// screen-space-constant width via ray-vs-line distance, like a normal CAD.
vec3 axisLine(vec3 ro, vec3 rd, float mt, vec3 col, vec3 axis, vec3 acol) {
    if(u_show_grid==0) return col;
    float b=dot(rd,axis);
    float den=1.-b*b;
    if(abs(den)<1e-6) return col;          // ray parallel to the axis
    float d=dot(rd,ro);
    float e=dot(axis,ro);
    float t=(b*e-d)/den;                    // closest param along the camera ray
    if(t<=0.||t>=mt) return col;
    float s=(e-b*d)/den;                    // closest param along the axis
    vec3 pr=ro+rd*t;
    vec3 pa=axis*s;
    float dist=length(pr-pa);
    float wpp=t*2./(u_focal_length*max(u_resolution.y,1.));  // world units / pixel
    float px=dist/max(wpp,1e-9);
    float linev=1.-smoothstep(.9,2.2,px);
    float ft=t/max(u_grid_spacing,1.);
    float fade=clamp(1./(1.+ft*ft*.0008),0.,1.);
    return mix(col,acol,linev*fade);
}
void main() {
    vec2 px = gl_FragCoord.xy;
    vec2 uv = (px - 0.5*u_resolution)/max(u_resolution.y, 1.0);
    if (u_fb_y_up == 0) uv.y = -uv.y;
    vec3 fwd = normalize(u_camera_target - u_camera_position);
    vec3 rd = normalize(2.0*uv.x*normalize(u_camera_right)
                      + 2.0*uv.y*normalize(u_camera_up) + u_focal_length*fwd);
    vec3 col = gridA(u_camera_position, rd, u_max_ray_distance,
                     u_background_color, 0.6);
    col = axisLine(u_camera_position, rd, u_max_ray_distance, col,
                   vec3(1.,0.,0.), vec3(1.00,0.34,0.25));   // X red
    col = axisLine(u_camera_position, rd, u_max_ray_distance, col,
                   vec3(0.,1.,0.), vec3(0.33,0.92,0.41));   // Y green
    col = axisLine(u_camera_position, rd, u_max_ray_distance, col,
                   vec3(0.,0.,1.), vec3(0.36,0.57,1.00));   // Z blue
    frag_color = vec4(col, 1.0);
}
"""

_LINE_UBO_MEMBERS = (
    ("vec3", "cam_pos", None),
    ("vec3", "cam_right", None),
    ("vec3", "cam_up", None),
    ("vec3", "cam_target", None),
    ("float", "focal", None),
    ("float", "aspect", None),
    ("vec2", "res", None),
    ("float", "half_px", None),
    ("float", "clip_y_sign", None),
)

_LINE_VERT = """\
#version 450
layout(location = 0) in vec3 in_a;
layout(location = 1) in vec3 in_b;
layout(location = 2) in vec3 in_col;
layout(location = 3) in vec2 in_param;
layout(location = 0) out vec3 v_col;
layout(std140, binding = 0) uniform LineUBO {
    vec3 cam_pos;
    vec3 cam_right;
    vec3 cam_up;
    vec3 cam_target;
    float focal;
    float aspect;
    vec2 res;
    float half_px;
    float clip_y_sign;
};
vec4 project(vec3 P) {
    vec3 fwd = normalize(cam_target - cam_pos);
    vec3 r = normalize(cam_right);
    vec3 u = normalize(cam_up);
    vec3 v = P - cam_pos;
    return vec4(focal * dot(v, r) * aspect, -focal * dot(v, u),
                0.0, max(dot(v, fwd), 1e-4));
}
void main() {
    vec4 ca = project(in_a);
    vec4 cb = project(in_b);
    vec2 sa = (ca.xy / ca.w * 0.5 + 0.5) * res;
    vec2 sb = (cb.xy / cb.w * 0.5 + 0.5) * res;
    vec2 dir = sb - sa;
    float len = length(dir);
    dir = len > 1e-5 ? dir / len : vec2(1.0, 0.0);
    vec2 nrm = vec2(-dir.y, dir.x);
    vec4 cthis = in_param.x < 0.5 ? ca : cb;
    vec2 sthis = in_param.x < 0.5 ? sa : sb;
    vec2 soff = sthis + nrm * in_param.y * half_px;
    vec2 ndc = soff / res * 2.0 - 1.0;
    gl_Position = vec4(ndc.x * cthis.w, ndc.y * cthis.w * clip_y_sign,
                       cthis.z, cthis.w);
    v_col = in_col;
}
"""

_LINE_FRAG = """\
#version 450
layout(location = 0) in vec3 v_col;
layout(location = 0) out vec4 frag_color;
void main() { frag_color = vec4(v_col, 1.0); }
"""

_ALIGN = {"float": 4, "int": 4, "uint": 4, "vec2": 8, "vec3": 16, "vec4": 16}
_SIZE = {"float": 4, "int": 4, "uint": 4, "vec2": 8, "vec3": 12, "vec4": 16}


def _bake(ext: str, glsl: str) -> QShader:
    temp_dir = tempfile.mkdtemp(prefix="casocad_viewport_surface_qrhi_")
    source = os.path.join(temp_dir, f"shader.{ext}")
    output = source + ".qsb"
    with open(source, "w", encoding="utf-8") as stream:
        stream.write(glsl)
    qsb = os.path.join(os.path.dirname(sys.executable), "pyside6-qsb")
    subprocess.run([qsb, "--glsl", "430", "-o", output, source], check=True)
    with open(output, "rb") as stream:
        return QShader.fromSerialized(QByteArray(stream.read()))


def _std140(members: tuple[tuple[str, str, object], ...], values: dict[str, object]) -> bytes:
    offsets: dict[str, tuple[int, str]] = {}
    offset = 0
    for glsl_type, name, _array in members:
        alignment = _ALIGN[glsl_type]
        offset = (offset + alignment - 1) // alignment * alignment
        offsets[name] = (offset, glsl_type)
        offset += _SIZE[glsl_type]
    data = bytearray((offset + 15) // 16 * 16)
    for name, (member_offset, glsl_type) in offsets.items():
        value = values[name]
        if glsl_type == "uint":
            struct.pack_into("<I", data, member_offset, int(value))
        elif glsl_type == "int":
            struct.pack_into("<i", data, member_offset, int(value))
        elif glsl_type == "float":
            struct.pack_into("<f", data, member_offset, float(value))
        elif glsl_type == "vec2":
            struct.pack_into("<2f", data, member_offset, *map(float, value))
        elif glsl_type == "vec3":
            struct.pack_into("<3f", data, member_offset, *map(float, value))
        elif glsl_type == "vec4":
            struct.pack_into("<4f", data, member_offset, *map(float, value))
    return bytes(data)


def _destroy_resource(resource: object | None) -> None:
    if resource is None:
        return
    destroy = getattr(resource, "destroy", None)
    if destroy is None:
        return
    try:
        destroy()
    except RuntimeError:
        pass


def _normalize(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return vector
    return vector / norm


def _pack_surface_ubo(matrix: NDArray[np.float32], opacity: float) -> bytes:
    return struct.pack("<16f4f", *matrix.T.reshape(-1), float(opacity), 0.0, 0.0, 0.0)


class QRhiSurfaceRenderer:
    def __init__(self) -> None:
        self._rhi = None
        self._render_pass_descriptor = None
        self._surface_vertex_shader = None
        self._surface_fragment_shader = None
        self._fullscreen_vertex_shader = None
        self._grid_fragment_shader = None
        self._line_vertex_shader = None
        self._line_fragment_shader = None
        self._surface_ubo = None
        self._surface_srb = None
        self._surface_pipeline = None
        self._transparent_surface_pipeline = None
        self._wire_pipeline = None
        self._grid_ubo = None
        self._grid_ubo_members = None
        self._grid_srb = None
        self._grid_pipeline = None
        self._line_ubo = None
        self._line_vbuf = None
        self._line_srb = None
        self._line_pipeline = None
        self._chunks: list[_SurfaceChunk] = []
        self._scene: ViewportSurfaceScene | None = None
        self._clip_y_sign = 1.0
        self._fb_y_up = 1
        self._depth_zero_to_one = True
        self._update_cb = None
        self._last_interaction_t = 0.0
        self._pending_uploads = False
        self._retired_chunks: list[tuple[_SurfaceChunk, int]] = []

    def initialize(self, rhi, render_target) -> None:
        self._rhi = rhi
        self._render_pass_descriptor = render_target.renderPassDescriptor()
        self._clip_y_sign = -1.0 if rhi.isYUpInNDC() else 1.0
        self._fb_y_up = 1 if rhi.isYUpInFramebuffer() else 0
        if hasattr(rhi, "isClipDepthZeroToOne"):
            self._depth_zero_to_one = bool(rhi.isClipDepthZeroToOne())
        backend = rhi.backendName() if hasattr(rhi, "backendName") else "?"
        log.info(
            "viewport surface qrhi: initialize backend=%s clip_y_sign=%+.0f "
            "fb_y_up=%d depth_zero_to_one=%s",
            backend,
            self._clip_y_sign,
            self._fb_y_up,
            self._depth_zero_to_one,
        )
        self._surface_vertex_shader = _bake("vert", _SURFACE_VERT)
        self._surface_fragment_shader = _bake("frag", _SURFACE_FRAG)
        self._fullscreen_vertex_shader = _bake("vert", _FULLSCREEN_VERT)
        self._grid_fragment_shader = _bake("frag", vulkanize(_GRID_FRAG))
        self._line_vertex_shader = _bake("vert", _LINE_VERT)
        self._line_fragment_shader = _bake("frag", _LINE_FRAG)
        self._surface_ubo = rhi.newBuffer(
            QRhiBuffer.Type.Dynamic,
            QRhiBuffer.UsageFlag.UniformBuffer,
            80,
        )
        self._surface_ubo.create()
        self._grid_ubo_members = uniform_block_members(_GRID_FRAG)
        grid_ubo_bytes = _std140(self._grid_ubo_members, self._grid_zero_uniforms())
        self._grid_ubo = rhi.newBuffer(
            QRhiBuffer.Type.Dynamic,
            QRhiBuffer.UsageFlag.UniformBuffer,
            len(grid_ubo_bytes),
        )
        self._grid_ubo.create()
        line_ubo_bytes = _std140(_LINE_UBO_MEMBERS, self._zero_line_ubo())
        self._line_ubo = rhi.newBuffer(
            QRhiBuffer.Type.Dynamic,
            QRhiBuffer.UsageFlag.UniformBuffer,
            len(line_ubo_bytes),
        )
        self._line_ubo.create()
        self._line_vbuf = rhi.newBuffer(
            QRhiBuffer.Type.Dynamic,
            QRhiBuffer.UsageFlag.VertexBuffer,
            _LINE_MAX_VERTS * _LINE_STRIDE,
        )
        self._line_vbuf.create()
        self._build_surface_pipelines()
        self._build_grid_pipeline()
        self._build_line_pipeline()
        for chunk in self._chunks:
            self._build_chunk_buffers(chunk)

    def set_surface_scene(self, scene: ViewportSurfaceScene | None) -> None:
        self._retire_current_chunks()
        self._scene = scene
        if scene is None:
            return
        for surface in scene.surfaces:
            chunk = _chunk_from_surface(surface)
            if chunk is None:
                continue
            self._chunks.append(chunk)
            if self._rhi is not None:
                self._build_chunk_buffers(chunk)
        self._pending_uploads = any(chunk.has_pending_uploads for chunk in self._chunks)
        log.info(
            "viewport surface qrhi: scene revision=%d chunks=%d vertices=%d "
            "triangles=%d build_ms=%.1f",
            scene.revision,
            len(self._chunks),
            scene.vertex_count,
            scene.triangle_count,
            scene.build_ms,
        )

    def clear(self) -> None:
        for chunk in self._chunks:
            self._destroy_chunk_resources(chunk)
        self._chunks.clear()
        for chunk, _frames in self._retired_chunks:
            self._destroy_chunk_resources(chunk)
        self._retired_chunks.clear()
        self._pending_uploads = False

    def _retire_current_chunks(self) -> None:
        if not self._chunks:
            self._pending_uploads = False
            return
        if self._rhi is None:
            for chunk in self._chunks:
                self._destroy_chunk_resources(chunk)
        else:
            self._retired_chunks.extend(
                (chunk, _RESOURCE_RETIRE_FRAMES) for chunk in self._chunks
            )
            self._trim_retired_chunks()
        self._chunks.clear()
        self._pending_uploads = False

    def _trim_retired_chunks(self) -> None:
        retired_bytes = sum(
            chunk.allocated_byte_count for chunk, _frames in self._retired_chunks
        )
        while (
            self._retired_chunks
            and retired_bytes > _MAX_RETIRED_SURFACE_BYTES
        ):
            chunk, _frames = self._retired_chunks.pop(0)
            retired_bytes -= chunk.allocated_byte_count
            self._destroy_chunk_resources(chunk)

    def _collect_retired_chunks(self) -> None:
        if not self._retired_chunks:
            return
        survivors: list[tuple[_SurfaceChunk, int]] = []
        for chunk, frames in self._retired_chunks:
            frames -= 1
            if frames <= 0:
                self._destroy_chunk_resources(chunk)
            else:
                survivors.append((chunk, frames))
        self._retired_chunks = survivors

    def _destroy_chunk_resources(self, chunk: "_SurfaceChunk") -> None:
        _destroy_resource(chunk.vertex_buffer)
        _destroy_resource(chunk.index_buffer)
        _destroy_resource(chunk.wire_index_buffer)
        chunk.vertex_buffer = None
        chunk.index_buffer = None
        chunk.wire_index_buffer = None

    def shutdown(self) -> None:
        self.clear()
        for name in (
            "_surface_pipeline",
            "_transparent_surface_pipeline",
            "_wire_pipeline",
            "_surface_srb",
            "_surface_ubo",
            "_grid_pipeline",
            "_grid_srb",
            "_grid_ubo",
            "_line_pipeline",
            "_line_srb",
            "_line_ubo",
            "_line_vbuf",
        ):
            _destroy_resource(getattr(self, name))
            setattr(self, name, None)
        self._rhi = None
        self._render_pass_descriptor = None

    def set_update_callback(self, cb) -> None:
        self._update_cb = cb

    def set_telemetry_callback(self, cb) -> None:
        del cb

    def mark_interaction(self) -> None:
        self._last_interaction_t = time.perf_counter()

    def should_prewarm_tool_pipeline(self) -> bool:
        return False

    def prewarm_for_tool(self, *args, **kwargs) -> None:
        del args, kwargs

    def save_pipeline_cache(self) -> None:
        return None

    def _build_chunk_buffers(self, chunk: "_SurfaceChunk") -> None:
        assert self._rhi is not None
        if chunk.vertex_buffer is None:
            chunk.vertex_buffer = self._rhi.newBuffer(
                QRhiBuffer.Type.Static,
                QRhiBuffer.UsageFlag.VertexBuffer,
                max(len(chunk.vertex_bytes), _SURFACE_STRIDE),
            )
            chunk.vertex_buffer.create()
        if chunk.index_count > 0 and chunk.index_buffer is None:
            chunk.index_buffer = self._rhi.newBuffer(
                QRhiBuffer.Type.Static,
                QRhiBuffer.UsageFlag.IndexBuffer,
                max(len(chunk.index_bytes), 4),
            )
            chunk.index_buffer.create()
        if chunk.wire_index_count > 0 and chunk.wire_index_buffer is None:
            chunk.wire_index_buffer = self._rhi.newBuffer(
                QRhiBuffer.Type.Static,
                QRhiBuffer.UsageFlag.IndexBuffer,
                max(len(chunk.wire_index_bytes), 4),
            )
            chunk.wire_index_buffer.create()

    def _build_surface_pipelines(self) -> None:
        assert self._rhi is not None
        surface_stages = (
            QRhiShaderResourceBinding.StageFlag.VertexStage
            | QRhiShaderResourceBinding.StageFlag.FragmentStage
        )
        self._surface_srb = self._rhi.newShaderResourceBindings()
        self._surface_srb.setBindings(
            [
                QRhiShaderResourceBinding.uniformBuffer(
                    0,
                    surface_stages,
                    self._surface_ubo,
                )
            ]
        )
        self._surface_srb.create()
        float3 = QRhiVertexInputAttribute.Format.Float3
        layout = QRhiVertexInputLayout()
        layout.setBindings([QRhiVertexInputBinding(_SURFACE_STRIDE)])
        layout.setAttributes(
            [
                QRhiVertexInputAttribute(0, 0, float3, 0),
                QRhiVertexInputAttribute(0, 1, float3, 12),
                QRhiVertexInputAttribute(0, 2, float3, 24),
            ]
        )
        self._surface_pipeline = self._new_surface_pipeline(
            QRhiGraphicsPipeline.Topology.Triangles,
            depth_test=True,
            depth_write=True,
            blend=False,
            layout=layout,
        )
        self._transparent_surface_pipeline = self._new_surface_pipeline(
            QRhiGraphicsPipeline.Topology.Triangles,
            depth_test=True,
            depth_write=False,
            blend=True,
            layout=layout,
        )
        self._wire_pipeline = self._new_surface_pipeline(
            QRhiGraphicsPipeline.Topology.Lines,
            depth_test=True,
            depth_write=False,
            blend=False,
            layout=layout,
        )

    def _new_surface_pipeline(
        self,
        topology: QRhiGraphicsPipeline.Topology,
        *,
        depth_test: bool,
        depth_write: bool,
        blend: bool,
        layout: QRhiVertexInputLayout,
    ):
        pipeline = self._rhi.newGraphicsPipeline()
        pipeline.setTopology(topology)
        pipeline.setCullMode(QRhiGraphicsPipeline.CullMode.None_)
        pipeline.setDepthTest(depth_test)
        pipeline.setDepthWrite(depth_write)
        if blend:
            target = QRhiGraphicsPipeline.TargetBlend()
            target.enable = True
            target.srcColor = QRhiGraphicsPipeline.BlendFactor.SrcAlpha
            target.dstColor = QRhiGraphicsPipeline.BlendFactor.OneMinusSrcAlpha
            target.srcAlpha = QRhiGraphicsPipeline.BlendFactor.One
            target.dstAlpha = QRhiGraphicsPipeline.BlendFactor.OneMinusSrcAlpha
            pipeline.setTargetBlends([target])
        pipeline.setShaderStages(
            [
                QRhiShaderStage(
                    QRhiShaderStage.Type.Vertex,
                    self._surface_vertex_shader,
                ),
                QRhiShaderStage(
                    QRhiShaderStage.Type.Fragment,
                    self._surface_fragment_shader,
                ),
            ]
        )
        pipeline.setVertexInputLayout(layout)
        pipeline.setShaderResourceBindings(self._surface_srb)
        pipeline.setRenderPassDescriptor(self._render_pass_descriptor)
        if not pipeline.create():
            log.warning("viewport surface qrhi: pipeline create() failed")
        return pipeline

    def _build_grid_pipeline(self) -> None:
        fragment_stage = QRhiShaderResourceBinding.StageFlag.FragmentStage
        self._grid_srb = self._rhi.newShaderResourceBindings()
        self._grid_srb.setBindings(
            [
                QRhiShaderResourceBinding.uniformBuffer(
                    UBO_BINDING,
                    fragment_stage,
                    self._grid_ubo,
                )
            ]
        )
        self._grid_srb.create()
        pipeline = self._rhi.newGraphicsPipeline()
        pipeline.setShaderStages(
            [
                QRhiShaderStage(
                    QRhiShaderStage.Type.Vertex,
                    self._fullscreen_vertex_shader,
                ),
                QRhiShaderStage(
                    QRhiShaderStage.Type.Fragment,
                    self._grid_fragment_shader,
                ),
            ]
        )
        pipeline.setVertexInputLayout(QRhiVertexInputLayout())
        pipeline.setShaderResourceBindings(self._grid_srb)
        pipeline.setRenderPassDescriptor(self._render_pass_descriptor)
        if not pipeline.create():
            log.warning("viewport surface qrhi: grid pipeline create() failed")
        self._grid_pipeline = pipeline

    def _build_line_pipeline(self) -> None:
        vertex_stage = QRhiShaderResourceBinding.StageFlag.VertexStage
        self._line_srb = self._rhi.newShaderResourceBindings()
        self._line_srb.setBindings(
            [QRhiShaderResourceBinding.uniformBuffer(0, vertex_stage, self._line_ubo)]
        )
        self._line_srb.create()
        float3 = QRhiVertexInputAttribute.Format.Float3
        float2 = QRhiVertexInputAttribute.Format.Float2
        layout = QRhiVertexInputLayout()
        layout.setBindings([QRhiVertexInputBinding(_LINE_STRIDE)])
        layout.setAttributes(
            [
                QRhiVertexInputAttribute(0, 0, float3, 0),
                QRhiVertexInputAttribute(0, 1, float3, 12),
                QRhiVertexInputAttribute(0, 2, float3, 24),
                QRhiVertexInputAttribute(0, 3, float2, 36),
            ]
        )
        pipeline = self._rhi.newGraphicsPipeline()
        pipeline.setTopology(QRhiGraphicsPipeline.Topology.Triangles)
        pipeline.setCullMode(QRhiGraphicsPipeline.CullMode.None_)
        pipeline.setDepthTest(False)
        pipeline.setDepthWrite(False)
        pipeline.setShaderStages(
            [
                QRhiShaderStage(QRhiShaderStage.Type.Vertex, self._line_vertex_shader),
                QRhiShaderStage(QRhiShaderStage.Type.Fragment, self._line_fragment_shader),
            ]
        )
        pipeline.setVertexInputLayout(layout)
        pipeline.setShaderResourceBindings(self._line_srb)
        pipeline.setRenderPassDescriptor(self._render_pass_descriptor)
        if not pipeline.create():
            log.warning("viewport surface qrhi: line pipeline create() failed")
        self._line_pipeline = pipeline

    def render(self, cb, render_target, camera: dict[str, object], overlay=None) -> None:
        assert self._rhi is not None
        size = render_target.pixelSize()
        width = max(size.width(), 1)
        height = max(size.height(), 1)
        background = _camera_tuple(camera, "u_background_color", (0.07, 0.08, 0.10))
        if int(camera.get("u_interacting", 0)) != 0:
            self._last_interaction_t = time.perf_counter()

        rub = self._rhi.nextResourceUpdateBatch()
        upload_budget = _MAX_UPLOAD_BYTES_PER_FRAME
        for chunk in self._chunks:
            upload_budget = self._upload_pending_chunk(rub, chunk, upload_budget)
        self._pending_uploads = any(chunk.has_pending_uploads for chunk in self._chunks)
        rub.updateDynamicBuffer(
            self._surface_ubo,
            0,
            _pack_surface_ubo(
                self._camera_matrix(width, height, camera),
                float(camera.get("u_surface_opacity", 1.0)),
            ),
        )
        if int(camera.get("u_show_grid", 1)) == 1:
            rub.updateDynamicBuffer(
                self._grid_ubo,
                0,
                _std140(
                    self._grid_ubo_members,
                    self._grid_uniforms(width, height, camera),
                ),
            )
        dynamic_line_count = self._upload_dynamic_lines(
            rub,
            width,
            height,
            camera,
            overlay,
        )

        cb.beginPass(
            render_target,
            QColor.fromRgbF(*background, 1.0),
            QRhiDepthStencilClearValue(1.0, 0),
            rub,
        )
        if int(camera.get("u_show_grid", 1)) == 1 and self._grid_pipeline is not None:
            cb.setGraphicsPipeline(self._grid_pipeline)
            cb.setViewport(QRhiViewport(0, 0, width, height))
            cb.setShaderResources(self._grid_srb)
            cb.draw(3)
        self._draw_surface_chunks(cb, width, height, camera)
        if dynamic_line_count > 0:
            cb.setGraphicsPipeline(self._line_pipeline)
            cb.setViewport(QRhiViewport(0, 0, width, height))
            cb.setShaderResources(self._line_srb)
            cb.setVertexInput(0, [(self._line_vbuf, 0)])
            cb.draw(dynamic_line_count)
        cb.endPass()
        self._collect_retired_chunks()
        if self._pending_uploads and self._update_cb is not None:
            self._update_cb()

    def _draw_surface_chunks(
        self,
        cb,
        width: int,
        height: int,
        camera: dict[str, object],
    ) -> None:
        if self._surface_srb is None:
            return
        draw_filled = bool(camera.get("filled_visible", True))
        draw_solid_wire = bool(camera.get("wireframe_visible", False))
        opacity = max(0.0, min(1.0, float(camera.get("u_surface_opacity", 1.0))))
        surface_pipeline = (
            self._surface_pipeline
            if opacity >= 0.999
            else self._transparent_surface_pipeline
        )
        if draw_filled and surface_pipeline is not None:
            cb.setGraphicsPipeline(surface_pipeline)
            cb.setViewport(QRhiViewport(0, 0, width, height))
            cb.setShaderResources(self._surface_srb)
            for chunk in self._chunks:
                if chunk.index_count <= 0 or chunk.vertex_buffer is None:
                    continue
                if chunk.index_buffer is None or chunk.has_pending_main_uploads:
                    continue
                cb.setVertexInput(
                    0,
                    [(chunk.vertex_buffer, 0)],
                    chunk.index_buffer,
                    0,
                    QRhiCommandBuffer.IndexFormat.IndexUInt32,
                )
                cb.drawIndexed(chunk.index_count)
        if self._wire_pipeline is not None:
            cb.setGraphicsPipeline(self._wire_pipeline)
            cb.setViewport(QRhiViewport(0, 0, width, height))
            cb.setShaderResources(self._surface_srb)
            for chunk in self._chunks:
                if chunk.index_count > 0 and not draw_solid_wire:
                    continue
                if chunk.index_count == 0:
                    continue
                if chunk.wire_index_count <= 0 or chunk.vertex_buffer is None:
                    continue
                if chunk.wire_index_buffer is None or chunk.has_pending_wire_uploads:
                    continue
                cb.setVertexInput(
                    0,
                    [(chunk.vertex_buffer, 0)],
                    chunk.wire_index_buffer,
                    0,
                    QRhiCommandBuffer.IndexFormat.IndexUInt32,
                )
                cb.drawIndexed(chunk.wire_index_count)

    def _upload_pending_chunk(self, rub, chunk: "_SurfaceChunk", budget: int) -> int:
        budget = self._upload_buffer(
            rub,
            chunk.vertex_buffer,
            "vertex_bytes",
            "vertex_uploaded",
            chunk,
            budget,
        )
        budget = self._upload_buffer(
            rub,
            chunk.index_buffer,
            "index_bytes",
            "index_uploaded",
            chunk,
            budget,
        )
        return self._upload_buffer(
            rub,
            chunk.wire_index_buffer,
            "wire_index_bytes",
            "wire_index_uploaded",
            chunk,
            budget,
        )

    def _upload_buffer(
        self,
        rub,
        buffer,
        bytes_attr: str,
        uploaded_attr: str,
        chunk: "_SurfaceChunk",
        budget: int,
    ) -> int:
        data = getattr(chunk, bytes_attr)
        uploaded = bool(getattr(chunk, uploaded_attr))
        if buffer is None or uploaded or not data:
            return budget
        if budget <= 0:
            return budget
        rub.uploadStaticBuffer(buffer, data)
        setattr(chunk, bytes_attr, b"")
        setattr(chunk, uploaded_attr, True)
        return budget - len(data)

    def _upload_dynamic_lines(
        self,
        rub,
        width: int,
        height: int,
        camera: dict[str, object],
        overlay,
    ) -> int:
        if self._line_pipeline is None:
            return 0
        vertex_bytes, vertex_count = _dynamic_line_payload(self._chunks, overlay)
        if vertex_count <= 0:
            return 0
        rub.updateDynamicBuffer(self._line_vbuf, 0, vertex_bytes)
        rub.updateDynamicBuffer(
            self._line_ubo,
            0,
            _std140(
                _LINE_UBO_MEMBERS,
                {
                    "cam_pos": _camera_tuple(camera, "u_camera_position", (0, 0, 1)),
                    "cam_right": _camera_tuple(camera, "u_camera_right", (1, 0, 0)),
                    "cam_up": _camera_tuple(camera, "u_camera_up", (0, 1, 0)),
                    "cam_target": _camera_tuple(camera, "u_camera_target", (0, 0, 0)),
                    "focal": float(camera.get("u_focal_length", 1.5)),
                    "aspect": float(height) / float(width),
                    "res": (float(width), float(height)),
                    "half_px": _LINE_HALF_PX,
                    "clip_y_sign": self._clip_y_sign,
                },
            ),
        )
        return int(vertex_count)

    def _camera_matrix(
        self,
        width: int,
        height: int,
        camera: dict[str, object],
    ) -> NDArray[np.float32]:
        eye = np.asarray(
            _camera_tuple(camera, "u_camera_position", (0, 0, 6)),
            dtype=np.float64,
        )
        target = np.asarray(
            _camera_tuple(camera, "u_camera_target", (0, 0, 0)),
            dtype=np.float64,
        )
        right = _normalize(
            np.asarray(
                _camera_tuple(camera, "u_camera_right", (1, 0, 0)),
                dtype=np.float64,
            )
        )
        up = _normalize(
            np.asarray(
                _camera_tuple(camera, "u_camera_up", (0, 1, 0)),
                dtype=np.float64,
            )
        )
        forward = _normalize(target - eye)
        distance = max(float(np.linalg.norm(eye - target)), 0.1)
        near = max(distance / 1000.0, 0.001)
        far = max(distance * 100.0, 100.0)
        focal = float(camera.get("u_focal_length", 1.5))
        aspect_scale = float(height) / max(float(width), 1.0)
        if self._depth_zero_to_one:
            depth_scale = far / (far - near)
            depth_bias = -(far * near) / (far - near)
        else:
            depth_scale = (far + near) / (far - near)
            depth_bias = -(2.0 * far * near) / (far - near)

        matrix = np.zeros((4, 4), dtype=np.float64)
        matrix[0, :3] = focal * aspect_scale * right
        matrix[1, :3] = -focal * self._clip_y_sign * up
        matrix[2, :3] = depth_scale * forward
        matrix[3, :3] = forward
        matrix[0, 3] = -float(np.dot(matrix[0, :3], eye))
        matrix[1, 3] = -float(np.dot(matrix[1, :3], eye))
        matrix[2, 3] = depth_bias - depth_scale * float(np.dot(forward, eye))
        matrix[3, 3] = -float(np.dot(forward, eye))
        return np.asarray(matrix, dtype=np.float32)

    def _grid_uniforms(
        self,
        width: int,
        height: int,
        camera: dict[str, object],
    ) -> dict[str, object]:
        values = self._grid_zero_uniforms()
        for key in (
            "u_camera_position",
            "u_camera_target",
            "u_camera_right",
            "u_camera_up",
            "u_focal_length",
            "u_background_color",
            "u_show_grid",
            "u_grid_spacing",
            "u_max_ray_distance",
            "u_grid_plane",
        ):
            if key in camera:
                values[key] = camera[key]
        values["u_resolution"] = (float(width), float(height))
        values["u_fb_y_up"] = self._fb_y_up
        return values

    def _grid_zero_uniforms(self) -> dict[str, object]:
        return {
            "u_resolution": (1.0, 1.0),
            "u_camera_position": (0.0, 0.0, 1.0),
            "u_camera_target": (0.0, 0.0, 0.0),
            "u_camera_right": (1.0, 0.0, 0.0),
            "u_camera_up": (0.0, 1.0, 0.0),
            "u_focal_length": 1.5,
            "u_max_ray_distance": 100.0,
            "u_background_color": (0.07, 0.08, 0.10),
            "u_show_grid": 1,
            "u_grid_spacing": 1.0,
            "u_grid_plane": 0,
            "u_fb_y_up": 1,
        }

    def _zero_line_ubo(self) -> dict[str, object]:
        return {
            "cam_pos": (0, 0, 1),
            "cam_right": (1, 0, 0),
            "cam_up": (0, 1, 0),
            "cam_target": (0, 0, 0),
            "focal": 1.5,
            "aspect": 1.0,
            "res": (1.0, 1.0),
            "half_px": _LINE_HALF_PX,
            "clip_y_sign": 1.0,
        }


def _camera_tuple(
    camera: dict[str, object],
    key: str,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    value = camera.get(key, default)
    return tuple(float(component) for component in value)


def _chunk_from_surface(surface: ViewportSurface) -> "_SurfaceChunk | None":
    if not surface.has_geometry:
        return None
    vertices = np.asarray(surface.vertices, dtype=np.float32)
    normals = np.asarray(surface.normals, dtype=np.float32)
    if normals.shape != vertices.shape:
        normals = np.zeros_like(vertices)
        normals[:, 2] = 1.0
    normals = _safe_normal_array(normals)
    colors = np.broadcast_to(
        np.asarray(surface.color, dtype=np.float32),
        vertices.shape,
    )
    colors = np.clip(np.nan_to_num(colors, nan=0.75, posinf=1.0, neginf=0.0), 0.0, 1.0)
    interleaved = np.column_stack((vertices, normals, colors)).astype(
        np.float32,
        copy=False,
    )
    index_array = np.asarray(surface.indices, dtype=np.uint32)
    wire_array = np.asarray(surface.wire_indices, dtype=np.uint32)
    thick_line_bytes, thick_line_vertex_count = (
        _thick_line_bytes(vertices, wire_array, surface.color)
        if index_array.size == 0
        else (b"", 0)
    )
    return _SurfaceChunk(
        vertex_bytes=interleaved.tobytes(),
        vertex_count=int(vertices.shape[0]),
        index_bytes=index_array.tobytes(),
        index_count=int(index_array.size),
        wire_index_bytes=wire_array.tobytes(),
        wire_index_count=int(wire_array.size),
        thick_line_bytes=thick_line_bytes,
        thick_line_vertex_count=thick_line_vertex_count,
        object_id=surface.key.object_id,
    )


def _safe_normal_array(normals: NDArray[np.float32]) -> NDArray[np.float32]:
    out = np.nan_to_num(
        np.asarray(normals, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    lengths = np.linalg.norm(out, axis=1)
    valid = np.isfinite(lengths) & (lengths > 1.0e-12)
    if np.any(valid):
        out[valid] = out[valid] / lengths[valid, None]
    out[~valid] = (0.0, 0.0, 1.0)
    return out


def _thick_line_bytes(
    vertices: NDArray[np.float32],
    wire_indices: NDArray[np.uint32],
    color: tuple[float, float, float],
) -> tuple[bytes, int]:
    if wire_indices.size < 2 or vertices.size == 0:
        return b"", 0
    data = bytearray()
    color_tuple = tuple(float(component) for component in color)
    pack = struct.pack
    for raw_a, raw_b in wire_indices.reshape(-1, 2):
        a = int(raw_a)
        b = int(raw_b)
        if a == b or a < 0 or b < 0 or a >= len(vertices) or b >= len(vertices):
            continue
        ax, ay, az = (float(value) for value in vertices[a])
        bx, by, bz = (float(value) for value in vertices[b])
        for endpoint, side in _LINE_VERTEX_PATTERN:
            data.extend(
                pack(
                    "<11f",
                    ax,
                    ay,
                    az,
                    bx,
                    by,
                    bz,
                    color_tuple[0],
                    color_tuple[1],
                    color_tuple[2],
                    endpoint,
                    side,
                )
            )
    return bytes(data), len(data) // _LINE_STRIDE


def _dynamic_line_payload(
    chunks: list["_SurfaceChunk"],
    overlay,
) -> tuple[bytes, int]:
    parts: list[bytes] = []
    total_count = 0

    def append_limited(
        vertex_bytes: bytes,
        vertex_count: int,
        *,
        align_to_segment: bool,
    ) -> None:
        nonlocal total_count
        remaining = _LINE_MAX_VERTS - total_count
        if remaining <= 0 or vertex_count <= 0 or not vertex_bytes:
            return
        take = min(int(vertex_count), remaining)
        if align_to_segment:
            take -= take % len(_LINE_VERTEX_PATTERN)
        if take <= 0:
            return
        parts.append(vertex_bytes[: take * _LINE_STRIDE])
        total_count += take

    if overlay is not None and overlay[1] > 0:
        append_limited(overlay[0], int(overlay[1]), align_to_segment=False)
    for chunk in chunks:
        append_limited(
            chunk.thick_line_bytes,
            chunk.thick_line_vertex_count,
            align_to_segment=True,
        )
    return b"".join(parts), total_count


@dataclass
class _SurfaceChunk:
    vertex_bytes: bytes
    vertex_count: int
    index_bytes: bytes
    index_count: int
    wire_index_bytes: bytes
    wire_index_count: int
    thick_line_bytes: bytes
    thick_line_vertex_count: int
    object_id: int
    vertex_buffer: object | None = None
    index_buffer: object | None = None
    wire_index_buffer: object | None = None
    vertex_uploaded: bool = False
    index_uploaded: bool = False
    wire_index_uploaded: bool = False

    @property
    def has_pending_main_uploads(self) -> bool:
        return (
            not self.vertex_uploaded
            or (self.index_count > 0 and not self.index_uploaded)
        )

    @property
    def has_pending_wire_uploads(self) -> bool:
        return (
            not self.vertex_uploaded
            or (self.wire_index_count > 0 and not self.wire_index_uploaded)
        )

    @property
    def has_pending_uploads(self) -> bool:
        return (
            not self.vertex_uploaded
            or (self.index_count > 0 and not self.index_uploaded)
            or (self.wire_index_count > 0 and not self.wire_index_uploaded)
        )

    @property
    def allocated_byte_count(self) -> int:
        return (
            self.vertex_count * _SURFACE_STRIDE
            + self.index_count * 4
            + self.wire_index_count * 4
            + len(self.thick_line_bytes)
        )


__all__ = ["QRhiSurfaceRenderer"]
