from __future__ import annotations

"""QRhi interpreter renderer — fragment path (portable, single codebase).

Runs casoCAD's SDF interpreter through Qt's QRhi as a **fragment raymarcher**:
the scene bytecode lives in storage buffers, a fullscreen triangle's fragment
shader reads them and raymarches one pixel per fragment **straight to the render
target** — one pass, no compute/texture/blit. This mirrors the OpenGL fragment
path that already renders smoothly in the app, and is what the QRhi/Vulkan path
needs to be fast (the compute→texture→blit→composite chain was the lag, D-R1).

The *same* code drives Vulkan / Metal / D3D — only the backend pick differs.
Shaders are baked once by ``qsb`` into all backend variants (pre-baked SPIR-V =
no per-frame driver GLSL compile).

Resource discipline: **all GPU resources are built in ``initialize`` / ``set_scene``
(outside a frame); ``render`` only records the pass.** Building pipelines during a
frame corrupts the submit (a segfault we hit and fixed).

Reference: `progress/switch_to_QRhiWidget_progress.md` (Phase 3, D-R1).
"""

import os
import struct
import subprocess
import sys
import tempfile

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

from core.gpu_program import emit_program
from core.gpu_scene import serialize_scene
from app.viewport.renderers.interpreter_glsl.shader_assembly import (
    build_program_source,
)

from .vulkanize import UBO_BINDING, uniform_block_members, vulkanize

_INTERP_SHADER_DIR = os.path.join(
    os.path.dirname(__file__), "..", "interpreter_glsl", "shaders"
)
_FRAG_MAIN = os.path.join(os.path.dirname(__file__), "raymarch_frag_main.glsl")

# Fullscreen triangle; the fragment shader uses gl_FragCoord, so no varyings.
_FULLSCREEN_VERT = """\
#version 450
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_ALIGN = {"float": 4, "int": 4, "uint": 4, "vec2": 8, "vec3": 16, "vec4": 16}
_SIZE = {"float": 4, "int": 4, "uint": 4, "vec2": 8, "vec3": 12, "vec4": 16}

# ---- overlay pipeline (gizmos / previews drawn over the SDF pass) ------------
# Constant-pixel-width colored lines, drawn as screen-space-expanded TRIANGLES
# (GPU line width is 1px / unsupported on most backends, so thin lines are
# invisible). Each segment is a quad: the vertex shader projects both endpoints
# with the fragment raymarcher's exact ray math (see raymarch_frag_main.glsl) —
# v = P - camPos in the orthonormal (right, up, fwd) basis, NDC = (focal*a*
# aspect, -focal*b)/c — then offsets each vertex perpendicular to the segment in
# screen pixels, so gizmos stay a fixed crisp width at any zoom.
_LINE_MAX_VERTS = 8192
_LINE_STRIDE = 44  # vec3 a + vec3 b + vec3 col + vec2 (endpoint_sel, side)
_LINE_HALF_PX = 3.0  # half line width in pixels (=> 6px lines)
_LINE_UBO_MEMBERS = (
    ("vec3", "cam_pos", None),
    ("vec3", "cam_right", None),
    ("vec3", "cam_up", None),
    ("vec3", "cam_target", None),
    ("float", "focal", None),
    ("float", "aspect", None),
    ("vec2", "res", None),
    ("float", "half_px", None),
)
_LINE_VERT = """\
#version 450
layout(location = 0) in vec3 in_a;
layout(location = 1) in vec3 in_b;
layout(location = 2) in vec3 in_col;
layout(location = 3) in vec2 in_param;  // x = endpoint (0=a,1=b), y = side (-1/+1)
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
    gl_Position = vec4(ndc * cthis.w, cthis.z, cthis.w);
    v_col = in_col;
}
"""
_LINE_FRAG = """\
#version 450
layout(location = 0) in vec3 v_col;
layout(location = 0) out vec4 frag_color;
void main() { frag_color = vec4(v_col, 1.0); }
"""


def _std140(members, values) -> bytes:
    offsets, off = {}, 0
    for gtype, name, _arr in members:
        a = _ALIGN[gtype]
        off = (off + a - 1) // a * a
        offsets[name] = (off, gtype)
        off += _SIZE[gtype]
    data = bytearray((off + 15) // 16 * 16)
    for name, (o, gtype) in offsets.items():
        v = values[name]
        if gtype == "uint":
            struct.pack_into("<I", data, o, int(v))
        elif gtype == "int":
            struct.pack_into("<i", data, o, int(v))
        elif gtype == "float":
            struct.pack_into("<f", data, o, float(v))
        elif gtype == "vec2":
            struct.pack_into("<2f", data, o, *map(float, v))
        elif gtype == "vec3":
            struct.pack_into("<3f", data, o, *map(float, v))
        elif gtype == "vec4":
            struct.pack_into("<4f", data, o, *map(float, v))
    return bytes(data)


def _bake(ext: str, glsl: str) -> QShader:
    tmp = tempfile.mkdtemp(prefix="qrhi_render_")
    src = os.path.join(tmp, f"s.{ext}")
    out = src + ".qsb"
    with open(src, "w") as f:
        f.write(glsl)
    qsb = os.path.join(os.path.dirname(sys.executable), "pyside6-qsb")
    subprocess.run([qsb, "--glsl", "430", "-o", out, src], check=True)
    with open(out, "rb") as f:
        return QShader.fromSerialized(QByteArray(f.read()))


class QRhiInterpreterRenderer:
    # Per-thread value-stack depth — match the proven fragment path's cap.
    STACK_CAPACITY = 16

    def __init__(self) -> None:
        self._rhi = None
        self._baked = False
        self._vert_shader = self._frag_shader = None
        self._ubo_members = None
        self._ubo = None
        self._scene = None          # (nodes, params, children, bytecode, program_length)
        self._buffers = None
        self._uploaded = False
        self._srb = None
        self._gpipe = None
        # overlay line pipeline (gizmos)
        self._line_vert = self._line_frag = None
        self._line_ubo = None
        self._line_vbuf = None
        self._line_srb = None
        self._line_pipe = None

    # -- one-time bake -------------------------------------------------------

    def _bake_once(self) -> None:
        if self._baked:
            return
        frag_main = open(_FRAG_MAIN).read()
        gl_src = build_program_source(
            frozenset(), frag_main, stack_capacity=self.STACK_CAPACITY)
        self._ubo_members = uniform_block_members(gl_src)
        self._frag_shader = _bake("frag", vulkanize(gl_src))
        self._vert_shader = _bake("vert", _FULLSCREEN_VERT)
        ubo_bytes = _std140(self._ubo_members, self._zero_camera())
        self._ubo = self._rhi.newBuffer(
            QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.UniformBuffer, len(ubo_bytes))
        self._ubo.create()
        # overlay line resources (scene-independent; built once)
        self._line_vert = _bake("vert", _LINE_VERT)
        self._line_frag = _bake("frag", _LINE_FRAG)
        line_ubo_bytes = _std140(_LINE_UBO_MEMBERS, self._zero_line_ubo())
        self._line_ubo = self._rhi.newBuffer(
            QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.UniformBuffer,
            len(line_ubo_bytes))
        self._line_ubo.create()
        self._line_vbuf = self._rhi.newBuffer(
            QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.VertexBuffer,
            _LINE_MAX_VERTS * _LINE_STRIDE)
        self._line_vbuf.create()
        self._baked = True

    def _zero_line_ubo(self) -> dict:
        return {
            "cam_pos": (0, 0, 1), "cam_right": (1, 0, 0), "cam_up": (0, 1, 0),
            "cam_target": (0, 0, 0), "focal": 1.5, "aspect": 1.0,
            "res": (1.0, 1.0), "half_px": _LINE_HALF_PX,
        }

    def _zero_camera(self) -> dict:
        return {
            "u_program_length": 0, "u_resolution": (1.0, 1.0),
            "u_camera_position": (0, 0, 1), "u_camera_target": (0, 0, 0),
            "u_camera_right": (1, 0, 0), "u_camera_up": (0, 1, 0),
            "u_focal_length": 1.5, "u_surface_opacity": 1.0,
            "u_background_color": (0.07, 0.08, 0.10),
            "u_show_grid": 1, "u_grid_spacing": 1.0, "u_grid_plane": 0,
            "u_selected_object_id": 0,
        }

    # -- resource build — ALWAYS outside a frame -----------------------------

    def initialize(self, rhi, render_target) -> None:
        self._rhi = rhi
        self._bake_once()
        self._build_scene_buffers()
        self._build_pipeline(render_target.renderPassDescriptor())
        self._build_line_pipeline(render_target.renderPassDescriptor())

    def set_scene(self, render_ir) -> None:
        if render_ir is None or not getattr(render_ir, "supported", True) \
                or not getattr(render_ir, "nodes", None):
            self._scene = None
        else:
            scene = serialize_scene(render_ir)
            program = emit_program(
                render_ir, stack_capacity=self.STACK_CAPACITY,
                profile_capacity=self.STACK_CAPACITY)
            self._scene = (
                scene.nodes_bytes or b"\x00\x00\x00\x00",
                scene.params_bytes or b"\x00\x00\x00\x00",
                scene.children_bytes or b"\x00\x00\x00\x00",
                program.bytecode_bytes or b"\x00\x00\x00\x00",
                program.program_length,
            )
        if self._baked:  # already initialized -> rebuild buffers + SRB (+ pipeline)
            self._build_scene_buffers()
            if self._gpipe is not None:
                self._build_pipeline(self._gpipe.renderPassDescriptor())

    def _build_scene_buffers(self) -> None:
        self._buffers = None
        self._uploaded = False
        if self._scene is None:
            return
        bufs = []
        for data in self._scene[:4]:
            b = self._rhi.newBuffer(
                QRhiBuffer.Type.Static, QRhiBuffer.UsageFlag.StorageBuffer, len(data))
            b.create()
            bufs.append(b)
        self._buffers = tuple(bufs)

    def _build_pipeline(self, rpd) -> None:
        if self._buffers is None:
            self._gpipe = self._srb = None
            return
        rhi = self._rhi
        F = QRhiShaderResourceBinding.StageFlag.FragmentStage
        nodes, params, children, bytecode = self._buffers
        self._srb = rhi.newShaderResourceBindings()
        self._srb.setBindings([
            QRhiShaderResourceBinding.bufferLoad(0, F, nodes),
            QRhiShaderResourceBinding.bufferLoad(1, F, params),
            QRhiShaderResourceBinding.bufferLoad(2, F, children),
            QRhiShaderResourceBinding.bufferLoad(3, F, bytecode),
            QRhiShaderResourceBinding.uniformBuffer(UBO_BINDING, F, self._ubo),
        ])
        self._srb.create()
        self._gpipe = rhi.newGraphicsPipeline()
        self._gpipe.setShaderStages([
            QRhiShaderStage(QRhiShaderStage.Type.Vertex, self._vert_shader),
            QRhiShaderStage(QRhiShaderStage.Type.Fragment, self._frag_shader)])
        self._gpipe.setVertexInputLayout(QRhiVertexInputLayout())
        self._gpipe.setShaderResourceBindings(self._srb)
        self._gpipe.setRenderPassDescriptor(rpd)
        self._gpipe.create()

    def _build_line_pipeline(self, rpd) -> None:
        """Scene-independent overlay pipeline: colored world-space line lists,
        depth test off so gizmos draw over the SDF pass. Built once."""
        rhi = self._rhi
        V = QRhiShaderResourceBinding.StageFlag.VertexStage
        self._line_srb = rhi.newShaderResourceBindings()
        self._line_srb.setBindings([
            QRhiShaderResourceBinding.uniformBuffer(0, V, self._line_ubo)])
        self._line_srb.create()
        F3 = QRhiVertexInputAttribute.Format.Float3
        F2 = QRhiVertexInputAttribute.Format.Float2
        vil = QRhiVertexInputLayout()
        vil.setBindings([QRhiVertexInputBinding(_LINE_STRIDE)])
        vil.setAttributes([
            QRhiVertexInputAttribute(0, 0, F3, 0),
            QRhiVertexInputAttribute(0, 1, F3, 12),
            QRhiVertexInputAttribute(0, 2, F3, 24),
            QRhiVertexInputAttribute(0, 3, F2, 36),
        ])
        pipe = rhi.newGraphicsPipeline()
        pipe.setTopology(QRhiGraphicsPipeline.Topology.Triangles)
        pipe.setCullMode(QRhiGraphicsPipeline.CullMode.None_)
        pipe.setDepthTest(False)
        pipe.setDepthWrite(False)
        pipe.setShaderStages([
            QRhiShaderStage(QRhiShaderStage.Type.Vertex, self._line_vert),
            QRhiShaderStage(QRhiShaderStage.Type.Fragment, self._line_frag)])
        pipe.setVertexInputLayout(vil)
        pipe.setShaderResourceBindings(self._line_srb)
        pipe.setRenderPassDescriptor(rpd)
        pipe.create()
        self._line_pipe = pipe

    # -- per-frame: one pass, no resource creation ---------------------------

    def render(self, cb, render_target, camera, overlay=None) -> None:
        """overlay: (vertex_bytes, vertex_count) of world-space colored lines, or
        None. Drawn over the SDF pass via the line pipeline."""
        rhi = self._rhi
        size = render_target.pixelSize()
        w, h = max(size.width(), 1), max(size.height(), 1)
        bg = camera.get("u_background_color", (0.07, 0.08, 0.10))
        have_scene = self._gpipe is not None and self._buffers is not None

        vcount = 0
        if overlay is not None and self._line_pipe is not None and overlay[1] > 0:
            vbytes, vcount = overlay
            cap = _LINE_MAX_VERTS * _LINE_STRIDE
            if len(vbytes) > cap:
                vbytes, vcount = vbytes[:cap], cap // _LINE_STRIDE

        rub = rhi.nextResourceUpdateBatch()
        if have_scene:
            if not self._uploaded:
                for buf, data in zip(self._buffers, self._scene[:4]):
                    rub.uploadStaticBuffer(buf, data)
                self._uploaded = True
            values = dict(camera)
            values["u_program_length"] = self._scene[4]
            values["u_resolution"] = (float(w), float(h))
            rub.updateDynamicBuffer(self._ubo, 0, _std140(self._ubo_members, values))
        if vcount:
            rub.updateDynamicBuffer(self._line_vbuf, 0, vbytes)
            rub.updateDynamicBuffer(self._line_ubo, 0, _std140(
                _LINE_UBO_MEMBERS, {
                    "cam_pos": camera["u_camera_position"],
                    "cam_right": camera["u_camera_right"],
                    "cam_up": camera["u_camera_up"],
                    "cam_target": camera["u_camera_target"],
                    "focal": camera["u_focal_length"],
                    "aspect": float(h) / float(w),
                    "res": (float(w), float(h)),
                    "half_px": _LINE_HALF_PX,
                }))

        cb.beginPass(render_target, QColor.fromRgbF(*bg, 1.0),
                     QRhiDepthStencilClearValue(1.0, 0), rub)
        if have_scene:
            cb.setGraphicsPipeline(self._gpipe)
            cb.setViewport(QRhiViewport(0, 0, w, h))
            cb.setShaderResources(self._srb)
            cb.draw(3)
        if vcount:
            cb.setGraphicsPipeline(self._line_pipe)
            cb.setViewport(QRhiViewport(0, 0, w, h))
            cb.setShaderResources(self._line_srb)
            cb.setVertexInput(0, [(self._line_vbuf, 0)])
            cb.draw(vcount)
        cb.endPass()


__all__ = ["QRhiInterpreterRenderer"]
