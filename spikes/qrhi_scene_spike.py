"""Real-scene QRhi render spike — the bridge from the toy spike to the renderer.

Runs the ACTUAL casoCAD interpreter through QRhi end-to-end:

    a real SDF scene (a sphere) -> serialize_scene / emit_program
      -> 4 storage buffers (nodes/params/children/bytecode) + a std140 camera UBO
      -> the vulkanized raymarch_interpreter.comp (baked by qsb) as a COMPUTE pass
      -> writes an RGBA8 texture
      -> a fullscreen blit draws the texture into the QRhiWidget

If a sphere shows up in the window, the whole renderer core works through QRhi and
the rest of the port is app wiring. Run on your GPU box:

    .venv/bin/python spikes/qrhi_scene_spike.py

A traceback means a QRhi API detail needs fixing — paste it back.
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

from PySide6.QtCore import QByteArray, QSize
from PySide6.QtGui import (
    QColor,
    QRhi,
    QRhiBuffer,
    QRhiDepthStencilClearValue,
    QRhiGraphicsPipeline,
    QRhiSampler,
    QRhiShaderResourceBinding,
    QRhiShaderStage,
    QRhiTexture,
    QRhiVertexInputLayout,
    QRhiViewport,
    QShader,
)
from PySide6.QtWidgets import QApplication, QRhiWidget

from core.gpu_program import emit_program
from core.gpu_scene import serialize_scene
from core.render_ir import build_render_ir
from core.sdf import SDFTree, Sphere
from app.viewport.renderers.interpreter_glsl.shader_assembly import (
    build_program_source,
)
from app.viewport.renderers.qrhi.vulkanize import (
    IMAGE_BINDING,
    UBO_BINDING,
    uniform_block_members,
    vulkanize,
)

RES = 512  # square compute target; the blit stretches it to the widget
_SH = "app/viewport/renderers/interpreter_glsl/shaders"

_BLIT_VERT = """\
#version 450
layout(location = 0) out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""
_BLIT_FRAG = """\
#version 450
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag;
layout(binding = 0) uniform sampler2D u_tex;
void main() { frag = texture(u_tex, v_uv); }
"""

_ALIGN = {"float": 4, "int": 4, "uint": 4, "vec2": 8, "vec3": 16, "vec4": 16}
_SIZE = {"float": 4, "int": 4, "uint": 4, "vec2": 8, "vec3": 12, "vec4": 16}


def _std140(members, values) -> bytes:
    """Pack a uniform block in std140 order (no arrays in the compute path)."""
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
    tmp = tempfile.mkdtemp(prefix="qrhi_scene_")
    src = os.path.join(tmp, f"s.{ext}")
    out = src + ".qsb"
    with open(src, "w") as f:
        f.write(glsl)
    qsb = os.path.join(os.path.dirname(sys.executable), "pyside6-qsb")
    subprocess.run([qsb, "--glsl", "430", "-o", out, src], check=True)
    with open(out, "rb") as f:
        return QShader.fromSerialized(QByteArray(f.read()))


def _sphere_scene():
    root = Sphere(name="ball", object_id=1, center=(0.0, 0.0, 0.0), radius=1.0)
    ir = build_render_ir(SDFTree(root=root))
    return serialize_scene(ir), emit_program(ir), ir


def _camera_values(program_length: int):
    pos = np.array([3.2, 2.4, 3.0]); tgt = np.array([0.0, 0.0, 0.0])
    fwd = tgt - pos; fwd /= np.linalg.norm(fwd)
    up0 = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up0); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return {
        "u_program_length": program_length,
        "u_resolution": (RES, RES),
        "u_camera_position": tuple(pos),
        "u_camera_target": tuple(tgt),
        "u_camera_right": tuple(right),
        "u_camera_up": tuple(up),
        "u_focal_length": 1.5,
        "u_surface_opacity": 1.0,
        "u_background_color": (0.07, 0.08, 0.10),
    }


_BACKENDS = {
    "vulkan": QRhiWidget.Api.Vulkan,
    "opengl": QRhiWidget.Api.OpenGL,
    "metal": getattr(QRhiWidget.Api, "Metal", None),
    "d3d11": getattr(QRhiWidget.Api, "Direct3D11", None),
}


class SceneSpikeWidget(QRhiWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(RES, RES)
        want = os.environ.get("QRHI_BACKEND", "").lower()
        if want in _BACKENDS and _BACKENDS[want] is not None:
            self.setApi(_BACKENDS[want])
        self._built = False

    def initialize(self, cb) -> None:
        try:
            self._initialize(cb)
        except BaseException:
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise

    def _initialize(self, cb) -> None:
        def log(*a):
            print(*a, flush=True)
        rhi = self.rhi()
        if self._built:
            return
        log("BACKEND:", rhi.backendName(),
            "| compute:", rhi.isFeatureSupported(QRhi.Feature.Compute))

        # --- scene + shader ---
        scene, program, _ = _sphere_scene()
        comp_main = open(os.path.join(_SH, "raymarch_interpreter.comp")).read()
        gl_src = build_program_source(frozenset(), comp_main)  # core only
        self._ubo_members = uniform_block_members(gl_src)
        comp_shader = _bake("comp", vulkanize(gl_src))
        log("compute shader baked:", comp_shader.isValid())

        # --- storage buffers (uploaded once) ---
        def sbuf(data: bytes):
            data = data or b"\x00\x00\x00\x00"
            b = rhi.newBuffer(QRhiBuffer.Type.Static,
                              QRhiBuffer.UsageFlag.StorageBuffer, len(data))
            b.create()
            return b, data

        self._nodes, self._nodes_d = sbuf(scene.nodes_bytes)
        self._params, self._params_d = sbuf(scene.params_bytes)
        self._children, self._children_d = sbuf(scene.children_bytes)
        self._bytecode, self._bytecode_d = sbuf(program.bytecode_bytes)
        log("storage buffers created")

        # --- camera UBO (updated each frame) ---
        ubo_bytes = _std140(self._ubo_members, _camera_values(program.program_length))
        self._ubo = rhi.newBuffer(QRhiBuffer.Type.Dynamic,
                                  QRhiBuffer.UsageFlag.UniformBuffer, len(ubo_bytes))
        self._ubo.create()
        self._ubo_bytes = ubo_bytes
        log("UBO created", len(ubo_bytes), "bytes")

        # --- output texture (compute writes, blit samples) ---
        self._tex = rhi.newTexture(QRhiTexture.Format.RGBA8, QSize(RES, RES), 1,
                                   QRhiTexture.Flag.UsedWithLoadStore)
        self._tex.create()
        self._sampler = rhi.newSampler(
            QRhiSampler.Filter.Linear, QRhiSampler.Filter.Linear,
            QRhiSampler.Filter.None_,
            QRhiSampler.AddressMode.ClampToEdge, QRhiSampler.AddressMode.ClampToEdge)
        self._sampler.create()
        log("texture + sampler created")

        # --- compute pipeline ---
        C = QRhiShaderResourceBinding.StageFlag.ComputeStage
        self._csrb = rhi.newShaderResourceBindings()
        self._csrb.setBindings([
            QRhiShaderResourceBinding.bufferLoad(0, C, self._nodes),
            QRhiShaderResourceBinding.bufferLoad(1, C, self._params),
            QRhiShaderResourceBinding.bufferLoad(2, C, self._children),
            QRhiShaderResourceBinding.bufferLoad(3, C, self._bytecode),
            QRhiShaderResourceBinding.imageStore(IMAGE_BINDING, C, self._tex, 0),
            QRhiShaderResourceBinding.uniformBuffer(UBO_BINDING, C, self._ubo),
        ])
        self._csrb.create()
        log("compute SRB created")
        self._cpipe = rhi.newComputePipeline()
        self._cpipe.setShaderStage(QRhiShaderStage(QRhiShaderStage.Type.Compute, comp_shader))
        self._cpipe.setShaderResourceBindings(self._csrb)
        log("compute pipeline:", self._cpipe.create())

        # --- blit (graphics) pipeline ---
        F = QRhiShaderResourceBinding.StageFlag.FragmentStage
        self._gsrb = rhi.newShaderResourceBindings()
        self._gsrb.setBindings([
            QRhiShaderResourceBinding.sampledTexture(0, F, self._tex, self._sampler)])
        self._gsrb.create()
        self._gpipe = rhi.newGraphicsPipeline()
        self._gpipe.setShaderStages([
            QRhiShaderStage(QRhiShaderStage.Type.Vertex, _bake("vert", _BLIT_VERT)),
            QRhiShaderStage(QRhiShaderStage.Type.Fragment, _bake("frag", _BLIT_FRAG))])
        self._gpipe.setVertexInputLayout(QRhiVertexInputLayout())
        self._gpipe.setShaderResourceBindings(self._gsrb)
        self._gpipe.setRenderPassDescriptor(self.renderTarget().renderPassDescriptor())
        log("blit pipeline:", self._gpipe.create())

        self._uploaded = False
        self._built = True
        log("initialize() complete")

    def render(self, cb) -> None:
        rhi = self.rhi()
        rub = rhi.nextResourceUpdateBatch()
        if not self._uploaded:
            rub.uploadStaticBuffer(self._nodes, self._nodes_d)
            rub.uploadStaticBuffer(self._params, self._params_d)
            rub.uploadStaticBuffer(self._children, self._children_d)
            rub.uploadStaticBuffer(self._bytecode, self._bytecode_d)
            self._uploaded = True
        rub.updateDynamicBuffer(self._ubo, 0, self._ubo_bytes)

        cb.beginComputePass(rub)
        cb.setComputePipeline(self._cpipe)
        cb.setShaderResources(self._csrb)
        cb.dispatch((RES + 7) // 8, (RES + 7) // 8, 1)
        cb.endComputePass()

        out = self.renderTarget()
        cb.beginPass(out, QColor.fromRgbF(0.07, 0.08, 0.10, 1.0),
                     QRhiDepthStencilClearValue(1.0, 0))
        cb.setGraphicsPipeline(self._gpipe)
        sz = out.pixelSize()
        cb.setViewport(QRhiViewport(0, 0, sz.width(), sz.height()))
        cb.setShaderResources(self._gsrb)
        cb.draw(3)
        cb.endPass()


def main() -> int:
    app = QApplication(sys.argv)
    w = SceneSpikeWidget()
    w.setWindowTitle("QRhi real-scene spike — a sphere = the renderer works")
    w.resize(640, 640)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
