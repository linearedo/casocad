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

import logging
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time

from PySide6.QtCore import QByteArray, QObject, Signal
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

import numpy as np

from core.gpu_cull import build_term_grid, leaf_bounds
from core.gpu_codegen import (
    emit_fragment_shader, flatten_terms, group_capacity, profiles_are_simple,
    scene_structure_signature, selector_indices, supported as cg_supported,
)
from core.gpu_scene import serialize_scene

from .vulkanize import UBO_BINDING, uniform_block_members, vulkanize

log = logging.getLogger(__name__)

# Codegen spatial-cull grid resolution (GxGxG).
_CULL_GRID_DIM = 16

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
# with the fragment raymarcher's exact ray math —
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
    ("float", "clip_y_sign", None),
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
    // clip_y_sign flips NDC-Y for a Y-up-NDC backend (OpenGL); +1 on Vulkan
    // leaves the proven path unchanged.
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


# The qsb compile runs on a worker thread so a structure-change edit never freezes
# the GUI; the previous scene keeps rendering until the new shader (and pipeline)
# are ready. Plus an in-session pipeline cache so revisiting a scene structure
# skips the driver pipeline compile. See progress/async_bake.md.
_PIPE_CACHE_PATH = os.path.join(
    tempfile.gettempdir(), "casocad_qrhi_pipeline_cache.bin")


class _BakeSignals(QObject):
    """GUI-thread marshaling for the worker bake (emitted from the worker, the
    queued connection delivers it on the thread the QObject lives on = GUI)."""
    done = Signal(object, object)    # (sig, QShader)
    error = Signal(object, str)      # (sig, message)


class QRhiInterpreterRenderer:
    def __init__(self) -> None:
        self._rhi = None
        self._baked = False
        self._vert_shader = None             # shared fullscreen-triangle vertex
        # --- codegen path (the ONLY render path; the bytecode VM was removed) ----
        # Every supported scene renders via a small scene-specialized shader keyed
        # by (kinds, group-capacity, simple-profiles). See core/gpu_codegen.py.
        self._cg_active = False              # is the current scene renderable by codegen?
        self._cg_render_ir = None
        self._cg_sig = None                  # baked (kinds, cap, simple) bake key
        self._cg_frag = None
        self._cg_frag_cache: dict = {}
        self._cg_ubo_members = None
        self._cg_ubo = None
        self._cg_pipe = None
        self._cg_srb = None
        self._cg_bufs = None
        self._cg_data = None
        self._cg_uploaded = False
        # --- async bake + pipeline cache ----------------------------------------
        self._cg_pipe_cache: dict = {}   # sig -> QRhiGraphicsPipeline (in-session)
        self._cg_baking: set = set()     # sigs with a worker bake in flight
        self._cg_pending_sig = None      # the structure the user most recently wants
        self._update_cb = None           # viewport.update, called when a bake lands
        self._bake_signals = _BakeSignals()
        self._bake_signals.done.connect(self._on_async_bake_done)
        self._bake_signals.error.connect(self._on_async_bake_error)
        self._rpd = None             # render pass descriptor (stashed for rebuilds)
        self._fb_y_up = 1            # framebuffer Y-up? (set from QRhi in initialize)
        self._line_clip_y_sign = 1.0  # overlay NDC-Y sign (set from QRhi)
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
        self._vert_shader = _bake("vert", _FULLSCREEN_VERT)
        # overlay line resources (gizmos; scene-independent, built once)
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
            "res": (1.0, 1.0), "half_px": _LINE_HALF_PX, "clip_y_sign": 1.0,
        }

    # -- resource build — ALWAYS outside a frame -----------------------------

    def initialize(self, rhi, render_target) -> None:
        self._rhi = rhi
        # Backend coordinate conventions (constant per backend): OpenGL's
        # framebuffer + NDC are Y-up, Vulkan's are Y-down. The raymarcher reads
        # gl_FragCoord (framebuffer space, u_fb_y_up); the overlay writes
        # gl_Position (NDC, clip_y_sign). QRhi does not auto-correct either.
        self._fb_y_up = 1 if rhi.isYUpInFramebuffer() else 0
        self._line_clip_y_sign = -1.0 if rhi.isYUpInNDC() else 1.0
        backend = rhi.backendName() if hasattr(rhi, "backendName") else "?"
        log.info("qrhi: initialize backend=%s fb_y_up=%d clip_y_sign=%+.0f",
                 backend, self._fb_y_up, self._line_clip_y_sign)
        self._rpd = render_target.renderPassDescriptor()
        self._load_pipeline_cache()
        self._bake_once()
        self._build_line_pipeline(self._rpd)
        if self._cg_active:   # seeded scene is renderable by codegen
            self._build_codegen_resources()

    def set_scene(self, render_ir) -> None:
        # Codegen is the only renderer. A scene it can't emit (e.g. > CG_GROUP_CEILING
        # cross-product groups) renders nothing, with a warning — the VM fallback
        # that used to cover it has been removed.
        self._cg_active = bool(cg_supported(render_ir))
        self._cg_render_ir = render_ir if self._cg_active else None
        if not self._cg_active and getattr(render_ir, "nodes", None):
            log.warning("qrhi: scene not renderable by codegen (too complex — "
                        "exceeds the group ceiling); nothing drawn")
        if not self._cg_active:
            self._cg_pipe = None   # nothing renderable this scene
            return
        # Before initialize() there is no QRhi yet: just stage the data; the seeded
        # build happens synchronously in initialize().
        if not self._baked:
            self._prepare_codegen_data(render_ir)
            return
        # A structure whose shader is already baked finalizes inline (fast: no qsb;
        # pipeline cached or one driver compile). This also covers the common
        # moved-object edit (sig == current). Otherwise bake OFF-THREAD and keep
        # rendering the PREVIOUS scene until it lands.
        sig = (scene_structure_signature(render_ir), group_capacity(render_ir),
               profiles_are_simple(render_ir))
        self._cg_pending_sig = sig
        if sig in self._cg_frag_cache:
            self._prepare_codegen_data(render_ir)
            self._build_codegen_resources()
            return
        if sig not in self._cg_baking:
            self._cg_baking.add(sig)
            gl_src = emit_fragment_shader(render_ir)
            threading.Thread(target=self._async_bake_worker,
                             args=(sig, gl_src), daemon=True).start()

    def set_update_callback(self, cb) -> None:
        """Viewport hands us its ``update`` so a landed async bake can request a
        redraw (the edit that triggered it may have already gone idle)."""
        self._update_cb = cb

    # -- async bake (worker thread = qsb only; NEVER touch QRhi here) ---------

    def _async_bake_worker(self, sig, gl_src) -> None:
        try:
            shader = _bake("frag", vulkanize(gl_src))
        except Exception as exc:  # noqa: BLE001
            self._bake_signals.error.emit(sig, str(exc))
            return
        self._bake_signals.done.emit(sig, shader)   # queued -> GUI thread

    def _on_async_bake_done(self, sig, shader) -> None:
        # GUI thread (queued). Cache the shader; finalize only if it's still the
        # structure the user wants (a newer edit may have superseded it).
        self._cg_frag_cache[sig] = shader
        self._cg_baking.discard(sig)
        if sig != self._cg_pending_sig or self._cg_render_ir is None:
            return
        # Build buffers + pipeline now (outside any frame). The frag is cached, so
        # _build_codegen_resources skips qsb; only the driver pipeline compile (or
        # an in-session cache hit) remains.
        self._prepare_codegen_data(self._cg_render_ir)
        self._build_codegen_resources()
        if self._update_cb is not None:
            self._update_cb()

    def _on_async_bake_error(self, sig, message) -> None:
        self._cg_baking.discard(sig)
        log.warning("qrhi: async bake failed for sig=%s: %s", sig, message)

    # -- pipeline cache (in-session always; disk best-effort) ----------------

    def _load_pipeline_cache(self) -> None:
        try:
            if os.path.exists(_PIPE_CACHE_PATH):
                with open(_PIPE_CACHE_PATH, "rb") as f:
                    self._rhi.setPipelineCacheData(QByteArray(f.read()))
                log.info("qrhi: loaded driver pipeline cache (%s)", _PIPE_CACHE_PATH)
        except Exception as exc:  # noqa: BLE001
            log.info("qrhi: pipeline cache load skipped (%s)", exc)

    def save_pipeline_cache(self) -> None:
        """Persist the driver's compiled-pipeline blob, if the backend collects it
        (requires EnablePipelineCacheDataSave at QRhi creation — may be empty under
        QRhiWidget, in which case this is a harmless no-op)."""
        if self._rhi is None:
            return
        try:
            data = bytes(self._rhi.pipelineCacheData())
            if data:
                with open(_PIPE_CACHE_PATH, "wb") as f:
                    f.write(data)
                log.info("qrhi: saved driver pipeline cache (%d bytes)", len(data))
        except Exception as exc:  # noqa: BLE001
            log.info("qrhi: pipeline cache save skipped (%s)", exc)

    # -- codegen path --------------------------------------------------------

    def _cg_zero_uniforms(self) -> dict:
        return {
            "u_term_count": 0, "u_sel_count": 0, "u_group_count": 1,
            "u_cull_enabled": 0, "u_grid_dim": 1, "u_grid_origin": (0.0, 0.0, 0.0),
            "u_grid_cell": (1.0, 1.0, 1.0),
            "u_resolution": (1.0, 1.0), "u_camera_position": (0, 0, 1),
            "u_camera_target": (0, 0, 0), "u_camera_right": (1, 0, 0),
            "u_camera_up": (0, 1, 0), "u_focal_length": 1.5,
            "u_background_color": (0.07, 0.08, 0.10), "u_show_grid": 1,
            "u_grid_spacing": 1.0, "u_fb_y_up": 1,
        }

    def _prepare_codegen_data(self, render_ir) -> None:
        """Pack the scene's data buffers for the codegen shader (nodes/params from
        serialize_scene; carved terms + a flat carve list from flatten_terms; and a
        spatial cull grid binning the terms). Each term is (leaf, group_id,
        carve_offset, carve_count) into the carve buffer. Data only — never changes
        the shader."""
        sc = serialize_scene(render_ir)
        groups = flatten_terms(render_ir)
        # Flatten groups -> a uvec4 per term plus a shared carve-index buffer.
        term_rows, carve_idx = [], []
        for gid, grp in enumerate(groups):
            for leaf, carves in grp:
                term_rows.append([leaf, gid, len(carve_idx), len(carves)])
                carve_idx.extend(carves)
        terms = (np.array(term_rows, dtype=np.uint32) if term_rows
                 else np.zeros((0, 4), dtype=np.uint32))
        carves = np.array(carve_idx, dtype=np.uint32)
        sel = np.array(selector_indices(render_ir), dtype=np.uint32)
        # Spatial cull grid over the terms' positive-leaf bounds (None if any leaf
        # is unbounded -> brute-force map() with cull disabled).
        grid = None
        if term_rows:
            pos_leaves = terms[:, 0]
            grid = build_term_grid(pos_leaves, leaf_bounds(render_ir),
                                   dim=_CULL_GRID_DIM)
        if grid is not None:
            origin, cell, dim, goff, gcnt, gitem = grid
            cull, gdim, gorigin, gcell = 1, dim, origin, cell
            goff_b, gcnt_b, gitem_b = goff.tobytes(), gcnt.tobytes(), gitem.tobytes()
        else:
            cull, gdim, gorigin, gcell = 0, 1, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
            goff_b = gcnt_b = gitem_b = b""
        z = b"\x00\x00\x00\x00"
        # Buffer order matches bindings: nodes@0, params@1, children@2, sel@3,
        # terms@4, carves@5, grid_off@6, grid_cnt@7, grid_items@8; then scalars.
        self._cg_data = (
            sc.nodes_bytes or z, sc.params_bytes or z, sc.children_bytes or z,
            sel.tobytes() or z, terms.tobytes() or z, carves.tobytes() or z,
            goff_b or z, gcnt_b or z, gitem_b or z,
            int(len(terms)), int(len(sel)), int(len(groups)),
            int(cull), int(gdim), tuple(gorigin), tuple(gcell),
        )

    def _build_codegen_resources(self) -> None:
        """Bake (cached by structure signature + group capacity) + build the codegen
        pipeline and buffers. Re-bakes only when the set of primitive kinds OR the
        group-capacity bucket changes."""
        ir = self._cg_render_ir
        if ir is None:
            self._cg_pipe = None
            return
        # Bake key = (leaf kinds, group-capacity bucket, simple-profiles flag). The
        # capacity sizes the g[] array; the profiles flag selects the straight-line
        # vs stack-VM profile block (a stack interpreter is Vulkan-pathological).
        sig = (scene_structure_signature(ir), group_capacity(ir),
               profiles_are_simple(ir))
        rebuilt = False
        if sig != self._cg_sig or self._cg_frag is None:
            gl_src = emit_fragment_shader(ir)
            if self._cg_ubo_members is None:
                self._cg_ubo_members = uniform_block_members(gl_src)
            frag = self._cg_frag_cache.get(sig)
            if frag is None:
                t = time.perf_counter()
                frag = _bake("frag", vulkanize(gl_src))
                self._cg_frag_cache[sig] = frag
                log.info("qrhi: codegen baked variant kinds=%s cap=%s in %.2fs",
                         sorted(sig[0]), sig[1], time.perf_counter() - t)
            self._cg_frag = frag
            self._cg_sig = sig
            rebuilt = True
            if self._cg_ubo is None:
                ubo_bytes = _std140(self._cg_ubo_members, self._cg_zero_uniforms())
                self._cg_ubo = self._rhi.newBuffer(
                    QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.UniformBuffer,
                    len(ubo_bytes))
                self._cg_ubo.create()
        # Scene data buffers are rebuilt every set_scene (sizes vary):
        # nodes, params, children, sel, terms, carves, grid_off/cnt/items.
        bufs = []
        for data in self._cg_data[:9]:
            b = self._rhi.newBuffer(QRhiBuffer.Type.Static,
                                    QRhiBuffer.UsageFlag.StorageBuffer, max(len(data), 4))
            b.create()
            bufs.append(b)
        self._cg_bufs = tuple(bufs)
        self._cg_uploaded = False
        self._build_cg_srb()
        # In-session pipeline cache (async path): revisiting a scene STRUCTURE
        # reuses its already-compiled pipeline, skipping the driver pipeline
        # compile. The codegen SRB layout is identical across scenes, so a pipeline
        # built with an earlier SRB is layout-compatible with the current one at
        # draw time.
        cached = self._cg_pipe_cache.get(sig)
        if cached is not None:
            self._cg_pipe = cached
        elif rebuilt or self._cg_pipe is None:
            F = QRhiShaderResourceBinding.StageFlag.FragmentStage  # noqa: F841
            pipe = self._rhi.newGraphicsPipeline()
            pipe.setShaderStages([
                QRhiShaderStage(QRhiShaderStage.Type.Vertex, self._vert_shader),
                QRhiShaderStage(QRhiShaderStage.Type.Fragment, self._cg_frag)])
            pipe.setVertexInputLayout(QRhiVertexInputLayout())
            pipe.setShaderResourceBindings(self._cg_srb)
            pipe.setRenderPassDescriptor(self._rpd)
            t = time.perf_counter()
            if not pipe.create():
                log.warning("qrhi: codegen pipeline create() FAILED")
            else:
                log.info("qrhi: pipeline driver-compiled in %.2fs",
                         time.perf_counter() - t)
            self._cg_pipe = pipe
            self._cg_pipe_cache[sig] = pipe

    def _build_cg_srb(self) -> None:
        F = QRhiShaderResourceBinding.StageFlag.FragmentStage
        n, p, c, sel, a, s, goff, gcnt, gitem = self._cg_bufs
        srb = self._rhi.newShaderResourceBindings()
        srb.setBindings([
            QRhiShaderResourceBinding.bufferLoad(0, F, n),
            QRhiShaderResourceBinding.bufferLoad(1, F, p),
            QRhiShaderResourceBinding.bufferLoad(2, F, c),
            QRhiShaderResourceBinding.bufferLoad(3, F, sel),
            QRhiShaderResourceBinding.bufferLoad(4, F, a),
            QRhiShaderResourceBinding.bufferLoad(5, F, s),
            QRhiShaderResourceBinding.bufferLoad(6, F, goff),
            QRhiShaderResourceBinding.bufferLoad(7, F, gcnt),
            QRhiShaderResourceBinding.bufferLoad(8, F, gitem),
            QRhiShaderResourceBinding.uniformBuffer(UBO_BINDING, F, self._cg_ubo)])
        srb.create()
        self._cg_srb = srb

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
        cg = (self._cg_active and self._cg_pipe is not None
              and self._cg_bufs is not None)

        vcount = 0
        if overlay is not None and self._line_pipe is not None and overlay[1] > 0:
            vbytes, vcount = overlay
            cap = _LINE_MAX_VERTS * _LINE_STRIDE
            if len(vbytes) > cap:
                vbytes, vcount = vbytes[:cap], cap // _LINE_STRIDE

        rub = rhi.nextResourceUpdateBatch()
        if cg:
            if not self._cg_uploaded:
                for buf, data in zip(self._cg_bufs, self._cg_data[:9]):
                    rub.uploadStaticBuffer(
                        buf, data if len(data) >= 4 else b"\x00\x00\x00\x00")
                self._cg_uploaded = True
            vals = self._cg_zero_uniforms()
            for k in ("u_camera_position", "u_camera_target", "u_camera_right",
                      "u_camera_up", "u_focal_length", "u_background_color",
                      "u_show_grid", "u_grid_spacing"):
                if k in camera:
                    vals[k] = camera[k]
            vals["u_term_count"] = self._cg_data[9]
            vals["u_sel_count"] = self._cg_data[10]
            vals["u_group_count"] = self._cg_data[11]
            vals["u_cull_enabled"] = self._cg_data[12]
            vals["u_grid_dim"] = self._cg_data[13]
            vals["u_grid_origin"] = self._cg_data[14]
            vals["u_grid_cell"] = self._cg_data[15]
            vals["u_resolution"] = (float(w), float(h))
            vals["u_fb_y_up"] = self._fb_y_up
            rub.updateDynamicBuffer(
                self._cg_ubo, 0, _std140(self._cg_ubo_members, vals))
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
                    "clip_y_sign": self._line_clip_y_sign,
                }))

        cb.beginPass(render_target, QColor.fromRgbF(*bg, 1.0),
                     QRhiDepthStencilClearValue(1.0, 0), rub)
        if cg:
            cb.setGraphicsPipeline(self._cg_pipe)
            cb.setViewport(QRhiViewport(0, 0, w, h))
            cb.setShaderResources(self._cg_srb)
            cb.draw(3)
        if vcount:
            cb.setGraphicsPipeline(self._line_pipe)
            cb.setViewport(QRhiViewport(0, 0, w, h))
            cb.setShaderResources(self._line_srb)
            cb.setVertexInput(0, [(self._line_vbuf, 0)])
            cb.draw(vcount)
        cb.endPass()


__all__ = ["QRhiInterpreterRenderer"]
