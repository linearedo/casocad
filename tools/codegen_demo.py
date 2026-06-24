#!/usr/bin/env python3
"""Render a scene through the THIN CODEGEN path (not the interpreter VM) and
screenshot it — a standalone proof that codegen emit -> compile -> data-upload ->
render works end to end. Prototype of the post-VM renderer.

    __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
    .venv/bin/python tools/codegen_demo.py [out.png] [scene] [per_union]

scene: intersection (default) | union | difference. NOTE: grabFramebuffer for the
screenshot works on OpenGL (the default here); add QRHI_BACKEND=vulkan to compile
on Vulkan (faster) — the on-screen window renders, but the saved PNG may be black
on Vulkan due to a Qt grab limitation.
"""
import math
import os
import random
import sys

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import (
    QColor, QRhiBuffer, QRhiDepthStencilClearValue, QRhiShaderResourceBinding,
    QRhiShaderStage, QRhiVertexInputLayout, QRhiViewport)
from PySide6.QtWidgets import QApplication

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.render_ir import RenderIR, RenderIRNode                       # noqa: E402
from core.gpu_codegen import emit_fragment_shader, supported            # noqa: E402
from core.gpu_codegen import flatten_terms, selector_indices          # noqa: E402
from core.gpu_scene import serialize_scene                              # noqa: E402
from app.viewport.renderers.qrhi.viewport import QRhiViewportWidget     # noqa: E402
from app.viewport.renderers.qrhi.renderer import (                      # noqa: E402
    _bake, _std140, _FULLSCREEN_VERT)
from app.viewport.renderers.qrhi.vulkanize import (                     # noqa: E402
    vulkanize, uniform_block_members, UBO_BINDING)


def build_scene(kind: str, per: int) -> RenderIR:
    rng = random.Random(7)
    nodes: list[RenderIRNode] = []

    def cluster(n):
        out = []
        for _ in range(n):
            nodes.append(RenderIRNode(kind="sphere", object_id=len(nodes) + 1,
                dimension=3, children=(),
                params=(rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-2, 2), 1.2)))
            out.append(len(nodes) - 1)
        return out

    def chain(op, idx):
        cur = idx[0]
        for t in idx[1:]:
            nodes.append(RenderIRNode(kind=op, object_id=0, dimension=3,
                                      children=(cur, t), params=())); cur = len(nodes) - 1
        return cur

    if kind == "selector":
        # spheres with a region tagged where inside a selector volume (Layer 2).
        rng2 = random.Random(13)
        geom_idx, sel_idx = [], []
        for k in range(per):
            cx, cy, cz = (rng2.uniform(-2, 2) for _ in range(3))
            nodes.append(RenderIRNode(kind="sphere", object_id=k + 1, dimension=3,
                children=(), params=(cx, cy, cz, 1.2)))
            geom_idx.append(len(nodes) - 1)
            nodes.append(RenderIRNode(kind="sphere", object_id=1000 + k, dimension=3,
                children=(), params=(cx + 0.7, cy, cz, 0.9)))   # selector volume
            vol = len(nodes) - 1
            nodes.append(RenderIRNode(kind="region_selector", object_id=k + 1,
                dimension=3, children=(vol,), params=(0.05,), flags=k + 1))
            sel_idx.append(len(nodes) - 1)
        root = chain("union", geom_idx)
        return RenderIR(nodes=tuple(nodes), root_indices=(root,),
                        component_indices=tuple(sel_idx))
    if kind == "carveunion":
        # union( sphere - hole , sphere2 ) -> carve-under-union (signed DNF).
        rng2 = random.Random(11)
        idx = []
        for _ in range(per):
            cx, cy, cz = (rng2.uniform(-2, 2) for _ in range(3))
            nodes.append(RenderIRNode(kind="sphere", object_id=len(nodes) + 1,
                dimension=3, children=(), params=(cx, cy, cz, 1.1)))
            solid = len(nodes) - 1
            nodes.append(RenderIRNode(kind="sphere", object_id=len(nodes) + 1,
                dimension=3, children=(), params=(cx + 0.6, cy, cz, 0.7)))
            hole = len(nodes) - 1
            nodes.append(RenderIRNode(kind="difference", object_id=0, dimension=3,
                children=(solid, hole), params=()))
            idx.append(len(nodes) - 1)
        root = chain("union", idx)   # union OF differences = carve-under-union
    elif kind == "profiles":
        # Extruded/revolved solids built from a 2D profile sub-graph -> exercises
        # the embedded profile stack-VM (evalProfileSDF) via codegen.
        rng2 = random.Random(9)
        b = (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)  # section world basis
        idx = []
        for _ in range(per):
            cx, cy, cz = (rng2.uniform(-2, 2) for _ in range(3))
            origin = (cx, cy, cz)
            sect_b = origin + (1, 0, 0, 0, 1, 0, 0, 0, 1)
            if rng2.random() < 0.5:                       # extrude(circle)
                nodes.append(RenderIRNode(kind="profile_circle_2d",
                    object_id=0, dimension=2, children=(), params=(0.0, 0.0, 0.6)))
                prof = len(nodes) - 1
                nodes.append(RenderIRNode(kind="extrude_profile_2d",
                    object_id=len(nodes) + 1, dimension=3, children=(prof,),
                    params=sect_b + (1.2, cz)))
            else:                                          # extrude(rectangle)
                nodes.append(RenderIRNode(kind="profile_rectangle_2d",
                    object_id=0, dimension=2, children=(), params=(0.0, 0.0, 0.6, 0.4)))
                prof = len(nodes) - 1
                nodes.append(RenderIRNode(kind="extrude_profile_2d",
                    object_id=len(nodes) + 1, dimension=3, children=(prof,),
                    params=sect_b + (1.2, cz)))
            idx.append(len(nodes) - 1)
        root = chain("union", idx)
    elif kind == "tubes":
        rng2 = random.Random(5)
        idx = []
        for _ in range(per):
            cx, cy, cz = (rng2.uniform(-2, 2) for _ in range(3))
            pts = [(cx, cy, cz), (cx + rng2.uniform(-1, 1), cy + 1.2, cz),
                   (cx + rng2.uniform(-1, 1), cy + 2.2, cz + rng2.uniform(-1, 1))]
            k = rng2.choice(["polyline_tube", "bezier_tube"])
            params = tuple(c for pt in pts for c in pt) + (0.3, 0.0, 0.0)
            nodes.append(RenderIRNode(kind=k, object_id=len(nodes) + 1, dimension=1,
                                      children=(), params=params))
            idx.append(len(nodes) - 1)
        root = chain("union", idx)
    elif kind == "sections":
        # Placed 2D analytic sections (thin discs/quads on the XY plane), unioned.
        rng2 = random.Random(3)
        base = (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)  # origin, axis_u, axis_v, normal
        idx = []
        for _ in range(per):
            cx, cy = rng2.uniform(-2, 2), rng2.uniform(-2, 2)
            k = rng2.choice(["placed_circle_2d", "placed_ellipse_2d",
                             "placed_rectangle_2d"])
            extra = {"placed_circle_2d": (cx, cy, 0.9),
                     "placed_ellipse_2d": (cx, cy, 1.0, 0.6),
                     "placed_rectangle_2d": (cx, cy, 0.8, 0.5)}[k]
            nodes.append(RenderIRNode(kind=k, object_id=len(nodes) + 1, dimension=2,
                                      children=(), params=base + extra))
            idx.append(len(nodes) - 1)
        root = chain("union", idx)
    elif kind == "union":
        root = chain("union", cluster(per * 2))
    elif kind == "difference":
        base = chain("union", cluster(per))
        cur = base
        for t in cluster(per):
            nodes.append(RenderIRNode(kind="difference", object_id=0, dimension=3,
                                      children=(cur, t), params=())); cur = len(nodes) - 1
        root = cur
    else:  # intersection
        a, b = chain("union", cluster(per)), chain("union", cluster(per))
        nodes.append(RenderIRNode(kind="intersection", object_id=0, dimension=3,
                                  children=(a, b), params=())); root = len(nodes) - 1
    return RenderIR(nodes=tuple(nodes), root_indices=(root,), component_indices=())


class Demo(QRhiViewportWidget):
    def set_demo_scene(self, ir):
        self._ir = ir

    def initialize(self, cb):
        rhi = self.rhi(); self._rhi = rhi
        self._fbup = 1 if rhi.isYUpInFramebuffer() else 0
        rpd = self.renderTarget().renderPassDescriptor()
        ir = self._ir
        gl_src = emit_fragment_shader(ir)
        self._members = uniform_block_members(gl_src)       # BEFORE vulkanize
        vert = _bake("vert", _FULLSCREEN_VERT)
        frag = _bake("frag", vulkanize(gl_src))
        print(f"[codegen-demo] backend={rhi.backendName()} ir_nodes={len(ir.nodes)} "
              f"supported={supported(ir)}", flush=True)

        sc = serialize_scene(ir); groups = flatten_terms(ir)
        term_rows, carve_idx = [], []
        for gid, grp in enumerate(groups):
            for leaf, carves in grp:
                term_rows.append([leaf, gid, len(carve_idx), len(carves)])
                carve_idx.extend(carves)
        terms = (np.array(term_rows, dtype=np.uint32) if term_rows
                 else np.zeros((0, 4), dtype=np.uint32))
        carves = np.array(carve_idx, dtype=np.uint32)
        sel = np.array(selector_indices(ir), dtype=np.uint32)
        self._gcount, self._tcount = len(groups), len(terms)
        self._selcount = len(sel)

        def sb(data):
            data = data if len(data) >= 4 else b"\x00\x00\x00\x00"
            b = rhi.newBuffer(QRhiBuffer.Type.Static, QRhiBuffer.UsageFlag.StorageBuffer, len(data))
            b.create(); return b, data
        # bindings 0-5 = scene data; 6-8 = dummy cull grid (demo runs brute-force,
        # u_cull_enabled=0 — the live renderer is what exercises the grid).
        self._bufs = [sb(sc.nodes_bytes), sb(sc.params_bytes), sb(sc.children_bytes),
                      sb(sel.tobytes()), sb(terms.tobytes()), sb(carves.tobytes()),
                      sb(b""), sb(b""), sb(b"")]
        self._ubo = rhi.newBuffer(QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.UniformBuffer,
                                  len(_std140(self._members, self._vals(1280, 800))))
        self._ubo.create()
        F = QRhiShaderResourceBinding.StageFlag.FragmentStage
        srb = rhi.newShaderResourceBindings()
        srb.setBindings([
            QRhiShaderResourceBinding.bufferLoad(b, F, self._bufs[b][0]) for b in range(9)
        ] + [QRhiShaderResourceBinding.uniformBuffer(UBO_BINDING, F, self._ubo)])
        srb.create(); self._srb = srb
        pipe = rhi.newGraphicsPipeline()
        pipe.setShaderStages([QRhiShaderStage(QRhiShaderStage.Type.Vertex, vert),
                              QRhiShaderStage(QRhiShaderStage.Type.Fragment, frag)])
        pipe.setVertexInputLayout(QRhiVertexInputLayout())
        pipe.setShaderResourceBindings(srb); pipe.setRenderPassDescriptor(rpd)
        print(f"[codegen-demo] pipeline ok={pipe.create()}", flush=True)
        self._pipe = pipe; self._uploaded = False; self._frames = 0
        QTimer.singleShot(1500, self._grab); self.update()

    def _vals(self, w, h):
        yaw, pitch, dist = math.radians(35), math.radians(28), 9.0
        cp = math.cos(pitch)
        pos = dist * np.array([cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch)])
        fwd = -pos / np.linalg.norm(pos)
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return {"u_term_count": self._tcount,
                "u_sel_count": self._selcount,
                "u_cull_enabled": 0, "u_grid_dim": 1,
                "u_grid_origin": (0.0, 0.0, 0.0), "u_grid_cell": (1.0, 1.0, 1.0),
                "u_group_count": self._gcount, "u_resolution": (float(w), float(h)),
                "u_camera_position": tuple(pos), "u_camera_target": (0, 0, 0),
                "u_camera_right": tuple(right), "u_camera_up": tuple(up),
                "u_focal_length": 1.5, "u_background_color": (0.07, 0.08, 0.10),
                "u_show_grid": 1, "u_grid_spacing": 1.0, "u_fb_y_up": self._fbup}

    def render(self, cb):
        rt = self.renderTarget(); sz = rt.pixelSize()
        w, h = max(sz.width(), 1), max(sz.height(), 1)
        rub = self._rhi.nextResourceUpdateBatch()
        if not self._uploaded:
            for buf, data in self._bufs:
                rub.uploadStaticBuffer(buf, data)
            self._uploaded = True
        rub.updateDynamicBuffer(self._ubo, 0, _std140(self._members, self._vals(w, h)))
        cb.beginPass(rt, QColor.fromRgbF(0.07, 0.08, 0.10, 1.0),
                     QRhiDepthStencilClearValue(1.0, 0), rub)
        cb.setGraphicsPipeline(self._pipe); cb.setViewport(QRhiViewport(0, 0, w, h))
        cb.setShaderResources(self._srb); cb.draw(3)
        cb.endPass()
        self._frames += 1; self.update()

    def _grab(self):
        path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/codegen_demo.png"
        img = self.grabFramebuffer()
        ok = (not img.isNull()) and img.save(path)
        print(f"[codegen-demo] frames={self._frames} screenshot={ok} -> {path}", flush=True)
        QTimer.singleShot(50, QApplication.instance().quit)


def main():
    kind = sys.argv[2] if len(sys.argv) > 2 else "intersection"
    per = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    app = QApplication(sys.argv[:1])
    w = Demo(); w.set_demo_scene(build_scene(kind, per)); w.resize(1280, 800); w.show()
    QTimer.singleShot(40000, app.quit); app.exec()


if __name__ == "__main__":
    main()
