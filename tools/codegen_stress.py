#!/usr/bin/env python3
"""Hard stress + timing harness for the thin-codegen renderer.

Two subcommands:

  create  — build a scene and time the FULL creation pipeline stage-by-stage
            (render_ir build -> flatten_terms -> signature -> capacity ->
            serialize_scene -> emit shader -> vulkanize -> qsb compile -> host
            buffer pack). Headless-safe (no window, no GPU): measures the CPU +
            offline-compile cost, the thing that gates "freeze on edit".

  fps     — open the real QRhiViewportWidget, orbit the camera, and report true
            on-GPU frame-time stats (mean/median/p95/min/max + FPS) plus the
            time-to-first-frame (the real driver pipeline compile). Codegen is the
            only renderer. Run under the NVIDIA offload env for the dGPU; add
            QRHI_BACKEND=vulkan|opengl to pick a backend (else platform default):

    __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
    __VK_LAYER_NV_optimus=NVIDIA_only \
    .venv/bin/python tools/codegen_stress.py fps --scene alltypes

Scenes:
  alltypes      one of EVERY codegen leaf kind (all 3D prims, placed sections,
                tubes, open curves, extrude/revolve/placed-profile 2D+1D) unioned,
                plus a region selector — the widest specialized shader (max compile).
  carveunion N  union of N carved solids (term-DNF; 1 group, linear).
  polytope N    intersection of N planes/solids (N groups -> adaptive g[] bucket).
  union N       union of N*2 solids (1 group, N leaves).
  mixed N       N instances cycling through every leaf kind, unioned (heavy data).
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.render_ir import RenderIR, RenderIRNode                       # noqa: E402

_IDB = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)   # axis_u, axis_v, normal


def _basis(pos):
    return tuple(float(c) for c in pos) + _IDB           # origin@0 + u,v,n (12 floats)


# --- one-of-each leaf builders. Each appends node(s) to `nodes`, returns root idx.
def _add(nodes, kind, params, dim=3, children=(), oid=None, flags=0, bound=None):
    # `bound` = world (cx,cy,cz,r) enclosing sphere. The real app fills this from
    # the SDF's bounding_box(); hand-built feature leaves (tubes/profiles) have no
    # closed-form fallback, so we set it here — otherwise ONE unbounded term makes
    # the whole cull grid bail to brute force. Must CONTAIN the geometry (over-
    # estimate is safe, just less selective).
    nodes.append(RenderIRNode(
        kind=kind, object_id=(len(nodes) + 1 if oid is None else oid),
        dimension=dim, children=tuple(children), params=tuple(params), flags=flags,
        bound=bound))
    return len(nodes) - 1


def _circle_profile(nodes):                              # 2D profile sub-graph leaf
    return _add(nodes, "profile_circle_2d", (0.0, 0.0, 0.6), dim=2, oid=0)


def _segment_profile(nodes):                             # 1D profile sub-graph leaf
    return _add(nodes, "profile_segment_1d", (0.0, 0.5), dim=1, oid=0)


def _every_leaf(nodes, pos):
    """Append one node of every codegen leaf kind near `pos`; return their indices."""
    x, y, z = pos
    b = _basis(pos)
    out = []
    out.append(_add(nodes, "sphere", (x, y, z, 0.8)))
    out.append(_add(nodes, "box", b + (0.6, 0.6, 0.6)))
    out.append(_add(nodes, "cylinder", b + (0.6, 0.8)))
    out.append(_add(nodes, "cone", b + (0.7, 0.8)))
    out.append(_add(nodes, "capped_cone", b + (0.7, 0.3, 0.8)))   # r1, r2, h
    out.append(_add(nodes, "box_frame", b + (0.6, 0.6, 0.6, 0.08)))
    out.append(_add(nodes, "pyramid", b + (0.6, 0.8)))
    out.append(_add(nodes, "torus", b + (0.6, 0.22)))
    # placed 2D analytic sections (origin/u/v/n @0..11, profile @12+)
    out.append(_add(nodes, "placed_circle_2d", b + (0.0, 0.0, 0.7), dim=2))
    out.append(_add(nodes, "placed_rectangle_2d", b + (0.0, 0.0, 0.7, 0.5), dim=2))
    out.append(_add(nodes, "placed_square_2d", b + (0.0, 0.0, 0.6), dim=2))
    out.append(_add(nodes, "placed_rounded_rectangle_2d",
                    b + (0.0, 0.0, 0.7, 0.5, 0.15), dim=2))
    out.append(_add(nodes, "placed_ellipse_2d", b + (0.0, 0.0, 0.8, 0.5), dim=2))
    # tubes (3 pts inline + radius, inner, flat_caps). Span pos..pos+(0,2,0.4).
    pts = (x, y, z, x + 0.4, y + 1.0, z, x, y + 2.0, z + 0.4)
    tube_b = (x + 0.1, y + 1.0, z + 0.1, 1.9)
    out.append(_add(nodes, "polyline_tube", pts + (0.25, 0.0, 0.0), dim=1, bound=tube_b))
    out.append(_add(nodes, "bezier_tube", pts + (0.25, 0.0, 0.0), dim=1, bound=tube_b))
    # placed open curves (points @12+ as 2D pairs)
    crv_b = (x, y, z, 1.6)
    out.append(_add(nodes, "placed_polyline_2d",
                    b + (0.0, 0.0, 0.5, 0.4, 1.0, -0.2), dim=2, bound=crv_b))
    out.append(_add(nodes, "placed_bezier_curve_2d",
                    b + (0.0, 0.0, 0.5, 0.6, 1.0, 0.0), dim=2, bound=crv_b))
    # profile-VM leaves (need a child profile sub-graph)
    prof_b = (x, y, z, 1.8)
    c2 = _circle_profile(nodes)
    out.append(_add(nodes, "extrude_profile_2d", b + (1.0, z), children=(c2,), bound=prof_b))
    c2b = _circle_profile(nodes)
    out.append(_add(nodes, "placed_profile_2d", b, dim=2, children=(c2b,), bound=prof_b))
    c2c = _circle_profile(nodes)
    rev = b + tuple(pos) + (0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0, 2.0 * math.pi)   # revolve basis + full angle
    out.append(_add(nodes, "revolve_profile_2d", rev, children=(c2c,), bound=prof_b))
    s1 = _segment_profile(nodes)
    out.append(_add(nodes, "placed_profile_1d",
                    (x, y, z) + (1.0, 0.0, 0.0), children=(s1,), bound=prof_b))
    return out


def build_scene(scene: str, n: int) -> RenderIR:
    nodes: list[RenderIRNode] = []
    comps: tuple = ()

    def cluster(count, r=1.0, spread=2.4):
        import random
        rng = random.Random(7)
        out = []
        for _ in range(count):
            out.append(_add(nodes, "sphere", (rng.uniform(-spread, spread),
                            rng.uniform(-spread, spread),
                            rng.uniform(-spread, spread), r)))
        return out

    def chain(op, idx):
        cur = idx[0]
        for t in idx[1:]:
            cur = _add(nodes, op, (), children=(cur, t), oid=0)
        return cur

    if scene == "alltypes":
        # a grid of every-leaf clusters so the scene is geometrically spread out
        leaves = []
        side = max(1, int(math.isqrt(max(1, n))))
        for i in range(n):
            gx, gy = (i % side) * 3.0 - side * 1.5, (i // side) * 3.0 - side * 1.5
            leaves += _every_leaf(nodes, (gx, gy, 0.0))
        # tag a region selector over a sphere volume (Layer 2 exercise)
        vol = _add(nodes, "sphere", (0.0, 0.0, 0.0, 3.0))
        sel = _add(nodes, "region_selector", (0.05,), children=(vol,), oid=1, flags=1)
        comps = (sel,)
        root = chain("union", leaves)
    elif scene == "carveunion":
        import random
        rng = random.Random(11)
        idx = []
        for _ in range(n):
            cx, cy, cz = (rng.uniform(-3, 3) for _ in range(3))
            solid = _add(nodes, "sphere", (cx, cy, cz, 1.1))
            hole = _add(nodes, "sphere", (cx + 0.6, cy, cz, 0.7))
            idx.append(_add(nodes, "difference", (), children=(solid, hole), oid=0))
        root = chain("union", idx)
    elif scene == "polytope":
        # intersection of N solids -> N groups (adaptive g[] bucket stress)
        import random
        rng = random.Random(13)
        idx = [_add(nodes, "sphere", (rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6),
                    rng.uniform(-0.6, 0.6), 3.0)) for _ in range(n)]
        root = chain("intersection", idx)
    elif scene == "union":
        root = chain("union", cluster(n * 2))
    elif scene == "mixed":
        # n clusters of every-leaf, spread 4 units apart -> spatially distributed
        # feature-heavy assembly (every cluster is bounded, so it culls).
        leaves = []
        side = max(1, int(math.isqrt(max(1, n))))
        for i in range(n):
            gx = (i % side) * 4.0 - side * 2.0
            gy = (i // side) * 4.0 - side * 2.0
            leaves += _every_leaf(nodes, (gx, gy, 0.0))
        root = chain("union", leaves)
    else:
        raise SystemExit(f"unknown scene: {scene}")
    return RenderIR(nodes=tuple(nodes), root_indices=(root,), component_indices=comps)


# --------------------------------------------------------------------------- create
def _time(fn, repeat):
    best = math.inf
    out = None
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best * 1e3, out          # ms, result


def cmd_create(args) -> None:
    from core.gpu_codegen import (flatten_terms, scene_structure_signature,
                                  group_capacity, emit_fragment_shader, supported)
    from core.gpu_scene import serialize_scene
    from app.viewport.renderers.qrhi.vulkanize import vulkanize, uniform_block_members
    from app.viewport.renderers.qrhi.renderer import _bake

    r = args.repeat
    tb, ir = _time(lambda: build_scene(args.scene, args.n), 1)
    print(f"\n=== create  scene={args.scene} n={args.n}  "
          f"ir_nodes={len(ir.nodes)}  supported={supported(ir)} ===")
    groups = flatten_terms(ir)
    ng = len(groups) if groups else 0
    nterms = sum(len(g) for g in groups) if groups else 0
    print(f"  flattened: groups={ng}  terms={nterms}  "
          f"capacity=g[{group_capacity(ir)}]  kinds={len(scene_structure_signature(ir))}")

    rows = [("build render_ir", tb)]
    rows.append(("flatten_terms", _time(lambda: flatten_terms(ir), r)[0]))
    rows.append(("scene_signature", _time(lambda: scene_structure_signature(ir), r)[0]))
    rows.append(("group_capacity", _time(lambda: group_capacity(ir), r)[0]))
    rows.append(("serialize_scene", _time(lambda: serialize_scene(ir), r)[0]))
    t_emit, gl_src = _time(lambda: emit_fragment_shader(ir), r)
    rows.append(("emit_fragment_shader", t_emit))
    t_vk, vsrc = _time(lambda: vulkanize(gl_src), r)
    rows.append(("vulkanize", t_vk))
    rows.append(("uniform_block_members", _time(lambda: uniform_block_members(gl_src), r)[0]))
    t_bake, _ = _time(lambda: _bake("frag", vsrc), 1)     # qsb subprocess: once
    rows.append(("qsb compile (offline)", t_bake))

    total = sum(ms for _, ms in rows)
    print(f"  shader source: {len(gl_src):,} chars  ({gl_src.count(chr(10))} lines)")
    print("  ---- stage timings (min of "
          f"{r}, qsb once) -------------------------")
    for name, ms in rows:
        bar = "#" * min(40, int(ms / max(total, 1e-9) * 40))
        print(f"    {name:24s} {ms:8.2f} ms  {bar}")
    print(f"    {'TOTAL':24s} {total:8.2f} ms")


# ----------------------------------------------------------------------------- fps
def cmd_fps(args) -> None:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from app.viewport.renderers.qrhi.viewport import QRhiViewportWidget

    app = QApplication(sys.argv[:1])
    w = QRhiViewportWidget()
    w.resize(args.width, args.height)
    ir = build_scene(args.scene, args.n)
    # Report whether codegen will cull or brute-force this scene (and warn if a
    # large brute-force scene risks an OOM/watchdog kill). Codegen is the only
    # renderer now, so this always applies.
    from core.gpu_codegen import flatten_terms
    from core.gpu_cull import build_term_grid, leaf_bounds
    import numpy as np
    groups = flatten_terms(ir)
    if groups:
        pos = np.array([leaf for g in groups for leaf, _c in g], dtype=np.uint32)
        grid = build_term_grid(pos, leaf_bounds(ir), dim=16)
        if grid is None:
            print(f"[stress-fps] WARNING: {len(pos)} terms but cull grid bailed "
                  "(an unbounded leaf) -> BRUTE FORCE; large N may be killed.",
                  flush=True)
        else:
            ne = (grid[4] > 0).sum()
            print(f"[stress-fps] cull ON: {len(pos)} terms, "
                  f"~{grid[5].size / max(1, ne):.1f} terms/cell (max {int(grid[4].max())})",
                  flush=True)
    t_set = time.perf_counter()
    w.set_scene(ir)
    w.frame_target((0.0, 0.0, 0.0), 12.0)
    w.show(); w.raise_(); w.activateWindow()
    print(f"[stress-fps] scene={args.scene} n={args.n} ir_nodes={len(ir.nodes)} "
          f"win={args.width}x{args.height} (backend logged by the renderer at init)",
          flush=True)

    state = {"measure": False, "first": None}
    stamps: list[float] = []
    _orig = w._renderer.render

    def _wrapped(*a, **k):
        now = time.perf_counter()
        if state["first"] is None:
            state["first"] = now
            print(f"[stress-fps] time-to-first-frame "
                  f"(real pipeline compile) = {(now - t_set) * 1e3:.0f} ms", flush=True)
        if state["measure"]:
            stamps.append(now)
        return _orig(*a, **k)

    w._renderer.render = _wrapped

    def drive():
        w._yaw += 0.012
        w._begin_interaction()
        w.update()

    driver = QTimer(); driver.timeout.connect(drive); driver.start(5)
    QTimer.singleShot(int(args.warmup * 1000), lambda: state.update(measure=True))

    def done():
        if len(stamps) >= 3:
            dts = [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:])]
            dts_sorted = sorted(dts)
            p95 = dts_sorted[min(len(dts_sorted) - 1, int(len(dts_sorted) * 0.95))]
            fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
            print(f"[stress-fps] FPS={fps:6.1f}  frames={len(stamps)}", flush=True)
            print(f"[stress-fps] frame ms  mean={statistics.mean(dts):.2f} "
                  f"median={statistics.median(dts):.2f}  p95={p95:.2f}  "
                  f"min={min(dts):.2f}  max={max(dts):.2f}", flush=True)
        else:
            print(f"[stress-fps] too few frames ({len(stamps)}) — shader may have "
                  "failed to compile/link", flush=True)
        if args.screenshot:
            try:
                w.grabFramebuffer().save(args.screenshot)
                print(f"[stress-fps] saved {args.screenshot}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[stress-fps] screenshot failed: {exc}", flush=True)
        app.quit()

    QTimer.singleShot(int((args.warmup + args.measure) * 1000), done)
    QTimer.singleShot(int((args.warmup + args.measure + 4) * 1000), app.quit)
    app.exec()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("create", help="time the full creation pipeline (headless)")
    sc.add_argument("--scene", default="alltypes")
    sc.add_argument("--n", type=int, default=1)
    sc.add_argument("--repeat", type=int, default=20)
    sc.set_defaults(func=cmd_create)
    sf = sub.add_parser("fps", help="orbit the real viewport and time frames")
    sf.add_argument("--scene", default="alltypes")
    sf.add_argument("--n", type=int, default=1)
    sf.add_argument("--width", type=int, default=1280)
    sf.add_argument("--height", type=int, default=800)
    sf.add_argument("--warmup", type=float, default=12.0)
    sf.add_argument("--measure", type=float, default=6.0)
    sf.add_argument("--screenshot", default=None)
    sf.set_defaults(func=cmd_fps)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
