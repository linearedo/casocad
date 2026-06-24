#!/usr/bin/env python3
"""Real-GPU orbit-FPS benchmark for the QRhi viewport.

Why this exists: headless / CPU logs are misleading for viewport performance
(they showed millisecond cull-grid builds while the GPU was at ~1 FPS). The only
trustworthy measure is the real app, on the real GPU, with the camera MOVING.
This tool seeds a chosen scene, shows the actual `QRhiViewportWidget`, orbits the
camera continuously, and reports the true render FPS measured from real frame
timestamps (it wraps `renderer.render`, a plain Python call — the QRhiWidget
`render` virtual cannot be wrapped at the instance level).

Run it via the NVIDIA offload env (same as ./outbin/casocad-nvidia):

    __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
    __VK_LAYER_NV_optimus=NVIDIA_only QT_OPENGL=desktop \
    .venv/bin/python tools/fps_bench.py --scene intersection --count 19

Scenes (built from spheres so cost is geometric, not parsing):
    union            min over leaves                       (1 cull group)
    difference       a union base carved by half the leaves (1 group + carves)
    carve-into-union large union then many carves           (1 group + carves)
    intersection     max(union A, union B)                  (2 cull groups)
    intersection3    A ∩ B ∩ C                              (3 cull groups)

A window opens briefly and closes itself. `--screenshot PATH` saves the final
frame (also a visual correctness check). Shrinking --width/--height emulates
internal resolution scaling (FPS scales with pixel count).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PySide6.QtCore import QTimer                       # noqa: E402
from PySide6.QtWidgets import QApplication              # noqa: E402

from core.render_ir import RenderIR, RenderIRNode       # noqa: E402
from app.viewport.renderers.qrhi.viewport import QRhiViewportWidget  # noqa: E402


def _sphere(oid: int, x: float, y: float, z: float, r: float) -> RenderIRNode:
    return RenderIRNode(kind="sphere", object_id=oid, dimension=3,
                        children=(), params=(x, y, z, r))


def _chain(nodes: list, kind: str, leaves: list[int]) -> int:
    cur = leaves[0]
    for tool in leaves[1:]:
        nodes.append(RenderIRNode(kind=kind, object_id=0, dimension=3,
                                  children=(cur, tool), params=()))
        cur = len(nodes) - 1
    return cur


def build_scene(scene: str, count: int, seed: int = 7) -> RenderIR:
    """`count` is the per-group leaf count (so intersection has 2x leaves)."""
    random.seed(seed)
    nodes: list[RenderIRNode] = []

    def cluster(n: int) -> list[int]:
        out = []
        for _ in range(n):
            nodes.append(_sphere(len(nodes) + 1,
                                 random.uniform(-2.2, 2.2),
                                 random.uniform(-2.2, 2.2),
                                 random.uniform(-2.2, 2.2), 1.2))
            out.append(len(nodes) - 1)
        return out

    if scene == "union":
        root = _chain(nodes, "union", cluster(count * 2))
    elif scene == "difference":
        a = cluster(count)
        base = _chain(nodes, "union", a)
        cur = base
        for tool in cluster(count):
            nodes.append(RenderIRNode(kind="difference", object_id=0,
                                      dimension=3, children=(cur, tool),
                                      params=()))
            cur = len(nodes) - 1
        root = cur
    elif scene == "carve-into-union":
        base = _chain(nodes, "union", cluster(count * 2))
        cur = base
        for tool in cluster(max(1, count // 2)):
            nodes.append(RenderIRNode(kind="difference", object_id=0,
                                      dimension=3, children=(cur, tool),
                                      params=()))
            cur = len(nodes) - 1
        root = cur
    elif scene == "intersection":
        a = _chain(nodes, "union", cluster(count))
        b = _chain(nodes, "union", cluster(count))
        nodes.append(RenderIRNode(kind="intersection", object_id=0, dimension=3,
                                  children=(a, b), params=()))
        root = len(nodes) - 1
    elif scene == "intersection3":
        kids = [_chain(nodes, "union", cluster(count)) for _ in range(3)]
        nodes.append(RenderIRNode(kind="intersection", object_id=0, dimension=3,
                                  children=tuple(kids), params=()))
        root = len(nodes) - 1
    else:
        raise SystemExit(f"unknown scene: {scene}")
    return RenderIR(nodes=tuple(nodes), root_indices=(root,),
                    component_indices=())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="intersection",
                    choices=["union", "difference", "carve-into-union",
                             "intersection", "intersection3"])
    ap.add_argument("--count", type=int, default=19, help="per-group leaf count")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--warmup", type=float, default=12.0,
                    help="seconds for shader compile + settle before measuring")
    ap.add_argument("--measure", type=float, default=6.0, help="measure seconds")
    ap.add_argument("--screenshot", default=None, help="save final frame to PATH")
    args = ap.parse_args()

    app = QApplication(sys.argv[:1])
    w = QRhiViewportWidget()
    w.resize(args.width, args.height)
    ir = build_scene(args.scene, args.count)
    w.set_scene(ir)
    w.frame_target((0.0, 0.0, 0.0), 9.0)
    w.show()
    w.raise_()
    w.activateWindow()
    print(f"[fps_bench] scene={args.scene} ir_nodes={len(ir.nodes)} "
          f"win={args.width}x{args.height}", flush=True)

    state = {"measure": False}
    stamps: list[float] = []
    _orig = w._renderer.render

    def _wrapped(*a, **k):
        if state["measure"]:
            stamps.append(time.perf_counter())
        return _orig(*a, **k)

    w._renderer.render = _wrapped

    def drive():
        w._yaw += 0.012             # continuous orbit (fixed-scene FPS is meaningless)
        w._begin_interaction()
        w.update()

    driver = QTimer()
    driver.timeout.connect(drive)
    driver.start(5)

    QTimer.singleShot(int(args.warmup * 1000),
                      lambda: state.update(measure=True))

    def done():
        if len(stamps) >= 2:
            fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
            print(f"[fps_bench] ORBIT FPS={fps:6.1f}  frames={len(stamps)}",
                  flush=True)
        else:
            print(f"[fps_bench] too few frames ({len(stamps)}) — GPU may have "
                  "failed to compile/link the shader", flush=True)
        if args.screenshot:
            try:
                w.grabFramebuffer().save(args.screenshot)
                print(f"[fps_bench] saved {args.screenshot}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[fps_bench] screenshot failed: {exc}", flush=True)
        app.quit()

    QTimer.singleShot(int((args.warmup + args.measure) * 1000), done)
    QTimer.singleShot(int((args.warmup + args.measure + 4) * 1000), app.quit)
    app.exec()


if __name__ == "__main__":
    main()
