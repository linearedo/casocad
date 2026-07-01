#!/usr/bin/env python3
"""End-to-end CAD workflow frame stress test.

This is intentionally a tool, not a normal pytest test. It opens the real
MainWindow, keeps orbiting the camera, and performs randomized but replayable CAD
operations while collecting frame, surface-artifact, and render-wait timings.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from core.scene import SceneDocument
from core.sdf import Box, PlacedSDF2D, SDFNode


DRAG_KINDS = (
    "sphere",
    "box",
    "cylinder",
    "cone",
    "capped_cone",
    "box_frame",
    "pyramid",
    "torus",
    "segment",
    "circle",
    "rectangle",
    "square",
    "rounded_rectangle",
    "ellipse",
    "regular_polygon",
)

POINT_KINDS = (
    "polyline",
    "quadratic_bezier_curve",
    "quadratic_bezier_polycurve",
    "polyline_tube",
    "quadratic_bezier_tube",
    "quadratic_bezier_surface",
    "polygon",
)

ALL_KINDS = DRAG_KINDS + POINT_KINDS

ROTATION_AXES = ("x", "y", "z")
REFERENCE_PLANES = ("xy", "xz", "yz")
DEFAULT_LOG_DIR = "perf_logs"

_INIT_RE = re.compile(r"(?:viewport surface )?qrhi: initialize backend=([^ ]+)")
_INIT_VALUE_RE = re.compile(r"\b(fb_y_up|clip_y_sign)=([+-]?\d+)")
_LARGE_SCENE_RE = re.compile(
    r"viewport-governor: mode=large exact=(\d+) total=(\d+) "
    r"no_blur=(True|False) reason=([^ ]*)"
)
_SURFACE_ARTIFACT_RE = re.compile(
    r"Render artifact built: total=(?P<total>[0-9.]+) ms, "
    r"surface=(?P<surface>[0-9.]+) ms, "
    r"render_wait=(?P<render_wait>[0-9.]+) ms, "
    r"tree_nodes=(?P<tree_nodes>\d+), "
    r"surface_resolution=(?P<surface_resolution>\d+), "
    r"surface_vertices=(?P<surface_vertices>\d+), "
    r"surface_triangles=(?P<surface_triangles>\d+), "
    r"large_scene=(?P<large_scene>yes|no), "
    r"objects=(?P<objects>\d+), "
    r"exact=(?P<exact>\d+), "
    r"no_blur=(?P<no_blur>True|False), "
    r"reason=(?P<reason>[^ ]*)"
)


def _now_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
    }


@dataclass
class Metrics:
    start: float = field(default_factory=time.perf_counter)
    jsonl_path: Path | None = None
    fsync_jsonl: bool = False
    _jsonl_handle: Any | None = field(default=None, init=False, repr=False)
    frame_stamps: list[float] = field(default_factory=list)
    render_ms: list[float] = field(default_factory=list)
    action_ms: list[float] = field(default_factory=list)
    artifact_ms: list[float] = field(default_factory=list)
    surface_ms: list[float] = field(default_factory=list)
    artifact_wait_ms: list[float] = field(default_factory=list)
    render_wait_ms: list[float] = field(default_factory=list)
    backend_info: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def open_stream(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = path
        self._jsonl_handle = path.open("w", encoding="utf-8", buffering=1)

    def close_stream(self) -> None:
        if self._jsonl_handle is None:
            return
        self._jsonl_handle.flush()
        self._jsonl_handle.close()
        self._jsonl_handle = None

    def event(self, kind: str, **data: Any) -> None:
        data = {"t_ms": round(_now_ms(self.start), 3), "kind": kind, **data}
        self.events.append(data)
        if self._jsonl_handle is not None:
            self._jsonl_handle.write(json.dumps(data, sort_keys=True) + "\n")
            self._jsonl_handle.flush()
            if self.fsync_jsonl:
                os.fsync(self._jsonl_handle.fileno())

    def record_log(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        init_match = _INIT_RE.search(msg)
        if init_match is not None:
            values = dict(_INIT_VALUE_RE.findall(msg))
            self.backend_info = {
                "backend": init_match.group(1),
                "fb_y_up": int(values.get("fb_y_up", "0")),
                "clip_y_sign": int(values.get("clip_y_sign", "1")),
            }
            self.event("qrhi_init", logger=record.name, **self.backend_info)
        large_scene_match = _LARGE_SCENE_RE.search(msg)
        if large_scene_match is not None:
            self.event(
                "large_scene",
                logger=record.name,
                exact=int(large_scene_match.group(1)),
                total=int(large_scene_match.group(2)),
                no_blur=large_scene_match.group(3) == "True",
                reason=large_scene_match.group(4),
            )
        artifact_match = _SURFACE_ARTIFACT_RE.search(msg)
        if artifact_match is not None:
            groups = artifact_match.groupdict()
            wait_ms = float(groups["render_wait"])
            self.artifact_ms.append(float(groups["total"]))
            surface_ms = float(groups["surface"])
            self.surface_ms.append(surface_ms)
            self.artifact_wait_ms.append(wait_ms)
            self.event(
                "artifact",
                total_ms=float(groups["total"]),
                surface_ms=surface_ms,
                render_wait_ms=wait_ms,
                tree_nodes=int(groups["tree_nodes"]),
                surface_resolution=int(groups["surface_resolution"]),
                surface_vertices=int(groups["surface_vertices"]),
                surface_triangles=int(groups["surface_triangles"]),
                large_scene=groups["large_scene"] == "yes",
                objects=int(groups["objects"]),
                exact=int(groups["exact"]),
                no_blur=groups["no_blur"] == "True",
                reason=groups["reason"],
            )

    def recent_frame_stats(self, count: int = 120) -> dict[str, float | int]:
        stamps = self.frame_stamps[-(count + 1):]
        intervals = [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])]
        stats = _stats(intervals)
        if len(stamps) > 1:
            stats["fps"] = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        else:
            stats["fps"] = 0.0
        return stats

    def summary(self, renderer: Any | None) -> dict[str, Any]:
        frame_intervals = [
            (b - a) * 1000.0 for a, b in zip(self.frame_stamps, self.frame_stamps[1:])
        ]
        elapsed = time.perf_counter() - self.start
        fps = (
            (len(self.frame_stamps) - 1) / (self.frame_stamps[-1] - self.frame_stamps[0])
            if len(self.frame_stamps) > 1
            else 0.0
        )
        render_state: dict[str, Any] = {}
        if renderer is not None:
            render_state = {"class": type(renderer).__name__}
        return {
            "elapsed_s": elapsed,
            "frames": len(self.frame_stamps),
            "fps": fps,
            "frame_interval_ms": _stats(frame_intervals),
            "render_call_ms": _stats(self.render_ms),
            "action_ms": _stats(self.action_ms),
            "artifact_ms": _stats(self.artifact_ms),
            "surface_ms": _stats(self.surface_ms),
            "artifact_wait_ms": _stats(self.artifact_wait_ms),
            "render_wait_ms": _stats(self.render_wait_ms),
            "backend": self.backend_info,
            "slow_frames": {
                "over_33ms": sum(1 for value in frame_intervals if value > 33.0),
                "over_50ms": sum(1 for value in frame_intervals if value > 50.0),
                "over_100ms": sum(1 for value in frame_intervals if value > 100.0),
            },
            "failures": len(self.failures),
            "renderer": render_state,
        }


class MetricsLogHandler(logging.Handler):
    def __init__(self, metrics: Metrics) -> None:
        super().__init__(logging.INFO)
        self.metrics = metrics

    def emit(self, record: logging.LogRecord) -> None:
        self.metrics.record_log(record)


class UltimateFrameRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(args.seed)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
        jsonl = args.jsonl
        if not jsonl:
            jsonl = str(Path(args.log_dir) / f"ultimate_frame_test_{self.run_id}.jsonl")
        self.jsonl_path = Path(jsonl)
        self.metrics = Metrics(fsync_jsonl=bool(args.fsync_jsonl))
        self.metrics.open_stream(self.jsonl_path)
        print(f"[ultimate-frame] streaming {self.jsonl_path}", flush=True)
        self.metrics_handler = MetricsLogHandler(self.metrics)
        logging.getLogger().addHandler(self.metrics_handler)
        self.app = QApplication(sys.argv[:1])
        self.window = MainWindow()
        self.window.resize(args.width, args.height)
        if args.start_empty:
            self.window.document = SceneDocument()
            self.window._publish_document(clear_selection=False, render=False)
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        self.created_handles: list[int] = []
        self.loop_handles: list[int] = []
        self.action_index = 0
        self.current_loop = 0
        # Outward translation applied to this cycle's moves. Its radius grows
        # with the cycle index so each cycle's objects land farther from the
        # origin than the previous cycle's (see _begin_cycle_move).
        self._cycle_move_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.todo: list[tuple[int, str]] = []
        self.done = False
        self._latest_render_version = -1
        self._await_render_version: int | None = None
        self._await_render_start = 0.0
        self._await_render_deadline = 0.0
        self._finish_after_render_wait = False
        self._render_wait_timeouts = 0
        self.exit_code = 0
        self.window.artifacts.render_ready.connect(self._on_tool_render_ready)
        self._install_render_probe()
        self.synthetic_wait_timer = QTimer()
        self.synthetic_wait_timer.timeout.connect(self._poll_synthetic_render_wait)
        self.orbit_timer = QTimer()
        self.orbit_timer.timeout.connect(self._orbit_camera)
        self.action_timer = QTimer()
        self.action_timer.timeout.connect(self._step)

    def close(self) -> None:
        root_logger = logging.getLogger()
        if self.metrics_handler in root_logger.handlers:
            root_logger.removeHandler(self.metrics_handler)
        self.metrics.close_stream()

    def _install_render_probe(self) -> None:
        renderer = self.window.viewport._renderer
        original_render = renderer.render

        def wrapped_render(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return original_render(*args, **kwargs)
            finally:
                end = time.perf_counter()
                if self.metrics.frame_stamps:
                    interval_ms = (end - self.metrics.frame_stamps[-1]) * 1000.0
                    if interval_ms >= self.args.slow_frame_ms:
                        self.metrics.event(
                            "slow_frame",
                            interval_ms=interval_ms,
                            render_ms=(end - start) * 1000.0,
                            action_index=self.action_index,
                            loop=self.current_loop,
                        )
                self.metrics.frame_stamps.append(end)
                self.metrics.render_ms.append((end - start) * 1000.0)

        renderer.render = wrapped_render

    def _on_tool_render_ready(self, artifact: Any) -> None:
        version = int(getattr(artifact, "version", -1))
        self._latest_render_version = max(self._latest_render_version, version)
        self.metrics.event(
            "render_ready",
            version=version,
            awaiting=self._await_render_version,
        )
        self._complete_render_wait_if_ready()

    def _start_render_wait(self, version: int, reason: str) -> None:
        if not self.args.wait_render:
            return
        self._await_render_version = int(version)
        self._await_render_start = time.perf_counter()
        self._await_render_deadline = (
            self._await_render_start + self.args.render_wait_timeout_ms / 1000.0
        )
        self.metrics.event(
            "render_wait_start",
            version=int(version),
            reason=reason,
            action_index=self.action_index,
            loop=self.current_loop,
        )
        self._complete_render_wait_if_ready()

    def _complete_render_wait_if_ready(self) -> bool:
        version = self._await_render_version
        if version is None or self._latest_render_version < version:
            return False
        wait_ms = (time.perf_counter() - self._await_render_start) * 1000.0
        self.metrics.render_wait_ms.append(wait_ms)
        self.metrics.event(
            "render_wait_done",
            version=version,
            wait_ms=wait_ms,
            latest_render_version=self._latest_render_version,
        )
        self._await_render_version = None
        if self._finish_after_render_wait:
            self._finish_after_render_wait = False
            self._schedule_finish_after_cooldown()
        return True

    def _check_render_wait_timeout(self) -> bool:
        version = self._await_render_version
        if version is None:
            return False
        if time.perf_counter() <= self._await_render_deadline:
            return False
        self._render_wait_timeouts += 1
        wait_ms = (time.perf_counter() - self._await_render_start) * 1000.0
        self.metrics.render_wait_ms.append(wait_ms)
        self.metrics.event(
            "render_wait_timeout",
            version=version,
            wait_ms=wait_ms,
            latest_render_version=self._latest_render_version,
            timeout_ms=self.args.render_wait_timeout_ms,
        )
        self._await_render_version = None
        if self._finish_after_render_wait:
            self._finish_after_render_wait = False
            self._schedule_finish_after_cooldown()
        return True

    def _poll_synthetic_render_wait(self) -> None:
        if self._await_render_version is None:
            self.synthetic_wait_timer.stop()
            return
        if self._complete_render_wait_if_ready():
            self.synthetic_wait_timer.stop()
            return
        if self._check_render_wait_timeout():
            self.synthetic_wait_timer.stop()

    def _orbit_camera(self) -> None:
        camera = self.window.viewport._camera
        camera.yaw += self.args.orbit_yaw_step
        camera.pitch = max(
            -1.2,
            min(1.2, camera.pitch + math.sin(time.perf_counter() * 0.7) * 0.0009),
        )
        viewport = self.window.viewport
        begin_interaction = getattr(viewport, "_begin_interaction", None)
        if callable(begin_interaction):
            begin_interaction()
        viewport.update()

    def _random_drag(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        base = np.asarray(
            (
                self.rng.uniform(-2.5, 2.5),
                self.rng.uniform(-2.5, 2.5),
                self.rng.uniform(-0.8, 0.8),
            ),
            dtype=np.float64,
        )
        size = np.asarray(
            (
                self.rng.uniform(0.25, 0.95),
                self.rng.uniform(0.25, 0.95),
                self.rng.uniform(0.25, 0.95),
            ),
            dtype=np.float64,
        )
        if self.rng.random() < 0.72:
            size[2] = 0.0
        end = base + size
        return tuple(float(v) for v in base), tuple(float(v) for v in end)

    def _random_points(self, kind: str) -> tuple[tuple[float, float, float], ...]:
        cx = self.rng.uniform(-2.2, 2.2)
        cy = self.rng.uniform(-2.2, 2.2)
        cz = self.rng.uniform(-0.35, 0.35)
        radius = self.rng.uniform(0.25, 0.85)
        templates = {
            "polyline": ((-1.0, -0.35, 0.0), (-0.15, 0.55, 0.0), (0.9, -0.15, 0.0)),
            "polyline_tube": ((-1.0, -0.3, -0.2), (-0.2, 0.55, 0.0), (0.9, -0.15, 0.25)),
            "quadratic_bezier_curve": ((-1.0, 0.0, 0.0), (0.0, 0.9, 0.0), (1.0, 0.0, 0.0)),
            "quadratic_bezier_polycurve": (
                (-1.0, 0.0, 0.0), (-0.55, 0.9, 0.0), (0.0, 0.05, 0.0),
                (0.55, -0.75, 0.0), (1.0, 0.0, 0.0),
            ),
            "quadratic_bezier_tube": (
                (-1.0, 0.0, -0.2), (-0.55, 0.85, -0.05), (0.0, 0.05, 0.1),
                (0.55, -0.75, 0.2), (1.0, 0.0, 0.35),
            ),
            "quadratic_bezier_surface": (
                (-1.0, 0.0, 0.0), (-0.55, 0.9, 0.0), (0.0, 0.05, 0.0),
                (0.55, -0.75, 0.0), (1.0, 0.0, 0.0),
            ),
            "polygon": ((-0.9, -0.6, 0.0), (0.8, -0.5, 0.0), (0.95, 0.55, 0.0), (-0.7, 0.75, 0.0)),
        }[kind]
        out: list[tuple[float, float, float]] = []
        for x, y, z in templates:
            jitter = 0.05 * radius
            out.append(
                (
                    cx + x * radius + self.rng.uniform(-jitter, jitter),
                    cy + y * radius + self.rng.uniform(-jitter, jitter),
                    cz + z * radius,
                )
            )
        return tuple(out)

    def _alive(self, handle: int) -> bool:
        try:
            self.window.document.node(handle)
        except KeyError:
            return False
        return True

    def _cleanup_handles(self) -> None:
        self.created_handles = [handle for handle in self.created_handles if self._alive(handle)]
        self.loop_handles = [handle for handle in self.loop_handles if self._alive(handle)]

    def _node_dimension(self, handle: int) -> int:
        node = self.window.document.node(handle)
        return int(getattr(node, "dimension", 0))

    def _node_sphere(self, handle: int) -> tuple[tuple[float, float, float], float] | None:
        """World-space (center, radius) of a 3D object's bounding box, or None when
        unavailable/unbounded."""
        node = self.window.document.node(handle)
        bbox = getattr(node, "bounding_box", None)
        if bbox is None:
            return None
        try:
            box = node.bounding_box()
        except Exception:  # noqa: BLE001 - some nodes have no finite box.
            return None
        lo = (box.x_min, box.y_min, box.z_min)
        hi = (box.x_max, box.y_max, box.z_max)
        if not all(math.isfinite(v) for v in (*lo, *hi)):
            return None
        center = tuple(0.5 * (a + b) for a, b in zip(lo, hi))
        return center, 0.5 * math.dist(lo, hi)

    def _maybe_combine_near(self, new_handle: int) -> int:
        """With probability ``--combine-prob``, combine a freshly created 3D object
        with the nearest existing 3D object whose bounding sphere overlaps it (so the
        boolean is never empty). Returns the resulting handle (combined or original).
        Same-loop objects share a ring and overlap; prior-loop objects are far and are
        skipped, so booleans always produce visible geometry."""
        if self.rng.random() >= self.args.combine_prob:
            return new_handle
        if self._node_dimension(new_handle) != 3:
            return new_handle
        new_sphere = self._node_sphere(new_handle)
        if new_sphere is None:
            return new_handle
        center, radius = new_sphere
        best: int | None = None
        best_dist: float | None = None
        for handle in self.created_handles:
            if handle == new_handle or not self._alive(handle):
                continue
            if self._node_dimension(handle) != 3:
                continue
            if not self.window.document.can_combine(new_handle, handle):
                continue
            other = self._node_sphere(handle)
            if other is None:
                continue
            dist = math.dist(center, other[0])
            if dist <= radius + other[1] and (best_dist is None or dist < best_dist):
                best, best_dist = handle, dist
        if best is None:
            return new_handle
        operation = self.rng.choice(("union", "intersection", "difference"))
        try:
            combined = self.window.document.combine(new_handle, best, operation)
        except Exception as exc:  # noqa: BLE001 - stress tool logs and continues.
            self._record_failure(f"combine={operation}", exc)
            return new_handle
        for consumed in (new_handle, best):
            if consumed in self.created_handles:
                self.created_handles.remove(consumed)
            if consumed in self.loop_handles:
                self.loop_handles.remove(consumed)
        self.created_handles.append(combined)
        self.loop_handles.append(combined)
        self.metrics.event(
            "combine",
            operation=operation,
            result_handle=combined,
            objects=len(self.window.document.objects),
        )
        return combined

    def _publish(self) -> None:
        version = int(self.window.document.version)
        self.window._publish_document(clear_selection=False, render=True)
        if self.window.document.objects:
            self._start_render_wait(version, "publish")

    def _draw(self, kind: str) -> int:
        if kind in POINT_KINDS:
            plane = self.rng.choice(REFERENCE_PLANES)
            return self.window.document.add_point_shape_from_world_points(
                kind,
                self._random_points(kind),
                plane,
            )
        start, end = self._random_drag()
        parameters: dict[str, float] = {}
        if kind == "capped_cone":
            parameters["top_diameter"] = self.rng.uniform(0.08, 0.45)
        if kind == "torus":
            parameters["minor_diameter"] = self.rng.uniform(0.05, 0.22)
        return self.window.document.add_primitive_from_drag(kind, start, end, parameters)

    def _begin_cycle_move(self, loop_index: int) -> None:
        """Pick this cycle's outward translation.

        The ring radius grows linearly with ``loop_index`` and is aimed in a
        per-cycle random direction (kept mostly in the XY plane so the orbiting
        camera keeps the spread in view). Cycle 0 stays on the origin so the
        first batch is the central cluster; every later cycle is pushed farther
        out than the one before it, keeping objects from piling up together.
        """
        if loop_index <= 0:
            self._cycle_move_center = (0.0, 0.0, 0.0)
            return
        radius = self.args.move_cycle_step * loop_index
        theta = self.rng.uniform(0.0, 2.0 * math.pi)
        phi = self.rng.uniform(-0.35, 0.35)
        horizontal = radius * math.cos(phi)
        self._cycle_move_center = (
            horizontal * math.cos(theta),
            horizontal * math.sin(theta),
            radius * math.sin(phi),
        )

    def _cycle_move_delta(self) -> tuple[float, float, float]:
        center_x, center_y, center_z = self._cycle_move_center
        jitter = self.args.move_jitter
        return (
            center_x + self.rng.uniform(-jitter, jitter),
            center_y + self.rng.uniform(-jitter, jitter),
            center_z + self.rng.uniform(-jitter, jitter),
        )

    def _move(self, handle: int) -> int:
        """Move a freshly created object onto the current cycle's outward ring."""
        return self.window.document.move_object(handle, self._cycle_move_delta())

    def _recap_move(self, handle: int) -> int:
        """Re-move a loop result in place: jitter only, so the recap churns the
        scene without compounding the cycle's outward push."""
        jitter = self.args.move_jitter
        delta = (
            self.rng.uniform(-jitter, jitter),
            self.rng.uniform(-jitter, jitter),
            self.rng.uniform(-jitter, jitter),
        )
        return self.window.document.move_object(handle, delta)

    def _rotate(self, handle: int) -> int:
        return self.window.document.rotate_object(
            handle,
            self.rng.choice(ROTATION_AXES),
            self.rng.uniform(-55.0, 55.0),
        )

    def _solid_from_2d(self, handle: int) -> int:
        node = self.window.document.node(handle)
        if not isinstance(node, PlacedSDF2D):
            return handle
        if self.rng.choice(("extrude", "revolve")) == "extrude":
            return self.window.document.solid_from_2d(
                [handle],
                "extrude",
                signed_height=self.rng.uniform(0.25, 1.4),
            )
        origin = tuple(float(value) for value in node.origin)
        axis_direction = tuple(float(value) for value in node.axis_v)
        radial_direction = tuple(float(value) for value in node.axis_u)
        return self.window.document.solid_from_2d(
            [handle],
            "revolve",
            revolve_axis_origin=origin,
            revolve_axis_direction=axis_direction,
            revolve_radial_direction=radial_direction,
            revolve_angle_degrees=self.rng.uniform(90.0, 360.0),
        )

    def _record_failure(self, action: str, exc: Exception) -> None:
        text = f"loop={self.current_loop} action={action}: {type(exc).__name__}: {exc}"
        self.metrics.failures.append(text)
        self.metrics.event("failure", action=action, error=text)
        print(f"[ultimate-frame] skip {text}", flush=True)

    def _run_kind(self, loop_index: int, kind: str) -> None:
        start = time.perf_counter()
        action_data: dict[str, Any] = {"loop": loop_index, "kind_name": kind}
        try:
            handle = self._draw(kind)
            action_data["draw_handle"] = handle
            handle = self._move(handle)
            action_data["move_handle"] = handle
            handle = self._rotate(handle)
            action_data["rotate_handle"] = handle
            if self._node_dimension(handle) == 2:
                solid = self._solid_from_2d(handle)
                action_data["solid_handle"] = solid
                handle = solid
            self.loop_handles.append(handle)
            self.created_handles.append(handle)
            pre_combine = handle
            handle = self._maybe_combine_near(handle)
            action_data["combined"] = handle != pre_combine
            self._publish()
            self.window.scene_tree.select_handle(handle)
            action_data["result_handle"] = handle
            action_data["objects"] = len(self.window.document.objects)
            action_data["tracked_handles"] = len(self.created_handles)
        except Exception as exc:  # noqa: BLE001 - stress tool logs and continues.
            self._record_failure(f"kind={kind}", exc)
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.metrics.action_ms.append(elapsed)
            self.metrics.event("action", ms=elapsed, **action_data)

    def _build_loop_todo(self, loop_index: int) -> list[tuple[int, str]]:
        kinds = list(ALL_KINDS)
        self.rng.shuffle(kinds)
        return [(loop_index, kind) for kind in kinds]

    def _move_loop_results(self) -> None:
        self._cleanup_handles()
        start = time.perf_counter()
        moved = 0
        for handle in tuple(self.loop_handles):
            if not self._alive(handle):
                continue
            try:
                self._recap_move(handle)
                moved += 1
            except Exception as exc:  # noqa: BLE001
                self._record_failure(f"recap_move={handle}", exc)
        self._publish()
        elapsed = (time.perf_counter() - start) * 1000.0
        self.metrics.action_ms.append(elapsed)
        self.metrics.event(
            "loop_recap_move",
            loop=self.current_loop,
            moved=moved,
            ms=elapsed,
            objects=len(self.window.document.objects),
        )
        print(
            f"[ultimate-frame] loop {self.current_loop + 1}/{self.args.nloop} "
            f"recap moved={moved} objects={len(self.window.document.objects)}",
            flush=True,
        )

    def _synthetic_large_scene_objects(self, count: int) -> list[SDFNode]:
        side = max(1, math.ceil(count ** (1.0 / 3.0)))
        spacing = float(self.args.large_scene_spacing)
        half = max(0.01, spacing * 0.22)
        origin = (side - 1) * spacing * -0.5
        objects: list[SDFNode] = []
        for index in range(count):
            ix = index % side
            iy = (index // side) % side
            iz = index // (side * side)
            objects.append(
                Box(
                    name=f"synthetic_box_{index + 1}",
                    object_id=index + 1,
                    center=(
                        origin + ix * spacing,
                        origin + iy * spacing,
                        origin + iz * spacing,
                    ),
                    half_size=(half, half, half),
                )
            )
        return objects

    def _run_synthetic_large_scene(self) -> None:
        count = int(self.args.large_scene_count)
        if count <= 0:
            return
        start = time.perf_counter()
        self.window.document = SceneDocument(
            objects=self._synthetic_large_scene_objects(count)
        )
        self.window.document.mark_changed()
        version = int(self.window.document.version)
        self.window._publish_document(clear_selection=False, render=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.metrics.action_ms.append(elapsed_ms)
        self.metrics.event(
            "synthetic_large_scene",
            objects=count,
            version=version,
            ms=elapsed_ms,
        )
        self.done = True
        if self.args.wait_render:
            self._finish_after_render_wait = True
            self._start_render_wait(version, "synthetic_large_scene")
            self.synthetic_wait_timer.start(20)
        else:
            self._schedule_finish_after_cooldown()

    def _step(self) -> None:
        if self._await_render_version is not None:
            if not self._complete_render_wait_if_ready():
                if not self._check_render_wait_timeout():
                    return
        if self.done:
            return
        if not self.todo:
            self._move_loop_results()
            self.current_loop += 1
            if self.current_loop >= self.args.nloop:
                self.done = True
                if self._await_render_version is not None:
                    self._finish_after_render_wait = True
                else:
                    self._schedule_finish_after_cooldown()
                return
            self.loop_handles = []
            self._begin_cycle_move(self.current_loop)
            self.todo = self._build_loop_todo(self.current_loop)
        loop_index, kind = self.todo.pop(0)
        self.action_index += 1
        self._cleanup_handles()
        self._run_kind(loop_index, kind)
        if self.action_index % self.args.report_every == 0:
            summary = self.metrics.summary(self.window.viewport._renderer)
            frame_stats = summary["frame_interval_ms"]
            recent = self.metrics.recent_frame_stats()
            self.metrics.event(
                "report",
                action_index=self.action_index,
                frames=summary["frames"],
                fps=summary["fps"],
                recent_fps=float(recent.get("fps", 0.0)),
                frame_p95_ms=float(frame_stats.get("p95", 0.0)),
                recent_p95_ms=float(recent.get("p95", 0.0)),
                artifact_count=summary["artifact_ms"]["count"],
                render_wait_timeouts=self._render_wait_timeouts,
                failures=summary["failures"],
            )
            print(
                "[ultimate-frame] "
                f"actions={self.action_index} frames={summary['frames']} "
                f"fps={summary['fps']:.1f} recent_fps={float(recent.get('fps', 0.0)):.1f} "
                f"frame_p95={float(frame_stats.get('p95', 0.0)):.2f}ms "
                f"recent_p95={float(recent.get('p95', 0.0)):.2f}ms "
                f"artifacts={summary['artifact_ms']['count']} "
                f"render_wait_timeouts={self._render_wait_timeouts} "
                f"failures={summary['failures']}",
                flush=True,
            )

    def start(self) -> int:
        self._begin_cycle_move(0)
        self.todo = self._build_loop_todo(0)
        self.metrics.event(
            "start",
            run_id=self.run_id,
            nloop=self.args.nloop,
            seed=self.args.seed,
            backend=os.environ.get("QRHI_BACKEND", ""),
            qpa=os.environ.get("QT_QPA_PLATFORM", ""),
            glx_vendor=os.environ.get("__GLX_VENDOR_LIBRARY_NAME", ""),
            nv_prime=os.environ.get("__NV_PRIME_RENDER_OFFLOAD", ""),
            vk_optimus=os.environ.get("__VK_LAYER_NV_optimus", ""),
            vk_icd=os.environ.get("VK_ICD_FILENAMES", ""),
            pid=os.getpid(),
            orbit=True,
            wait_render=bool(self.args.wait_render),
            kinds=len(ALL_KINDS),
            synthetic_large_scene_count=int(self.args.large_scene_count),
            move_cycle_step=float(self.args.move_cycle_step),
        )
        if self.args.large_scene_count > 0:
            QTimer.singleShot(self.args.start_delay_ms, self._run_synthetic_large_scene)
        else:
            self.orbit_timer.start(self.args.orbit_interval_ms)
            QTimer.singleShot(
                self.args.start_delay_ms,
                lambda: self.action_timer.start(self.args.action_interval_ms),
            )
        QTimer.singleShot(self.args.timeout_ms, self._timeout)
        return self.app.exec()

    def _schedule_finish_after_cooldown(self) -> None:
        self.action_timer.stop()
        self.orbit_timer.stop()
        self.synthetic_wait_timer.stop()
        QTimer.singleShot(self.args.cooldown_ms, self.finish)

    def _timeout(self) -> None:
        if self.done:
            return
        self.metrics.event("timeout", timeout_ms=self.args.timeout_ms)
        self.finish()

    def finish(self) -> None:
        self.done = True
        self.action_timer.stop()
        self.orbit_timer.stop()
        self.synthetic_wait_timer.stop()
        set_refinement_callback = getattr(
            self.window.viewport,
            "set_refinement_callback",
            None,
        )
        if callable(set_refinement_callback):
            set_refinement_callback(None)
        self.window.artifacts.shutdown(timeout_ms=10000)
        summary = self.metrics.summary(self.window.viewport._renderer)
        summary["render_wait_timeouts"] = self._render_wait_timeouts
        summary["run_id"] = self.run_id
        large_scene_count = sum(
            1 for event in self.metrics.events if event.get("kind") == "large_scene"
        )
        summary["large_scene_events"] = large_scene_count
        if self.args.assert_no_render_wait_timeouts and self._render_wait_timeouts:
            self.exit_code = 2
            self.metrics.event(
                "assertion_failed",
                assertion="no_render_wait_timeouts",
                render_wait_timeouts=self._render_wait_timeouts,
            )
        if self.args.assert_large_scene and large_scene_count <= 0:
            self.exit_code = 2
            self.metrics.event(
                "assertion_failed",
                assertion="large_scene_activated",
            )
        if self.args.assert_qrhi_frames and int(summary["frames"]) <= 0:
            self.exit_code = 2
            self.metrics.event(
                "assertion_failed",
                assertion="qrhi_frames_rendered",
                frames=int(summary["frames"]),
            )
        self.metrics.event("summary", **summary)
        if summary["frames"] == 0:
            print(
                "[ultimate-frame] warning: no QRhi frames were rendered. "
                "Run without QT_QPA_PLATFORM=offscreen in a real desktop session "
                "to collect FPS metrics.",
                flush=True,
            )
        path = self.metrics.jsonl_path
        if path is None:
            path = self.jsonl_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for event in self.metrics.events:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
        self.close()
        print(f"[ultimate-frame] wrote {path}", flush=True)
        print("[ultimate-frame] summary", json.dumps(summary, indent=2, sort_keys=True), flush=True)
        self.app.exit(self.exit_code)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nloop", type=int, default=5, help="Full cycles over all SDF kinds.")
    parser.add_argument("--seed", type=int, default=1, help="Replayable random seed.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--start-empty", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--action-interval-ms",
        type=int,
        default=1000,
        help="Delay between SDF creation actions. Default: 1000 ms.",
    )
    parser.add_argument(
        "--move-cycle-step",
        type=float,
        default=3.0,
        help=(
            "Extra outward radius added per loop cycle when moving objects, so "
            "each cycle's geometry lands farther from the origin than the "
            "previous cycle's. Set 0 to keep every cycle near the origin."
        ),
    )
    parser.add_argument(
        "--move-jitter",
        type=float,
        default=0.35,
        help="Random per-axis jitter (units) added to every move.",
    )
    parser.add_argument("--orbit-interval-ms", type=int, default=16)
    parser.add_argument("--orbit-yaw-step", type=float, default=0.012)
    parser.add_argument("--start-delay-ms", type=int, default=1500)
    parser.add_argument("--cooldown-ms", type=int, default=2500)
    parser.add_argument("--timeout-ms", type=int, default=120000)
    parser.add_argument("--report-every", type=int, default=8)
    parser.add_argument(
        "--combine-prob",
        type=float,
        default=0.4,
        help=(
            "Probability that a freshly created 3D object is combined (random "
            "union/intersection/difference) with an existing object whose bounding "
            "sphere overlaps it. Only overlapping operands are combined, so the "
            "boolean is never empty. Set 0 to disable booleans."
        ),
    )
    parser.add_argument(
        "--large-scene-count",
        type=int,
        default=0,
        help=(
            "Create one synthetic sharp-box scene with this many objects instead "
            "of running the interactive action loop. Use 1000+ for proxy stress."
        ),
    )
    parser.add_argument(
        "--large-scene-spacing",
        type=float,
        default=0.42,
        help="Grid spacing for --large-scene-count synthetic objects.",
    )
    parser.add_argument(
        "--slow-frame-ms",
        type=float,
        default=50.0,
        help="Emit a JSONL slow_frame event when frame interval exceeds this.",
    )
    parser.add_argument(
        "--boolean-mode",
        choices=("renderable", "all", "off"),
        default="off",
        help=(
            "Deprecated no-op. Boolean operations are not inserted by the "
            "ultimate frame stress loop."
        ),
    )
    parser.add_argument(
        "--wait-render",
        dest="wait_render",
        action="store_true",
        default=True,
        help=(
            "Wait for the async render artifact after each action before issuing "
            "the next action. This measures incremental object visibility instead "
            "of coalescing every edit into the final scene."
        ),
    )
    parser.add_argument(
        "--no-wait-render",
        dest="wait_render",
        action="store_false",
        help="Keep issuing actions on schedule even if render artifacts are pending.",
    )
    parser.add_argument(
        "--render-wait-timeout-ms",
        type=int,
        default=6000,
        help="Maximum time to wait for a render artifact before continuing.",
    )
    parser.add_argument(
        "--assert-no-render-wait-timeouts",
        action="store_true",
        help="Exit nonzero if any render wait timeout is recorded.",
    )
    parser.add_argument(
        "--assert-large-scene",
        action="store_true",
        help="Exit nonzero if no large_scene event is recorded.",
    )
    parser.add_argument(
        "--assert-qrhi-frames",
        action="store_true",
        help="Exit nonzero if no QRhi frames are rendered.",
    )
    parser.add_argument(
        "--jsonl",
        default="",
        help="JSONL output path. Overrides --log-dir.",
    )
    parser.add_argument(
        "--fsync-jsonl",
        action="store_true",
        help="fsync after every JSONL event. Slower, but useful for crash/OOM forensics.",
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help=(
            "Directory for automatic JSONL output. "
            "Default: perf_logs/ultimate_frame_test_<run_id>.jsonl"
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    if args.nloop < 1:
        raise SystemExit("--nloop must be >= 1")
    if args.large_scene_count < 0:
        raise SystemExit("--large-scene-count must be >= 0")
    if args.large_scene_count > 65_535:
        raise SystemExit("--large-scene-count must be <= 65535")
    runner: UltimateFrameRunner | None = None
    try:
        runner = UltimateFrameRunner(args)
        raise SystemExit(runner.start())
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
