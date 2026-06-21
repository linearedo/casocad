"""
flow_sim_view.py – OpenGL flow visualisation widget.

Architecture
------------
_SimThread   runs FlowMapSimulation.step() in a background QThread as fast
             as the hardware allows.  After each step it copies positions and
             (if enabled) streamline vertices into a lock-protected buffer.

FlowSimView  is a QOpenGLWidget driven by a 16 ms QTimer.  Each tick it
             pops the latest snapshot from _SimThread and uploads it to the
             GPU—no simulation work happens on the main thread.

Result: render at 60 fps is completely decoupled from simulation speed.
"""
from __future__ import annotations

from collections import deque
import logging
import threading
import time

import moderngl
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPoint, QThread, QTimer, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from app.flow_sim_map import FlowMapSimulation
from app.viewport.camera import OrbitCamera
from core.sdf.base import BoundingBox3D

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color defaults (imported by flow_sim_panel)
# ---------------------------------------------------------------------------

DEFAULT_BACKGROUND = QColor(11, 16, 24)
DEFAULT_LATTICE_COLOR = QColor(60, 90, 120)
DEFAULT_BOUNDARY_COLOR = QColor(172, 190, 208)
DEFAULT_UNTAGGED_BOUNDARY_COLOR = QColor(248, 250, 252)
DEFAULT_PARTICLE_COLOR = QColor(50, 228, 255)
DEFAULT_STREAMLINE_COLOR = QColor(64, 142, 255)
TAG_PALETTE = (
    QColor(244, 114, 22),
    QColor(59, 130, 246),
    QColor(16, 185, 129),
    QColor(234, 179, 8),
)

# Render budget caps
_MAX_PARTICLES = 40_000
_MAX_SL_SEGS = 20_000
_TARGET_MS = 16.0  # 60 fps
_PARTICLE_MOVE_EPS = 1e-6
_SIM_FPS = 60.0
_SIM_DT = 1.0 / _SIM_FPS
_STATIC_FRAME_WARN_EVERY = 120
_FRAME_COMPARE_LOG = 120
_STATIC_MOVED_RATIO_THRESHOLD = 0.02

# ---------------------------------------------------------------------------
# GLSL shaders
# ---------------------------------------------------------------------------

_VERT_STATIC = """
#version 330
uniform mat4 u_mvp;
uniform float u_sz;
in vec3 in_pos;
in vec3 in_col;
out vec3 v_col;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    gl_PointSize = u_sz;
    v_col = in_col;
}
"""
_FRAG_STATIC = """
#version 330
in vec3 v_col;
out vec4 f;
void main() { f = vec4(v_col, 1.0); }
"""
_VERT_DYN = """
#version 330
uniform mat4 u_mvp;
uniform float u_sz;
in vec3 in_pos;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    gl_PointSize = u_sz;
}
"""
_VERT_LINE = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
void main() { gl_Position = u_mvp * vec4(in_pos, 1.0); }
"""
_FRAG_UNI = """
#version 330
uniform vec3 u_col;
out vec4 f;
void main() { f = vec4(u_col, 1.0); }
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb(c: QColor) -> tuple[float, float, float]:
    return (c.redF(), c.greenF(), c.blueF())


def _rgb32(c: QColor) -> NDArray[np.float32]:
    return np.array(_rgb(c), dtype=np.float32)


def _subsample(a: NDArray, n: int) -> NDArray:
    if len(a) <= n:
        return a
    return a[np.linspace(0, len(a) - 1, n, dtype=np.intp)]


# ---------------------------------------------------------------------------
# Background simulation thread
# ---------------------------------------------------------------------------

class _SimThread(QThread):
    """
    Owns the FlowMapSimulation and runs step() in a tight background loop.

    Thread-safety contract
    ----------------------
    * self._sim and self._want_sl are written from the main thread under _lock.
    * The background thread reads them under _lock at the top of each iteration.
    * self._pos_out / self._sl_out are written by the background thread and
      read (then cleared) by the main thread, both under _lock.
    * FlowMapSimulation itself is only ever touched by the background thread
      after being handed over via set_sim().
    """

    def __init__(self, parent: object = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._sim: FlowMapSimulation | None = None
        self._want_sl: bool = False
        self._new_sim: bool = False
        self._running: bool = False
        self._pos_out: NDArray[np.float32] | None = None
        self._sl_out: NDArray[np.float32] | None = None
        self._produced_frames: int = 0
        self._consumed_frames: int = 0
        self._dropped_frames: int = 0
        self._next_step: float = 0.0

    # -- Main-thread API ---------------------------------------------------

    def set_sim(self, sim: FlowMapSimulation | None, streamlines: bool) -> None:
        with self._lock:
            self._sim = sim
            self._want_sl = streamlines
            self._new_sim = True
            self._pos_out = None
            self._sl_out = None
            self._produced_frames = 0
            self._consumed_frames = 0
            self._dropped_frames = 0

    def set_streamlines(self, enabled: bool) -> None:
        with self._lock:
            self._want_sl = enabled

    def take(
        self,
    ) -> tuple[NDArray[np.float32] | None, NDArray[np.float32] | None]:
        """Pop latest (positions, streamline_verts).  Returns (None, None) when unchanged."""
        with self._lock:
            pos, sl = self._pos_out, self._sl_out
            self._pos_out = None
            self._sl_out = None
            if pos is not None or sl is not None:
                self._consumed_frames += 1
            return pos, sl

    def frame_stats(self) -> tuple[int, int, int]:
        """Return (produced_frames, consumed_frames, dropped_frames)."""
        with self._lock:
            return (
                self._produced_frames,
                self._consumed_frames,
                self._dropped_frames,
            )

    def has_sim(self) -> bool:
        with self._lock:
            return self._sim is not None

    # -- Thread body -------------------------------------------------------

    def run(self) -> None:
        self._running = True
        cur_sl = False
        self._next_step = time.perf_counter()
        while self._running:
            with self._lock:
                sim = self._sim
                want_sl = self._want_sl
                new_sim = self._new_sim
                self._new_sim = False

            if sim is None:
                time.sleep(0.005)
                self._next_step = time.perf_counter() + _SIM_DT
                continue

            now = time.perf_counter()
            if new_sim:
                self._next_step = now

            if now < self._next_step:
                time.sleep(self._next_step - now)
                continue

            self._next_step += _SIM_DT
            if now - self._next_step > _SIM_DT:
                self._next_step = now + _SIM_DT

            try:
                # Sync streamlines flag (may allocate trail buffers)
                if new_sim or cur_sl != want_sl:
                    sim.set_streamlines_enabled(want_sl)
                    cur_sl = want_sl
                sim.step(_SIM_DT)

                pos = sim.positions.copy().astype(np.float32, copy=False)
                sl_verts: NDArray[np.float32] | None = None
                if cur_sl and sim.trail_count >= 2:
                    segs = sim.streamline_segments_limited(_MAX_SL_SEGS)
                    if segs.size:
                        sl_verts = segs.reshape(-1, 3).astype(np.float32, copy=False)

                with self._lock:
                    if self._pos_out is not None or self._sl_out is not None:
                        self._dropped_frames += 1
                    self._pos_out = pos
                    self._sl_out = sl_verts
                    self._produced_frames += 1
            except Exception:
                logger.exception("SimThread step failed")
                time.sleep(0.1)

    def stop(self) -> None:
        self._running = False
        self.wait(2000)


# ---------------------------------------------------------------------------
# OpenGL widget
# ---------------------------------------------------------------------------

class FlowSimView(QOpenGLWidget):
    """
    Independent 3-D OpenGL view for flow particles over a lattice.

    Static lattice geometry is uploaded once when it changes.
    Dynamic particle / streamline geometry comes from _SimThread every frame.
    """

    def __init__(self, parent: object = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)

        # Colors
        self._bg = QColor(DEFAULT_BACKGROUND)
        self._lat_col = QColor(DEFAULT_LATTICE_COLOR)
        self._bnd_col = QColor(DEFAULT_BOUNDARY_COLOR)
        self._ubnd_col = QColor(DEFAULT_UNTAGGED_BOUNDARY_COLOR)
        self._par_col = QColor(DEFAULT_PARTICLE_COLOR)
        self._sl_col = QColor(DEFAULT_STREAMLINE_COLOR)
        self._tag_colors: dict[int, QColor] = {}
        self._particle_size = 3
        self._sl_visible = False

        # Camera
        self._camera = OrbitCamera()
        self._camera.set_standard_view()
        self._drag: QPoint | None = None

        # Simulation reference (read-only on main thread, only for status display)
        self._simulation: FlowMapSimulation | None = None

        # Static geometry (float32, subsampled)
        self._lat: NDArray[np.float32] = np.empty((0, 3), dtype=np.float32)
        self._bnd: NDArray[np.float32] = np.empty((0, 3), dtype=np.float32)
        self._ubnd: NDArray[np.float32] = np.empty((0, 3), dtype=np.float32)
        self._tagged: NDArray[np.float32] = np.empty((0, 3), dtype=np.float32)
        self._tag_ids: NDArray[np.uint16] = np.empty(0, dtype=np.uint16)

        # Latest data from _SimThread
        self._cur_pos: NDArray[np.float32] | None = None
        self._cur_sl: NDArray[np.float32] | None = None
        self._prev_pos: NDArray[np.float32] | None = None
        self._move_mean = 0.0
        self._move_max = 0.0
        self._move_ratio = 0.0
        self._move_zero_ratio = 0.0
        self._frame_seq = 0
        self._stuck_counter = 0
        self._frame_cmp_log: deque[dict[str, float | int]] = deque(maxlen=_FRAME_COMPARE_LOG)
        self._last_frame_stats: dict[str, float | int] = {
            "frame": 0,
            "mean_disp": 0.0,
            "max_disp": 0.0,
            "moved_ratio": 0.0,
            "zero_ratio": 0.0,
            "max_index": 0,
        }

        # Dirty flags
        self._static_dirty = True
        self._dyn_dirty = True

        # Adaptive render budget
        self._par_limit = _MAX_PARTICLES
        self._sl_limit = _MAX_SL_SEGS
        self._paint_ms = 0.0

        # OpenGL objects (created in initializeGL)
        self._ctx: moderngl.Context | None = None
        self._p_static: moderngl.Program | None = None
        self._p_dyn: moderngl.Program | None = None
        self._p_line: moderngl.Program | None = None
        # key → {buf, vao, n, cap}
        self._gl: dict[str, dict] = {}

        # Render timer: poll thread + repaint at 60 fps
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._on_timer)

        # Background thread (starts idle, waits for set_sim)
        self._sim_thread = _SimThread(self)
        self._sim_thread.start()

    # -----------------------------------------------------------------------
    # Public API (called from FlowSimPanel)
    # -----------------------------------------------------------------------

    def set_simulation(
        self,
        simulation: FlowMapSimulation | None,
        streamlines_visible: bool,
    ) -> None:
        self._simulation = simulation
        self._sl_visible = streamlines_visible
        self._sim_thread.set_sim(simulation, streamlines_visible)
        self._cur_pos = None
        self._cur_sl = None
        self._prev_pos = None
        self._move_mean = 0.0
        self._move_max = 0.0
        self._move_ratio = 0.0
        self._move_zero_ratio = 0.0
        self._frame_seq = 0
        self._stuck_counter = 0
        self._frame_cmp_log.clear()
        self._last_frame_stats = {
            "frame": 0,
            "mean_disp": 0.0,
            "max_disp": 0.0,
            "moved_ratio": 0.0,
            "zero_ratio": 1.0,
            "max_index": 0,
        }
        self._dyn_dirty = True
        self._thread_stats: tuple[int, int, int] = (0, 0, 0)
        if simulation is None:
            self._timer.stop()
        else:
            self._reset_budget()
            if not self._timer.isActive():
                self._timer.start()
        self.update()

    def set_lattice_points(
        self,
        lattice: NDArray[np.float64],
        boundary: NDArray[np.float64],
        untagged: NDArray[np.float64],
        tagged: NDArray[np.float64],
        tag_ids: NDArray[np.uint16],
    ) -> None:
        self._lat = _subsample(np.asarray(lattice, np.float32), 80_000)
        self._bnd = _subsample(np.asarray(boundary, np.float32), 40_000)
        self._ubnd = _subsample(np.asarray(untagged, np.float32), 20_000)
        if tagged.size and tag_ids.size:
            idx = np.linspace(0, len(tagged) - 1, min(len(tagged), 20_000), dtype=np.intp)
            self._tagged = np.asarray(tagged[idx], np.float32)
            self._tag_ids = tag_ids[idx]
        else:
            self._tagged = np.empty((0, 3), np.float32)
            self._tag_ids = np.empty(0, np.uint16)
        self._static_dirty = True
        self.update()

    def set_tag_colors(self, colors: dict[int, QColor]) -> None:
        self._tag_colors = {k: QColor(v) for k, v in colors.items()}
        self._static_dirty = True
        self.update()

    def set_background_color(self, color: QColor) -> None:
        self._bg = QColor(color)
        self.update()

    def set_lattice_color(self, color: QColor) -> None:
        self._lat_col = QColor(color)
        self._static_dirty = True
        self.update()

    def set_boundary_color(self, color: QColor) -> None:
        self._bnd_col = QColor(color)
        self._static_dirty = True
        self.update()

    def set_untagged_boundary_color(self, color: QColor) -> None:
        self._ubnd_col = QColor(color)
        self._static_dirty = True
        self.update()

    def set_particle_color(self, color: QColor) -> None:
        self._par_col = QColor(color)
        self.update()

    def set_streamline_color(self, color: QColor) -> None:
        self._sl_col = QColor(color)
        self.update()

    def set_particle_size(self, size: int) -> None:
        self._particle_size = max(1, int(size))
        self.update()

    def set_active(self, active: bool) -> None:
        if active and self._simulation is not None:
            self._reset_budget()
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()

    def set_streamlines_visible(self, visible: bool) -> None:
        self._sl_visible = visible
        if not visible:
            self._cur_sl = None
        self._sim_thread.set_streamlines(visible)
        self._dyn_dirty = True
        self.update()

    def frame_points(self, points: NDArray[np.float64]) -> None:
        if not points.size:
            return
        ok = points[np.all(np.isfinite(points), axis=1)]
        if not ok.size:
            return
        lo, hi = ok.min(0), ok.max(0)
        self._camera.frame(
            BoundingBox3D(float(lo[0]), float(hi[0]),
                          float(lo[1]), float(hi[1]),
                          float(lo[2]), float(hi[2]))
        )
        self.update()

    # -----------------------------------------------------------------------
    # Render timer
    # -----------------------------------------------------------------------

    def _on_timer(self) -> None:
        pos, sl = self._sim_thread.take()
        if pos is not None:
            self._update_particle_motion(pos)
            self._cur_pos = pos
            self._dyn_dirty = True
        if sl is not None:
            self._cur_sl = sl
            self._dyn_dirty = True
        self._thread_stats = self._sim_thread.frame_stats()
        self._adapt_budget()
        self.update()

    def _reset_budget(self) -> None:
        self._par_limit = _MAX_PARTICLES
        self._sl_limit = _MAX_SL_SEGS
        self._paint_ms = 0.0

    def _adapt_budget(self) -> None:
        ms = self._paint_ms
        if ms <= 0:
            return
        if ms > _TARGET_MS:
            self._par_limit = max(2_000, int(self._par_limit * 0.9))
            self._sl_limit = max(500, int(self._sl_limit * 0.88))
        elif ms < _TARGET_MS * 0.65:
            self._par_limit = min(_MAX_PARTICLES, int(self._par_limit * 1.05) + 512)
            self._sl_limit = min(_MAX_SL_SEGS, int(self._sl_limit * 1.05) + 256)

    # -----------------------------------------------------------------------
    # OpenGL lifecycle
    # -----------------------------------------------------------------------

    def initializeGL(self) -> None:
        try:
            self._ctx = moderngl.create_context(require=330)
        except moderngl.Error as e:
            logger.warning("flow sim GL init failed: %s", e)
            return
        ctx = self._ctx
        ctx.enable(moderngl.PROGRAM_POINT_SIZE | moderngl.DEPTH_TEST | moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self._p_static = ctx.program(vertex_shader=_VERT_STATIC, fragment_shader=_FRAG_STATIC)
        self._p_dyn = ctx.program(vertex_shader=_VERT_DYN, fragment_shader=_FRAG_UNI)
        self._p_line = ctx.program(vertex_shader=_VERT_LINE, fragment_shader=_FRAG_UNI)
        self._static_dirty = True
        self._dyn_dirty = True

    def paintGL(self) -> None:
        t0 = time.perf_counter()
        if self._ctx is None:
            self._fallback_paint()
            return

        ctx = self._ctx
        # QOpenGLWidget renders to an internal FBO, not framebuffer 0.
        fbo = ctx.detect_framebuffer(glo=self.defaultFramebufferObject())
        fbo.use()
        # Re-enable GL flags each frame — QPainter corrupts GL state.
        ctx.enable_only(
            moderngl.PROGRAM_POINT_SIZE | moderngl.DEPTH_TEST | moderngl.BLEND
        )
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        ctx.viewport = (0, 0, max(1, self.width()), max(1, self.height()))
        ctx.clear(*_rgb(self._bg), 1.0, depth=1.0)

        if self._static_dirty:
            self._upload_static()
        if self._dyn_dirty:
            self._upload_dynamic()

        mvp = (
            self._camera.view_projection(max(self.width(), 1) / max(self.height(), 1))
            .T.astype(np.float32).tobytes()
        )

        if self._p_static and self._gl.get("static", {}).get("n", 0) > 0:
            self._p_static["u_mvp"].write(mvp)
            self._p_static["u_sz"].value = 2.0
            self._gl["static"]["vao"].render(moderngl.POINTS, vertices=self._gl["static"]["n"])

        if self._p_dyn and self._gl.get("particle", {}).get("n", 0) > 0:
            # Particles are intentionally rendered without depth-write/read to avoid
            # per-frame artifacts where older depth values hide moving points.
            ctx.disable(moderngl.DEPTH_TEST)
            self._p_dyn["u_mvp"].write(mvp)
            self._p_dyn["u_sz"].value = float(self._particle_size)
            self._p_dyn["u_col"].value = _rgb(self._par_col)
            self._gl["particle"]["vao"].render(moderngl.POINTS, vertices=self._gl["particle"]["n"])
            ctx.enable(moderngl.DEPTH_TEST)

        if (self._sl_visible and self._p_line
                and self._gl.get("sl", {}).get("n", 0) > 0):
            self._p_line["u_mvp"].write(mvp)
            self._p_line["u_col"].value = _rgb(self._sl_col)
            self._gl["sl"]["vao"].render(moderngl.LINES, vertices=self._gl["sl"]["n"])

        # HUD overlay
        painter = QPainter(self)
        painter.setPen(QColor(190, 208, 228))
        sim = self._simulation
        if sim is None:
            painter.drawText(8, 18, "No simulation — load an .arrow file")
            painter.drawText(8, 50, "Thread: idle")
        else:
            painter.drawText(8, 18, f"Particles: {sim.particle_count:,}  |  {sim.backend_name}")
            painter.drawText(8, 34, f"Streamlines: {'on' if self._sl_visible else 'off'}")
            produced, consumed, dropped = self._thread_stats
            color = QColor(190, 208, 228)
            if dropped > 0:
                color = QColor(251, 191, 36)
            if produced == consumed == 0:
                color = QColor(156, 163, 175)
            painter.setPen(color)
            painter.drawText(
                8,
                50,
                f"Sim thread frames: {consumed}/{produced}, dropped: {dropped}",
            )
            painter.setPen(QColor(190, 208, 228))
            painter.drawText(
                8,
                66,
                (
                    f"Particle motion | mean: {self._move_mean:.4f}  "
                    f"max: {self._move_max:.4f}  "
                    f"moved: {self._move_ratio:.1%}"
                ),
            )
            painter.drawText(
                8,
                82,
                (
                    f"Frame compare | zero: {self._move_zero_ratio:.1%}  "
                    f"frame #{self._last_frame_stats.get('frame', 0)}  "
                    f"stuck run: {self._stuck_counter}"
                ),
            )
        painter.end()

        ms = (time.perf_counter() - t0) * 1_000
        self._paint_ms = ms if self._paint_ms <= 0 else self._paint_ms * 0.85 + ms * 0.15

    # -----------------------------------------------------------------------
    # GPU buffer helpers
    # -----------------------------------------------------------------------

    def _update_particle_motion(self, pos: NDArray[np.float32]) -> None:
        if pos is None or pos.size == 0:
            self._move_mean = 0.0
            self._move_max = 0.0
            self._move_ratio = 0.0
            self._move_zero_ratio = 1.0
            self._prev_pos = None
            return

        self._frame_seq += 1
        stats = self.compare_particle_frames(pos, self._prev_pos)
        self._move_mean = float(stats["mean_disp"])
        self._move_max = float(stats["max_disp"])
        self._move_ratio = float(stats["moved_ratio"])
        self._move_zero_ratio = float(stats["zero_ratio"])
        self._last_frame_stats = stats
        self._frame_cmp_log.append(stats)

        if self._move_ratio < _STATIC_MOVED_RATIO_THRESHOLD:
            self._stuck_counter += 1
            if self._stuck_counter % _STATIC_FRAME_WARN_EVERY == 0:
                logger.warning(
                    "FlowSimView frame #%s appears almost static: moved=%s mean=%s max=%s",
                    self._frame_seq,
                    f"{self._move_ratio:.1%}",
                    f"{self._move_mean:.6f}",
                    f"{self._move_max:.6f}",
                )
        else:
            self._stuck_counter = 0

        self._prev_pos = pos.copy()

    def frame_debug_log(self, max_frames: int | None = None) -> list[dict[str, float | int]]:
        """Return recent frame-to-frame movement metrics (newest last)."""
        if max_frames is None:
            max_frames = _FRAME_COMPARE_LOG
        if max_frames <= 0:
            return []
        return list(self._frame_cmp_log)[-max_frames:]

    def has_frame_stagnation(self, lookback: int = 90) -> bool:
        """Return True if all recent frames report essentially zero movement."""
        if lookback <= 0 or not self._frame_cmp_log:
            return False
        history = self.frame_debug_log(lookback)
        if len(history) < lookback:
            return False
        return all(
            item["moved_ratio"] <= _STATIC_MOVED_RATIO_THRESHOLD for item in history
        )

    def frame_stagnation_report(
        self,
        lookback: int = 90,
    ) -> dict[str, float | int | bool]:
        """Return a compact frame-diff summary for debugging static particle visuals."""
        if lookback <= 0:
            return {
                "window_frames": 0,
                "particle_count": int(self._last_frame_stats.get("particle_count", 0)),
                "static_frames": 0,
                "static_ratio": 1.0,
                "moved_ratio_latest": 0.0,
                "mean_disp_latest": 0.0,
                "max_disp_latest": 0.0,
                "exact_match_streak": 0,
                "longest_static_streak": 0,
                "stuck": False,
            }

        history = self.frame_debug_log(max(1, lookback))
        if not history:
            return {
                "window_frames": 0,
                "particle_count": int(self._last_frame_stats.get("particle_count", 0)),
                "static_frames": 0,
                "static_ratio": 1.0,
                "moved_ratio_latest": 0.0,
                "mean_disp_latest": 0.0,
                "max_disp_latest": 0.0,
                "exact_match_streak": 0,
                "longest_static_streak": 0,
                "stuck": False,
            }

        static_count = 0
        longest_static_streak = 0
        run = 0
        exact_match_count = 0
        for item in history:
            is_static = item["moved_ratio"] <= _STATIC_MOVED_RATIO_THRESHOLD
            if is_static:
                static_count += 1
                run += 1
                longest_static_streak = max(longest_static_streak, run)
            else:
                run = 0
            exact_match_count += int(item["exact_match"])

        latest = history[-1]
        return {
            "window_frames": int(len(history)),
            "particle_count": int(latest.get("particle_count", 0)),
            "static_frames": int(static_count),
            "static_ratio": float(static_count / len(history)),
            "moved_ratio_latest": float(latest.get("moved_ratio", 0.0)),
            "mean_disp_latest": float(latest.get("mean_disp", 0.0)),
            "max_disp_latest": float(latest.get("max_disp", 0.0)),
            "exact_match_frames": int(exact_match_count),
            "longest_static_streak": int(longest_static_streak),
            "stuck": self.has_frame_stagnation(lookback),
        }

    def compare_particle_frames(
        self,
        cur_pos: NDArray[np.float32],
        prev_pos: NDArray[np.float32] | None = None,
    ) -> dict[str, float | int]:
        """Return movement comparison metrics between two particle snapshots."""
        if prev_pos is None or prev_pos.shape != cur_pos.shape or cur_pos.size == 0:
            return {
                "frame": self._frame_seq,
                "mean_disp": 0.0,
                "max_disp": 0.0,
                "moved_ratio": 0.0,
                "zero_ratio": 1.0,
                "max_index": 0,
                "exact_match": 1,
                "particle_count": int(cur_pos.shape[0]),
            }
        disp = np.linalg.norm(cur_pos - prev_pos, axis=1)
        moved = np.count_nonzero(disp > _PARTICLE_MOVE_EPS)
        moved_ratio = float(moved / len(disp))
        max_index = int(np.argmax(disp))
        exact_match = int(np.array_equal(cur_pos, prev_pos))
        return {
            "frame": self._frame_seq,
            "mean_disp": float(np.mean(disp)),
            "max_disp": float(np.max(disp)),
            "moved_ratio": moved_ratio,
            "zero_ratio": float(1.0 - moved_ratio),
            "max_index": max_index,
            "exact_match": exact_match,
            "particle_count": int(len(disp)),
        }

    def _buf_write(
        self,
        key: str,
        data: NDArray[np.float32],
        prog: moderngl.Program,
        fmt: str,
        attrs: tuple[str, ...],
    ) -> None:
        if self._ctx is None:
            return
        payload = data.astype(np.float32, copy=False).tobytes()
        size = len(payload)
        e = self._gl.setdefault(key, {"buf": None, "vao": None, "n": 0, "cap": 0})
        if e["buf"] is None or size > e["cap"]:
            if e["vao"]:
                e["vao"].release()
            if e["buf"]:
                e["buf"].release()
            cap = max(size, 24)
            e["buf"] = self._ctx.buffer(reserve=cap)
            e["vao"] = self._ctx.vertex_array(prog, [(e["buf"], fmt, *attrs)])
            e["cap"] = cap
        if size:
            e["buf"].write(payload)
        e["n"] = data.shape[0]

    def _upload_static(self) -> None:
        if self._ctx is None or self._p_static is None:
            return
        parts: list[NDArray[np.float32]] = []

        def add(pts: NDArray[np.float32], col: QColor) -> None:
            if not pts.size:
                return
            colors = np.tile(_rgb32(col), (len(pts), 1))
            parts.append(np.concatenate((pts, colors), axis=1, dtype=np.float32))

        add(self._lat, self._lat_col)
        add(self._bnd, self._bnd_col)
        add(self._ubnd, self._ubnd_col)
        if self._tagged.size:
            for tid in np.unique(self._tag_ids):
                if tid == 0:
                    continue
                add(self._tagged[self._tag_ids == tid], self._tag_color(int(tid)))

        data = np.concatenate(parts) if parts else np.empty((0, 6), np.float32)
        self._buf_write("static", data, self._p_static, "3f 3f", ("in_pos", "in_col"))
        self._static_dirty = False

    def _upload_dynamic(self) -> None:
        if self._ctx is None or self._p_dyn is None or self._p_line is None:
            return

        # Particles
        pos = self._cur_pos
        if pos is not None and pos.size:
            n = min(self._par_limit, max(2_000, (self.width() * self.height()) // 180))
            par_data = _subsample(pos, n)
        else:
            par_data = np.empty((0, 3), np.float32)
        self._buf_write("particle", par_data, self._p_dyn, "3f", ("in_pos",))

        # Streamlines
        sl = self._cur_sl if self._sl_visible else None
        sl_data = sl if sl is not None else np.empty((0, 3), np.float32)
        self._buf_write("sl", sl_data, self._p_line, "3f", ("in_pos",))

        self._dyn_dirty = False

    def _tag_color(self, tag_id: int) -> QColor:
        return self._tag_colors.get(tag_id, TAG_PALETTE[tag_id % len(TAG_PALETTE)])

    def _fallback_paint(self) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._bg)
        painter.setPen(QColor(190, 208, 228))
        painter.drawText(8, 18, "OpenGL 3.3 unavailable — software fallback")
        painter.end()

    # -----------------------------------------------------------------------
    # Camera interaction
    # -----------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position().toPoint()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag is None:
            super().mouseMoveEvent(event)
            return
        cur = event.position().toPoint()
        self._camera.orbit(float(cur.x() - self._drag.x()), float(cur.y() - self._drag.y()))
        self._drag = cur
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        self._camera.zoom(event.angleDelta().y() / 120.0)
        self.update()
        event.accept()

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def closeEvent(self, event: object) -> None:
        self._timer.stop()
        self._sim_thread.stop()
        for e in self._gl.values():
            for key in ("vao", "buf"):
                obj = e.get(key)
                if obj:
                    try:
                        obj.release()
                    except Exception:
                        pass
        self._gl.clear()
        self._p_static = self._p_dyn = self._p_line = None
        self._ctx = None
        super().closeEvent(event)
