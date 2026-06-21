"""
Single integration test for the Sim feature.

Covers:
  - flow_sim_arrow: FlowLatticeData dataclass and derived masks
  - flow_sim_map: FlowMapSimulation creation, step, streamlines, backend name
  - flow_sim_view: _SimThread runs in background and produces position snapshots
"""
from __future__ import annotations

from collections import deque
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Minimal mock for core.io.read_lattice so we don't need a real .arrow file
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_read_lattice(monkeypatch):
    """Replace read_lattice with a minimal in-memory table/metadata."""

    class FakeTable:
        column_names = ["x", "y", "z", "node_type", "i", "j", "k"]

        def column(self, name: str):
            N = 200
            mapping = {
                "x": np.linspace(0.0, 1.0, N),
                "y": np.zeros(N),
                "z": np.zeros(N),
                "node_type": np.where(
                    np.arange(N) % 10 == 0, np.uint8(1), np.uint8(0)
                ),
                "i": np.arange(N, dtype=np.int64),
                "j": np.zeros(N, dtype=np.int64),
                "k": np.zeros(N, dtype=np.int64),
            }
            return mapping[name]

    metadata = {
        "grid": {
            "dx": 1.0 / 199,
            "nx": 200,
            "ny": 1,
            "nz": 1,
            "dimension": 2,
            "lattice_origin": [0.0, 0.0, 0.0],
            "axis_i": [1.0, 0.0, 0.0],
            "axis_j": [0.0, 1.0, 0.0],
            "axis_k": [0.0, 0.0, 1.0],
        },
        "object_directory": [
            {"kind": "boundary_region", "object_id": 1,
             "name": "inlet", "outside_direction": 0},
        ],
    }

    import app.flow_sim_arrow as _fsa
    monkeypatch.setattr(_fsa, "read_lattice", lambda _path: (FakeTable(), metadata))
    yield


# ---------------------------------------------------------------------------
# flow_sim_arrow
# ---------------------------------------------------------------------------

class TestFlowLatticeData:
    def _load(self):
        from app.flow_sim_arrow import load_arrow_lattice
        return load_arrow_lattice(Path("fake.arrow"))

    def test_loads_positions(self):
        lat = self._load()
        assert lat.positions.shape == (200, 3)
        assert lat.positions.dtype == np.float64

    def test_fluid_and_boundary_masks(self):
        lat = self._load()
        assert lat.boundary_mask.sum() == 20   # every 10th node
        assert lat.fluid_mask.sum() == 180

    def test_untagged_boundary_mask(self):
        lat = self._load()
        # No tag_ids column → all boundary nodes are untagged
        assert int(lat.untagged_boundary_mask.sum()) == 20

    def test_flow_direction_from_inlet(self):
        lat = self._load()
        # axis_direction 0 → (-1,0,0); inlet negates → (1,0,0)
        assert np.allclose(lat.flow_direction, (1.0, 0.0, 0.0))

    def test_grid_metadata(self):
        lat = self._load()
        assert lat.nx == 200
        assert lat.ny == 1
        assert lat.nz == 1
        assert lat.dimension == 2


# ---------------------------------------------------------------------------
# flow_sim_map
# ---------------------------------------------------------------------------

class TestFlowMapSimulation:
    def _lattice(self):
        from app.flow_sim_arrow import load_arrow_lattice
        return load_arrow_lattice(Path("fake.arrow"))

    def _planar_lattice(self, z_scale: float = 0.0):
        from app.flow_sim_arrow import load_arrow_lattice
        import app.flow_sim_arrow as _fsa

        class FakeTable:
            column_names = ["x", "y", "z", "node_type", "i", "j", "k"]

            def __init__(self, nx: int = 200, ny: int = 40) -> None:
                self.nx = nx
                self.ny = ny
                X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
                self.x = X.reshape(-1).astype(np.float64)
                self.y = Y.reshape(-1).astype(np.float64)
                self.z = (self.y * float(z_scale)).astype(np.float64)
                boundary = (self.x % 30 == 0) & (self.y % 2 == 0)
                self.node_type = np.where(boundary, np.uint8(1), np.uint8(0))
                self.i = self.x.astype(np.int64)
                self.j = self.y.astype(np.int64)
                self.k = np.zeros_like(self.i)

            def column(self, name: str):
                return {
                    "x": self.x,
                    "y": self.y,
                    "z": self.z,
                    "node_type": self.node_type,
                    "i": self.i,
                    "j": self.j,
                    "k": self.k,
                }[name]

        metadata = {
            "grid": {
                "dx": 1.0,
                "nx": 200,
                "ny": 40,
                "nz": 1,
                "dimension": 2,
                "lattice_origin": [0.0, 0.0, 0.0],
                "axis_i": [1.0, 0.0, 0.0],
                "axis_j": [0.0, 1.0, 0.0],
                "axis_k": [0.0, 0.0, 1.0],
            },
            "object_directory": [
                {
                    "kind": "boundary_region",
                    "object_id": 1,
                    "name": "inlet",
                    "outside_direction": 0,
                }
            ],
        }

        _fsa.read_lattice = lambda _path: (FakeTable(), metadata)
        return load_arrow_lattice(Path("fake_planar.arrow"))

    def _thin_axis_lattice(
        self,
        nx: int = 1200,
        ny: int = 4,
        nz: int = 5,
        dx: float = 1e-5,
    ):
        from app.flow_sim_arrow import load_arrow_lattice
        import app.flow_sim_arrow as _fsa

        class FakeTable:
            column_names = ["x", "y", "z", "node_type", "i", "j", "k"]

            def __init__(self) -> None:
                X, Y, Z = np.indices((nx, ny, nz))
                self.x = (X * dx).reshape(-1).astype(np.float64)
                self.y = (Y * dx).reshape(-1).astype(np.float64)
                self.z = (Z * dx).reshape(-1).astype(np.float64)
                boundary = (X % 10 == 0) & (Y % 2 == 0)
                self.node_type = np.where(
                    boundary, np.uint8(1), np.uint8(0)
                ).reshape(-1)
                self.i = X.astype(np.int64).reshape(-1)
                self.j = Y.astype(np.int64).reshape(-1)
                self.k = Z.astype(np.int64).reshape(-1)

            def column(self, name: str):
                return {
                    "x": self.x,
                    "y": self.y,
                    "z": self.z,
                    "node_type": self.node_type,
                    "i": self.i,
                    "j": self.j,
                    "k": self.k,
                }[name]

        metadata = {
            "grid": {
                "dx": dx,
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "dimension": 3,
                "lattice_origin": [0.0, 0.0, 0.0],
                "axis_i": [1.0, 0.0, 0.0],
                "axis_j": [0.0, 1.0, 0.0],
                "axis_k": [0.0, 0.0, 1.0],
            },
            "object_directory": [
                {
                    "kind": "boundary_region",
                    "object_id": 1,
                    "name": "inlet",
                    "outside_direction": 0,
                }
            ],
        }

        _fsa.read_lattice = lambda _path: (FakeTable(), metadata)
        return load_arrow_lattice(Path("fake_thin.axis.arrow"))

    def test_thin_axis_box_tolerance_does_not_block_particles(self):
        from app.flow_sim_map import FlowMapSimulation

        lat = self._thin_axis_lattice(nx=800, ny=6, nz=5, dx=1e-5)
        assert lat.dimension == 3
        assert lat.nz == 5
        sim = FlowMapSimulation.from_lattice(
            lat,
            particle_count=1024,
            velocity=0.02,
            diffusion=0.0,
            seed=31,
        )

        before = sim.positions.copy()
        for _ in range(90):
            sim.step(1.0 / 60.0)
        proj0 = before @ sim.direction
        proj1 = sim.positions @ sim.direction
        span = float(sim.proj_max - sim.proj_min)
        raw_shift = proj1 - proj0
        if span > 0:
            mean_shift = float(np.mean(np.mod(raw_shift, span)))
        else:
            mean_shift = float(np.mean(raw_shift))
        assert mean_shift > 0.002

    def test_planar_collapsed_axis_moves_particles_along_flow(self):
        from app.flow_sim_map import FlowMapSimulation

        lat = self._planar_lattice()
        sim = FlowMapSimulation.from_lattice(
            lat,
            particle_count=2000,
            velocity=5.0,
            diffusion=0.08,
            seed=7,
        )
        before = sim.positions.copy()
        for _ in range(360):
            sim.step(1.0 / 60.0)

        proj0 = before @ sim.direction
        proj1 = sim.positions @ sim.direction
        mean_shift = float(np.mean(proj1 - proj0))
        near_inlet_band = float(
            np.mean(
                proj1 <= sim.proj_min + 0.15 * max(sim.proj_max - sim.proj_min, 1.0)
            )
        )
        assert mean_shift > 1.0
        assert near_inlet_band < 0.95

    def test_indices_origin_mismatch_still_allows_flow(self):
        from app.flow_sim_map import FlowMapSimulation

        class OffsetTable:
            column_names = ["x", "y", "z", "node_type", "i", "j", "k"]

            def __init__(self, nx: int = 64, ny: int = 8, nz: int = 1, dx: float = 0.5):
                X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
                self.x = (X.astype(np.float64) * dx - 2.0)
                self.y = (Y.astype(np.float64) * dx - 1.0)
                self.z = np.zeros_like(self.x)
                self.node_type = np.zeros_like(self.x, dtype=np.uint8)
                self.i = X.astype(np.int64)
                self.j = Y.astype(np.int64)
                self.k = np.zeros_like(self.i)

            def column(self, name: str):
                return {
                    "x": self.x.reshape(-1),
                    "y": self.y.reshape(-1),
                    "z": self.z.reshape(-1),
                    "node_type": self.node_type.reshape(-1),
                    "i": self.i.reshape(-1),
                    "j": self.j.reshape(-1),
                    "k": self.k.reshape(-1),
                }[name]

        def make_lattice():
            import app.flow_sim_arrow as fsa
            from app.flow_sim_arrow import load_arrow_lattice

            metadata = {
                "grid": {
                    "dx": 0.5,
                    "nx": 64,
                    "ny": 8,
                    "nz": 1,
                    "dimension": 2,
                    "lattice_origin": [0.0, 0.0, 0.0],
                    "axis_i": [1.0, 0.0, 0.0],
                    "axis_j": [0.0, 1.0, 0.0],
                    "axis_k": [0.0, 0.0, 1.0],
                },
                "object_directory": [
                    {"kind": "boundary_region", "object_id": 1, "name": "inlet", "outside_direction": 0}
                ],
            }
            fsa.read_lattice = lambda _path: (OffsetTable(), metadata)
            return load_arrow_lattice(Path("fake_offset.arrow"))

        lat = make_lattice()
        sim = FlowMapSimulation.from_lattice(
            lat,
            particle_count=800,
            velocity=3.0,
            diffusion=0.0,
            seed=3,
        )
        before = sim.positions.copy()
        for _ in range(120):
            sim.step(1.0 / 60.0)

        proj0 = before @ sim.direction
        proj1 = sim.positions @ sim.direction
        assert np.mean(proj1 - proj0) > 0.5
        assert np.mean(proj1 <= sim.proj_min + 0.15 * max(sim.proj_max - sim.proj_min, 1.0)) < 0.95

    def test_planar_axis_mask_ignores_noisy_thickness(self):
        from app.flow_sim_map import FlowMapSimulation

        lat = self._planar_lattice(z_scale=1e-10)
        sim = FlowMapSimulation.from_lattice(
            lat,
            particle_count=64,
            velocity=4.0,
            diffusion=0.08,
            seed=11,
        )
        assert not sim.axis_motion_mask[2]
        assert sim.axis_motion_mask[:2].all()

        before = sim.positions.copy()
        for _ in range(30):
            sim.step(1.0 / 60.0)
        assert not np.allclose(sim.positions, before)

    def test_from_lattice_creates_simulation(self):
        from app.flow_sim_map import FlowMapSimulation
        sim = FlowMapSimulation.from_lattice(self._lattice(), particle_count=50)
        assert sim.particle_count == 50
        assert sim.positions.shape == (50, 3)
        assert sim.positions.dtype == np.float32

    def test_step_moves_particles(self):
        from app.flow_sim_map import FlowMapSimulation
        sim = FlowMapSimulation.from_lattice(self._lattice(), particle_count=50)
        before = sim.positions.copy()
        sim.step(0.016)
        # At least some particles should have moved
        assert not np.allclose(sim.positions, before)

    def test_step_keeps_particles_in_fluid(self):
        from app.flow_sim_map import FlowMapSimulation
        lat = self._lattice()
        sim = FlowMapSimulation.from_lattice(lat, particle_count=100)
        for _ in range(20):
            sim.step(0.016)
        # All positions should be within bounding box
        lo = lat.positions.min(0).astype(np.float32) - 0.02
        hi = lat.positions.max(0).astype(np.float32) + 0.02
        assert np.all(sim.positions >= lo)
        assert np.all(sim.positions <= hi)

    def test_backend_name_is_string(self):
        from app.flow_sim_map import FlowMapSimulation
        sim = FlowMapSimulation.from_lattice(self._lattice(), particle_count=10)
        assert isinstance(sim.backend_name, str)
        assert len(sim.backend_name) > 0

    def test_streamlines_disabled_by_default(self):
        from app.flow_sim_map import FlowMapSimulation
        sim = FlowMapSimulation.from_lattice(self._lattice(), particle_count=20)
        assert not sim.streamlines_enabled
        segs = sim.streamline_segments_limited()
        assert segs.shape == (0, 2, 3)

    def test_streamlines_enable_disable(self):
        from app.flow_sim_map import FlowMapSimulation
        sim = FlowMapSimulation.from_lattice(
            self._lattice(), particle_count=20, streamline_length=5
        )
        sim.set_streamlines_enabled(True)
        assert sim.streamlines_enabled
        # Step a few times to fill trail buffer
        for _ in range(10):
            sim.step(0.016)
        segs = sim.streamline_segments_limited()
        assert segs.ndim == 3
        assert segs.shape[1] == 2
        assert segs.shape[2] == 3
        # Disable and check cleared
        sim.set_streamlines_enabled(False)
        assert not sim.streamlines_enabled
        assert sim.streamline_segments_limited().size == 0

    def test_particle_vertices_returns_float32(self):
        from app.flow_sim_map import FlowMapSimulation
        sim = FlowMapSimulation.from_lattice(self._lattice(), particle_count=30)
        verts = sim.particle_vertices()
        assert verts.dtype == np.float32
        assert verts.shape == (30, 3)

    def test_seeded_particles_start_near_inlet_band(self):
        from app.flow_sim_map import FlowMapSimulation
        sim = FlowMapSimulation.from_lattice(
            self._lattice(),
            particle_count=120,
            velocity=4.0,
            diffusion=0.0,
            seed=11,
        )
        proj = sim.positions @ sim.direction
        span = sim.proj_max - sim.proj_min
        assert np.all(proj <= sim.proj_min + 0.2 * max(span, 1e-6))

    def test_multiple_steps_keep_particles_advecting(self):
        from app.flow_sim_map import FlowMapSimulation
        sim = FlowMapSimulation.from_lattice(
            self._lattice(),
            particle_count=120,
            velocity=4.0,
            diffusion=0.0,
            seed=42,
        )
        before = sim.positions.copy()
        moved_ratios = []
        for _ in range(8):
            sim.step(1.0 / 60.0)
            disp = np.linalg.norm(sim.positions - before, axis=1)
            moved = np.count_nonzero(disp > 1e-6)
            moved_ratios.append(float(moved / len(disp)))
            before = sim.positions.copy()
        assert max(moved_ratios) > 0.1
        assert max(sim.positions[:, 0]) >= sim.proj_min


# ---------------------------------------------------------------------------
# flow_sim_view._SimThread
# ---------------------------------------------------------------------------

class TestSimThread:
    def _lattice(self):
        from app.flow_sim_arrow import load_arrow_lattice
        return load_arrow_lattice(Path("fake.arrow"))

    def test_thread_produces_positions(self):
        from app.flow_sim_map import FlowMapSimulation
        from app.flow_sim_view import _SimThread

        sim = FlowMapSimulation.from_lattice(self._lattice(), particle_count=30)
        thread = _SimThread()
        thread.set_sim(sim, streamlines=False)
        thread.start()
        try:
            deadline = time.monotonic() + 3.0
            pos = None
            while time.monotonic() < deadline:
                pos, _ = thread.take()
                if pos is not None:
                    break
                time.sleep(0.01)
            assert pos is not None, "thread never produced positions"
            assert pos.shape == (30, 3)
            assert pos.dtype == np.float32
        finally:
            thread.stop()

    def test_thread_set_sim_none_stops_producing(self):
        from app.flow_sim_map import FlowMapSimulation
        from app.flow_sim_view import _SimThread

        sim = FlowMapSimulation.from_lattice(self._lattice(), particle_count=20)
        thread = _SimThread()
        thread.set_sim(sim, streamlines=False)
        thread.start()
        try:
            # Let it produce at least one frame
            time.sleep(0.1)
            thread.set_sim(None, streamlines=False)
            time.sleep(0.05)
            # Drain any pending output
            thread.take()
            thread.take()
            # After draining, should produce nothing
            time.sleep(0.05)
            pos, _ = thread.take()
            assert pos is None
        finally:
            thread.stop()

    def test_thread_produces_changing_frames(self):
        from app.flow_sim_map import FlowMapSimulation
        from app.flow_sim_view import _SimThread

        sim = FlowMapSimulation.from_lattice(
            self._lattice(),
            particle_count=60,
            velocity=5.0,
            diffusion=0.0,
            seed=21,
        )
        thread = _SimThread()
        thread.set_sim(sim, streamlines=False)
        thread.start()
        try:
            frames = []
            deadline = time.monotonic() + 3.5
            while len(frames) < 12 and time.monotonic() < deadline:
                pos, _ = thread.take()
                if pos is not None:
                    frames.append(pos.copy())
                time.sleep(0.005)
            assert len(frames) >= 8
            produced, consumed, dropped = thread.frame_stats()
            assert produced > 0
            assert consumed > 0
            assert produced >= consumed
            ratios = []
            for a, b in zip(frames[:-1], frames[1:]):
                disp = np.linalg.norm(a - b, axis=1)
                moved = np.count_nonzero(disp > 1e-6)
                ratios.append(float(moved / len(disp)))
            assert max(ratios) > 0.5
            assert len(ratios) == len(frames) - 1
            assert dropped <= produced
        finally:
            thread.stop()

    def test_thread_frame_stats_report_progress(self):
        from app.flow_sim_map import FlowMapSimulation
        from app.flow_sim_view import _SimThread

        sim = FlowMapSimulation.from_lattice(self._lattice(), particle_count=20)
        thread = _SimThread()
        thread.set_sim(sim, streamlines=False)
        thread.start()
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                thread.take()
                time.sleep(0.01)
            produced, consumed, dropped = thread.frame_stats()
            assert produced >= 1
            assert consumed >= 1
            assert dropped >= 0
            assert dropped < max(1, produced)
        finally:
            thread.stop()

    def test_thread_streamlines(self):
        from app.flow_sim_map import FlowMapSimulation
        from app.flow_sim_view import _SimThread

        sim = FlowMapSimulation.from_lattice(
            self._lattice(), particle_count=50, streamline_length=5
        )
        thread = _SimThread()
        thread.set_sim(sim, streamlines=True)
        thread.start()
        try:
            deadline = time.monotonic() + 3.0
            sl = None
            while time.monotonic() < deadline:
                _, sl = thread.take()
                if sl is not None:
                    break
                time.sleep(0.01)
            # Streamlines may be None until trail buffer fills; that's OK
            # as long as the thread is still alive and producing positions
            assert thread.isRunning()  # thread must not have crashed
        finally:
            thread.stop()


class TestFlowSimViewDebug:
    def _view(self) -> object:
        from app.flow_sim_view import FlowSimView
        # Avoid creating a full QWidget/QOpenGL context; we only need frame helpers.
        view = FlowSimView.__new__(FlowSimView)
        view._frame_seq = 0
        view._prev_pos = None
        view._move_mean = 0.0
        view._move_max = 0.0
        view._move_ratio = 0.0
        view._move_zero_ratio = 1.0
        view._stuck_counter = 0
        view._frame_cmp_log = deque(maxlen=20)
        view._last_frame_stats = {
            "frame": 0,
            "mean_disp": 0.0,
            "max_disp": 0.0,
            "moved_ratio": 0.0,
            "zero_ratio": 1.0,
            "max_index": 0,
            "exact_match": 1,
            "particle_count": 0,
        }
        return view

    def test_frame_debug_log_catches_stagnant_frames(self) -> None:
        from app.flow_sim_view import FlowSimView

        view = self._view()
        assert isinstance(view, FlowSimView)

        base = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        view._update_particle_motion(base)
        view._update_particle_motion(base.copy())
        view._update_particle_motion(base.copy())

        log = view.frame_debug_log()
        assert len(log) == 3
        assert all(item["exact_match"] == 1 for item in log)
        assert view.has_frame_stagnation(lookback=3)

    def test_frame_debug_log_catches_moving_frames(self) -> None:
        from app.flow_sim_view import FlowSimView

        view = self._view()
        assert isinstance(view, FlowSimView)

        base = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        moved = np.array([[0.1, 0.2, 0.4], [0.4, 0.5, 0.6]], dtype=np.float32)

        view._update_particle_motion(base)
        view._update_particle_motion(base.copy())
        view._update_particle_motion(moved)

        log = view.frame_debug_log()
        assert len(log) == 3
        assert log[0]["exact_match"] == 1
        assert log[1]["exact_match"] == 1
        assert log[2]["exact_match"] == 0
        assert log[2]["moved_ratio"] > 0.0
        assert not view.has_frame_stagnation(lookback=2)

    def test_frame_report_flags_moving_batch(self) -> None:
        from app.flow_sim_map import FlowMapSimulation
        from app.flow_sim_view import FlowSimView, _SimThread

        # Use a non-collapsed geometry where advection is expected for each frame.
        from app.flow_sim_arrow import load_arrow_lattice

        sim = FlowMapSimulation.from_lattice(
            load_arrow_lattice(Path("fake.arrow")),
            particle_count=128,
            velocity=6.0,
            diffusion=0.0,
            seed=4,
        )

        thread = _SimThread()
        thread.set_sim(sim, streamlines=False)
        thread.start()
        try:
            view = FlowSimView.__new__(FlowSimView)
            view._frame_seq = 0
            view._prev_pos = None
            view._frame_cmp_log = deque(maxlen=60)
            view._move_mean = 0.0
            view._move_max = 0.0
            view._move_ratio = 0.0
            view._move_zero_ratio = 1.0
            view._stuck_counter = 0
            view._last_frame_stats = {
                "frame": 0,
                "mean_disp": 0.0,
                "max_disp": 0.0,
                "moved_ratio": 0.0,
                "zero_ratio": 1.0,
                "max_index": 0,
                "exact_match": 1,
                "particle_count": 0,
            }

            frames = 0
            deadline = time.monotonic() + 3.0
            while frames < 12 and time.monotonic() < deadline:
                pos, _ = thread.take()
                if pos is not None:
                    view._update_particle_motion(pos)
                    frames += 1
                time.sleep(0.005)
            report = view.frame_stagnation_report(12)
            assert frames >= 8
            assert report["moved_ratio_latest"] > 0.0
            assert report["exact_match_frames"] <= 1
            assert report["static_ratio"] < 0.6
            assert report["stuck"] is False
        finally:
            thread.stop()
