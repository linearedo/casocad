"""
flow_sim_map.py – Fluid-particle simulation math.

Uses JAX (with @jax.jit) when available for near-zero step cost after
warm-up compilation; falls back to NumPy transparently.

No Qt, no file I/O, no OpenGL here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.flow_sim_arrow import FlowLatticeData

FloatArray = NDArray[np.float64]
F32Array = NDArray[np.float32]

_DEFAULT_SL_STEPS = 20
_DEFAULT_SL_PARTICLE_LIMIT = 2048
_PLANAR_AXIS_EPS = 1e-12
_MAX_BOUNDARY_TOLERANCE = 1e-4
_BOUNDS_TOLERANCE_FRACTION = 0.45
_MIN_AXIS_BOUNDARY_TOLERANCE = 1e-12
_OUTLET_TOLERANCE_FRACTION = 0.25
_WALL_FRICTION = 0.8  # speed reduction when sliding along wall

# ---------------------------------------------------------------------------
# Optional JAX import
# ---------------------------------------------------------------------------

try:
    import jax
    import jax.numpy as jnp
    _JAX = True
except Exception:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _JAX = False


def jax_backend_name() -> str | None:
    if not _JAX or jax is None:
        return None
    try:
        return str(jax.default_backend())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# JAX JIT kernel
# ---------------------------------------------------------------------------

if _JAX and jax is not None and jnp is not None:

    @jax.jit
    def _jax_step(
        positions: Any, speed_scale: Any,
        source_positions: Any, fluid_keys: Any,
        origin: Any, axis_i: Any, axis_j: Any, axis_k: Any,
        low: Any, high: Any, direction: Any, key: Any,
        axis_tolerance: Any, outlet_tolerance: Any,
        axis_mask: Any, cross_centroid: Any, cross_max_r: float,
        flow_axis: int,
        speed: float, diffusion: float, dt: float, dx: float,
        nx: int, ny: int, nz: int, proj_min: float, proj_max: float,
    ) -> tuple[Any, Any, Any]:
        n = positions.shape[0]
        key, k_drift, k_respawn, k_source, k_speed = jax.random.split(key, 5)

        # Advect + diffuse (tangent only – no through-boundary drift)
        noise = jax.random.normal(k_drift, positions.shape, dtype=jnp.float32)
        tangent = noise - jnp.sum(noise * direction, axis=1, keepdims=True) * direction
        tangent = tangent * axis_mask[None, :]
        adv = direction * (speed * speed_scale[:, None] * dt)
        cand = positions + adv + tangent * jnp.sqrt(jnp.maximum(dt, 0.0)) * diffusion * dx

        # Outlet wrap (particles that pass outlet reappear at inlet)
        proj = jnp.sum(cand * direction, axis=1)
        flow_span = jnp.maximum(
            proj_max - proj_min,
            jnp.float32(1e-12),
        )
        needs_wrap = (proj < proj_min) | (proj > proj_max)
        wrapped_proj = proj_min + jnp.mod(proj - proj_min, flow_span)
        wrapped_delta = (wrapped_proj - proj) * needs_wrap.astype(jnp.float32)
        cand = cand + wrapped_delta[:, None] * direction

        # Fluid-cell membership via sorted-key binary search
        rel = cand - origin
        ci = jnp.rint(jnp.sum(rel * axis_i, axis=1) / dx).astype(jnp.int32)
        cj = jnp.rint(jnp.sum(rel * axis_j, axis=1) / dx).astype(jnp.int32)
        ck = jnp.rint(jnp.sum(rel * axis_k, axis=1) / dx).astype(jnp.int32)
        in_bounds = (
            (ci >= 0) & (ci < jnp.int32(nx)) &
            (cj >= 0) & (cj < jnp.int32(ny)) &
            (ck >= 0) & (ck < jnp.int32(nz))
        )
        keys = ci + jnp.int32(nx) * (cj + jnp.int32(ny) * ck)
        idx = jnp.searchsorted(fluid_keys, keys, side="left")
        clipped = jnp.clip(idx, 0, jnp.maximum(fluid_keys.shape[0] - 1, 0))
        occupied = (idx < fluid_keys.shape[0]) & (fluid_keys[clipped] == keys)

        in_fluid = in_bounds & occupied

        # Separate wall hits from outlet wraps
        hit_wall = ~in_fluid & ~needs_wrap   # lateral → follow wall
        hit_outlet = ~in_fluid & needs_wrap   # passed outlet → respawn

        # Wall hits: slide along wall tangent (project flow onto wall surface)
        move = cand - positions
        move_len = jnp.sqrt(jnp.sum(move ** 2, axis=1, keepdims=True))
        wall_normal = move / jnp.maximum(move_len, jnp.float32(1e-12))
        # Remove wall-normal component from flow direction → tangent flow
        flow_dot_n = jnp.sum(direction * wall_normal, axis=1, keepdims=True)
        tang_dir = direction - flow_dot_n * wall_normal
        t_len = jnp.sqrt(jnp.sum(tang_dir ** 2, axis=1, keepdims=True))
        tang_dir = jnp.where(t_len > jnp.float32(1e-6), tang_dir / t_len, jnp.float32(0.0))
        wall_pos = positions + tang_dir * (jnp.float32(_WALL_FRICTION) * speed * speed_scale[:, None] * dt)

        # Validate wall_pos is in fluid; fall back to current position if not
        wrel = wall_pos - origin
        wci = jnp.rint(jnp.sum(wrel * axis_i, axis=1) / dx).astype(jnp.int32)
        wcj = jnp.rint(jnp.sum(wrel * axis_j, axis=1) / dx).astype(jnp.int32)
        wck = jnp.rint(jnp.sum(wrel * axis_k, axis=1) / dx).astype(jnp.int32)
        w_in_bounds = (
            (wci >= 0) & (wci < jnp.int32(nx)) &
            (wcj >= 0) & (wcj < jnp.int32(ny)) &
            (wck >= 0) & (wck < jnp.int32(nz))
        )
        w_keys = wci + jnp.int32(nx) * (wcj + jnp.int32(ny) * wck)
        w_idx = jnp.searchsorted(fluid_keys, w_keys, side="left")
        w_clipped = jnp.clip(w_idx, 0, jnp.maximum(fluid_keys.shape[0] - 1, 0))
        w_occupied = (w_idx < fluid_keys.shape[0]) & (fluid_keys[w_clipped] == w_keys)
        wall_ok = w_in_bounds & w_occupied
        wall_final = jnp.where(wall_ok[:, None], wall_pos, positions)

        # Respawn at inlet for outlet hits
        src_idx = jax.random.randint(k_source, (n,), 0, jnp.maximum(source_positions.shape[0], 1))
        jitter = jax.random.normal(k_respawn, positions.shape, dtype=jnp.float32) * jnp.float32(dx * 0.18)
        tang_jitter = jitter - jnp.sum(jitter * direction, axis=1, keepdims=True) * direction
        tang_jitter = tang_jitter * axis_mask[None, :]
        replacement = source_positions[src_idx] + tang_jitter

        # Compose: outlet→respawn, wall→slide, valid→advance
        next_pos = jnp.where(
            hit_outlet[:, None], replacement,
            jnp.where(hit_wall[:, None], wall_final, cand),
        )
        # Laminar profile for respawned particles
        rep_proj = jnp.sum(replacement * direction, axis=1, keepdims=True)
        rep_cross = replacement - rep_proj * direction
        rep_r = jnp.sqrt(jnp.sum((rep_cross - cross_centroid) ** 2, axis=1))
        rep_t = jnp.clip(rep_r / jnp.maximum(cross_max_r, jnp.float32(1e-12)), 0.0, 1.0)
        rep_profile = jnp.clip(1.0 - rep_t * rep_t, 0.05, 1.0)
        next_scale = jnp.where(hit_outlet, rep_profile, speed_scale)
        return next_pos, next_scale, key

    class _JaxEngine:
        def __init__(
            self,
            positions: F32Array,
            speed_scale: F32Array,
            source_positions: F32Array,
            fluid_keys: NDArray[np.uint64],
            origin: FloatArray,
            axis_i: FloatArray, axis_j: FloatArray, axis_k: FloatArray,
            low: FloatArray, high: FloatArray,
            direction: FloatArray,
            axis_tolerance: FloatArray,
            outlet_tolerance: float,
            axis_mask: FloatArray,
            cross_centroid: F32Array,
            cross_max_r: float,
            flow_axis: int,
            dx: float, nx: int, ny: int, nz: int,
            proj_min: float, proj_max: float, seed: int,
        ) -> None:
            int_keys = fluid_keys.astype(np.int64, copy=False)
            if np.any(int_keys < 0) or np.any(int_keys > np.iinfo(np.int32).max):
                raise OverflowError("lattice keys exceed int32 — too large for JAX kernel")
            self._p = jnp.asarray(positions, dtype=jnp.float32)
            self._s = jnp.asarray(speed_scale, dtype=jnp.float32)
            self._src = jnp.asarray(source_positions, dtype=jnp.float32)
            self._fk = jnp.asarray(int_keys.astype(np.int32), dtype=jnp.int32)
            self._orig = jnp.asarray(origin, dtype=jnp.float32)
            self._ai = jnp.asarray(axis_i, dtype=jnp.float32)
            self._aj = jnp.asarray(axis_j, dtype=jnp.float32)
            self._ak = jnp.asarray(axis_k, dtype=jnp.float32)
            self._low = jnp.asarray(low, dtype=jnp.float32)
            self._high = jnp.asarray(high, dtype=jnp.float32)
            self._dir = jnp.asarray(direction, dtype=jnp.float32)
            self._axis_tol = jnp.asarray(axis_tolerance, dtype=jnp.float32)
            self._out_tol = float(outlet_tolerance)
            self._axis_mask = jnp.asarray(axis_mask, dtype=jnp.float32)
            self._cross_centroid = jnp.asarray(cross_centroid, dtype=jnp.float32)
            self._cross_max_r = float(cross_max_r)
            self._flow_axis = int(flow_axis)
            self._dx = float(dx)
            self._nx = int(nx)
            self._ny = int(ny)
            self._nz = int(nz)
            self._pm = float(proj_max)
            self._pmn = float(proj_min)
            self._key = jax.random.PRNGKey(seed)

        def step(self, speed: float, diffusion: float, dt: float) -> F32Array:
            self._p, self._s, self._key = _jax_step(
                self._p, self._s, self._src, self._fk,
                self._orig, self._ai, self._aj, self._ak,
                self._low, self._high, self._dir, self._key,
                self._axis_tol, self._out_tol,
                self._axis_mask, self._cross_centroid, self._cross_max_r,
                self._flow_axis,
                float(speed), float(diffusion), float(dt),
                self._dx, self._nx, self._ny, self._nz, self._pmn, self._pm,
            )
            return np.asarray(self._p, dtype=np.float32)

else:

    class _JaxEngine:  # type: ignore[no-redef]
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("JAX is not available")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _norm(v: FloatArray) -> FloatArray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array((1.0, 0.0, 0.0), dtype=np.float64)


def _inlet_positions(
    fluid_pts: F32Array,
    direction: FloatArray,
    proj_min: float,
    proj_max: float,
    axis_cells: int | None = None,
    dx: float = 1.0,
) -> F32Array:
    span = max(proj_max - proj_min, 1e-6)
    proj = fluid_pts @ direction
    n_cells = int(axis_cells) if axis_cells and axis_cells > 0 else 0
    if n_cells <= 0:
        width = 0.14 * span
    else:
        cells = max(1, min(4, max(1, int(round(n_cells * 0.02)))))
        width = min(0.5 * span, cells * max(float(dx), 1e-12))
    mask = proj <= proj_min + width
    src = fluid_pts[mask]
    return src if src.size else fluid_pts


def _axis_activity_mask(
    low: FloatArray,
    high: FloatArray,
    dimension: int,
    nx: int,
    ny: int,
    nz: int,
) -> NDArray[np.bool_]:
    span = high - low
    spans = np.asarray(span, dtype=np.float64)
    counts = np.array((nx > 1, ny > 1, nz > 1), dtype=np.bool_)
    mask = counts & np.isfinite(spans) & (np.abs(spans) > _PLANAR_AXIS_EPS)

    target = int(dimension)
    if target <= 0:
        target = 1
    if np.sum(mask) <= target:
        return mask

    # Keep only the `target` widest axes when more than one dimension is collapsed
    # by numeric noise but metadata says movement is lower-dimensional.
    keep = set(np.argsort(np.abs(spans))[-target:].tolist())
    final = np.zeros(3, dtype=np.bool_)
    for idx in keep:
        final[int(idx)] = bool(mask[idx])
    return final


def _flow_axis_index(direction: FloatArray) -> int:
    d = np.asarray(direction, dtype=np.float64)
    if d.size == 0:
        return 0
    mag = np.abs(d)
    if not np.any(np.isfinite(mag)):
        return 0
    return int(np.argmax(mag))


@dataclass
class FlowMapSimulation:
    lattice: FlowLatticeData
    axis_bounds_tolerance: F32Array
    axis_motion_mask: NDArray[np.bool_]
    flow_axis: int
    direction: FloatArray
    direction_f32: F32Array
    origin: FloatArray
    axis_i: FloatArray
    axis_j: FloatArray
    axis_k: FloatArray
    low: FloatArray
    high: FloatArray
    proj_min: float
    proj_max: float
    outlet_tolerance: float
    speed: float
    diffusion: float
    cross_centroid: F32Array   # cross-section centroid (perpendicular to flow)
    cross_max_r: float         # max radius in cross-section
    trail_length: int
    streamlines_enabled: bool
    speed_scale: F32Array
    positions: F32Array
    trail_positions: F32Array
    trail_particle_indices: NDArray[np.intp]
    trail_cursor: int
    trail_count: int
    fluid_positions: F32Array
    fluid_keys: NDArray[np.uint64]
    inlet_positions: F32Array
    rng: np.random.Generator
    _jax: _JaxEngine | None = field(default=None, init=False, repr=False)

    @property
    def backend_name(self) -> str:
        return f"JAX/{jax_backend_name() or 'unknown'}" if self._jax else "NumPy"

    @property
    def particle_count(self) -> int:
        return int(self.positions.shape[0])

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_lattice(
        cls,
        lattice: FlowLatticeData,
        particle_count: int = 1800,
        velocity: float = 5.0,
        diffusion: float = 0.08,
        streamline_length: int = _DEFAULT_SL_STEPS,
        seed: int = 11,
    ) -> "FlowMapSimulation":
        direction = _norm(np.asarray(lattice.flow_direction, dtype=np.float64))
        low = lattice.positions.min(0).astype(np.float64)
        high = lattice.positions.max(0).astype(np.float64)
        proj = lattice.positions @ direction
        proj_min, proj_max = float(proj.min()), float(proj.max())

        fluid_mask = lattice.fluid_mask
        if not fluid_mask.any():
            fluid_mask = np.ones(len(lattice.positions), dtype=np.bool_)
        fluid_pos = lattice.positions[fluid_mask].astype(np.float32)
        origin = np.asarray(lattice.lattice_origin, dtype=np.float64)
        derived_origin = _origin_from_indices(
            fluid_pos,
            np.asarray(lattice.axis_i, dtype=np.float64),
            np.asarray(lattice.axis_j, dtype=np.float64),
            np.asarray(lattice.axis_k, dtype=np.float64),
            lattice.i[fluid_mask],
            lattice.j[fluid_mask],
            lattice.k[fluid_mask],
            lattice.dx,
        )
        if np.linalg.norm(derived_origin - origin) > max(abs(lattice.dx), 1e-12) * 1e-3:
            origin = derived_origin

        flow_axis = _flow_axis_index(direction)
        axis_cells = (
            (lattice.nx, lattice.ny, lattice.nz)[flow_axis]
            if 0 <= flow_axis < 3 else None
        )
        inlet_pos = _inlet_positions(
            fluid_pos,
            direction,
            proj_min,
            proj_max,
            axis_cells=axis_cells,
            dx=lattice.dx,
        )
        if not inlet_pos.size:
            inlet_pos = fluid_pos

        # Cross-section geometry for parabolic (laminar) velocity profile
        proj_along = fluid_pos @ direction.astype(np.float32)
        cross_coords = fluid_pos - np.outer(proj_along, direction.astype(np.float32))
        cross_centroid = cross_coords.mean(0).astype(np.float32)
        cross_r = np.linalg.norm(cross_coords - cross_centroid, axis=1)
        cross_max_r = max(float(cross_r.max()), 1e-12)

        n = max(1, int(particle_count))
        steps = max(2, int(streamline_length))
        rng = np.random.default_rng(seed)

        fluid_keys = np.sort(_make_keys(
            lattice.i[fluid_mask], lattice.j[fluid_mask], lattice.k[fluid_mask],
            lattice.nx, lattice.ny,
        ))

        sim = cls(
            lattice=lattice,
            axis_bounds_tolerance=_axis_bounds_tolerance(low, high, lattice.dx),
            axis_motion_mask=_axis_activity_mask(
                low,
                high,
                lattice.dimension,
                lattice.nx,
                lattice.ny,
                lattice.nz,
            ),
            flow_axis=flow_axis,
            direction=direction,
            direction_f32=direction.astype(np.float32),
            origin=origin,
            axis_i=np.asarray(lattice.axis_i, dtype=np.float64),
            axis_j=np.asarray(lattice.axis_j, dtype=np.float64),
            axis_k=np.asarray(lattice.axis_k, dtype=np.float64),
            low=low, high=high,
            proj_min=proj_min, proj_max=proj_max,
            outlet_tolerance=_outlet_tolerance(proj_max - proj_min),
            speed=max(float(velocity), 0.001),
            diffusion=max(float(diffusion), 0.0),
            cross_centroid=cross_centroid,
            cross_max_r=cross_max_r,
            trail_length=steps,
            streamlines_enabled=False,
            speed_scale=np.ones(n, dtype=np.float32),  # set by _update_speed_profile
            positions=np.empty((n, 3), dtype=np.float32),
            trail_positions=np.empty((1, 0, 3), dtype=np.float32),
            trail_particle_indices=np.empty(0, dtype=np.intp),
            trail_cursor=0, trail_count=0,
            fluid_positions=fluid_pos,
            fluid_keys=fluid_keys,
            inlet_positions=inlet_pos.astype(np.float32),
            rng=rng,
        )
        sim._seed_particles()
        sim._update_speed_profile()
        sim._init_jax(seed)
        sim._warmup_jax()
        return sim

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, dt: float = 0.016) -> None:
        if self._jax is not None:
            self.positions[:] = self._jax.step(self.speed, self.diffusion, float(dt))
            if self.streamlines_enabled:
                self._record_trail()
            return

        # NumPy fallback
        base_vel = self.direction[None, :] * (self.speed * self.speed_scale[:, None])
        noise = self._diffusion_noise(dt)
        cand = self.positions + base_vel * float(dt) + noise
        proj = cand @ self.direction
        flow_span = float(self.proj_max - self.proj_min)
        if flow_span <= 0.0:
            flow_span = 1e-12
        needs_wrap = (proj < self.proj_min) | (proj > self.proj_max)
        wrapped_proj = self.proj_min + np.mod(proj - self.proj_min, flow_span)
        wrapped_delta = (wrapped_proj - proj) * needs_wrap
        cand = cand + wrapped_delta[:, None] * self.direction

        in_fluid = self._in_fluid(cand)
        hit_wall = ~in_fluid & ~needs_wrap
        hit_outlet = ~in_fluid & needs_wrap

        # Wall hits: slide along wall tangent with reduced speed
        if hit_wall.any():
            hw_idx = np.where(hit_wall)[0]
            hw_pos = self.positions[hw_idx]
            hw_cand = cand[hw_idx]
            # Wall normal estimate from failed movement direction
            move = hw_cand - hw_pos
            move_len = np.sqrt(np.sum(move ** 2, axis=1, keepdims=True))
            wall_normal = move / np.maximum(move_len, 1e-12)
            # Project flow direction onto wall tangent plane
            d = self.direction_f32[None, :]
            flow_dot_n = np.sum(d * wall_normal, axis=1, keepdims=True)
            tang_dir = d - flow_dot_n * wall_normal
            t_len = np.sqrt(np.sum(tang_dir ** 2, axis=1, keepdims=True))
            tang_dir = np.where(t_len > 1e-6, tang_dir / t_len, 0.0)
            # Slide along tangent with friction
            wall_slide = hw_pos + tang_dir * (
                _WALL_FRICTION * self.speed * self.speed_scale[hw_idx, None] * float(dt)
            )
            wall_slide = wall_slide.astype(np.float32)
            # Only accept if the slid position is in fluid
            slide_ok = self._in_fluid(wall_slide)
            cand[hw_idx] = np.where(slide_ok[:, None], wall_slide, hw_pos)

        # Outlet hits: respawn at inlet
        if hit_outlet.any():
            self._respawn(cand, hit_outlet)

        self.positions[:] = cand
        if self.streamlines_enabled:
            self._record_trail()

    # ------------------------------------------------------------------
    # Streamlines
    # ------------------------------------------------------------------

    def set_streamlines_enabled(self, enabled: bool) -> None:
        self.streamlines_enabled = bool(enabled)
        if not enabled:
            self.trail_count = 0
            self.trail_cursor = 0
            self.trail_positions = np.empty((1, 0, 3), dtype=np.float32)
            self.trail_particle_indices = np.empty(0, dtype=np.intp)
            return
        n = self.positions.shape[0]
        if n == 0:
            self.trail_positions = np.empty((1, 0, 3), dtype=np.float32)
            self.trail_particle_indices = np.empty(0, dtype=np.intp)
            self.trail_count = 0
            return
        tracked = min(n, _DEFAULT_SL_PARTICLE_LIMIT)
        self.trail_particle_indices = np.linspace(0, n - 1, tracked, dtype=np.intp)
        self.trail_positions = np.empty((self.trail_length, tracked, 3), dtype=np.float32)
        self.trail_positions[:] = self.positions[self.trail_particle_indices][None]
        self.trail_cursor = 0
        self.trail_count = 1

    def streamline_segments_limited(self, max_segs: int = 0) -> NDArray[np.float32]:
        if not self.streamlines_enabled or self.trail_count < 2:
            return np.empty((0, 2, 3), dtype=np.float32)
        tracked = self.trail_particle_indices.shape[0]
        if tracked == 0:
            return np.empty((0, 2, 3), dtype=np.float32)
        full = self.trail_count >= self.trail_length
        steps = (self.trail_length - 1) if full else (self.trail_count - 1)
        if full:
            order = (np.arange(self.trail_length) + self.trail_cursor + 1) % self.trail_length
        else:
            order = np.arange(self.trail_count, dtype=np.intp)
        total = steps * tracked
        if total <= 0:
            return np.empty((0, 2, 3), dtype=np.float32)
        if max_segs > 0 and max_segs < total:
            sample = np.linspace(0, total - 1, max_segs, dtype=np.intp)
        else:
            sample = np.arange(total, dtype=np.intp)
        ts = sample // tracked
        ps = sample % tracked
        start = self.trail_positions[order[ts], ps]
        stop = self.trail_positions[order[ts + 1], ps]
        return np.stack((start.astype(np.float32), stop.astype(np.float32)), axis=1)

    def particle_vertices(self) -> F32Array:
        return self.positions.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_jax(self, seed: int) -> None:
        if not _JAX or self.positions.shape[0] == 0:
            return
        try:
            self._jax = _JaxEngine(
                positions=self.positions,
                speed_scale=self.speed_scale,
                source_positions=self.inlet_positions,
                fluid_keys=self.fluid_keys,
                origin=self.origin,
                axis_i=self.axis_i,
                axis_j=self.axis_j,
                axis_k=self.axis_k,
                low=self.low, high=self.high,
                direction=self.direction,
                axis_tolerance=self.axis_bounds_tolerance,
                outlet_tolerance=self.outlet_tolerance,
                axis_mask=self.axis_motion_mask.astype(np.float32),
                cross_centroid=self.cross_centroid,
                cross_max_r=self.cross_max_r,
                flow_axis=int(self.flow_axis),
                dx=float(self.lattice.dx),
                nx=self.lattice.nx, ny=self.lattice.ny, nz=self.lattice.nz,
                proj_min=self.proj_min, proj_max=self.proj_max, seed=seed,
            )
        except Exception:
            self._jax = None

    def _warmup_jax(self) -> None:
        """Pre-compile JAX JIT kernels so the first real step has no stutter."""
        if self._jax is None:
            return
        try:
            for _ in range(3):
                self._jax.step(self.speed, self.diffusion, 0.016)
        except Exception:
            self._jax = None

    def _laminar_profile(self, pts: F32Array) -> F32Array:
        """Parabolic velocity profile: fastest at center, zero at walls."""
        proj_along = pts @ self.direction_f32
        cross = pts - np.outer(proj_along, self.direction_f32)
        r = np.linalg.norm(cross - self.cross_centroid, axis=1)
        t = np.clip(r / self.cross_max_r, 0.0, 1.0)
        # Parabolic: v(r) = v_max * (1 - (r/R)^2), clamped to [0.05, 1.0]
        return np.clip(1.0 - t * t, 0.05, 1.0).astype(np.float32)

    def _update_speed_profile(self) -> None:
        """Set speed_scale from laminar parabolic profile based on position."""
        self.speed_scale[:] = self._laminar_profile(self.positions)

    def _seed_particles(self) -> None:
        source = self.inlet_positions if self.inlet_positions.size else self.fluid_positions
        self.positions[:] = self._sample(source, self.positions.shape[0])

    def _record_trail(self) -> None:
        if self.trail_particle_indices.size == 0:
            return
        self.trail_cursor = (self.trail_cursor + 1) % self.trail_length
        self.trail_positions[self.trail_cursor] = self.positions[self.trail_particle_indices]
        if self.trail_count < self.trail_length:
            self.trail_count += 1

    def _respawn(self, cand: F32Array, mask: NDArray[np.bool_]) -> None:
        n = int(mask.sum())
        if n == 0:
            return
        pts = self._sample(self.inlet_positions, n)
        cand[mask] = pts
        self.positions[mask] = pts
        self.speed_scale[mask] = self._laminar_profile(pts)

    def _diffusion_noise(self, dt: float) -> F32Array:
        noise = self.rng.standard_normal(self.positions.shape).astype(np.float32)
        d = self.direction_f32
        tang = noise - (noise @ d)[:, None] * d[None, :]
        tang = tang * self.axis_motion_mask.astype(np.float32)[None, :]
        return tang * sqrt(max(float(dt), 0.0)) * self.diffusion * float(self.lattice.dx)

    def _movement_side_axes(self) -> NDArray[np.bool_]:
        side = np.array(self.axis_motion_mask, copy=True)
        axis = int(self.flow_axis)
        if 0 <= axis < side.size:
            side[axis] = False
        return side

    def _clamp_to_side_axes(self, pts: F32Array) -> F32Array:
        side = self._movement_side_axes()
        if not np.any(side):
            return pts
        lo = self.low[side] + self.axis_bounds_tolerance[side]
        hi = self.high[side] - self.axis_bounds_tolerance[side]
        pts[:, side] = np.clip(pts[:, side], lo, hi)
        return pts

    def _outside_box(self, pts: F32Array) -> NDArray[np.bool_]:
        if not np.any(self.axis_motion_mask):
            return np.zeros(pts.shape[0], dtype=np.bool_)
        tol = self.axis_bounds_tolerance
        side = self._movement_side_axes()
        if not np.any(side):
            return np.zeros(pts.shape[0], dtype=np.bool_)
        return np.any(
            (pts[:, side] < self.low[side] + tol[side])
            | (pts[:, side] > self.high[side] - tol[side]),
            axis=1,
        )

    def _in_fluid(self, pts: F32Array) -> NDArray[np.bool_]:
        rel = pts - self.origin[None, :]
        i = np.rint((rel @ self.axis_i) / self.lattice.dx).astype(np.int64)
        j = np.rint((rel @ self.axis_j) / self.lattice.dx).astype(np.int64)
        k = np.rint((rel @ self.axis_k) / self.lattice.dx).astype(np.int64)
        in_bounds = (
            (i >= 0) & (i < self.lattice.nx) &
            (j >= 0) & (j < self.lattice.ny) &
            (k >= 0) & (k < self.lattice.nz)
        )
        keys = _make_keys(i, j, k, self.lattice.nx, self.lattice.ny)
        idx = np.searchsorted(self.fluid_keys, keys)
        safe = np.clip(idx, 0, self.fluid_keys.size - 1)
        return in_bounds & (idx < self.fluid_keys.size) & (self.fluid_keys[safe] == keys)

    def _sample(self, source: F32Array, n: int) -> F32Array:
        idx = self.rng.integers(0, source.shape[0], n)
        pts = source[idx].copy()
        jitter = self.rng.standard_normal(pts.shape).astype(np.float32) * float(self.lattice.dx * 0.18)
        d = self.direction_f32
        tang = jitter - (jitter @ d)[:, None] * d[None, :]
        tang = tang * self.axis_motion_mask.astype(np.float32)[None, :]
        pts += tang
        bad = ~self._in_fluid(pts)
        if bad.any():
            pts[bad] = source[self.rng.integers(0, source.shape[0], int(bad.sum()))]
        return pts


def _axis_bounds_tolerance(
    low: FloatArray,
    high: FloatArray,
    dx: float,
) -> F32Array:
    span = np.abs(np.asarray(high, dtype=np.float64) - np.asarray(low, dtype=np.float64))
    cell = float(dx) if np.isfinite(float(dx)) and float(dx) > 0 else 1.0
    tol = np.minimum(
        span * _BOUNDS_TOLERANCE_FRACTION,
        np.full(3, 0.5 * cell, dtype=np.float64),
    )
    tol = np.minimum(tol, _MAX_BOUNDARY_TOLERANCE)
    return np.maximum(tol, _MIN_AXIS_BOUNDARY_TOLERANCE).astype(np.float32)


def _origin_from_indices(
    positions: F32Array,
    axis_i: FloatArray,
    axis_j: FloatArray,
    axis_k: FloatArray,
    i: NDArray[np.int64],
    j: NDArray[np.int64],
    k: NDArray[np.int64],
    dx: float,
) -> FloatArray:
    if positions.size == 0 or i.size == 0:
        return np.array((0.0, 0.0, 0.0), dtype=np.float64)
    ax_i = np.asarray(axis_i, dtype=np.float64)
    ax_j = np.asarray(axis_j, dtype=np.float64)
    ax_k = np.asarray(axis_k, dtype=np.float64)
    rel = positions.astype(np.float64) - (
        np.asarray(i, dtype=np.float64)[:, None] * ax_i
        + np.asarray(j, dtype=np.float64)[:, None] * ax_j
        + np.asarray(k, dtype=np.float64)[:, None] * ax_k
    ) * float(dx)
    if not np.isfinite(rel).all():
        return np.array((0.0, 0.0, 0.0), dtype=np.float64)
    return np.median(rel, axis=0)


def _outlet_tolerance(span: float) -> float:
    span = float(span)
    if not np.isfinite(span) or span <= 0.0:
        return _MIN_AXIS_BOUNDARY_TOLERANCE
    return float(
        max(
            _MIN_AXIS_BOUNDARY_TOLERANCE,
            min(_MAX_BOUNDARY_TOLERANCE, span * _OUTLET_TOLERANCE_FRACTION),
        ),
    )

def _make_keys(
    i: NDArray[np.int64], j: NDArray[np.int64], k: NDArray[np.int64],
    nx: int, ny: int,
) -> NDArray[np.uint64]:
    return (
        i.astype(np.uint64)
        + np.uint64(nx) * (j.astype(np.uint64) + np.uint64(ny) * k.astype(np.uint64))
    )
