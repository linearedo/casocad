from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.io import read_lattice


VELOCITIES = np.asarray(
    (
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        (1, 1),
        (-1, 1),
        (-1, -1),
        (1, -1),
    ),
    dtype=np.int8,
)
OPPOSITE = np.asarray((0, 3, 4, 1, 2, 7, 8, 5, 6), dtype=np.int8)
WEIGHTS = np.asarray(
    (4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
     1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0),
    dtype=np.float64,
)

REQUIRED_REGIONS = ("lateral_up", "lateral_down", "inlet", "outlet", "wall")
REQUIRED_TAGGED_REGIONS = ("lateral_up", "lateral_down", "inlet", "outlet")
VIDEO_FRAMES_PER_SECOND = 30
DEFAULT_RENDER_SCALE = 4
MIN_RECOMMENDED_OBSTACLE_CELLS = 20.0


@dataclass(frozen=True)
class ArrowGeometry2D:
    nx: int
    ny: int
    dx: float
    inflow_direction: int
    fluid: NDArray[np.bool_]
    boundary: NDArray[np.bool_]
    regions: dict[str, NDArray[np.bool_]]

    @property
    def no_slip(self) -> NDArray[np.bool_]:
        return self.regions["wall"]

    @property
    def inlet(self) -> NDArray[np.bool_]:
        return self.regions["inlet"] & ~self.no_slip

    @property
    def outlet(self) -> NDArray[np.bool_]:
        return self.regions["outlet"] & ~self.no_slip

    @property
    def open_boundary(self) -> NDArray[np.bool_]:
        return self.inlet | self.outlet

    @property
    def flow(self) -> NDArray[np.bool_]:
        return self.fluid & ~self.no_slip


def _normalized_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def load_arrow_geometry(path: str | Path) -> ArrowGeometry2D:
    table, metadata = read_lattice(path)
    grid = metadata.get("grid", {})
    if metadata.get("dimension") != 2 or grid.get("dimension") != 2:
        raise ValueError("LBM example expects a 2D casoCAD lattice Arrow file")

    nx = int(grid["nx"])
    ny = int(grid["ny"])
    dx = float(grid["dx"])
    i_values = table["i"].to_numpy(zero_copy_only=False).astype(np.int64)
    j_values = table["j"].to_numpy(zero_copy_only=False).astype(np.int64)
    node_type = table["node_type"].to_numpy(zero_copy_only=False)
    tag_ids = table["tag_ids"].to_pylist()

    object_ids_by_name = _object_ids_by_name(metadata)
    missing = [
        name for name in REQUIRED_TAGGED_REGIONS if name not in object_ids_by_name
    ]
    if missing:
        names = ", ".join(sorted(object_ids_by_name))
        raise ValueError(
            "Arrow metadata is missing required LBM regions "
            f"{missing}; available names: {names}"
        )

    fluid = np.zeros((nx, ny), dtype=np.bool_)
    boundary = np.zeros((nx, ny), dtype=np.bool_)
    fluid[i_values, j_values] = True
    boundary[i_values, j_values] = node_type == 1

    regions: dict[str, NDArray[np.bool_]] = {}
    tag_sets = [set(int(item) for item in row) for row in tag_ids]
    for name in REQUIRED_TAGGED_REGIONS:
        object_id = object_ids_by_name[name]
        mask = np.zeros((nx, ny), dtype=np.bool_)
        row_mask = np.fromiter(
            (object_id in items for items in tag_sets),
            dtype=np.bool_,
            count=len(tag_sets),
        )
        mask[i_values[row_mask], j_values[row_mask]] = True
        if not mask.any():
            raise ValueError(f"required LBM region '{name}' has no tagged nodes")
        regions[name] = mask

    wall = np.zeros((nx, ny), dtype=np.bool_)
    wall_object_id = object_ids_by_name.get("wall")
    if wall_object_id is not None:
        wall_row_mask = np.fromiter(
            (wall_object_id in items for items in tag_sets),
            dtype=np.bool_,
            count=len(tag_sets),
        )
        wall[i_values[wall_row_mask], j_values[wall_row_mask]] = True
    if not wall.any():
        assigned_boundary = np.zeros((nx, ny), dtype=np.bool_)
        for mask in regions.values():
            assigned_boundary |= mask
        wall = boundary & ~assigned_boundary
    regions["wall"] = wall
    inflow_direction = (
        1
        if np.mean(np.nonzero(regions["inlet"])[0])
        < np.mean(np.nonzero(regions["outlet"])[0])
        else -1
    )

    return ArrowGeometry2D(
        nx=nx,
        ny=ny,
        dx=dx,
        inflow_direction=inflow_direction,
        fluid=fluid,
        boundary=boundary,
        regions=regions,
    )


def _object_ids_by_name(metadata: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in metadata.get("object_directory", ()):
        result[_normalized_name(str(item["name"]))] = int(item["object_id"])
    return result


def equilibrium(
    density: NDArray[np.float64],
    velocity: NDArray[np.float64],
) -> NDArray[np.float64]:
    projected = (
        velocity[:, :, 0, None] * VELOCITIES[:, 0]
        + velocity[:, :, 1, None] * VELOCITIES[:, 1]
    )
    magnitude_sq = velocity[:, :, 0] ** 2 + velocity[:, :, 1] ** 2
    return (
        density[:, :, None]
        * WEIGHTS
        * (1.0 + 3.0 * projected + 4.5 * projected**2 - 1.5 * magnitude_sq[:, :, None])
    )


def macroscopic(
    populations: NDArray[np.float64],
    fluid: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    density = populations.sum(axis=2)
    density = np.where(fluid, np.maximum(density, 1e-12), 1.0)
    momentum_x = (populations * VELOCITIES[:, 0]).sum(axis=2)
    momentum_y = (populations * VELOCITIES[:, 1]).sum(axis=2)
    velocity = np.zeros((*density.shape, 2), dtype=np.float64)
    velocity[:, :, 0] = np.where(fluid, momentum_x / density, 0.0)
    velocity[:, :, 1] = np.where(fluid, momentum_y / density, 0.0)
    return density, velocity


def estimate_obstacle_length(geometry: ArrowGeometry2D) -> float:
    """Estimate a generic wall-obstacle length in lattice cells."""
    wall_indices = np.argwhere(geometry.regions["wall"] & geometry.fluid)
    if wall_indices.size == 0:
        return 1.0
    transverse_axis = 1
    return max(1.0, float(np.ptp(wall_indices[:, transverse_axis]) + 1))


def apply_pressure_outlet(
    populations: NDArray[np.float64],
    geometry: ArrowGeometry2D,
    *,
    outlet_density: float = 1.0,
) -> None:
    outlet = geometry.outlet
    if geometry.inflow_direction > 0:
        known = outlet
        ux = -1.0 + (
            populations[:, :, 0]
            + populations[:, :, 2]
            + populations[:, :, 4]
            + 2.0
            * (populations[:, :, 1] + populations[:, :, 5] + populations[:, :, 8])
        ) / outlet_density
        uy = np.zeros_like(ux)
        populations[known, 3] = (
            populations[known, 1] - (2.0 / 3.0) * outlet_density * ux[known]
        )
        populations[known, 7] = (
            populations[known, 5]
            + 0.5 * (populations[known, 2] - populations[known, 4])
            - (1.0 / 6.0) * outlet_density * ux[known]
            - 0.5 * outlet_density * uy[known]
        )
        populations[known, 6] = (
            populations[known, 8]
            + 0.5 * (populations[known, 4] - populations[known, 2])
            - (1.0 / 6.0) * outlet_density * ux[known]
            + 0.5 * outlet_density * uy[known]
        )
    else:
        known = outlet
        ux = 1.0 - (
            populations[:, :, 0]
            + populations[:, :, 2]
            + populations[:, :, 4]
            + 2.0
            * (populations[:, :, 3] + populations[:, :, 6] + populations[:, :, 7])
        ) / outlet_density
        uy = np.zeros_like(ux)
        populations[known, 1] = (
            populations[known, 3] + (2.0 / 3.0) * outlet_density * ux[known]
        )
        populations[known, 5] = (
            populations[known, 7]
            + 0.5 * (populations[known, 4] - populations[known, 2])
            + (1.0 / 6.0) * outlet_density * ux[known]
            + 0.5 * outlet_density * uy[known]
        )
        populations[known, 8] = (
            populations[known, 6]
            + 0.5 * (populations[known, 2] - populations[known, 4])
            + (1.0 / 6.0) * outlet_density * ux[known]
            - 0.5 * outlet_density * uy[known]
        )


def apply_velocity_inlet(
    populations: NDArray[np.float64],
    inlet: NDArray[np.bool_],
    inflow_velocity: float,
) -> None:
    ux = float(inflow_velocity)
    uy = 0.0
    if ux >= 0.0:
        density = (
            populations[:, :, 0]
            + populations[:, :, 2]
            + populations[:, :, 4]
            + 2.0
            * (populations[:, :, 3] + populations[:, :, 6] + populations[:, :, 7])
        ) / (1.0 - ux)
        populations[inlet, 1] = populations[inlet, 3] + (2.0 / 3.0) * density[inlet] * ux
        populations[inlet, 5] = (
            populations[inlet, 7]
            + 0.5 * (populations[inlet, 4] - populations[inlet, 2])
            + (1.0 / 6.0) * density[inlet] * ux
            + 0.5 * density[inlet] * uy
        )
        populations[inlet, 8] = (
            populations[inlet, 6]
            + 0.5 * (populations[inlet, 2] - populations[inlet, 4])
            + (1.0 / 6.0) * density[inlet] * ux
            - 0.5 * density[inlet] * uy
        )
    else:
        density = (
            populations[:, :, 0]
            + populations[:, :, 2]
            + populations[:, :, 4]
            + 2.0
            * (populations[:, :, 1] + populations[:, :, 5] + populations[:, :, 8])
        ) / (1.0 + ux)
        populations[inlet, 3] = populations[inlet, 1] - (2.0 / 3.0) * density[inlet] * ux
        populations[inlet, 7] = (
            populations[inlet, 5]
            + 0.5 * (populations[inlet, 2] - populations[inlet, 4])
            - (1.0 / 6.0) * density[inlet] * ux
            - 0.5 * density[inlet] * uy
        )
        populations[inlet, 6] = (
            populations[inlet, 8]
            + 0.5 * (populations[inlet, 4] - populations[inlet, 2])
            - (1.0 / 6.0) * density[inlet] * ux
            + 0.5 * density[inlet] * uy
        )


def stream_with_bounce_back(
    post_collision: NDArray[np.float64],
    geometry: ArrowGeometry2D,
) -> NDArray[np.float64]:
    streamed = np.zeros_like(post_collision)
    fluid = geometry.flow
    open_boundary = geometry.open_boundary
    for q, lattice_velocity in enumerate(VELOCITIES):
        cx, cy = (int(lattice_velocity[0]), int(lattice_velocity[1]))
        if cx == 0 and cy == 0:
            streamed[:, :, q] += np.where(fluid, post_collision[:, :, q], 0.0)
            continue

        if cx >= 0:
            source_x = slice(0, geometry.nx - cx)
            dest_x = slice(cx, geometry.nx)
        else:
            source_x = slice(-cx, geometry.nx)
            dest_x = slice(0, geometry.nx + cx)

        destination = np.zeros_like(fluid)
        destination[source_x, :] = np.roll(fluid[dest_x, :], shift=-cy, axis=1)
        source = fluid[source_x, :]
        can_stream = source & destination[source_x, :]
        streamed[dest_x, :, q] += np.roll(
            np.where(can_stream, post_collision[source_x, :, q], 0.0),
            shift=cy,
            axis=1,
        )

        blocked = fluid & ~destination & ~open_boundary
        streamed[:, :, OPPOSITE[q]] += np.where(
            blocked,
            post_collision[:, :, q],
            0.0,
        )
    streamed[~fluid, :] = 0.0
    return streamed


def compute_curl(velocity: NDArray[np.float64]) -> NDArray[np.float64]:
    ux = velocity[:, :, 0]
    uy = velocity[:, :, 1]
    dudy = np.empty_like(ux)
    dvdx = np.empty_like(uy)
    dudy[:, 1:-1] = 0.5 * (ux[:, 2:] - ux[:, :-2])
    dudy[:, 0] = ux[:, 1] - ux[:, 0]
    dudy[:, -1] = ux[:, -1] - ux[:, -2]
    dvdx[1:-1, :] = 0.5 * (uy[2:, :] - uy[:-2, :])
    dvdx[0, :] = uy[1, :] - uy[0, :]
    dvdx[-1, :] = uy[-1, :] - uy[-2, :]
    return dvdx - dudy


def seed_vortex_shedding_perturbation(
    geometry: ArrowGeometry2D,
    velocity: NDArray[np.float64],
    inflow_velocity: float,
) -> None:
    wall_indices = np.argwhere(geometry.regions["wall"] & geometry.fluid)
    if wall_indices.size == 0:
        return
    obstacle_center_x = float(np.mean(wall_indices[:, 0]))
    obstacle_center_y = float(np.mean(wall_indices[:, 1]))
    obstacle_length = estimate_obstacle_length(geometry)
    x = np.arange(geometry.nx, dtype=np.float64)[:, None]
    y = np.arange(geometry.ny, dtype=np.float64)[None, :]
    downstream = (
        (x - obstacle_center_x) * float(geometry.inflow_direction)
        > 0.25 * obstacle_length
    )
    wake = np.exp(
        -((y - obstacle_center_y) / max(1.0, 0.8 * obstacle_length)) ** 2
    )
    wave = np.sin(2.0 * np.pi * (x - obstacle_center_x) / max(4.0, 6.0 * obstacle_length))
    perturbation = 0.03 * abs(inflow_velocity) * wake * wave
    active = geometry.fluid & ~geometry.no_slip & downstream
    velocity[:, :, 1] = np.where(active, perturbation, velocity[:, :, 1])


def run_simulation(
    geometry: ArrowGeometry2D,
    *,
    iterations: int,
    reynolds_number: float,
    inflow_velocity: float,
    characteristic_length: float | None = None,
    plot_every: int,
    skip_first: int,
    video_path: Path | None,
    render_scale: int = DEFAULT_RENDER_SCALE,
    speed_scale: float | None = None,
    curl_scale: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if render_scale <= 0:
        raise ValueError("render scale must be positive")
    obstacle_length = (
        characteristic_length
        if characteristic_length is not None
        else estimate_obstacle_length(geometry)
    )
    if obstacle_length <= 0.0:
        raise ValueError("characteristic length must be positive")
    inflow_speed = abs(inflow_velocity)
    signed_inflow_velocity = geometry.inflow_direction * inflow_speed
    viscosity = inflow_speed * obstacle_length / reynolds_number
    omega = 1.0 / (3.0 * viscosity + 0.5)
    if not 0.0 < omega < 2.0:
        raise ValueError(
            "unstable relaxation parameter "
            f"omega={omega:.6g}; reduce velocity or Reynolds number"
        )
    render_speed_scale = (
        speed_scale if speed_scale is not None else max(0.02, 1.8 * inflow_speed)
    )
    render_curl_scale = (
        curl_scale
        if curl_scale is not None
        else max(0.003, 6.0 * inflow_speed / obstacle_length)
    )
    if render_speed_scale <= 0.0:
        raise ValueError("speed scale must be positive")
    if render_curl_scale <= 0.0:
        raise ValueError("curl scale must be positive")

    density = np.ones((geometry.nx, geometry.ny), dtype=np.float64)
    velocity = np.zeros((geometry.nx, geometry.ny, 2), dtype=np.float64)
    flow = geometry.flow
    velocity[flow, 0] = signed_inflow_velocity
    seed_vortex_shedding_perturbation(geometry, velocity, signed_inflow_velocity)
    velocity[geometry.no_slip, :] = 0.0
    populations = equilibrium(density, velocity)
    populations[~flow, :] = 0.0
    if (
        video_path is not None
        and characteristic_length is None
        and obstacle_length < MIN_RECOMMENDED_OBSTACLE_CELLS
    ):
        print(
            "warning: obstacle is resolved by only "
            f"{obstacle_length:.0f} lattice cells; use a finer Arrow lattice "
            "for a credible von Karman street"
        )

    video_writer = (
        Mp4Writer(
            video_path,
            width=geometry.nx * render_scale,
            height=(geometry.ny * 2 + 4) * render_scale,
            frames_per_second=VIDEO_FRAMES_PER_SECOND,
        )
        if video_path is not None
        else None
    )
    try:
        for iteration in range(1, iterations + 1):
            density, velocity = macroscopic(populations, flow)
            feq = equilibrium(density, velocity)
            post_collision = populations - omega * (populations - feq)
            post_collision[~flow, :] = 0.0
            populations = stream_with_bounce_back(post_collision, geometry)
            apply_pressure_outlet(populations, geometry)
            density, velocity = macroscopic(populations, flow)
            apply_velocity_inlet(
                populations,
                geometry.inlet,
                signed_inflow_velocity,
            )
            density, velocity = macroscopic(populations, flow)
            velocity[geometry.no_slip, :] = 0.0

            if (
                video_writer is not None
                and iteration > skip_first
                and iteration % plot_every == 0
            ):
                density, velocity = macroscopic(populations, flow)
                velocity[geometry.no_slip, :] = 0.0
                video_writer.write(
                    render_frame(
                        geometry,
                        velocity,
                        scale=render_scale,
                        speed_scale=render_speed_scale,
                        curl_scale=render_curl_scale,
                    )
                )

            if iteration % max(1, iterations // 20) == 0:
                print(f"iteration {iteration}/{iterations}")
    finally:
        if video_writer is not None:
            video_writer.close()

    density, velocity = macroscopic(populations, flow)
    velocity[geometry.no_slip, :] = 0.0
    return density, velocity


class Mp4Writer:
    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        frames_per_second: int,
    ) -> None:
        ffmpeg = _ffmpeg_executable()
        if ffmpeg is None:
            raise RuntimeError(
                "writing MP4 output requires ffmpeg or the imageio-ffmpeg package"
            )
        if frames_per_second <= 0:
            raise ValueError("frames per second must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        command = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(frames_per_second),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._closed = False

    def write(self, image: NDArray[np.uint8]) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed MP4 writer")
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self._process.stdin.write(np.ascontiguousarray(image).tobytes())

    def close(self) -> None:
        if self._closed:
            return
        if self._process.stdin is not None:
            self._process.stdin.close()
        return_code = self._process.wait()
        self._closed = True
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed while writing {self.path}")


def _ffmpeg_executable() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable is not None:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def render_frame(
    geometry: ArrowGeometry2D,
    velocity: NDArray[np.float64],
    *,
    scale: int = 1,
    speed_scale: float | None = None,
    curl_scale: float | None = None,
) -> NDArray[np.uint8]:
    velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
    speed = np.sqrt(velocity[:, :, 0] ** 2 + velocity[:, :, 1] ** 2)
    curl = compute_curl(velocity)
    speed_maximum = (
        speed_scale
        if speed_scale is not None
        else max(0.06, _finite_percentile(speed, geometry.flow, 99.0))
    )
    curl_magnitude = (
        curl_scale
        if curl_scale is not None
        else max(
            0.02,
            _finite_percentile(np.abs(curl), geometry.flow, 99.0),
        )
    )
    speed_rgb = _colorize(speed, geometry.fluid, 0.0, speed_maximum, "ice")
    curl_rgb = _colorize(
        curl,
        geometry.fluid,
        -curl_magnitude,
        curl_magnitude,
        "balance",
    )
    wall = geometry.no_slip & geometry.fluid
    speed_rgb[wall] = (25, 110, 45)
    curl_rgb[wall] = (25, 110, 45)
    separator = np.full((geometry.nx, 4, 3), 20, dtype=np.uint8)
    image = np.concatenate((speed_rgb, separator, curl_rgb), axis=1)
    frame = np.flipud(np.swapaxes(image, 0, 1))
    if scale <= 1:
        return frame
    return _resize_bilinear(frame, scale)


def _resize_bilinear(image: NDArray[np.uint8], scale: int) -> NDArray[np.uint8]:
    if scale <= 1:
        return image
    height, width, channels = image.shape
    output_height = height * scale
    output_width = width * scale
    y = (np.arange(output_height, dtype=np.float64) + 0.5) / scale - 0.5
    x = (np.arange(output_width, dtype=np.float64) + 0.5) / scale - 0.5
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    x0 = np.clip(x0, 0, width - 1)
    wy = (y - y0).clip(0.0, 1.0)
    wx = (x - x0).clip(0.0, 1.0)
    top = (
        image[y0[:, None], x0[None, :], :].astype(np.float64)
        * (1.0 - wx)[None, :, None]
        + image[y0[:, None], x1[None, :], :].astype(np.float64)
        * wx[None, :, None]
    )
    bottom = (
        image[y1[:, None], x0[None, :], :].astype(np.float64)
        * (1.0 - wx)[None, :, None]
        + image[y1[:, None], x1[None, :], :].astype(np.float64)
        * wx[None, :, None]
    )
    resized = top * (1.0 - wy)[:, None, None] + bottom * wy[:, None, None]
    return np.clip(resized, 0.0, 255.0).astype(np.uint8).reshape(
        output_height,
        output_width,
        channels,
    )


def _finite_percentile(
    values: NDArray[np.float64],
    mask: NDArray[np.bool_],
    percentile: float,
) -> float:
    finite_values = values[mask & np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0
    return float(np.percentile(finite_values, percentile))


def _colorize(
    values: NDArray[np.float64],
    fluid: NDArray[np.bool_],
    minimum: float,
    maximum: float,
    palette: str,
) -> NDArray[np.uint8]:
    values = np.nan_to_num(values, nan=minimum, posinf=maximum, neginf=minimum)
    t = np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)
    if palette == "balance":
        low = np.asarray((35, 94, 168), dtype=np.float64)
        mid = np.asarray((28, 31, 36), dtype=np.float64)
        high = np.asarray((214, 80, 56), dtype=np.float64)
        rgb = np.where(
            t[:, :, None] < 0.5,
            low + (mid - low) * (2.0 * t[:, :, None]),
            mid + (high - mid) * (2.0 * t[:, :, None] - 1.0),
        )
    else:
        low = np.asarray((10, 18, 35), dtype=np.float64)
        mid = np.asarray((34, 127, 160), dtype=np.float64)
        high = np.asarray((246, 188, 65), dtype=np.float64)
        rgb = np.where(
            t[:, :, None] < 0.65,
            low + (mid - low) * (t[:, :, None] / 0.65),
            mid + (high - mid) * ((t[:, :, None] - 0.65) / 0.35),
        )
    rgb[~fluid] = (0, 0, 0)
    return rgb.astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a D2Q9 LBM channel simulation from a casoCAD 2D Arrow lattice."
    )
    parser.add_argument("arrow_path", type=Path)
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--reynolds", type=float, default=80.0)
    parser.add_argument("--inflow-velocity", type=float, default=0.04)
    parser.add_argument(
        "--characteristic-length",
        type=float,
        default=None,
        help=(
            "Obstacle length in lattice cells for Reynolds scaling. "
            "Defaults to a wall-tag estimate."
        ),
    )
    parser.add_argument("--plot-every", type=int, default=100)
    parser.add_argument("--skip-first", type=int, default=500)
    parser.add_argument(
        "--render-scale",
        type=int,
        default=DEFAULT_RENDER_SCALE,
        help="Nearest-neighbor scale factor for MP4 frames.",
    )
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=None,
        help=(
            "Fixed velocity magnitude mapped to the top of the video palette. "
            "Defaults to a value derived from inlet speed."
        ),
    )
    parser.add_argument(
        "--curl-scale",
        type=float,
        default=None,
        help=(
            "Fixed absolute vorticity mapped to the ends of the video palette. "
            "Defaults to a value derived from inlet speed and wall-obstacle size."
        ),
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path("lbm.mp4"),
        help="Output MP4 path for side-by-side velocity/vorticity animation.",
    )
    parser.add_argument(
        "--frames-dir",
        dest="legacy_frames_dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-visualization", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geometry = load_arrow_geometry(args.arrow_path)
    video_path = args.video_path
    if args.legacy_frames_dir is not None and video_path == Path("lbm.mp4"):
        video_path = args.legacy_frames_dir.with_suffix(".mp4")
    if args.no_visualization:
        video_path = None
    run_simulation(
        geometry,
        iterations=args.iterations,
        reynolds_number=args.reynolds,
        inflow_velocity=args.inflow_velocity,
        characteristic_length=args.characteristic_length,
        plot_every=args.plot_every,
        skip_first=args.skip_first,
        video_path=video_path,
        render_scale=args.render_scale,
        speed_scale=args.speed_scale,
        curl_scale=args.curl_scale,
    )
    if video_path is not None:
        print(f"wrote visualization video to {video_path}")


if __name__ == "__main__":
    main()
