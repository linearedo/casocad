from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

from core.io.arrow_writer import ArrowWriter


def _load_lbm_module() -> object:
    path = Path(__file__).resolve().parents[1] / "examples" / "lbm_d2q9_arrow.py"
    spec = importlib.util.spec_from_file_location("lbm_d2q9_arrow", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_channel_arrow(
    path: Path,
    *,
    tag_wall: bool = True,
    reverse_flow: bool = False,
    nx: int = 8,
    ny: int = 5,
) -> None:
    object_ids = {
        "inlet": 1,
        "outlet": 2,
        "lateral_up": 3,
        "lateral_down": 4,
        "wall": 5,
    }
    rows: list[tuple[int, int]] = []
    tag_ids: list[list[int]] = []
    node_type: list[int] = []
    if nx >= 24 and ny >= 12:
        center_i = nx // 4
        center_j = ny // 2
        wall_nodes = {
            (i, j)
            for i in range(center_i - 2, center_i + 3)
            for j in range(center_j - 2, center_j + 3)
        }
    else:
        wall_nodes = {(2, 2), (3, 1), (3, 3), (4, 2)}
    for i in range(nx):
        for j in range(ny):
            if (i, j) == (3, 2):
                continue
            tags = []
            if i == 0:
                tags.append(
                    object_ids["outlet" if reverse_flow else "inlet"]
                )
            if i == nx - 1:
                tags.append(
                    object_ids["inlet" if reverse_flow else "outlet"]
                )
            if j == 0:
                tags.append(object_ids["lateral_down"])
            if j == ny - 1:
                tags.append(object_ids["lateral_up"])
            if tag_wall and (i, j) in wall_nodes:
                tags.append(object_ids["wall"])
            rows.append((i, j))
            tag_ids.append(tags)
            node_type.append(int(bool(tags) or (i, j) in wall_nodes))

    i_values = np.asarray([row[0] for row in rows], dtype=np.uint64)
    j_values = np.asarray([row[1] for row in rows], dtype=np.uint64)
    metadata = {
        "dimension": 2,
        "grid": {"dimension": 2, "nx": nx, "ny": ny, "nz": 1, "dx": 0.1},
        "object_directory": [
            {"object_id": object_id, "name": name, "kind": "placed_sdf_1d"}
            for name, object_id in object_ids.items()
        ],
    }
    with ArrowWriter(path, metadata) as writer:
        writer.write_batch(
            x=i_values.astype(np.float64),
            y=j_values.astype(np.float64),
            z=np.zeros(i_values.shape, dtype=np.float64),
            i=i_values,
            j=j_values,
            k=np.zeros(i_values.shape, dtype=np.uint64),
            node_type=np.asarray(node_type, dtype=np.uint8),
            tag_ids=tag_ids,
            level=np.zeros(i_values.shape, dtype=np.uint8),
        )


def test_lbm_arrow_geometry_maps_required_regions(tmp_path) -> None:
    module = _load_lbm_module()
    path = tmp_path / "channel.arrow"
    _write_channel_arrow(path)

    geometry = module.load_arrow_geometry(path)

    assert geometry.nx == 8
    assert geometry.ny == 5
    assert geometry.inflow_direction == 1
    assert not geometry.fluid[3, 2]
    for name in module.REQUIRED_REGIONS:
        assert geometry.regions[name].any()
    assert np.count_nonzero(geometry.regions["inlet"]) == 5
    assert np.count_nonzero(geometry.inlet) == 5
    assert geometry.open_boundary[0, 0]
    assert geometry.open_boundary[0, 4]
    assert not geometry.no_slip[0, 0]
    assert not geometry.no_slip[0, 4]


def test_lbm_arrow_geometry_infers_untagged_wall_boundaries(tmp_path) -> None:
    module = _load_lbm_module()
    path = tmp_path / "channel.arrow"
    _write_channel_arrow(path, tag_wall=False)

    geometry = module.load_arrow_geometry(path)

    wall_indices = set(zip(*np.nonzero(geometry.regions["wall"]), strict=True))
    assert wall_indices == {(2, 2), (3, 1), (3, 3), (4, 2)}


def test_lbm_arrow_geometry_detects_reversed_inlet_outlet(tmp_path) -> None:
    module = _load_lbm_module()
    path = tmp_path / "channel.arrow"
    _write_channel_arrow(path, reverse_flow=True)

    geometry = module.load_arrow_geometry(path)

    assert geometry.inflow_direction == -1


def _write_fake_ffmpeg(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "#!/usr/bin/env python3",
                "from pathlib import Path",
                "import sys",
                "data = sys.stdin.buffer.read()",
                "Path(sys.argv[-1]).write_bytes(b'fake mp4\\n' + data[:16])",
            )
        ),
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | 0o111)


def test_lbm_runs_short_simulation_and_writes_mp4(tmp_path, monkeypatch) -> None:
    module = _load_lbm_module()
    path = tmp_path / "channel.arrow"
    video_path = tmp_path / "lbm.mp4"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_ffmpeg(bin_dir / "ffmpeg")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    _write_channel_arrow(path, nx=48, ny=20)
    geometry = module.load_arrow_geometry(path)

    density, velocity = module.run_simulation(
        geometry,
        iterations=3,
        reynolds_number=40.0,
        inflow_velocity=0.04,
        plot_every=1,
        skip_first=0,
        video_path=video_path,
    )

    assert density.shape == (48, 20)
    assert velocity.shape == (48, 20, 2)
    assert np.isfinite(density[geometry.flow]).all()
    assert np.isfinite(velocity[geometry.flow]).all()
    assert video_path.read_bytes().startswith(b"fake mp4\n")


def test_lbm_streaming_handles_lattices_wider_than_int8(tmp_path) -> None:
    module = _load_lbm_module()
    path = tmp_path / "wide_channel.arrow"
    _write_channel_arrow(path, nx=200)
    geometry = module.load_arrow_geometry(path)

    density, velocity = module.run_simulation(
        geometry,
        iterations=1,
        reynolds_number=80.0,
        inflow_velocity=0.02,
        plot_every=1,
        skip_first=0,
        video_path=None,
    )

    assert density.shape == (200, 5)
    assert velocity.shape == (200, 5, 2)


def test_lbm_streaming_wraps_lateral_boundaries_periodically() -> None:
    module = _load_lbm_module()
    nx = 5
    ny = 4
    fluid = np.ones((nx, ny), dtype=np.bool_)
    false = np.zeros((nx, ny), dtype=np.bool_)
    regions = {
        "inlet": false.copy(),
        "outlet": false.copy(),
        "lateral_up": false.copy(),
        "lateral_down": false.copy(),
        "wall": false.copy(),
    }
    regions["lateral_down"][:, 0] = True
    regions["lateral_up"][:, -1] = True
    geometry = module.ArrowGeometry2D(
        nx=nx,
        ny=ny,
        dx=1.0,
        inflow_direction=1,
        fluid=fluid,
        boundary=false.copy(),
        regions=regions,
    )
    populations = np.zeros((nx, ny, 9), dtype=np.float64)
    populations[2, ny - 1, 2] = 0.75
    populations[3, 0, 4] = 0.5

    streamed = module.stream_with_bounce_back(populations, geometry)

    assert streamed[2, 0, 2] == 0.75
    assert streamed[3, ny - 1, 4] == 0.5
    assert streamed[2, ny - 1, 4] == 0.0
    assert streamed[3, 0, 2] == 0.0


def test_lbm_estimates_obstacle_length_from_wall_span(tmp_path) -> None:
    module = _load_lbm_module()
    path = tmp_path / "channel.arrow"
    _write_channel_arrow(path)
    geometry = module.load_arrow_geometry(path)

    assert module.estimate_obstacle_length(geometry) == 3.0


def test_lbm_render_frame_can_be_scaled(tmp_path) -> None:
    module = _load_lbm_module()
    path = tmp_path / "channel.arrow"
    _write_channel_arrow(path)
    geometry = module.load_arrow_geometry(path)
    velocity = np.zeros((geometry.nx, geometry.ny, 2), dtype=np.float64)

    frame = module.render_frame(geometry, velocity, scale=3)

    assert frame.shape == ((geometry.ny * 2 + 4) * 3, geometry.nx * 3, 3)


def test_lbm_vorticity_midpoint_is_not_white(tmp_path) -> None:
    module = _load_lbm_module()
    path = tmp_path / "channel.arrow"
    _write_channel_arrow(path)
    geometry = module.load_arrow_geometry(path)
    velocity = np.zeros((geometry.nx, geometry.ny, 2), dtype=np.float64)

    frame = module.render_frame(geometry, velocity, scale=1)

    middle_row = geometry.ny // 2
    middle_column = geometry.nx // 2
    assert frame[middle_row, middle_column].max() < 120
