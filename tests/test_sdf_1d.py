from __future__ import annotations

import numpy as np

from core.sdf import (
    BinaryProfile1D,
    IntervalProfile,
    PlacedSDF1D,
    SDFTree,
)


def test_interval_profile_has_signed_intrinsic_distance() -> None:
    interval = IntervalProfile(center=0.5, half_length=1.0)
    coordinates = np.asarray((-1.0, -0.5, 0.5, 1.5, 2.0))

    np.testing.assert_allclose(
        interval.to_numpy(coordinates),
        (0.5, 0.0, -1.0, 0.0, 0.5),
    )
    assert interval.bounds() == (-0.5, 1.5)


def test_placed_1d_projects_to_its_line() -> None:
    line = PlacedSDF1D(
        name="inlet",
        object_id=1,
        profile=IntervalProfile(half_length=1.0),
        origin=(1.0, 2.0, 3.0),
        axis_u=(0.0, 1.0, 0.0),
    )
    positions = np.asarray(
        (
            (1.0, 2.5, 3.0),
            (1.0, 3.5, 3.0),
            (1.1, 2.5, 3.0),
        ),
        dtype=np.float64,
    )
    coordinate, radial = line.project_numpy(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
    )

    np.testing.assert_allclose(coordinate, (0.5, 1.5, 0.5))
    np.testing.assert_allclose(radial, (0.0, 0.0, 0.1))
    assert line.contains_points(positions, tolerance=1e-8).tolist() == [
        True,
        False,
        False,
    ]


def test_1d_boolean_profile_preserves_interval_signs() -> None:
    profile = BinaryProfile1D(
        IntervalProfile(center=-0.5, half_length=0.75),
        IntervalProfile(center=0.5, half_length=0.75),
        "union",
    )
    coordinates = np.asarray((-1.5, -1.0, 0.0, 1.0, 1.5))

    assert (profile.to_numpy(coordinates) <= 0.0).tolist() == [
        False,
        True,
        True,
        True,
        False,
    ]


def test_placed_1d_generates_selectable_glsl() -> None:
    line = PlacedSDF1D(
        name="line",
        object_id=7,
        profile=IntervalProfile(),
    )
    tree = SDFTree(line, components=(line,))
    source = tree.to_glsl()

    assert "length(" in line.to_glsl()
    assert "return 1;" in source
    assert f"selected_object_id == {line.object_id}" in source
