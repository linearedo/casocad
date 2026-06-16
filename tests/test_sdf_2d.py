from __future__ import annotations

import numpy as np

from core.sdf import (
    CircleProfile,
    Extrude,
    LoftImplicit,
    PlacedSDF2D,
    RectangleProfile,
    Sweep,
)


def _point(value: tuple[float, float, float]) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray([coordinate], dtype=np.float64) for coordinate in value)


def test_placed_profile_projects_to_its_plane_and_filled_area() -> None:
    placed = PlacedSDF2D(
        name="inlet",
        object_id=1,
        profile=CircleProfile(radius=1.0),
    )
    u, v, plane = placed.project_numpy(*_point((0.0, 0.0, 0.0)))
    assert plane[0] == 0.0
    assert placed.profile is not None
    assert placed.profile.to_numpy(u, v)[0] < 0.0
    _u, _v, plane = placed.project_numpy(*_point((0.0, 0.0, 0.1)))
    assert plane[0] != 0.0
    u, v, _plane = placed.project_numpy(*_point((2.0, 0.0, 0.0)))
    assert placed.profile.to_numpy(u, v)[0] > 0.0


def test_extrude_is_closed_and_signed() -> None:
    section = PlacedSDF2D(
        name="section", profile=RectangleProfile(half_size=(1.0, 1.0))
    )
    solid = Extrude(name="solid", section=section, height=2.0)
    assert solid.to_numpy(*_point((0.0, 0.0, 0.0)))[0] < 0.0
    assert solid.to_numpy(*_point((0.0, 0.0, 1.5)))[0] > 0.0
    assert solid.to_numpy(*_point((2.0, 0.0, 0.0)))[0] > 0.0


def test_sweep_does_not_move_source_section() -> None:
    section = PlacedSDF2D(
        name="section",
        profile=CircleProfile(radius=0.5),
        origin=(0.0, 0.0, 0.0),
    )
    sweep = Sweep(name="sweep", section=section, end=(0.0, 0.0, 2.0))
    assert section.origin == (0.0, 0.0, 0.0)
    assert sweep.to_numpy(*_point((0.0, 0.0, 1.0)))[0] < 0.0


def test_loft_is_closed_between_ordered_sections() -> None:
    lower = PlacedSDF2D(
        name="lower",
        profile=CircleProfile(radius=0.5),
        origin=(0.0, 0.0, -1.0),
    )
    upper = PlacedSDF2D(
        name="upper",
        profile=CircleProfile(radius=1.0),
        origin=(0.0, 0.0, 1.0),
    )
    loft = LoftImplicit(name="loft", sections=(lower, upper))
    assert loft.to_numpy(*_point((0.0, 0.0, 0.0)))[0] < 0.0
    assert loft.to_numpy(*_point((0.0, 0.0, 2.0)))[0] > 0.0
