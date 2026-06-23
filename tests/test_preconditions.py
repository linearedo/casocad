from __future__ import annotations

import pytest

from core.model import Model, ModelCompileError, compile_model
from core.preconditions import (
    erosion_violations,
    precondition_violations,
    revolve_violations,
)
from core.sdf import CircleProfile, DistanceOffsetProfile, PlacedSDF2D, Revolve
from core.sdf.roles import Domain, DomainKind


def _section(profile) -> PlacedSDF2D:
    return PlacedSDF2D(
        name="sec",
        object_id=1,
        profile=profile,
        origin=(0.0, 0.0, 0.0),
        axis_u=(1.0, 0.0, 0.0),
        axis_v=(0.0, 1.0, 0.0),
    )


def _revolve(profile, *, object_id: int = 2) -> Revolve:
    # axis="v" -> revolve about +Y; radial axis is +X (the profile's u coord).
    return Revolve(name="rev", object_id=object_id, section=_section(profile), axis="v")


# --- Revolve axis-crossing (§5) ---------------------------------------------


def test_revolve_off_axis_profile_is_exact() -> None:
    # Circle entirely at u in [0.85, 1.55] -> stays on one side of the axis.
    rev = _revolve(CircleProfile(center=(1.2, 0.0), radius=0.35))
    assert revolve_violations(rev) == []


def test_revolve_profile_crossing_axis_is_flagged() -> None:
    # Circle at u in [-0.5, 0.5] straddles the axis (u=0).
    rev = _revolve(CircleProfile(center=(0.0, 0.0), radius=0.5))
    issues = revolve_violations(rev)
    assert issues
    assert "crosses the revolution axis" in issues[0]


# --- Erosion reach (§6) -----------------------------------------------------


def test_dilation_is_unconditional() -> None:
    prof = DistanceOffsetProfile(
        child=CircleProfile(center=(0.0, 0.0), radius=0.5), offset=0.3
    )
    assert erosion_violations(prof) == []


def test_small_erosion_within_reach_is_ok() -> None:
    # Depth of a radius-0.5 disk is 0.5; eroding 0.2 stays within reach.
    prof = DistanceOffsetProfile(
        child=CircleProfile(center=(0.0, 0.0), radius=0.5), offset=-0.2
    )
    assert erosion_violations(prof) == []


def test_erosion_beyond_reach_vanishes() -> None:
    # Eroding 0.6 > the 0.5 inscribed depth -> the shape vanishes.
    prof = DistanceOffsetProfile(
        child=CircleProfile(center=(0.0, 0.0), radius=0.5), offset=-0.6
    )
    issues = erosion_violations(prof)
    assert issues
    assert "vanishes" in issues[0]


# --- aggregator + compile_model wiring --------------------------------------


def test_precondition_violations_walks_region() -> None:
    bad = _revolve(CircleProfile(center=(0.0, 0.0), radius=0.5))
    assert precondition_violations(bad)
    good = _revolve(CircleProfile(center=(1.2, 0.0), radius=0.35))
    assert precondition_violations(good) == []


def test_compile_model_rejects_precondition_violation() -> None:
    bad = _revolve(CircleProfile(center=(0.0, 0.0), radius=0.5))
    domain = Domain(name="part", kind=DomainKind.SOLID, region=bad)
    with pytest.raises(ModelCompileError):
        compile_model(Model(domains=(domain,)))
