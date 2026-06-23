from __future__ import annotations

"""Exactness preconditions for generators and offsets (spec §5, §6).

Most exact operations are unconditional, but a few are exact only under a
geometric precondition. This module provides sampled validators for the two
well-defined ones, plus an aggregator wired into ``compile_model``:

* **Revolve** is exact only when the section profile stays on **one side of the
  revolution axis** (§5). The revolve folds every point to ``radial >= 0``, so a
  profile that crosses the axis is double-covered and non-exact.
* **Erosion** (a negative ``DistanceOffsetProfile``) is exact only while the
  radius stays below the shape's reach (§6). A *necessary* check is enforced
  here: the erosion must not reach the shape's maximum inscribed depth, else the
  shape vanishes / the field stops being a distance. (True medial-axis reach can
  be stricter at concave features -- documented, not fully computed.)

Not covered (deferred, documented): **sweep / tube self-overlap** (radius vs.
path curvature). The tube primitives need a curvature analysis that is left for a
later pass; flagged here so it is not silently assumed exact.

These checks **sample** the field (like the §7 disjointness probe), so they are
part of the expensive ``compile_model`` gate, not the live grammar diagnostics.

Reference: ``docs/exact_signed_distance_field_cfd_migration_v2.md`` (§5, §6).
"""

import numpy as np

from core.sdf.base import SDFNode
from core.sdf.primitives_2d import DistanceOffsetProfile
from core.sdf.solid_from_2d import Revolve

_REVOLVE_RESOLUTION = 48
_EROSION_RESOLUTION = 64
_TOL = 1.0e-9


def revolve_violations(
    revolve: Revolve, *, resolution: int = _REVOLVE_RESOLUTION
) -> list[str]:
    """Return a violation if the revolve's profile crosses its axis (§5).

    The radial coordinate of each section point (its signed distance along the
    revolve's radial axis from the axis origin) is sampled over the profile; if
    the profile *interior* straddles ``radial = 0`` the revolve is non-exact.
    A profile that merely touches the axis (radial >= 0) is fine.
    """

    section = revolve.section
    assert section is not None and section.profile is not None
    origin, _axis, radial, _tangent = revolve._axis_frame()
    section_origin = np.asarray(section.origin, dtype=np.float64)
    axis_u = np.asarray(section.axis_u, dtype=np.float64)
    axis_v = np.asarray(section.axis_v, dtype=np.float64)

    u_min, u_max, v_min, v_max = section.profile.bounds()
    grid_u, grid_v = np.meshgrid(
        np.linspace(u_min, u_max, resolution),
        np.linspace(v_min, v_max, resolution),
        indexing="ij",
    )
    # radial coord(u,v) = ((origin_s - axis_origin) + u*axis_u + v*axis_v) . radial
    base = float(np.dot(section_origin - origin, radial))
    radial_coord = (
        base
        + grid_u * float(np.dot(axis_u, radial))
        + grid_v * float(np.dot(axis_v, radial))
    )
    interior = section.profile.to_numpy(grid_u, grid_v) < 0.0
    if not bool(interior.any()):
        return []
    r_min = float(radial_coord[interior].min())
    r_max = float(radial_coord[interior].max())
    if r_min < -_TOL and r_max > _TOL:
        return [
            f"Revolve {revolve.name!r}: profile crosses the revolution axis "
            f"(radial coord spans {r_min:.3g}..{r_max:.3g}); a revolve is exact "
            f"only when the profile stays on one side of the axis (§5)"
        ]
    return []


def erosion_violations(
    profile: DistanceOffsetProfile, *, resolution: int = _EROSION_RESOLUTION
) -> list[str]:
    """Return a violation if a negative offset erodes past the shape's reach (§6).

    Dilation (``offset >= 0``) is unconditionally exact. For erosion the radius
    must stay below the shape's maximum inscribed depth; otherwise the eroded set
    vanishes and ``child - offset`` is no longer a true distance. This is a
    *necessary* condition -- the true medial-axis reach can be stricter at
    concave features.
    """

    if profile.offset >= 0.0:
        return []  # dilation: unconditional (§6)
    child = profile.child
    u_min, u_max, v_min, v_max = child.bounds()
    grid_u, grid_v = np.meshgrid(
        np.linspace(u_min, u_max, resolution),
        np.linspace(v_min, v_max, resolution),
        indexing="ij",
    )
    min_child = float(child.to_numpy(grid_u, grid_v).min())  # deepest interior
    # Eroded set = {child < offset}; empty once offset <= min_child.
    if profile.offset <= min_child + _TOL:
        return [
            f"DistanceOffsetProfile: erosion {profile.offset:.3g} reaches/exceeds "
            f"the shape's max inscribed depth ({-min_child:.3g}); the eroded shape "
            f"vanishes or its field is no longer exact (§6, r < reach). Necessary "
            f"check only -- true reach can be stricter at concave features."
        ]
    return []


def _walk_nodes(node: SDFNode):
    yield node
    for child in node.children():
        yield from _walk_nodes(child)


def _walk_profiles(profile):
    yield profile
    for attr in ("child", "left", "right"):
        sub = getattr(profile, attr, None)
        if sub is not None and hasattr(sub, "to_numpy"):
            yield from _walk_profiles(sub)


def precondition_violations(region: SDFNode) -> list[str]:
    """Aggregate all generator/offset precondition violations in a region tree."""

    violations: list[str] = []
    for node in _walk_nodes(region):
        if isinstance(node, Revolve):
            violations.extend(revolve_violations(node))
        profile = getattr(node, "profile", None)
        if profile is not None:
            for sub in _walk_profiles(profile):
                if isinstance(sub, DistanceOffsetProfile):
                    violations.extend(erosion_violations(sub))
    return violations


__all__ = [
    "revolve_violations",
    "erosion_violations",
    "precondition_violations",
]
