from __future__ import annotations

"""Surface provenance + physics tags for Domain boundaries (spec v2 §4, §8).

The exact operators are ``min``/``max`` *selections*: at any boundary point
exactly one leaf is active, so every boundary surface already knows the leaf that
owns it. ``core.boundary_patches`` implements that walk -- it carries the
``cut_surface`` flag and ``owner`` down through ``Difference`` / ``Union`` /
``Intersection`` / transforms, so a surface cut by a ``Subtract`` is attributed to
the *obstacle* that cut it.

This module makes that provenance **first-class and Domain-/physics-aware**: it
maps each :class:`~core.boundary_patches.BoundaryPatch` of a Domain region to a
:class:`SurfaceProvenance` carrying its owner, whether it is a cut surface, and a
:class:`PatchTag` (physics role). The spec rule (§4) is that a cut surface
produced by subtracting an obstacle defaults to ``WALL``; inlet/outlet are
explicit user overrides. The *mechanism* (provenance riding the operators) is
settled here; the full tag *taxonomy* + UI remains open (§12).

Reference: ``docs/exact_signed_distance_field_cfd_migration_v2.md`` (§4, §8).
"""

from dataclasses import dataclass
from enum import Enum

from core.boundary_patches import boundary_patches
from core.sdf.roles import Domain


class PatchTag(Enum):
    """Physics tag a boundary surface carries downstream. Minimal default set;
    the full taxonomy is open (spec §12). ``WALL`` is the default for every solid
    surface (including obstacle cut surfaces); ``INLET``/``OUTLET`` are explicit
    overrides."""

    WALL = "wall"
    INLET = "inlet"
    OUTLET = "outlet"


@dataclass(frozen=True)
class SurfaceProvenance:
    """One boundary surface of a Domain, with its provenance and physics tag.

    * ``owner_object_id`` / ``owner_kind`` -- the leaf that owns this surface.
    * ``patch_id`` -- the surface's stable sub-id within its owner.
    * ``is_cut_surface`` -- True if produced by subtracting an obstacle (the
      surface inherits the obstacle's identity, spec §4).
    * ``tag`` -- the physics role (default ``WALL``).
    """

    owner_object_id: int
    owner_kind: str
    patch_id: str
    is_cut_surface: bool
    tag: PatchTag


def domain_surface_provenance(
    domain: Domain,
    *,
    tag_overrides: dict[int, PatchTag] | None = None,
) -> tuple[SurfaceProvenance, ...]:
    """Return the provenance + physics tag of every boundary surface of a Domain.

    Each surface defaults to :attr:`PatchTag.WALL`. ``tag_overrides`` maps an
    owner ``object_id`` to an explicit tag (e.g. mark the inlet/outlet face's
    owner as ``INLET``/``OUTLET``); the override applies to every surface owned
    by that object.
    """

    overrides = tag_overrides or {}
    provenance: list[SurfaceProvenance] = []
    for patch in boundary_patches(domain.region):
        provenance.append(
            SurfaceProvenance(
                owner_object_id=patch.owner_object_id,
                owner_kind=patch.owner.kind,
                patch_id=patch.patch_id,
                is_cut_surface=patch.patch_type == "cut_surface",
                tag=overrides.get(patch.owner_object_id, PatchTag.WALL),
            )
        )
    return tuple(provenance)


__all__ = [
    "PatchTag",
    "SurfaceProvenance",
    "domain_surface_provenance",
]
