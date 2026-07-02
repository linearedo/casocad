from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.sdf.base import SDFNode

CUT_SIDES = ("inside", "outside")


@dataclass(frozen=True)
class BoundaryCut:
    """One knife-half in a region's cut chain (boundary_region_v2 §2).

    ``ghost`` is detached selector geometry: never part of the scene graph,
    never rendered; it is embedded in the region's serialized record. A 3D
    ghost classifies by its own sign; a lower-dimensional ghost is extruded
    through the scene at classification time."""

    side: str
    ghost: "SDFNode"

    def __post_init__(self) -> None:
        if self.side not in CUT_SIDES:
            raise ValueError("cut side must be 'inside' or 'outside'")


@dataclass
class BoundaryRegion:
    """A named subset of a Domain boundary (boundary_region_v2 §2).

    Identity = owner surface (provenance) + optional analytic patch scope +
    the ordered chain of cuts that carved it. ``tag`` is an opaque physics
    label the kernel never interprets. The ``selector_*``/``outside_direction``
    fields are the legacy single-selector schema, kept readable until every
    creation/load path emits cut chains."""

    name: str
    object_id: int
    owner_object_id: int
    outside_direction: int | None = None
    patch_id: str | None = None
    patch_type: str | None = None
    selector_id: str | None = None
    selector_type: str | None = None
    selector_side: str = "inside"
    selector_start: float | None = None
    selector_end: float | None = None
    cuts: tuple[BoundaryCut, ...] = field(default=())
    tag: str | None = None

    def __post_init__(self) -> None:
        if self.outside_direction is not None and not 0 <= self.outside_direction < 6:
            raise ValueError("outside_direction must be in the range 0..5")
        if self.selector_side not in {"inside", "outside"}:
            raise ValueError("selector_side must be 'inside' or 'outside'")
        if self.selector_start is not None and self.selector_end is None:
            raise ValueError("selector_end is required when selector_start is set")
        if self.selector_end is not None and self.selector_start is None:
            raise ValueError("selector_start is required when selector_end is set")
        self.cuts = tuple(self.cuts)
        if not all(isinstance(cut, BoundaryCut) for cut in self.cuts):
            raise ValueError("cuts must be BoundaryCut instances")

    @property
    def kind(self) -> str:
        return "boundary_region"

    @property
    def dimension(self) -> int:
        return 2
