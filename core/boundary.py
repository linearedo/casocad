from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoundaryRegion:
    """A named subset of the final fluid boundary selected by boundary owner."""

    name: str
    object_id: int
    owner_object_id: int
    outside_direction: int | None = None
    patch_id: str | None = None
    patch_type: str | None = None
    selector_id: str | None = None
    selector_type: str | None = None
    selector_start: float | None = None
    selector_end: float | None = None

    def __post_init__(self) -> None:
        if self.outside_direction is not None and not 0 <= self.outside_direction < 6:
            raise ValueError("outside_direction must be in the range 0..5")
        if self.selector_start is not None and self.selector_end is None:
            raise ValueError("selector_end is required when selector_start is set")
        if self.selector_end is not None and self.selector_start is None:
            raise ValueError("selector_start is required when selector_end is set")

    @property
    def kind(self) -> str:
        return "boundary_region"

    @property
    def dimension(self) -> int:
        return 2
