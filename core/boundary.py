from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoundaryRegion:
    """A named subset of the final fluid boundary selected by boundary owner."""

    name: str
    object_id: int
    owner_object_id: int
    outside_direction: int | None = None

    def __post_init__(self) -> None:
        if self.outside_direction is not None and not 0 <= self.outside_direction < 6:
            raise ValueError("outside_direction must be in the range 0..5")

    @property
    def kind(self) -> str:
        return "boundary_region"

    @property
    def dimension(self) -> int:
        return 2
