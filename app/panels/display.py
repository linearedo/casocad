from __future__ import annotations

import re

from core.boundary import BoundaryRegion
from core.sdf import PlacedPolyline2D, PlacedSDF1D, PlacedSDF2D


def _camel_to_snake(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()


def _profile_display_kind(profile: object) -> str:
    raw = str(getattr(profile, "kind", type(profile).__name__.lower()))
    class_name = type(profile).__name__
    if raw == class_name.lower() and class_name.endswith("Profile"):
        return _camel_to_snake(class_name[: -len("Profile")])
    return raw


def display_kind(node: object) -> str:
    """Return the user-facing shape kind for panels.

    Placed 1D/2D nodes are implementation containers: users care about the
    profile they drew, not the placement wrapper.
    """
    if isinstance(node, (PlacedSDF1D, PlacedSDF2D, PlacedPolyline2D)):
        profile = getattr(node, "profile", None)
        if profile is not None:
            return _profile_display_kind(profile)
    if isinstance(node, BoundaryRegion):
        return node.kind
    return str(getattr(node, "kind", type(node).__name__.lower()))


__all__ = ["display_kind"]
