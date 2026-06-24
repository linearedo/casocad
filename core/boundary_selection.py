from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.boundary import BoundaryRegion
from core.boundary_patches import (
    boundary_region_scope_mask,
    surface_selector_values,
)
from core.sdf.base import SDFNode
from core.sdf.placed_2d import PlacedSDF2D
from core.sdf.primitives_2d import (
    CircleProfile,
    EllipseProfile,
    RectangleProfile,
    SquareProfile,
)


def interval_parameter_mask(
    parameter: NDArray[np.float64],
    start: float,
    end: float,
) -> NDArray[np.bool_]:
    parameter = np.where(np.isclose(parameter, 1.0, atol=1.0e-12), 0.0, parameter)
    start = float(np.mod(start, 1.0))
    end = float(np.mod(end, 1.0))
    if start <= end:
        return np.asarray(
            (parameter >= start - 1.0e-9) & (parameter <= end + 1.0e-9),
            dtype=np.bool_,
        )
    return np.asarray(
        (parameter >= start - 1.0e-9) | (parameter <= end + 1.0e-9),
        dtype=np.bool_,
    )


def boundary_interval_mask(
    tag: BoundaryRegion,
    owner: PlacedSDF2D,
    positions: NDArray[np.float64],
) -> NDArray[np.bool_]:
    profile = owner.profile
    if isinstance(profile, SquareProfile):
        profile = profile._rectangle()
    assert tag.selector_start is not None and tag.selector_end is not None
    u, v, _plane = owner.project_numpy(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
    )
    if isinstance(profile, CircleProfile) and tag.patch_id == "curve":
        cu, cv = profile.center
        parameter = np.mod(np.arctan2(v - cv, u - cu), 2.0 * np.pi) / (2.0 * np.pi)
        return interval_parameter_mask(parameter, tag.selector_start, tag.selector_end)
    if isinstance(profile, EllipseProfile) and tag.patch_id == "curve":
        cu, cv = profile.center
        au, av = profile.semi_axes
        parameter = (
            np.mod(np.arctan2((v - cv) / av, (u - cu) / au), 2.0 * np.pi)
            / (2.0 * np.pi)
        )
        return interval_parameter_mask(parameter, tag.selector_start, tag.selector_end)
    if not isinstance(profile, RectangleProfile):
        return np.ones(positions.shape[0], dtype=np.bool_)
    if tag.patch_id not in {"-U", "+U", "-V", "+V"}:
        return np.ones(positions.shape[0], dtype=np.bool_)
    start, end = sorted((tag.selector_start, tag.selector_end))
    cu, cv = profile.center
    hu, hv = profile.half_size
    if tag.patch_id in {"-U", "+U"}:
        parameter = (v - (cv - hv)) / (2.0 * hv)
    else:
        parameter = (u - (cu - hu)) / (2.0 * hu)
    return np.asarray(
        (parameter >= start - 1.0e-9) & (parameter <= end + 1.0e-9),
        dtype=np.bool_,
    )


def _selector_object_from_id(
    selector_id: str,
    selector_by_id: dict[int, SDFNode],
) -> SDFNode | None:
    prefix = "selector:"
    if not selector_id.startswith(prefix):
        return None
    try:
        object_id = int(selector_id[len(prefix):])
    except ValueError:
        return None
    return selector_by_id.get(object_id)


def surface_split_selector_mask(
    selector_id: str,
    selector_by_id: dict[int, SDFNode],
    root: SDFNode,
    positions: NDArray[np.float64],
    *,
    region: BoundaryRegion | None = None,
    side: str = "inside",
    tolerance: float,
) -> NDArray[np.bool_]:
    selector = _selector_object_from_id(selector_id, selector_by_id)
    if selector is None:
        return np.zeros(positions.shape[0], dtype=np.bool_)
    values = surface_selector_values(
        root,
        selector,
        positions,
        scope_region=region,
    )
    inside = np.asarray(values <= tolerance, dtype=np.bool_)
    if region is not None:
        scope = boundary_region_scope_mask(
            root,
            region,
            positions,
            tolerance=tolerance,
        )
        inside &= scope
    else:
        scope = np.ones(positions.shape[0], dtype=np.bool_)
    if side == "outside":
        return np.asarray(scope & ~inside, dtype=np.bool_)
    return inside


__all__ = [
    "boundary_interval_mask",
    "interval_parameter_mask",
    "surface_split_selector_mask",
]
