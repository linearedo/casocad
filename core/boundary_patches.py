from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from math import pi
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from core.boundary import BoundaryRegion
from core.sdf import (
    BinaryProfile,
    Box,
    CappedCone,
    CircleProfile,
    Cone,
    Cylinder,
    DistanceOffsetProfile,
    Difference,
    EllipseProfile,
    Extrude,
    Intersection,
    PlacedPolyline2D,
    PlacedSDF1D,
    PlacedSDF2D,
    PolylineProfile,
    RectangleProfile,
    Rotate,
    Scale,
    Sphere,
    SquareProfile,
    SegmentProfile,
    Torus,
    Translate,
    Union,
)
from core.sdf.base import SDFNode
from core.sdf.operators import BinarySDFOperator

PATCH_TOLERANCE = 1.5e-3
CURVE_PATCH_PICK_TOLERANCE = 0.05
SURFACE_SELECTOR_TYPES = {
    "surface_sdf_subregion",
    "surface_split_curve",
    "surface_split_profile",
}


@dataclass(frozen=True)
class BoundarySelector:
    """Selector geometry that marks a subregion of a boundary patch."""

    selector_id: str
    selector_type: str
    object_id: int | None = None
    name: str | None = None
    side: str = "inside"


@dataclass(frozen=True)
class BoundaryIntervalSelector(BoundarySelector):
    start: float = 0.0
    end: float = 1.0


@dataclass(frozen=True)
class BoundaryObjectSelector(BoundarySelector):
    pass


@dataclass(frozen=True)
class BoundarySurfacePatch:
    owner_object_id: int
    patch_id: str
    patch_type: str
    owner: SDFNode
    normal: tuple[float, float, float] | None = None
    outside_direction: int | None = None
    normal_sign: float = 1.0
    selector: BoundarySelector | None = None


@dataclass(frozen=True)
class BoundaryCurvePatch:
    owner_object_id: int
    patch_id: str
    patch_type: str
    owner: PlacedSDF2D
    normal: tuple[float, float, float] | None = None
    outside_direction: int | None = None
    selector: BoundarySelector | None = None


@dataclass(frozen=True)
class BoundaryPatchHit:
    point: tuple[float, float, float]
    owner_object_id: int
    patch_id: str
    patch_type: str
    normal: tuple[float, float, float]
    outside_direction: int | None = None
    selector: BoundarySelector | None = None


BoundaryPatch = BoundarySurfacePatch | BoundaryCurvePatch


def boundary_owner_ids(root: SDFNode) -> set[int]:
    """Return object IDs that can own explicit CAD boundary patches."""
    return {patch.owner_object_id for patch in boundary_patches(root)}


def boundary_patches(root: SDFNode) -> tuple[BoundaryPatch, ...]:
    if root.dimension == 2:
        if isinstance(root, PlacedSDF2D):
            return curve_patches_for_owner(root)
        return tuple()
    return surface_patches_for_root(root)


def surface_patches_for_root(root: SDFNode) -> tuple[BoundarySurfacePatch, ...]:
    return tuple(_surface_patches_for_node(root, cut_surface=False, normal_sign=1.0))


def curve_patches_for_owner(owner: PlacedSDF2D) -> tuple[BoundaryCurvePatch, ...]:
    profile = owner.profile
    if profile is None:
        return tuple()
    axis_u = np.asarray(owner.axis_u, dtype=np.float64)
    axis_v = np.asarray(owner.axis_v, dtype=np.float64)
    if isinstance(profile, SquareProfile):
        rectangle = profile._rectangle()
        return _rectangle_curve_patches(owner, rectangle, axis_u, axis_v)
    if isinstance(profile, RectangleProfile):
        return _rectangle_curve_patches(owner, profile, axis_u, axis_v)
    if isinstance(profile, (CircleProfile, EllipseProfile)):
        return (
            BoundaryCurvePatch(
                owner_object_id=owner.object_id,
                patch_id="curve",
                patch_type="curve",
                owner=owner,
                normal=None,
            ),
        )
    return (
        BoundaryCurvePatch(
            owner_object_id=owner.object_id,
            patch_id=f"{profile.kind}_boundary",
            patch_type="curve",
            owner=owner,
            normal=None,
        ),
    )


def boundary_selector_from_node(
    node: SDFNode,
    *,
    domain_dimension: int,
) -> BoundarySelector | None:
    if domain_dimension == 3 and isinstance(node, (PlacedSDF1D, PlacedPolyline2D)):
        return BoundaryObjectSelector(
            selector_id=f"selector:{node.object_id}",
            selector_type="surface_split_curve",
            object_id=node.object_id,
            name=node.name,
        )
    if domain_dimension == 3 and isinstance(node, PlacedSDF2D):
        return BoundaryObjectSelector(
            selector_id=f"selector:{node.object_id}",
            selector_type="surface_split_profile",
            object_id=node.object_id,
            name=node.name,
        )
    if domain_dimension == 3 and node.dimension == 3:
        return BoundaryObjectSelector(
            selector_id=f"selector:{node.object_id}",
            selector_type="surface_sdf_subregion",
            object_id=node.object_id,
            name=node.name,
        )
    if domain_dimension == 2 and isinstance(node, (PlacedSDF1D, PlacedPolyline2D)):
        return BoundaryObjectSelector(
            selector_id=f"selector:{node.object_id}",
            selector_type="boundary_curve_selector",
            object_id=node.object_id,
            name=node.name,
        )
    return None


def boundary_interval_selector_from_node(
    patch: BoundaryCurvePatch,
    selector: SDFNode,
    *,
    tolerance: float = 1.0e-6,
) -> BoundaryIntervalSelector | None:
    endpoints = _selector_patch_parameters(patch, selector, tolerance)
    if endpoints is None:
        return None
    start, end = sorted(endpoints)
    return BoundaryIntervalSelector(
        selector_id=f"selector:{selector.object_id}",
        selector_type="boundary_curve_interval",
        object_id=selector.object_id,
        name=selector.name,
        start=max(0.0, min(1.0, start)),
        end=max(0.0, min(1.0, end)),
    )


def pick_boundary_patch(
    root: SDFNode,
    ray_origin: NDArray[np.float64],
    ray_direction: NDArray[np.float64],
    *,
    selector_objects: Sequence[SDFNode] = (),
    hit_tolerance: float = 0.0008,
    maximum_travel: float = 100.0,
) -> BoundaryPatchHit | None:
    if root.dimension == 2:
        return _pick_2d_boundary_patch(root, ray_origin, ray_direction)

    patch_hits = _pick_surface_patch_candidates(
        root,
        ray_origin,
        ray_direction,
        hit_tolerance,
    )
    if patch_hits:
        hit = _select_surface_patch_hit(root, patch_hits)
        if not selector_objects:
            return hit
        return _surface_hit_with_selector(root, hit, selector_objects, hit_tolerance)

    point = _pick_sdf_surface(
        root,
        ray_origin,
        ray_direction,
        hit_tolerance=hit_tolerance,
        maximum_travel=maximum_travel,
    )
    if point is None:
        return None
    normal = _sdf_normal(root, point, hit_tolerance)
    candidates = [
        _surface_patch_hit(root, patch, point, normal, hit_tolerance)
        for patch in surface_patches_for_root(root)
    ]
    hits = [hit for hit in candidates if hit is not None]
    if not hits:
        return None
    hit = max(hits, key=lambda item: _normal_alignment(item.normal, normal))
    if not selector_objects:
        return hit
    return _surface_hit_with_selector(root, hit, selector_objects, hit_tolerance)


def boundary_patch_preview_node(
    root: SDFNode,
    hit: BoundaryPatchHit,
    *,
    selector_objects: Sequence[SDFNode] = (),
    thickness: float = 0.006,
) -> SDFNode | None:
    """Build transient RenderIR-compatible geometry for patch highlighting."""
    owner = _find_node_by_object_id(root, hit.owner_object_id)
    if owner is None:
        return None
    preview: SDFNode | None
    if root.dimension == 2 and isinstance(owner, PlacedSDF2D):
        preview = _curve_patch_preview_node(owner, hit, thickness)
    elif isinstance(owner, Box):
        preview = _box_patch_preview_node(owner, hit, thickness)
    elif isinstance(owner, (Cylinder, Cone, CappedCone)):
        preview = _cylinder_patch_preview_node(owner, hit, thickness)
    elif isinstance(owner, Sphere):
        preview = _sphere_patch_preview_node(owner, hit, thickness)
    elif isinstance(owner, Torus):
        preview = _torus_patch_preview_node(owner, hit, thickness)
    else:
        preview = None
    if preview is None:
        return None
    if root.dimension == 3 and hit.selector is not None:
        return _clip_surface_selector_preview(
            root,
            preview,
            hit.selector.selector_id,
            hit.selector.selector_type,
            hit.selector.side,
            selector_objects,
            name=f"{owner.name}_{hit.patch_id}_selector_highlight",
        )
    return preview


def boundary_region_preview_node(
    root: SDFNode,
    region: BoundaryRegion,
    *,
    selector_objects: Sequence[SDFNode] = (),
    thickness: float = 0.006,
) -> SDFNode | None:
    patch = _region_boundary_patch(root, region)
    if patch is None:
        return None
    if (
        root.dimension == 2
        and isinstance(patch, BoundaryCurvePatch)
        and region.selector_start is not None
        and region.selector_end is not None
    ):
        return _curve_interval_preview_node(patch, region, thickness)
    preview = _region_patch_preview_node(root, region, thickness)
    if preview is None:
        return None
    if root.dimension == 3 and region.selector_id is not None:
        return _clip_surface_selector_preview(
            root,
            preview,
            region.selector_id,
            region.selector_type,
            region.selector_side,
            selector_objects,
            name=f"{region.name}_selector_highlight",
        )
    return preview


def _region_boundary_patch(
    root: SDFNode,
    region: BoundaryRegion,
) -> BoundaryPatch | None:
    return next(
        (
            patch
            for patch in boundary_patches(root)
            if patch.owner_object_id == region.owner_object_id
            and patch.patch_id == region.patch_id
        ),
        None,
    )


def _region_patch_preview_node(
    root: SDFNode,
    region: BoundaryRegion,
    thickness: float,
) -> SDFNode | None:
    patch = _region_boundary_patch(root, region)
    if patch is None:
        return None
    normal = _region_patch_normal(patch)
    hit = BoundaryPatchHit(
        point=(0.0, 0.0, 0.0),
        owner_object_id=region.owner_object_id,
        patch_id=patch.patch_id,
        patch_type=patch.patch_type,
        normal=normal,
        outside_direction=patch.outside_direction,
        selector=patch.selector,
    )
    return boundary_patch_preview_node(root, hit, thickness=thickness)


def _region_patch_scope_volume(
    root: SDFNode,
    region: BoundaryRegion,
    thickness: float,
) -> SDFNode | None:
    patch = _region_boundary_patch(root, region)
    if patch is None:
        return None
    if isinstance(patch, BoundarySurfacePatch) and isinstance(patch.owner, Box):
        face = patch.patch_id.split(".")[-1]
        axis_index = {
            "-X": 0,
            "+X": 0,
            "-Y": 1,
            "+Y": 1,
            "-Z": 2,
            "+Z": 2,
        }.get(face)
        if axis_index is not None:
            sign = -1.0 if face.startswith("-") else 1.0
            axes = (
                np.asarray(patch.owner.axis_u, dtype=np.float64),
                np.asarray(patch.owner.axis_v, dtype=np.float64),
                np.asarray(patch.owner.axis_w, dtype=np.float64),
            )
            half_size = list(patch.owner.half_size)
            half_size[axis_index] = thickness
            center = (
                np.asarray(patch.owner.center, dtype=np.float64)
                + sign * patch.owner.half_size[axis_index] * axes[axis_index]
            )
            return Box(
                name=f"{patch.owner.name}_{patch.patch_id}_scope",
                object_id=0,
                center=_tuple(center),
                half_size=tuple(half_size),
                axis_u=patch.owner.axis_u,
                axis_v=patch.owner.axis_v,
                axis_w=patch.owner.axis_w,
            )
    return _region_patch_preview_node(root, region, thickness)


def _surface_patches_for_node(
    node: SDFNode,
    *,
    cut_surface: bool,
    normal_sign: float,
) -> Sequence[BoundarySurfacePatch]:
    if isinstance(node, (Translate, Scale, Rotate)):
        assert node.child is not None
        return _surface_patches_for_node(
            node.child,
            cut_surface=cut_surface,
            normal_sign=normal_sign,
        )
    if isinstance(node, Difference):
        assert node.left is not None and node.right is not None
        return (
            *_surface_patches_for_node(
                node.left,
                cut_surface=cut_surface,
                normal_sign=normal_sign,
            ),
            *_surface_patches_for_node(
                node.right,
                cut_surface=True,
                normal_sign=-normal_sign,
            ),
        )
    if isinstance(node, (Union, Intersection)):
        assert node.left is not None and node.right is not None
        return (
            *_surface_patches_for_node(
                node.left,
                cut_surface=cut_surface,
                normal_sign=normal_sign,
            ),
            *_surface_patches_for_node(
                node.right,
                cut_surface=cut_surface,
                normal_sign=normal_sign,
            ),
        )
    if isinstance(node, Box):
        return _box_surface_patches(node, cut_surface, normal_sign)
    if isinstance(node, Cylinder):
        return _cylinder_surface_patches(node, cut_surface, normal_sign)
    if isinstance(node, (Cone, CappedCone)):
        return _cylinder_surface_patches(node, cut_surface, normal_sign)
    if isinstance(node, (Sphere, Torus)):
        patch_id = _patch_id("surface", cut_surface)
        return (
            BoundarySurfacePatch(
                owner_object_id=node.object_id,
                patch_id=patch_id,
                patch_type="cut_surface" if cut_surface else "surface",
                owner=node,
                normal=None,
                normal_sign=normal_sign,
            ),
        )
    if isinstance(node, BinarySDFOperator):
        return tuple()
    return tuple()


def _box_patch_preview_node(
    owner: Box,
    hit: BoundaryPatchHit,
    thickness: float,
) -> SDFNode | None:
    face = hit.patch_id.split(".")[-1]
    axis_index = {"-X": 0, "+X": 0, "-Y": 1, "+Y": 1, "-Z": 2, "+Z": 2}.get(face)
    if axis_index is None:
        return None
    sign = -1.0 if face.startswith("-") else 1.0
    axes = (
        np.asarray(owner.axis_u, dtype=np.float64),
        np.asarray(owner.axis_v, dtype=np.float64),
        np.asarray(owner.axis_w, dtype=np.float64),
    )
    half_size = list(owner.half_size)
    half_size[axis_index] = thickness
    center = (
        np.asarray(owner.center, dtype=np.float64)
        + sign * owner.half_size[axis_index] * axes[axis_index]
        + np.asarray(hit.normal, dtype=np.float64) * thickness * 2.0
    )
    return Box(
        name=f"{owner.name}_{hit.patch_id}_highlight",
        object_id=0,
        center=_tuple(center),
        half_size=tuple(float(value) for value in half_size),
        axis_u=owner.axis_u,
        axis_v=owner.axis_v,
        axis_w=owner.axis_w,
    )


def _cylinder_patch_preview_node(
    owner: Cylinder | Cone | CappedCone,
    hit: BoundaryPatchHit,
    thickness: float,
) -> SDFNode | None:
    patch_name = hit.patch_id.split(".")[-1]
    radius = _cylinder_like_radius(owner)
    if radius is None:
        return None
    axis_w = np.asarray(owner.axis_w, dtype=np.float64)
    if patch_name == "-Z_cap":
        center = (
            np.asarray(owner.center, dtype=np.float64)
            - owner.half_height * axis_w
            + np.asarray(hit.normal, dtype=np.float64) * thickness * 2.0
        )
        return Cylinder(
            name=f"{owner.name}_{hit.patch_id}_highlight",
            object_id=0,
            center=_tuple(center),
            radius=radius,
            half_height=thickness,
            axis_u=owner.axis_u,
            axis_v=owner.axis_v,
            axis_w=owner.axis_w,
        )
    if patch_name == "+Z_cap":
        center = (
            np.asarray(owner.center, dtype=np.float64)
            + owner.half_height * axis_w
            + np.asarray(hit.normal, dtype=np.float64) * thickness * 2.0
        )
        return Cylinder(
            name=f"{owner.name}_{hit.patch_id}_highlight",
            object_id=0,
            center=_tuple(center),
            radius=radius,
            half_height=thickness,
            axis_u=owner.axis_u,
            axis_v=owner.axis_v,
            axis_w=owner.axis_w,
        )
    if patch_name == "side_wall":
        outer = Cylinder(
            name=f"{owner.name}_{hit.patch_id}_highlight_outer",
            object_id=0,
            center=owner.center,
            radius=radius + thickness,
            half_height=owner.half_height,
            axis_u=owner.axis_u,
            axis_v=owner.axis_v,
            axis_w=owner.axis_w,
        )
        inner = Cylinder(
            name=f"{owner.name}_{hit.patch_id}_highlight_inner",
            object_id=0,
            center=owner.center,
            radius=max(radius - thickness, thickness),
            half_height=owner.half_height + thickness * 2.0,
            axis_u=owner.axis_u,
            axis_v=owner.axis_v,
            axis_w=owner.axis_w,
        )
        return Difference(
            name=f"{owner.name}_{hit.patch_id}_highlight",
            object_id=0,
            left=outer,
            right=inner,
        )
    return None


def _sphere_patch_preview_node(
    owner: Sphere,
    hit: BoundaryPatchHit,
    thickness: float,
) -> SDFNode:
    outer = Sphere(
        name=f"{owner.name}_{hit.patch_id}_highlight_outer",
        object_id=0,
        center=owner.center,
        radius=owner.radius + thickness,
    )
    inner = Sphere(
        name=f"{owner.name}_{hit.patch_id}_highlight_inner",
        object_id=0,
        center=owner.center,
        radius=max(owner.radius - thickness, thickness),
    )
    return Difference(
        name=f"{owner.name}_{hit.patch_id}_highlight",
        object_id=0,
        left=outer,
        right=inner,
    )


def _torus_patch_preview_node(
    owner: Torus,
    hit: BoundaryPatchHit,
    thickness: float,
) -> SDFNode:
    outer = Torus(
        name=f"{owner.name}_{hit.patch_id}_highlight_outer",
        object_id=0,
        center=owner.center,
        major_radius=owner.major_radius,
        minor_radius=owner.minor_radius + thickness,
        axis_u=owner.axis_u,
        axis_v=owner.axis_v,
        axis_w=owner.axis_w,
    )
    inner = Torus(
        name=f"{owner.name}_{hit.patch_id}_highlight_inner",
        object_id=0,
        center=owner.center,
        major_radius=owner.major_radius,
        minor_radius=max(owner.minor_radius - thickness, thickness),
        axis_u=owner.axis_u,
        axis_v=owner.axis_v,
        axis_w=owner.axis_w,
    )
    return Difference(
        name=f"{owner.name}_{hit.patch_id}_highlight",
        object_id=0,
        left=outer,
        right=inner,
    )


def _clip_surface_selector_preview(
    root: SDFNode,
    preview: SDFNode,
    selector_id: str | None,
    selector_type: str | None,
    selector_side: str,
    selector_objects: Sequence[SDFNode],
    *,
    name: str,
    scope_region: BoundaryRegion | None = None,
) -> SDFNode:
    if selector_type not in SURFACE_SELECTOR_TYPES:
        return preview
    selector = _find_selector_node(root, selector_id, selector_objects)
    if selector is None:
        return preview
    selector_volume = surface_selector_volume(
        root,
        selector,
        scope_region=scope_region,
    )
    if selector_volume is None:
        return preview
    if selector_side == "outside":
        return Difference(
            name=name,
            object_id=0,
            left=preview,
            right=selector_volume,
        )
    return Intersection(name=name, object_id=0, left=preview, right=selector_volume)


def surface_selector_volume(
    root: SDFNode,
    selector: SDFNode,
    *,
    scope_region: BoundaryRegion | None = None,
) -> SDFNode | None:
    """Return the 3D SDF field used to classify boundary subregions."""

    if selector.dimension == 3:
        volume: SDFNode | None = deepcopy(selector)
    elif isinstance(selector, PlacedSDF2D):
        volume = _placed_2d_selector_volume(root, selector)
    elif isinstance(selector, PlacedPolyline2D):
        volume = _polyline_selector_volume(root, selector)
    elif isinstance(selector, PlacedSDF1D) and isinstance(
        selector.profile,
        SegmentProfile,
    ):
        lower, upper = selector.profile.bounds()
        half_length = max(0.5 * (upper - lower), PATCH_TOLERANCE)
        center_offset = 0.5 * (upper + lower)
        axis_u = np.asarray(selector.axis_u, dtype=np.float64)
        axis_u /= max(np.linalg.norm(axis_u), 1.0e-12)
        axis_v, axis_w = _orthonormal_completion(axis_u)
        bounds = root.bounding_box()
        span = max(
            bounds.x_max - bounds.x_min,
            bounds.y_max - bounds.y_min,
            bounds.z_max - bounds.z_min,
            1.0,
        )
        center = np.asarray(selector.origin, dtype=np.float64) + center_offset * axis_u
        volume = Box(
            name=f"{selector.name}_extruded_selector",
            object_id=0,
            center=_tuple(center),
            half_size=(half_length, span * 2.0, span * 2.0),
            axis_u=_tuple(axis_u),
            axis_v=_tuple(axis_v),
            axis_w=_tuple(axis_w),
        )
    else:
        volume = None
    if volume is None or scope_region is None:
        return volume
    scope = _region_patch_scope_volume(
        root,
        scope_region,
        max(PATCH_TOLERANCE * 4.0, 0.006),
    )
    if scope is None:
        return volume
    return Intersection(
        name=f"{selector.name}_scoped_selector",
        object_id=0,
        left=volume,
        right=scope,
    )


def surface_selector_values(
    root: SDFNode,
    selector: SDFNode,
    positions: NDArray[np.float64],
    *,
    scope_region: BoundaryRegion | None = None,
) -> NDArray[np.float64]:
    selector_volume = surface_selector_volume(
        root,
        selector,
        scope_region=scope_region,
    )
    if selector_volume is None:
        return np.full(positions.shape[0], np.inf, dtype=np.float64)
    return selector_volume.to_numpy(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
    )


def boundary_region_scope_mask(
    root: SDFNode,
    region: BoundaryRegion,
    positions: NDArray[np.float64],
    *,
    tolerance: float = PATCH_TOLERANCE,
) -> NDArray[np.bool_]:
    scope = _region_patch_scope_volume(
        root,
        region,
        max(PATCH_TOLERANCE * 4.0, 0.006),
    )
    if scope is None:
        return np.ones(positions.shape[0], dtype=np.bool_)
    values = scope.to_numpy(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
    )
    return np.asarray(values <= max(tolerance, PATCH_TOLERANCE), dtype=np.bool_)


def _placed_2d_selector_volume(
    root: SDFNode,
    selector: PlacedSDF2D,
) -> SDFNode | None:
    if selector.profile is None:
        return None
    bounds = root.bounding_box()
    span = max(
        bounds.x_max - bounds.x_min,
        bounds.y_max - bounds.y_min,
        bounds.z_max - bounds.z_min,
        1.0,
    )
    section = deepcopy(selector)
    section.name = f"{selector.name}_selector_section"
    section.object_id = 0
    return Extrude(
        name=f"{selector.name}_extruded_selector",
        object_id=0,
        section=section,
        height=span * 4.0,
    )


def _polyline_selector_volume(
    root: SDFNode,
    selector: PlacedPolyline2D,
) -> SDFNode | None:
    if selector.profile is None:
        return None
    bounds = root.bounding_box()
    span = max(
        bounds.x_max - bounds.x_min,
        bounds.y_max - bounds.y_min,
        bounds.z_max - bounds.z_min,
        1.0,
    )
    band_profile = DistanceOffsetProfile(
        deepcopy(selector.profile),
        PATCH_TOLERANCE,
    )
    section = PlacedSDF2D(
        name=f"{selector.name}_selector_section",
        object_id=0,
        profile=band_profile,
        origin=selector.origin,
        axis_u=selector.axis_u,
        axis_v=selector.axis_v,
    )
    return Extrude(
        name=f"{selector.name}_extruded_selector",
        object_id=0,
        section=section,
        height=span * 4.0,
    )


def _selector_has_surface_preview_volume(selector: SDFNode) -> bool:
    return selector.dimension == 3 or isinstance(selector, PlacedSDF2D) or (
        isinstance(selector, PlacedSDF1D)
        and isinstance(selector.profile, SegmentProfile)
    ) or (
        isinstance(selector, PlacedPolyline2D)
        and selector.profile is not None
    )


def _curve_patch_preview_node(
    owner: PlacedSDF2D,
    hit: BoundaryPatchHit,
    thickness: float,
) -> SDFNode | None:
    profile = owner.profile
    if isinstance(profile, CircleProfile):
        return _circle_curve_preview_node(owner, profile, hit, thickness)
    if isinstance(profile, EllipseProfile):
        return _ellipse_curve_preview_node(owner, profile, hit, thickness)
    if isinstance(profile, SquareProfile):
        profile = profile._rectangle()
    if not isinstance(profile, RectangleProfile):
        return None
    cu, cv = profile.center
    hu, hv = profile.half_size
    axis_u = np.asarray(owner.axis_u, dtype=np.float64)
    axis_v = np.asarray(owner.axis_v, dtype=np.float64)
    origin = np.asarray(owner.origin, dtype=np.float64)
    if hit.patch_id == "-U":
        line_origin = origin + (cu - hu) * axis_u + cv * axis_v
        line_axis = owner.axis_v
        normal_axis = owner.axis_u
        half_length = hv
    elif hit.patch_id == "+U":
        line_origin = origin + (cu + hu) * axis_u + cv * axis_v
        line_axis = owner.axis_v
        normal_axis = owner.axis_u
        half_length = hv
    elif hit.patch_id == "-V":
        line_origin = origin + cu * axis_u + (cv - hv) * axis_v
        line_axis = owner.axis_u
        normal_axis = owner.axis_v
        half_length = hu
    elif hit.patch_id == "+V":
        line_origin = origin + cu * axis_u + (cv + hv) * axis_v
        line_axis = owner.axis_u
        normal_axis = owner.axis_v
        half_length = hu
    else:
        return None
    return PlacedSDF2D(
        name=f"{owner.name}_{hit.patch_id}_highlight",
        object_id=0,
        profile=RectangleProfile(half_size=(thickness, max(half_length, thickness))),
        origin=_tuple(line_origin),
        axis_u=normal_axis,
        axis_v=line_axis,
    )


def _circle_curve_preview_node(
    owner: PlacedSDF2D,
    profile: CircleProfile,
    hit: BoundaryPatchHit,
    thickness: float,
) -> SDFNode:
    outer = CircleProfile(
        center=profile.center,
        radius=profile.radius + thickness,
    )
    inner = CircleProfile(
        center=profile.center,
        radius=max(profile.radius - thickness, thickness),
    )
    return PlacedSDF2D(
        name=f"{owner.name}_{hit.patch_id}_highlight",
        object_id=0,
        profile=BinaryProfile(outer, inner, "difference"),
        origin=owner.origin,
        axis_u=owner.axis_u,
        axis_v=owner.axis_v,
    )


def _ellipse_curve_preview_node(
    owner: PlacedSDF2D,
    profile: EllipseProfile,
    hit: BoundaryPatchHit,
    thickness: float,
) -> SDFNode:
    outer = EllipseProfile(
        center=profile.center,
        semi_axes=(
            profile.semi_axes[0] + thickness,
            profile.semi_axes[1] + thickness,
        ),
    )
    inner = EllipseProfile(
        center=profile.center,
        semi_axes=(
            max(profile.semi_axes[0] - thickness, thickness),
            max(profile.semi_axes[1] - thickness, thickness),
        ),
    )
    return PlacedSDF2D(
        name=f"{owner.name}_{hit.patch_id}_highlight",
        object_id=0,
        profile=BinaryProfile(outer, inner, "difference"),
        origin=owner.origin,
        axis_u=owner.axis_u,
        axis_v=owner.axis_v,
    )


def _curve_interval_preview_node(
    patch: BoundaryCurvePatch,
    region: BoundaryRegion,
    thickness: float,
) -> SDFNode | None:
    assert region.selector_start is not None and region.selector_end is not None
    owner = patch.owner
    profile = owner.profile
    if isinstance(profile, SquareProfile):
        profile = profile._rectangle()
    if isinstance(profile, CircleProfile):
        return _closed_curve_interval_preview_node(
            owner,
            patch,
            region,
            profile.center,
            (profile.radius, profile.radius),
            thickness,
        )
    if isinstance(profile, EllipseProfile):
        return _closed_curve_interval_preview_node(
            owner,
            patch,
            region,
            profile.center,
            profile.semi_axes,
            thickness,
        )
    if not isinstance(profile, RectangleProfile):
        return None
    start, end = sorted((region.selector_start, region.selector_end))
    start = max(0.0, min(1.0, start))
    end = max(0.0, min(1.0, end))
    cu, cv = profile.center
    hu, hv = profile.half_size
    axis_u = np.asarray(owner.axis_u, dtype=np.float64)
    axis_v = np.asarray(owner.axis_v, dtype=np.float64)
    origin = np.asarray(owner.origin, dtype=np.float64)
    if patch.patch_id == "-U":
        local_start = np.asarray((cu - hu, cv - hv + 2.0 * hv * start))
        local_end = np.asarray((cu - hu, cv - hv + 2.0 * hv * end))
        line_axis = owner.axis_v
        normal_axis = owner.axis_u
    elif patch.patch_id == "+U":
        local_start = np.asarray((cu + hu, cv - hv + 2.0 * hv * start))
        local_end = np.asarray((cu + hu, cv - hv + 2.0 * hv * end))
        line_axis = owner.axis_v
        normal_axis = owner.axis_u
    elif patch.patch_id == "-V":
        local_start = np.asarray((cu - hu + 2.0 * hu * start, cv - hv))
        local_end = np.asarray((cu - hu + 2.0 * hu * end, cv - hv))
        line_axis = owner.axis_u
        normal_axis = owner.axis_v
    elif patch.patch_id == "+V":
        local_start = np.asarray((cu - hu + 2.0 * hu * start, cv + hv))
        local_end = np.asarray((cu - hu + 2.0 * hu * end, cv + hv))
        line_axis = owner.axis_u
        normal_axis = owner.axis_v
    else:
        return None
    world_start = origin + local_start[0] * axis_u + local_start[1] * axis_v
    world_end = origin + local_end[0] * axis_u + local_end[1] * axis_v
    center = 0.5 * (world_start + world_end)
    half_length = 0.5 * float(np.linalg.norm(world_end - world_start))
    return PlacedSDF2D(
        name=f"{owner.name}_{patch.patch_id}_interval_highlight",
        object_id=0,
        profile=RectangleProfile(half_size=(thickness, max(half_length, thickness))),
        origin=_tuple(center),
        axis_u=normal_axis,
        axis_v=line_axis,
    )


def _closed_curve_interval_preview_node(
    owner: PlacedSDF2D,
    patch: BoundaryCurvePatch,
    region: BoundaryRegion,
    center: tuple[float, float],
    semi_axes: tuple[float, float],
    thickness: float,
) -> SDFNode:
    assert region.selector_start is not None and region.selector_end is not None
    start = max(0.0, min(1.0, region.selector_start))
    end = max(0.0, min(1.0, region.selector_end))
    if end < start:
        end += 1.0
    span = max(end - start, 1.0e-6)
    sample_count = max(8, int(np.ceil(span * 64.0)) + 1)
    angles = np.linspace(start * 2.0 * pi, end * 2.0 * pi, sample_count)
    cu, cv = center
    au, av = semi_axes
    points = tuple(
        (
            float(cu + au * np.cos(angle)),
            float(cv + av * np.sin(angle)),
        )
        for angle in angles
    )
    return PlacedPolyline2D(
        name=f"{owner.name}_{patch.patch_id}_interval_highlight",
        object_id=0,
        profile=PolylineProfile(points=points),
        origin=owner.origin,
        axis_u=owner.axis_u,
        axis_v=owner.axis_v,
    )


def _selector_patch_parameters(
    patch: BoundaryCurvePatch,
    selector: SDFNode,
    tolerance: float,
) -> tuple[float, float] | None:
    world_endpoints = _selector_world_endpoints(selector)
    if world_endpoints is None:
        return None
    owner = patch.owner
    profile = owner.profile
    if isinstance(profile, SquareProfile):
        profile = profile._rectangle()
    u, v, plane = owner.project_numpy(
        np.asarray([world_endpoints[0][0], world_endpoints[1][0]], dtype=np.float64),
        np.asarray([world_endpoints[0][1], world_endpoints[1][1]], dtype=np.float64),
        np.asarray([world_endpoints[0][2], world_endpoints[1][2]], dtype=np.float64),
    )
    if np.max(np.abs(plane)) > tolerance:
        return None
    if isinstance(profile, CircleProfile) and patch.patch_id == "curve":
        return _closed_curve_patch_parameters(profile, u, v, tolerance)
    if isinstance(profile, EllipseProfile) and patch.patch_id == "curve":
        return _closed_curve_patch_parameters(profile, u, v, tolerance)
    if not isinstance(profile, RectangleProfile):
        return None
    if patch.patch_id not in {"-U", "+U", "-V", "+V"}:
        return None
    cu, cv = profile.center
    hu, hv = profile.half_size
    if patch.patch_id == "-U":
        if np.max(np.abs(u - (cu - hu))) > tolerance:
            return None
        return tuple(float((value - (cv - hv)) / (2.0 * hv)) for value in v)
    if patch.patch_id == "+U":
        if np.max(np.abs(u - (cu + hu))) > tolerance:
            return None
        return tuple(float((value - (cv - hv)) / (2.0 * hv)) for value in v)
    if patch.patch_id == "-V":
        if np.max(np.abs(v - (cv - hv))) > tolerance:
            return None
        return tuple(float((value - (cu - hu)) / (2.0 * hu)) for value in u)
    if np.max(np.abs(v - (cv + hv))) > tolerance:
        return None
    return tuple(float((value - (cu - hu)) / (2.0 * hu)) for value in u)


def _selector_world_endpoints(
    selector: SDFNode,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    if isinstance(selector, PlacedSDF1D):
        if not isinstance(selector.profile, SegmentProfile):
            return None
        t_min, t_max = selector.profile.bounds()
        origin = np.asarray(selector.origin, dtype=np.float64)
        axis = np.asarray(selector.axis_u, dtype=np.float64)
        return origin + t_min * axis, origin + t_max * axis
    if isinstance(selector, PlacedPolyline2D):
        profile = selector.profile
        if profile is None or not hasattr(profile, "points"):
            return None
        points = profile.points
        if len(points) < 2:
            return None
        origin = np.asarray(selector.origin, dtype=np.float64)
        axis_u = np.asarray(selector.axis_u, dtype=np.float64)
        axis_v = np.asarray(selector.axis_v, dtype=np.float64)
        first = np.asarray(points[0], dtype=np.float64)
        last = np.asarray(points[-1], dtype=np.float64)
        return (
            origin + first[0] * axis_u + first[1] * axis_v,
            origin + last[0] * axis_u + last[1] * axis_v,
        )
    return None


def _closed_curve_patch_parameters(
    profile: CircleProfile | EllipseProfile,
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    tolerance: float,
) -> tuple[float, float] | None:
    distances = profile.to_numpy(u, v)
    if np.max(np.abs(distances)) > max(tolerance, PATCH_TOLERANCE):
        return None
    cu, cv = profile.center
    if isinstance(profile, CircleProfile):
        du = u - cu
        dv = v - cv
    else:
        au, av = profile.semi_axes
        du = (u - cu) / au
        dv = (v - cv) / av
    angles = np.mod(np.arctan2(dv, du), 2.0 * pi)
    parameters = angles / (2.0 * pi)
    parameters = np.where(np.isclose(parameters, 1.0, atol=1.0e-12), 0.0, parameters)
    return tuple(float(value) for value in parameters)


def _box_surface_patches(
    owner: Box,
    cut_surface: bool,
    normal_sign: float,
) -> tuple[BoundarySurfacePatch, ...]:
    axes = (
        np.asarray(owner.axis_u, dtype=np.float64),
        np.asarray(owner.axis_v, dtype=np.float64),
        np.asarray(owner.axis_w, dtype=np.float64),
    )
    names = ("-X", "+X", "-Y", "+Y", "-Z", "+Z")
    normals = (-axes[0], axes[0], -axes[1], axes[1], -axes[2], axes[2])
    return tuple(
        BoundarySurfacePatch(
            owner_object_id=owner.object_id,
            patch_id=_patch_id(name, cut_surface),
            patch_type="cut_surface" if cut_surface else "face",
            owner=owner,
            normal=_tuple(normal),
            outside_direction=index,
            normal_sign=normal_sign,
        )
        for index, (name, normal) in enumerate(zip(names, normals, strict=True))
    )


def _cylinder_surface_patches(
    owner: Cylinder | Cone | CappedCone,
    cut_surface: bool,
    normal_sign: float,
) -> tuple[BoundarySurfacePatch, ...]:
    axis_w = np.asarray(owner.axis_w, dtype=np.float64)
    return (
        BoundarySurfacePatch(
            owner_object_id=owner.object_id,
            patch_id=_patch_id("side_wall", cut_surface),
            patch_type="cut_surface" if cut_surface else "side_wall",
            owner=owner,
            normal=None,
            normal_sign=normal_sign,
        ),
        BoundarySurfacePatch(
            owner_object_id=owner.object_id,
            patch_id=_patch_id("-Z_cap", cut_surface),
            patch_type="cut_surface" if cut_surface else "cap",
            owner=owner,
            normal=_tuple(-axis_w),
            outside_direction=4,
            normal_sign=normal_sign,
        ),
        BoundarySurfacePatch(
            owner_object_id=owner.object_id,
            patch_id=_patch_id("+Z_cap", cut_surface),
            patch_type="cut_surface" if cut_surface else "cap",
            owner=owner,
            normal=_tuple(axis_w),
            outside_direction=5,
            normal_sign=normal_sign,
        ),
    )


def _rectangle_curve_patches(
    owner: PlacedSDF2D,
    profile: RectangleProfile,
    axis_u: NDArray[np.float64],
    axis_v: NDArray[np.float64],
) -> tuple[BoundaryCurvePatch, ...]:
    return (
        BoundaryCurvePatch(
            owner.object_id,
            "-U",
            "edge",
            owner,
            _tuple(-axis_u),
            0,
        ),
        BoundaryCurvePatch(
            owner.object_id,
            "+U",
            "edge",
            owner,
            _tuple(axis_u),
            1,
        ),
        BoundaryCurvePatch(
            owner.object_id,
            "-V",
            "edge",
            owner,
            _tuple(-axis_v),
            2,
        ),
        BoundaryCurvePatch(
            owner.object_id,
            "+V",
            "edge",
            owner,
            _tuple(axis_v),
            3,
        ),
    )


def _pick_surface_patch_candidates(
    root: SDFNode,
    ray_origin: NDArray[np.float64],
    ray_direction: NDArray[np.float64],
    tolerance: float,
) -> list[tuple[float, BoundaryPatchHit]]:
    ray_points: list[tuple[float, BoundarySurfacePatch, NDArray[np.float64]]] = []
    direction = np.asarray(ray_direction, dtype=np.float64)
    length = np.linalg.norm(direction)
    if length <= 1.0e-12:
        return []
    direction = direction / length
    for patch in surface_patches_for_root(root):
        for travel, point in _surface_patch_ray_points(patch, ray_origin, direction):
            if travel < 0.0:
                continue
            ray_points.append((travel, patch, point))
    ray_points.sort(key=lambda item: item[0])
    first_hit: tuple[float, BoundaryPatchHit] | None = None
    for travel, patch, point in ray_points:
        hit = _surface_patch_hit(root, patch, point, None, tolerance)
        if hit is not None:
            if hit.patch_type == "cut_surface":
                return [(travel, hit)]
            if first_hit is None:
                first_hit = (travel, hit)
    return [first_hit] if first_hit is not None else []


def _surface_patch_ray_points(
    patch: BoundarySurfacePatch,
    ray_origin: NDArray[np.float64],
    ray_direction: NDArray[np.float64],
) -> tuple[tuple[float, NDArray[np.float64]], ...]:
    owner = patch.owner
    if isinstance(owner, Box):
        return _box_patch_ray_points(owner, patch, ray_origin, ray_direction)
    if isinstance(owner, (Cylinder, Cone, CappedCone)):
        return _cylinder_patch_ray_points(owner, patch, ray_origin, ray_direction)
    if isinstance(owner, Sphere):
        return _sphere_patch_ray_points(owner, ray_origin, ray_direction)
    return tuple()


def _box_patch_ray_points(
    owner: Box,
    patch: BoundarySurfacePatch,
    ray_origin: NDArray[np.float64],
    ray_direction: NDArray[np.float64],
) -> tuple[tuple[float, NDArray[np.float64]], ...]:
    face = patch.patch_id.split(".")[-1]
    axis_index = {"-X": 0, "+X": 0, "-Y": 1, "+Y": 1, "-Z": 2, "+Z": 2}.get(face)
    if axis_index is None:
        return tuple()
    sign = -1.0 if face.startswith("-") else 1.0
    axes = (
        np.asarray(owner.axis_u, dtype=np.float64),
        np.asarray(owner.axis_v, dtype=np.float64),
        np.asarray(owner.axis_w, dtype=np.float64),
    )
    normal = sign * axes[axis_index]
    point_on_plane = (
        np.asarray(owner.center, dtype=np.float64)
        + sign * owner.half_size[axis_index] * axes[axis_index]
    )
    denominator = float(np.dot(ray_direction, normal))
    if abs(denominator) <= 1.0e-12:
        return tuple()
    travel = float(np.dot(point_on_plane - ray_origin, normal) / denominator)
    if travel < 0.0:
        return tuple()
    point = ray_origin + travel * ray_direction
    return ((travel, point),)


def _cylinder_patch_ray_points(
    owner: Cylinder | Cone | CappedCone,
    patch: BoundarySurfacePatch,
    ray_origin: NDArray[np.float64],
    ray_direction: NDArray[np.float64],
) -> tuple[tuple[float, NDArray[np.float64]], ...]:
    radius = _cylinder_like_radius(owner)
    if radius is None:
        return tuple()
    patch_name = patch.patch_id.split(".")[-1]
    origin_local = _oriented_local(
        ray_origin,
        owner.center,
        owner.axis_u,
        owner.axis_v,
        owner.axis_w,
    )
    direction_local = np.asarray(
        (
            np.dot(ray_direction, np.asarray(owner.axis_u, dtype=np.float64)),
            np.dot(ray_direction, np.asarray(owner.axis_v, dtype=np.float64)),
            np.dot(ray_direction, np.asarray(owner.axis_w, dtype=np.float64)),
        ),
        dtype=np.float64,
    )
    axes = (
        np.asarray(owner.axis_u, dtype=np.float64),
        np.asarray(owner.axis_v, dtype=np.float64),
        np.asarray(owner.axis_w, dtype=np.float64),
    )
    result: list[tuple[float, NDArray[np.float64]]] = []
    if patch_name == "side_wall":
        a = float(direction_local[0] ** 2 + direction_local[1] ** 2)
        if a <= 1.0e-12:
            return tuple()
        b = float(2.0 * (origin_local[0] * direction_local[0] + origin_local[1] * direction_local[1]))
        c = float(origin_local[0] ** 2 + origin_local[1] ** 2 - radius**2)
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return tuple()
        root = float(np.sqrt(discriminant))
        for travel in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
            z = float(origin_local[2] + travel * direction_local[2])
            if travel >= 0.0 and abs(z) <= owner.half_height + PATCH_TOLERANCE:
                result.append((travel, ray_origin + travel * ray_direction))
        return tuple(sorted(result, key=lambda item: item[0]))
    cap_sign = -1.0 if patch_name == "-Z_cap" else 1.0 if patch_name == "+Z_cap" else None
    if cap_sign is None:
        return tuple()
    if abs(float(direction_local[2])) <= 1.0e-12:
        return tuple()
    travel = float((cap_sign * owner.half_height - origin_local[2]) / direction_local[2])
    if travel < 0.0:
        return tuple()
    point_local = origin_local + travel * direction_local
    if point_local[0] ** 2 + point_local[1] ** 2 > (radius + PATCH_TOLERANCE) ** 2:
        return tuple()
    point = (
        np.asarray(owner.center, dtype=np.float64)
        + point_local[0] * axes[0]
        + point_local[1] * axes[1]
        + point_local[2] * axes[2]
    )
    return ((travel, point),)


def _sphere_patch_ray_points(
    owner: Sphere,
    ray_origin: NDArray[np.float64],
    ray_direction: NDArray[np.float64],
) -> tuple[tuple[float, NDArray[np.float64]], ...]:
    relative = ray_origin - np.asarray(owner.center, dtype=np.float64)
    b = float(2.0 * np.dot(relative, ray_direction))
    c = float(np.dot(relative, relative) - owner.radius**2)
    discriminant = b * b - 4.0 * c
    if discriminant < 0.0:
        return tuple()
    root = float(np.sqrt(discriminant))
    result = []
    for travel in ((-b - root) * 0.5, (-b + root) * 0.5):
        if travel >= 0.0:
            result.append((travel, ray_origin + travel * ray_direction))
    return tuple(sorted(result, key=lambda item: item[0]))


def _select_surface_patch_hit(
    root: SDFNode,
    candidates: list[tuple[float, BoundaryPatchHit]],
) -> BoundaryPatchHit:
    first = candidates[0][1]
    for _travel, candidate in candidates:
        if candidate.patch_type == "cut_surface":
            return candidate
    return first


def _surface_hit_with_selector(
    root: SDFNode,
    hit: BoundaryPatchHit,
    selector_objects: Sequence[SDFNode],
    tolerance: float,
) -> BoundaryPatchHit:
    inside_candidates: list[tuple[float, BoundarySelector]] = []
    outside_candidates: list[tuple[float, BoundarySelector]] = []
    point = np.asarray(hit.point, dtype=np.float64)
    for selector in selector_objects:
        if not _selector_has_surface_preview_volume(selector):
            continue
        metadata = boundary_selector_from_node(selector, domain_dimension=3)
        if metadata is None:
            continue
        value = float(surface_selector_values(root, selector, point.reshape(1, 3))[0])
        if value <= max(tolerance, PATCH_TOLERANCE):
            inside_candidates.append((value, metadata))
        else:
            outside_candidates.append((value, metadata))
    if inside_candidates:
        _value, selector = min(inside_candidates, key=lambda item: item[0])
        return replace(hit, selector=replace(selector, side="inside"))
    if outside_candidates:
        _value, selector = min(outside_candidates, key=lambda item: item[0])
        return replace(hit, selector=replace(selector, side="outside"))
    if not inside_candidates and not outside_candidates:
        return hit
    return hit


def _surface_patch_hit(
    root: SDFNode,
    patch: BoundarySurfacePatch,
    point: NDArray[np.float64],
    final_normal: NDArray[np.float64] | None,
    tolerance: float,
) -> BoundaryPatchHit | None:
    if abs(_evaluate_scalar(patch.owner, point)) > max(4.0 * tolerance, PATCH_TOLERANCE):
        return None
    if not _surface_patch_contains(patch, point, tolerance):
        return None
    if abs(_evaluate_scalar(root, point)) > max(tolerance, PATCH_TOLERANCE):
        return None
    if final_normal is None:
        final_normal = _sdf_normal(root, point, tolerance)
    patch_normal = _surface_patch_normal(patch, point)
    if patch_normal is None:
        return None
    if _normal_alignment(patch_normal, final_normal) < 0.82:
        return None
    return BoundaryPatchHit(
        point=_tuple(point),
        owner_object_id=patch.owner_object_id,
        patch_id=patch.patch_id,
        patch_type=patch.patch_type,
        normal=_tuple(patch_normal),
        outside_direction=patch.outside_direction,
        selector=patch.selector,
    )


def _pick_2d_boundary_patch(
    root: SDFNode,
    ray_origin: NDArray[np.float64],
    ray_direction: NDArray[np.float64],
) -> BoundaryPatchHit | None:
    if not isinstance(root, PlacedSDF2D):
        return None
    normal = np.asarray(root.normal, dtype=np.float64)
    denominator = float(np.dot(ray_direction, normal))
    if abs(denominator) <= 1.0e-12:
        return None
    travel = float(np.dot(np.asarray(root.origin) - ray_origin, normal) / denominator)
    if travel < 0.0:
        return None
    point = ray_origin + travel * ray_direction
    u, v, plane = root.project_numpy(
        np.asarray([point[0]], dtype=np.float64),
        np.asarray([point[1]], dtype=np.float64),
        np.asarray([point[2]], dtype=np.float64),
    )
    if abs(float(plane[0])) > PATCH_TOLERANCE:
        return None
    if abs(float(root.profile.to_numpy(u, v)[0])) > CURVE_PATCH_PICK_TOLERANCE:
        return None
    candidates = [
        _curve_patch_hit(patch, point, CURVE_PATCH_PICK_TOLERANCE)
        for patch in curve_patches_for_owner(root)
    ]
    hits = [hit for hit in candidates if hit is not None]
    if not hits:
        return None
    return hits[0]


def _curve_patch_hit(
    patch: BoundaryCurvePatch,
    point: NDArray[np.float64],
    tolerance: float,
) -> BoundaryPatchHit | None:
    if not _curve_patch_contains(patch, point, tolerance):
        return None
    normal = _curve_patch_normal(patch, point)
    if normal is None:
        return None
    return BoundaryPatchHit(
        point=_tuple(point),
        owner_object_id=patch.owner_object_id,
        patch_id=patch.patch_id,
        patch_type=patch.patch_type,
        normal=_tuple(normal),
        outside_direction=patch.outside_direction,
        selector=patch.selector,
    )


def _surface_patch_contains(
    patch: BoundarySurfacePatch,
    point: NDArray[np.float64],
    tolerance: float,
) -> bool:
    owner = patch.owner
    if isinstance(owner, Box):
        local = _oriented_local(point, owner.center, owner.axis_u, owner.axis_v, owner.axis_w)
        half = np.asarray(owner.half_size, dtype=np.float64)
        face = patch.patch_id.split(".")[-1]
        index = {"-X": 0, "+X": 0, "-Y": 1, "+Y": 1, "-Z": 2, "+Z": 2}.get(face)
        if index is None:
            return False
        sign = -1.0 if face.startswith("-") else 1.0
        if abs(float(local[index] - sign * half[index])) > max(4.0 * tolerance, PATCH_TOLERANCE):
            return False
        other = [axis for axis in range(3) if axis != index]
        return all(abs(float(local[axis])) <= half[axis] + PATCH_TOLERANCE for axis in other)
    if isinstance(owner, (Cylinder, Cone, CappedCone)):
        local = _oriented_local(point, owner.center, owner.axis_u, owner.axis_v, owner.axis_w)
        patch_name = patch.patch_id.split(".")[-1]
        if patch_name == "side_wall":
            return abs(float(abs(local[2]) - owner.half_height)) > PATCH_TOLERANCE
        if patch_name == "-Z_cap":
            return abs(float(local[2] + owner.half_height)) <= max(4.0 * tolerance, PATCH_TOLERANCE)
        if patch_name == "+Z_cap":
            return abs(float(local[2] - owner.half_height)) <= max(4.0 * tolerance, PATCH_TOLERANCE)
    return True


def _curve_patch_contains(
    patch: BoundaryCurvePatch,
    point: NDArray[np.float64],
    tolerance: float = CURVE_PATCH_PICK_TOLERANCE,
) -> bool:
    owner = patch.owner
    assert owner.profile is not None
    u, v, _plane = owner.project_numpy(
        np.asarray([point[0]], dtype=np.float64),
        np.asarray([point[1]], dtype=np.float64),
        np.asarray([point[2]], dtype=np.float64),
    )
    profile = owner.profile
    u_value = float(u[0])
    v_value = float(v[0])
    if isinstance(profile, SquareProfile):
        profile = profile._rectangle()
    if isinstance(profile, RectangleProfile):
        cu, cv = profile.center
        hu, hv = profile.half_size
        if patch.patch_id == "-U":
            return abs(u_value - (cu - hu)) <= tolerance and cv - hv - tolerance <= v_value <= cv + hv + tolerance
        if patch.patch_id == "+U":
            return abs(u_value - (cu + hu)) <= tolerance and cv - hv - tolerance <= v_value <= cv + hv + tolerance
        if patch.patch_id == "-V":
            return abs(v_value - (cv - hv)) <= tolerance and cu - hu - tolerance <= u_value <= cu + hu + tolerance
        if patch.patch_id == "+V":
            return abs(v_value - (cv + hv)) <= tolerance and cu - hu - tolerance <= u_value <= cu + hu + tolerance
    return abs(float(owner.profile.to_numpy(u, v)[0])) <= tolerance


def _surface_patch_normal(
    patch: BoundarySurfacePatch,
    point: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    if patch.normal is not None:
        normal = np.asarray(patch.normal, dtype=np.float64)
    else:
        normal = _sdf_normal(patch.owner, point, PATCH_TOLERANCE)
    length = np.linalg.norm(normal)
    if length <= 1.0e-12:
        return None
    return np.asarray(normal * patch.normal_sign / length, dtype=np.float64)


def _curve_patch_normal(
    patch: BoundaryCurvePatch,
    point: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    if patch.normal is not None:
        normal = np.asarray(patch.normal, dtype=np.float64)
    else:
        normal = _sdf_normal(patch.owner, point, PATCH_TOLERANCE)
    length = np.linalg.norm(normal)
    if length <= 1.0e-12:
        return None
    return np.asarray(normal / length, dtype=np.float64)


def _region_patch_normal(patch: BoundaryPatch) -> tuple[float, float, float]:
    if isinstance(patch, BoundarySurfacePatch):
        if patch.normal is None:
            return (0.0, 0.0, 0.0)
        normal = np.asarray(patch.normal, dtype=np.float64) * patch.normal_sign
        length = np.linalg.norm(normal)
        if length <= 1.0e-12:
            return (0.0, 0.0, 0.0)
        return _tuple(normal / length)
    if patch.normal is None:
        return (0.0, 0.0, 0.0)
    return patch.normal


def _pick_sdf_surface(
    root: SDFNode,
    ray_origin: NDArray[np.float64],
    ray_direction: NDArray[np.float64],
    *,
    hit_tolerance: float,
    maximum_travel: float,
) -> NDArray[np.float64] | None:
    travel = 0.0
    for _step in range(160):
        point = ray_origin + ray_direction * travel
        value = _evaluate_scalar(root, point)
        if abs(value) < hit_tolerance:
            return point
        travel += max(abs(value), 0.0002)
        if travel > maximum_travel:
            break
    return None


def _sdf_normal(
    root: SDFNode,
    point: NDArray[np.float64],
    step: float,
) -> NDArray[np.float64]:
    gradient = np.empty(3, dtype=np.float64)
    for axis in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[axis] = step
        gradient[axis] = _evaluate_scalar(root, point + offset) - _evaluate_scalar(root, point - offset)
    length = np.linalg.norm(gradient)
    return np.asarray(gradient / max(length, 1.0e-12), dtype=np.float64)


def _evaluate_scalar(root: SDFNode, point: NDArray[np.float64]) -> float:
    return float(
        root.to_numpy(
            np.asarray([point[0]], dtype=np.float64),
            np.asarray([point[1]], dtype=np.float64),
            np.asarray([point[2]], dtype=np.float64),
        )[0]
    )


def _find_node_by_object_id(root: SDFNode, object_id: int) -> SDFNode | None:
    if root.object_id == object_id:
        return root
    for child in root.children():
        found = _find_node_by_object_id(child, object_id)
        if found is not None:
            return found
    return None


def _find_selector_node(
    root: SDFNode,
    selector_id: str | None,
    selector_objects: Sequence[SDFNode],
) -> SDFNode | None:
    object_id = _selector_object_id(selector_id)
    if object_id is None:
        return None
    for selector in selector_objects:
        if selector.object_id == object_id:
            return selector
    return _find_node_by_object_id(root, object_id)


def _selector_object_id(selector_id: str | None) -> int | None:
    if selector_id is None:
        return None
    prefix = "selector:"
    if not selector_id.startswith(prefix):
        return None
    try:
        return int(selector_id[len(prefix):])
    except ValueError:
        return None


def _cylinder_like_radius(owner: Cylinder | Cone | CappedCone) -> float | None:
    if isinstance(owner, Cylinder):
        return owner.radius
    if isinstance(owner, Cone):
        return owner.radius
    if isinstance(owner, CappedCone):
        return max(owner.radius_a, owner.radius_b)
    return None


def _oriented_local(
    point: NDArray[np.float64],
    center: tuple[float, float, float],
    axis_u: tuple[float, float, float],
    axis_v: tuple[float, float, float],
    axis_w: tuple[float, float, float],
) -> NDArray[np.float64]:
    relative = point - np.asarray(center, dtype=np.float64)
    return np.asarray(
        (
            np.dot(relative, np.asarray(axis_u, dtype=np.float64)),
            np.dot(relative, np.asarray(axis_v, dtype=np.float64)),
            np.dot(relative, np.asarray(axis_w, dtype=np.float64)),
        ),
        dtype=np.float64,
    )


def _normal_alignment(
    first: tuple[float, float, float] | NDArray[np.float64],
    second: tuple[float, float, float] | NDArray[np.float64],
) -> float:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    first_array /= max(np.linalg.norm(first_array), 1.0e-12)
    second_array /= max(np.linalg.norm(second_array), 1.0e-12)
    return float(np.dot(first_array, second_array))


def _orthonormal_completion(
    axis_u: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    reference = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(np.dot(axis_u, reference))) > 0.9:
        reference = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    axis_v = np.cross(reference, axis_u)
    axis_v /= max(np.linalg.norm(axis_v), 1.0e-12)
    axis_w = np.cross(axis_u, axis_v)
    axis_w /= max(np.linalg.norm(axis_w), 1.0e-12)
    return axis_v, axis_w


def _patch_id(name: str, cut_surface: bool) -> str:
    return f"cut_surface.{name}" if cut_surface else name


def _tuple(values: NDArray[np.float64]) -> tuple[float, float, float]:
    return tuple(float(value) for value in values)
