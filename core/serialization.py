from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .boundary import BoundaryRegion
from .mesher import FluidDomain
from .scene import SceneDocument
from .sdf import (
    BezierCurveProfile,
    BezierSurfaceProfile,
    BezierTube,
    BinaryProfile1D,
    BinaryProfile,
    Box,
    BoxFrame,
    CappedCone,
    CircleProfile,
    Cone,
    Cylinder,
    Difference,
    EllipseProfile,
    Extrude,
    Intersection,
    OffsetProfile,
    OffsetProfile1D,
    PlacedSDF1D,
    PlacedPolyline2D,
    PlacedSDF2D,
    PolygonProfile,
    PolylineTube,
    PolylineProfile,
    Pyramid,
    RectangleProfile,
    RegularPolygonProfile,
    Revolve,
    Rotate,
    RoundedRectangleProfile,
    Scale,
    SegmentProfile,
    SmoothUnion,
    Sphere,
    SquareProfile,
    Torus,
    Translate,
    Union,
)
from .sdf.base import SDFNode
from .sdf.operators import BinarySDFOperator
from .sdf.primitives_1d import Profile1D
from .sdf.primitives_2d import Profile2D
from .sdf.transforms import UnaryTransform

SCENE_FORMAT_VERSION = 4


def save_scene(document: SceneDocument, path: str | Path) -> None:
    records: dict[int, dict[str, Any]] = {}
    for _handle, node, _parent in document.walk():
        if isinstance(node, SDFNode):
            records.setdefault(node.object_id, _node_to_record(node))
    payload = {
        "format": "casocad_scene",
        "version": SCENE_FORMAT_VERSION,
        "unit": "m",
        "root_object_ids": [node.object_id for node in document.objects],
        "objects": [records[key] for key in sorted(records)],
        "boundary_regions": [
            {
                "object_id": region.object_id,
                "name": region.name,
                "owner_object_id": region.owner_object_id,
                "outside_direction": region.outside_direction,
                "patch_id": region.patch_id,
                "patch_type": region.patch_type,
                "selector_id": region.selector_id,
                "selector_type": region.selector_type,
                "selector_side": region.selector_side,
                "selector_start": region.selector_start,
                "selector_end": region.selector_end,
            }
            for region in sorted(
                document.boundary_regions, key=lambda item: item.object_id
            )
        ],
        "fluid_domain": (
            {
                "root_object_id": document.fluid_domain.root.object_id,
                "tag_object_ids": [
                    tag.object_id for tag in document.fluid_domain.tag_objects
                ],
                "selector_object_ids": [
                    selector.object_id
                    for selector in document.fluid_domain.selector_objects
                ],
            }
            if document.fluid_domain is not None
            else {
                "root_object_id": None,
                "tag_object_ids": [],
                "selector_object_ids": [],
            }
        ),
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_scene(path: str | Path) -> SceneDocument:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "casocad_scene":
        raise ValueError("not a casoCAD scene file")
    version = payload.get("version")
    if version not in {2, 3, SCENE_FORMAT_VERSION}:
        raise ValueError(f"unsupported scene version: {payload.get('version')}")
    records = {int(item["object_id"]): item for item in payload.get("objects", [])}
    built: dict[int, SDFNode] = {}

    def build(object_id: int) -> SDFNode:
        if object_id in built:
            return built[object_id]
        record = records[object_id]
        node = _node_from_record(record, build)
        built[object_id] = node
        return node

    roots = [build(int(object_id)) for object_id in payload.get("root_object_ids", [])]
    boundary_regions = [
        BoundaryRegion(
            name=str(item["name"]),
            object_id=int(item["object_id"]),
            owner_object_id=int(item["owner_object_id"]),
            outside_direction=(
                int(item["outside_direction"])
                if item.get("outside_direction") is not None
                else None
            ),
            patch_id=(
                str(item["patch_id"])
                if item.get("patch_id") is not None
                else None
            ),
            patch_type=(
                str(item["patch_type"])
                if item.get("patch_type") is not None
                else None
            ),
            selector_id=(
                str(item["selector_id"])
                if item.get("selector_id") is not None
                else None
            ),
            selector_type=(
                str(item["selector_type"])
                if item.get("selector_type") is not None
                else None
            ),
            selector_side=str(item.get("selector_side") or "inside"),
            selector_start=(
                float(item["selector_start"])
                if item.get("selector_start") is not None
                else None
            ),
            selector_end=(
                float(item["selector_end"])
                if item.get("selector_end") is not None
                else None
            ),
        )
        for item in payload.get("boundary_regions", [])
    ]
    document = SceneDocument(roots, boundary_regions=boundary_regions)
    fluid = payload.get("fluid_domain", {})
    root_id = fluid.get("root_object_id")
    if root_id is not None:
        root = build(int(root_id))
        regions_by_id = {
            region.object_id: region for region in document.boundary_regions
        }
        tag_items: list[PlacedSDF1D | PlacedPolyline2D | PlacedSDF2D | BoundaryRegion] = []
        for object_id in (int(item) for item in fluid.get("tag_object_ids", [])):
            region = regions_by_id.get(object_id)
            if region is not None:
                tag_items.append(region)
            elif object_id in records:
                tag = build(object_id)
                if isinstance(tag, (PlacedSDF1D, PlacedPolyline2D, PlacedSDF2D)):
                    tag_items.append(tag)
        tags = tuple(tag_items)
        selector_items: list[SDFNode] = []
        for object_id in (
            int(item) for item in fluid.get("selector_object_ids", [])
        ):
            if object_id in records:
                selector_items.append(build(object_id))
        if root.dimension == 2:
            migrated: list[PlacedSDF1D | PlacedPolyline2D | BoundaryRegion] = []
            for tag in tags:
                if isinstance(tag, (PlacedSDF1D, PlacedPolyline2D, BoundaryRegion)):
                    migrated.append(tag)
                else:
                    raise ValueError(
                        "2D fluid tag objects must be PlacedSDF1D, "
                        "PlacedPolyline2D, or BoundaryRegion"
                    )
            tags = tuple(migrated)
        if not all(
            isinstance(tag, (PlacedSDF1D, PlacedPolyline2D, PlacedSDF2D, BoundaryRegion))
            for tag in tags
        ):
            raise ValueError(
                "fluid tag objects have an unsupported dimension"
            )
        document.fluid_domain = FluidDomain(
            root,
            tuple(
                tag
                for tag in tags
                if isinstance(
                    tag,
                    (PlacedSDF1D, PlacedPolyline2D, PlacedSDF2D, BoundaryRegion),
                )
            ),
            tuple(selector_items),
        )
    document._reindex()
    return document


def _common(node: SDFNode) -> dict[str, Any]:
    return {
        "type": type(node).__name__,
        "name": node.name,
        "object_id": node.object_id,
    }


def _node_to_record(node: SDFNode) -> dict[str, Any]:
    data = _common(node)
    if isinstance(node, Sphere):
        data.update(center=list(node.center), radius=node.radius)
    elif isinstance(node, Box):
        data.update(
            center=list(node.center),
            half_size=list(node.half_size),
            axis_u=list(node.axis_u),
            axis_v=list(node.axis_v),
            axis_w=list(node.axis_w),
        )
    elif isinstance(node, BoxFrame):
        data.update(
            center=list(node.center),
            half_size=list(node.half_size),
            thickness=node.thickness,
            axis_u=list(node.axis_u),
            axis_v=list(node.axis_v),
            axis_w=list(node.axis_w),
        )
    elif isinstance(node, Cylinder):
        data.update(
            center=list(node.center),
            radius=node.radius,
            half_height=node.half_height,
            axis_u=list(node.axis_u),
            axis_v=list(node.axis_v),
            axis_w=list(node.axis_w),
        )
    elif isinstance(node, CappedCone):
        data.update(
            center=list(node.center),
            radius_a=node.radius_a,
            radius_b=node.radius_b,
            half_height=node.half_height,
            axis_u=list(node.axis_u),
            axis_v=list(node.axis_v),
            axis_w=list(node.axis_w),
        )
    elif isinstance(node, Cone):
        data.update(
            center=list(node.center),
            radius=node.radius,
            half_height=node.half_height,
            axis_u=list(node.axis_u),
            axis_v=list(node.axis_v),
            axis_w=list(node.axis_w),
        )
    elif isinstance(node, Pyramid):
        data.update(
            center=list(node.center),
            base_half_size=node.base_half_size,
            half_height=node.half_height,
            axis_u=list(node.axis_u),
            axis_v=list(node.axis_v),
            axis_w=list(node.axis_w),
        )
    elif isinstance(node, Torus):
        data.update(
            center=list(node.center),
            major_radius=node.major_radius,
            minor_radius=node.minor_radius,
            axis_u=list(node.axis_u),
            axis_v=list(node.axis_v),
            axis_w=list(node.axis_w),
        )
    elif isinstance(node, PlacedSDF1D):
        assert node.profile is not None
        data.update(
            profile=_profile_1d_to_dict(node.profile),
            origin=list(node.origin),
            axis_u=list(node.axis_u),
            source_ids=[child.object_id for child in node.sources],
        )
    elif isinstance(node, PlacedPolyline2D):
        assert node.profile is not None
        data.update(
            profile=_profile_to_dict(node.profile),
            origin=list(node.origin),
            axis_u=list(node.axis_u),
            axis_v=list(node.axis_v),
        )
    elif isinstance(node, PlacedSDF2D):
        assert node.profile is not None
        data.update(
            profile=_profile_to_dict(node.profile),
            origin=list(node.origin),
            axis_u=list(node.axis_u),
            axis_v=list(node.axis_v),
            source_ids=[child.object_id for child in node.sources],
        )
    elif isinstance(node, BinarySDFOperator):
        assert node.left is not None and node.right is not None
        data.update(left_id=node.left.object_id, right_id=node.right.object_id)
        if isinstance(node, SmoothUnion):
            data["smoothing"] = node.smoothing
    elif isinstance(node, UnaryTransform):
        assert node.child is not None
        data["child_id"] = node.child.object_id
        if isinstance(node, Translate):
            data["offset"] = list(node.offset)
        elif isinstance(node, Rotate):
            data.update(axis=node.axis, angle_degrees=node.angle_degrees)
        elif isinstance(node, Scale):
            data["factor"] = node.factor
    elif isinstance(node, Extrude):
        assert node.section is not None
        data.update(
            section_id=node.section.object_id,
            height=node.height,
            center_offset=node.center_offset,
        )
    elif isinstance(node, Revolve):
        assert node.section is not None
        data["section_id"] = node.section.object_id
        data["axis"] = node.axis
        data["angle_degrees"] = node.angle_degrees
        if node.axis_origin is not None:
            data["axis_origin"] = list(node.axis_origin)
        if node.axis_direction is not None:
            data["axis_direction"] = list(node.axis_direction)
        if node.radial_direction is not None:
            data["radial_direction"] = list(node.radial_direction)
    elif isinstance(node, PolylineTube):
        data.update(
            points=[list(point) for point in node.points],
            radius=node.radius,
            inner_radius=node.inner_radius,
            caps=node.caps,
        )
    elif isinstance(node, BezierTube):
        data.update(
            points=[list(point) for point in node.points],
            radius=node.radius,
            inner_radius=node.inner_radius,
            caps=node.caps,
        )
    else:
        raise TypeError(f"cannot serialize {type(node).__name__}")
    return data


def _node_from_record(
    data: dict[str, Any], build: Any
) -> SDFNode:
    node_type = str(data["type"])
    common = {
        "name": str(data["name"]),
        "object_id": int(data["object_id"]),
    }
    if node_type == "Sphere":
        return Sphere(**common, center=tuple(data["center"]), radius=float(data["radius"]))
    if node_type == "Box":
        return Box(
            **common,
            center=tuple(data["center"]),
            half_size=tuple(data["half_size"]),
            axis_u=tuple(data.get("axis_u", (1.0, 0.0, 0.0))),
            axis_v=tuple(data.get("axis_v", (0.0, 1.0, 0.0))),
            axis_w=tuple(data.get("axis_w", (0.0, 0.0, 1.0))),
        )
    if node_type == "BoxFrame":
        return BoxFrame(
            **common,
            center=tuple(data["center"]),
            half_size=tuple(data["half_size"]),
            thickness=float(data["thickness"]),
            axis_u=tuple(data.get("axis_u", (1.0, 0.0, 0.0))),
            axis_v=tuple(data.get("axis_v", (0.0, 1.0, 0.0))),
            axis_w=tuple(data.get("axis_w", (0.0, 0.0, 1.0))),
        )
    if node_type == "Cylinder":
        return Cylinder(
            **common,
            center=tuple(data["center"]),
            radius=float(data["radius"]),
            half_height=float(data["half_height"]),
            axis_u=tuple(data.get("axis_u", (1.0, 0.0, 0.0))),
            axis_v=tuple(data.get("axis_v", (0.0, 1.0, 0.0))),
            axis_w=tuple(data.get("axis_w", (0.0, 0.0, 1.0))),
        )
    if node_type == "CappedCone":
        return CappedCone(
            **common,
            center=tuple(data["center"]),
            radius_a=float(data["radius_a"]),
            radius_b=float(data["radius_b"]),
            half_height=float(data["half_height"]),
            axis_u=tuple(data.get("axis_u", (1.0, 0.0, 0.0))),
            axis_v=tuple(data.get("axis_v", (0.0, 1.0, 0.0))),
            axis_w=tuple(data.get("axis_w", (0.0, 0.0, 1.0))),
        )
    if node_type == "Cone":
        return Cone(
            **common,
            center=tuple(data["center"]),
            radius=float(data["radius"]),
            half_height=float(data["half_height"]),
            axis_u=tuple(data.get("axis_u", (1.0, 0.0, 0.0))),
            axis_v=tuple(data.get("axis_v", (0.0, 1.0, 0.0))),
            axis_w=tuple(data.get("axis_w", (0.0, 0.0, 1.0))),
        )
    if node_type == "Pyramid":
        return Pyramid(
            **common,
            center=tuple(data["center"]),
            base_half_size=float(data["base_half_size"]),
            half_height=float(data["half_height"]),
            axis_u=tuple(data.get("axis_u", (1.0, 0.0, 0.0))),
            axis_v=tuple(data.get("axis_v", (0.0, 1.0, 0.0))),
            axis_w=tuple(data.get("axis_w", (0.0, 0.0, 1.0))),
        )
    if node_type == "Torus":
        return Torus(
            **common,
            center=tuple(data["center"]),
            major_radius=float(data["major_radius"]),
            minor_radius=float(data["minor_radius"]),
            axis_u=tuple(data.get("axis_u", (1.0, 0.0, 0.0))),
            axis_v=tuple(data.get("axis_v", (0.0, 1.0, 0.0))),
            axis_w=tuple(data.get("axis_w", (0.0, 0.0, 1.0))),
        )
    if node_type == "PlacedSDF2D":
        return PlacedSDF2D(
            **common,
            profile=_profile_from_dict(data["profile"]),
            origin=tuple(data["origin"]),
            axis_u=tuple(data["axis_u"]),
            axis_v=tuple(data["axis_v"]),
            sources=tuple(build(int(item)) for item in data.get("source_ids", [])),
        )
    if node_type == "PlacedPolyline2D":
        return PlacedPolyline2D(
            **common,
            profile=_profile_from_dict(data["profile"]),
            origin=tuple(data["origin"]),
            axis_u=tuple(data["axis_u"]),
            axis_v=tuple(data["axis_v"]),
        )
    if node_type == "PlacedSDF1D":
        return PlacedSDF1D(
            **common,
            profile=_profile_1d_from_dict(data["profile"]),
            origin=tuple(data["origin"]),
            axis_u=tuple(data["axis_u"]),
            sources=tuple(
                build(int(item)) for item in data.get("source_ids", [])
            ),
        )
    binary_types = {
        "Union": Union,
        "Intersection": Intersection,
        "Difference": Difference,
        "SmoothUnion": SmoothUnion,
    }
    if node_type in binary_types:
        extra = {"smoothing": float(data["smoothing"])} if node_type == "SmoothUnion" else {}
        return binary_types[node_type](
            **common,
            left=build(int(data["left_id"])),
            right=build(int(data["right_id"])),
            **extra,
        )
    if node_type == "Translate":
        return Translate(**common, child=build(int(data["child_id"])), offset=tuple(data["offset"]))
    if node_type == "Rotate":
        return Rotate(
            **common,
            child=build(int(data["child_id"])),
            axis=str(data["axis"]),
            angle_degrees=float(data["angle_degrees"]),
        )
    if node_type == "Scale":
        return Scale(**common, child=build(int(data["child_id"])), factor=float(data["factor"]))
    section = (
        build(int(data["section_id"]))
        if "section_id" in data
        else None
    )
    if section is not None and not isinstance(section, PlacedSDF2D):
        raise ValueError(f"{node_type} section must be PlacedSDF2D")
    if node_type == "Extrude":
        return Extrude(
            **common,
            section=section,
            height=float(data["height"]),
            center_offset=float(data.get("center_offset", 0.0)),
        )
    if node_type == "Revolve":
        return Revolve(
            **common,
            section=section,
            axis=str(data.get("axis", "v")),
            axis_origin=(
                tuple(data["axis_origin"])
                if data.get("axis_origin") is not None
                else None
            ),
            axis_direction=(
                tuple(data["axis_direction"])
                if data.get("axis_direction") is not None
                else None
            ),
            radial_direction=(
                tuple(data["radial_direction"])
                if data.get("radial_direction") is not None
                else None
            ),
            angle_degrees=float(data.get("angle_degrees", 360.0)),
        )
    if node_type == "PolylineTube":
        return PolylineTube(
            **common,
            points=tuple(tuple(point) for point in data["points"]),
            radius=float(data["radius"]),
            inner_radius=float(data.get("inner_radius", 0.0)),
            caps=str(data.get("caps", "round")),
        )
    if node_type == "BezierTube":
        return BezierTube(
            **common,
            points=tuple(tuple(point) for point in data["points"]),
            radius=float(data["radius"]),
            inner_radius=float(data.get("inner_radius", 0.0)),
            caps=str(data.get("caps", "round")),
        )
    raise ValueError(f"unknown SDF node type: {node_type}")


def _profile_to_dict(profile: Profile2D) -> dict[str, Any]:
    data: dict[str, Any] = {"type": type(profile).__name__}
    for key, value in vars(profile).items():
        if isinstance(value, Profile2D):
            data[key] = _profile_to_dict(value)
        elif isinstance(value, tuple):
            data[key] = list(value)
        else:
            data[key] = value
    return data


def _profile_from_dict(data: dict[str, Any]) -> Profile2D:
    profile_types = {
        "CircleProfile": CircleProfile,
        "RectangleProfile": RectangleProfile,
        "SquareProfile": SquareProfile,
        "RoundedRectangleProfile": RoundedRectangleProfile,
        "EllipseProfile": EllipseProfile,
        "RegularPolygonProfile": RegularPolygonProfile,
        "BezierCurveProfile": BezierCurveProfile,
        "BezierSurfaceProfile": BezierSurfaceProfile,
        "PolylineProfile": PolylineProfile,
        "PolygonProfile": PolygonProfile,
        "OffsetProfile": OffsetProfile,
        "BinaryProfile": BinaryProfile,
    }
    profile_type = str(data["type"])
    constructor = profile_types.get(profile_type)
    if constructor is None:
        raise ValueError(f"unknown 2D profile type: {profile_type}")
    kwargs = {key: value for key, value in data.items() if key != "type"}
    for key in ("center", "half_size", "semi_axes"):
        if key in kwargs and isinstance(kwargs[key], list):
            kwargs[key] = tuple(kwargs[key])
    if "points" in kwargs and isinstance(kwargs["points"], list):
        kwargs["points"] = tuple(tuple(point) for point in kwargs["points"])
    for key in ("left", "right", "child"):
        if key in kwargs:
            kwargs[key] = _profile_from_dict(kwargs[key])
    return constructor(**kwargs)


def _profile_1d_to_dict(profile: Profile1D) -> dict[str, Any]:
    data: dict[str, Any] = {"type": type(profile).__name__}
    for key, value in vars(profile).items():
        data[key] = (
            _profile_1d_to_dict(value)
            if isinstance(value, Profile1D)
            else value
        )
    return data


def _profile_1d_from_dict(data: dict[str, Any]) -> Profile1D:
    profile_types = {
        "IntervalProfile": SegmentProfile,
        "OffsetProfile1D": OffsetProfile1D,
        "BinaryProfile1D": BinaryProfile1D,
        "SegmentProfile": SegmentProfile,
    }
    profile_type = str(data["type"])
    constructor = profile_types.get(profile_type)
    if constructor is None:
        raise ValueError(f"unknown 1D profile type: {profile_type}")
    kwargs = {key: value for key, value in data.items() if key != "type"}
    for key in ("left", "right", "child"):
        if key in kwargs:
            kwargs[key] = _profile_1d_from_dict(kwargs[key])
    return constructor(**kwargs)
