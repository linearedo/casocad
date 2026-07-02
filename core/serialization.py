from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .boundary import BoundaryCut, BoundaryRegion
from .domain import FluidDomain
from .scene import SceneDocument
from .sdf import (
    BinaryProfile,
    BinaryProfile1D,
    Box,
    BoxFrame,
    CappedCone,
    CircleProfile,
    Cone,
    Cylinder,
    Difference,
    DistanceOffsetProfile,
    EllipseProfile,
    Extrude,
    Intersection,
    OffsetProfile,
    OffsetProfile1D,
    PlacedPolyline1D,
    PlacedSDF1D,
    PlacedSDF2D,
    PolygonProfile,
    PolylineProfile,
    PolylineTube,
    Pyramid,
    QuadraticBezierCurveProfile,
    QuadraticBezierSurfaceProfile,
    QuadraticBezierTube,
    RectangleProfile,
    RegularPolygonProfile,
    Revolve,
    Rotate,
    RoundedRectangleProfile,
    Scale,
    SegmentProfile,
    Sphere,
    SquareProfile,
    Torus,
    Translate,
    Union,
    Xor,
)
from .sdf.base import SDFNode
from .sdf.operators import BinarySDFOperator
from .sdf.primitives_1d import Profile1D
from .sdf.primitives_2d import Profile2D
from .sdf.roles import DomainKind
from .sdf.transforms import UnaryTransform

SCENE_FORMAT_VERSION = 1
FORMAT_NAME = "casocad"
DEFAULT_AXIS_U = (1.0, 0.0, 0.0)
DEFAULT_AXIS_V = (0.0, 1.0, 0.0)
DEFAULT_AXIS_W = (0.0, 0.0, 1.0)


def save_scene(document: SceneDocument, path: str | Path) -> None:
    names = _scene_names(document)
    node_items = sorted(
        (
            (node.object_id, key, node)
            for key, node in names.nodes_by_key.items()
        ),
        key=lambda item: item[0],
    )
    region_items = sorted(
        (
            (region.object_id, key, region)
            for key, region in names.regions_by_key.items()
        ),
        key=lambda item: item[0],
    )
    payload: dict[str, Any] = {
        "format": FORMAT_NAME,
        "version": SCENE_FORMAT_VERSION,
        "unit": "m",
        "root_objects": [names.node_keys[id(node)] for node in document.objects],
        "objects": {
            key: _node_to_record(node, names)
            for _object_id, key, node in node_items
        },
    }
    if region_items:
        payload["boundary_regions"] = {
            key: _boundary_region_to_record(region, names)
            for _object_id, key, region in region_items
        }
    domain_records: dict[str, Any] = {}
    nodes_by_object_id = {
        node.object_id: node
        for node in names.nodes_by_key.values()
        if node.object_id > 0
    }
    fluid = document.fluid_domain
    effective_domain_kinds = dict(document.domain_kinds)
    if fluid is not None:
        effective_domain_kinds.setdefault(fluid.root.object_id, DomainKind.FLUID)
    for object_id, kind in sorted(effective_domain_kinds.items()):
        root = nodes_by_object_id.get(object_id)
        if root is None:
            continue
        root_key = names.node_keys[id(root)]
        record: dict[str, Any] = {
            "type": kind.value,
            "root": root_key,
        }
        if fluid is not None and fluid.root is root:
            record["tags"] = [names.key_for_tag(tag) for tag in fluid.tag_objects]
            record["selectors"] = [
                names.node_keys[id(selector)]
                for selector in fluid.selector_objects
            ]
        domain_records[root_key] = record
    if domain_records:
        payload["domains"] = domain_records
    Path(path).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def load_scene(path: str | Path) -> SceneDocument:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT_NAME:
        raise ValueError("not a casoCAD scene file")
    version = payload.get("version")
    if version != SCENE_FORMAT_VERSION:
        raise ValueError(f"unsupported scene version: {version}")
    raw_objects = payload.get("objects", {})
    if not isinstance(raw_objects, dict):
        raise ValueError("casoCAD scene objects must be a name-keyed object")
    allocator = _ObjectIdAllocator()
    built: dict[str, SDFNode] = {}

    def build(name: str) -> SDFNode:
        if name in built:
            return built[name]
        if name not in raw_objects:
            raise ValueError(f"unknown object reference: {name}")
        record = raw_objects[name]
        if not isinstance(record, dict):
            raise ValueError(f"object '{name}' must be a JSON object")
        node = _node_from_record(
            name,
            record,
            build,
            allocator.allocate(),
        )
        built[name] = node
        return node

    root_names = payload.get("root_objects", [])
    if not isinstance(root_names, list):
        raise ValueError("root_objects must be a list")
    roots = [build(str(name)) for name in root_names]
    regions_by_name: dict[str, BoundaryRegion] = {}
    raw_regions = payload.get("boundary_regions", {})
    if not isinstance(raw_regions, dict):
        raise ValueError("boundary_regions must be a name-keyed object")
    for name, raw_region in raw_regions.items():
        if not isinstance(raw_region, dict):
            raise ValueError(f"boundary region '{name}' must be a JSON object")
        regions_by_name[str(name)] = _boundary_region_from_record(
            str(name),
            raw_region,
            build,
            allocator.allocate(),
        )
    document = SceneDocument(
        roots,
        boundary_regions=list(regions_by_name.values()),
    )
    raw_domains = payload.get("domains", {})
    if raw_domains:
        if not isinstance(raw_domains, dict):
            raise ValueError("domains must be a name-keyed object")
        for domain_key, domain_record in raw_domains.items():
            if not isinstance(domain_record, dict):
                raise ValueError(f"domain '{domain_key}' must be a JSON object")
            root_name = str(domain_record["root"])
            root = build(root_name)
            try:
                kind = DomainKind(str(domain_record.get("type", "fluid")))
            except ValueError as error:
                raise ValueError(
                    f"domain '{domain_key}' has unknown type "
                    f"{domain_record.get('type')!r}"
                ) from error
            document.domain_kinds[root.object_id] = kind
            if kind is not DomainKind.FLUID:
                continue
            tags = []
            for tag_name in domain_record.get("tags", []):
                key = str(tag_name)
                if key in regions_by_name:
                    tags.append(regions_by_name[key])
                else:
                    tag = build(key)
                    if not isinstance(
                        tag,
                        (PlacedSDF1D, PlacedPolyline1D, PlacedSDF2D),
                    ):
                        raise ValueError(
                            f"fluid tag '{key}' has an unsupported dimension"
                        )
                    tags.append(tag)
            selectors = tuple(
                build(str(name)) for name in domain_record.get("selectors", [])
            )
            if document.fluid_domain is None:
                document.fluid_domain = FluidDomain(root, tuple(tags), selectors)
    _drop_orphaned_internal_selectors(document)
    document._reindex()
    return document


def _drop_orphaned_internal_selectors(document: SceneDocument) -> None:
    """Remove hidden ``__boundary_selector_*`` nodes left behind by the legacy
    format once no region references them (volume selectors are migrated into
    cut chains at load; interval selectors may still need theirs)."""
    referenced: set[int] = set()
    prefix = "selector:"
    for region in document.boundary_regions:
        if region.selector_id is not None and region.selector_id.startswith(prefix):
            try:
                referenced.add(int(region.selector_id[len(prefix):]))
            except ValueError:
                continue
    document.objects = [
        node
        for node in document.objects
        if not (
            document.is_internal_scene_node(node)
            and node.object_id not in referenced
        )
    ]
    fluid = document.fluid_domain
    if fluid is not None:
        kept = tuple(
            selector
            for selector in fluid.selector_objects
            if selector.object_id in referenced
        )
        if len(kept) != len(fluid.selector_objects):
            document.fluid_domain = FluidDomain(fluid.root, fluid.tag_objects, kept)


class _ObjectIdAllocator:
    def __init__(self) -> None:
        self._next = 1

    def allocate(self) -> int:
        object_id = self._next
        self._next += 1
        return object_id


class _SceneNames:
    def __init__(
        self,
        nodes_by_key: dict[str, SDFNode],
        regions_by_key: dict[str, BoundaryRegion],
    ) -> None:
        self.nodes_by_key = nodes_by_key
        self.regions_by_key = regions_by_key
        self.node_keys = {id(node): key for key, node in nodes_by_key.items()}
        self.region_keys = {
            id(region): key for key, region in regions_by_key.items()
        }

    def key_for_tag(
        self,
        tag: PlacedSDF1D | PlacedPolyline1D | PlacedSDF2D | BoundaryRegion,
    ) -> str:
        if isinstance(tag, BoundaryRegion):
            return self.region_keys[id(tag)]
        return self.node_keys[id(tag)]


def _scene_names(document: SceneDocument) -> _SceneNames:
    nodes: list[SDFNode] = []
    seen: set[int] = set()
    for _handle, item, _parent in document.walk():
        if isinstance(item, SDFNode) and id(item) not in seen:
            seen.add(id(item))
            nodes.append(item)
    used: set[str] = set()
    nodes_by_key = {
        _unique_key(node.name, used): node
        for node in sorted(nodes, key=lambda item: item.object_id)
    }
    regions_by_key = {
        _unique_key(region.name, used): region
        for region in sorted(
            document.boundary_regions,
            key=lambda item: item.object_id,
        )
    }
    return _SceneNames(nodes_by_key, regions_by_key)


def _unique_key(name: str, used: set[str]) -> str:
    base = _clean_key(name)
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _clean_key(name: str) -> str:
    text = name.strip()
    if not text:
        return "object"
    return text


def _display_name(key: str, data: dict[str, Any]) -> str:
    return str(data.get("name", key))


def _node_to_record(node: SDFNode, names: _SceneNames) -> dict[str, Any]:
    data: dict[str, Any] = {"type": _node_type(node)}
    key = names.node_keys[id(node)]
    if node.name != key:
        data["name"] = node.name
    if isinstance(node, Sphere):
        data.update(center=list(node.center), radius=node.radius)
    elif isinstance(node, Box):
        data.update(center=list(node.center), size=_doubled(node.half_size))
        _write_axes(data, node)
    elif isinstance(node, BoxFrame):
        data.update(
            center=list(node.center),
            size=_doubled(node.half_size),
            thickness=node.thickness,
        )
        _write_axes(data, node)
    elif isinstance(node, Cylinder):
        data.update(
            center=list(node.center),
            radius=node.radius,
            height=node.half_height * 2.0,
        )
        _write_axes(data, node)
    elif isinstance(node, CappedCone):
        data.update(
            center=list(node.center),
            radius_a=node.radius_a,
            radius_b=node.radius_b,
            height=node.half_height * 2.0,
        )
        _write_axes(data, node)
    elif isinstance(node, Cone):
        data.update(
            center=list(node.center),
            radius=node.radius,
            height=node.half_height * 2.0,
        )
        _write_axes(data, node)
    elif isinstance(node, Pyramid):
        data.update(
            center=list(node.center),
            base_size=node.base_half_size * 2.0,
            height=node.half_height * 2.0,
        )
        _write_axes(data, node)
    elif isinstance(node, Torus):
        data.update(
            center=list(node.center),
            major_radius=node.major_radius,
            minor_radius=node.minor_radius,
        )
        _write_axes(data, node)
    elif isinstance(node, PlacedSDF1D):
        assert node.profile is not None
        data.update(
            profile=_profile_1d_to_dict(node.profile),
            origin=list(node.origin),
            axis_u=list(node.axis_u),
        )
        if node.sources:
            data["sources"] = [names.node_keys[id(child)] for child in node.sources]
    elif isinstance(node, PlacedPolyline1D):
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
        )
        if node.sources:
            data["sources"] = [names.node_keys[id(child)] for child in node.sources]
    elif isinstance(node, BinarySDFOperator):
        assert node.left is not None and node.right is not None
        data.update(
            left=names.node_keys[id(node.left)],
            right=names.node_keys[id(node.right)],
        )
    elif isinstance(node, UnaryTransform):
        assert node.child is not None
        data["object"] = names.node_keys[id(node.child)]
        if isinstance(node, Translate):
            data["offset"] = list(node.offset)
        elif isinstance(node, Rotate):
            data.update(axis=node.axis, angle_degrees=node.angle_degrees)
        elif isinstance(node, Scale):
            data["factor"] = node.factor
    elif isinstance(node, Extrude):
        assert node.section is not None
        data.update(
            section=names.node_keys[id(node.section)],
            height=node.height,
        )
        if node.center_offset != 0.0:
            data["center_offset"] = node.center_offset
    elif isinstance(node, Revolve):
        assert node.section is not None
        data.update(
            section=names.node_keys[id(node.section)],
            axis=node.axis,
            angle_degrees=node.angle_degrees,
        )
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
    elif isinstance(node, QuadraticBezierTube):
        data.update(
            points=[list(point) for point in node.points],
            radius=node.radius,
            inner_radius=node.inner_radius,
            caps=node.caps,
        )
    else:
        raise TypeError(f"cannot serialize {type(node).__name__}")
    return data


def _node_type(node: SDFNode) -> str:
    names = {
        Sphere: "sphere",
        Box: "box",
        BoxFrame: "box_frame",
        Cylinder: "cylinder",
        CappedCone: "capped_cone",
        Cone: "cone",
        Pyramid: "pyramid",
        Torus: "torus",
        PlacedSDF1D: "placed_sdf_1d",
        PlacedPolyline1D: "placed_polyline_1d",
        PlacedSDF2D: "placed_sdf_2d",
        Union: "union",
        Intersection: "intersection",
        Difference: "difference",
        Xor: "xor",
        Translate: "translate",
        Rotate: "rotate",
        Scale: "scale",
        Extrude: "extrude",
        Revolve: "revolve",
        PolylineTube: "polyline_tube",
        QuadraticBezierTube: "quadratic_bezier_tube",
    }
    return names[type(node)]


def _node_from_record(
    key: str,
    data: dict[str, Any],
    build: Any,
    object_id: int,
) -> SDFNode:
    node_type = str(data["type"])
    common = {
        "name": _display_name(key, data),
        "object_id": object_id,
    }
    if node_type == "sphere":
        return Sphere(
            **common,
            center=_tuple3(data.get("center", (0.0, 0.0, 0.0))),
            radius=float(data["radius"]),
        )
    if node_type == "box":
        return Box(
            **common,
            center=_tuple3(data.get("center", (0.0, 0.0, 0.0))),
            half_size=_half_size(data),
            **_read_axes(data),
        )
    if node_type == "box_frame":
        return BoxFrame(
            **common,
            center=_tuple3(data.get("center", (0.0, 0.0, 0.0))),
            half_size=_half_size(data),
            thickness=float(data["thickness"]),
            **_read_axes(data),
        )
    if node_type == "cylinder":
        return Cylinder(
            **common,
            center=_tuple3(data.get("center", (0.0, 0.0, 0.0))),
            radius=float(data["radius"]),
            half_height=_half_height(data),
            **_read_axes(data),
        )
    if node_type == "capped_cone":
        return CappedCone(
            **common,
            center=_tuple3(data.get("center", (0.0, 0.0, 0.0))),
            radius_a=float(data["radius_a"]),
            radius_b=float(data["radius_b"]),
            half_height=_half_height(data),
            **_read_axes(data),
        )
    if node_type == "cone":
        return Cone(
            **common,
            center=_tuple3(data.get("center", (0.0, 0.0, 0.0))),
            radius=float(data["radius"]),
            half_height=_half_height(data),
            **_read_axes(data),
        )
    if node_type == "pyramid":
        return Pyramid(
            **common,
            center=_tuple3(data.get("center", (0.0, 0.0, 0.0))),
            base_half_size=float(data["base_size"]) * 0.5
            if "base_size" in data
            else float(data["base_half_size"]),
            half_height=_half_height(data),
            **_read_axes(data),
        )
    if node_type == "torus":
        return Torus(
            **common,
            center=_tuple3(data.get("center", (0.0, 0.0, 0.0))),
            major_radius=float(data["major_radius"]),
            minor_radius=float(data["minor_radius"]),
            **_read_axes(data),
        )
    if node_type == "placed_sdf_2d":
        return PlacedSDF2D(
            **common,
            profile=_profile_from_dict(data["profile"]),
            origin=_tuple3(data.get("origin", (0.0, 0.0, 0.0))),
            axis_u=_tuple3(data.get("axis_u", DEFAULT_AXIS_U)),
            axis_v=_tuple3(data.get("axis_v", DEFAULT_AXIS_V)),
            sources=tuple(build(str(item)) for item in data.get("sources", [])),
        )
    if node_type == "placed_polyline_1d":
        return PlacedPolyline1D(
            **common,
            profile=_profile_from_dict(data["profile"]),
            origin=_tuple3(data.get("origin", (0.0, 0.0, 0.0))),
            axis_u=_tuple3(data.get("axis_u", DEFAULT_AXIS_U)),
            axis_v=_tuple3(data.get("axis_v", DEFAULT_AXIS_V)),
        )
    if node_type == "placed_sdf_1d":
        return PlacedSDF1D(
            **common,
            profile=_profile_1d_from_dict(data["profile"]),
            origin=_tuple3(data.get("origin", (0.0, 0.0, 0.0))),
            axis_u=_tuple3(data.get("axis_u", DEFAULT_AXIS_U)),
            sources=tuple(build(str(item)) for item in data.get("sources", [])),
        )
    binary_types = {
        "union": Union,
        "intersection": Intersection,
        "difference": Difference,
        "xor": Xor,
    }
    if node_type in binary_types:
        return binary_types[node_type](
            **common,
            left=build(str(data["left"])),
            right=build(str(data["right"])),
        )
    if node_type == "translate":
        return Translate(
            **common,
            child=build(str(data["object"])),
            offset=_tuple3(data["offset"]),
        )
    if node_type == "rotate":
        return Rotate(
            **common,
            child=build(str(data["object"])),
            axis=str(data["axis"]),
            angle_degrees=float(data["angle_degrees"]),
        )
    if node_type == "scale":
        return Scale(
            **common,
            child=build(str(data["object"])),
            factor=float(data["factor"]),
        )
    section = build(str(data["section"])) if "section" in data else None
    if section is not None and not isinstance(section, PlacedSDF2D):
        raise ValueError(f"{node_type} section must be placed_sdf_2d")
    if node_type == "extrude":
        return Extrude(
            **common,
            section=section,
            height=float(data["height"]),
            center_offset=float(data.get("center_offset", 0.0)),
        )
    if node_type == "revolve":
        return Revolve(
            **common,
            section=section,
            axis=str(data.get("axis", "v")),
            axis_origin=(
                _tuple3(data["axis_origin"])
                if data.get("axis_origin") is not None
                else None
            ),
            axis_direction=(
                _tuple3(data["axis_direction"])
                if data.get("axis_direction") is not None
                else None
            ),
            radial_direction=(
                _tuple3(data["radial_direction"])
                if data.get("radial_direction") is not None
                else None
            ),
            angle_degrees=float(data.get("angle_degrees", 360.0)),
        )
    if node_type == "polyline_tube":
        return PolylineTube(
            **common,
            points=_points3(data["points"]),
            radius=float(data["radius"]),
            inner_radius=float(data.get("inner_radius", 0.0)),
            caps=str(data.get("caps", "round")),
        )
    if node_type == "quadratic_bezier_tube":
        return QuadraticBezierTube(
            **common,
            points=_points3(data["points"]),
            radius=float(data["radius"]),
            inner_radius=float(data.get("inner_radius", 0.0)),
            caps=str(data.get("caps", "round")),
        )
    raise ValueError(f"unknown SDF node type: {node_type}")


def _ghost_to_record(node: SDFNode) -> dict[str, Any]:
    """Self-contained record of a cut ghost (boundary_region_v2 §5).

    Ghosts are always leaf shapes (the cutter draws one shape at a time), so
    the record never needs child references."""
    names = _SceneNames({node.name or "ghost": node}, {})
    try:
        record = _node_to_record(node, names)
    except KeyError as error:
        raise TypeError(
            "boundary cut ghosts must be self-contained leaf shapes"
        ) from error
    record["name"] = node.name or "ghost"
    return record


def _no_ghost_references(name: str) -> SDFNode:
    raise ValueError(f"boundary cut ghosts cannot reference scene objects: {name}")


def _ghost_from_record(data: dict[str, Any]) -> SDFNode:
    name = str(data.get("name", "ghost"))
    return _node_from_record(name, data, _no_ghost_references, 0)


def _boundary_region_to_record(
    region: BoundaryRegion,
    names: _SceneNames,
) -> dict[str, Any]:
    owner = next(
        node
        for node in names.nodes_by_key.values()
        if node.object_id == region.owner_object_id
    )
    data: dict[str, Any] = {"owner": names.node_keys[id(owner)]}
    key = names.region_keys[id(region)]
    if region.name != key:
        data["name"] = region.name
    if region.patch_id is not None:
        data["patch"] = region.patch_id
    if region.patch_type is not None:
        data["patch_type"] = region.patch_type
    if region.outside_direction is not None:
        data["outside_direction"] = region.outside_direction
    if region.tag is not None:
        data["tag"] = region.tag
    cuts: list[dict[str, Any]] = []
    # Legacy volume selectors are inlined at save time so the file is always
    # the new self-contained format; interval selectors (2D curve params)
    # remain in the legacy fields until 2D parity (v2 §9).
    if region.selector_id is not None and region.selector_start is None:
        selector_key = _selector_name(region.selector_id, names)
        cuts.append(
            {
                "side": region.selector_side,
                "ghost": _ghost_to_record(names.nodes_by_key[selector_key]),
            }
        )
    elif region.selector_id is not None:
        data["selector"] = _selector_name(region.selector_id, names)
        if region.selector_type is not None:
            data["selector_type"] = region.selector_type
        if region.selector_side != "inside":
            data["selector_side"] = region.selector_side
        data["selector_start"] = region.selector_start
        data["selector_end"] = region.selector_end
    cuts.extend(
        {"side": cut.side, "ghost": _ghost_to_record(cut.ghost)}
        for cut in region.cuts
    )
    if cuts:
        data["cuts"] = cuts
    return data


def _boundary_region_from_record(
    key: str,
    data: dict[str, Any],
    build: Any,
    object_id: int,
) -> BoundaryRegion:
    owner = build(str(data["owner"]))
    cuts: list[BoundaryCut] = []
    selector_id = None
    is_interval = data.get("selector_start") is not None
    if data.get("selector") is not None:
        selector = build(str(data["selector"]))
        if is_interval:
            # 2D interval selectors stay legacy until 2D parity (v2 §9).
            selector_id = f"selector:{selector.object_id}"
        else:
            # One-way migration: a legacy volume selector becomes the first
            # entry of the cut chain; the hidden node is dropped afterwards.
            knife = deepcopy(selector)
            knife.object_id = 0
            cuts.append(
                BoundaryCut(str(data.get("selector_side") or "inside"), knife)
            )
    raw_cuts = data.get("cuts", [])
    if not isinstance(raw_cuts, list):
        raise ValueError(f"boundary region '{key}' cuts must be a list")
    for raw_cut in raw_cuts:
        if not isinstance(raw_cut, dict) or not isinstance(raw_cut.get("ghost"), dict):
            raise ValueError(
                f"boundary region '{key}' cuts must be objects with a ghost record"
            )
        cuts.append(
            BoundaryCut(str(raw_cut.get("side", "inside")), _ghost_from_record(raw_cut["ghost"]))
        )
    return BoundaryRegion(
        name=_display_name(key, data),
        object_id=object_id,
        owner_object_id=owner.object_id,
        outside_direction=(
            int(data["outside_direction"])
            if data.get("outside_direction") is not None
            else None
        ),
        patch_id=(
            str(data["patch"])
            if data.get("patch") is not None
            else None
        ),
        patch_type=(
            str(data["patch_type"])
            if data.get("patch_type") is not None
            else None
        ),
        selector_id=selector_id,
        selector_type=(
            str(data["selector_type"])
            if data.get("selector_type") is not None and is_interval
            else None
        ),
        selector_side=str(data.get("selector_side") or "inside"),
        selector_start=(
            float(data["selector_start"])
            if data.get("selector_start") is not None
            else None
        ),
        selector_end=(
            float(data["selector_end"])
            if data.get("selector_end") is not None
            else None
        ),
        cuts=tuple(cuts),
        tag=(str(data["tag"]) if data.get("tag") is not None else None),
    )


def _selector_name(selector_id: str, names: _SceneNames) -> str:
    prefix = "selector:"
    if not selector_id.startswith(prefix):
        return selector_id
    object_id = int(selector_id[len(prefix):])
    for key, node in names.nodes_by_key.items():
        if node.object_id == object_id:
            return key
    raise ValueError(f"unknown selector object id: {object_id}")


def _profile_to_dict(profile: Profile2D) -> dict[str, Any]:
    data: dict[str, Any] = {"type": _profile_type(profile)}
    for key, value in vars(profile).items():
        if isinstance(value, Profile2D):
            data[key] = _profile_to_dict(value)
        elif isinstance(value, tuple):
            if key == "half_size":
                data["size"] = _doubled(value)
            else:
                data[key] = _tuple_to_list(value)
        else:
            data[key] = value
    return data


def _profile_type(profile: Profile2D) -> str:
    names = {
        CircleProfile: "circle",
        RectangleProfile: "rectangle",
        SquareProfile: "square",
        RoundedRectangleProfile: "rounded_rectangle",
        EllipseProfile: "ellipse",
        RegularPolygonProfile: "regular_polygon",
        QuadraticBezierCurveProfile: "quadratic_bezier_curve",
        QuadraticBezierSurfaceProfile: "quadratic_bezier_surface",
        PolylineProfile: "polyline",
        PolygonProfile: "polygon",
        OffsetProfile: "offset",
        DistanceOffsetProfile: "distance_offset",
        BinaryProfile: "binary",
    }
    return names[type(profile)]


def _profile_from_dict(data: dict[str, Any]) -> Profile2D:
    profile_types = {
        "circle": CircleProfile,
        "rectangle": RectangleProfile,
        "square": SquareProfile,
        "rounded_rectangle": RoundedRectangleProfile,
        "ellipse": EllipseProfile,
        "regular_polygon": RegularPolygonProfile,
        "quadratic_bezier_curve": QuadraticBezierCurveProfile,
        "quadratic_bezier_surface": QuadraticBezierSurfaceProfile,
        "polyline": PolylineProfile,
        "polygon": PolygonProfile,
        "offset": OffsetProfile,
        "distance_offset": DistanceOffsetProfile,
        "binary": BinaryProfile,
    }
    profile_type = str(data["type"])
    constructor = profile_types.get(profile_type)
    if constructor is None:
        raise ValueError(f"unknown 2D profile type: {profile_type}")
    kwargs = {key: value for key, value in data.items() if key != "type"}
    if "size" in kwargs and "half_size" not in kwargs:
        kwargs["half_size"] = tuple(float(value) * 0.5 for value in kwargs.pop("size"))
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
    data: dict[str, Any] = {"type": _profile_1d_type(profile)}
    for key, value in vars(profile).items():
        if isinstance(value, Profile1D):
            data[key] = _profile_1d_to_dict(value)
        elif key == "half_length":
            data["length"] = float(value) * 2.0
        else:
            data[key] = value
    return data


def _profile_1d_type(profile: Profile1D) -> str:
    names = {
        SegmentProfile: "segment",
        OffsetProfile1D: "offset",
        BinaryProfile1D: "binary",
    }
    return names[type(profile)]


def _profile_1d_from_dict(data: dict[str, Any]) -> Profile1D:
    profile_types = {
        "segment": SegmentProfile,
        "offset": OffsetProfile1D,
        "binary": BinaryProfile1D,
    }
    profile_type = str(data["type"])
    constructor = profile_types.get(profile_type)
    if constructor is None:
        raise ValueError(f"unknown 1D profile type: {profile_type}")
    kwargs = {key: value for key, value in data.items() if key != "type"}
    if "length" in kwargs and "half_length" not in kwargs:
        kwargs["half_length"] = float(kwargs.pop("length")) * 0.5
    for key in ("left", "right", "child"):
        if key in kwargs:
            kwargs[key] = _profile_1d_from_dict(kwargs[key])
    return constructor(**kwargs)


def _write_axes(data: dict[str, Any], node: Any) -> None:
    if (
        tuple(node.axis_u) == DEFAULT_AXIS_U
        and tuple(node.axis_v) == DEFAULT_AXIS_V
        and tuple(node.axis_w) == DEFAULT_AXIS_W
    ):
        return
    data["axes"] = {
        "u": list(node.axis_u),
        "v": list(node.axis_v),
        "w": list(node.axis_w),
    }


def _read_axes(data: dict[str, Any]) -> dict[str, tuple[float, float, float]]:
    axes = data.get("axes")
    if isinstance(axes, dict):
        return {
            "axis_u": _tuple3(axes.get("u", DEFAULT_AXIS_U)),
            "axis_v": _tuple3(axes.get("v", DEFAULT_AXIS_V)),
            "axis_w": _tuple3(axes.get("w", DEFAULT_AXIS_W)),
        }
    return {
        "axis_u": _tuple3(data.get("axis_u", DEFAULT_AXIS_U)),
        "axis_v": _tuple3(data.get("axis_v", DEFAULT_AXIS_V)),
        "axis_w": _tuple3(data.get("axis_w", DEFAULT_AXIS_W)),
    }


def _half_size(data: dict[str, Any]) -> tuple[float, float, float]:
    if "size" in data:
        return tuple(float(value) * 0.5 for value in data["size"])
    return _tuple3(data["half_size"])


def _half_height(data: dict[str, Any]) -> float:
    if "height" in data:
        return float(data["height"]) * 0.5
    return float(data["half_height"])


def _doubled(values: tuple[float, ...]) -> list[float]:
    return [float(value) * 2.0 for value in values]


def _tuple_to_list(value: tuple[Any, ...]) -> list[Any]:
    return [
        _tuple_to_list(item)
        if isinstance(item, tuple)
        else item
        for item in value
    ]


def _tuple3(value: Any) -> tuple[float, float, float]:
    items = tuple(float(item) for item in value)
    if len(items) != 3:
        raise ValueError("expected a 3D vector")
    return items


def _points3(value: Any) -> tuple[tuple[float, float, float], ...]:
    return tuple(_tuple3(point) for point in value)
