from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import numpy as np

from core.boundary import BoundaryRegion
from core.boundary_patches import surface_selector_volume
from core.boundary_region import (
    boundary_region_mask,
    cut_volume,
    find_node_by_object_id,
)
from core.model import Model, compile_model, model_from_document
from core.serialization import load_scene
from core.sdf.base import BoundingBox3D, FloatArray, SDFNode
from core.sdf.roles import DomainKind


SDFCallable = Callable[[np.ndarray], np.ndarray]
PointsPredicate = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class MeshableBoundaryRegion:
    """One boundary region, callable from mesher scripts (v2 §6).

    * ``contains`` -- exact membership of (N, 3) world points; the same
      classifier the viewport uses, so what is highlighted is what you get.
    * ``owner_sdf`` -- the exact field of the region's generating surface
      (y+ layers, grading, refinement bands).
    * ``selector_sdf`` -- combined signed field of the cut chain (negative
      inside every kept knife-half); ``None`` for whole-surface regions.
    * ``tag`` -- opaque physics label; the kernel never interprets it.
    """

    name: str
    tag: str | None
    owner_object_id: int
    contains: PointsPredicate
    owner_sdf: SDFCallable
    selector_sdf: SDFCallable | None = None


@dataclass(frozen=True)
class MeshableBoundaryRegions:
    """Name-keyed collection of a domain's boundary regions."""

    _items: tuple[MeshableBoundaryRegion, ...] = ()

    def __iter__(self) -> Iterator[MeshableBoundaryRegion]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, key: int) -> MeshableBoundaryRegion: ...

    @overload
    def __getitem__(self, key: slice) -> tuple[MeshableBoundaryRegion, ...]: ...

    @overload
    def __getitem__(self, key: str) -> MeshableBoundaryRegion: ...

    def __getitem__(
        self,
        key: int | slice | str,
    ) -> MeshableBoundaryRegion | tuple[MeshableBoundaryRegion, ...]:
        if isinstance(key, (int, slice)):
            return self._items[key]
        for region in self._items:
            if region.name == key:
                return region
        available = ", ".join(self.keys())
        raise KeyError(f"unknown boundary region {key!r}; available: {available}")

    def keys(self) -> tuple[str, ...]:
        return tuple(region.name for region in self._items)

    def as_tuple(self) -> tuple[MeshableBoundaryRegion, ...]:
        return self._items


@dataclass(frozen=True)
class MeshableDomain:
    name: str
    kind: tuple[str, ...]
    dimension: int
    bounds: BoundingBox3D
    domain_sdf: SDFCallable
    boundary_tags: tuple[tuple[str, SDFCallable], ...] = ()
    boundary_regions: MeshableBoundaryRegions = MeshableBoundaryRegions()


@dataclass(frozen=True)
class MeshableDomains:
    """Small script-facing collection for meshable domain lookup.

    Integer indexing is kept for Python sequence compatibility, but mesher
    scripts should prefer string lookup by domain name or unique domain kind.
    """

    _items: tuple[MeshableDomain, ...]

    def __iter__(self) -> Iterator[MeshableDomain]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, key: int) -> MeshableDomain: ...

    @overload
    def __getitem__(self, key: slice) -> tuple[MeshableDomain, ...]: ...

    @overload
    def __getitem__(self, key: str) -> MeshableDomain: ...

    def __getitem__(
        self,
        key: int | slice | str,
    ) -> MeshableDomain | tuple[MeshableDomain, ...]:
        if isinstance(key, (int, slice)):
            return self._items[key]
        for domain in self._items:
            if domain.name == key:
                return domain
        matches = tuple(domain for domain in self._items if key in domain.kind)
        if len(matches) == 1:
            return matches[0]
        if matches:
            names = ", ".join(domain.name for domain in matches)
            raise KeyError(f"domain kind {key!r} is ambiguous: {names}")
        available = ", ".join(self.keys())
        raise KeyError(f"unknown meshable domain {key!r}; available: {available}")

    def keys(self) -> tuple[str, ...]:
        names = [domain.name for domain in self._items]
        unique_kinds = sorted(
            kind
            for kind in {kind for domain in self._items for kind in domain.kind}
            if sum(kind in domain.kind for domain in self._items) == 1
        )
        return tuple([*names, *unique_kinds])

    def by_kind(self, kind: str) -> tuple[MeshableDomain, ...]:
        return tuple(domain for domain in self._items if kind in domain.kind)

    def as_tuple(self) -> tuple[MeshableDomain, ...]:
        return self._items


def _ensure_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("SDF query points must have shape (N, 3)")
    return array


def sdf_callable(node: SDFNode) -> SDFCallable:
    """Wrap an SDF node as a batch callable over ``(N, 3)`` world points."""

    def evaluate(points: np.ndarray) -> np.ndarray:
        array = _ensure_points(points)
        values: FloatArray = node.to_numpy(array[:, 0], array[:, 1], array[:, 2])
        return np.asarray(values, dtype=np.float64).reshape(-1)

    return evaluate


def _boundary_region_callable(
    root: SDFNode,
    region: BoundaryRegion,
    selectors: dict[int, SDFNode],
) -> SDFCallable | None:
    """Return an SDF-backed query for selector-defined boundary regions.

    Directional/owner-only boundary metadata remains valid scene metadata, but
    it is not an SDF query by itself. The public mesher contract exposes only
    SDF-backed tags.
    """

    prefix = "selector:"
    selector_id = (
        int(region.selector_id[len(prefix):])
        if region.selector_id is not None and region.selector_id.startswith(prefix)
        else None
    )
    if selector_id is None:
        return None
    selector = selectors.get(selector_id)
    if selector is None:
        return None
    try:
        volume = surface_selector_volume(root, selector)
    except (TypeError, ValueError, NotImplementedError):
        return None
    query = sdf_callable(volume)
    if region.selector_side != "outside":
        return query

    def outside(points: np.ndarray) -> np.ndarray:
        return -query(points)

    return outside


def _cut_chain_field(root: SDFNode, region: BoundaryRegion) -> SDFCallable | None:
    """Combined signed field of a region's cut chain: negative exactly where
    every knife-half is satisfied (max over per-cut signed fields)."""
    if not region.cuts:
        return None
    parts: list[tuple[float, SDFCallable]] = []
    for cut in region.cuts:
        volume = cut_volume(root, cut)
        parts.append((1.0 if cut.side == "inside" else -1.0, sdf_callable(volume)))

    def field(points: np.ndarray) -> np.ndarray:
        return np.maximum.reduce([sign * query(points) for sign, query in parts])

    return field


def _region_contains(root: SDFNode, region: BoundaryRegion) -> PointsPredicate:
    def contains(points: np.ndarray) -> np.ndarray:
        return boundary_region_mask(root, region, _ensure_points(points))

    return contains


def _fluid_boundary_regions(
    document: object,
    root: SDFNode,
) -> MeshableBoundaryRegions:
    fluid = getattr(document, "fluid_domain", None)
    if fluid is None or fluid.root is not root:
        return MeshableBoundaryRegions()
    selector_by_id = {
        selector.object_id: selector
        for selector in fluid.selector_objects
        if selector.object_id > 0
    }
    regions: list[MeshableBoundaryRegion] = []
    for tag in fluid.tag_objects:
        if not isinstance(tag, BoundaryRegion):
            continue
        owner = find_node_by_object_id(root, tag.owner_object_id)
        if owner is None:
            continue
        try:
            selector_field = _cut_chain_field(root, tag)
        except (TypeError, ValueError, NotImplementedError):
            selector_field = None
        if selector_field is None:
            selector_field = _boundary_region_callable(root, tag, selector_by_id)
        regions.append(
            MeshableBoundaryRegion(
                name=tag.name,
                tag=tag.tag,
                owner_object_id=tag.owner_object_id,
                contains=_region_contains(root, tag),
                owner_sdf=sdf_callable(owner),
                selector_sdf=selector_field,
            )
        )
    return MeshableBoundaryRegions(tuple(regions))


def _fluid_boundary_tags(document: object, root: SDFNode) -> tuple[tuple[str, SDFCallable], ...]:
    """Back-compat SDF-backed tag list: knife fields for selector/cut-chain
    regions, raw fields for placed 1D/2D tag objects."""
    fluid = getattr(document, "fluid_domain", None)
    if fluid is None or fluid.root is not root:
        return ()
    selector_by_id = {
        selector.object_id: selector
        for selector in fluid.selector_objects
        if selector.object_id > 0
    }
    tags: list[tuple[str, SDFCallable]] = []
    for tag in fluid.tag_objects:
        if isinstance(tag, BoundaryRegion):
            try:
                query = _cut_chain_field(root, tag)
            except (TypeError, ValueError, NotImplementedError):
                query = None
            if query is None:
                query = _boundary_region_callable(root, tag, selector_by_id)
            if query is not None:
                tags.append((tag.name, query))
        elif isinstance(tag, SDFNode):
            tags.append((tag.name, sdf_callable(tag)))
    return tuple(tags)


def meshable_domains_from_model(
    model: Model,
    *,
    validate: bool = True,
    resolution: int = 32,
    boundary_tags: dict[str, tuple[tuple[str, SDFCallable], ...]] | None = None,
    boundary_regions: dict[str, MeshableBoundaryRegions] | None = None,
) -> MeshableDomains:
    """Expose an exact-SDF ``Model`` through the public meshing API.

    ``compile_model`` is the mesh-time hard gate: invalid role wiring,
    generator precondition failures, or overlapping Domains are refused before
    a mesher script receives field callables.
    """

    if validate:
        compile_model(model, resolution=resolution)
    boundary_tags = boundary_tags or {}
    boundary_regions = boundary_regions or {}
    return MeshableDomains(
        tuple(
            MeshableDomain(
                name=domain.name,
                kind=(domain.kind.value,),
                dimension=domain.region.dimension,
                bounds=domain.region.bounding_box(),
                domain_sdf=sdf_callable(domain.region),
                boundary_tags=boundary_tags.get(domain.name, ()),
                boundary_regions=boundary_regions.get(
                    domain.name, MeshableBoundaryRegions()
                ),
            )
            for domain in model.domains
        )
    )


def load_meshable_domains(scene_path: str | Path) -> MeshableDomains:
    """Load meshable domains from a saved casoCAD ``scene.json``.

    Declared Fluid/Solid Domains feed the same ``MeshableDomain`` contract.
    Fluid boundary regions/tags are exposed when the declared Domain is also
    the legacy ``fluid_domain`` root.
    """

    document = load_scene(scene_path)
    model = model_from_document(document)
    fluid_domains = [
        domain for domain in model.domains if domain.kind is DomainKind.FLUID
    ]
    tags = {
        domain.name: _fluid_boundary_tags(document, domain.region)
        for domain in fluid_domains
    }
    regions = {
        domain.name: _fluid_boundary_regions(document, domain.region)
        for domain in fluid_domains
    }
    return meshable_domains_from_model(
        model,
        boundary_tags=tags,
        boundary_regions=regions,
    )


__all__ = [
    "MeshableBoundaryRegion",
    "MeshableBoundaryRegions",
    "MeshableDomain",
    "MeshableDomains",
    "PointsPredicate",
    "SDFCallable",
    "load_meshable_domains",
    "meshable_domains_from_model",
    "sdf_callable",
]
