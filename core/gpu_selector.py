from __future__ import annotations

"""Build Layer 2 ``region_selector`` IR nodes from boundary selectors.

Parity-by-construction with the CPU Boundary Planar Cutter / Surface Cutter
(design §5.2, §11): the selector volume is produced by the *same*
``surface_selector_volume`` the CPU path uses, then serialized as an ordinary
SDF subtree. The GPU re-tags a boundary region iff a point is inside (or
outside) that identical volume, so the GPU inside-test can only agree with
``surface_selector_values``.
"""

from dataclasses import dataclass, replace

from .boundary import BoundaryRegion
from .boundary_patches import PATCH_TOLERANCE, surface_selector_volume
from .render_ir import RenderIR, RenderIRNode, build_render_ir
from .sdf import SDFNode, SDFTree


@dataclass(frozen=True)
class RegionSelectorSpec:
    """One Layer 2 split: re-tag ``base_owner_id``'s boundary with ``region_id``.

    ``selector`` is any selector geometry ``surface_selector_volume`` accepts
    (extruded 2D profile, offset polyline band, oriented segment slab, or a full
    3D SDF). ``side`` is ``"inside"`` or ``"outside"`` with the same meaning as
    the CPU cutter. ``scope_region`` optionally restricts the volume to a patch.
    """

    base_owner_id: int
    region_id: int
    selector: SDFNode
    side: str = "inside"
    scope_region: BoundaryRegion | None = None
    tolerance: float = PATCH_TOLERANCE


def selector_volume_ir(
    root: SDFNode,
    selector: SDFNode,
    *,
    scope_region: BoundaryRegion | None = None,
) -> RenderIR | None:
    """Serialize a selector volume alone (used to assert volume parity)."""

    volume = surface_selector_volume(root, selector, scope_region=scope_region)
    if volume is None:
        return None
    return build_render_ir(SDFTree(root=volume))


def _signed_tol(side: str, tolerance: float) -> float:
    # >= 0 => inside (sel_d <= tol); < 0 => outside (sel_d > -tol). Keep this
    # in parity with core.boundary_selection.surface_split_selector_mask().
    if side == "outside":
        return -abs(tolerance)
    return abs(tolerance)


def _merge_subtree(
    nodes: list[RenderIRNode],
    subtree: RenderIR,
) -> int:
    """Append a sub-IR's nodes to ``nodes``, re-indexing children; return its root."""

    offset = len(nodes)
    for node in subtree.nodes:
        nodes.append(
            replace(node, children=tuple(child + offset for child in node.children))
        )
    return subtree.root_indices[0] + offset


def attach_region_selectors(
    scene_ir: RenderIR,
    root: SDFNode,
    specs: tuple[RegionSelectorSpec, ...],
) -> RenderIR:
    """Return ``scene_ir`` augmented with ``region_selector`` nodes.

    The selector-volume subtrees are merged into the node list (unreachable from
    the scene root, so the main bytecode never walks them — they are evaluated
    only by ``applyRegionSelector`` via ``evalSubtreeSDF``). One
    ``region_selector`` node per spec carries the owner scope, the signed
    tolerance, and the region id (in ``flags``).
    """

    nodes = list(scene_ir.nodes)
    for spec in specs:
        volume_ir = selector_volume_ir(
            root, spec.selector, scope_region=spec.scope_region
        )
        if volume_ir is None:
            continue
        volume_root = _merge_subtree(nodes, volume_ir)
        nodes.append(
            RenderIRNode(
                kind="region_selector",
                object_id=spec.base_owner_id,
                dimension=3,
                children=(volume_root,),
                params=(_signed_tol(spec.side, spec.tolerance),),
                flags=spec.region_id,
            )
        )

    return RenderIR(
        nodes=tuple(nodes),
        root_indices=scene_ir.root_indices,
        component_indices=scene_ir.component_indices,
        aliases=scene_ir.aliases,
        material_refs=scene_ir.material_refs,
        component_refs=scene_ir.component_refs,
        unsupported_kinds=scene_ir.unsupported_kinds,
    )


__all__ = [
    "RegionSelectorSpec",
    "selector_volume_ir",
    "attach_region_selectors",
]
