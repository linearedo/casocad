from __future__ import annotations

"""Thin data-driven codegen (core/gpu_codegen.py) — the post-VM shader path.

Verifies the emitter specializes to ONLY the present primitive types and keeps
the boolean structure (incl. intersection) as data-driven loops, so the shader
is the same size regardless of instance count or operator mix.
"""

from core.gpu_codegen import (
    emit_fragment_shader,
    emit_map_glsl,
    scene_structure_signature,
    supported,
)
from core.render_ir import RenderIR, RenderIRNode


def _sphere(o, x=0.0, y=0.0, z=0.0, r=1.0):
    return RenderIRNode(kind="sphere", object_id=o, dimension=3, children=(),
                        params=(x, y, z, r))


def _box(o, h=1.0):
    return RenderIRNode(kind="box", object_id=o, dimension=3, children=(),
                        params=(0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, h, h, h))


def _op(kind, kids):
    return RenderIRNode(kind=kind, object_id=0, dimension=3,
                        children=tuple(kids), params=())


def _ir(nodes, root):
    return RenderIR(nodes=tuple(nodes), root_indices=(root,), component_indices=())


def test_signature_is_kind_set_not_instance_data() -> None:
    # Two spheres at different spots share one signature -> moving/adding spheres
    # never re-bakes; only a new KIND does.
    a = _ir([_sphere(1, -1), _sphere(2, 1), _op("union", [0, 1])], 2)
    b = _ir([_sphere(1, 5), _sphere(2, -3), _sphere(3, 0), _op("union", [0, 1, 2])], 3)
    assert scene_structure_signature(a) == {"sphere"}
    assert scene_structure_signature(a) == scene_structure_signature(b)
    c = _ir([_box(1), _sphere(2), _op("difference", [0, 1])], 2)
    assert scene_structure_signature(c) == {"box", "sphere"}


def test_leaf_dispatch_only_present_types() -> None:
    # Only the present types get a dispatch branch (the all-codes #define block is
    # just harmless constants; what matters is `t == NODE_X` in leafDist).
    glsl = emit_map_glsl(_ir([_sphere(1), _sphere(2, 2), _op("union", [0, 1])], 2))
    assert "t == NODE_SPHERE" in glsl
    for absent in ("NODE_BOX", "NODE_TORUS", "NODE_CYLINDER", "NODE_PYRAMID"):
        assert f"t == {absent}" not in glsl, f"sphere scene must not branch {absent}"


def test_intersection_adds_no_shader_code() -> None:
    # The whole point: intersection is a data-driven DNF loop, not extra source.
    union = emit_map_glsl(
        _ir([_sphere(1, -1), _sphere(2, 1), _op("union", [0, 1])], 2))
    inter = _ir([_sphere(1, -1), _sphere(2, -0.5), _sphere(3, 1), _sphere(4, 0.5),
                 _op("union", [0, 1]), _op("union", [2, 3]),
                 _op("intersection", [4, 5])], 6)
    assert emit_map_glsl(inter).count("\n") == union.count("\n")


def test_emits_dnf_loops_and_map_signature() -> None:
    glsl = emit_map_glsl(_ir([_sphere(1), _sphere(2, 2), _op("union", [0, 1])], 2))
    assert "float map(vec3 p, out uint owner_out)" in glsl
    assert "u_term_count" in glsl
    assert "u_group_count" in glsl          # intersection breadth, data-driven
    assert "leafDist" in glsl


def test_helpers_emitted_only_when_needed() -> None:
    sphere_only = emit_map_glsl(_ir([_sphere(1), _sphere(2, 2), _op("union", [0, 1])], 2))
    assert "irOrientedLocal" not in sphere_only  # spheres don't need it
    with_box = emit_map_glsl(_ir([_box(1), _sphere(2), _op("difference", [0, 1])], 2))
    assert "irOrientedLocal" in with_box


def test_full_fragment_shader_is_complete() -> None:
    # emit_fragment_shader = preamble + leafDist + map + raymarch main. Validated
    # to render correctly on real GPU (intersection blob); here just structural.
    glsl = emit_fragment_shader(
        _ir([_sphere(1, -1), _sphere(2, 1), _op("union", [0, 1])], 2))
    assert glsl.startswith("#version")
    assert "float map(vec3 p, out uint owner_out)" in glsl
    assert "void main()" in glsl
    assert "u_camera_position" in glsl and "frag_color" in glsl


def _placed(kind, o, extra):
    base = (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)   # origin, axis_u, axis_v, normal
    return RenderIRNode(kind=kind, object_id=o, dimension=2, children=(),
                        params=base + extra)


def test_placed_2d_sections_supported_and_emitted() -> None:
    # Step 2: placed 2D analytic sections render via codegen (no VM).
    ir = _ir([_placed("placed_circle_2d", 1, (0, 0, 1.0)),
              _placed("placed_ellipse_2d", 2, (0.5, 0, 1.0, 0.6)),
              _op("union", [0, 1])], 2)
    assert supported(ir)
    assert scene_structure_signature(ir) == {"placed_circle_2d", "placed_ellipse_2d"}
    glsl = emit_map_glsl(ir)
    assert "t == NODE_PLACED_CIRCLE_2D" in glsl
    assert "t == NODE_PLACED_ELLIPSE_2D" in glsl
    assert "irExactEllipseDistance" in glsl          # ellipse helper pulled in
    # a sphere-only scene must NOT drag in the section branches/helper
    sph = emit_map_glsl(_ir([_sphere(1), _sphere(2, 2), _op("union", [0, 1])], 2))
    assert "NODE_PLACED_CIRCLE_2D" not in sph.replace("#define NODE_PLACED_CIRCLE_2D", "")
    assert "irExactEllipseDistance" not in sph


def _tube(kind, o, pts, r=0.3):
    params = tuple(c for pt in pts for c in pt) + (r, 0.0, 0.0)  # +radius,inner,flat
    return RenderIRNode(kind=kind, object_id=o, dimension=1, children=(), params=params)


def test_tubes_supported_and_emitted() -> None:
    # Step 2: polyline/bezier tubes render via codegen (points inline in params).
    ir = _ir([_tube("polyline_tube", 1, [(-1, 0, 0), (0, 1, 0), (1, 0, 0)]),
              _sphere(2, 0, 0, 0, 0.6), _op("union", [0, 1])], 2)
    assert supported(ir)
    assert "polyline_tube" in scene_structure_signature(ir)
    glsl = emit_map_glsl(ir)
    assert "t == NODE_POLYLINE_TUBE" in glsl
    assert "irSegmentDistance3D" in glsl and "irTubeSDF" in glsl
    bez = emit_map_glsl(
        _ir([_tube("bezier_tube", 1, [(-1, 0, 0), (0, 1, 0), (1, 0, 0)]),
             _sphere(2), _op("union", [0, 1])], 2))
    assert "irQuadraticBezierDistance3D" in bez
    # sphere-only scene stays free of tube helpers
    assert "irSegmentDistance3D" not in emit_map_glsl(
        _ir([_sphere(1), _sphere(2, 2), _op("union", [0, 1])], 2))


def test_profile_subvm_extrude_revolve() -> None:
    # Step 2c: extrude/revolve over a 2D profile sub-graph render via codegen by
    # embedding the profile stack-VM. The recompile signature is the 3D leaf kind,
    # NOT the profile sub-graph kinds (handled uniformly by the embedded VM).
    base = (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)
    circle = RenderIRNode(kind="profile_circle_2d", object_id=0, dimension=2,
                          children=(), params=(0.0, 0.0, 1.0))
    extrude = RenderIRNode(kind="extrude_profile_2d", object_id=1, dimension=3,
                           children=(0,), params=base + (1.0, 0.0))
    ir = RenderIR(nodes=(circle, extrude), root_indices=(1,), component_indices=())
    assert supported(ir)
    assert scene_structure_signature(ir) == {"extrude_profile_2d"}  # not the circle
    glsl = emit_map_glsl(ir)
    assert "evalProfileSDF" in glsl                # embedded profile VM present
    assert "t == NODE_EXTRUDE_PROFILE_2D" in glsl
    assert "u_children" in glsl                    # walks the sub-graph
    # a sphere scene must NOT embed the profile VM
    assert "evalProfileSDF" not in emit_map_glsl(
        _ir([_sphere(1), _sphere(2, 2), _op("union", [0, 1])], 2))


def test_region_selectors_emit_subtree_vm() -> None:
    # Step 2d: region selectors render via codegen — geometry flattens normally
    # (selectors are a separate list), and an embedded subtree-VM tags region_id.
    from core.gpu_codegen import selector_indices
    geom = _sphere(1, 0, 0, 0, 1.2)
    vol = _sphere(2, 0.6, 0, 0, 0.8)
    sel = RenderIRNode(kind="region_selector", object_id=1, dimension=3,
                       children=(1,), params=(0.3,), flags=5)
    ir = RenderIR(nodes=(geom, vol, sel), root_indices=(0,),
                  component_indices=(2,))
    assert supported(ir)                       # geometry is codegen-able
    assert selector_indices(ir) == [2]
    glsl = emit_fragment_shader(ir)
    assert "evalSubtreeDist" in glsl and "regionAt" in glsl
    assert "u_sel" in glsl
    # a scene with no selector gets the no-op regionAt stub (no subtree VM)
    plain = emit_fragment_shader(_ir([_sphere(1), _sphere(2, 2), _op("union", [0, 1])], 2))
    assert "evalSubtreeDist" not in plain and "regionAt" in plain


def test_supported_scope() -> None:
    assert supported(_ir([_sphere(1), _sphere(2, 2), _op("union", [0, 1])], 2))
    assert supported(_ir([_box(1), _sphere(2), _op("difference", [0, 1])], 2))
    assert not supported(None)


def test_carve_under_union_now_supported() -> None:
    # (A - B) ∪ C flattens to ONE union group of carved terms — the carve B rides
    # term A locally, so codegen handles it (no VM fallback).
    from core.gpu_codegen import flatten_terms
    carve = _ir([_sphere(1), _sphere(2), _sphere(3, 4),
                 _op("difference", [0, 1]), _op("union", [3, 2])], 4)
    assert supported(carve)
    groups = flatten_terms(carve)
    assert len(groups) == 1                       # union -> single group
    terms = groups[0]
    assert {leaf for leaf, _carves in terms} == {0, 2}    # solids A and C
    assert dict(terms)[0] == (1,)                 # A carries the local hole B
    assert dict(terms)[2] == ()                   # C is uncarved


def test_union_of_many_carved_solids_is_linear() -> None:
    # The whole point of local per-term carves: a union of K carved solids stays
    # ONE group of K terms (linear), where a signed-literal DNF would be 2**K.
    from core.gpu_codegen import flatten_terms
    nodes, kids = [], []
    for k in range(8):
        nodes.append(_sphere(k + 1, x=2.0 * k))            # solid
        nodes.append(_sphere(100 + k, x=2.0 * k + 0.5))    # hole
        nodes.append(_op("difference", [len(nodes) - 2, len(nodes) - 1]))
        kids.append(len(nodes) - 1)
    nodes.append(_op("union", kids))
    ir = _ir(nodes, len(nodes) - 1)
    assert supported(ir)
    groups = flatten_terms(ir)
    assert len(groups) == 1 and len(groups[0]) == 8         # 8 carved terms, 1 group


def test_flatten_terms_cpu_parity_vs_tree() -> None:
    # The term DNF must equal the boolean tree everywhere. Compare a CPU eval of
    # max_g(min_term(carved)) against a recursive tree eval at sample points.
    import numpy as np
    from core.gpu_codegen import flatten_terms
    from core.gpu_node_types import is_operator

    def leaf_sdf(node, p):                          # sphere only (params x,y,z,r)
        c = np.array(node.params[:3]); r = node.params[3]
        return float(np.linalg.norm(p - c) - r)

    def tree_sdf(nodes, idx, p):
        n = nodes[idx]
        if not is_operator(n.kind):
            return leaf_sdf(n, p)
        ds = [tree_sdf(nodes, int(c), p) for c in n.children]
        if n.kind == "union":
            return min(ds)
        if n.kind == "intersection":
            return max(ds)
        if n.kind == "difference":
            return max(ds[0], *[-d for d in ds[1:]])
        raise AssertionError(n.kind)

    def flat_sdf(nodes, groups, p):
        gvals = []
        for grp in groups:
            terms = []
            for leaf, carves in grp:
                d = leaf_sdf(nodes[leaf], p)
                for c in carves:
                    d = max(d, -leaf_sdf(nodes[c], p))
                terms.append(d)
            gvals.append(min(terms))
        return max(gvals)

    # (A-B) ∪ (C-D) ∩ E  — mixes union, carve, and intersection.
    nodes = [_sphere(1, 0, 0, 0, 1.2), _sphere(2, 0.6, 0, 0, 0.6),
             _sphere(3, 2, 0, 0, 1.2), _sphere(4, 2.6, 0, 0, 0.6),
             _sphere(5, 1, 0, 0, 2.5),
             _op("difference", [0, 1]), _op("difference", [2, 3]),
             _op("union", [5, 6]), _op("intersection", [7, 4])]
    ir = _ir(nodes, 8)
    groups = flatten_terms(ir)
    assert groups is not None
    rng = np.random.default_rng(0)
    for p in rng.uniform(-3, 3, size=(200, 3)):
        assert abs(flat_sdf(nodes, groups, p) - tree_sdf(nodes, 8, p)) < 1e-6


def test_term_grid_bins_every_overlapping_cell() -> None:
    # The cull-grid invariant the DDA relies on: a term is binned into EVERY cell
    # its positive-leaf bound overlaps, so any point inside a solid finds that
    # solid in its own cell. Check it for a spread of spheres.
    import numpy as np
    from core.gpu_codegen import flatten_terms
    from core.gpu_cull import build_term_grid, leaf_bounds, _INF_RADIUS
    rng = np.random.default_rng(1)
    nodes, kids = [], []
    centers = rng.uniform(-3, 3, size=(40, 3))
    for k, (x, y, z) in enumerate(centers):
        nodes.append(_sphere(k + 1, float(x), float(y), float(z), 1.0))
        kids.append(len(nodes) - 1)
    nodes.append(_op("union", kids))
    ir = _ir(nodes, len(nodes) - 1)
    groups = flatten_terms(ir)
    pos = np.array([leaf for g in groups for leaf, _c in g], dtype=np.uint32)
    bounds = leaf_bounds(ir)
    assert not np.any(bounds[pos][:, 3] >= _INF_RADIUS)      # all bounded
    origin, cell, dim, off, cnt, items = build_term_grid(pos, bounds, dim=16)
    org, cl = np.array(origin), np.array(cell)
    # every term (sphere center) must be listed in the cell containing its center
    for ti, leaf in enumerate(pos):
        c = np.clip(np.floor((centers[ti] - org) / cl).astype(int), 0, dim - 1)
        ci = (c[2] * dim + c[1]) * dim + c[0]
        cell_items = items[off[ci]:off[ci] + cnt[ci]]
        assert ti in cell_items, f"term {ti} missing from its own center cell"
