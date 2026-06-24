from __future__ import annotations

"""Thin data-driven codegen for the SDF viewport (the post-VM path).

Instead of one giant interpreter shader (a bytecode VM with a value stack and a
switch over every node type — big, slow to compile, link-limited), emit a SMALL
fragment shader specialized to ONE scene's structure:

* only the **primitive types actually present** get a leaf-distance branch
  (a sphere scene compiles only the sphere case),
* the boolean structure is the cull DNF — ``max_g(min(group_g)) `` carved by
  ``max(-sub)`` — emitted as plain ``for`` LOOPS over instance arrays, not unrolled.

Because the shader **loops over data** (positions/radii live in buffers) rather
than baking each primitive into source, it is the same size for 100 or 100000
primitives, so it compiles in ~tens of ms at any scale (measured) and never hits
the driver link limit. It only re-bakes when the scene's **structure** changes —
the set of primitive types, not the instance data — so moving objects is free.

Backend: target Vulkan (codegen compiles far faster there than OpenGL; the old
"Vulkan pathological" result was VM-specific). This module is backend-agnostic —
it just emits GLSL.

Data buffers (host fills these; positions etc. are world-space params, same
layout as ``core.gpu_scene.serialize_scene``):
  binding 0  Nodes   (GpuNode[])      — per-leaf type + param_offset + owner
  binding 1  Params  (float[])        — flat leaf params
  binding 4  Terms   (uvec4[])        — (leaf, group_id, carve_offset, carve_count)
  binding 5  Carves  (uint[])         — flat carve leaf indices (per-term slices)
"""

import os

from core.gpu_node_types import emit_glsl_defines
from core.render_ir import RenderIR

# ADAPTIVE group capacity. The shader holds a `float g[N]` accumulator array whose
# size N is baked PER SCENE (group_capacity below); the cull-grid DDA evaluates a
# cell's terms into these accumulators, so the size is just the scene's group count.
#
# Group count grows with: an intersection (groups ADD — a convex polytope of N
# planes -> N groups, LINEAR) or a union OF intersections (groups MULTIPLY ->
# exponential, rare). The linear case is covered by baking a bigger g[] bucket; the
# multiplicative case can't be arrayed away and bails past the ceiling. Sizes are
# BUCKETED so edits that nudge the group count (20 -> 21) stay in one variant (no
# re-bake); a bucket change re-bakes (~30ms) and the capacity is part of the bake
# key. See progress/codegen_full_migration.md.
_CG_GROUP_BUCKETS = (8, 16, 32, 64, 128, 256)
CG_GROUP_CEILING = _CG_GROUP_BUCKETS[-1]   # > this many groups -> scene bails


def group_capacity(render_ir: RenderIR | None) -> int | None:
    """Smallest baked ``g[]`` capacity (a bucket) that fits the scene's flattened
    group count, or None if the scene is unsupported or needs > CG_GROUP_CEILING
    groups. Drives BOTH the emitted array size and the bake-cache key — two scenes
    with the same leaf kinds but different group counts may bake different variants."""
    groups = flatten_terms(render_ir)
    if groups is None:
        return None
    n = len(groups)
    for bucket in _CG_GROUP_BUCKETS:
        if n <= bucket:
            return bucket
    return None

# Per-core-primitive leaf distance, given `base` (param offset) and `p`. This is
# the source of truth for the primitive SDFs — codegen emits ONLY the present types.
_LEAF_GLSL: dict[str, str] = {
    "sphere":
        "return length(p - irP3(base, 0u)) - irP(base, 3u);",
    "box": """vec3 q = abs(irOrientedLocal(p, irP3(base,0u), irP3(base,3u), irP3(base,6u), irP3(base,9u))) - irP3(base,12u);
        return length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);""",
    "cylinder": """vec3 lc = irOrientedLocal(p, irP3(base,0u), irP3(base,3u), irP3(base,6u), irP3(base,9u));
        vec2 d2 = abs(vec2(length(lc.xy), lc.z)) - vec2(irP(base,12u), irP(base,13u));
        return min(max(d2.x, d2.y), 0.0) + length(max(d2, vec2(0.0)));""",
    "cone": """vec3 lc = irOrientedLocal(p, irP3(base,0u), irP3(base,3u), irP3(base,6u), irP3(base,9u));
        float rr = irP(base,12u); float hh = irP(base,13u); float hgt = 2.0*hh;
        vec2 qq = vec2(rr, -hgt); vec2 ww = vec2(length(lc.xy), lc.z - hh);
        vec2 aa = ww - qq * clamp(dot(ww,qq)/dot(qq,qq), 0.0, 1.0);
        vec2 bb2 = ww - qq * vec2(clamp(ww.x/qq.x, 0.0, 1.0), 1.0);
        float dd = min(dot(aa,aa), dot(bb2,bb2));
        float ss = max(-(ww.x*qq.y - ww.y*qq.x), -(ww.y - qq.y));
        return sqrt(dd) * sign(ss);""",
    "capped_cone": """vec3 lc = irOrientedLocal(p, irP3(base,0u), irP3(base,3u), irP3(base,6u), irP3(base,9u));
        return irCappedConeSDF(lc, irP(base,14u), irP(base,12u), irP(base,13u));""",
    "box_frame": """vec3 lc = irOrientedLocal(p, irP3(base,0u), irP3(base,3u), irP3(base,6u), irP3(base,9u));
        return irBoxFrameSDF(lc, irP3(base,12u), irP(base,15u));""",
    "pyramid": """vec3 lc = irOrientedLocal(p, irP3(base,0u), irP3(base,3u), irP3(base,6u), irP3(base,9u));
        float bh = irP(base,12u); float hh = irP(base,13u); float sc = 2.0*bh;
        float hgt = (2.0*hh)/sc; vec3 qq = vec3(lc.x/sc, (lc.z+hh)/sc, lc.y/sc);
        return sc * irPyramidUnitSDF(qq, hgt);""",
    "torus": """vec3 lc = irOrientedLocal(p, irP3(base,0u), irP3(base,3u), irP3(base,6u), irP3(base,9u));
        return length(vec2(length(lc.xy) - irP(base,12u), lc.z)) - irP(base,13u);""",
    # Placed 2D analytic sections — thin discs/quads on a world plane (origin@0,
    # axis_u@3, axis_v@6, normal@9; profile params @12+). Mirrors sdf_profiles.glsl.
    "placed_circle_2d": """vec3 l3 = p - irP3(base,0u);
        vec2 q = vec2(dot(l3, irP3(base,3u)), dot(l3, irP3(base,6u)));
        float plane = dot(l3, irP3(base,9u));
        float prof = length(q - vec2(irP(base,12u), irP(base,13u))) - irP(base,14u);
        return max(prof, abs(plane) - 0.002);""",
    "placed_rectangle_2d": """vec3 l3 = p - irP3(base,0u);
        vec2 q = vec2(dot(l3, irP3(base,3u)), dot(l3, irP3(base,6u)));
        float plane = dot(l3, irP3(base,9u));
        vec2 d = abs(q - vec2(irP(base,12u), irP(base,13u))) - vec2(irP(base,14u), irP(base,15u));
        float prof = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
        return max(prof, abs(plane) - 0.002);""",
    "placed_square_2d": """vec3 l3 = p - irP3(base,0u);
        vec2 q = vec2(dot(l3, irP3(base,3u)), dot(l3, irP3(base,6u)));
        float plane = dot(l3, irP3(base,9u));
        vec2 d = abs(q - vec2(irP(base,12u), irP(base,13u))) - vec2(irP(base,14u));
        float prof = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
        return max(prof, abs(plane) - 0.002);""",
    "placed_rounded_rectangle_2d": """vec3 l3 = p - irP3(base,0u);
        vec2 q = vec2(dot(l3, irP3(base,3u)), dot(l3, irP3(base,6u)));
        float plane = dot(l3, irP3(base,9u));
        float cr = irP(base,16u); vec2 inner = vec2(irP(base,14u), irP(base,15u)) - vec2(cr);
        vec2 d = abs(q - vec2(irP(base,12u), irP(base,13u))) - inner;
        float prof = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - cr;
        return max(prof, abs(plane) - 0.002);""",
    "placed_ellipse_2d": """vec3 l3 = p - irP3(base,0u);
        vec2 q = vec2(dot(l3, irP3(base,3u)), dot(l3, irP3(base,6u)));
        float plane = dot(l3, irP3(base,9u));
        float prof = irExactEllipseDistance(q - vec2(irP(base,12u), irP(base,13u)),
                                            vec2(irP(base,14u), irP(base,15u)));
        return max(prof, abs(plane) - 0.002);""",
    # Tubes (sweeps): radius around a polyline / bezier centerline. Points live
    # inline in params; pc = (param_count-3)/3, then radius/inner/flat_caps.
    "polyline_tube": """uint pc = (node.param_count - 3u) / 3u;
        float radius = irP(base, node.param_count - 3u);
        float inner = irP(base, node.param_count - 2u);
        bool flat_caps = irP(base, node.param_count - 1u) > 0.5;
        float cl = IR_FAR;
        for (uint i = 0u; i + 1u < pc; i++)
            cl = min(cl, irSegmentDistance3D(p, irP3(base, i*3u), irP3(base, (i+1u)*3u)));
        if (!flat_caps) return irTubeSDF(cl, radius, inner);
        float outer = IR_FAR;
        for (uint i = 0u; i + 1u < pc; i++)
            outer = min(outer, irFlatCappedSegmentTubeSDF3D(p, irP3(base, i*3u), irP3(base, (i+1u)*3u), radius));
        return irFlatTubeSDF(outer, cl, inner);""",
    "bezier_tube": """uint pc = (node.param_count - 3u) / 3u;
        float radius = irP(base, node.param_count - 3u);
        float inner = irP(base, node.param_count - 2u);
        bool flat_caps = irP(base, node.param_count - 1u) > 0.5;
        float cl = IR_FAR;
        for (uint i = 0u; i + 2u < pc; i += 2u)
            cl = min(cl, irQuadraticBezierDistance3D(p, irP3(base, i*3u), irP3(base, (i+1u)*3u), irP3(base, (i+2u)*3u)));
        if (!flat_caps) return irTubeSDF(cl, radius, inner);
        vec3 sp0 = irP3(base, 0u); vec3 fc = irP3(base, 3u); vec3 fe = irP3(base, 6u);
        vec3 ls = irP3(base, (pc-3u)*3u); vec3 lc = irP3(base, (pc-2u)*3u); vec3 ep = irP3(base, (pc-1u)*3u);
        vec3 st = irSafeDirection(fc - sp0, fe - sp0);
        vec3 et = irSafeDirection(ep - lc, ep - ls);
        float outer = max(max(cl - radius, dot(sp0 - p, st)), dot(p - ep, et));
        return irFlatTubeSDF(outer, cl, inner);""",
    # Placed 2D open curves (polyline / bezier) — points inline in params @12+.
    "placed_polyline_2d": """vec3 l3 = p - irP3(base,0u);
        vec2 q = vec2(dot(l3, irP3(base,3u)), dot(l3, irP3(base,6u)));
        float plane = dot(l3, irP3(base,9u));
        uint pc = (node.param_count - 12u) / 2u; float d = IR_FAR;
        for (uint i = 0u; i + 1u < pc; i++)
            d = min(d, irSegmentDistance2D(q, vec2(irP(base,12u+i*2u), irP(base,12u+i*2u+1u)),
                                              vec2(irP(base,12u+(i+1u)*2u), irP(base,12u+(i+1u)*2u+1u))));
        return max(d - 0.004, abs(plane) - 0.002);""",
    "placed_bezier_curve_2d": """vec3 l3 = p - irP3(base,0u);
        vec2 q = vec2(dot(l3, irP3(base,3u)), dot(l3, irP3(base,6u)));
        float plane = dot(l3, irP3(base,9u));
        uint pc = (node.param_count - 12u) / 2u; float d = IR_FAR;
        for (uint i = 0u; i + 2u < pc; i += 2u)
            d = min(d, irQuadraticBezierDistance(q, vec2(irP(base,12u+i*2u), irP(base,12u+i*2u+1u)),
                                                    vec2(irP(base,12u+(i+1u)*2u), irP(base,12u+(i+1u)*2u+1u)),
                                                    vec2(irP(base,12u+(i+2u)*2u), irP(base,12u+(i+2u)*2u+1u))));
        return max(d - 0.004, abs(plane) - 0.002);""",
    # Profile sub-graph leaves — call the embedded 2D/1D profile stack-VM
    # (evalProfileSDF / evalProfile1DSDF) over the node's children sub-graph.
    "placed_profile_2d": """vec3 l3 = p - irP3(base,0u);
        vec2 q = vec2(dot(l3, irP3(base,3u)), dot(l3, irP3(base,6u)));
        float plane = dot(l3, irP3(base,9u));
        float prof = evalProfileSDF(u_children[node.child_offset], q);
        return max(prof, abs(plane) - 0.002);""",
    "extrude_profile_2d": """vec3 l3 = p - irP3(base,0u);
        vec2 q = vec2(dot(l3, irP3(base,3u)), dot(l3, irP3(base,6u)));
        float plane = dot(l3, irP3(base,9u));
        float prof = evalProfileSDF(u_children[node.child_offset], q);
        float axial = abs(plane - irP(base,13u)) - irP(base,12u)*0.5;
        vec2 pr = vec2(prof, axial);
        return length(max(pr, vec2(0.0))) + min(max(prof, axial), 0.0);""",
    "revolve_profile_2d": """vec3 lc = p - irP3(base,12u);
        vec3 ax = irP3(base,15u); vec3 rdir = irP3(base,18u); vec3 tg = irP3(base,21u);
        float axial = dot(lc, ax); float rx = dot(lc, rdir); float ry = dot(lc, tg);
        float radial = sqrt(max(rx*rx + ry*ry, 0.0));
        vec3 sp = irP3(base,12u) + axial*ax + radial*rdir; vec3 sl = sp - irP3(base,0u);
        float prof = evalProfileSDF(u_children[node.child_offset],
                                    vec2(dot(sl, irP3(base,3u)), dot(sl, irP3(base,6u))));
        float angle = irP(base,24u);
        if (angle >= 6.283184307179586) return prof;
        float angular = irAngularSectorSDF(vec2(rx, ry), angle);
        vec2 pr = vec2(prof, angular);
        return length(max(pr, vec2(0.0))) + min(max(prof, angular), 0.0);""",
    "placed_profile_1d": """vec3 origin = irP3(base,0u); vec3 axis_u = irP3(base,3u);
        vec3 lc = p - origin; float tt = dot(lc, axis_u); vec3 radial = lc - tt*axis_u;
        float prof = evalProfile1DSDF(u_children[node.child_offset], tt);
        return max(prof, length(radial) - 0.004);""",
}

# Kinds whose leafDist calls the embedded profile stack-VM (evalProfileSDF /
# evalProfile1DSDF over a children sub-graph).
_PROFILE_KINDS = frozenset((
    "placed_profile_2d", "extrude_profile_2d", "revolve_profile_2d",
    "placed_profile_1d",
))
# Helpers the embedded profile-VM block already defines — don't double-emit them.
_PROFILE_VM_PROVIDES = frozenset((
    "irExactEllipseDistance", "irSegmentDistance2D", "irQuadraticBezierDistance",
))
# 2D/1D profile COMBINATOR kinds (boolean/offset of profiles). These are the only
# profile nodes that need the stack machine; a single analytic leaf does not.
_PROFILE_COMBINATORS = frozenset((
    "profile_union_2d", "profile_intersection_2d", "profile_difference_2d",
    "profile_offset_2d", "profile_distance_offset_2d",
    "profile_union_1d", "profile_intersection_1d", "profile_difference_1d",
))
# sdf_profiles.glsl lives next to this module — codegen is its only consumer.
_SHADER_DIR = os.path.dirname(__file__)
_profile_src_cache: list = []


def profiles_are_simple(render_ir: "RenderIR | None") -> bool:
    """True if every profile node's child sub-graph is a single analytic leaf (no
    2D/1D combinator). Then codegen calls irProfileLeafValue directly and SKIPS the
    stack-machine evalProfileSDF — a stack interpreter is pathological for the
    Vulkan shader compiler (it's the same shape as the bytecode VM), so straight-
    lining the common case keeps the shader Vulkan-friendly. Combinator profiles
    still emit the full sub-VM. Part of the bake key (renderer)."""
    if render_ir is None:
        return True
    nodes = render_ir.nodes
    for n in nodes:
        if n.kind in _PROFILE_KINDS and n.children:
            if nodes[int(n.children[0])].kind in _PROFILE_COMBINATORS:
                return False
    return True


def _profile_src() -> str:
    if not _profile_src_cache:
        _profile_src_cache.append(
            open(os.path.join(_SHADER_DIR, "sdf_profiles.glsl")).read())
    return _profile_src_cache[0]


def _profile_vm_glsl(simple: bool = False) -> str:
    """The nested 2D/1D profile evaluator, sliced from sdf_profiles.glsl by content
    markers (so it can't drift from the source). Embedded only when a profile kind
    is present.

    ``simple=True`` (single-leaf profiles): emit only the straight-line leaf
    evaluators (irProfileLeafValue / irProfile1DLeafValue) + thin wrappers, OMITTING
    the stack machines — keeps the shader free of a stack interpreter (Vulkan-fast).
    ``simple=False`` (combinator profiles): the full stack-VM."""
    src = _profile_src()
    if not simple:
        return src[src.index("float irSegmentDistance2D"):
                   src.index("bool irProfileLeaf(")].rstrip() + "\n"
    # helpers + irProfileLeafValue + irIsProfileCombinator (everything up to the
    # 2D stack machine), then just the 1D leaf evaluator, then wrappers that route
    # evalProfile*SDF straight to the leaf — no vstack, no program loop.
    leaf2d = src[src.index("float irSegmentDistance2D"):
                 src.index("float evalProfileSDF(")].rstrip()
    leaf1d = src[src.index("float irProfile1DLeafValue"):
                 src.index("float evalProfile1DSDF(")].rstrip()
    return (leaf2d + "\n" + leaf1d + "\n"
            + "float evalProfileSDF(uint root, vec2 q) { return irProfileLeafValue(root, q); }\n"
            + "float evalProfile1DSDF(uint root, float t) { return irProfile1DLeafValue(root, t); }\n")

# Analytic helpers each kind needs (emitted only when a using kind is present).
_HELPERS = {
    "irOrientedLocal": ("box", "cylinder", "cone", "capped_cone", "box_frame",
                        "pyramid", "torus"),
    "irPyramidUnitSDF": ("pyramid",),
    "irCappedConeSDF": ("capped_cone",),
    "irBoxFrameSDF": ("box_frame",),
    "irExactEllipseDistance": ("placed_ellipse_2d",),
    # Tube helpers (order matters: bezier-distance uses segment-distance).
    "irSegmentDistance3D": ("polyline_tube", "bezier_tube"),
    "irQuadraticBezierDistance3D": ("bezier_tube",),
    "irFlatCappedSegmentTubeSDF3D": ("polyline_tube",),
    "irSafeDirection": ("bezier_tube",),
    "irTubeSDF": ("polyline_tube", "bezier_tube"),
    "irFlatTubeSDF": ("polyline_tube", "bezier_tube"),
    "irSegmentDistance2D": ("placed_polyline_2d",),
    "irQuadraticBezierDistance": ("placed_bezier_curve_2d",),
}

_HELPER_GLSL = {
    "irOrientedLocal": """vec3 irOrientedLocal(vec3 p, vec3 c, vec3 au, vec3 av, vec3 aw) {
    vec3 l = p - c; return vec3(dot(l,au), dot(l,av), dot(l,aw));
}""",
    "irPyramidUnitSDF": """float irPyramidUnitSDF(vec3 p, float h) {
    float m2 = h*h + 0.25; p.xz = abs(p.xz); p.xz = (p.z > p.x) ? p.zx : p.xz; p.xz -= 0.5;
    vec3 q = vec3(p.z, h*p.y - 0.5*p.x, h*p.x + 0.5*p.y);
    float s = max(-q.x, 0.0); float t = clamp((q.y - 0.5*p.z)/(m2 + 0.25), 0.0, 1.0);
    float a = m2*(q.x+s)*(q.x+s) + q.y*q.y;
    float b = m2*(q.x+0.5*t)*(q.x+0.5*t) + (q.y - m2*t)*(q.y - m2*t);
    float d2 = min(q.y, -q.x*m2 - q.y*0.5) > 0.0 ? 0.0 : min(a,b);
    return sqrt((d2 + q.z*q.z)/m2) * sign(max(q.z, -p.y));
}""",
    "irCappedConeSDF": """float irCappedConeSDF(vec3 p, float h, float r1, float r2) {
    vec2 q = vec2(length(p.xy), p.z); vec2 k1 = vec2(r2, h); vec2 k2 = vec2(r2-r1, 2.0*h);
    vec2 ca = vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - h);
    vec2 cb = q - k1 + k2*clamp(dot(k1-q, k2)/dot(k2,k2), 0.0, 1.0);
    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s * sqrt(min(dot(ca,ca), dot(cb,cb)));
}""",
    "irBoxFrameSDF": """float irBoxFrameSDF(vec3 p, vec3 b, float e) {
    p = abs(p) - b; vec3 q = abs(p + e) - e;
    return min(min(
        length(max(vec3(p.x,q.y,q.z),0.0)) + min(max(p.x,max(q.y,q.z)),0.0),
        length(max(vec3(q.x,p.y,q.z),0.0)) + min(max(q.x,max(p.y,q.z)),0.0)),
        length(max(vec3(q.x,q.y,p.z),0.0)) + min(max(q.x,max(q.y,p.z)),0.0));
}""",
    "irExactEllipseDistance": """float irExactEllipseDistance(vec2 p, vec2 ab) {
    p = abs(p);
    if (p.x > p.y) { p = p.yx; ab = ab.yx; }
    float l = ab.y*ab.y - ab.x*ab.x;
    float m = ab.x*p.x/l; float m2 = m*m;
    float n = ab.y*p.y/l; float n2 = n*n;
    float c = (m2 + n2 - 1.0)/3.0; float c3 = c*c*c;
    float d = c3 + m2*n2; float q = d + m2*n2; float g = m + m*n2; float co;
    if (d < 0.0) {
        float h = acos(clamp(q/c3, -1.0, 1.0))/3.0;
        float s = cos(h); float t = sin(h)*sqrt(3.0);
        float rx = sqrt(max(-c*(s+t+2.0)+m2, 0.0));
        float ry = sqrt(max(-c*(s-t+2.0)+m2, 0.0));
        co = (ry + sign(l)*rx + abs(g)/(rx*ry) - m)*0.5;
    } else {
        float h = 2.0*m*n*sqrt(max(d, 0.0));
        float s = sign(q+h)*pow(abs(q+h), 1.0/3.0);
        float t = sign(q-h)*pow(abs(q-h), 1.0/3.0);
        float rx = -s - t - c*4.0 + 2.0*m2; float ry = (s-t)*sqrt(3.0);
        float rm = sqrt(max(rx*rx + ry*ry, 0.0));
        co = (ry/sqrt(max(rm-rx, 1.0e-12)) + 2.0*g/rm - m)*0.5;
    }
    co = clamp(co, 0.0, 1.0);
    vec2 r = ab*vec2(co, sqrt(max(1.0 - co*co, 0.0)));
    return length(r - p)*sign(p.y - r.y);
}""",
    "irSegmentDistance3D": """float irSegmentDistance3D(vec3 pos, vec3 A, vec3 B) {
    vec3 pa = pos - A; vec3 ba = B - A;
    float h = clamp(dot(pa, ba)/max(dot(ba, ba), 1.0e-12), 0.0, 1.0);
    return length(pa - ba*h);
}""",
    "irQuadraticBezierDistance3D": """float irQuadraticBezierDistance3D(vec3 pos, vec3 A, vec3 B, vec3 C) {
    vec3 a = B - A; vec3 b = A - 2.0*B + C; vec3 c = a*2.0; vec3 d = A - pos;
    float bb = dot(b, b);
    if (bb <= 1.0e-12) return irSegmentDistance3D(pos, A, C);
    float kk = 1.0/bb; float kx = kk*dot(a, b);
    float ky = (kk*(2.0*dot(a, a) + dot(d, b)))/3.0; float kz = kk*dot(d, a);
    float pp = ky - kx*kx; float qq = kx*(2.0*kx*kx - 3.0*ky) + kz;
    float hh = qq*qq + 4.0*pp*pp*pp; float res = 0.0;
    if (hh >= 0.0) {
        hh = sqrt(hh); vec2 x = (vec2(hh, -hh) - qq)*0.5;
        vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
        float t = clamp(uv.x + uv.y - kx, 0.0, 1.0);
        vec3 w = d + (c + b*t)*t; res = dot(w, w);
    } else {
        float z = sqrt(max(-pp, 0.0)); float den = 2.0*pp*z;
        float ang = den == 0.0 ? 0.0 : qq/den;
        float v = acos(clamp(ang, -1.0, 1.0))/3.0;
        float m = cos(v); float n = sin(v)*1.732050808;
        vec3 t = clamp(vec3(m + m, -n - m, n - m)*z - kx, 0.0, 1.0);
        vec3 qx = d + (c + b*t.x)*t.x; vec3 qy = d + (c + b*t.y)*t.y;
        res = min(dot(qx, qx), dot(qy, qy));
    }
    return sqrt(max(res, 0.0));
}""",
    "irFlatCappedSegmentTubeSDF3D": """float irFlatCappedSegmentTubeSDF3D(vec3 pos, vec3 A, vec3 B, float radius) {
    vec3 ba = B - A; float sl = length(ba);
    if (sl <= 1.0e-8) return length(pos - A) - radius;
    vec3 axis = ba/sl; vec3 pa = pos - A; float proj = dot(pa, axis);
    float radial = length(pa - axis*proj) - radius;
    float axial = abs(proj - 0.5*sl) - 0.5*sl;
    vec2 pair = vec2(radial, axial);
    return length(max(pair, vec2(0.0))) + min(max(radial, axial), 0.0);
}""",
    "irSafeDirection": """vec3 irSafeDirection(vec3 preferred, vec3 fallback) {
    float pl = length(preferred);
    if (pl > 1.0e-12) return preferred/pl;
    return fallback/max(length(fallback), 1.0e-12);
}""",
    "irTubeSDF": """float irTubeSDF(float cl, float radius, float inner) {
    float outer = cl - radius;
    if (inner <= 0.0) return outer;
    return max(outer, inner - cl);
}""",
    "irFlatTubeSDF": """float irFlatTubeSDF(float outer, float cl, float inner) {
    if (inner <= 0.0) return outer;
    return max(outer, inner - cl);
}""",
    "irSegmentDistance2D": """float irSegmentDistance2D(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a; vec2 ba = b - a;
    float h = clamp(dot(pa, ba)/max(dot(ba, ba), 1.0e-12), 0.0, 1.0);
    return length(pa - ba*h);
}""",
    "irQuadraticBezierDistance": """float irQuadraticBezierDistance(vec2 pos, vec2 A, vec2 B, vec2 C) {
    vec2 a = B - A; vec2 b = A - 2.0*B + C; vec2 c = a*2.0; vec2 d = A - pos;
    float bb = dot(b, b);
    if (bb <= 1.0e-12) {
        vec2 pa = pos - A; vec2 ba = C - A;
        float hh = clamp(dot(pa, ba)/max(dot(ba, ba), 1.0e-12), 0.0, 1.0);
        return length(pa - ba*hh);
    }
    float kk = 1.0/bb; float kx = kk*dot(a, b);
    float ky = (kk*(2.0*dot(a, a) + dot(d, b)))/3.0; float kz = kk*dot(d, a);
    float pp = ky - kx*kx; float qq = kx*(2.0*kx*kx - 3.0*ky) + kz;
    float hh = qq*qq + 4.0*pp*pp*pp; float res = 0.0;
    if (hh >= 0.0) {
        hh = sqrt(hh); vec2 x = (vec2(hh, -hh) - qq)*0.5;
        vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
        float t = clamp(uv.x + uv.y - kx, 0.0, 1.0);
        vec2 w = d + (c + b*t)*t; res = dot(w, w);
    } else {
        float z = sqrt(max(-pp, 0.0)); float den = 2.0*pp*z;
        float ang = den == 0.0 ? 0.0 : qq/den;
        float v = acos(clamp(ang, -1.0, 1.0))/3.0;
        float m = cos(v); float n = sin(v)*1.732050808;
        vec3 t = clamp(vec3(m + m, -n - m, n - m)*z - kx, 0.0, 1.0);
        vec2 qx = d + (c + b*t.x)*t.x; vec2 qy = d + (c + b*t.y)*t.y;
        res = min(dot(qx, qx), dot(qy, qy));
    }
    return sqrt(max(res, 0.0));
}""",
}


# Region selectors (Layer 2): tag region_id on a target object's surface where a
# point is inside a selector volume (and optional scope). They don't change
# distance, so they run as a post-process over the geometry. Embedded only when a
# region_selector is present; a no-op stub otherwise. Uses a float-only 3D subtree
# stack-VM (only .dist is needed) over leafDist.
_SELECTOR_STUB = "uint regionAt(vec3 p, uint owner) { return 0u; }\n"
_SELECTOR_GLSL = """bool irSelIsOp(uint t) {
    return t == NODE_UNION || t == NODE_INTERSECTION || t == NODE_DIFFERENCE;
}
float evalSubtreeDist(uint root, vec3 p) {
    uint wn[IR_STACK_CAPACITY]; int ws[IR_STACK_CAPACITY]; float vs[IR_STACK_CAPACITY];
    int wsp = 0; int vsp = 0; wn[0] = root; ws[0] = 0; wsp = 1;
    while (wsp > 0) {
        wsp--; uint idx = wn[wsp]; int st = ws[wsp]; GpuNode n = u_nodes[idx];
        if (st == 0 && irSelIsOp(n.type)) {
            wn[wsp] = idx; ws[wsp] = 1; wsp++;
            for (int c = int(n.child_count) - 1; c >= 0; c--) {
                wn[wsp] = u_children[n.child_offset + uint(c)]; ws[wsp] = 0; wsp++;
            }
        } else if (st == 1) {
            int base = vsp - int(n.child_count); float acc = vs[base];
            for (int i = 1; i < int(n.child_count); i++) {
                float b = vs[base + i];
                if (n.type == NODE_UNION) acc = min(acc, b);
                else if (n.type == NODE_INTERSECTION) acc = max(acc, b);
                else acc = max(acc, -b);
            }
            vsp = base; vs[vsp] = acc; vsp++;
        } else { vs[vsp] = leafDist(idx, p); vsp++; }
    }
    return vs[0];
}
uint regionAt(vec3 p, uint owner) {
    uint region = 0u;
    for (uint si = 0u; si < u_sel_count; si++) {
        GpuNode sn = u_nodes[u_sel[si]];
        if (owner != sn.base_owner_id) continue;
        float sel_d = evalSubtreeDist(u_children[sn.child_offset], p);
        if (sn.child_count > 1u) {
            if (evalSubtreeDist(u_children[sn.child_offset + 1u], p) > 1.5e-3) continue;
        }
        float tol = irP(sn.param_offset, 0u);
        bool inside = (tol >= 0.0) ? (sel_d <= tol) : (sel_d > -tol);
        if (inside) region = sn.flags;
    }
    return region;
}
"""


def selector_indices(render_ir: RenderIR | None) -> list[int]:
    """Node indices of region_selector nodes (applied as a post-process)."""
    if render_ir is None:
        return []
    return [i for i, n in enumerate(render_ir.nodes)
            if n.kind == "region_selector"]


def flatten_terms(render_ir: RenderIR | None):
    """DNF over CARVED TERMS for codegen — the scene SDF is

        max_g( min_{term in g} termDist(term) ),
        termDist((leaf, carves)) = max( leafDist(leaf), max_{c in carves} -leafDist(c) )

    i.e. each group is a union (min) of carved terms, and groups are intersected
    (max). A *term* is one positive leaf with a LOCAL carve list (the holes cut out
    of it). Because a carve rides its term locally — instead of becoming a negative
    literal distributed through the union cross-product — a union of K carved solids
    stays K terms in ONE group (linear), where a signed-literal DNF would explode to
    2**K groups.

    Returns a tuple of groups (each a tuple of ``(leaf, carves)`` terms, ``carves``
    a tuple of leaf indices) or None — unbounded, an unsupported operator shape
    (e.g. a carved/compound subtraction tool), or more than CG_GROUP_CEILING groups.
    """
    from core.gpu_node_types import is_operator
    if render_ir is None or not render_ir.nodes \
            or len(render_ir.root_indices) != 1:
        return None
    nodes = render_ir.nodes

    # iterative post-order, memoised per node index (context-free)
    order, work = [], [(render_ir.root_indices[0], False)]
    while work:
        idx, done = work.pop()
        if done:
            order.append(idx)
            continue
        work.append((idx, True))
        if is_operator(nodes[idx].kind):
            for c in nodes[idx].children:
                work.append((int(c), False))

    memo: dict = {}
    for idx in order:
        if idx in memo:
            continue
        node = nodes[idx]
        if not is_operator(node.kind):
            memo[idx] = (((idx, ()),),)            # 1 group, 1 bare term
            continue
        kids = [memo[int(c)] for c in node.children]
        if any(k is None for k in kids):
            memo[idx] = None
            continue
        if node.kind == "union":
            # min over children: cross-product the children's GROUP lists, each
            # combined group = the union (concatenation) of one group's terms from
            # every child. All-single-group children -> 1 group (terms appended).
            acc = ((),)
            ok = True
            for kg in kids:
                acc = tuple(a + b for a in acc for b in kg)
                if len(acc) > CG_GROUP_CEILING:
                    ok = False
                    break
            memo[idx] = acc if ok else None
        elif node.kind == "intersection":
            # max over children: concatenate their group lists.
            groups = tuple(g for kg in kids for g in kg)
            memo[idx] = groups if len(groups) <= CG_GROUP_CEILING else None
        elif node.kind == "difference":
            # carve the tool leaves into EVERY term of the base (distributes over
            # the base's groups/terms). Tool must be a single group of bare terms
            # (a solid or union of solids, no nested carve/intersection) else bail.
            base = kids[0]
            carves: list[int] = []
            ok = True
            for tool in kids[1:]:
                if len(tool) != 1 or any(cv for _leaf, cv in tool[0]):
                    ok = False
                    break
                carves += [leaf for leaf, _cv in tool[0]]
            if not ok:
                memo[idx] = None
                continue
            extra = tuple(carves)
            memo[idx] = tuple(
                tuple((leaf, cv + extra) for leaf, cv in grp) for grp in base)
        else:
            memo[idx] = None

    result = memo[render_ir.root_indices[0]]
    if result is None:
        return None
    groups = tuple(g for g in result if g)
    if not groups:
        return None
    return groups


def _dnf_leaf_indices(groups) -> set[int]:
    """The 3D-scene leaf node indices the term DNF references (each term's positive
    leaf + its local carves). NOT the profile sub-graph nodes under extrude/revolve/
    placed_profile — those are reached via children and handled by the embedded
    profile VM."""
    leaves: set[int] = set()
    for group in groups:
        for leaf, carves in group:
            leaves.add(leaf)
            leaves.update(carves)
    return leaves


def scene_structure_signature(render_ir: RenderIR | None) -> frozenset[str]:
    """The recompile key: the set of 3D-scene leaf KINDS. Instance data changes
    (and profile sub-graph contents) never re-bake; only a new leaf kind does."""
    if render_ir is None:
        return frozenset()
    groups = flatten_terms(render_ir)
    if groups is None:
        return frozenset()
    return frozenset(
        render_ir.nodes[i].kind for i in _dnf_leaf_indices(groups)
        if render_ir.nodes[i].kind in _LEAF_GLSL)


def supported(render_ir: RenderIR | None) -> bool:
    """True if the scene flattens to a bounded term DNF (incl. carve-under-union)
    and every 3D-scene leaf kind is one the codegen path handles. Profile sub-graph
    kinds (under extrude/revolve/placed_profile) are covered by the embedded
    profile VM, so they're not gated here."""
    if render_ir is None or not render_ir.nodes:
        return False
    groups = flatten_terms(render_ir)
    if groups is None:
        return False
    return all(render_ir.nodes[i].kind in _LEAF_GLSL
               for i in _dnf_leaf_indices(groups))


def _leaf_dist_fn(kinds: frozenset[str]) -> str:
    """Emit leafDist() with ONLY the present primitive kinds branched."""
    branches = []
    for kind in sorted(kinds):
        body = _LEAF_GLSL[kind]
        branches.append(f"    if (t == NODE_{kind.upper()}) {{ {body} }}")
    return ("float leafDist(uint ni, vec3 p) {\n"
            "    GpuNode node = u_nodes[ni];\n"
            "    uint base = node.param_offset; uint t = node.type;\n"
            + "\n".join(branches)
            + "\n    return IR_FAR;\n}")


def _helpers_for(kinds: frozenset[str], skip: frozenset[str] = frozenset()) -> str:
    out = []
    for name, users in _HELPERS.items():
        if name in skip:
            continue
        if any(k in kinds for k in users):
            out.append(_HELPER_GLSL[name])
    return "\n".join(out)


def emit_map_glsl(render_ir: RenderIR) -> str:
    """Emit the GLSL preamble + leafDist + the term-DNF ``map()`` for a scene.

    ``map(p, owner)`` returns the scene SDF and the owning object id of the
    nearest surface: it loops over the Terms buffer, computing each carved term
    (``leafDist`` carved by its local Carves slice), scatters it into its group's
    accumulator by ``min``, then ``max``es the groups (intersection). See
    flatten_terms for the algebra.
    """
    kinds = scene_structure_signature(render_ir)
    use_profile_vm = bool(kinds & _PROFILE_KINDS)
    simple_profiles = use_profile_vm and profiles_are_simple(render_ir)
    profile_block = _profile_vm_glsl(simple_profiles) if use_profile_vm else ""
    skip = _PROFILE_VM_PROVIDES if use_profile_vm else frozenset()
    selector_block = _SELECTOR_GLSL if selector_indices(render_ir) else _SELECTOR_STUB
    # Adaptive-capacity group accumulators (NOT unrolled per group): the term loop
    # scatters into g[gid] by dynamic index, then the combine loop maxes over the
    # runtime u_group_count. cap = the scene's bucketed group capacity, baked into
    # the array size (part of the bake-cache key); the shader is fixed-size for any
    # group count <= cap.
    cap = group_capacity(render_ir) or _CG_GROUP_BUCKETS[-1]
    return f"""{emit_glsl_defines()}
struct GpuNode {{ uint type; uint dim; uint base_owner_id; uint flags;
                 uint param_offset; uint param_count; uint child_offset; uint child_count; }};
layout(std430, binding = 0) readonly buffer Nodes  {{ GpuNode u_nodes[]; }};
layout(std430, binding = 1) readonly buffer Params {{ float   u_params[]; }};
layout(std430, binding = 2) readonly buffer Children  {{ uint  u_children[]; }};
layout(std430, binding = 3) readonly buffer Selectors {{ uint  u_sel[]; }};
layout(std430, binding = 4) readonly buffer Terms  {{ uvec4 u_terms[]; }};
layout(std430, binding = 5) readonly buffer Carves {{ uint  u_carves[]; }};
// Spatial cull grid (binned terms; DDA-marched). Items are term indices; built
// host-side by core.gpu_cull.build_term_grid. Gated by u_cull_enabled.
layout(std430, binding = 6) readonly buffer GridOff  {{ uint u_goff[]; }};
layout(std430, binding = 7) readonly buffer GridCnt  {{ uint u_gcnt[]; }};
layout(std430, binding = 8) readonly buffer GridItem {{ uint u_gitem[]; }};
const float IR_FAR = 1.0e6;
uniform uint u_term_count;
uniform uint u_sel_count;
uniform int  u_group_count;
uniform vec3 u_grid_origin;
uniform vec3 u_grid_cell;
uniform int  u_grid_dim;
uniform int  u_cull_enabled;
float irP(uint base, uint i) {{ return u_params[base + i]; }}
vec3 irP3(uint base, uint i) {{ return vec3(u_params[base+i], u_params[base+i+1u], u_params[base+i+2u]); }}
{_helpers_for(kinds, skip)}
{profile_block}
{_leaf_dist_fn(kinds)}
{selector_block}
// One carved term: the positive leaf raised by -leafDist of each local carve.
float termDist(uint ti, vec3 p, out uint owner) {{
    uvec4 t = u_terms[ti]; uint leaf = t.x; uint co = t.z; uint cc = t.w;
    float d = leafDist(leaf, p);
    for (uint j = 0u; j < cc; j++) {{
        float h = -leafDist(u_carves[co + j], p);
        if (h > d) d = h;
    }}
    owner = u_nodes[leaf].base_owner_id;
    return d;
}}
// term DNF: max over groups of (min over carved terms within the group). Brute
// force over ALL terms (used for normals and the no-cull path).
float map(vec3 p, out uint owner_out) {{
    float g[{cap}]; uint o[{cap}];
    for (uint k = 0u; k < {cap}u; k++) {{ g[k] = IR_FAR; o[k] = 0u; }}
    for (uint i = 0u; i < u_term_count; i++) {{
        uint gid = u_terms[i].y; uint own; float d = termDist(i, p, own);
        if (gid < {cap}u && d < g[gid]) {{ g[gid] = d; o[gid] = own; }}
    }}
    float res = g[0]; uint owner = o[0];
    for (uint k = 1u; k < uint(u_group_count) && k < {cap}u; k++)
        if (g[k] > res) {{ res = g[k]; owner = o[k]; }}
    owner_out = owner;
    return res;
}}
// Same DNF but over ONLY the terms binned into grid cell `ci` (the cull path).
float cellDist(int ci, vec3 p, out uint owner_out) {{
    float g[{cap}]; uint o[{cap}];
    for (uint k = 0u; k < {cap}u; k++) {{ g[k] = IR_FAR; o[k] = 0u; }}
    uint off = u_goff[ci]; uint cnt = u_gcnt[ci];
    for (uint i = 0u; i < cnt; i++) {{
        uint ti = u_gitem[off + i]; uint gid = u_terms[ti].y;
        uint own; float d = termDist(ti, p, own);
        if (gid < {cap}u && d < g[gid]) {{ g[gid] = d; o[gid] = own; }}
    }}
    float res = g[0]; uint owner = o[0];
    for (uint k = 1u; k < uint(u_group_count) && k < {cap}u; k++)
        if (g[k] > res) {{ res = g[k]; owner = o[k]; }}
    owner_out = owner;
    return res;
}}
bool cgRayGrid(vec3 ro, vec3 rd, out float t0, out float t1) {{
    vec3 lo = u_grid_origin; vec3 hi = u_grid_origin + u_grid_cell * float(u_grid_dim);
    vec3 inv = 1.0 / rd; vec3 a = (lo - ro) * inv; vec3 b = (hi - ro) * inv;
    vec3 tmn = min(a, b); vec3 tmx = max(a, b);
    t0 = max(max(tmn.x, tmn.y), tmn.z); t1 = min(min(tmx.x, tmx.y), tmx.z);
    return t1 >= max(t0, 0.0);
}}
ivec3 cgCellCoord(vec3 p) {{
    return clamp(ivec3(floor((p - u_grid_origin) / u_grid_cell)),
                 ivec3(0), ivec3(u_grid_dim - 1));
}}
float cgCellExit(vec3 p, vec3 rd, ivec3 c) {{
    vec3 cmin = u_grid_origin + vec3(c) * u_grid_cell; vec3 cmax = cmin + u_grid_cell;
    vec3 inv = 1.0 / rd; vec3 tb = max((cmin - p) * inv, (cmax - p) * inv);
    return min(min(tb.x, tb.y), tb.z);
}}
"""


# Raymarch main: camera ray -> cull-grid DDA (or brute-force map()) -> Lambert +
# owner palette, grid plane behind. Camera/grid uniforms are loose (the host
# collects them into one std140 block).
_RAYMARCH_MAIN = """
layout(location = 0) out vec4 frag_color;
uniform vec2 u_resolution;
uniform vec3 u_camera_position;
uniform vec3 u_camera_target;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform float u_focal_length;
uniform vec3 u_background_color;
uniform int u_show_grid;
uniform float u_grid_spacing;
uniform int u_fb_y_up;
vec3 palette(uint id) {
    uint s = (max(id, 1u) - 1u) % 6u;
    if (s == 0u) return vec3(0.95, 0.28, 0.16);
    if (s == 1u) return vec3(0.20, 0.72, 1.00);
    if (s == 2u) return vec3(0.25, 0.88, 0.38);
    if (s == 3u) return vec3(1.00, 0.76, 0.16);
    if (s == 4u) return vec3(0.82, 0.36, 1.00);
    return vec3(0.12, 0.90, 0.82);
}
float mapd(vec3 p) { uint o; return map(p, o); }
vec3 mapNormal(vec3 p) {
    const vec2 k = vec2(1.0, -1.0); const float e = 0.0008;
    return normalize(k.xyy*mapd(p+k.xyy*e) + k.yyx*mapd(p+k.yyx*e)
                   + k.yxy*mapd(p+k.yxy*e) + k.xxx*mapd(p+k.xxx*e));
}
vec3 gridBg(vec3 ro, vec3 rd, vec3 bg) {
    if (u_show_grid == 0 || abs(rd.z) < 1e-6) return bg;
    float tt = -ro.z / rd.z; if (tt <= 0.0) return bg;
    vec2 g = (ro + rd*tt).xy;
    vec2 w = fwidth(g);
    vec2 a = abs(fract(g/u_grid_spacing + 0.5) - 0.5) * u_grid_spacing;
    float line = 1.0 - smoothstep(0.0, max(max(w.x,w.y),1e-5)*1.5, min(a.x, a.y));
    float fade = clamp(1.0/(1.0 + tt*tt*0.002), 0.0, 1.0);
    return mix(bg, vec3(0.33, 0.38, 0.47), line*0.6*fade);
}
void main() {
    vec2 px = gl_FragCoord.xy;
    vec2 uv = (px - 0.5*u_resolution)/max(u_resolution.y, 1.0);
    if (u_fb_y_up == 0) uv.y = -uv.y;
    vec3 fwd = normalize(u_camera_target - u_camera_position);
    vec3 rd = normalize(2.0*uv.x*normalize(u_camera_right)
                      + 2.0*uv.y*normalize(u_camera_up) + u_focal_length*fwd);
    vec3 ro = u_camera_position;
    float t = 0.0; bool hit = false; uint owner = 0u; vec3 hp = ro;
    if (u_cull_enabled == 1) {
        // DDA the cull grid: evaluate only the current cell's terms, never step
        // past the cell boundary (the next cell may hold nearer geometry); empty
        // cells jump straight to their exit face.
        float t0, t1;
        if (cgRayGrid(ro, rd, t0, t1)) {
            t = max(t0, 0.0);
            for (int i = 0; i < 256; i++) {
                if (t > t1 + 0.01) break;
                hp = ro + rd*t;
                ivec3 c = cgCellCoord(hp);
                int ci = (c.z*u_grid_dim + c.y)*u_grid_dim + c.x;
                float d = cellDist(ci, hp, owner);
                if (d < 0.0008) { hit = true; break; }
                float step = min(max(d, 0.0002), cgCellExit(hp, rd, c) + 0.0008);
                t += max(step, 0.0008);
            }
        }
    } else {
        for (int i = 0; i < 160; i++) {
            hp = ro + rd*t; float d = map(hp, owner);
            if (d < 0.0008) { hit = true; break; }
            t += max(d, 0.0002); if (t > 100.0) break;
        }
    }
    vec3 col = gridBg(ro, rd, u_background_color);
    if (hit) {
        vec3 n = mapNormal(hp);
        float diff = max(dot(n, normalize(vec3(0.7, 1.0, 0.45))), 0.0);
        col = palette(owner) * (0.25 + 0.75*diff);
        uint region = regionAt(hp, owner);   // Layer 2 region tag (no-op if none)
        if (region != 0u) col = mix(col, vec3(1.0, 0.30, 0.28), 0.45);
    }
    frag_color = vec4(col, 1.0);
}
"""


def emit_fragment_shader(render_ir: RenderIR, *, version: str = "#version 460") -> str:
    """Complete codegen fragment shader (preamble + leafDist + map + raymarch
    main) for a supported scene. Loose uniforms still need ``vulkanize`` before
    ``qsb`` (host convention)."""
    return f"{version}\n{emit_map_glsl(render_ir)}{_RAYMARCH_MAIN}"


__all__ = [
    "emit_map_glsl", "emit_fragment_shader", "scene_structure_signature",
    "supported", "flatten_terms", "group_capacity", "profiles_are_simple",
    "selector_indices",
]
