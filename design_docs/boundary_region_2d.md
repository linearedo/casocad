# Boundary Region support for 2D domains

Status: IMPLEMENTED (2026-07-19), except §7's SU2 marker-membership test,
which is still to be written on the meshing side.
Closes the "2D domain parity" open item of `boundary_region_v2.md` §9.

## 1. Motivation and scope

A 2D domain root (`PlacedSdf2D`, allowed by `set_fluid_domain` — scene.rs
`dimension != 2 && dimension != 3` check) is the natural setup for planar CFD
cases (e.g. the NACA 0012 airfoil SU2 case in `meshing/examples/`). Today the
Boundary Region tool can never tag anything on such a domain: the pick path is
3D-only, so the button silently does nothing.

In scope:

- pick / create / select curve regions on the outline of a 2D domain,
- hover / selected / split-preview highlights for curve regions,
- the cutter for curve regions (point knife + dimension-aware segment knife),
- meshing consumption verified end to end (2D markers).

Every phase is the one-dimension-down translation of a mechanism the 3D tool
already has, mapped to the user-visible flow:

| user action | 3D mechanism today | 2D analog (this plan) |
|---|---|---|
| hover names a part | box → 6 faces, `pick_boundary_patch` | rectangle → 4 edges, ray∩plane pick (§4) |
| yellow/cyan highlight | triangle filter `region_highlight_mesh` | outline-arc ribbon (§5) |
| cutter splits | plane knife, seam root-found | line knife, split point root-found (§6) |
| mesher consumes | classifier masks | same masks, 2D markers (§7) |

Out of scope (see §9): non-flat 2D manifolds, stencil knives on 2D domains,
zero-thickness baffles.

## 2. Where it breaks today (gap analysis)

- `pick_boundary_patch` (kernel/src/boundary_ops.rs:873) is explicitly the
  "3D path". Patch generation (`surface_patches_for_node`) bails for any leaf
  with `dimension() != 3` (boundary_ops.rs:576), so a 2D root yields zero
  patches; the sphere-trace fallback then finds a surface point but has no
  patch to attribute it to, and the pick returns `None`.
- `CURVE_PATCH_PICK_TOLERANCE` (boundary_ops.rs:30) is defined and unused —
  the Python 2D curve-pick path was never ported.
- The highlight overlay (`region_highlight_mesh`,
  app/src/boundary_tool.rs:73) filters display *triangles*. A 2D domain's
  filled sheet has fan/ear-clip triangles whose interior vertices fail
  criterion 1 (interior of the sheet has f < 0), so no whole triangle ever
  passes: the overlay is always empty for 2D.
- `straight_knife` (kernel/src/boundary_paths.rs:68) builds
  `side_axis = normal × line_axis` from the *surface gradient*. On a 2D root
  the gradient and the click line are both in-plane, so `side_axis` comes out
  perpendicular to the workplane — the knife classifies by height above the
  plane, which is degenerate for curve points (all sit at height ≈ 0).

What already works unchanged (and must not be forked):

- classifier criteria 1–2 (`boundary_region_base_mask` / `walk_owner`) are
  pure sign tests, dimension-agnostic;
- `BoundaryCut` storage, chain conjunction, serialization, undo;
- `cut_volume` / `surface_selector_volume` already extrude a `PlacedSdf2D`
  ghost into a 3D prism (boundary_ops.rs:960);
- `add_boundary_region` / `split_boundary_region` (scene.rs:1947 / :2003) are
  patch-id-driven and generic;
- the meshing contract (`MeshableBoundaryRegion`, `MeshableDomain.dimension`)
  is mask/field based.

## 3. Structural facts that drive the design

1. **2D booleans merge into one leaf.** A boolean of two coplanar 2D objects
   produces a single `Placed2D` payload whose profile is
   `Profile2D::Binary { left, right: Offset { … } }`, with `sources` keeping
   the two operand objects (scene.rs:804-846). So unlike 3D, a 2D domain has
   exactly ONE provenance leaf: node-level attribution (criterion 2) is
   trivially satisfied, and telling "airfoil edge" from "farfield edge" must
   happen at the **profile tree** level, not the node tree level.
2. **`PlacedSdf2D::eval_point` ignores the plane offset** (placed.rs:63): the
   field is a prism, constant along the workplane normal. Criterion 1's
   on-boundary band is therefore an infinite prism around the outline — fine
   for classification (cut ghosts extrude to prisms too, and mesh points lie
   on the plane), but it means sphere-tracing a ray against the root converges
   to the prism, not to the plane. 2D picking must intersect the workplane
   analytically instead.
3. **2D domains render as filled sheet + wire outline**
   (`placed_2d_outline`, surfaces/src/profiles2d.rs:704): the outline polyline
   is already computed by `profile_outline` at display resolution, and
   `ViewportSurface.wire_indices` already draws line lists (the 1D objects use
   exactly this). The curve highlight should be wire-based, not triangle-based.
4. **The legal outline of a 2D domain is always planar.** Coplanarity is
   enforced for 2D booleans, transforms are refused for non-3D objects
   (scene.rs:929), and 1D objects cannot be domain roots (scene.rs:1053). The
   cutter never has to handle a non-planar boundary curve.

## 3b. Non-regression contract (hard requirement)

3D behavior stays byte-identical. Every touched entry point gains only a
branch or match arm for inputs that today produce nothing (`_` fallthrough,
`None`, empty overlay, unreachable path); all 2D handling sits behind
`root.dimension() == 2` dispatch. `straight_knife`, `stencil_knife`, the
classifier (`boundary_region_base_mask` / `boundary_region_mask`),
`walk_owner`, `cut_volume`, `add_boundary_region`, `split_boundary_region`
and serialization are not modified. The existing 3D test suite must pass
UNMODIFIED — this plan adds tests, it never edits or deletes one. Any step
that would require changing an existing test is a design error: stop and
revisit the plan instead.

## 4. Phase 1 — hover names a part of the outline (kernel: curve patches + pick)

All in `kernel/src/boundary_ops.rs` unless noted.

### 4.1 Curve patch catalogue

New arm in `surface_patches_for_node` for `Shape::PlacedSdf2D`, emitting
`BoundarySurfacePatch` values whose `patch_type` marks them as curves:

| Profile2D variant | patches | patch_type |
|---|---|---|
| Rectangle / Square | `-U`, `+U`, `-V`, `+V` edges | `edge` |
| Polygon / RegularPolygon | `edge_0` … `edge_{n-1}` | `edge` |
| Circle / Ellipse | one `outline` patch | `outline` |
| QuadraticBezierSurface | one `outline` patch | `outline` |
| Offset / DistanceOffset | recurse into child, keep patch ids | — |
| Binary (boolean profile) | recurse both sides, prefix ids (§4.2) | — |
| anything else | one `outline` fallback patch | `outline` |

Straight edges get `normal = Some(in-plane outward normal)` and
`outside_direction` 0..3 mapped onto the existing `PlacedSdf2D` arm of
`owner_outside_direction_vectors` (±axis_u, ±axis_v) where the edge normal
aligns with an axis; curved patches get `normal = None` (like `side_wall`).
The `owner` field holds the placed node as today; patches also need the
sub-profile (for containment), so the 2D arm carries a resolved
`Profile2D` + in-plane transform per patch (extend `BoundarySurfacePatch`
with an optional 2D payload rather than forking the struct).

### 4.2 Profile-level attribution (the 2D criterion 2/3)

A small profile walk, mirroring `evaluate_with_attribution` but over
`Profile2D::Binary` / `Offset` / `DistanceOffset`:

- `profile_eval_with_attribution(profile, u, v) -> (f64, path)` where `path`
  identifies the controlling sub-profile (index path through the Binary tree,
  stable across sessions; when `sources` provide named objects, patch ids use
  the source object name for readability: `airfoil.outline`,
  `flowbox.-U`).
- Difference provenance mirrors 3D: for `A − B` the subtracted profile owns
  the cut portion of the outline, patch ids get the `cut_surface.` prefix and
  a negated normal sign, matching `surface_patches_for_node`'s Difference arm.

`surface_patch_contains` for curve patches: the point projects into the
workplane; containment = |sub-profile boundary distance| ≤ tolerance AND the
controlling sub-profile at (u, v) is this patch's sub-profile (plus the edge
interval test for straight edges, analogous to the box face test).

`region_patch_scope_volume` for curve patches: a `DistanceOffset` band around
the sub-profile (reusing the exact pattern of the `PlacedPolyline1D` arm of
`surface_selector_volume`, boundary_ops.rs:974) placed on the domain plane and
extruded — a thin prism that limits the region to its edge/outline.

### 4.3 2D pick path

In `pick_boundary_patch`, branch on `root.dimension() == 2`:

1. ray ∩ workplane (reject near-parallel rays),
2. project to (u, v), evaluate each curve patch's in-plane distance to its
   curve,
3. hit = nearest patch within `CURVE_PATCH_PICK_TOLERANCE × root diagonal`
   (make the constant scale-relative — everything else in this module is;
   0.05 absolute was the Python leftover),
4. the returned `BoundaryPatchHit.point` is the *nearest point on the curve*
   (not the raw plane hit), `normal` is the in-plane outward normal there.

Cut-surface patches win outright, as in 3D. A helper
`pick_outline_point(root, ray) -> Option<Vec3>` (plane-hit + snap to outline)
replaces `pick_sdf_surface` for 2D cutter clicks (fact §3.2 makes the
sphere-trace unusable).

### 4.4 Region creation plumbing

`add_boundary_region` works as-is once patches exist. Small items:

- name pattern already generic (`"{owner} {patch}"` → "Domain airfoil.outline");
- `BoundaryRegion::dimension()` hardcodes 2 (boundary.rs:90); audit usages
  (none found in app/serialization today) and either leave it or derive from
  the root — decide during implementation, do not block on it.

## 5. Phase 2 — you see what you tag (app: curve highlight)

All in `app/src/boundary_tool.rs` + `app/src/tools.rs`.

- `handle_boundary_region` (tools.rs:620) is generic; only the status strings
  need a dimension-aware variant ("hover the domain outline …").
- New `region_highlight_polyline(root, region, scene) -> wire ViewportSurface`
  next to `region_highlight_mesh`:
  1. take outline points from the domain's display surface wire (or re-run
     `profile_outline` at the display resolution — same source, no drift),
  2. mask vertices with `boundary_region_base_mask`,
  3. for each cut in the chain, root-find the exact crossing parameter along
     any outline segment where the cut volume's sign changes (bisection on the
     segment — the 1D analog of the 3D seam root-finding). Both split-preview
     children run the identical deterministic routine with opposite keep
     sides, so the shared split vertex is bitwise-identical — the same
     crack-free guarantee the 3D pipeline documents,
  4. emit a wire surface (`indices` empty, `wire_indices` line list), lifted
     in-plane along the outward normal by diagonal·1e-3 (never along the plane
     normal — the overlay must stay in the sheet plane to read correctly from
     both sides).
- `region_highlight_surface` dispatches on `root.dimension()`; overlay colors
  and revision bookkeeping (`set_hover`) unchanged.
- `validation_points` for 2D should feed outline vertices (they are already in
  the display surface's vertex list; the filled-sheet interior vertices are
  harmless — the base mask rejects them).

## 6. Phase 3 — splitting an arc (cutter)

### 6.1 Kernel (`boundary_paths.rs`)

- **Segment knife, dimension-aware:** for a 2D root, call `straight_knife`
  with the *domain workplane normal* in place of the mean surface gradient.
  Then `side_axis = plane_normal × line_axis` is the in-plane perpendicular
  and the ghost is a half-plane in the domain plane bounded by the click
  line — the correct field (§2 explains why the gradient version is
  degenerate). The ghost is still a `PlacedSdf2D`, so `cut_volume` extrudes
  it to a prism with zero new code.
- **Point knife (new, 2D only):** `point_knife(root, click)` — the knife line
  passes through the click along the outline's local in-plane outward normal
  (so it crosses the curve transversally at the click); `side_axis` is the
  tangent, discriminating "before/after the click" along the curve. Built as
  the same half-plane `PlacedSdf2D` ghost via `straight_knife(click,
  click + normal · ε_span, plane_normal)` internals.
- **Warnings and refusals** (same discipline as `KnifeGhost`):
  - count transversal crossings of the knife line with the outline (sample
    `profile_outline`, count sign changes of the ghost field). Exactly 2 →
    silent. 4+ → warn "cut line crosses the outline at N points; each side
    will contain multiple arcs" at preview AND commit (a line crosses a
    closed curve an even number of times; only convex outlines guarantee
    exactly 2 — mirror of the 3D planar-slice warning).
  - refuse a segment knife whose clicks lie on the same straight edge (line
    collinear with the edge → the edge sits on the zero set; mirror of the
    3D "opposing normals" refusal).

### 6.2 App

- `cutter_ghost` (boundary_tool.rs:264) dispatches on the fluid root's
  dimension; knife kinds for 2D: `"segment"` (existing drag gesture) and
  `"point"` (one click places/moves the point; the split preview and any
  crossing warning show BEFORE Enter commits — the same
  preview-then-commit grammar and warning discipline as the other knives).
- Cutter clicks use `pick_outline_point` (§4.3) instead of
  `pick_surface_point`.
- The Point button appears (or is enabled) only when the fluid domain is 2D;
  Segment stays one button for both dimensions. Stencil knives
  (polygon/bezier) stay 3D-only for now and error cleanly on 2D.
- Split preview reuses the wire highlight (§5) with
  `PREVIEW_INSIDE_COLOR` / `PREVIEW_OUTSIDE_COLOR`.
- `split_boundary_region` validation already samples a near-boundary point
  cloud (prism band — works for 2D) plus the dense display points; no change
  expected, verify in tests.

## 7. Phase 4 — meshing and export verification

No code expected; verify with tests:

- `MeshableBoundaryRegion::contains` on outline-adjacent mesh points of a 2D
  case selects exactly the tagged arcs (masks are the same classifier the
  viewport uses);
- the SU2 converter path exercised by the `n0012` example maps curve regions
  to 2D line markers; add a scene fixture that tags `airfoil` vs `farfield`
  edges and asserts marker membership.

## 8. Phase 5 — tests

Mirror the existing 3D tests in `boundary_tool.rs` / kernel with a 2D fixture
(rectangle flow box minus a circle or the n0012 profile, on a placed
workplane):

- pick each rectangle edge and the obstacle outline; cut-surface (obstacle)
  patch wins when both are under the ray;
- create → select round-trip via `add_boundary_region`;
- point cut on an arc region: children partition the parent's outline
  **length** exactly (1D analog of the 3D area-partition test) and share a
  bitwise-identical split vertex;
- point cut on the closed outline: two arcs, warning present when the second
  crossing is uncontrolled (non-convex fixture → 4 crossings → warning);
- segment knife click-order independence (port of the existing 3D test);
- segment knife on the same straight edge refuses;
- classifier: base mask on outline samples matches patch expectations for the
  merged Binary profile (airfoil vs farfield attribution).

## 9. Deliberately deferred

- **Stencil knives on 2D domains** — a closed stencil in the plane selecting
  an arc set; the machinery generalizes, but segment + point cover the
  practical planar-CFD cases.
- **Zero-thickness baffles** (internal walls inside a domain) — model as thin
  3D solids subtracted from the domain, or later as a mesher-level internal
  boundary; never as a kernel sheet object.
- **Non-flat 2D manifolds.** A curved open sheet has no inside/outside, so it
  cannot be a signed-SDF domain root without breaking the exactness contract
  (CSG as min/max, sign-test classification) that every feature here relies
  on. If curved-surface flow (shallow-water on terrain, film flow on a blade)
  is ever wanted, the recommended route is NOT a new 2D object class but the
  boundary-region side of the system: a curved 2D manifold already exists in
  the kernel as the zero set of a 3D domain, boundary regions already name
  and cut exactly those manifolds, and the missing piece would be a mesher
  backend that emits a *surface mesh* of a tagged region. Planar 2D domains +
  terrain-as-data also cover the common shallow-water formulation with no
  kernel change. Keeping the classifier and cut machinery dimension-clean (as
  this plan does) is what keeps that door open.

## 9b. Domain generality (added 2026-07-19)

Boundary regions are not fluid-only: they attach to ANY marked domain
(fluid, solid, and future `DomainKind`s).

- `BoundaryRegion.domain_root` records the owning domain; regions from
  older files carry none and resolve to the fluid domain
  (`region_domain_root`). Saved as a `domain` node key in the region record.
- `add_boundary_region_in(domain_root, …)` is the explicit API the viewport
  uses (it knows which domain it picked); `add_boundary_region` auto-resolves
  from owner + patch, fluid first, for callers that don't. Splits resolve
  the root from the region. The fluid record's tag list stays in sync for
  fluid regions only; other domains attach purely by `domain_root`.
- A marked root's evolution (combine, transform wrap, solid-from-2D) moves
  its regions' `domain_root` along with the mark.
- The viewport picks against every marked domain's boundary; the nearest
  hit wins. KNOWN LIMIT: where two domains' boundaries coincide exactly (a
  nested solid's wall is also the fluid's cut surface — the CHT setup),
  ties go to the fluid-first domain order, so the solid's coincident wall
  can be tagged through the API but not yet by hover. A modifier to cycle
  coincident domains is the natural follow-up.
- Meshing attaches each domain's regions under that domain
  (`domain_boundary_entries`); the fluid path is unchanged (tag-driven,
  including `TagRef::Node` tag objects, which remain fluid-only).

## 10. Open decisions (resolve during implementation, none blocking)

- Patch id naming for merged profiles: source object names
  (`airfoil.outline`) when `sources` are resolvable, index paths as fallback.
- Whether `CURVE_PATCH_PICK_TOLERANCE` becomes `0.05 × diagonal` or a screen-
  space radius like the sketch snap (`tools.rs` snap uses screen points);
  start with scale-relative world tolerance for kernel purity, revisit UX.
- `BoundaryRegion::dimension()` hardcoded 2 — audit and possibly derive.
