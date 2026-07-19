# Meshing Toolkit — the base layer for future meshers

Status: implementing. This document specifies the query/derivation layer any
FEA/CFD mesh generator needs, built on the `MeshableDomains` facade
(`kernel/src/meshing.rs`). The default meshers (2D + 3D) are future work and
explicitly out of scope here; this layer is what they — and user Rhai
scripts — will consume.

## 1. Scope and non-goals

In scope:

- Differential queries: batch normals, projection onto the boundary,
  curvature (`kernel/src/differential.rs`).
- Total boundary classification: "which region owns this boundary point"
  with a defined precedence, plus owner attribution for untagged boundary.
- Domain interfaces: the exact shared wall between nested marked domains
  (spec: `exact_signed_distance_field_cfd_migration_v2.md` §8).
- Exact, tagged, closed, oriented 2D boundary loops
  (`meshing/src/toolkit/loops2d.rs`).
- A sizing field with analytic gradation (`meshing/src/toolkit/sizing.rs`).
- Rhai bindings mirroring all of the above.

Not in scope: any mesh generation algorithm; element quality; solver export
modes (the connected/separated interface choice stays per-export, spec §8).

## 2. The interior-exactness contract (binding rule)

The exactness grammar guarantees a true distance only on the **negative
(interior) side** of a field. Outside, only the sign is guaranteed; the
magnitude is a lower bound. Therefore:

- **The toolkit never treats a positive field value as a distance.** Every
  distance it consumes is either a negative-side evaluation of an exact
  field, or an evaluation of a leaf-primitive field (`owner_sdf`), which is
  exact everywhere by construction.
- `project_to_zero_set` **refuses** a start point on the positive side
  (immediate `converged: false`, no iteration).
- Interface work seeds from the inner domain's interior band
  (`−h < f < 0`); each adjacent domain reaches the shared wall through its
  **own** interior field.
- Layer/band distances (y+, refinement bands) go through `owner_sdf` — a
  leaf primitive, exact on both sides — never through an operator tree's
  positive side.
- The only positive-side read anywhere is the on-boundary membership band
  `|f| ≤ tol` (§3), whose semantics predate this toolkit.

## 3. The two tolerance bands

Membership of the boundary (`{f = 0}`) is tested as `|f(p)| ≤ tol`, exactly
as the classifier has always done (`boundary_region_mask` criterion 1). The
zero set is exact from both sides (sign correctness), so the band is always
centered on the true wall. Two bands, two jobs:

| Band | Value | Job |
| --- | --- | --- |
| projection acceptance | `1e-9 ×` bbox diagonal | "this projected point landed on the wall" |
| classification default | `1e-3 ×` owner bbox diagonal (`RELATIVE_SURFACE_TOLERANCE`, `boundary_ops.rs`) | "this unprojected sample belongs to the wall" — e.g. the centroid of a straight face on a curved wall, offset by the sagitta `h²/8r` |

`classify_boundary` takes an explicit `Option<f64>` tolerance (mirroring
`boundary_region_mask`): `None` means the classification default; a caller
classifying projected vertices passes the tight band.

## 4. Differential queries (`kernel/src/differential.rs`)

Constants (relative to the field's bbox diagonal): normal step `1e-6`,
curvature step `1e-4`, zero band `1e-9`.

- `batch_normals` — central differences (`sdf_normal`), normalized.
- `project_to_zero_set` — Newton from the interior: `p ← p − n·f(p)·t`,
  ≤ 24 iterations, backtracking `t` (≤ 8 halvings) when the residual does
  not improve; success is `|f| ≤ zero_band`. On an interior-exact field the
  full step is the true correction, so convergence is essentially one step;
  backtracking only fires at C⁰ creases (equidistant seams), where the
  loop reports `converged: false` honestly rather than returning a bad
  point. Positive starts are refused (§2).
- In-plane variant for 2D domains: start and gradient are re-projected onto
  the domain plane every iteration.
- Curvature: mean curvature `H = ∇²f / 2` (3D, 7-point stencil) and
  in-plane `κ = ∇²f` (2D, 5-point stencil). Valid for points on/near the
  wall reached from the interior — project first. At creases the stencil
  returns O(1/h): read it as "refine here", not as a curvature.

Facade: `MeshableDomain::{normals, project_to_boundary, curvature}` (2D
dispatches to in-plane via the `mesh_space` frame);
`MeshableBoundaryRegion::normals` (gradient of the domain root — outward
from the domain, correct on cut surfaces) and
`MeshableBoundaryRegion::project_to_owner` (owner leaf field, exact
everywhere — layer seeding).

## 5. Total boundary classification (`kernel/src/meshing.rs`)

`MeshableDomain::classify_boundary(points, tolerance)` returns per point
`BoundaryClass { on_boundary, owner_object_id, region_index }`:

- `on_boundary`: the band test (§3) on the domain region field.
- `owner_object_id`: the controlling leaf from `evaluate_with_attribution` —
  the same owner attribution picking uses. Untagged boundary keeps this so
  a mesher can form default patches per owner leaf.
- `region_index`: the winning region among those whose
  `boundary_region_mask` accepts the point (the identical classifier the
  viewport highlights use — what is highlighted is what the mesher gets).

Precedence when several regions match: lexicographic maximum of
`(cuts.len(), patch_scoped, index)` — more knife cuts is more specific; a
patch- or direction-scoped region beats a whole-surface one at equal cuts;
the final tie goes to the later-created region, so newer refinements
override older broad tags. `regions_containing` returns all matches for
callers that want the multi-label view.

## 6. Domain interfaces (`kernel/src/meshing.rs`)

Where two marked domains are directly nested, the interface is the inner
domain's embedded additive base — the *same* `Node` that `domain_region`
subtracts from the outer region. The interface therefore **is** the surface
the outer domain was cut with: one known exact field, no numerical
differencing (spec §8).

- Derivation: each marked domain pairs with its nearest marked strict
  ancestor (direct nesting only: `sea ⊃ pipe ⊃ gas` yields sea↔pipe and
  pipe↔gas, never sea↔gas). Unequal dimensions are skipped. Sibling
  *touching* domains (adjacent, not nested) are a documented future
  extension — the current scene grammar produces adjacency by nesting.
- `MeshableInterface`: `surface_sdf` (the shared node), `project` (field
  chosen by sign: a start inside the inner domain projects through the
  inner node; a start inside the outer domain through the outer region
  node; a start in neither is refused — both sides reach the wall through
  their own exact interior distance, §2), `contains` (on the shared surface
  ∧ on both domains' boundaries, band test §3).
- Lookup is keyed, never positional: `interface_between(a, b)`
  (order-independent; unknown pairs error listing the available ones),
  `interfaces_of(name)`, `interfaces()` for iteration.
- Seeding recipe for interface meshing: the inner domain's interior band
  `−h < f_inner < 0` (exact by the grammar), projected outward; both volume
  meshes then adopt the resulting faces (owner cell on one side, neighbor
  cell on the other — MeshIR already models this).

## 7. Exact 2D boundary loops (`meshing/src/toolkit/loops2d.rs`)

`boundary_loops(domain, resolution)` returns closed, oriented, region-tagged
loops of a 2D domain:

1. Arcs per curve patch from `surfaces::boundary_outline::curve_patch_arcs`
   — already exactly on the outline with bisected junction endpoints. This
   adds the internal workspace dependency `caso-meshing → caso-surfaces`
   (downward; external dependency sets are unchanged).
2. Arcs are split where the region classification (§5) changes, bisecting
   the transition (48 iterations, as `bisect_membership`) so both sides
   share the identical junction vertex.
3. Stitching into closed loops welds endpoints within `1e-7 ×` diagonal,
   first-wins so vertices stay bitwise-shared; an unclosable gap is a
   `GeometryError` naming the patch — never a silent open chain.
4. Orientation is "material on the left": tested by evaluating the domain
   field at a small **interior** offset (negative side, §2). Outer loops
   are CCW, holes CW in `mesh_space` chart coordinates; `signed_area` is
   computed in the chart; `is_outer = signed_area > 0`.

## 8. Sizing field (`meshing/src/toolkit/sizing.rs`)

`SizingSpec { background = diag/20, min_size = diag·1e-4, gradation = 0.3,
bands, curvature_factor }`, validated on construction (unknown region names
error listing the available ones). `size_at(p)`:

- band contribution: `size + gradation × max(0, |owner_sdf(p)| − distance)`
  — `owner_sdf` is a leaf-primitive exact distance, so the gradation bound
  holds analytically (Lipschitz by construction, no smoothing grid).
- curvature contribution (opt-in): only for interior points within
  `2 × background` of the wall; project (interior start), take κ at the
  landed point; `factor/|κ| + gradation × |domain_sdf(p)|`; flat walls
  (`|κ| ≤ 1/diag`) are skipped.
- result clamped to `[min_size, background]`.

## 9. Rhai bindings

Same information, same names, thin handles: `d.normals / d.project /
d.curvature / d.classify / d.boundary_loops`; `domains.interface(a, b)` (by
names or handles, order-independent) + `domains.interfaces()`;
`itf.sdf / itf.project / itf.contains`; `sizing(domain, spec_map)` →
`size_at / sizes`. Scripts and future built-in meshers consume the identical
API — no privileged path.

## 10. Verification

New tests: `kernel/tests/differential.rs`, `kernel/tests/meshing_toolkit.rs`
(classification + interfaces), `meshing/tests/toolkit.rs` (loops + sizing),
one Rhai smoke test. Conventions: 1e-12 exact geometry, 1e-9·diag
on-surface, 1e-5 relative for stencil-limited curvature. Standing checks:
`cargo test`, `cargo clippy --all-targets`,
`cargo build --target wasm32-unknown-unknown`.
