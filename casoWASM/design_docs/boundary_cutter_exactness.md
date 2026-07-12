# Boundary Cutter Exactness — visualization and knife edge cases

Status: implemented. Round 1 (2026-07-11): exact-seam highlight + segment
knife + polyline edge cases. Round 2 (2026-07-12): area knives bounded to the
clicked sheet. Round 3 (2026-07-12): **the smooth-polyline knife was removed
as unproven** (user decision; §6 is its archived design record). Scope:
casoWASM only (`kernel`, `surfaces`, `app` crates). The Python original
shares the visualization defect this fixes but is not touched — **Python
format parity was explicitly dropped by user decision on 2026-07-12.**

Active knives: **segment** (planar slice), **polygon** and **quadratic
bezier surface** (planar point stencils).

## Why

casoCAD serves geometry to FEA/CFD meshing, so boundary-region cuts must be
exact. The region *definition* always was: `boundary_region_mask`
(`kernel/src/boundary_ops.rs`) is pure per-point SDF sign tests. But two
things around it were not:

1. **The visualization was quantized to the display mesh.** The highlight kept
   only display triangles whose all three vertices passed the mask; no
   triangle was ever split where a cut's zero set crossed it, so the rendered
   cut edge was a staircase that could be off by a full display triangle
   (flat faces are seeded at extent/24 — ~4% of the object).
2. **The knives had silent surprising edge cases** on curved and closed
   surfaces that produced mathematically well-defined but visually arbitrary
   cuts (see §3).

## Governing principle

The committed ghost must classify **exactly and only along geometry the user
saw in the preview** — never an implicit phantom extension. When that cannot
be guaranteed, the cut is refused with an actionable message; it is never
silently altered.

## 1. Exact-seam highlight (the staircase fix)

`region_highlight_mesh` in `app/src/boundary_tool.rs`:

- Criteria 1–3 of the classifier (on the Domain boundary, owner provenance,
  patch scope) are **mesh-aligned** — their boundaries coincide with display
  feature edges — so whole-triangle filtering with
  `boundary_region_base_mask` (new; `boundary_region_mask` ≡ base ∧ cuts,
  pinned by the `mask_composes_base_mask_and_cut_chain` test) stays correct.
- Criterion 4 (the cut chain) crosses triangle interiors, so the surviving
  triangles are **clipped** against each `cut_volume` SDF in chain order:
  `tessellate_for_clip` (refine long near-seam edges, ≤6 passes, target edge
  diagonal/64) then `clip_mesh_to_sdf` (marching triangles, seam vertices
  Hermite-root-found onto the exact zero set). Both were existing private
  machinery in `surfaces/src/clipping.rs`, now `pub`.
- Clipping runs on **unlifted** f64 positions; the anti-z-fight lift along
  normals (diagonal·1e-3) happens after, in `region_highlight_surface`.
- **Tolerance choice**: the overlay follows the exact `<= 0` zero set (the
  geometrically true knife); the per-point classifier keeps its
  scale-relative `tol` band for robust membership (validation, FEA/CFD
  export). They can disagree only within an O(tol) band around the seam —
  invisible at lift scale.
- **Determinism invariant**: the inside and outside split previews run the
  identical pipeline with opposite `keep_inside`, so their shared seam
  vertices are bitwise-identical → crack-free complementary previews (pinned
  by `split_previews_partition_the_region_area`). Never perturb or
  parallelize the two preview calls independently.
- The hover candidate has no cuts → the clipping stage is skipped entirely;
  its cost is unchanged.

## 2. Segment knife (planar slice, by definition)

A segment cut IS a planar slice: two clicked points plus an orientation
define a half-plane classification volume (`straight_knife`). Decisions:

| Case | Behavior |
|---|---|
| Flat face | Exact planar cut through the segment (unchanged). |
| Curved surface (endpoint normals dot < 0.95) | Cut proceeds as a planar slice, plus a **warning**: the cut follows the plane–surface intersection. |
| Endpoint normals nearly opposite | **Error** — no meaningful plane exists; cut across a flatter part of the boundary. |
| Clicks in reverse order | **Identical partition** — the plane is oriented by the **mean of both endpoint normals** (was: first click's normal only, order-dependent). Inside/Outside labels may swap; the partition may not. |

## 3. Area knives are bounded to the clicked sheet

On a **closed surface** (sphere, torus, …) the naive area knives cut a
second phantom copy the user never drew:

- The polygon / bezier stencil ghost was a `PlacedSdf2D` on a hardcoded "xy"
  workplane; `surface_selector_volume` extruded it **symmetrically** 4·span
  tall — an effectively infinite prism that punches through the body and also
  selects the **antipodal sheet** (a mirrored cut). The exact seam preview
  renders this volume faithfully; the old staircase filter merely masked it.

Fix (`stencil_knife`, `kernel/src/boundary_paths.rs`): the stencil volume is
confined to the sheet the user clicked on:

- Workplane fitted to the clicks: origin = centroid, normal n̂ = **mean click
  surface normal** — order-independent, like the segment knife. Profile in
  that plane, then a **one-sided `Extrude`**: the volume spans
  `[min click offset − ε, +4·diagonal]` along n̂
  (`ε = 0.05·footprint_diagonal + 1e-3·diagonal` absorbs the surface dipping
  below the lowest click on quasi-convex sheets; on a convex cap the interior
  bulges upward, so the bound is inactive). The antipodal sheet lies far
  below the lower bound → excluded.
- `|mean normal| < MEAN_NORMAL_MINIMUM` (0.1) → **error** (opposing normals,
  no stencil plane exists).
- Curved clicks (any normal·n̂ < `KNIFE_CURVATURE_WARNING_ALIGNMENT` = 0.95,
  shared with the segment knife) → planar-stencil **warning**.
- **Documented limitation**: on strongly non-convex sheets the bound plane
  may truncate the cut — but the truncation is visible in the preview
  (preview ≡ commit), never a silent phantom.

| Case | Behavior |
|---|---|
| Clicks on a flat face | Planar stencil cut on the face plane. |
| Clicks on a sphere/curved cap | Stencil on the mean-normal plane, bounded to the clicked sheet — no antipodal mirror copy. Warning when click normals disagree. |
| Clicks with opposing normals (mean ≈ 0) | **Error** — no stencil plane. |
| < 3 points (or even count for bezier) | **Error**. |
| Click order reversed/rotated | Same volume (centroid + mean normal are order-independent). |
| Surface dips below the lowest click inside the footprint | Cut truncates at the bound plane; visible in preview. |

## 4. Warnings plumbing

`cutter_ghost` (`app/src/boundary_tool.rs`) returns `KnifeGhost { node,
warnings }`. Warnings surface in the status line at **preview** time and are
repeated at **commit** ("Region split — …") so the user always knows what was
stored. Commit-time validation (both children must select boundary points,
`split_boundary_region`) is unchanged.

## 5. Composite ghost records (serialization format extension)

Stencil knives produce composite ghosts, so `ghost_to_json` /
`ghost_from_json` (`kernel/src/serialization.rs`) accept one **recursive**
record type in addition to self-contained leaves:

- `"extrude"`: `{ section: <ghost>, height, center_offset }`

Still never scene references. This is a casoWASM format extension — the
Python loader does not know it (parity dropped 2026-07-12). An
`"intersection"` record existed briefly for bounded polyline loops and was
removed with the polyline knife (round 3, no saved scenes used it). Note:
constructors re-normalize unit vectors on load, so round-trip fidelity is
field-exact only to last-ULP (pinned by eval-agreement in tests, ≤1e-9).

## 6. ARCHIVED: the smooth-polyline knife (removed 2026-07-12)

A smooth on-surface polyline knife shipped in rounds 1–2 and was **removed in
round 3**: in practice it was unusable and its usefulness unproven. This
section is the design record a future reintroduction should start from
(the implementation is in git history, rounds 1–2 on branch
`wasmintegration`).

- **Mechanism**: clicked points chained by discrete geodesics
  (`surface_shortest_path`: midpoint smoothing + Newton projection onto the
  boundary), wrapped in the `NormalCurtain` classification field — signed
  distance to the path, sign = nearest segment's binormal side.
  `NormalCurtain` itself still exists (`kernel/src/sdf/curtain.rs`): it is an
  independent scene payload, only its knife use was removed.
- **Edge cases solved and worth re-solving**:
  - The curtain signs EVERY point by nearest-segment binormal → open paths
    implicitly extend past their endpoints. Fix was: geodesically
    auto-extend open paths to the region border (march along the end
    tangent, re-project each step, stop when `boundary_region_base_mask`
    fails + one margin step); error if the region has no border ("close the
    polyline into a loop").
  - Closed loops (last click within 1%·diagonal of the first): snap exactly,
    no extension.
  - Self-intersecting paths rejected (nearest-segment side classification is
    inconsistent on them).
  - **Closed-surface phantom**: the curtain's sign-flip sheet extends along
    the surface normals — radial on a sphere — so a closed loop's sheet is a
    cone through the sphere center re-emerging antipodally (mirror image +
    sign speckle). Fix was: intersect the curtain with the half-space just
    below the loop's lowest point along the mean path normal (skip when the
    mean normal ≈ 0 — equator-like loops are the self-separating case).
    This required the `intersection` composite ghost record (also removed).
- **Why it still failed**: even with the sheet bound the interactive behavior
  on spheres was judged not good enough to keep; the tool needs a rethink of
  the classification field (not just bounding) before returning.

## Tests

- `app/src/boundary_tool.rs` unit tests: seam vertices exactly on the knife
  zero set (≤1e-9·diagonal); every preview triangle strictly on its side;
  inside+outside areas partition the parent (≤1e-6 rel.); cut-free hover path
  unchanged; click-order independence; curved-surface warning; opposing
  normals error.
- `kernel/tests/boundary_paths_knives.rs`: polygon and bezier stencils on a
  sphere cap have no antipodal mirror (negative samples confined to the
  clicked cap); flat-face stencil matches the drawn polygon and excludes the
  opposite face; stencil click-order independence; opposing-normals
  rejection; `extrude` ghost JSON round-trip (field agreement ≤1e-9).
- `kernel/tests/boundary_classifier.rs::mask_composes_base_mask_and_cut_chain`:
  pins the base/full mask split.

## Terminology

SDF operators (never "CSG"); viewport output is display surfaces /
tessellation ("meshing" is reserved for FEA/CFD).
