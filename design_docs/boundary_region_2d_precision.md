# Precise 1D boundary regions for 2D domains

Status: IMPLEMENTED (2026-07-19). Follow-up to `boundary_region_2d.md` —
resolves its §10 pick-tolerance open decision and replaces the MVP
highlight pipeline with a patch-exact one.

## 1. Problem

The 2D boundary region MVP was too imprecise for CAD. Reported case: a
bezier surface minus a polygon — hovering showed vertex-like fragments
instead of clean edges. Measured on a bezier blob (~8×6) minus square
holes:

1. **Highlight polylines were resampled approximations.** The ribbon drew
   arcs from `placed_outline_rings` — marching-squares rings of the merged
   profile. `clean_polygon_ring` then deleted collinear vertices, so a
   square hole's ring collapsed to ~12 vertices with chamfered corners
   (measured perimeter 0.737 for a true 0.8).
2. **Segments were kept whole or dropped whole** (both endpoints had to
   pass `boundary_region_base_mask`; only cut ghosts were bisected). A 0.2
   hole edge highlighted only 0.131 (35% missing — reads as a dot); a 0.8
   edge highlighted 0.809 (bleeding past the corners); a 2.0 edge 1.927.
3. **Picking was far too loose.** `CURVE_PATCH_PICK_TOLERANCE = 0.05 ×
   diagonal` meant a hole edge was picked from 0.55 world units away; cut
   patches won *outright* over closer regular patches; edge snapping
   clamps to segment endpoints, so a large hover area near a corner
   reported the corner **vertex** as the hit.

## 2. What changed

### Patch-exact arcs — `surfaces/src/boundary_outline.rs` (new)

`curve_patch_arcs(root, patch, resolution)` builds the highlight polyline
from the patch's own analytic geometry:

- `Edge` patches: the exact segment, uniformly resampled (corners exact).
- `Outline` patches: the operand's `profile_outline` (exact polygon
  corners, on-curve bezier/ellipse samples; straight runs subdivided only
  when the midpoint verifiably stays on the operand outline), with
  operand-field contour rings as the fallback for profiles without an
  analytic outline.
- Clipping to the final boundary uses `|root| <= |operand| + band` — NOT
  `|root| <= band`. On a curved outline a chord's interior points sit off
  the zero set by the sagitta; both fields carry that offset so it
  cancels, and membership transitions exactly at the swallowing operand's
  boundary. On straight edges the operand term is zero and the test is
  exact outright. Transitions are bisected on the segment parameter (48
  iterations, deterministic — shared junction vertices are bitwise
  identical).

Junction accuracy is the chord sagitta (~6e-6 world units at display
sampling in the test fixture) — i.e. exactly as accurate as the drawn
polyline can be, since the junction lies on a chord. Straight-edge ends
(corners) are exact.

### Ribbon — `app/src/boundary_tool.rs`

`region_highlight_ribbon` draws `curve_patch_arcs` whenever the region's
`patch_id` resolves to a curve patch (`region_curve_patch`); criteria 1
and 3 are then exact by construction, and the existing exact cut-chain
bisection (`bisect_crossing`) is unchanged, so split previews inherit the
precision. Regions without a resolvable patch keep the legacy
ring+base-mask path (`region_highlight_arcs` fallback).

### Pick — `kernel/src/boundary_ops.rs`

- `pick_curve_patch` is nearest-wins: a cut surface is preferred only when
  its distance ties the nearest regular patch within the surface band (the
  2D reading of the 3D "coincident cut surface wins" rule).
- `pick_boundary_patch_with_radius` / `pick_outline_point_with_radius`
  accept an explicit world-space radius. The viewport passes a
  screen-derived one (`workplane_pixel_radius`, app/src/tools.rs): ~8 px
  at the workplane for hover, ~12 px for cutter clicks.
- Defaults without a caller radius: `CURVE_PATCH_PICK_TOLERANCE` dropped
  0.05 → 0.01 × diagonal (hover); knife-click snap keeps the forgiving
  0.05 as `OUTLINE_SNAP_TOLERANCE`.

### Scope — `kernel/src/boundary_ops.rs`

`boundary_region_scope_mask` evaluates curve-patch scopes against zero
with the slack baked into the volume (`curve_patch_scope_volume` takes
lateral + tangential pads) instead of an isotropic eval limit:

- Edge scopes: lateral band stays absolute (membership points — outline
  rings, mesh boundary nodes — lie ON the line); tangential slack past a
  corner is a hairline. Previously the scale-relative eval limit let an
  edge region claim a stretch of the neighbor edge past the corner.
- Outline scopes: unchanged semantics (closed curves have no corners); the
  band keeps the tolerance slack for curved/faceted robustness.

3D patches are untouched, and `region_patch_scope_volume` keeps its old
behavior for external callers.

### Ribbon size follows the patch owner (follow-up fix)

The highlight ribbon's `half_width` and `lift` used to scale with the
WHOLE root's bounding-box diagonal, so enlarging one operand (the flowbox
rectangle) visibly fattened the highlight of an unchanged one (the
subtracted ellipse). `region_highlight_ribbon` now sizes the ribbon from
the bounding box of `patch.owner` — the operand the region tags — falling
back to the root diagonal only when no curve patch resolves (legacy ring
path). Numerical epsilons (gradient step, degenerate-chord drop) stay on
the root diagonal; they are field tolerances, not visuals. Deferred:
constant screen-pixel ribbon width, which would need camera-dependent
overlay rebuilds (`refresh_boundary_overlays` keys only on scene/overlay
revision and selection today).

## 3. Tests

- `surfaces/tests/boundary_outline_arcs.rs`: interior-hole edges run
  corner-to-corner at full length (0.8 and 0.2 fixtures), untouched
  outline stays one closed ring, bite-fixture arcs end exactly on the
  junctions (both curves), fully-outside edges yield nothing.
- `kernel/tests/boundary_region_2d.rs`: nearest-wins between cut and edge,
  pick radius honored, edge scope stops at the corner (shared corner still
  belongs to both edges).
- `app/src/boundary_tool.rs` tests: ribbon width of the subtracted
  circle's highlight is unchanged when the flowbox rectangle triples;
  a flowbox edge's ribbon still scales with the flowbox.
- Existing suites pass unmodified.

## 4. Deliberately unchanged / deferred

- The knife crossing census and 2D validation points still use
  `placed_outline_rings` — they only need sign changes, not exact ends.
- The bezier outline remains ONE patch (`blob.outline`); per-span bezier
  patches (selecting a single span between on-curve points) would be the
  next granularity step if users ask.
- A junction "polish" below sagitta accuracy (2-variable Newton on both
  operand fields) is possible but invisible at any zoom the polyline
  itself is valid for.
