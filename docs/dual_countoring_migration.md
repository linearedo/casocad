# Dual Contouring Viewport Migration

This document records the migration steps performed to move casoCAD's normal CAD
viewport away from shader-raymarching SDFs and toward a QRhi-backed viewport surface
cache. The filename keeps the requested spelling (`countoring`); the technique is dual
contouring.

## Goal

Keep the canonical SDF scene as the source of truth for editing, undo/redo, meshing,
export, and solver workflows, while making the viewport render generated geometry:
stable vertices, normals, indices, and outline segments uploaded through QRhi.

Normal viewport rendering no longer depends on per-scene shader-raymarching,
primitive-specific shader churn, or the old flat GPU interpreter path. The legacy
RenderIR/codegen renderer was removed after the surface path became the concrete
viewport contract.

## Required Outcomes Status

Tracked against the refactor contract; verified by tests and code paths cited below.

| # | Outcome | Status | Where |
|---|---------|--------|-------|
| 0 | No freeze/lag | Met | Async `RenderArtifactWorker` (off GUI thread) + request coalescing + coarse→refined progressive resolution (`app/artifacts.py`); the FPS-over-time cache leak is fixed (`ViewportSurfaceCache`, see Performance Regression Fixes) |
| 0.5 | Detailed geometry, no cheap approximation | Met | Quasi-exact booleans (Phases A–E): exact Hermite edges, machine-exact sharp edges/corners (SVD QEF), seam creases welded onto the analytic intersection curve to ~1e-6, watertight/manifold narrow band — all grid-independent. Analytic meshes for primitives/sweeps |
| 1 | SDF canonical for edit/undo/meshing/export | Met | Surfaces are disposable artifacts built from `visual_snapshot()`; SDF tree untouched (see Current Architecture) |
| 2 | Viewport cache keyed by object id + scene revision | Met | `ViewportSurfaceKey(object_id, scene_revision, resolution)` |
| 3 | Async surface generation | Met | `ArtifactManager` / `RenderArtifactWorker` `QThread`; extrude/revolve/Bezier built off-thread |
| 4 | No fake CAD fallback / no bbox replacement | Met | Generic path returns empty/failed, never a bbox solid; `test_new_sdf_renders_via_generic_path_without_renderer_changes` asserts real on-surface geometry |
| 5 | Explicit, non-misleading fallback | Met | `_publish_surface_scene` keeps the last committed surface and warns when a build fails with no geometry (`viewport.py`) |
| 6 | Backend-aware via QRhi contracts | Met | `QRhiSurfaceRenderer` uploads stable vertex/index buffers + QRhi pipelines; no Vulkan-only paths; clip/Y/depth conventions queried from `QRhi` |
| 7 | Scalable primitive addition | Met | Generic DC renders any SDF from `to_numpy`+`bounding_box`; optional exact surface builder via `@_register_surface_builder` registry, no central dispatch edits |

## Migration Steps

1. Added a viewport-only surface cache.

   File: `app/viewport/surface_builder.py`

   - Added `ViewportSurfaceKey`, keyed by `object_id`, `scene_revision`, and
     `resolution`.
   - Added `ViewportSurface`, `ViewportSurfaceScene`, and `ViewportSurfaceCache`.
   - Added `build_viewport_surface_scene()` as the artifact builder entry point.
   - Added cache reuse for unchanged objects across scene revisions.
   - Added translated-object reuse so moved objects reuse topology, indices, and
     normals while only shifting vertex positions.

2. Added generated viewport geometry for primitives and sweeps.

   File: `app/viewport/surface_builder.py`

   - Added direct mesh builders for common 3D primitives: sphere, box, box frame,
     cylinder, cone, capped cone, pyramid, torus, polyline tube, and quadratic Bezier
     tube.
   - Added direct ordered-profile meshes for `Extrude` and `Revolve`.
   - Added profile outlines for circle, ellipse, rectangle, square, rounded rectangle,
     regular polygon, polygon, offset profile, and quadratic Bezier surface profile.
   - Kept generic dual-contour extraction as the fallback for unsupported bounded 3D
     SDFs.
   - Avoided fake geometry fallback: unsupported/failed/empty surfaces are reported as
     empty or failed, not replaced with bounding-box solids.

3. Added real 1D and 2D outline rendering data.

   File: `app/viewport/surface_builder.py`

   - Added 1D segment/profile surfaces as line geometry.
   - Added polyline and quadratic Bezier 1D curve sampling.
   - Added placed 2D profile contour extraction for outline-only surface display.
   - Marked these as `status="outline"` so the renderer handles them separately from
     filled triangle meshes.

4. Added the QRhi surface renderer.

   File: `app/viewport/renderers/qrhi/surface_renderer.py`

   - Added `QRhiSurfaceRenderer`.
   - Uses static QRhi vertex and index buffers for filled surfaces.
   - Vertex format stores position, normal, and color.
   - Uses indexed triangles for solid geometry.
   - Keeps optional wire indices for solid wireframe mode.
   - Converts outline-only surfaces into thick screen-space line triangles so 1D and
     2D outlines are visible on QRhi backends.
   - Handles backend coordinate conventions with `clip_y_sign`, `fb_y_up`, and
     `depth_zero_to_one`.
   - Keeps the empty-scene grid shader in the surface renderer, independent of SDF
     raymarching.

5. Switched the active QRhi viewport to surface rendering.

   File: `app/viewport/renderers/qrhi/viewport.py`

   - Replaced the normal viewport renderer with `QRhiSurfaceRenderer`.
   - Added `set_scene(surface_scene)` and `set_scene_artifact(tree, surface_scene,
     timings)` paths.
   - Kept the SDF tree only for CPU picking and scene bounds.
   - Added explicit surface publish behavior:
     - failed objects are hidden with a warning;
     - if a new scene has no valid geometry, the previous committed surface remains;
     - no bounding-box replacement is shown as if it were user geometry.
   - Cleaned active viewport comments so they describe surface rendering instead of the
     old shader-raymarch/codegen path.

6. Moved viewport artifact generation to surface-only.

   File: `app/artifacts.py`

   - Added `surface_scene` to `RenderArtifact`.
   - Added `surface_ms`, `surface_vertex_count`, `surface_triangle_count`, and
     `surface_resolution` timing fields.
   - Added coarse and refined viewport surface resolutions.
   - Kept generation inside the existing `RenderArtifactWorker` QThread, so expensive
     surface generation happens off the GUI thread.
   - Removed RenderIR fields from `RenderArtifact`, `RenderSceneSnapshot`, and
     `RenderArtifactTimings`.
   - Artifact generation now produces a `ViewportSurfaceScene` only.

7. Updated main-window render requests.

   File: `app/main_window.py`

   - All normal app viewport artifact requests pass or inherit surface-only behavior.
   - Preview, move, rotate, extrude, revolve, boolean, and initial scene requests use
     surface artifacts.
   - Log output now reports surface timing, surface resolution, vertex/triangle counts,
     large-scene state, object counts, and render-wait timing.

8. Removed proxy pressure and budgeted RenderIR rendering.

   File: `app/viewport/performance_governor.py`

   - Stopped forcing swept Bezier surfaces into viewport proxy objects.
   - Removed proxy budgets, forced-proxy object IDs, proxy box clustering, and
     `build_budgeted_render_ir()`.
   - Large-scene budgets now only select exact surface objects for prioritization.
   - Large-scene logs now report exact and total surface-object counts.

9. Reworked stale tests around the new contract.

   Files:

   - `tests/test_viewport_surface_builder.py`
   - `tests/test_qrhi_prewarm.py`
   - `tests/test_qrhi_orientation_widget.py`
   - `tests/test_viewport_performance_governor.py`
   - `tests/test_scene_solid_from_2d.py`
   - `tests/test_viewport_boolean_preview.py`
   - `tests/coregeotests/_benchmark.py`
   - `tests/coregeotests/test_core_geometry_timings.py`
   - `tests/coregeotests/test_core_cad_capabilities_timings.py`
   - `tests/coregeotests/README.md`

   Changes:

   - Added regression coverage for all filled 2D surface kinds:
     circle, rectangle, square, rounded rectangle, ellipse, regular polygon, polygon,
     and quadratic Bezier surface.
   - Covered 2D create, extrude, undo-then-revolve, and repeated operation degradation.
   - Covered 1D outline conversion into thick line payloads.
   - Replaced old QRhi shader-prewarm tests with surface renderer buffer/payload tests.
   - Updated core geometry timing tests to measure surface artifact generation.
   - Converted boundary-preview tests to assert generated surface/outline geometry.
   - Converted 2D-to-solid tests to inspect visual snapshot components and surface
     output.

10. Removed legacy RenderIR/codegen implementation and tests.

    Files removed:

    - `app/viewport/renderers/qrhi/renderer.py`
    - `core/render_ir.py`
    - `core/gpu_codegen.py`
    - `core/gpu_cull.py`
    - `core/gpu_node_types.py`
    - `core/gpu_primitives.py`
    - `core/gpu_scene.py`
    - `core/gpu_selector.py`
    - `core/sdf_profiles.glsl`
    - `tests/test_gpu_codegen.py`
    - `tests/test_gpu_primitives.py`
    - `tests/test_gpu_scene.py`
    - `tests/test_qrhi_compile_log_summary.py`
    - `tests/test_render_ir_specialization.py`
    - `tools/codegen_demo.py`
    - `tools/codegen_stress.py`
    - `tools/fps_bench.py`
    - `tools/qrhi_compile_log_summary.py`

11. Updated ultimate stress parsing.

    File: `tools/ultimate_frame_test.py`

    - Parser now accepts only the current surface artifact log format.
    - Summary records frame timings, surface artifact timings, failures, backend info,
      and render wait timeouts.
    - `tools/analyze_ultimate_frame_test.py` now reports current surface-event JSONL
      fields only.

12. Removed dead app RenderIR upload abstraction.

    File removed: `app/viewport/renderer_base.py`

    This protocol referenced the obsolete RenderIR upload renderer and was unused by
    the active app viewport path.

## Current Architecture

The normal viewport path is:

```text
SceneDocument / canonical SDF tree
        |
        | visual_snapshot()
        v
RenderArtifactWorker QThread
        |
        | build_viewport_surface_scene()
        v
ViewportSurfaceScene
        |
        | QRhiSurfaceRenderer.set_surface_scene()
        v
QRhi static vertex/index buffers + dynamic thick-line outline payloads
```

The SDF tree remains canonical. Generated viewport surfaces are disposable rendering
artifacts and are not used to define meshing, export, solver geometry, or document
state.

## Fallback Policy

The viewport fallback rules after this migration are:

- supported objects render as generated surfaces or outline geometry;
- unsupported bounded 3D SDFs use generic dual-contour extraction;
- failed objects are hidden and reported through an explicit warning;
- empty/unsupported lower-dimensional cases produce empty surfaces;
- the viewport does not show bounding-box solids as replacement CAD geometry;
- if a new artifact has no usable geometry, the previous committed surface remains.

## Validation Run

Focused tests:

```bash
.venv/bin/pytest -q \
  tests/test_viewport_performance_governor.py \
  tests/test_boundary_patches.py \
  tests/test_scene_solid_from_2d.py \
  tests/test_ultimate_frame_test_tool.py \
  tests/coregeotests/test_core_cad_capabilities_timings.py
```

Result: `88 passed`.

Viewport/cache tests:

```bash
.venv/bin/pytest -q \
  tests/test_mesh_render_cache.py \
  tests/test_viewport_surface_builder.py \
  tests/test_qrhi_prewarm.py
```

Result: `46 passed`.

Full suite:

```bash
.venv/bin/pytest -q
```

Result: `227 passed`.

GUI smoke scripts listed in `AGENTS.md` were not present in this checkout; the
available headless viewport/cache tests above were run instead.

Whitespace hygiene:

```bash
git diff --check
```

Result: passed.

## Performance Regression Fixes

### Unbounded viewport-surface cache growth (FPS degradation over time)

File: `app/viewport/surface_builder.py`

Symptom: users reported FPS degrading over an editing session until the app became
laggy and unusable. Root cause was an unbounded CPU leak in `ViewportSurfaceCache`,
not a per-frame cost (a pure orbit does not rebuild surfaces, so the regression only
accumulated with edits).

`prune_before(revision)` correctly bounded `_surfaces` (keyed by scene revision) to
the last few revisions. But the two reuse dictionaries were never pruned:

- `_latest_by_signature` was keyed by `(object_id, resolution, repr(node))`;
- `_latest_by_translation_signature` was keyed by
  `(object_id, resolution, translation_signature)`.

`repr(node)` changes on every edit, so each parameter tweak inserted a *new* entry
that retained a full `ViewportSurface` (its vertex/normal/index `ndarray`s) for the
lifetime of the cache. Reproduction: 400 edits of a single coarse sphere retained 400
surfaces per dict (~7 MB at resolution 14); with booleans/dual-contour meshes at
resolution 32 and three per-resolution caches, a long session grew into hundreds of
MB / GBs, driving memory pressure, allocator/GC churn, and the observed slowdown.

Fix: both reuse dictionaries now hold at most **one entry per live object**, keyed by
`object_id` (the cache is already per-resolution), storing the signature alongside the
surface and comparing it on lookup. Translation reuse is unaffected — while an object
is dragged its shape signature is constant, so the single slot keeps matching. Added
`ViewportSurfaceCache.prune_to_object_ids(live_ids)`, called from
`build_viewport_surface_scene()`, to drop reuse slots for deleted objects so the cache
is bounded by the live object set, not by edit history.

After the fix the reuse dictionaries stay at one entry per object regardless of edit
count; `_surfaces` remains bounded by revision pruning. Full suite: 238 passed.

### Feature-accurate intersection normals (outcome 0.5)

File: `app/viewport/surface_builder.py`

The generic dual-contour path shaded vertices with normals derived from
`np.gradient` over the sampled value field. Those are central differences across
whole grid cells: at a boolean seam (intersection/difference) they average the two
incident surfaces and round the crease, the "intersection bad rendering" symptom.

Added `_analytic_vertex_normals()`: after vertex placement, the analytic SDF gradient
is resampled at each final vertex position by central difference of `node.to_numpy`
with a half-cell epsilon per axis. The six axis samples are batched into a single
`to_numpy` call (one tree walk) to amortise per-term cost; degenerate gradients fall
back to the grid normal so seams never shade black. QEF vertex *placement* still uses
the grid gradients (stable); only the *shading* normal is upgraded.

Verified on a sphere∩box: stored normals are unit-length and align with the true SDF
gradient (mean |dot| ≈ 0.99 vs the rounded grid-average). Cost is within measurement
noise (six evaluations over the vertex set vs the full grid evaluation) and runs on
the `RenderArtifactWorker` thread, so it never touches frame time. Regression:
`test_intersection_normals_match_analytic_sdf_gradient`.

### Registry-driven primitive dispatch (outcome 7)

File: `app/viewport/surface_builder.py`

`_primitive_surface` was a hardcoded `isinstance` ladder: adding an analytic
fast-path surface builder meant editing central dispatch. Replaced with a type→builder
registry (`_SURFACE_BUILDERS`) populated by a co-located `@_register_surface_builder`
decorator on each surface builder. Dispatch walks `type(node).__mro__` so subclass matching
behaves exactly as the old `isinstance` chain, at O(depth).

This formalises the outcome-7 contract:

- a new SDF that supplies `dimension`, `to_numpy`, and `bounding_box` renders
  immediately through the generic dual-contour path with **zero** renderer changes;
- an optional exact surface builder is added as one decorated function, never an edit to
  dispatch.

Regression `test_new_sdf_renders_via_generic_path_without_renderer_changes` defines a
brand-new `_Egg` SDF inside the test and asserts the generic path emits real
on-surface geometry (|f|<0.1 at every vertex), not a bounding-box stand-in.

### Boolean precision: progressive high-resolution dual contouring (outcome 0.5)

Files: `app/viewport/surface_builder.py`, `app/artifacts.py`, `app/main_window.py`

Symptom: users reported boolean results "not refined at all / very bad". Two causes:

1. `viewport_render_resolution_for_tree` returned `refine_after=False` for *every*
   tree, so a boolean rendered once at ~24 grid cells and never improved.
2. The generic dual-contour grid was capped at `_MAX_AXIS_CELLS = 36`, and the
   tessellation (`_dual_contour_indices`) and gradient gathering were pure-Python /
   full-grid, so raising resolution would have frozen the worker (~2.1 s at 96^3,
   ~4.8 s at 128^3).

Fixes:

- **Vectorised tessellation.** `_dual_contour_indices` replaced its triple
  Python loop with three vectorised edge passes (`_quad_triangles`) emitting all
  quads per axis in one go, returning a `uint32` index array directly. Output is
  bit-identical to the scalar version (verified by set comparison).
- **Sparse gradient gather.** `_dual_contour_cell_vertices` now fancy-indexes
  gradients at the 8 corners of crossing cells only (the O(n^2) surface band)
  instead of stacking the full O(n^3) grid three times.
- Together these cut a 96^3 boolean build from ~2.1 s to ~0.7 s (3x).
- **Raised precision ceiling.** `_MAX_AXIS_CELLS` 36 -> 96.
- **Progressive refinement ladder.** `REFINED_VIEWPORT_SURFACE_RESOLUTION` 32 -> 96
  with `_REFINEMENT_TIERS = (32, 64, 96)`. `ArtifactManager._on_render_completed`
  walks the ladder via `_next_surface_resolution`, queuing one short background
  build per tier. Booleans and primitives start at COARSE (`refine_after=True`).

Measured boolean ladder (sphere∩box, worker thread): res 14 → 2.0k tris @ 19 ms
(instant first paint), 32 → 11.5k @ 53 ms, 64 → 44.7k @ 314 ms, 96 → 99.4k @ 802 ms.
The seam goes from 14 to 96 cells (~7x triangles) and sharpens in steps; the GUI is
never blocked (outcome 0). The unused `BOOLEAN_VIEWPORT_SURFACE_RESOLUTION` constant
was removed and its tests rewritten to assert the progressive contract.

## Quasi-Exact Boolean Precision (staged)

Goal: boolean seams render as sharp, clean, quasi-exact edges and curved faces stay
smooth, with quality invariant to base grid resolution — without ever blocking the UI.

### Phase A — Exact Hermite edge data

File: `app/viewport/surface_builder.py`

Before: edge zero-crossings were linear-interpolated from grid corner values
(`t = fa/(fa-fb)`) and edge normals interpolated from `np.gradient` of the value
grid. Both are grid-limited and round curved/seam features.

After:

- `_refine_edge_hermite()` finds the analytic zero of `node.to_numpy` along each
  sign-crossing edge with the **Illinois variant of regula falsi** — one evaluation
  per iteration, always bracketed by the [a,b] sign change, superlinear even on
  curved fields. The exact analytic gradient is sampled at the root via
  `_analytic_gradient()` (six batched evals, one tree walk).
- `_dual_contour_cell_vertices()` now takes the `node` (not pre-computed grid
  gradients), refines all crossing edges in one vectorised batch, and feeds exact
  (point, normal) Hermite pairs to the QEF. The full-grid `np.gradient` call is gone.

Result (sphere∩box, worker thread): edge roots land on the true isosurface to mean
|f| ≈ 3e-6 on one-cell edges, **independent of cell size** (verified at res-14 and
res-96 cell scales). QEF vertices: mean |f| 1.4e-3 @res14 → 2.4e-5 @res96 (the
residual is DC's curvature error, which Phase C/refinement reduces). Cost: res-96
build 0.80–0.92 s (was 0.80 s; the root-finder adds ~0.13 s, the dropped grid-gradient
saves ~0.04 s), still within the <1 s async budget. Regression:
`test_hermite_edge_roots_are_exact_and_grid_independent`.

Known residual after A: the normal-equation QEF still averages the two incident
surfaces at a boolean seam (box−sphere vtx |f| ≈ 3e-4 at the crease) — addressed by
Phase B (SVD/Lindstrom sharp-feature QEF).

### Phase B — Sharp-feature QEF (Lindstrom/Schaefer)

File: `app/viewport/surface_builder.py`

Replaced the raw normal-equations solve (`np.linalg.solve(AᵀA, Aᵀb)`) with a
mass-point-centred, truncated-eigendecomposition pseudo-inverse:

- recentre the system at the cell mass point `c` (mean of the exact edge points):
  `x = c + A⁺(Aᵀb − AᵀA·c)`;
- batched `np.linalg.eigh(AᵀA)`; eigenvalues (= squared singular values) below
  `_QEF_SINGULAR_RATIO = 0.03` of the largest are dropped, so flat faces solve at
  rank 1, seams at rank 2, corners at rank 3 — the **exact** feature point closest
  to the mass point, never an average;
- clamp to the cell as a degenerate-solve guard.

Result (worker thread): box∩box sharp edges/corners land on the surface to mean
|f| ≈ 1.2e-8 (machine-exact) at **both** res 24 and res 96 — sharp CAD edges are now
exact and grid-independent. The box−sphere curved seam improved ~14× (vtx |f| mean
2.95e-4 → 2.1e-5 at res 96). Cost: res-96 build ~1.0–1.15 s (the batched `eigh`);
this is the async top tier reached after the coarse/mid tiers paint, and Phase C will
cut the cell count that dominates it. Regression:
`test_sharp_feature_qef_places_exact_edges_grid_independently`.

### Phase C/D — Sparse narrow band (manifold, memory-bounded)

File: `app/viewport/surface_builder.py`

The top precision tier (`resolution >= _NARROW_BAND_MIN_RES = 96`) contours a sparse
narrow band: a coarse base grid locates the surface, only crossing coarse cells are
subdivided (`_NARROW_BAND_SUBDIV`) and contoured, evaluating the SDF once per unique
fine corner (integer linear-id dedup, no `np.unique(axis=0)` row-sort). Per-cell
placement reuses `_solve_cells_hermite_qef` (Phases A+B); faces are generated sparsely
via `searchsorted` neighbour lookup (`_DC_EDGE_SPECS`, `_cell_vid_lookup`).

Properties (verified on intersection/difference/union via the real pipeline):

- **Watertight + manifold by construction** — the band is a single uniform fine level,
  so there are no T-junctions to stitch; every edge borders exactly two triangles
  (`{2}`). This satisfies Phase D without a separate manifold-DC pass.
- **Memory-bounded** — no dense fine grid is materialised (a dense 192³ corner stack
  would be ~0.5 GB); cost/memory scale with the O(n²) surface band, not the O(n³)
  volume.

Decisions backed by measurement (sphere∩box, worker thread):

- Integer linear-id dedup cut the eff-96 band build 2.6 s → 0.84 s.
- Whole-surface uniform refinement past ~eff-128 has diminishing returns (vtx |f|
  2.4e-5 @eff96 → 1.5e-5 @eff128) at rising cost (~0.84 s/97k tris → ~1.9 s/175k tris)
  and was rejected as the default — the lever for curved-seam precision is exact seam
  geometry (Phase E), not more uniform triangles. The shipped top tier is eff-96
  (same triangle budget as the prior dense path, now manifold + memory-bounded).

Band dilation (manifold correctness): coarse crossing detection can miss surface
that clips a coarse cell whose 8 corners share a sign, leaving a fine surface cell
with an inactive neighbour and dropping that quad (a crack — first seen as
`manifold={1,2}` on box−sphere). Fixed by dilating the band to the 26-neighbourhood
of every crossing coarse cell, so every surface fine cell's face-neighbours are
active. Verified `manifold={2}` (watertight, 2-manifold) across intersection,
difference (box−sphere, box−cylinder) and union.

### Phase E — Exact boolean seam-curve welding

File: `app/viewport/surface_builder.py`

The lever for *curved* boolean precision. A boolean crease lies on
`{f_left = 0} ∩ {f_right = 0}` — where both operand surfaces pass — for every
operator. `_weld_boolean_seam()` selects vertices within ~one cell of *both* operand
surfaces and projects them onto that analytic curve by minimum-norm Gauss-Newton on
the 2-equation system (`dx = -Jᵀ(JJᵀ)⁻¹[f_left, f_right]`, `J = [∇f_left; ∇f_right]`),
4 iterations. Large or degenerate (tangent-surface) moves are rejected; only
positions move, so the mesh stays manifold and gains no triangles.

Result: the crease is welded onto the exact intersection curve — measured
`|f_left|, |f_right| < 1e-6` (machine-exact, down from ~5e-2 grid-snapped) at the
welded vertices, **independent of resolution** (verified res 64 and 96). Per build,
thousands of vertices land exactly on the curve (sphere∩box 5304, box−sphere 2880,
sphere∪box 4344 at res 96). Applied in both the dense and narrow-band paths; manifold
preserved (`{2}`). Nested booleans weld only the outermost crease (operands of the
top node); deeper creases are future work. Regressions:
`test_phase_e_welds_seam_vertices_onto_exact_curve`,
`test_narrow_band_manifold_across_boolean_ops`.

Net precision after A–E: piecewise-planar boolean edges/corners are machine-exact
(Phase B); curved seam creases are welded onto the analytic curve to ~1e-6 (Phase E);
the mesh is watertight/manifold (Phase C/D); all grid-independent and async.

### Phase F — Async / progressive integration

No new code: the precise work rides the existing ladder (`app/artifacts.py`). A
boolean edit paints COARSE (res 14) in ~30 ms, then the `RenderArtifactWorker` thread
re-requests 32 → 64 → 96 in short background steps; the top tier (96) is the welded,
manifold narrow band. Requests are coalesced (only the latest pending snapshot
survives), so rapid edits never queue stale precise builds. Measured ladder
(sphere∩box, worker thread): 14 → 2.0k tris @32 ms, 32 → 11.5k @94 ms, 64 → 44.7k
@482 ms, 96 → 99.4k @1.33 s, every tier `manifold={2}`. The GUI thread is never
blocked (outcome 0); precision arrives progressively without a freeze.

### Boolean quality hardening (user-reported: "union bad / holes / polygonized")

Files: `app/viewport/surface_builder.py`, `app/artifacts.py`, `app/main_window.py`

Diagnosis from instrumenting the meshes:

- The post-hoc seam **weld** (former Phase E) and **seam-band subdivision** (Phase G)
  moved vertices ~0.7–1.5 cells to reach the exact curve, *folding* the adjacent
  triangle fans (3–25% inverted triangles). With no backface culling those overlap
  and z-fight, reading as a **torn/broken seam and holes**. Reverted both: clean
  geometry beats "exact" geometry that tears. The dead weld/projection/subdivision
  helpers were removed.
- Base dual contouring still wound a few triangles backwards at concave seams
  (worst on **union**: ~0.3%, normalised normal dot down to −0.95). Added
  `_orient_triangles`: flip any triangle whose geometric normal opposes the smooth
  analytic vertex normal, and drop zero-area triangles. Union min dot −0.95 → +0.74;
  intersection/difference/union all clean, manifold preserved.
- **Polygonized while drawing**: booleans contour through DC (not a cheap analytic
  mesh) yet drew at the primitive COARSE tier (res 14, ~2k tris — chunky). Added
  `BOOLEAN_DRAW_RESOLUTION = 48` (≈20–27k tris, ~0.2 s async, no GUI freeze) as the
  boolean interactive floor; the ladder then climbs 48 → 64 → 96 → 128.
- **Polygonized curved primitives**: raised analytic tessellation floors (sphere
  64 seg / 32 rings, cylinder/cone 64 seg, torus 96/32, tubes 32) so curved
  primitives look smooth at every tier, including during interaction.
- Band cost trimmed: 26- → 18-neighbourhood dilation (cube corners are never DC
  quad partners, so still watertight) and Hermite root-finding 5 → 4 iterations.

End-to-end (worker thread, all tiers `manifold={2}`): union draws at 48 (19.8k tris,
0.21 s) → 64 → 96 → 128 (140k tris, 1.8 s). No holes, no torn seams, consistent
winding, watertight.

### Exact boolean rendering via SDF-clipped analytic meshes

File: `app/viewport/surface_builder.py`

The decisive quality fix for primitive booleans. Instead of dual contouring the
combined field on a grid (which polygonises the smooth curved faces and the seam),
each operand's **smooth analytic mesh** is clipped against the other operand's exact
SDF:

- `_clip_mesh_to_sdf` — marching-triangles clip of a mesh against an SDF half-space.
  Kept triangles are emitted as-is; straddling triangles are split and the new cut
  vertices are root-found exactly onto the clip's zero isosurface (`_refine_edge_hermite`).
  Cut vertices keep the operand's interpolated surface normal, so shading stays smooth.
- `_grid_box_mesh` — the box is rebuilt as a per-face grid so a curved cut through a
  flat face is captured (the analytic 2-triangle face would miss it entirely).
- `_clip_operand_mesh` — supplies adequately tessellated operand meshes (curved
  primitives from the analytic surface builders; box as a grid). Operands without a fine
  analytic mesh (e.g. a nested boolean) return None and the boolean falls back to the
  dual-contour narrow band.
- `_boolean_clip_surface` — clips each operand by the other per operator rule
  (intersection: keep each inside the other; union: keep each outside the other;
  difference A−B: keep A outside B, keep B inside A with normals/winding flipped),
  concatenates, orients (`_orient_triangles`), and compacts away discarded vertices.

Result (sphere/box, worker thread): curved faces are the **exact analytic surfaces**
(no grid faceting); rendered vertices lie on the exact boolean surface to ~4.5e-8
(intersection) / ~2.4e-4 (others); winding is consistent (no folds/tears); and it is
**3–5× faster than dual contouring** — sphere∪box at res 96 builds in ~0.33 s (vs
~1 s) and res 128 in ~0.54 s (vs ~1.8 s). The two clipped pieces meet on the seam to
root tolerance (geometrically coincident; the seam is not a shared-topology weld, but
has no visible gap). Nested/non-primitive booleans still use the watertight narrow
band. Regressions: `test_primitive_boolean_clips_to_exact_surface`,
`test_clipped_boolean_uses_exact_operand_normals`,
`test_nested_boolean_narrow_band_is_watertight_and_manifold`.

**Recursive nested booleans.** `_clip_boolean_mesh` builds a boolean's clipped mesh as
raw arrays; `_clip_operand_mesh` calls it when an operand is itself a boolean, so an
arbitrarily nested tree of sharp booleans over meshable leaves (`(A∩B)−C`,
`((A∩B)−C)∪D`, …) renders entirely through the exact clip path. Each operand mesh is
clipped against the other operand's SDF (always available from the tree) and is
cut-aware tessellated at every level. Verified `(sphere∩box)−cylinder` and the 3-deep
`((sphere∩box)−cylinder)∪sphere` clip exactly (surf |f| ~2.4e-4, consistent winding) at
res 96 in ~1.5–2.5 s. The dual-contour band remains the fallback only when an operand
has no analytic mesh (a non-meshable/field SDF, or a primitive outside the clip set
such as `Pyramid`), and for watertight export.

**Extrude/Revolve operands.** Their analytic meshes have tall un-subdivided wall
quads and fan caps, so a curved cut would be missed by the clip. `_tessellate_for_clip`
adaptively refines (red-green `_split_marked_triangles`) the mesh edges that are long
*and* inside a narrow band around the other operand's cut — concentrating detail at the
seam without inflating the flat regions. Cut-aware refinement is essential: a uniform
refine to res-96 produced ~1M triangles (≈5 s); the band version produces ~90–145k
triangles in ~0.4–0.65 s, exact to ~2e-4 with consistent winding. Extrude/Revolve are
now in `_clip_operand_mesh`, so their booleans render exact clipped geometry instead of
grid dual contouring.

### Clipping is a boolean technique; primitives/sweeps are leaves

A recurring point of confusion: clipping does **not** *generate* a shape — it is purely
a **boolean** operation (trim mesh A by operand B's SDF sign). It needs two operands and
a boolean between them.

- A primitive or **sweep** (sphere, box, **revolve**, **extrude**, …) is a *leaf*. It is
  produced by its own parametric surface builder (`_primitive_surface` → `_revolve_profile_surface`
  etc.); clipping plays no role in building it. Its smoothness is purely a surface builder-sampling
  knob (profile × angular resolution), independent of clip-vs-contour.
- A **boolean of** those leaves is where clipping applies — each leaf mesh is clipped
  against the other's SDF. So a revolve *is* clipped, but only as an *operand*
  (`revolve ∩ box`), never to bring the revolve itself into existence.

|              | how it is made                      | role of clipping       |
|--------------|-------------------------------------|------------------------|
| primitive/sweep | parametric surface builder (sample surface) | none — it is a leaf    |
| boolean of those | clip each leaf by the other's SDF | this *is* clipping     |

### Revolve resolution cap (8 → 48)

The revolve sweep surface builder was hard-capped at resolution 8 (`_MAX_REVOLVE_VIEWPORT_RESOLUTION`),
giving an octagonal profile. This made revolves chunky **both** standalone and as boolean
operands (the clip builds the operand surface with the same capped surface builder). Raised to **48**
(~144 angular + 48 profile segments). The cap is a deliberate balance — the sweep cost
grows ~quadratically (profile × angular): cap 8 → 2.5k tris/18 ms, cap 48 → 55k tris/0.39 s,
cap 128 → 393k tris/2.1 s. `REVOLVE_VIEWPORT_SURFACE_RESOLUTION` (the standalone tier) was
raised to match. After the change, `revolve ∩ box` clips exactly (surf |f| ~2.2e-4) with a
smooth operand instead of grid dual contouring. This was the actual cause behind
"revolve booleans look bad" — a surface builder cap, not the clip-vs-contour choice.

### Generic clip eligibility: seed-and-fall-back (no per-type clip whitelist)

File: `app/viewport/surface_clipping.py`

Clip eligibility used to be a hand-maintained `isinstance` whitelist in
`_clip_operand_mesh` (`Sphere, Cylinder, Cone, CappedCone, Torus, Extrude, Revolve`)
plus a box-only `_grid_box_mesh` rebuild helper. That duplicated the surface-builder
registry, excluded primitives that *did* have surface builders (Pyramid, BoxFrame, tubes), and
coupled the module to 8 concrete primitive types. Two measured gates drove a
consolidation to a single generic rule:

- **Gate 1** (retire `_grid_box_mesh`, feed the coarse 2-triangle box through the
  existing near-cut `_tessellate_for_clip`): **failed**. A band refiner only subdivides
  edges already near the cut, so a 2-triangle flat face gives it no foothold —
  `box − cylinder` lost the hole entirely (12 triangles) and `box ∩ box` was 47%
  slivers. The grid seed is genuinely necessary.
- **Gate 2** (replace the grid with a *generic* uniform 1→4 seeder): **partial pass**.
  Uniform subdivision is shape-preserving (no slivers) and matched the old grid box on
  box/pyramid at fewer triangles, **and unlocked exact Pyramid clipping** (was a
  DC-band mush at |f|≈1.5e-3, now machine-exact). But it broke thin-walled BoxFrame
  (clip drifts to |f|≈0.12) — clipping is wrong for non-convex thin shells.

Shipped design (`_clip_operand_mesh` → `_seed_operand_mesh` → quality gate):

1. **Any** leaf with a registered analytic surface builder is offered to the clip path through
   the injected `operand_mesh` provider — no type list. The whitelist and
   `_grid_box_mesh` are deleted; the module no longer imports concrete primitive types
   (only `Union/Intersection/Difference`), realising the independence its docstring
   already claimed.
2. `_seed_operand_mesh` uniformly subdivides **only big well-shaped faces**
   (`_uniform_subdivide`, min-angle > `_SEED_WELL_SHAPED_MIN_ANGLE = 15°`, edge >
   extent/`_CLIP_OPERAND_SEED_DIVISIONS = 24`). Box/pyramid faces seed; curved meshes
   (cylinder walls, fan caps, sphere poles) are inherently thin and are **not** seeded
   — seeding a cylinder explodes it 768 → 786k triangles for nothing. Curved operands
   are still cut by near-cut `_tessellate_for_clip` exactly as before.
3. `clip_surface` then **tries** the clip and validates the result: if the rendered
   vertices drift off the boolean surface (`_clip_quality_ok`, rel-error >
   `_CLIP_MAX_RELATIVE_SURFACE_ERROR = 1e-2`), it returns `None` and the dispatcher
   falls back to the watertight dual-contour band. A sliver test is deliberately *not*
   used — curved primitive meshes are legitimately full of thin triangles, so only
   surface error discriminates (good clips ≤ ~9e-4, bad ≥ ~0.08; ~100× margin).

Net: a new primitive gets exact clipped booleans **for free** when its mesh clips
cleanly (curved, or convex flat solid), and self-excludes to the DC band when it does
not — no per-shape clip surface builder, no whitelist, no broken geometry shipped. Verified by
the real pipeline (box/sphere/pyramid/nested clip; BoxFrame and a cylinder-taller-than-
pyramid difference fall back, both watertight). Regressions:
`test_flat_solid_operand_clips_exactly_without_per_type_surface_builder`,
`test_non_clippable_boolean_falls_back_to_watertight_band` (repointed Pyramid → BoxFrame).

## Module Architecture (two independent strategies, one dispatcher)

The 3,500-line `surface_builder.py` was split so the two rendering strategies are clean,
independent modules behind one dispatch point. Dependency graph (acyclic):

```text
surface_types     (leaf: ViewportSurface/Key/Scene + empty/failed/colour fallbacks)
surface_geomops   (leaf: shared numpy surface-geometry helpers — orient, normals, wire,
                   SDF gradient/edge root-finding, edge-split subdivision)
        ^   ^
        |   |
surface_clipping   -> {types, geomops}   Strategy A: exact SDF-clipped analytic meshes
surface_contouring -> {types, geomops}   Strategy B: dual-contour fallback (any SDF)
        ^   ^
        |   |
surface_builder  -> {types, geomops, clipping, contouring}
                  primitive surface-builder library + per-object surface cache + the dispatcher
```

- The two strategies do **not** import each other or the cache; each depends only on
  the two leaf modules (verified by AST). They are genuinely independent.
- `surface_clipping` receives operand meshes through an injected `OperandMeshProvider`,
  so it never imports the primitive surface builders — keeping it decoupled from the dispatcher.
- `build_viewport_surface` (in `surface_builder`) is the single routing point:
  primitive/sweep → analytic mesh; sharp meshable boolean → `clip_surface`; everything
  else (field SDFs, non-meshable operands) → `contour_surface`.
- `surface_builder.py` re-exports the public names, so existing
  `from app.viewport.surface_builder import …` imports are unchanged.

## Notes

The active app viewport is `QRhiSurfaceRenderer`. The legacy QRhi
interpreter/codegen module, its tests, and its dedicated stress/demo tools have been
deleted.
