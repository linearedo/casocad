# Wire spatial culling into the QRhi viewport

**Goal:** stop the FPS collapse (~10 fps) as object count grows by enabling the
existing `CULL` world-grid DDA in the QRhi fragment renderer.

## Diagnosis (why it lags)
The QRhi viewport is a brute-force fragment raymarcher with **no spatial culling**.
Per pixel: up to 160 march steps, each calling `evalSceneSDF` which runs the whole
`~2N`-instruction program (every object); +4 evals for the normal; + up to 96 evals for
the x-ray selection pass. Cost ≈ `pixels × ~160–260 × 2N` → **linear in object count**,
so FPS ≈ 1/N. Even tiny/off-screen objects are evaluated for every pixel every step.

## Existing machinery (already in the repo, just not wired to QRhi)
- `core/gpu_cull.py`: `flatten_scene` (union/difference → ADD/SUB leaf sets, else None),
  `leaf_bounds`, `build_grid` (GxGxG grid, dim=16; returns None for non-cullable or
  unbounded-additive scenes — e.g. profiles/placed sections → fall back to full VM).
- `app/.../shaders/sdf_cull.glsl` (`FEATURE_CULL`): SSBO bindings 8–13 + grid uniforms
  `u_grid_origin/cell/dim`; exposes `irMarchCulled` (DDA, clamps steps to cell exits,
  skips empty cells), `irCullDist`, `irCellEval`, `irCellCoord`, `irCellExit`.
- `core/gpu_features.py`: `CULL` is an assembly feature (not in `FULL_FEATURES`).

## Plan
1. **Frag shader dispatch** (`raymarch_frag_main.glsl`): add `u_cull_enabled`; when set,
   use `irMarchCulled` for the main march, `irCellEval` for `sceneSampleAt` (normals),
   and clamp the x-ray step to the cell exit. Bake `FULL_FEATURES | {CULL}`.
2. **Renderer host** (`renderer.py`): in `set_scene`, `flatten_scene`+`leaf_bounds`+
   `build_grid`; build 6 cull SSBOs (bindings 8–13; real grid or 1-elem dummies); bind
   them; add grid uniforms + `u_cull_enabled` to the UBO + per-frame update; upload once.
3. **Fallback**: non-cullable scene (intersection / sections) → `u_cull_enabled=0`, dummy
   buffers, exact full VM (unchanged correctness).

Binding set confirmed: fragment shader needs SSBO 0–3 (core) + 8–13 (cull) + UBO 15.

## Log
- (start) Diagnosis verified: `program_length` grows ~2N; shader evaluates the whole
  program per pixel per step. Cull machinery read end-to-end; plan above.
- DONE shader (`raymarch_frag_main.glsl`): added `u_cull_enabled`; `sceneSampleAt`
  dispatches to `irCellEval` (cull) or `evalSceneSDF` (full VM); main march uses
  `irMarchCulled` when culling; x-ray step clamped to `irCellExit` so it can't skip a
  cell. All cull refs are under `#ifdef FEATURE_CULL`, so the shader still compiles
  without it.
- DONE renderer (`renderer.py`): bakes `FULL_FEATURES | {CULL}` once. `set_scene` builds
  the grid (`_build_cull_grid` → flatten/leaf_bounds/build_grid, dim=16); 6 cull SSBOs
  (8–13) created + bound + uploaded (real grid or 4-byte dummies); grid uniforms +
  `u_cull_enabled` added to `_zero_camera` and the per-frame UBO update.
- VERIFIED headless (no GPU): FULL+CULL bakes via `qsb` (18 UBO members, all packed);
  6 spread/overlapping boxes → cull ENABLED with correct blob sizes (offsets/counts =
  16³·4 = 16384 B); a placed-2D section scene → cull falls back to the full VM (unbounded
  additive leaf), dummies bound. Test suite: **138 passed, 3 skipped**.
- Fallback behaviour confirmed in code: intersection scenes and any scene with an
  unbounded additive leaf (profiles / placed 1D/2D sections, tubes) run the exact full VM
  (`u_cull_enabled = 0`). Cull accelerates union/difference scenes of bounded 3D
  primitives — the "many objects" case.

## To verify on GPU (user)
- `./outbin/casocad-nvidia`, add many **spread-out** 3D objects → FPS should hold up far
  better than before (cull evaluates only each pixel-ray's nearby cells).
- Watch for `qrhi: ... pipeline create() FAILED` (FULL+CULL is a bigger shader; if the
  driver rejects it the log now says so instead of going black).

## GPU run results (user, NVIDIA OpenGL)
- Cull **confirmed working** for spread-out 3D objects: as boxes/spheres/box_frames were
  created across the grid (up to ~31 objects), `max_leaves/cell` stayed **2–5**, grid
  build **3–5 ms**. Count-scaling for separated 3D geometry is solved. Feels smoother.
- **Two remaining issues surfaced:**
  1. **~20 fps floor** even at low object count with the cull on. This is the FULL
     interpreter shader's *per-pixel* cost (big shader → low GPU occupancy, up to ~256
     DDA iterations, normals), **independent of object count**. Levers: dynamic
     resolution while interacting, fewer march steps, or a codegen shader (big change).
     Not a culling problem.
  2. **Adding a 2D section → `cull OFF — unbounded leaf` → very laggy.** A single placed
     section is an *unbounded additive leaf*, and `build_grid` returns None if ANY
     additive leaf is unbounded → the **whole** scene falls back to the full VM. So one
     2D object disables culling for all the 3D objects too. This is the acute regression.

## 2D-disables-cull — FIXED (bound placed sections)
`gpu_cull._leaf_bounding_sphere` now returns a real world bounding sphere for the
closed-form placed sections — **circle / rectangle / square / rounded-rectangle /
ellipse** (centre = origin + cu·axis_u + cv·axis_v on the orthonormal placement basis,
radius = profile extent + small thickness). They're gridded like any leaf, so adding such
a section no longer makes `build_grid` bail — **the cull stays on**.
- Verified: `box + circle + rectangle` → cull ON, `max_leaves/cell=3` (was `cull OFF`);
  section bounds correct (circle r=0.71, rect r=0.731). 138 passed, 3 skipped.
- The ~20 fps base-cost floor (heavy FULL shader) is unrelated and still open; dynamic
  resolution while interacting is the lever there (not done).

## All geometry kinds now culled (authoritative bounding_box)
Per-kind param formulas are fragile (a wrong offset silently culls geometry away), so
instead `build_render_ir` now captures each geometry leaf's **world bounding sphere from
the SDF node's `bounding_box()`** (transform applied to the box corners) onto
`RenderIRNode.bound`; `gpu_cull.leaf_bounds` uses it (falling back to the param formula
for hand-built IRs). `bounding_box()` is defined on every SDF node, so this bounds
**every** geometry kind correctly — present and future — including the ones that used to
disable the cull: polyline/bezier sections, **tubes, extrude, revolve, profile-graphs,
placed 1D**.
- Verified: polyline/bezier tube, placed_profile_1d, placed_bezier_curve_2d,
  extrude_profile_2d, revolve_profile_2d all report finite bounds and `cull ON`.
  138 passed, 3 skipped.
- Correctness rests on `bounding_box()` being conservative (it's the same bound used for
  framing/meshing). Only `intersection` operators and genuinely overlapping objects remain
  un-culled (the former by the flat ADD/SUB design; the latter is inherent).

## Intersection cull — ATTEMPTED, REVERTED (driver compile limit)
Tried to cull intersection (and subtractive-context difference) as a "compound leaf"
evaluated by a new 3D subtree VM `irEvalSubtree` (commit reverted as e27f9bb). It passed
the offline `qsb` bake and all CPU checks, but the **NVIDIA GL driver failed to LINK** the
shader: `error C5025: lvalue in assignment too complex` (×13, in irEvalSubtree *and* now
in pre-existing struct-array writes) → `pipeline create() FAILED` → nothing rendered. The
new `Sample vstack[]` work-stack (a third indexed array-of-struct VM, after evalSceneSDF +
the profile VMs) tipped the driver's complexity budget for dynamically-indexed struct
assignments. The checked `create()` surfaced it loudly instead of a black viewport.
- **Reverted** → back to: all geometry + union/difference culled; **intersection renders
  via the exact full-VM fallback** (correct, just not accelerated).
- Lesson: the offline bake does NOT predict the NVIDIA driver's link limits; shader
  additions can't be validated here, only on the GPU. A retry would need a lighter subtree
  eval (parallel `float`/`uint` arrays like `evalProfileSDF`, smaller stack) — still
  unverifiable locally.

## Notes / limits
- Speedup scales with spatial separation: objects clustered in one spot share cells, so
  the per-cell leaf list stays large (still correct, just less culling).
- Grid is rebuilt per `set_scene` on the CPU (cheap: a few ms for a 16³ grid).
- The FULL+CULL shader is near the NVIDIA GL driver's link complexity limit — adding more
  per-cell VM machinery risks `C5025 lvalue too complex` (see the intersection attempt).
