# Full migration to thin data-driven codegen (retire the interpreter VM)

Goal: codegen becomes the viewport's renderer; the bytecode interpreter VM is
deleted. Rationale + compile measurements: `codegen_vs_vm_compile_investigation.md`.

Honest scope note: this is a multi-step migration, not a single-commit swap. The
VM renders feature nodes (profiles/sweeps/selectors) and non-flattenable booleans
that codegen does not emit yet; deleting it before those are ported would drop
features the user requires. Each port is validated on the real GPU before the VM
fallback for it is removed.

## Plan
1. **Wire codegen into the live viewport** as a selectable path, VM as fallback
   for nodes codegen can't emit. (env `CASOCAD_CODEGEN=1` during migration.)
2. **Port each feature into codegen**, shrinking the fallback:
   - core 3D primitives + union/difference/intersection (DNF)  — DONE (emitter).
   - placed 2D sections (circle/rect/square/rounded-rect/ellipse) — analytic.
   - polyline/bezier tubes (sweeps).
   - extrude/revolve + profile sub-graph (the nested 2D profile VM) — hardest.
   - region selectors (Layer 2).
   - non-flattenable booleans (carve-under-union) → signed-literal DNF.
3. **Delete the VM** once the fallback is provably never hit (a guard logs any
   fallback; when it stays silent across the test scenes + real use, remove
   sdf_core VM bits, emit_program, the bytecode buffers).

## Status / log
- [x] **Step 1 DONE + validated on real GPU.** Codegen path lives inside
  `QRhiInterpreterRenderer`, gated by `CASOCAD_CODEGEN=1`: `set_scene` routes
  `cg_supported()` scenes (core primitives + flattenable booleans) to a codegen
  pipeline (shader cached by structure signature, re-baked only when the kind-set
  changes; data buffers rebuilt per scene), `render()` draws the codegen pass +
  the shared gizmo overlay; everything else falls back to the VM. A fallback log
  fires whenever the VM is used, to measure step-3 readiness. Verified: the live
  `QRhiViewportWidget` (CASOCAD_CODEGEN=1) renders an intersection via codegen
  (cg_active=True, screenshot correct); default VM path unchanged (169 tests).
- [ ] Step 2 feature ports — IN PROGRESS:
  - [x] **Placed 2D analytic sections DONE + GPU-validated** (`placed_circle_2d`,
    `placed_rectangle_2d`, `placed_square_2d`, `placed_rounded_rectangle_2d`,
    `placed_ellipse_2d`). Added their analytic SDF to `_LEAF_GLSL` (+ the
    `irExactEllipseDistance` helper, gated to ellipse), mirroring sdf_profiles.glsl;
    `supported()` now accepts them. qsb-compiles for circle/ellipse/rect/rounded/
    square scenes; the demo `tools/codegen_demo.py sections` renders them through
    codegen on the real GPU (flat discs/quads on the grid — screenshot correct);
    8 codegen tests + 170 total green. These no longer hit the VM fallback.
  - [x] **Polyline/bezier tubes (sweeps) DONE + GPU-validated** (`polyline_tube`,
    `bezier_tube`). Points live inline in params (pc=(param_count-3)/3, then
    radius/inner/flat_caps), so no child buffer needed — `leafDist` reads
    `node.param_count`. Added the SDF + helpers (irSegmentDistance3D,
    irQuadraticBezierDistance3D, irFlatCappedSegmentTubeSDF3D, irSafeDirection,
    irTubeSDF, irFlatTubeSDF) mirroring sdf_sweeps.glsl. qsb-compiles; renders on
    the real GPU (`tools/codegen_demo.py tubes` — bent + curved tubes, correct).
    9 codegen tests + 171 total green. (Codegen has no spatial cull yet — tubes
    are brute-forced like all leaves; fine for correctness, optimize later.)
  - [~] **extrude/revolve + profile sub-graph (2c) — STARTED, split in two:**
    - [x] **2c-i: placed 2D open curves DONE** (`placed_polyline_2d`,
      `placed_bezier_curve_2d`) — points inline in params @12+, analytic (no
      sub-VM); added with the 2D helpers irSegmentDistance2D /
      irQuadraticBezierDistance. qsb-compiles; 171 tests green.
    - [x] **2c-ii: the profile sub-VM DONE + GPU-validated** (placed_profile_2d,
      extrude_profile_2d, revolve_profile_2d, placed_profile_1d). The nested 2D/1D
      profile stack-VM is **sliced out of sdf_profiles.glsl by content markers**
      (`irSegmentDistance2D` → before `irProfileLeaf`, cached) and embedded ONLY
      when a profile kind is present, de-duping the 3 helpers it shares
      (irExactEllipseDistance / irSegmentDistance2D / irQuadraticBezierDistance).
      Added the `Children` buffer (binding 2) to the codegen shader + renderer SRB
      (data from serialize_scene.children_bytes). `supported()` /
      `scene_structure_signature()` now key off the **DNF leaves** (the 3D scene
      leaves), so profile sub-graph nodes don't gate or bloat the signature.
      Validated: extrude/revolve scenes qsb-compile and render on the real GPU
      (`tools/codegen_demo.py profiles` — extruded solids, correct). 172 tests
      green. CONFIRMS the thesis: the sub-VM's local stacks (the shape that blew
      the FULL shader's link limit) compile fine in the small specialized shader.
  - [x] **Carve-under-union (2e) DONE + GPU-validated** via a codegen-only
    SIGNED-literal DNF (`core.gpu_codegen.flatten_signed`): `union` converts a
    child's carves into NEGATIVE singleton groups before the min cross-product, so
    `union(A-B, C) = max(min(+A,+C), min(-B,+C))`. The VM's `flatten_scene` /
    spatial grid is UNTOUCHED (it can't cull negative-in-group literals; codegen
    brute-forces, so signed literals are exact). The sign rides bit 8 of the
    add-item group field; the map() add loop negates `d` for negative literals.
    Exact CPU parity vs the tree (incl. nested + unions of carves); renders on the
    real GPU (`tools/codegen_demo.py carveunion` — carved spheres unioned, holes
    visible). Cap: a union of >~2 independent carved solids exceeds
    MAX_CULL_GROUPS (DNF cross-product) and correctly falls back to the VM. 173
    tests green. (Also fixed a latent demo bug: the Children buffer was unbound in
    the standalone demo — now bound, so extruded circles render as cylinders.)
  - [x] **Region selectors (2d) DONE + GPU-validated** — the LAST port. Selectors
    are a separate post-process list (not in the boolean tree), so geometry
    flattens normally; an embedded FLOAT-only 3D subtree stack-VM (`evalSubtreeDist`
    over leafDist — no Sample needed, only `.dist`) + `regionAt()` tags region_id
    where a point is inside the selector volume (+ optional scope), matching
    sdf_selectors.glsl. Selector node indices ride a new `Selectors` buffer
    (binding 3); a no-op `regionAt` stub when no selector is present. The raymarch
    main tints tagged surfaces. Renders on the real GPU
    (`tools/codegen_demo.py selector` — spheres with red region patches where
    inside the volume). 174 tests green.

**STEP 2 COMPLETE — codegen covers every feature.** The VM fallback log should now
stay silent across all scenes. Ready for step 3 (delete the VM) after a soak.

## Post-STEP-2 hardening — group cap + local-carve representation (after revert)
Context: the post-2 experiment ("specialize the VM") regressed because the VM is
coupled to OpenGL (its dynamic loops/stack compile pathologically on Vulkan);
codegen is the only path that runs on Vulkan, so we returned to STEP 2 COMPLETE
and hardened codegen instead.
- [x] **Group cap is now codegen-local + data-driven.** Replaced the 4 unrolled
  `g0..g3` accumulators with a fixed `float g[CG_MAX_GROUPS]` (=16) array + a
  scatter-by-`gid` term loop and a combine loop bounded by the runtime
  `u_group_count`. `CG_MAX_GROUPS` is INDEPENDENT of the VM's `MAX_CULL_GROUPS=4`
  (which `sdf_cull.glsl` still unrolls in lock-step — untouched). Compile stays
  O(1)/fixed-size; structural edits up to 16 groups change a uniform, no re-bake.
  GPU-validated: `g[16]` did NOT cost FPS (orbit equal vs VM on the RTX 3050).
- [x] **`flatten_signed` → `flatten_terms` (local per-term carves).** The old
  signed-literal DNF distributed carve-under-union via a CNF cross-product, so a
  union of K distinctly-carved solids cost **2**K** groups (K=3 already bailed to
  the VM). Now a *term* is `(leaf, local carves)` and a carve rides its term
  locally, so a union of K carved solids is **1 group / K terms (linear)** —
  measured: K=50 supported in one group, where K=3 used to fall back. New buffers
  Terms@4 (uvec4 leaf,gid,carve_offset,carve_count) + Carves@5; the global Subs
  buffer and the bit-8 sign flag are gone. CPU parity vs the recursive tree (200
  random pts over mixed union/carve/intersection) + qsb-compile clean. 176 tests.
  `CG_MAX_GROUPS` now caps INTERSECTION breadth (cross-product of union groups),
  not carved-solid count.
- [x] GPU-render-validated the term path at K=16 carved solids (user confirmed
  `CASOCAD_CODEGEN=1 tools/codegen_demo.py /tmp/cg16.png carveunion 16`).
- [x] **Adaptive group cap DONE — codegen is now total for the LINEAR case.**
  `group_capacity(ir)` buckets the scene's flattened group count to
  `_CG_GROUP_BUCKETS=(8,16,32,64,128,256)`; `emit_map_glsl` bakes `float g[cap]`
  at that size, and the renderer's bake key is now `(kinds, cap)` so a bigger
  bucket bakes a distinct variant (edits that nudge the count, 20→21, stay in one
  bucket → no re-bake). Verified: a wide intersection (convex-polytope shape) of N
  planes flattens to N groups and bakes the right bucket — N=17→g[32], N=40→g[64]
  (qsb-compiles clean), N=100→g[128], N=256→g[256]; N>256 bails (flatten_terms
  caps at `CG_GROUP_CEILING=256`). 176 tests green.
- [x] **Spatial culling for codegen DONE (real-GPU validation pending).** The
  codegen path brute-forced every term at every pixel — single-digit FPS at scale
  (stress: carveunion 200 = 4.2 FPS, polytope 64 = 7.2, mixed 400 CRASHED). Ported
  the VM's world-grid DDA to the term model: `core.gpu_cull.build_term_grid` bins
  each term by its positive-leaf bound (carves only shrink it, so the solid bound
  is conservative; items are TERM indices, not the VM's packed leaf/group), and the
  shader gained `cellDist` + `cgRayGrid/cgCellCoord/cgCellExit` + a DDA march in
  main gated by `u_cull_enabled` (grid buffers @6/7/8, mirroring sdf_cull.glsl).
  Measured: carveunion 200 bins to ~7.5 terms/cell (max 29) vs 200 brute-force —
  ~25× less per-step work. Grid is None (cull off, brute-force kept) when any term
  leaf is unbounded. 177 tests (incl. a grid-binning-invariant test); cull shader
  qsb-compiles. NOTE: hand-built stress scenes with tubes/profiles lack node.bound
  so they stay brute-forced; real-app scenes have authoritative bounds and cull.
- [ ] **Policy for the MULTIPLICATIVE case (>256 groups).** A union of many
  intersected parts multiplies group counts and can't be arrayed away — it bails
  today (supported=False → VM fallback). Before the VM can be fully deleted, pick
  one: (1) warn-as-unsupported, (2) a dormant minimal VM as a correctness
  backstop, or (3) a codegen tree-eval shader (no flatten) for these rare scenes.
  Not blocking the realistic scenes — this is the last residual case.
- [ ] Step 3 VM deletion — blocked on step 2 coverage (delete only when the
  fallback log stays silent across the feature scenes).

HONEST NOTE: steps 2 and 3 are real engineering (porting ~1000 lines of profile/
sweep/selector SDF, incl. a nested 2D interpreter) and must be done + GPU-validated
incrementally — they are deliberately NOT rushed into this run.

## Coverage matrix (codegen)
| node group | VM | codegen | notes |
|---|---|---|---|
| sphere/box/cylinder/cone/capped_cone/box_frame/pyramid/torus | yes | **yes** | leafDist |
| union / difference / intersection (flattenable) | yes | **yes** | term-DNF loops |
| carve-under-union | yes | **yes** | local per-term carves (flatten_terms); linear in carved solids |
| placed 2D sections (circle/rect/square/rounded/ellipse) | yes | **yes** | analytic, in leafDist |
| polyline/bezier tubes | yes | **yes** | points inline in params; leafDist loop |
| placed 2D open curves (polyline/bezier) | yes | **yes** | inline points, leafDist |
| extrude / revolve / placed_profile (2D+1D) | yes | **yes** | embedded profile stack-VM (sliced from sdf_profiles.glsl) |
| region selectors (Layer 2) | yes | **yes** | embedded float subtree-VM + regionAt |
