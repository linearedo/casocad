# Fix: 1D / 2D SDF objects not rendered in the QRhi viewport

**Status: RESOLVED** (user-confirmed on NVIDIA + OpenGL: 1D and 2D shapes now render).
Date: 2026-06-24.

## Symptom
- 3D primitives (sphere/box/von_karman…) rendered fine.
- Placed **1D** and **2D** SDF objects showed **neither the live preview nor the final
  figure** — even when launched with `CASOCAD_RENDER_TIER=full ./outbin/casocad-nvidia`.
- Telltale clue from the user: *the move gizmo of a 1D/2D object was visible, but the
  figure itself was not.* The gizmo is drawn by the **line-overlay** pipeline from the
  object's bounding box, so the object existed in the document — only its **raymarched
  surface** was missing.

## Investigation — three layers (first two were real but NOT the actual cause)

### Layer 1 — `CASOCAD_RENDER_TIER` was a dead env var
`renderer.py` always baked the shader with `frozenset()` (LEAN / core-only). The
`CASOCAD_RENDER_TIER` env var the user was setting **was not read anywhere** — a no-op.
With a core-only shader the optional leaf handlers are `#ifdef`'d out
(`sdf_core.glsl` `irLeafDistance`: `FEATURE_PROFILES` / `FEATURE_SWEEPS`), so every
placed 1D/2D/sweep/selector leaf returned `IR_FAR` → invisible.

### Layer 2 — tier divide removed; FULL made the only path (user decision)
Per the user's explicit request ("make FULL the only available way; if a GPU can't
compile it, it's unsupported"), the LEAN/FULL selection was **removed entirely**:
- `renderer._bake_main()` now always bakes `FULL_FEATURES`, once, in `initialize()`
  (where the GL context is current — the robust path; no `set_scene` re-bake).
- No `_desired_features`, no auto-derive, no `CASOCAD_RENDER_TIER`.
- Linux backend default switched **Vulkan → OpenGL** (`viewport._choose_api`): the FULL
  interpreter shader compiles in **~8 s on the NVIDIA GL driver vs ~456 s on its Vulkan
  driver** (same GPU). `QRHI_BACKEND` still overrides.
- Backend-aware Y handling added so OpenGL isn't upside-down (the shader had hardcoded
  Vulkan flips): `u_fb_y_up` (raymarcher, from `isYUpInFramebuffer()`) and
  `clip_y_sign` (line overlay NDC, from `isYUpInNDC()`). **Vulkan behaviour is
  byte-for-byte unchanged** (`u_fb_y_up=0` flips as before, `clip_y_sign=+1` is a no-op).
- Added diagnostics: `qrhi: initialize backend=…`, `qrhi: baked FULL shader …`, and a
  **`graphics pipeline create() FAILED`** warning (previously a silent black viewport if
  a driver can't compile FULL).

**This was necessary but NOT sufficient** — the figures still didn't show, which led to
the real cause.

### Layer 3 — THE ROOT CAUSE: sections never entered the GPU program
`build_render_ir` produces a single render root = **`tree.root`, the solid boolean
only**. Standalone placed 1D/2D **sections** live in `component_indices`, *not* inside
`tree.root`. And `emit_program` **walks only the root**. So the bytecode the shader
executes never contained the section at all:

```
before:  [box + 2D section]  program len = 1:  PUSH_LEAF(box)          ← section MISSING
after:   [box + 2D section]  program len = 3:  PUSH_LEAF(box)  PUSH_LEAF(placed_rectangle_2d)  EVAL_OP(union)
```

No shader tier could ever fix this — the geometry simply wasn't in the program. This is
why the gizmo (drawn from the bbox) appeared but the surface (raymarched from the
program) did not.

## The fix (`core/render_ir.py`)
`build_render_ir` now **unions the standalone (non-root) components onto the root** so
they reach the shader:
- Compute the set of node indices reachable from `tree.root` (`_reachable_node_indices`).
- Any `component_indices` **not** reachable (the 1D/2D sections) are collected and a
  synthetic `union` node `(root, *extra)` becomes the new single root.
- Components already inside the root (solids in a boolean) are reachable and skipped, so
  **boolean scenes are unchanged** (verified: a two-solid scene's program is identical).

Both Layer-2 and Layer-3 fixes are required together:
- **render-IR union** → the section geometry reaches the GPU program.
- **FULL shader** → the shader has the profile/sweep handler to evaluate that section leaf.

## Verification
- Headless (no GPU): the 2D and 1D section now appear in `emit_program` output as a
  `union` of root + section; a two-solid boolean scene's program is byte-identical
  (no spurious union); the FULL shader bakes through `qsb`; `BoundaryRegion`s are not
  rendered (they never become render-IR nodes).
- Test suite: **138 passed, 3 skipped** (unchanged baseline).
- **User-confirmed**: 1D and 2D shapes render in `./outbin/casocad-nvidia`.

## Files changed
- `core/render_ir.py` — `_reachable_node_indices` + union non-root components onto the
  root in `build_render_ir`. **(The actual fix.)**
- `app/viewport/renderers/qrhi/renderer.py` — FULL-only bake; backend-aware `u_fb_y_up`
  / `clip_y_sign`; checked `create()` + diagnostics.
- `app/viewport/renderers/qrhi/raymarch_frag_main.glsl` — `u_fb_y_up` uniform; the two
  hardcoded Vulkan Y-flips made backend-aware.
- `app/viewport/renderers/qrhi/viewport.py` — Linux default backend Vulkan → OpenGL.

## How to run
```
./outbin/casocad-nvidia          # NVIDIA + OpenGL; FULL compiles ~8 s first launch (driver-cached)
```
A placed 2D section is a thin flat coin (~0.004 thick) — the `{x,y}` reference view shows
it face-on. A 1D segment is a very thin tube (~0.004 radius) — renders but is faint at
normal zoom.

## Follow-up fixes (same session)
1. **Section rotate/move previews — FIXED.** Rotating a placed 2D/1D section spammed
   `only SDF objects can be rotated`, and moving one crashed with
   `'BoundaryRegion' object has no attribute 'children'` — yet the *committed* rotate/move
   was correct. Root cause: `SceneDocument.snapshot()` deep-copied nodes and **renumbered
   handles from scratch**, so a LIVE handle (e.g. a drawn section at handle 5) resolved to
   a different node in the snapshot (a BoundaryRegion). The preview tools snapshot the
   document then address it with live handles, so every preview frame hit the wrong node;
   the commit ran on the live document and was fine. Fix (`core/scene.py`): `snapshot()`
   now preserves handle identity by re-mapping the live handles onto the copied nodes via
   `deepcopy`'s memo. One fix covers move/rotate/extrude/revolve previews. User-confirmed:
   1D/2D rotate works with no error log.

## Still open
- **1D/2D thickness**: sections render at a fixed ~0.004 thickness; a thicker / overlay
  visualization would make 1D segments easier to see. Cosmetic, not a blocker.
