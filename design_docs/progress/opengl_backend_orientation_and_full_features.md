# OpenGL backend: orientation fix + full 1D/2D features

**Goal (autonomous session):** deliver a fully runnable casoCAD on the OpenGL backend,
with 1D/2D object view working and the grid/axes oriented correctly. Verify by
actually running + screenshotting the app (X11, `import`/`gnome-screenshot` available).

## Context / what changed before this session
- Vulkan FULL-shader pipeline compile = 456 s (unusable); OpenGL = 8 s. So Linux
  default backend switched Vulkan -> OpenGL (`_choose_api`).
- Tier selection: optimistic FULL, `_LEAN_DENYLIST=("intel",)`; profile/selector
  sub-VM stacks shrunk (PROFILE_STACK_CAPACITY=8) to fit the per-pixel budget.

## Root cause of the "broken grid/axes" (diagnosed)
Shaders hardcode **Vulkan** conventions:
- `raymarch_frag_main.glsl` flips `gl_FragCoord.y` assuming Vulkan top-left origin.
  OpenGL `gl_FragCoord` is **bottom-left** -> the flip is now wrong -> image/grid/axes
  inverted.
- `viewport.py` ray-pick math (`suvy`) mirrors the same Vulkan y-flip.
- Line overlay (`_LINE_VERT`) writes `gl_Position` directly; NDC-Y and clip-Z differ
  between GL and Vulkan.

## Plan
1. [ ] Add backend-aware y/clip handling (uniform driven by `rhi.isYUpInFramebuffer()` /
   `isYUpInNDC()` / `isClipDepthZeroToOne()`), not a hardcoded Vulkan flip.
2. [ ] Fix raymarch ray construction + axis gizmo for GL.
3. [ ] Fix line/overlay clip space for GL.
4. [ ] Fix viewport pick math to match.
5. [ ] Test Intel Mesa + OpenGL + FULL (does it compile? decides whether "intel" stays
   in the LEAN denylist; needed for 2D/1D on the default Intel launch).
6. [ ] Run + screenshot to verify grid/axes correct and 1D/2D visible.
7. [ ] Run test suite; no regressions.

## Log
- (start) Recon done: X11, screenshot tools present. Root cause = Vulkan-hardcoded
  coordinate conventions exposed by the GL backend switch.
- FIX applied: backend-aware y/clip.
  - `raymarch_frag_main.glsl`: new `uniform int u_fb_y_up;`; flip screen_uv.y only when
    y-down (Vulkan); axis gizmo corner uses fy based on u_fb_y_up.
  - `renderer.py`: `_fb_y_up` from `rhi.isYUpInFramebuffer()`, `_line_clip_y_sign` from
    `rhi.isYUpInNDC()`; both packed into the raymarch UBO (u_fb_y_up) and line UBO
    (clip_y_sign). `_LINE_VERT` multiplies final clip.y by clip_y_sign.
  - viewport.py pick math LEFT AS-IS: it works in Qt top-left widget space (backend
    independent) and was tuned to the on-screen image, which is now identical across
    backends — so no change needed.
  - Verified: FULL shader bakes; u_fb_y_up present in UBO members; renderer imports.
- VERIFIED orientation fix: ran default (Intel/GL), log shows `fb_y_up=1 clip_y_sign=-1.0`,
  LEAN compiled in 2.3s, screenshot shows correctly-oriented 3D perspective grid + red/green
  axes + von_karman shape + corner gizmo bottom-left. Grid/axes un-broken. ✓
- DECISION on denylist: Intel Mesa + OpenGL + FULL = HANG (process state `Sl` sleeping, stuck
  at pipeline create; vs NVIDIA `Rl` actively compiling). This is the documented Mesa-Intel
  link hang. So **"intel" STAYS in `_LEAN_DENYLIST`**. Consequence: default Intel launch =
  LEAN (3D only, no 2D/1D). 2D/1D requires the NVIDIA dGPU.
- 2D/1D path = NVIDIA + OpenGL: `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
  ./outbin/casocad` → NVIDIA GL → FULL → ~8s compile.
- BUG found + fixed (ordering): the viewport seeds the scene in main_window __init__ BEFORE
  the QRhiWidget `initialize()` runs `_bake_once()` (which resolves the GPU tier). So
  `set_scene` ran with `self._features` still at the LEAN default, and the new
  `scene_fits_tier` guard wrongly SKIPPED a FULL-only (2D/1D) scene -> black viewport.
  Fix: `set_scene` now just stores `self._render_ir` and defers; `_recompute_scene()` does
  the tier check + serialize; `initialize()` calls `_recompute_scene()` after `_bake_once()`
  so the seeded scene is applied with the resolved tier. (First NVIDIA-GL FULL run with the
  2D/1D temp scene was black due to this; re-running after the fix.)
- CONFIRMED on NVIDIA-GL (post-fix): FULL tier compiles, scene renders (NOT black),
  ir_nodes=6 (incl. the 2D/1D objects), no "skipping" log. Screenshot nvidia_full2.png
  shows the scene rendering. **FPS counter read 378** → smooth, the "very laggy" Vulkan
  symptom is GONE on OpenGL (lag was Vulkan-specific, not a per-frame shader cost).
- COMPILE is one-time: first NVIDIA-GL FULL compile ~6-8 s; a relaunch logged
  `scene pipeline created in 0.0 s` — the NVIDIA driver disk-caches the compiled pipeline,
  so subsequent launches are instant. (Vulkan was 456 s every time.)
- Reverted the temp 2D/1D scene seed (core/scene.py back to original; default = von_karman
  only). Verified no leftovers.
- Test suite: **138 passed, 3 skipped** (after revert). No regressions.

## RESULT / how to run
- `./outbin/casocad` → Intel Mesa + OpenGL + **LEAN** (3D primitives). Fast, stable, grid/
  axes correctly oriented. Intel Mesa CANNOT compile FULL (hangs), so no 1D/2D here.
- `./outbin/casocad-nvidia` → NVIDIA dGPU + OpenGL + **FULL** → 1D/2D objects, tubes,
  selectors. First launch ~8 s compile (driver-cached after; then instant). Smooth (≈hundreds
  of FPS). **This is the launcher to use for 1D/2D work.**
- `QRHI_BACKEND=vulkan` still available as an override (not recommended: 456 s compile).

## Files changed this session
- `viewport.py`: Linux default backend Vulkan→OpenGL.
- `raymarch_frag_main.glsl` + `renderer.py`: backend-aware y/clip (u_fb_y_up, clip_y_sign)
  from QRhi.isYUpInFramebuffer()/isYUpInNDC() — fixes flipped grid/axes on OpenGL.
- `renderer.py`: tier selection (optimistic FULL, "intel" LEAN-denylist), profile/selector
  sub-VM stack shrink, set_scene seed-ordering fix (_recompute_scene after bake), GPU debug
  dump + compile timing logs.
- `shader_assembly.py` + `sdf_selectors.glsl`: independent profile_capacity for sub-VMs.

## Open / not addressed
- 1D/2D unavailable on the default Intel launcher (hardware: Mesa Intel hangs on FULL) — use
  casocad-nvidia. Not fixable in code without major shader-complexity reduction.
- Could persist QRhi's own pipeline cache to disk (PipelineCacheDataLoadSave=true) to make the
  first FULL compile instant even on a cold driver cache — not needed yet (driver caches it).
