# Async shader bake + pipeline cache (the bake path)

Goal: a structure-change edit must not FREEZE the GUI on the shader bake. The
viewport keeps showing the previous scene while the new shader compiles off the
GUI thread, then swaps in when ready.

STATUS: migration complete — async is the only bake path; the old blocking sync
path (and its CASOCAD_ASYNC_BAKE flag) has been removed.

## Diagnosis (where the freeze time lives)
Two separate costs hide behind "bake":
- **qsb** (GLSL→SPIR-V, a `subprocess.run` in `_bake`): ~0.4 s, runs on **any** thread.
- **driver pipeline compile** (`pipe.create()` / first draw): the big one —
  ~3 s OpenGL, ~65 s Vulkan on the widest shader. Bound to the **render thread**
  (QRhi), so it can't be moved to a worker.

So async-bake removes the qsb stall + keeps the UI live; the driver compile is a
residual handled by (a) defaulting to OpenGL, (b) an in-session pipeline cache so
it happens once per structure per session, (c) a best-effort disk cache.

## What was implemented (the only bake path; the sync path + flag were removed)
- **Piece 1 — async qsb bake** (`renderer.py`): `set_scene` for an unbaked
  structure kicks `_async_bake_worker` on a daemon thread (qsb only — touches NO
  QRhi), returns immediately; the old `_cg_pipe`/buffers keep rendering. On
  completion a `_BakeSignals.done` queued signal lands `_on_async_bake_done` on
  the GUI thread, which builds buffers+pipeline (frag now cached → no qsb) and
  calls the viewport's `update`. Stale bakes (superseded by a newer edit) are
  dropped via `_cg_pending_sig`; in-flight sigs tracked in `_cg_baking`.
- **Piece 2a — in-session pipeline cache** (`_cg_pipe_cache[sig]`): revisiting a
  scene structure reuses its compiled `QRhiGraphicsPipeline`, skipping the driver
  compile. Codegen SRB layout is constant across scenes, so a pipeline built with
  an earlier SRB is layout-compatible with the current one at draw.
- **Piece 2b — disk pipeline cache** (best-effort): `setPipelineCacheData` at
  init, `pipelineCacheData` saved on `closeEvent`. QRhiWidget doesn't expose
  `EnablePipelineCacheDataSave` at QRhi creation, so the SAVE may yield no data
  (harmless no-op); the LOAD still primes the driver if a blob exists.

## Validation (real GPU, OpenGL, RTX 3050)
- `set_scene(B)` on a union→alltypes structure change: **451 ms (async off) → 6 ms
  (async on)**; B renders identically.
- A→B→A revisit (cached-pipeline reuse with a fresh SRB): **bit-identical** to the
  first A (max_diff=0), no crash → piece 2a sound.
- Full unit suite green (140 passed).

## Honest residual / next
- The driver pipeline compile still hitches the GUI thread the FIRST time a
  structure is drawn (the old scene stays visible until then — no hard freeze on
  the edit action itself, but a ~3 s OpenGL hitch before the new scene appears).
  In-session cache removes it on revisit; piece 2b would remove it across launches
  IF the backend collected cache data.
- Possible follow-ups: render reduced-res / "preparing…" during that first-draw
  compile; confirm at runtime whether QRhiWidget's QRhi yields pipelineCacheData.
