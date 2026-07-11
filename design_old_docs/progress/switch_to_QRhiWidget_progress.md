# Switch to QRhiWidget — Progress Tracker

> **Audience:** an LLM agent (or human) resuming this work in a fresh session.
> Read top-to-bottom before touching code. Single source of truth for *what's
> decided, what's done, and what's next* on the renderer-portability migration.
>
> **Status: ⛔ NOT STARTED (planning + spike only). Do not begin the port until
> the user says so AND Phase 0 passes on real hardware.**

## Goal

Replace the OpenGL viewport renderers with **`QRhiWidget`** (Qt's portable
Rendering Hardware Interface) and **drop `moderngl` + `glcontext` entirely**. One
renderer that runs natively on **Metal (macOS), Vulkan/D3D (Windows),
Vulkan/OpenGL (Linux)** — no per-API backends, no MoltenVK.

**Why:** the current viewport hard-requires OpenGL **4.6** (`moderngl.create_context(require=460)`),
which macOS caps out of (4.1, deprecated) — so the app can't render on a Mac
today. QRhi removes the OpenGL dependency from the 3D path. (GUI panels are
CPU-raster-drawn by Qt and are **unaffected** by this work.)

## Spike result (already done — the binding question is ANSWERED ✅)

Probed on PySide6 **6.11.1** (headless, software GL). Proven from Python:

- ✅ Full QRhi API exposed (`QRhi`, `QRhiBuffer`, `QRhiComputePipeline`,
  `QRhiGraphicsPipeline`, `QRhiShaderResourceBindings`, `QShader`, …).
- ✅ `qsb` / `pyside6-qsb` present; **bakes a compute shader** (SPIR-V + GLSL 430).
- ✅ `QRhi.create()` brings up a real backend (OpenGL) **even headless**;
  `isFeatureSupported(Compute) == True`.
- ✅ Baked compute shader loads + validates; **storage buffer + shader-resource-
  bindings + compute pipeline all create successfully** (QRhi accepted the exact
  interpreter pattern: bytecode-in-SSBO + shader).
- ⚠️ The headless harness could NOT run the final dispatch+readback: PySide6
  mis-binds `QRhi.beginOffscreenFrame`'s `QRhiCommandBuffer **cb` *out*-param as an
  *input* you can't construct. **This is a harness artifact, not a capability gap**
  — the real path (`QRhiWidget.render(cb)` / `initialize(cb)`) is *handed* the
  command buffer by Qt, sidestepping `beginOffscreenFrame` entirely.

**Verdict: QRhiWidget is viable from your PySide6. Green light.** One thing still
to confirm on real hardware — see Phase 0.

## Scope / blast-radius (`moderngl` usage)

| Area | moderngl call sites | Disposition |
|------|--------------------|-------------|
| `app/viewport/renderer.py` | ~50 | **DELETE** — the legacy *codegen* renderer (see synergy) |
| `app/viewport/renderers/interpreter_glsl/*` (gl_renderer, renderer, scene_buffers, sdf_evaluator, scene_cull, compute_renderer) | ~36 | **PORT** — the real interpreter renderer (chosen path) |
| `app/viewport/viewport_widget.py` | ~4 | PORT: `QOpenGLWidget` → `QRhiWidget` |
| `app/flow_sim_view.py` | ~17 | PORT: separate, simpler (particles/streamlines) |
| 7 test files (`test_interpreter_renderer`, `test_sdf_vm`, `test_sdf_profiles`, `test_sdf_interpreter`, `test_gpu_region_parity`, `test_gpu_cull`, `coregeotests/_benchmark`) | — | Adapt to QRhi, or keep behind the CPU oracle |
| `glcontext` | — | **Drops automatically** (only moderngl's context helper) |
| 22 shader files (`app/viewport/renderers/**`) | — | Bake with `qsb`. **Good:** shaders are assembled by Python string-concat (`emit_glsl_defines` + chunks), NOT GLSL `#include`, so feeding `qsb` is straightforward |

### Synergy — the legacy codegen renderer can be DELETED, not ported

The codegen renderer (`renderer.py` + `renderers/opengl/`) exists as a
**cross-vendor fallback**. **QRhi is inherently cross-vendor**, so it *subsumes
that reason to exist.* Deleting it removes the single biggest moderngl user (~50
calls) without porting a line. The migration is a **simplification**, not just a
translation. (Confirm the interpreter path covers everything the codegen path did
before deleting.)

## Decisions to settle (Phase 1 — capture answers here)

- **D-R1 — Interpreter execution model — ✅ RESOLVED: FRAGMENT path (A).**
  Final answer: raymarch in a **fragment shader straight to the render target**
  (fullscreen triangle; reads the storage buffers + camera UBO in the fragment
  stage). One pass, no compute/texture/blit. **User-confirmed SMOOTH on NVIDIA
  Vulkan**, vs. the compute path (B) which rendered correctly but LAGGED (the
  compute→blit→QRhiWidget-composite chain). QRhi *does* support fragment-stage
  storage-buffer reads on Vulkan (and Metal/D3D); the old "fragment can't compile
  the value-stack" issue is moot because `qsb` pre-bakes to SPIR-V. Shader baked +
  verified headlessly; smoothness confirmed on hardware. (History of the B attempt
  kept below.)

  --- superseded B reasoning ---
  Original choice B: raymarch in a **compute shader** → write a texture → **blit**.
  Chosen over the fragment-reads-SSBO approach (A) on **portability** grounds
  (fragment-stage storage-buffer reads "not supported on every backend") — *not*
  perf; the overhead was assumed negligible.

  **Phase 3 disproved the perf assumption.** B renders correctly but **LAGS** in
  the interactive viewport, while the **normal app (OpenGL + a FRAGMENT raymarcher
  straight to the framebuffer) is SMOOTH on the same scene + machine** (user-
  confirmed). Ruled out: GPU compute cost (trivial 3-node scene), stack depth
  (16 didn't help, matching the fragment path), event flood (throttled), and
  resolution (capped 720). Remaining suspect: the **compute → texture → blit →
  QRhiWidget-composite** chain (extra passes + a compute→graphics barrier every
  frame) vs. the fragment path's single direct pass.

  **Reconsider A (fragment direct-to-render-target).** Why A's original downside
  is mostly moot now: (1) the portability worry was chiefly about **OpenGL**,
  which is NOT an interpreter target — on **Vulkan/Metal/D3D** fragment shaders
  *can* read storage buffers; (2) the "driver chokes compiling the fragment value-
  stack" issue was a **GLSL-compile** problem, and `qsb` pre-bakes to **SPIR-V**,
  so it likely no longer applies. A is also literally the path that's smooth in
  the normal app. **To verify before committing to A:** QRhi exposes/supports
  fragment-stage storage-buffer reads on the targets, and the value-stack fragment
  shader bakes + runs via SPIR-V. *(Alternative if A also disappoints: the lag is
  QRhiWidget composite overhead → use a QWindow + swapchain instead of QRhiWidget.)*
- **D-R2 — Shader baking — ✅ RESOLVED + PROVEN (headless).** Approach: a
  programmatic GL→Vulkan transform (`app/viewport/renderers/qrhi/vulkanize.py`)
  rewrites the *assembled* GL interpreter source into Vulkan GLSL — unique bindings
  (image moved to 14, UBO at 15), loose uniforms collapsed into one unnamed std140
  block (shader bodies unchanged), `#version 450`. The **original GL chunks are
  untouched** (ModernGL path keeps working). Verified: the real interpreter
  compute shader bakes through `qsb` — **core (15.8 KB) and full features
  (49 KB)** both compile to SPIR-V+GLSL. Tests in `tests/test_vulkanize.py`
  (6, incl. the qsb bakes). Baking at runtime (startup) for now.
- **D-R3 — Delete the legacy codegen renderer?** (Recommended yes — see synergy.)
  Confirm nothing depends on it that the interpreter doesn't cover.
- **D-R4 — Test strategy:** which GL-backed tests port to QRhi vs. fall back to the
  CPU oracle (`to_numpy`)?

## Phased plan

| Phase | What | Status |
|-------|------|--------|
| 0 | **GPU-box confirmation** — `QRhiWidget` round-trip on real NVIDIA hardware: storage buffer → **compute** shader → texture → readback. | ✅ **PASSED** (NVIDIA, OpenGL backend; green/PASS via `spikes/qrhi_compute_widget_spike.py`) |
| 1 | **Decision note** — settle D-R1…D-R4. | 🟡 D-R1 ✅, D-R2 ✅; D-R3/D-R4 open |
| 2 | **Shader pipeline** — Vulkan-style GLSL + `qsb`. | 🟡 `vulkanize` + qsb bake proven; remaining: load the baked `QShader` in the renderer |
| 2.5 | **Real-scene render PROVEN (Vulkan)** — `spikes/qrhi_scene_spike.py`: real sphere bytecode → storage buffers + std140 camera UBO → vulkanized compute raymarch → texture → blit → **red sphere on screen** (user-confirmed, NVIDIA Vulkan). The full renderer core works through QRhi. | ✅ **PASSED** |
| 3 | **Port the interpreter viewport** — reusable `QRhiInterpreterRenderer` (fragment path) + `QRhiViewportWidget` (orbit/zoom). | ✅ **DONE** — renders the real scene **smoothly** on NVIDIA Vulkan (user-confirmed). Fragment raymarcher direct to render target (D-R1=A). |
| 4 | **Port `FlowSimView`** to QRhiWidget. | ⬜ todo |
| 5 | **Delete** the legacy codegen renderer (`renderer.py`, `renderers/opengl/`). | ⬜ todo |
| 6 | **Remove `moderngl` + `glcontext`** from deps; adapt the 7 tests. | ⬜ todo |

## Conventions

- Branch/commits as the user directs; keep the app runnable at each phase.
- Run on the user's GPU box for anything that executes shaders (this dev box is
  software-GL / headless — fine for API checks, not a perf or correctness oracle).
- The throwaway spike lives in scratchpad; a reusable GPU-box `QRhiWidget` spike
  is the Phase 0 deliverable (user runs it with `! .venv/bin/python <file>`).

## IMPORTANT finding — target Vulkan; it also FIXES the shader-compile lag

Two things learned from `spikes/qrhi_scene_spike.py` on the user's NVIDIA box:

1. **GL is SLOW, Vulkan is FAST** (the confirmed fact): on the GL backend QRhi
   hands the *GLSL* to the driver, which **compiles the huge interpreter shader at
   pipeline-create time** — exactly the known "viewport lag = GLSL compile on the
   GUI thread" problem ([[lag-is-glsl-compile-on-gui-thread]]). On **Vulkan**,
   `qsb` pre-baked the shader to **SPIR-V**, so there is **no driver compile** →
   fast.
2. **The earlier GL "no window" runs were almost certainly this slowness, NOT a
   crash** (correcting an earlier overclaim). The log always stopped at the step
   right before `compute pipeline create` — i.e. mid shader-compile; waiting long
   enough lets GL finish and render. There is **no confirmed evidence** that the
   `Immutable` storage buffers crashed; the `Immutable → Static` change is kept as
   best-practice (matches the working spike) but is **not** credited with fixing a
   crash. (If we ever need certainty: run the GL backend and wait it out.)

**Consequence — the migration kills two birds:** QRhi+Vulkan gives portability
(Metal/Vulkan/D3D, incl. Mac) *and* **eliminates the GUI-thread shader-compile
stall** (pre-baked SPIR-V). **Action:** the QRhi renderer **forces the Vulkan
backend** (platform-native elsewhere: Metal/D3D); OpenGL stays a slow last-resort
we are moving off anyway.

## Open questions

- Does QRhi support storage-buffer reads in the **fragment** stage on the target
  backends (NVIDIA GL/Vulkan, Apple Metal)? → drives D-R1.
- Perf parity of QRhi vs. the current ModernGL interpreter on the heavy scenes?
- `QRhiWidget` + the existing FPS overlay / QPainter HUD interplay (QRhiWidget
  composites differently than QOpenGLWidget).
- Build-time `qsb` integration (where do baked `.qsb` artifacts live / get loaded)?

## Log

- **✅ Phase 3 DONE — fragment path is SMOOTH (D-R1 = A).** Rewrote
  `renderer.py` to the fragment raymarcher (`raymarch_frag_main.glsl`): fullscreen
  triangle, fragment shader reads the 4 storage buffers + camera UBO **in the
  fragment stage**, raymarches one pixel per fragment straight to the render
  target — one pass, no compute/texture/blit. User-confirmed **very smooth** on
  NVIDIA Vulkan. This both fixes the lag and confirms QRhi supports fragment-stage
  SSBO reads. Verified headlessly first (shader bakes to SPIR-V; UBO identical to
  the compute path) per the "be accurate" note. Next: **integration** — replace
  `ViewportWidget` (QOpenGLWidget) in the app, then delete legacy + drop moderngl.
- **Phase 3 LAG — confirmed QRhi-specific; D-R1 reopened.** After applying the
  throttle, resolution cap (720), and stack-cap-16, the viewport STILL lags.
  Decisive comparison: the **normal casoCAD app (OpenGL + fragment raymarcher)
  orbits the same scene SMOOTHLY** on the same machine. So the lag is not the
  scene/machine and not GPU compute — it's the **compute→blit→QRhiWidget-composite
  path**. Next decision (D-R1 reopened): switch the QRhi renderer to the **fragment
  raymarcher direct to the render target** (one pass, like the smooth normal app),
  after verifying QRhi fragment-stage storage-buffer support + SPIR-V bake. If that
  also lags, the cause is QRhiWidget composite overhead → try QWindow + swapchain.
  **Status: paused at user's discretion — renderer core proven & committed; the
  fragment-path rewrite is the next concrete step, needs a GPU run to confirm.**
- **Phase 3 — interactive viewport (renders; lag fix pending one run).**
  Added `app/viewport/renderers/qrhi/renderer.py` (`QRhiInterpreterRenderer`) +
  `viewport.py` (`QRhiViewportWidget`, orbit/zoom, Linux→Vulkan hint) + launcher
  `spikes/qrhi_viewport_run.py`. It renders the **real von-Kármán scene** through
  QRhi/Vulkan, camera orbit confirmed by the user. **Crash fix learned the hard
  way:** never create GPU resources during `render()` — build everything in
  `initialize()` (outside the frame); render only records passes. **Lag:** it was
  laggy; applied two fixes — (a) render throttle (input marks dirty, a 60fps timer
  renders only on change, decoupling from the touchpad event flood); (b) **the
  likely real cause — the compute shader baked at IR_STACK_CAPACITY=64 (huge
  per-thread register pressure); reduced to 16 to match the proven fragment path
  (`FRAGMENT_STACK_CAPACITY`).** Also caps raymarch res at 720px (blit upscales).
  **Not yet confirmed smooth on hardware** — needs one run of the launcher. If
  still laggy after stack-cap-16, suspect QRhiWidget composite/present overhead.
- **🎉 Real-scene render PASSED on Vulkan (real hardware).**
  `spikes/qrhi_scene_spike.py` rendered an actual sphere through the full QRhi
  path (vulkanized interpreter compute shader + real scene bytecode in 4 storage
  buffers + std140 camera UBO → compute → texture → blit). Red sphere on screen,
  user-confirmed. **The renderer core is proven end-to-end.** Finding: the QRhi
  OpenGL backend segfaults building the interpreter compute pipeline; **Vulkan
  works** — so target Vulkan/Metal/D3D (see "IMPORTANT finding" above). Fixes
  along the way: storage buffers must be `Static` not `Immutable`; clear values
  need `QColor`/`QRhiDepthStencilClearValue`. Next: **Phase 3** — turn the spike
  into a real `QRhiWidget` viewport + QRhi interpreter renderer wired into the app
  (camera controls, scene updates), forcing the Vulkan backend.
- **Big discovery — the compute raymarcher already exists.**
  `app/viewport/renderers/interpreter_glsl/compute_renderer.py`
  (`ComputeInterpreterRenderer`) + `shaders/raymarch_interpreter.comp` already
  raymarch one pixel per compute invocation into an image (the Approach-B path),
  built for cross-vendor support; its own comment says "a trivial blit puts it on
  screen (GUI integration to follow)." So the port is **wrap this existing compute
  raymarcher in QRhi + a QRhiWidget**, not rewrite the raymarcher. SSBO layout:
  Nodes=0, Params=1, Children=2, Program=3 (cull 8-13); loose uniforms = program
  length + camera; output `image2D`.
- **Shader pipeline foundation done (headless).** Added
  `app/viewport/renderers/qrhi/vulkanize.py` + `tests/test_vulkanize.py` (6, green).
  The real interpreter compute shader (core + full features) transforms to Vulkan
  GLSL and **bakes through `qsb`**. D-R2 resolved. Next: the QRhi renderer
  plumbing (port `ComputeInterpreterRenderer`'s ModernGL buffer/dispatch to QRhi:
  4 storage buffers + std140 UBO + output texture + compute pipeline), then a
  `QRhiWidget` viewport that runs compute → blit-to-screen. **That step needs the
  user to run it (GPU).**
- **Phase 0 PASSED (real hardware).** `spikes/qrhi_compute_widget_spike.py` ran on
  the user's NVIDIA box: `BACKEND: OpenGL | compute supported: True`, shader baked +
  compute pipeline created, compute→texture→readback verified (`pixel(0,0).blue=200`),
  window green / console PASS. (Two trivial first-run fixes: compute shaders need
  explicit GLSL versions for qsb; `beginPass` wants `QColor` + `QRhiDepthStencilClearValue`,
  not tuples.) The Approach-B path is proven end-to-end from PySide6. The migration
  is technically de-risked; remaining open items are housekeeping (D-R2), the legacy
  deletion (D-R3), and test strategy (D-R4). Still gated on the user's go-ahead to
  start the port (after the geometry/UX work).
- **(planning)** Spike confirmed PySide6 6.11.1 exposes the full QRhi + qsb
  toolchain; QRhi inits a backend headless, compute supported, interpreter pattern
  (storage buffer + compute pipeline) creates successfully. Only the headless
  offscreen-frame dispatch was blocked by a PySide6 `beginOffscreenFrame`
  out-param quirk (irrelevant to the `QRhiWidget.render(cb)` path). Mapped the
  moderngl blast-radius and the codegen-deletion synergy. **Next: Phase 0 GPU-box
  confirmation spike** (write it, user runs on NVIDIA).
