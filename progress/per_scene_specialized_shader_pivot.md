# Path change: single unified shader → per-scene specialized shaders

## The decision (supersedes "one shader source" for the raymarcher)

casoCAD's QRhi viewport raymarched through **one giant FULL shader** — a single
data-driven interpreter VM whose `irNodeSDF` can evaluate *every* node type, baked
once with *every* feature (profiles, sweeps, selectors, cull). That unified shader
already sits **at the NVIDIA GL driver's link limit** (C5025). Proven, repeatedly,
on the real RTX 3050:

* Adding intersection culling to `irCellEval` (a per-cell grouped max-of-min) —
  in any form that introduces a second `irNodeSDF` call site (a struct array
  forcing a second inline, or a second item loop) — makes `pipeline.create()`
  **fail to link**: the whole viewport renders nothing (only the grid background).
* This is the same wall the earlier "compound-leaf subtree VM" hit and was
  reverted for (commit 3752ff2).

So the single-shader path is **exhausted**: we cannot add the per-pixel work that
heavy scenes (intersection ≈ 1 FPS, see the FPS benchmark) need.

## New path: specialize the shader to each scene

Compile a shader containing **only the node types and features the current scene
actually uses**. A scene of spheres bakes a tiny shader (just the sphere
evaluator) instead of the all-types monster. Consequences:

* **Removes the link ceiling** — a small `irNodeSDF` leaves ample budget, so
  grouped intersection culling (and future per-pixel work) fits easily.
* **Faster per step** — fewer branches in the hot evaluator.
* **Still interpreter-fast on edits** — the shader is keyed by the scene's *set of
  node types / features*, not its geometry. Moving, adding, or removing objects of
  a type already present changes only the data buffers (no recompile). Only
  introducing a *new kind* of primitive triggers a re-bake — rare, a few seconds
  on GL, and cached by type-set key after.

This keeps the data-driven interpreter's "no freeze on edits" win while shedding
the unified-shader link limit. It is a finer-grained relative of the removed
LEAN/FULL tiers, but driven by scene **content**, not GPU capability.

## What landed in this checkpoint (carries over, backend-agnostic)

* **Grouped-cull flattener** (`core/gpu_cull.py`): boolean trees distribute to DNF
  `max_g( min(group_g) )` + global carves, so **intersection now culls**
  (`intersection(A,B)=max(min A, min B)`), not just union/difference. Algebraic
  parity vs the boolean tree is exact (tests). Bounded at `MAX_CULL_GROUPS=4`.
* **Vectorised cull-grid binning** (`_bin_entries`): ~20× faster than the old
  Python loop; the per-edit rebuild is milliseconds at thousands of objects.
* **Grouped `irCellEval`** (`sdf_cull.glsl`): one item loop, one `irNodeSDF` call
  site, up to 4 scalar group accumulators. Correct, and the *only* shape with a
  chance of linking — but it still needs the specialized (small) shader to clear
  the driver. This is why this checkpoint does not yet render on NVIDIA.
* **Stress tests** (`tests/test_stress_scale.py`) and a **real-GPU FPS benchmark
  tool** (`tools/fps_bench.py`) — orbit FPS on the actual GPU, the only honest
  measure (headless logs showed ms builds while the GPU was at ~1 FPS).
* Interaction-quality + create-time renderer fixes (carry over regardless).

## Next steps (this pivot)

1. Derive a scene's `(features, node-kinds)` signature from its `RenderIR`.
2. Make `irNodeSDF` / feature chunks compile only the needed cases (preprocessor-
   gated by the signature).
3. Bake + cache shaders by signature; re-bake only on signature change; reuse
   pipeline/buffers otherwise.
4. Validate on the real GPU: intersection links, renders correctly, and is fast
   (`tools/fps_bench.py --scene intersection`).
