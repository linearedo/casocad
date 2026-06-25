# Shader specialization integration progress

Audience: future agent or maintainer resuming the viewport compile-latency work.

## Context

The QRhi codegen renderer already specializes shaders by scene structure, but a
few common 2D UI shapes still went through the generic profile path:

```text
PlacedSDF2D(profile=BezierSurfaceProfile(...))
  -> RenderIR kind="placed_profile_2d"
  -> shader calls evalProfileSDF(child)
```

That generic path was originally useful for feature coverage: one profile
evaluator handled placed profiles, extrude/revolve profiles, offset profiles,
binary profile graphs, polygons, and Bezier surfaces. It is correct, but it is
too broad for first-use interactive drawing.

The user observed a freeze on first creation of a 2D Bezier surface. Logs before
this pass showed, on Vulkan:

```text
qrhi: pipeline driver-compiled in 6.14s
```

The expensive part was not `RenderIR` creation and not `qsb`; it was
`QRhiGraphicsPipeline.create()` compiling the generated shader variant.

## What changed in this pass

Implemented direct render/codegen leaves for common filled 2D profile shapes that
were still using the generic profile evaluator:

- `placed_polygon_2d`
- `placed_bezier_surface_2d`

Files changed:

- `core/gpu_node_types.py`
  - Added both new kinds at the end of the node registry, preserving existing
    node type codes.
- `core/render_ir.py`
  - `PolygonProfile` and `RegularPolygonProfile` now build a direct
    `placed_polygon_2d` `RenderIRNode`.
  - `BezierSurfaceProfile` now builds a direct `placed_bezier_surface_2d`
    `RenderIRNode`.
  - `placed_profile_2d` remains for generic profile graphs such as
    `OffsetProfile` and `BinaryProfile`.
- `core/gpu_codegen.py`
  - Added direct `leafDist` branches for polygon and Bezier surface leaves.
  - Added only the helper functions needed by those direct branches:
    segment distance, segment ray crossing, quadratic Bezier distance, and
    quadratic Bezier ray crossing.
- `tests/test_gpu_codegen.py`
  - Added coverage that direct polygon/Bezier-surface variants do not emit
    `evalProfileSDF`.
- `tests/test_render_ir_specialization.py`
  - Added coverage that CAD `PlacedSDF2D` polygon and Bezier-surface profiles
    lower to the direct render leaves.

## Validation

Focused tests:

```bash
.venv/bin/pytest -q tests/test_gpu_codegen.py tests/test_gpu_scene.py tests/test_render_ir_specialization.py
```

Result:

```text
24 passed
```

Full headless test suite:

```bash
.venv/bin/pytest -q
```

Result:

```text
165 passed, 3 skipped
```

Whitespace check:

```bash
git diff --check
```

Result: clean.

A direct qsb compile probe for a scene containing `placed_polygon_2d` plus
`placed_bezier_surface_2d` also succeeded.

## User-measured result after this pass

OpenGL run:

```text
bezier_polycurve pipeline driver-compiled in 0.07s
bezier_surface pipeline driver-compiled in 2.06s
```

Vulkan run:

```text
bezier_curve pipeline driver-compiled in 0.73s
bezier_surface pipeline driver-compiled in 1.98s
```

This is materially better than the earlier ~6.14s Bezier-surface compile, but
still too slow for ideal CAD interaction.

## Important interpretation

Backend choice is not the whole remaining problem. Vulkan is active in the
latest measurement, yet first-use Bezier-surface pipeline creation is still about
2 seconds.

`qsb` precompiles GLSL into portable shader bytecode, but the graphics driver
still compiles/optimizes the pipeline into device-specific machine code during
`QRhiGraphicsPipeline.create()`.

Also, the codegen renderer builds one shader variant for the whole visible scene.
When the user adds a Bezier surface after existing objects, the compiled variant
is not only:

```text
placed_bezier_surface_2d
```

It is more like:

```text
box + cylinder + placed_bezier_curve_2d + placed_bezier_surface_2d
```

so compile time is the combined scene-signature cost.

## Current design stance

The renderer should not try to specialize every possible CAD graph. That would
explode into too many variants.

Preferred split:

- Direct specialized leaves for common primitive/simple UI shapes.
- Generic `placed_profile_2d`, `extrude_profile_2d`, and `revolve_profile_2d`
  paths for arbitrary profile graphs and composition.

Already specialized or mostly direct:

- core 3D primitives: sphere, box, cylinder, cone, capped cone, box frame,
  pyramid, torus
- tubes: polyline tube, Bezier tube
- placed 2D primitives: circle, rectangle, square, rounded rectangle, ellipse,
  polyline curve, Bezier curve
- now: polygon and Bezier surface

Still intentionally generic:

- offset profiles
- binary profile graphs
- arbitrary extrude/revolve profile graphs
- selectors

## Recommended next steps

1. Add richer shader compile telemetry. DONE after the initial specialization
   pass.
   Logs now include signature, group-capacity bucket, profile mode, source byte
   count, qsb time, backend, and pipeline compile time.

2. Prewarm variants when draw tools are selected. FIRST PASS DONE.
   For example, when the user selects “Bezier Surface 2D”, start baking/creating
   the likely shader variant while the user is clicking points. This is probably
   the highest-value UX mitigation for the remaining ~2s first-use stall.

3. Consider further common-case variants only if telemetry justifies them.
   Possible candidates:
   - direct `extruded_circle_2d`
   - direct `extruded_rectangle_2d`
   - direct `revolved_circle_2d`

4. Keep the generic profile path as the correctness fallback for composed
   profiles.

## Notes

The scene tree/Properties panel UI naming was also improved during this session:
placed 1D/2D containers now display the user-facing profile kind, e.g. a
`PlacedSDF2D(profile=RectangleProfile)` named `floor` shows kind `rectangle`
instead of `placed_sdf_2d`.

## Follow-up pass: telemetry + draw-tool prewarm

Additional changes after the first direct-leaf specialization:

- `app/viewport/renderers/qrhi/renderer.py`
  - Added detailed codegen bake and pipeline logs:
    - sorted scene-signature kinds
    - group-capacity bucket
    - profile mode (`simple` / `full`)
    - emitted source byte count
    - qsb compile time
    - QRhi backend name
    - driver pipeline compile time
  - Added `prewarm_for_tool(render_ir, tool_kind)`.
  - Added a synthetic prewarm `RenderIR` builder for common placed 2D draw tools.
    It adds a tiny extra leaf to the committed scene only for computing and
    compiling the likely future shader signature; it does not swap the active
    visible scene.
  - Prewarm bakes on the worker thread and creates/caches the pipeline on the GUI
    thread using the existing QRhi layout, matching the normal renderer resource
    discipline.
- `app/viewport/renderers/qrhi/viewport.py`
  - `begin_create_tool()` now calls `renderer.prewarm_for_tool(...)` after a draw
    tool is selected.
- `tests/test_qrhi_prewarm.py`
  - Added coverage that Bezier-surface prewarm adds
    `placed_bezier_surface_2d` to the future shader signature.

Validation after this follow-up pass:

```bash
.venv/bin/pytest -q tests/test_qrhi_prewarm.py tests/test_gpu_codegen.py tests/test_render_ir_specialization.py
```

Result:

```text
20 passed
```

Full suite:

```bash
.venv/bin/pytest -q
```

Result:

```text
167 passed, 3 skipped
```

Whitespace check:

```bash
git diff --check
```

Result: clean.

## Follow-up pass: avoid point-tool prewarm stalls

User testing showed that the first prewarm pass fixed the post-creation shader
bake stall, but could still block active point drawing because QRhi pipeline
creation runs on the GUI/QRhi thread:

```text
prewarm bake start tool=bezier_surface ... source_bytes=15043
async bake done reason=prewarm ... qsb=583.1 ms
prewarm pipeline driver-compiled ... in 2.99s
```

That means the expensive driver compile moved earlier, but for click-to-create
tools it could still happen while the user was placing points.

Policy adjustment:

- Point-created tools now use shader-only prewarm while the tool is active.
  This still moves the `qsb` bake off the critical path, but avoids calling
  `QRhiGraphicsPipeline.create()` during point collection.
- Drag-created tools still request full pipeline prewarm because their creation
  gesture is short and less likely to overlap with multi-click interaction.
- The remaining known tradeoff is that a first Bezier-surface commit may still
  pay the driver pipeline compile if no idle/background-safe opportunity has
  compiled that exact scene signature yet. That stall should happen at commit,
  not during point picking.

Files changed:

- `app/viewport/renderers/qrhi/renderer.py`
  - `prewarm_for_tool(...)` now accepts `compile_pipeline`.
  - Shader-cache prewarm and pipeline-cache prewarm are separated.
- `app/viewport/renderers/qrhi/viewport.py`
  - Point-created tools call prewarm with `compile_pipeline=False`.
- `tests/test_qrhi_prewarm.py`
  - Added coverage for shader-only prewarm skipping the pipeline path.

Validation after this follow-up pass:

```bash
.venv/bin/pytest -q tests/test_qrhi_prewarm.py tests/test_gpu_codegen.py tests/test_render_ir_specialization.py
```

Result:

```text
21 passed
```

Full suite:

```bash
.venv/bin/pytest -q
```

Result:

```text
168 passed, 3 skipped
```

Whitespace check:

```bash
git diff --check
```

Result: clean.

User-measured Vulkan result after this follow-up:

```text
bezier_polycurve prewarm pipeline=no qsb=569.8 ms
bezier_polycurve commit pipeline driver-compiled in 0.01s
second bezier_polycurve prewarm cache hit
bezier_surface prewarm pipeline=no qsb=752.0 ms
bezier_surface commit pipeline driver-compiled in 0.02s
```

Interpretation: the click-to-create tools no longer run driver pipeline compile
during point collection. In this run, the remaining commit-time pipeline compile
was also tiny, likely helped by the driver's warmed cache and the already-baked
shader bytecode.

## Follow-up pass: defer cold commit-time pipeline finalization

A stricter cold-ish Vulkan run showed that shader-only prewarm protects point
collection, but the final commit can still pay the cold driver compile:

```text
bezier_curve prewarm pipeline=no qsb=460.9 ms
bezier_curve commit pipeline driver-compiled in 1.06s
bezier_surface prewarm pipeline=no qsb=582.2 ms
bezier_surface commit pipeline driver-compiled in 3.44s
```

The remaining bottleneck is therefore not `RenderIR` build time and not `qsb`.
It is Vulkan driver pipeline creation for a new scene signature.

Policy adjustment:

- If a requested scene has cached shader bytecode but no cached QRhi pipeline,
  `set_scene()` no longer synchronously finalizes that scene.
- The renderer keeps the previous drawable scene resources active and schedules
  a short deferred finalization task.
- The deferred task creates the cold pipeline outside the scene commit call path,
  then swaps the new scene resources in and requests a viewport update.
- Pipeline-cache hits still activate immediately.

This does not make the Vulkan driver compile faster. It prevents the expensive
compile from running directly inside the CAD edit/commit path, which is the next
step toward keeping interaction responsive while heavy renderer variants become
ready.

Files changed:

- `app/viewport/renderers/qrhi/renderer.py`
  - Added deferred scene/pipeline state.
  - Added `_activate_codegen_scene(...)`, `_defer_codegen_finalize(...)`, and
    `_finalize_deferred_codegen()`.
  - Routed shader-cached/pipeline-cold scene updates through deferred
    finalization.
- `tests/test_qrhi_prewarm.py`
  - Added coverage that a shader-cached/pipeline-cold scene does not activate
    synchronously and is activated by the queued deferred callback.

Validation after this follow-up pass:

```bash
.venv/bin/pytest -q tests/test_qrhi_prewarm.py tests/test_gpu_codegen.py tests/test_render_ir_specialization.py
```

Result:

```text
22 passed
```

Full suite:

```bash
.venv/bin/pytest -q
```

Result:

```text
169 passed, 3 skipped
```

## Follow-up pass: tiny-scene no-cull shader variant

Audit finding:

The slow Bezier 2D scene signature was already specialized by primitive kind, but
it still emitted the spatial cull-grid shader path:

```text
box + cylinder + placed_bezier_curve_2d + placed_bezier_surface_2d
terms=4
```

For this tiny interactive scene, culling is not worth the emitted shader
complexity. The cull path added:

- grid storage-buffer declarations
- cull-grid uniforms
- `cellDist(...)`
- `cgRayGrid(...)`
- `cgCellCoord(...)`
- `cgCellExit(...)`
- a DDA branch in `main()`

This is useful for large scenes, but it is avoidable shader complexity for the
small 2D drawing signatures where cold pipeline compile latency is currently the
UX problem.

Implemented backend-neutral codegen changes:

- `core/gpu_codegen.py`
  - Added `term_count(render_ir)`.
  - Added `uses_spatial_cull(render_ir)`.
  - Scenes with fewer than 16 flattened terms now emit a flat brute-force
    raymarch shader by default.
  - Large scenes still emit the spatial cull-grid path.
  - `emit_map_glsl(...)` and `emit_fragment_shader(...)` accept an optional
    `spatial_cull` override for probes/tests.
- `app/viewport/renderers/qrhi/renderer.py`
  - The QRhi shader/pipeline cache key now includes `cull_mode`.
  - The cache key also includes selector presence, because selector-enabled
    scenes emit a different shader block from no-selector scenes.
  - Runtime grid data is only built when the emitted shader variant uses spatial
    culling.
  - UBO layout is rebuilt when the shader variant changes, because flat and cull
    variants have different loose-uniform sets after `vulkanize`.
- `tools/codegen_stress.py`
  - Added `scene=bezier2d`.
  - Added `--spatial-cull auto|on|off` for local source/qsb comparison.
- `tests/test_gpu_codegen.py`
  - Added structural coverage that small Bezier-surface variants omit cull-grid
    code.
  - Added coverage that larger scenes still keep the cull-grid path.
- `tests/test_qrhi_prewarm.py`
  - Updated renderer signature tests for the cull-mode key.
  - Added coverage that selector presence changes the shader variant key.

Local probe commands:

```bash
.venv/bin/python tools/codegen_stress.py create --scene bezier2d --n 1 --repeat 5 --spatial-cull auto
.venv/bin/python tools/codegen_stress.py create --scene bezier2d --n 1 --repeat 5 --spatial-cull on
```

Measured on this development run:

```text
auto/no-cull:  source=12,230 chars, 286 lines, qsb=390.07 ms
forced-cull:   source=15,041 chars, 347 lines, qsb=405.24 ms
```

Compared with the previous user logs for the same kind signature
(`source_bytes=15043`), the default emitted source for the small Bezier 2D scene
is now about 18.7% smaller and removes the cull-grid/DDA helper block entirely.

Important limitation:

This is a real shader-complexity reduction, but it is not proof that cold driver
pipeline creation is now below the UX threshold on the user's Vulkan driver. The
next validation should be the same cold-ish GUI run as before and compare:

```text
source_bytes
cull_mode
pipeline driver-compiled ... in X.XXs
```

Expected log shape for the Bezier surface variant:

```text
kinds=[..., 'placed_bezier_surface_2d'] cap=8 profile_mode=simple cull_mode=flat selector_mode=no_selectors source_bytes≈12230
```

Validation after this follow-up pass:

```bash
.venv/bin/pytest -q tests/test_gpu_codegen.py tests/test_qrhi_prewarm.py tests/test_render_ir_specialization.py
```

Result:

```text
25 passed
```

Full suite:

```bash
.venv/bin/pytest -q
```

Result:

```text
172 passed, 3 skipped
```

## Follow-up pass: single-group DNF shader specialization

Second audit finding:

The same small Bezier 2D scene has only one flattened DNF group:

```text
box + cylinder + placed_bezier_curve_2d + placed_bezier_surface_2d
groups=1
terms=4
```

Before this pass it still emitted the generic group-accumulator path:

- `float g[8]`
- `uint o[8]`
- dynamic `gid` scatter
- `u_group_count`
- combine loop over groups

That generic path is required for intersections and other multi-group scenes,
but it is avoidable for the common union-only CAD cases. A single group is just
the minimum over all carved terms.

Implemented backend-neutral codegen changes:

- `core/gpu_codegen.py`
  - Expanded group-capacity buckets to include `1`, `2`, and `4`.
  - Scenes with one flattened group now bake as `cap=1`.
  - `cap=1` emits a single-group `map(...)` without group accumulator arrays or
    `u_group_count`.
  - The cull-grid `cellDist(...)` path also has a single-group version for large
    scenes that have many terms but only one union group.
- `tests/test_gpu_codegen.py`
  - Updated older group-loop assumptions.
  - Added assertions that small Bezier-surface variants omit `u_group_count` and
    `float g[...]`.

Local probe after this pass:

```bash
.venv/bin/python tools/codegen_stress.py create --scene bezier2d --n 1 --repeat 5 --spatial-cull auto
.venv/bin/python tools/codegen_stress.py create --scene bezier2d --n 1 --repeat 5 --spatial-cull on
```

Measured on this development run:

```text
auto/no-cull/single-group: source=11,856 chars, 280 lines, qsb=400.89 ms
forced-cull/single-group:  source=14,380 chars, 336 lines, qsb=400.72 ms
```

The qsb subprocess timings are noisy at this scale, but the emitted source for
the default small Bezier 2D variant is now:

```text
previous cull+generic-group: 15,041 chars
no-cull only:                 12,230 chars
no-cull + single-group:       11,856 chars
```

The cumulative source reduction for this representative small Bezier 2D
signature is about 21.2%, while preserving the same RenderIR and backend-neutral
QRhi shader path.

Expected log shape after this pass:

```text
kinds=[..., 'placed_bezier_surface_2d'] cap=1 profile_mode=simple cull_mode=flat selector_mode=no_selectors source_bytes≈11856
```

Additional qsb probe coverage:

```bash
.venv/bin/python tools/codegen_stress.py create --scene polytope --n 8 --repeat 3 --spatial-cull auto
.venv/bin/python tools/codegen_stress.py create --scene polytope --n 16 --repeat 3 --spatial-cull auto
```

Results:

```text
polytope n=8:  groups=8,  terms=8,  cap=8,  cull=False, qsb=314.03 ms
polytope n=16: groups=16, terms=16, cap=16, cull=True,  qsb=322.10 ms
```

Validation after this follow-up pass:

```bash
.venv/bin/pytest -q tests/test_gpu_codegen.py tests/test_qrhi_prewarm.py tests/test_render_ir_specialization.py
.venv/bin/pytest -q
git diff --check
```

Results:

```text
25 passed
172 passed, 3 skipped
diff check clean
```

## Follow-up pass: selector-free shader variant

Third audit finding:

The common Bezier 2D drawing scenes do not contain region selectors, but the
shader still carried no-op selector plumbing:

- `u_sel_count`
- selector storage-buffer declaration
- `regionAt(...)` no-op stub
- a `regionAt(hp, owner)` call in the hit shading path

Selector support is still required when `region_selector` nodes are present, but
the no-selector case is the common interactive drawing path.

Implemented backend-neutral codegen changes:

- `core/gpu_codegen.py`
  - No-selector variants now omit selector uniforms, selector storage-buffer
    declarations, the no-op `regionAt(...)` stub, and selector shading code.
  - Selector-enabled variants still emit the existing selector subtree evaluator
    and region highlight path.
- `tests/test_gpu_codegen.py`
  - Added assertions that small Bezier 2D variants omit selector plumbing.
  - Updated selector tests to verify that no-selector scenes no longer emit
    `regionAt(...)`, while selector scenes still do.

Local probe after this pass:

```bash
.venv/bin/python tools/codegen_stress.py create --scene bezier2d --n 1 --repeat 5 --spatial-cull auto
.venv/bin/python tools/codegen_stress.py create --scene alltypes --n 1 --repeat 3 --spatial-cull auto
```

Measured on this development run:

```text
bezier2d: source=11,554 chars, 276 lines, qsb=384.36 ms
alltypes: source=32,810 chars, 732 lines, qsb=509.73 ms
```

The `alltypes` probe includes a region selector, so it covers the selector-enabled
shader path after the split.

Current cumulative source reduction for the representative small Bezier 2D
signature:

```text
previous cull+generic-group+selector-stub: 15,041 chars
no-cull only:                          12,230 chars
no-cull + single-group:                11,856 chars
no-cull + single-group + no-selector:  11,554 chars
```

That is about a 23.2% reduction in emitted source size for this signature, while
keeping the QRhi path backend-neutral.

Expected log shape after this pass:

```text
kinds=[..., 'placed_bezier_surface_2d'] cap=1 profile_mode=simple cull_mode=flat selector_mode=no_selectors source_bytes≈11554
```

## Follow-up pass: no-carve/no-child shader variant

Fourth audit finding:

The common Bezier 2D drawing scene is union-only and has no local carve terms:

```text
groups=1
terms=4
has_carves=False
```

It also does not need the `Children` buffer unless profile-VM or selector code is
present. Before this pass, the no-selector/no-profile Bezier 2D variant still
declared:

- `Children` / `u_children`
- `Carves` / `u_carves`
- carve loop inside `termDist(...)`

Implemented backend-neutral codegen changes:

- `core/gpu_codegen.py`
  - Added `has_carves(render_ir)`.
  - No-carve variants now emit a simpler `termDist(...)` with no `u_carves`
    buffer and no carve loop.
  - `Children` buffer declarations are now emitted only when profile or selector
    shader code needs `u_children`.
- `app/viewport/renderers/qrhi/renderer.py`
  - The QRhi shader/pipeline cache key now includes `carve_mode`, because
    union-only and carved scenes can share the same primitive kind set while
    requiring different shader source.
- `tools/codegen_stress.py`
  - The `create` probe now prints `has_carves`.
- `tests/test_gpu_codegen.py`
  - Added assertions that small Bezier 2D variants omit `u_carves` and
    `u_children`.
  - Added coverage that carved scenes still emit the carve buffer and carve loop.
- `tests/test_qrhi_prewarm.py`
  - Updated renderer signature helpers for the carve-mode key.

Local probes after this pass:

```bash
.venv/bin/python tools/codegen_stress.py create --scene bezier2d --n 1 --repeat 3 --spatial-cull auto
.venv/bin/python tools/codegen_stress.py create --scene carveunion --n 2 --repeat 3 --spatial-cull auto
```

Measured on this development run:

```text
bezier2d:    source=11,182 chars, 270 lines, has_carves=False, qsb=317.34 ms
carveunion:  source=5,811 chars,  160 lines, has_carves=True,  qsb=371.47 ms
```

Current cumulative source reduction for the representative small Bezier 2D
signature:

```text
previous cull+generic-group+selector-stub+carve/child buffers: 15,041 chars
no-cull only:                                             12,230 chars
no-cull + single-group:                                   11,856 chars
no-cull + single-group + no-selector:                     11,554 chars
no-cull + single-group + no-selector + no-carves/children:11,182 chars
```

That is about a 25.7% emitted-source reduction for this signature.

Expected log shape after this pass:

```text
kinds=[..., 'placed_bezier_surface_2d'] cap=1 profile_mode=simple cull_mode=flat selector_mode=no_selectors carve_mode=no_carves source_bytes≈11182
```

## Follow-up pass: inline simple single-group terms

Fifth audit finding:

After the no-carve/no-child split, the small Bezier 2D variant still emitted a
generic `termDist(...)` function and called it from the single-group map loop.
For `cap=1` and `has_carves=False`, every term is just one positive leaf, so the
map loop can directly evaluate:

```text
leaf = u_terms[i].x
d = leafDist(leaf, p)
owner = u_nodes[leaf].base_owner_id
```

Implemented backend-neutral codegen changes:

- `core/gpu_codegen.py`
  - Single-group/no-carve variants now inline term evaluation in `map(...)`.
  - Single-group/no-carve cull variants also inline term evaluation in
    `cellDist(...)`.
  - Carved and multi-group variants keep the generic `termDist(...)` path.
- `tests/test_gpu_codegen.py`
  - Added assertions that the small Bezier 2D variant omits `termDist(...)`.
  - Existing carved-scene coverage verifies that carved variants still emit
    `termDist(...)` and the carve loop.

Local probes after this pass:

```bash
.venv/bin/python tools/codegen_stress.py create --scene bezier2d --n 1 --repeat 5 --spatial-cull auto
.venv/bin/python tools/codegen_stress.py create --scene carveunion --n 2 --repeat 3 --spatial-cull auto
.venv/bin/python tools/codegen_stress.py create --scene polytope --n 16 --repeat 3 --spatial-cull auto
.venv/bin/python tools/codegen_stress.py create --scene union --n 8 --repeat 3 --spatial-cull auto
```

Measured on this development run:

```text
bezier2d:    source=11,037 chars, 264 lines, qsb=359.26 ms
carveunion:  source=5,811 chars,  160 lines, qsb=381.85 ms
polytope16:  source=8,713 chars,  222 lines, qsb=376.41 ms
union8:      source=7,952 chars,  206 lines, qsb=380.97 ms
```

Current cumulative source reduction for the representative small Bezier 2D
signature:

```text
previous tracked variant: 15,041 chars
current tracked variant:  11,037 chars
```

That is about a 26.6% emitted-source reduction for this signature.

Expected log shape after this pass:

```text
kinds=[..., 'placed_bezier_surface_2d'] cap=1 profile_mode=simple cull_mode=flat selector_mode=no_selectors carve_mode=no_carves source_bytes≈11037
```

## Follow-up pass: single-quadratic Bezier placed leaves

Sixth audit finding:

The first-created Bezier curve and Bezier surface tools create exactly three
control points:

```text
anchor, control, anchor
```

Before this pass, those three-point shapes still lowered to the loop-capable
`placed_bezier_curve_2d` and `placed_bezier_surface_2d` shader leaves. Those
leaves support polycurves and multi-segment surfaces, but for a single quadratic
segment they emit avoidable point-count calculation and segment loops.

Implemented backend-neutral codegen/render-IR changes:

- `core/gpu_node_types.py`
  - Added `placed_quadratic_bezier_curve_2d`.
  - Added `placed_quadratic_bezier_surface_2d`.
  - Appended both kinds to preserve existing node-type codes.
- `core/render_ir.py`
  - `BezierCurveProfile` with exactly 3 points now lowers to
    `placed_quadratic_bezier_curve_2d`.
  - `BezierSurfaceProfile` with exactly 3 points now lowers to
    `placed_quadratic_bezier_surface_2d`.
  - Multi-segment Bezier curves/surfaces keep the existing loop-capable leaves.
- `core/gpu_codegen.py`
  - Added direct leaf branches for the two single-quadratic placed leaves.
  - The quadratic surface branch avoids the `irQuadraticBezierRayCrossValue` and
    `irSegmentRayCrossValue` helper wrappers used by the loop-capable surface.
- `app/viewport/renderers/qrhi/renderer.py`
  - Draw-tool prewarm now uses the single-quadratic leaves for `bezier_curve`
    and `bezier_surface`.
  - `bezier_polycurve` prewarm still uses the loop-capable curve leaf.
- `tools/codegen_stress.py`
  - The `bezier2d` probe now represents the common first-created quadratic
    Bezier curve/surface variant.
- Tests:
  - `tests/test_render_ir_specialization.py` covers three-point vs multi-segment
    Bezier lowering.
  - `tests/test_gpu_codegen.py` covers that the common quadratic variant omits
    loop-capable Bezier branches and ray-crossing wrapper helpers.
  - `tests/test_qrhi_prewarm.py` updated for the new Bezier-surface prewarm
    signature.

Local probe after this pass:

```bash
.venv/bin/python tools/codegen_stress.py create --scene bezier2d --n 1 --repeat 5 --spatial-cull auto
```

Measured on this development run:

```text
bezier2d: source=9,887 chars, 245 lines, qsb=361.52 ms
```

Current cumulative source reduction for the representative first-created Bezier
2D signature:

```text
previous tracked variant: 15,041 chars
current tracked variant:   9,887 chars
```

That is about a 34.3% emitted-source reduction for this signature.

Expected log shape after this pass:

```text
kinds=[..., 'placed_quadratic_bezier_curve_2d', 'placed_quadratic_bezier_surface_2d'] cap=1 profile_mode=simple cull_mode=flat selector_mode=no_selectors carve_mode=no_carves source_bytes≈9887
```

## Follow-up pass: source-size regression guard

The remaining large blocks in the representative Bezier 2D shader are now the
exact quadratic Bezier distance/crossing helpers and the shared raymarch/shading
code. Those are not safe to approximate or remove without changing rendering
behavior, so this pass adds regression coverage instead of another speculative
trim.

Added test coverage:

- `tests/test_gpu_codegen.py`
  - Added `test_representative_bezier2d_variant_stays_under_source_budget`.
  - The test builds a representative small 2D scene with:
    - `box`
    - `placed_circle_2d`
    - `placed_quadratic_bezier_curve_2d`
    - `placed_quadratic_bezier_surface_2d`
  - It verifies the low-complexity variant shape:
    - `cap=1`
    - `cull_mode=flat`
    - `carve_mode=no_carves`
  - It asserts emitted fragment shader source stays below 10,500 bytes.

Current measured size for that guarded scene:

```text
source=9,926 chars, 247 lines
```

Focused validation after this pass:

```bash
.venv/bin/pytest -q tests/test_gpu_codegen.py tests/test_qrhi_prewarm.py tests/test_render_ir_specialization.py
```

Result:

```text
30 passed
```

## Follow-up pass: QRhi compile log summarizer

After the source-size trims, the remaining proof needs real cold QRhi/Vulkan
pipeline timings from the GUI. To make those runs easier to compare, added a
small log summarizer:

```bash
.venv/bin/python tools/qrhi_compile_log_summary.py path/to/casocad.log
```

It can also read pasted logs from stdin:

```bash
pbpaste | .venv/bin/python tools/qrhi_compile_log_summary.py -
```

The tool extracts and sorts QRhi compile telemetry:

- `prewarm bake start`
- `async bake done`
- `codegen baked variant`
- `pipeline driver-compiled`
- `prewarm pipeline driver-compiled`

It reports:

- backend
- source bytes
- qsb time
- pipeline compile time
- full shader signature label

Files changed:

- `tools/qrhi_compile_log_summary.py`
  - New standalone parser/formatter for QRhi compile logs.
- `tests/test_qrhi_compile_log_summary.py`
  - Regression coverage for parsing qsb and pipeline compile records.

Focused validation after this pass:

```bash
.venv/bin/pytest -q tests/test_qrhi_compile_log_summary.py tests/test_gpu_codegen.py tests/test_qrhi_prewarm.py tests/test_render_ir_specialization.py
```

Result:

```text
31 passed
```

## Follow-up pass: measurement probe cleanup

The local `fps` stress probe now reports the shader cull policy before trying
to describe cull-grid behavior. For the small Bezier2D cold-compile scene this
matters because the emitted shader intentionally uses the flat/no-cull variant;
testing `gpu_cull.build_term_grid()` for that scene produced a misleading
"unbounded leaf -> brute force" warning for a grid the shader does not use.

Current headless create probe:

```bash
.venv/bin/python tools/codegen_stress.py create --scene bezier2d --n 1 --repeat 1
```

Result:

```text
flattened: groups=1  terms=4  capacity=g[1]  kinds=4
auto_cull=False  shader_cull=False  term_count=4  has_carves=False
shader source: 9,887 chars  (245 lines)
qsb compile (offline): 281.20 ms
```

Current offscreen QRhi probe:

```bash
QT_QPA_PLATFORM=offscreen QRHI_BACKEND=vulkan \
  .venv/bin/python tools/codegen_stress.py fps \
  --scene bezier2d --n 1 --warmup 2 --measure 1 --width 640 --height 480
```

Observed:

```text
QRhiWidget: QRhi is not supported on this platform.
[stress-fps] cull OFF by shader policy: 4 terms use the flat shader variant
[stress-fps] QRhi unavailable; cannot measure real pipeline compile
```

This confirms the probe output path, but it is not evidence for cold driver
pipeline time. Real compile timing still needs a GUI run on a platform where
QRhi can create the requested backend.

## Follow-up pass: filtered node-type defines for specialized shaders

Audit finding:

The codegen shader still emitted the complete generated `NODE_*` define block
for every SDF/profile/operator kind, even when the specialized shader only had
branches for a few leaf kinds. Those unused defines do not change runtime
behavior, but they are dead source text on exactly the cold-compile path we are
trying to shrink.

Implemented backend-neutral change:

- `core/gpu_node_types.py`
  - `emit_glsl_defines()` now keeps the previous default behavior: no argument
    emits every node kind.
  - Added an optional `used_kinds` filter for specialized shader emitters.
- `core/gpu_codegen.py`
  - Codegen now requests only the node constants referenced by the emitted leaf,
    selector, and profile blocks.
  - Selector/profile variants expand the filtered set conservatively so their
    optional shader blocks still have the constants they can reference.
- `tests/test_gpu_scene.py`
  - Added coverage that the default full-header behavior remains intact and the
    filtered form omits unrelated node constants.
- `tests/test_gpu_codegen.py`
  - The representative Bezier2D source-budget test now asserts unrelated node
    constants stay out of the emitted shader.
  - Tightened the source budget from 10,500 bytes to 9,000 bytes.

Current headless Bezier2D create probe after this pass:

```text
flattened: groups=1  terms=4  capacity=g[1]  kinds=4
auto_cull=False  shader_cull=False  term_count=4  has_carves=False
shader source: 8,372 chars  (201 lines)
```

Representative guarded test scene:

```text
source=8,420 chars, 203 lines
```

This is a source-size reduction from the prior Bezier2D probe value of 9,887
chars / 245 lines. A single local `qsb` sample was noisy, so no qsb/runtime
speedup is claimed from this pass until confirmed by repeated local runs or a
real QRhi GUI run.

## Follow-up pass: omit unused stack/opcode defines from simple codegen shaders

Audit finding:

After filtering node-kind defines, simple codegen shaders still emitted
interpreter support constants:

- `IR_STACK_CAPACITY`
- `IR_PROFILE_STACK_CAPACITY`
- `OP_*`
- `OPCODE_SHIFT`
- `PAYLOAD_MASK`

The small Bezier2D variants do not reference those constants. Selector shaders
need `IR_STACK_CAPACITY`; profile-VM shaders need `IR_PROFILE_STACK_CAPACITY`;
the specialized codegen path does not use the opcode constants.

Implemented backend-neutral change:

- `core/gpu_node_types.py`
  - `emit_glsl_defines()` now accepts optional `include_stack_defs` and
    `include_opcode_defs` flags.
  - Defaults still emit the full header for existing/interpreter callers.
- `core/gpu_codegen.py`
  - Specialized codegen omits opcode constants.
  - It emits stack-capacity constants only for profile or selector variants.
- `tests/test_gpu_scene.py`
  - Added coverage for filtered headers without stack/opcode constants.
- `tests/test_gpu_codegen.py`
  - The Bezier2D source-budget test now asserts those constants stay out of the
    simple shader.
  - Tightened the source budget from 9,000 bytes to 8,500 bytes.

Current headless Bezier2D create probe after this pass:

```text
shader source: 8,172 chars  (192 lines)
```

Representative guarded test scene:

```text
source=8,220 chars, 194 lines
```

## Follow-up pass: shared 2D parameter accessor

Audit finding:

The remaining 2D leaf branches repeated this source pattern many times:

```glsl
vec2(irP(base, X), irP(base, X + 1u))
```

This is the same existing `u_params` buffer access, just emitted verbosely in
each 2D branch.

Implemented backend-neutral change:

- `core/gpu_codegen.py`
  - Added a tiny `irP2(base, i)` accessor beside `irP()` / `irP3()`.
  - Converted 2D leaf snippets to use it for paired profile/control-point
    parameter loads.
  - Emits `irP2()` only when the specialized leaf set actually references it, so
    sphere-only or other non-2D shaders do not grow.
- `tests/test_gpu_codegen.py`
  - Added coverage that sphere-only shaders do not emit the 2D accessor.
  - Tightened the representative Bezier2D source budget from 8,500 bytes to
    8,300 bytes.

Current headless Bezier2D create probe after this pass:

```text
shader source: 8,113 chars  (191 lines)
```

Representative guarded test scene:

```text
source=8,141 chars, 193 lines
```

## Follow-up pass: conditional quadratic-Bezier segment fallback

Audit finding:

`irQuadraticBezierDistance()` included an inline degenerate-curve fallback that
computes point-to-segment distance. Bezier surface variants already emit
`irSegmentDistance2D()`, so the inline fallback duplicated source in the
curve+surface signatures. A curve-only shader must not pay for the segment
helper, though, because that would grow the first Bezier-curve variant.

Implemented backend-neutral change:

- `core/gpu_codegen.py`
  - Kept the inline fallback as the default `irQuadraticBezierDistance()`.
  - Added a compact alternate helper that calls `irSegmentDistance2D()` for the
    degenerate fallback.
  - `_helpers_for()` selects the compact helper only when the leaf set already
    requires `irSegmentDistance2D()`.
- `tests/test_gpu_codegen.py`
  - Added coverage that curve+surface variants use the compact fallback.
  - Added coverage that curve-only variants do not emit `irSegmentDistance2D()`
    and keep the inline fallback.
  - Tightened the representative Bezier2D source budget from 8,300 bytes to
    8,200 bytes.

Current headless Bezier2D create probe after this pass:

```text
shader source: 7,990 chars  (187 lines)
```

Representative guarded test scene:

```text
source=8,018 chars, 189 lines
```

Curve-only check:

```text
source=5,323 chars, 128 lines
```

The curve-only size matches the previous measurement, so this pass improves the
curve+surface cold signature without regressing the first curve-only variant.

## Follow-up pass: strip generated comments from QRhi-facing fragment shaders

Audit finding:

The generated fragment shader still carried human comments into the `qsb`/driver
compile path. These comments are useful while inspecting `emit_map_glsl()`, but
they are dead source text for the QRhi fragment shader.

Implemented backend-neutral change:

- `core/gpu_codegen.py`
  - Added a small line-comment stripper.
  - `emit_fragment_shader()` strips line comments by default.
  - `emit_fragment_shader(strip_comments=False)` keeps the readable source for
    debugging/tests.
  - `emit_map_glsl()` remains unchanged and readable.
- `tests/test_gpu_codegen.py`
  - Added coverage that the default fragment shader has no line comments.
  - Added coverage that the unstripped debug form still contains the expected
    comments.
  - Tightened the representative Bezier2D source budget from 8,200 bytes to
    8,050 bytes.

Current headless Bezier2D create probe after this pass:

```text
shader source: 7,861 chars  (178 lines)
```

Representative guarded test scene:

```text
source=7,887 chars, 180 lines
```

## Follow-up pass: repeated qsb samples in the headless probe

Audit finding:

The source-size checks are deterministic, but local `qsb` timings were noisy
because the stress probe ran the qsb subprocess only once. That made it too easy
to over-interpret one slow or fast sample after a source change.

Implemented measurement-tool change:

- `tools/codegen_stress.py`
  - Added `--qsb-repeat N` to the `create` subcommand.
  - CPU stages still use `--repeat` and report the best sample.
  - qsb now reports the best sample in the stage table plus min/median/max when
    `--qsb-repeat > 1`.

Example command:

```bash
.venv/bin/python tools/codegen_stress.py create \
  --scene bezier2d --n 1 --repeat 1 --qsb-repeat 3
```

Current local result:

```text
shader source: 7,861 chars  (178 lines)
qsb compile (offline): 277.88 ms best of 3
qsb samples: min=277.88 ms  median=280.62 ms  max=292.60 ms
```

This still does not replace real QRhi driver pipeline timing, but it gives a
less noisy offline-compile probe for comparing backend-neutral source changes.

## Follow-up pass: compact QRhi-facing shader whitespace

Audit finding:

After stripping comments, the default QRhi-facing fragment shader still carried
indentation whitespace from the readable generated GLSL. The driver/qsb path
does not need that whitespace; keeping it only inflates the source string being
parsed.

Implemented backend-neutral change:

- `core/gpu_codegen.py`
  - The default `emit_fragment_shader()` path now strips leading/trailing
    whitespace from each emitted source line after comments are removed.
  - `emit_fragment_shader(strip_comments=False)` still returns the readable,
    commented, indented debug source.
  - `emit_map_glsl()` remains readable.
- `tests/test_gpu_codegen.py`
  - Added coverage that the default fragment shader has no line comments and no
    indented lines.
  - Tightened the representative Bezier2D source budget from 8,050 bytes to
    7,200 bytes.

Current headless Bezier2D create probe after this pass:

```text
shader source: 7,046 chars  (178 lines)
```

Representative guarded test scene:

```text
source=7,056 chars, 180 lines
```

Repeated qsb sample from the same run:

```text
qsb samples: min=287.17 ms  median=322.31 ms  max=357.46 ms
```

As before, the qsb timing is useful only for local offline comparison. The real
driver pipeline compile time still needs GUI QRhi timing on a supported backend.

## Follow-up pass: collapse redundant QRhi-facing source whitespace

Audit finding:

After removing comments and indentation, a few generated lines still contained
redundant internal whitespace. This is also dead source text for qsb/driver
parsing.

Implemented backend-neutral change:

- `core/gpu_codegen.py`
  - The default `emit_fragment_shader()` path now collapses repeated whitespace
    inside non-preprocessor lines after comments are removed.
  - Preprocessor lines are left structurally intact.
  - The readable debug path remains available through
    `emit_fragment_shader(strip_comments=False)`.
- `tests/test_gpu_codegen.py`
  - Added coverage that the default fragment shader has no double-space lines.
  - Tightened the representative Bezier2D source budget from 7,200 bytes to
    7,100 bytes.

Current headless Bezier2D create probe after this pass:

```text
shader source: 7,042 chars  (178 lines)
```

Representative guarded test scene:

```text
source=7,052 chars, 180 lines
```

Repeated qsb sample from the same run:

```text
qsb samples: min=284.04 ms  median=285.43 ms  max=286.81 ms
```

## Follow-up pass: aggregate QRhi compile log summaries by signature

Audit finding:

The log summarizer listed QRhi compile events sorted by duration, but pasted GUI
runs still required manual grouping to answer the important question: "which
shader signature is still slow, and is it qsb or driver pipeline compile?"

Implemented measurement-tool change:

- `tools/qrhi_compile_log_summary.py`
  - Added per-signature aggregation.
  - Reports max qsb time, max driver pipeline time, backends, event count, source
    bytes, and the normalized signature label.
  - Normalizes qsb records so `qsb=... ms` is not treated as part of the shader
    signature.
- `tests/test_qrhi_compile_log_summary.py`
  - Added coverage for grouping qsb and pipeline records under one signature.

Example output:

```text
signature summary
backends source_bytes max_qsb_ms max_pipeline_s events label
-------- ------------ ---------- -------------- ------ -----
Vulkan           7042 420.0      1.75                2 kinds=[...]
```

This should make the next real GUI run easier to interpret: the slowest
signature and whether the cost is qsb or backend driver pipeline creation should
be visible without manually comparing log lines.

## Follow-up pass: real QRhi FPS probe measurements

The environment can run real Qt/QRhi windows when the FPS stress probe is run
through the GUI-approved proxy path. `tools/codegen_stress.py fps` now configures
logging, so the renderer's backend/source/qsb/pipeline telemetry appears in the
same output as the time-to-first-frame measurement.

OpenGL command:

```bash
.venv/bin/python tools/codegen_stress.py fps \
  --scene bezier2d --n 1 --warmup 2 --measure 1 --width 640 --height 480
```

Observed:

```text
backend=OpenGL
kinds=['box', 'cylinder', 'placed_quadratic_bezier_curve_2d', 'placed_quadratic_bezier_surface_2d']
cap=1 profile_mode=simple cull_mode=flat selector_mode=no_selectors carve_mode=no_carves
source_bytes=7042
qsb=324.4 ms
pipeline driver-compiled in 0.04s
time-to-first-frame=1363 ms
FPS=141.2
```

Vulkan command:

```bash
QRHI_BACKEND=vulkan .venv/bin/python tools/codegen_stress.py fps \
  --scene bezier2d --n 1 --warmup 2 --measure 1 --width 640 --height 480
```

Observed:

```text
backend=Vulkan
kinds=['box', 'cylinder', 'placed_quadratic_bezier_curve_2d', 'placed_quadratic_bezier_surface_2d']
cap=1 profile_mode=simple cull_mode=flat selector_mode=no_selectors carve_mode=no_carves
source_bytes=7042
qsb=333.3 ms
pipeline driver-compiled in 0.57s
time-to-first-frame=3720 ms
FPS=139.4
```

Interpretation:

- The optimized Bezier2D curve+surface signature is now materially smaller than
  the earlier 15k source variants and compiles as the expected flat/no-cull,
  single-group, no-selector, no-carve shader.
- On this machine/backend run, Vulkan driver pipeline creation for this shader
  is no longer multi-second by itself (`0.57s`), but total first-frame latency is
  still several seconds. That remaining time includes Qt/QRhi window/backend
  startup plus qsb and driver work in the stress harness, so GUI workflow logs
  from the actual draw-tool path are still useful before declaring the UX freeze
  fully solved.

## Follow-up pass: real draw-tool workflow measurements

Added a `workflow` subcommand to `tools/codegen_stress.py`:

```bash
.venv/bin/python tools/codegen_stress.py workflow --tool bezier_surface
```

The workflow probe opens the real `MainWindow`, arms the actual viewport draw
tool, waits for shader-only prewarm, commits the collected point-shape through
the same viewport/MainWindow signal path as the GUI, and reports the first frame
after commit. It also enables renderer logging, so qsb/pipeline telemetry appears
in the same output.

OpenGL, `bezier_surface`:

```text
prewarm: source_bytes=7007 qsb=345.8 ms
commit: RenderIR=2.4 ms
first frame after commit=10 ms
deferred pipeline finalize: 1.05s
```

Vulkan, `bezier_surface`:

```text
prewarm: source_bytes=7007 qsb=353.0 ms
commit: RenderIR=4.0 ms
first frame after commit=8 ms
deferred pipeline finalize: 0.00s
```

Vulkan, `bezier_curve`:

```text
prewarm: source_bytes=5665 qsb=373.7 ms
commit: RenderIR=4.0 ms
first frame after commit=9 ms
deferred pipeline finalize: 0.60s
```

Vulkan, `bezier_polycurve`:

```text
prewarm: source_bytes=5771 qsb=353.4 ms
commit: RenderIR=2.9 ms
first frame after commit=9 ms
deferred pipeline finalize: 0.75s
```

Interpretation:

- The original user-visible freeze was on committing a freshly created 2D Bezier
  SDF. In the measured real MainWindow workflow, commit-to-first-frame is now
  8-10 ms for curve/polycurve/surface.
- Cold qsb work is moved into shader-only prewarm while the user is collecting
  points.
- Cold QRhi pipeline creation is deferred after commit, so the previous scene
  remains drawable and the edit path returns quickly.
- The emitted Bezier draw-tool variants are backend-neutral QRhi shaders and
  keep the expected flat/no-cull, single-group shape.
