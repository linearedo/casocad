# Meshing Viewer Integration

## Goal

Add the first standalone Meshing Workspace viewer without coupling CAD rendering
to mesher output.

The viewer follows the boundary decided earlier:

```text
scene.json -> MeshableDomain -> user mesher script -> Arrow mesh artifact
Arrow mesh artifact -> async preview loader -> QRhi mesh viewer
```

CAD remains SDF/RenderIR based. The meshing viewer consumes only the generic
Arrow mesh artifact.

## Package Boundary

```text
app/meshing/viewer/loader.py
  Arrow IPC reader and bounded preview triangulation
  no QRhi dependency

app/meshing/viewer/renderer.py
  QRhi mesh renderer
  backend-aware shader/pipeline setup

app/meshing/viewer/widget.py
  QRhiWidget wrapper, camera controls, async loader wiring

app/meshing/workspace.py
  owns the viewer as an independent Meshing Workspace pane
```

The QRhi code lives under `app/meshing/viewer/`, not `core/`, because it is GUI
and backend resource management.

## Implemented

- Added bounded Arrow preview loading.
  - Reads Arrow IPC record batches without `read_all()`.
  - Scans bounds/tags first.
  - Emits triangulated preview chunks in a second pass.
  - Caps preview geometry with `max_preview_vertices` to avoid blindly loading
    a huge mesh into RAM.
- Added visualization triangulation rules:
  - `triangle` rows draw directly.
  - `quad`/polygon rows are fan-triangulated.
  - `point` rows become small tetra markers so the current default script is
    visible.
- Added deterministic tag colors from `tag_name`.
- Added `MeshArtifactLoader` with Qt signals:
  - `chunk_loaded`
  - `finished`
  - `failed`
- Added standalone `QRhiMeshRenderer`.
  - Uses one simple vertex/fragment shader pair.
  - Bakes shaders with `pyside6-qsb`.
  - Detects QRhi backend conventions at initialize:
    - `isYUpInNDC()` for clip-space Y sign
    - `isClipDepthZeroToOne()` when available
  - Builds GPU resources outside the render pass.
  - Draws static triangle buffers with depth test/write enabled.
- Added `QRhiMeshViewerWidget`.
  - Uses `QRHI_BACKEND` (`vulkan`, `opengl`, `metal`, `d3d11`) when provided.
  - Lets QRhi pick the platform default when no backend is forced.
  - Provides orbit camera and wheel zoom.
  - Loads Arrow artifacts asynchronously through the loader.
- Integrated the viewer into `MeshingWorkspace`.
  - Left pane: imported domains.
  - Center pane: artifact status, QRhi mesh viewer, bounded row preview.
  - Right pane: script editor and output log.
  - After a script run writes an Arrow artifact, the viewer loads it.
- Added `Open mesh artifact...` to the Meshing Workspace.
  - Existing `.arrow` mesh artifacts can be loaded into the viewer without
    importing a scene or running a script.
- Added a Meshing Workspace preview budget control.
  - `Max render triangles` exposes the maximum number of filled render
    triangles loaded into the visualization preview.
  - Internally this is still converted to a vertex cap because the cache and
    GPU upload path are vertex-buffer based.
  - `Auto` chooses a conservative preview limit from available system RAM.
  - `Auto` reloads the current artifact if one is already open.
- Clarified mesh preview status wording.
  - Script output is reported as `mesh element(s)`.
  - Viewer output is reported as `render triangle(s)`.
  - Wireframe overlay output is reported as `wire edge(s)`.
- Added wireframe preview support.
  - Loader derives wire edges from the original artifact element vertices.
  - Quads/polygons keep their semantic edges even though filled rendering uses
    triangulated preview geometry.
  - The QRhi renderer uses a separate `Lines` pipeline for the wire overlay,
    avoiding backend-specific polygon fill/line modes.
  - The Meshing Workspace exposes separate `Filled` and `Wireframe` toggles.
  - `Filled` is off by default for the slice preview.
  - `Wireframe` is on by default.
- Split the Meshing Workspace into top-level pages.
  - The top toolbar has `Viewer` and `Script` actions.
  - The Viewer page contains domain/artifact controls, preview budget, viewport,
    and preview table.
  - The Script page contains the script editor, run button, and output log.
  - After a successful script run, the workspace returns to the Viewer page.
- Replaced the default script's point-only sampling demo with a simple quad
  slice demo.
  - It uses `domains["fluid"]`.
  - It samples one XY slice through the fluid domain bounds.
  - It emits `quad` rows tagged as `fluid_slice`.
  - The script comments explicitly describe this as a bounded 2D slice preview,
    not a full 3D mesher.
- Replaced the default quad slice script with a marching-squares slice script.
  - It emits triangulated `triangle` rows tagged as `fluid_slice`.
  - Full interior cells share grid-edge coordinates.
  - Boundary cells are clipped against interpolated SDF zero crossings.
  - Edge intersections are cached so adjacent cells reuse the same boundary
    coordinates.
  - This is geometrically conforming for the sampled slice, but the current
    Arrow artifact still stores per-element vertex coordinates rather than a
    topological indexed mesh.

## Deliberate Limits

- This is a preview viewer, not the final solver-grade mesh database.
- The viewer intentionally caps loaded vertices for RAM safety.
- The first renderer handles static polygon/point preview only.
- Point elements are rendered as small polygonal tetra markers, not native GPU
  points. This is intentional for the first backend-portable viewer because
  fixed-size point sprites can be backend-specific; a future pass can add a
  dedicated point-display mode.
- No CAD selection, editing, or SDF raymarching is mixed into this viewer.
- Large artifact interaction beyond the preview cap should later use tiled or
  level-of-detail loading.

## Chunk And RAM Model

There are two separate limits:

```text
Arrow artifact
  Full mesh output written by the script.
  The schema does not impose a row limit.

Viewer preview
  Bounded visualization subset derived from the Arrow artifact.
  Current defaults:
    max_rows_per_chunk = 4096
    max_preview_vertices = 300_000 by default, adjustable in the workspace
```

The loader reads Arrow IPC record batches and emits preview chunks. This avoids
calling `read_all()` for visualization. The current QRhi viewer still combines
loaded preview chunks into one static vertex buffer for drawing, so
`max_preview_vertices` is the RAM/GPU safety valve. This value is not an Arrow
file limit; it only limits the visualization subset.

Future large-mesh work should replace the single combined preview buffer with
chunk-owned GPU buffers or tiled/LOD buffers, so the viewer can stream and evict
preview chunks instead of accumulating them.

## Verification

- Compile check passed:
  - `app/meshing/viewer/loader.py`
  - `app/meshing/viewer/renderer.py`
  - `app/meshing/viewer/widget.py`
  - `app/meshing/workspace.py`
  - `tests/test_mesh_viewer_loader.py`
- Focused tests passed:
  - `tests/test_mesh_viewer_loader.py`
  - `tests/test_mesh_artifact.py`
  - `tests/test_mesh_api.py`
  - result: `6 passed`
- Full test suite passed:
  - result: `146 passed, 3 skipped`
- `git diff --check` passed.
- Offscreen Meshing Workspace construction passed.
  - Qt reported `QRhiWidget: QRhi is not supported on this platform`, expected
    with `QT_QPA_PLATFORM=offscreen`.
  - The workspace still constructed and closed cleanly.
- PySide6 QRhi API introspection passed for the renderer methods used:
  - `QRhiGraphicsPipeline.setDepthTest`
  - `QRhiGraphicsPipeline.setDepthWrite`
  - `QRhiGraphicsPipeline.setCullMode`
  - `QRhiShaderResourceBinding.uniformBuffer`

Additional verification after `Open mesh artifact...` and the quad default
script:

- Compile check passed:
  - `app/meshing/workspace.py`
  - `app/meshing/viewer/loader.py`
  - `app/meshing/viewer/widget.py`
  - `app/meshing/viewer/renderer.py`
- Focused tests passed:
  - `tests/test_mesh_api.py`
  - `tests/test_mesh_artifact.py`
  - `tests/test_mesh_viewer_loader.py`
  - result: `7 passed`
- Full test suite passed:
  - result: `147 passed, 3 skipped`
- `git diff --check` passed.
- Offscreen workspace control check passed.
  - Confirmed `Open mesh artifact...` button is present.
- Offscreen default-script check passed.
  - Imported `scene.json`.
  - Ran the workspace default script.
  - Confirmed the emitted preview table contains `quad` rows.

Additional verification after adding the preview budget control:

- Compile check passed:
  - `app/meshing/workspace.py`
  - `app/meshing/viewer/loader.py`
  - `app/meshing/viewer/widget.py`
- Focused tests passed:
  - `tests/test_mesh_api.py`
  - `tests/test_mesh_artifact.py`
  - `tests/test_mesh_viewer_loader.py`
  - result: `7 passed`
- Full test suite passed:
  - result: `147 passed, 3 skipped`
- `git diff --check` passed.
- Offscreen preview-budget control check passed.
  - Confirmed manual value changes update the viewer.
  - Confirmed `Auto` updates the viewer limit.

Additional verification after wireframe/status wording:

- Compile check passed:
  - `app/meshing/viewer/loader.py`
  - `app/meshing/viewer/renderer.py`
  - `app/meshing/viewer/widget.py`
  - `app/meshing/workspace.py`
  - `tests/test_mesh_viewer_loader.py`
- Focused tests passed:
  - `tests/test_mesh_viewer_loader.py`
  - `tests/test_mesh_artifact.py`
  - `tests/test_mesh_api.py`
  - result: `7 passed`
- Full test suite passed:
  - result: `147 passed, 3 skipped`
- `git diff --check` passed.
- Offscreen wireframe toggle check passed.
  - Wireframe is enabled by default.
  - Toggling the checkbox updates the viewer state.

Additional verification after filled/wireframe split and Script page:

- Compile check passed:
  - `app/meshing/workspace.py`
  - `app/meshing/viewer/loader.py`
  - `app/meshing/viewer/renderer.py`
  - `app/meshing/viewer/widget.py`
- Focused tests passed:
  - `tests/test_mesh_viewer_loader.py`
  - `tests/test_mesh_artifact.py`
  - `tests/test_mesh_api.py`
  - result: `7 passed`
- Full test suite passed:
  - result: `147 passed, 3 skipped`
- `git diff --check` passed.
- Offscreen Meshing Workspace pages/toggles check passed.
  - Viewer page is default.
  - `Filled` is off by default.
  - `Wireframe` is on by default.
  - Toolbar actions switch between Viewer and Script pages.

Additional verification after marching-squares default script:

- Compile check passed:
  - `app/meshing/workspace.py`
  - `tests/test_meshing_workspace_script.py`
- Focused tests passed:
  - `tests/test_meshing_workspace_script.py`
  - `tests/test_mesh_viewer_loader.py`
  - `tests/test_mesh_artifact.py`
  - `tests/test_mesh_api.py`
  - result: `8 passed`
- Full test suite passed:
  - result: `148 passed, 3 skipped`
- `git diff --check` passed.
- Offscreen default-script check passed.
  - Imported `scene.json`.
  - Ran the workspace default script.
  - Confirmed the preview table contains `triangle` rows.

## Large Preview Freeze Fix

Observed issue:

- A large preview artifact can emit many Arrow record batches.
- The previous viewer path uploaded on every `chunk_loaded` signal.
- Each chunk appended arrays, then rebuilt the full accumulated NumPy array and
  QRhi buffer.
- That made loading effectively O(n^2) on the GUI thread and spammed the log
  with one line per chunk.

Fix:

- `QRhiMeshViewerWidget` now accumulates preview chunks without uploading on
  each chunk.
- The viewer uploads once when `MeshArtifactLoader.finished` arrives.
- Progress logging is throttled to the first chunk and then every 25 chunks.
- A final `Uploading preview to GPU` status marks the one remaining heavy step.

Remaining limitation:

- The current renderer still uses one combined static preview buffer.
- Very large preview limits can still create a noticeable final upload pause.
- The next scalability step is chunk-owned GPU buffers or a lower default/auto
  preview budget, not repeated full-buffer rebuilds.

## Meshing Worker Process

Observed issue:

- Running user meshing scripts with `exec(...)` inside `MeshingWorkspace`
  couples heavy Python meshing work to the GUI process.
- Large or pathological scripts can freeze the UI, contend on the GIL, or crash
  the whole application process.

Architecture update:

```text
MeshingWorkspace GUI process
  writes a JSON job file
  starts python -m app.meshing.worker with QProcess
  consumes JSON-line status messages
  loads the completed Arrow artifact into the viewer

app.meshing.worker process
  reads the job file
  reconstructs MeshableDomains from scene.json
  runs the user script
  writes the Arrow artifact
  emits started/log/done/error messages
```

Implemented:

- Added `app/meshing/script_runner.py`.
  - Owns `MeshScriptEmitter`.
  - Owns `run_meshing_script(...)`.
  - This keeps script execution logic out of the GUI class.
- Added `app/meshing/worker.py`.
  - CLI entry point for `python -m app.meshing.worker JOB`.
  - Emits JSON lines on stdout.
  - Captures user script stdout/stderr so prints do not corrupt the protocol.
- Updated `MeshingWorkspace.run_script()`.
  - Starts a `QProcess` instead of calling `exec(...)` in-process.
  - Disables `Run Script` while the worker is running.
  - Adds `Cancel`, which kills the worker process.
  - On `done`, populates the preview table and opens the Arrow artifact.

Remaining limitation:

- The worker currently reports script completion, logs, and errors.
- It does not yet stream incremental element-count progress while a script is
  running; the next step can expose progress calls through `emit()`.

Verification after moving script execution to a worker process:

- Compile check passed:
  - `app/meshing/script_runner.py`
  - `app/meshing/worker.py`
  - `app/meshing/workspace.py`
  - `tests/test_meshing_worker.py`
- Focused tests passed:
  - `tests/test_meshing_worker.py`
  - `tests/test_meshing_workspace_script.py`
  - `tests/test_mesh_viewer_loader.py`
  - `tests/test_mesh_artifact.py`
  - `tests/test_mesh_api.py`
  - result: `9 passed`
- Full test suite passed:
  - result: `149 passed, 3 skipped`
- `git diff --check` passed.
- Offscreen Meshing Workspace worker run passed.
  - Imported `scene.json`.
  - Started the script through `QProcess`.
  - Waited for process completion through the Qt event loop.
  - Confirmed the Arrow artifact exists and preview rows were populated.

Verification after batching chunk uploads:

- Compile check passed:
  - `app/meshing/viewer/widget.py`
  - `app/meshing/viewer/loader.py`
  - `app/meshing/workspace.py`
  - `tests/test_meshing_workspace_script.py`
- Focused tests passed:
  - `tests/test_meshing_workspace_script.py`
  - `tests/test_mesh_viewer_loader.py`
  - `tests/test_mesh_artifact.py`
  - `tests/test_mesh_api.py`
  - result: `8 passed`
- Full test suite passed:
  - result: `148 passed, 3 skipped`
- `git diff --check` passed.

## Render Preview Cache

Design decision:

- Keep the mesher artifact as semantic Arrow:
  - `element_type`
  - `vertices`
  - `tag_name`
- Add a disposable viewer cache next to the artifact:
  - `mesh.preview-<max_preview_vertices>.arrow`
  - `mesh.preview-<max_preview_vertices>.json`
- The semantic artifact remains the user/solver-facing output.
- The preview cache is only a render acceleration file and can be regenerated.

Preview cache schema:

```text
chunk_id: int64
primitive_type: string        # "triangle" or "line"
position: fixed_size_list<float32, 3>
color: fixed_size_list<float32, 3>
```

Implemented:

- Added `app/meshing/viewer/render_cache.py`.
  - Builds a packed float32 preview cache from the semantic Arrow artifact.
  - Reuses the cache when the source path, source mtime, and preview vertex
    limit still match.
  - Writes a JSON summary with mesh element count, preview triangle count,
    preview wire-edge count, tags, bounds, and truncation state.
- Replaced the old viewer loader conversion path.
  - `app/meshing/viewer/loader.py` now asks for a render cache and streams
    packed triangle/line batches from it.
  - Semantic triangulation/color conversion no longer lives in the loader.
- Updated viewer progress wording.
  - Cache batches are render chunks, so progress reports render triangles and
    wire edges.
  - Final status still reports the semantic mesh element count from the cache
    summary.
- Added `tests/test_mesh_render_cache.py`.
  - Verifies semantic Arrow -> render cache packing.
  - Verifies triangle/line primitive batches.
  - Verifies cache reuse.

Important limitation:

- This is not GPU zero-copy.
- The cache reduces repeated CPU conversion and keeps the viewer path simpler,
  but QRhi still receives CPU-owned float32 data for upload.
- Very high preview limits can still cause one large final upload pause because
  the renderer currently owns one combined static buffer.
- The next scalability step is chunk-owned GPU buffers, so preview cache chunks
  can become independently uploaded/evicted render resources.

Verification after adding the render preview cache:

- Compile check passed:
  - `app/meshing/viewer/render_cache.py`
  - `app/meshing/viewer/loader.py`
  - `app/meshing/viewer/widget.py`
- Focused tests passed:
  - `tests/test_mesh_render_cache.py`
  - `tests/test_mesh_viewer_loader.py`
  - `tests/test_mesh_artifact.py`
  - `tests/test_mesh_api.py`
  - `tests/test_meshing_worker.py`
  - `tests/test_meshing_workspace_script.py`
  - result: `10 passed`
- Full test suite passed:
  - result: `150 passed, 3 skipped`
- `git diff --check` passed.

## Preview Budget Wording

Observed issue:

- The UI exposed `Preview vertices`.
- For triangle artifacts this was confusing because one render triangle needs
  three fill vertices.
- Example: a limit around 4.28 million fill vertices shows about 1.42 million
  render triangles, not 4.28 million triangles.

Fix:

- Renamed the user-facing control to `Max render triangles`.
- The workspace now stores and logs the auto value in render triangles.
- The viewer still converts that number to an internal fill-vertex limit before
  building/loading the render cache.

## Render Cache Worker Process

Observed issue:

- After `Loading mesh artifact...`, the GUI could still freeze before the first
  preview chunk appeared.
- The reason was render-cache generation:
  - semantic Arrow rows were converted to Python lists
  - vertices were packed to float32 render buffers
  - the preview cache Arrow file was written
- This happened in a Python thread inside the GUI process. Even though it was
  not on the Qt main thread, heavy Python/Arrow conversion could still contend
  for the GIL and starve the GUI event loop.

Fix:

- Added `app/meshing/viewer/cache_worker.py`.
  - CLI: `python -m app.meshing.viewer.cache_worker ARTIFACT MAX_PREVIEW_VERTICES`
  - Builds or reuses the render preview cache in a child process.
  - Emits JSON-line status messages.
- Updated `MeshArtifactLoader`.
  - Starts the cache worker with `QProcess`.
  - Streams cached triangle/line chunks only after the worker exits cleanly.
  - Emits status messages for cache preparation and cache readiness.
- Renamed the synchronous helper to `iter_mesh_preview_chunks_sync(...)`.
  - This makes it clear that the helper is for tests/debug/non-GUI use.
  - The GUI path uses the cache worker process.
- The GUI process no longer performs semantic Arrow -> render-cache conversion.

Remaining limitation:

- Streaming the packed cache and uploading to QRhi still happen from the GUI
  process architecture.
- Very large preview limits can still pause during final static-buffer upload.
- The next scalability step remains chunk-owned GPU buffers instead of one
  combined preview buffer.

Verification after moving render-cache generation to a worker process:

- Compile check passed:
  - `app/meshing/viewer/cache_worker.py`
  - `app/meshing/viewer/loader.py`
  - `app/meshing/viewer/render_cache.py`
  - `app/meshing/viewer/widget.py`
- Focused tests passed:
  - `tests/test_mesh_render_cache.py`
  - `tests/test_mesh_viewer_loader.py`
  - `tests/test_mesh_artifact.py`
  - `tests/test_mesh_api.py`
  - `tests/test_meshing_worker.py`
  - `tests/test_meshing_workspace_script.py`
  - result: `11 passed`
- Full test suite passed:
  - result: `151 passed, 3 skipped`

## Render Cache Worker Memory Fix

Observed issue:

- The cache worker could crash with exit code `9`.
- That usually means the OS killed the worker process under memory pressure.
- Root cause:
  - `build_render_cache(...)` called `vertices.to_pylist()` for a whole Arrow
    record batch.
  - It also accumulated converted triangle/line arrays for the whole input
    batch before writing cache chunks.
  - If the meshing script emitted one huge Arrow batch, the cache worker still
    had an unbounded memory spike even though it was outside the GUI process.

Fix:

- `build_render_cache(...)` now reads Arrow rows incrementally from each record
  batch instead of materializing whole columns with `to_pylist()`.
- Added small `_RenderPrimitiveBuffer` buffers for triangle and line cache
  output.
  - Buffers flush to the preview Arrow cache at `max_rows_per_chunk`.
  - Cache memory is bounded by render-cache chunk size, not input artifact
    batch size.
- Added a regression test that writes a larger single input batch and verifies
  the render cache flushes multiple bounded chunks.

Verification after bounding render-cache writer memory:

- Compile check passed:
  - `app/meshing/viewer/render_cache.py`
  - `app/meshing/viewer/cache_worker.py`
  - `app/meshing/viewer/loader.py`
- Focused tests passed:
  - `tests/test_mesh_render_cache.py`
  - `tests/test_mesh_viewer_loader.py`
  - `tests/test_mesh_artifact.py`
  - `tests/test_mesh_api.py`
  - `tests/test_meshing_worker.py`
  - `tests/test_meshing_workspace_script.py`
  - result: `12 passed`
- Full test suite passed:
  - result: `152 passed, 3 skipped`

## Chunk-Owned QRhi Preview Buffers

Observed issue:

- The render cache was chunked on disk, but the widget still accumulated all
  preview chunks in Python lists.
- At load completion it combined them with `np.vstack(...)` and uploaded one
  huge static QRhi buffer.
- This created large temporary CPU copies and one heavy final GPU upload.

Fix:

- `QRhiMeshRenderer` now owns many render chunks instead of one filled buffer
  and one wire buffer.
- Each preview cache chunk becomes its own QRhi vertex buffer.
- `QRhiMeshViewerWidget` streams `MeshPreviewChunk` objects directly into the
  renderer as they arrive.
- Removed the widget-side chunk accumulation and final `np.vstack(...)`.
- Uploaded CPU bytes are released after `uploadStaticBuffer(...)` submits them
  to QRhi.
- Added a per-frame upload budget so large previews are uploaded across
  multiple render frames instead of one large final upload.

Current behavior:

- The cache remains chunked on disk.
- The GUI streams chunks to chunk-owned QRhi buffers.
- The renderer draws all uploaded filled chunks and wire chunks each frame.

Remaining limitation:

- There is still no view-dependent loading or eviction.
- All chunks within the selected preview budget are eventually kept as GPU
  buffers.
- Very high budgets can still consume a lot of GPU memory, especially with
  wireframe enabled.

Verification after adding chunk-owned QRhi buffers:

- Compile check passed:
  - `app/meshing/viewer/renderer.py`
  - `app/meshing/viewer/widget.py`
- Focused tests passed:
  - `tests/test_mesh_render_cache.py`
  - `tests/test_mesh_viewer_loader.py`
  - `tests/test_meshing_workspace_script.py`
  - `tests/test_meshing_worker.py`
  - result: `7 passed`
- Full test suite passed:
  - result: `152 passed, 3 skipped`

## Optional GPU Memory Probe For Auto Budget

Design decision:

- VRAM querying is kept separate from QRhi rendering.
- The renderer remains backend-aware and cross-platform.
- GPU memory information is optional metadata used only by the Meshing
  Workspace `Auto` preview budget.

Implemented:

- Added `app/meshing/viewer/gpu_memory.py`.
  - Defines `GpuMemoryInfo`.
  - Tries `nvidia-smi` on Linux/Windows with a timeout.
  - Reports free and total memory separately.
  - Treats macOS memory as unified memory metadata.
  - Returns `None` when no safe probe is available.
- Updated the Meshing Workspace `Auto` button.
  - Uses GPU free memory when available.
  - Falls back to total GPU memory, then system RAM, then a conservative
    default.
  - Applies a lower budget when wireframe is enabled.
  - Logs the source used, for example `nvidia-smi` or `system-ram`.
- Tied GPU-memory probing to the current QRhi render device.
  - `QRhiMeshViewerWidget` records `driverInfo()` after QRhi initialization.
  - The metadata includes backend name, vendor id, device id, device name, and
    device type.
  - `nvidia-smi` is trusted when the current QRhi render device is NVIDIA or
    the process was launched with the NVIDIA offload environment used by
    `outbin/casocad-nvidia`.
  - If QRhi is rendering on another GPU and no NVIDIA offload launch state is
    present, NVIDIA telemetry is ignored so Auto does not budget for the wrong
    device.
- Added `tests/test_gpu_memory_budget.py`.
  - Verifies free-VRAM based budgeting.
  - Verifies wireframe reduces the budget.
  - Verifies system-RAM fallback.
  - Verifies NVIDIA telemetry is ignored for a non-NVIDIA QRhi render device.
  - Verifies NVIDIA telemetry is used when the QRhi render device matches.
- Verification:
  - full test suite passed: `157 passed, 3 skipped`

Current policy:

- Filled-only previews can use a larger fraction of detected free VRAM.
- Wireframe previews are more conservative because triangle wireframe can add
  about twice the vertex storage of filled triangles.
- No speculative allocation is used.

Follow-up fix:

- The `Max render triangles` spinbox still had the old cap of `6,666,666`.
- That value came from the previous `20,000,000` vertex cap divided by three.
- The spinbox now shares the same `50,000,000` render-triangle maximum used by
  the Auto budget clamp.

Follow-up fix:

- Plain `./outbin/casocad` could still report `nvidia-smi` in Auto logs.
- Root cause:
  - QRhi render-device metadata is only available after QRhi initialization.
  - When Auto ran with unknown render-device metadata, the probe still allowed
    `nvidia-smi`.
- Fix:
  - `nvidia-smi` is now used only when QRhi has reported a known NVIDIA render
    device or when the process carries the NVIDIA offload launch environment.
  - If the QRhi render device is unknown and there is no NVIDIA offload launch
    state, Auto falls back instead of assuming that visible NVIDIA telemetry
    belongs to the active renderer.
- Verification:
  - full test suite passed: `159 passed, 3 skipped`

## Repeated Auto Worker Restart Fix

Observed issue:

- Repeatedly pressing `Auto` could log cache worker crashes with exit code `9`.
- This was not necessarily an OOM condition.
- The loader killed the old cache worker when starting a new load, but stale
  Qt process callbacks could still fire.
- Because the loader stored only one `_cache_process`, an old worker callback
  could interact with the new worker state.

Fix:

- Cache worker callbacks now capture their owning `QProcess`.
- Stale callbacks are ignored unless they belong to the current process.
- Stopping a worker clears the current process before killing it, so later
  stale crash/finished signals are ignored.
- Pressing `Auto` with the same computed limit no longer reloads the current
  artifact.

Verification:

- Focused tests passed:
  - `tests/test_mesh_viewer_loader.py`
  - `tests/test_mesh_render_cache.py`
  - `tests/test_gpu_memory_budget.py`
  - `tests/test_meshing_workspace_script.py`
  - result: `12 passed`
- Full test suite passed:
  - result: `158 passed, 3 skipped`

## Meshing Workspace Close Cleanup

Observed issue:

- After closing the Meshing Workspace, CAD interaction could remain laggy.
- The main window kept a reference to the Meshing Workspace.
- The meshing viewer had no explicit release path for QRhi chunk buffers.
- Closing the window could therefore leave large preview buffers/resources alive
  longer than expected.

Fix:

- `QRhiMeshRenderer.shutdown()` now destroys chunk buffers, pipelines, shader
  resource bindings, and the uniform buffer.
- The renderer asks QRhi to `releaseCachedResources()` after clearing its own
  resources.
- `QRhiMeshViewerWidget.release_resources()` cancels cache loading and shuts
  down the renderer.
- `MeshingWorkspace.closeEvent(...)` cancels any running script worker and
  releases viewer resources.
- `MeshingWorkspace` now uses `WA_DeleteOnClose`.
- `MainWindow` clears its `_meshing_workspace` reference when the workspace is
  destroyed, so reopening creates a fresh workspace.

Verification:

- Compile check passed:
  - `app/meshing/viewer/renderer.py`
  - `app/meshing/viewer/widget.py`
  - `app/meshing/viewer/loader.py`
  - `app/meshing/workspace.py`
  - `app/main_window.py`
- Focused tests passed:
  - `tests/test_mesh_viewer_loader.py`
  - `tests/test_mesh_render_cache.py`
  - `tests/test_meshing_workspace_script.py`
  - `tests/test_gpu_memory_budget.py`
  - result: `13 passed`
- Offscreen Meshing Workspace close smoke passed.
  - Qt reported `QRhiWidget: QRhi is not supported on this platform`, expected
    with `QT_QPA_PLATFORM=offscreen`.
- Full test suite passed:
  - result: `159 passed, 3 skipped`
