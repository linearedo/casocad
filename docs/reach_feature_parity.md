# Reach Feature Parity — QRhi viewport

Handoff doc for a **fresh conversation**. The moderngl→QRhi migration is done; this
doc is about restoring the viewport's **interaction/creation features** that were
lost when the old `app/viewport/viewport_widget.py` (~3000 lines) was deleted.

Read this top-to-bottom before touching code. It is written to be self-contained.

---

## 0. Hard constraints (do not violate)

1. **Single codebase, single shader source.** The whole point of the QRhi switch is
   ONE unified codebase + ONE shader source. Backend selection (Vulkan/Metal/D3D/GL)
   is a ~3-line config hint (`_choose_api`), never backend-specific branches in code
   or shaders. Never "just force Vulkan" as a fix.
2. **SDF terminology only.** This is an SDF CAD. Call boolean ops "SDF operators".
   Never say "CSG".
3. **Be accurate — verify before claiming.** Especially perf/backend claims. Do your
   own headless verification; don't ask the user to run things you can check yourself.
   Don't guess-and-test on the user's machine.
4. **Resource discipline (QRhi).** ALL GPU resources are built in
   `initialize()`/`set_scene()` (outside the frame). `render()` only records passes.
   Building pipelines/buffers during `render()` corrupts the submit → segfault.
5. **No QOpenGLWidget anywhere.** One graphics API per top-level window; mixing a
   QOpenGLWidget with the QRhiWidget breaks the window. (This is why flow-sim was
   deleted.)

---

## 1. How casoCAD renders (the data flow)

```
Scene tree panel / Draw buttons  ──emit──> app/signals.py Signals
        │                                         │
        ▼                                         ▼
   app/main_window.py handlers (_on_add_primitive, _on_viewport_shape_drawn, …)
        │  mutate core Document, then _publish_document()
        ▼
   _publish_document() renders a RenderIR + tree, calls:
        self.viewport.set_scene_artifact(tree, render_ir)      (main_window.py:511)
        │
        ▼
   QRhiViewportWidget.set_scene_artifact()  →  renderer.set_scene(render_ir)
        │
        ▼
   QRhiInterpreterRenderer: serialize_scene + emit_program → 4 storage buffers,
   fragment raymarcher reads them and draws ONE fullscreen pass to the target.
```

- **The render path WORKS.** The user already sees scenes ("box with the hole").
- **Creation/edit features go through signals → main_window → Document → re-render.**
  So most "missing" features are missing UI/tools in the viewport, NOT missing render.

### Key files (current, QRhi)
| File | Role |
|------|------|
| `app/viewport/renderers/qrhi/viewport.py` | `QRhiViewportWidget` — camera, input, command panel, tool state, picking. **This is where most parity work goes.** |
| `app/viewport/renderers/qrhi/renderer.py` | `QRhiInterpreterRenderer` — fragment raymarch path. Builds buffers/pipeline, records one pass. |
| `app/viewport/renderers/qrhi/raymarch_frag_main.glsl` | Fragment shader `main` + grid + corner axis gizmo + selection highlight. |
| `app/viewport/renderers/qrhi/vulkanize.py` | Moves loose `uniform`s into std140 UBO (binding 15), image→14, version 450. **Regex requires NO trailing comment on `uniform` lines.** |
| `app/viewport/renderers/interpreter_glsl/shader_assembly.py` | `build_program_source` (moderngl-free, KEPT). |
| `app/viewport/renderers/interpreter_glsl/shaders/` | `sdf_core.glsl`, etc. (KEPT). |
| `app/panels/scene_tree.py` | Scene tree panel — **has real "Add SDF" + "Draw" buttons** (lines 86–94). NOT deleted. |
| `app/main_window.py` | All signal handlers, document mutation, `_publish_document`. |

### Shader notes
- Uniforms in `raymarch_frag_main.glsl`: `u_resolution`, `u_camera_position/target/right/up`,
  `u_focal_length`, `u_surface_opacity`, `u_background_color`, `u_show_grid` (int),
  `u_grid_spacing`, `u_grid_plane` (0=XY,1=XZ,2=YZ), `u_selected_object_id`.
- Vulkan `gl_FragCoord` origin is top-left → shader flips Y (`screen_uv.y = -screen_uv.y`).
- `STACK_CAPACITY = 16` (64 caused register pressure).
- Adding a uniform = add it to the shader (no trailing comment on the line), add a
  default in `renderer.py` `_zero_camera()`, and pass it from
  `viewport._camera_values()`. std140 packing is automatic via `uniform_block_members`.

---

## 2. What ALREADY works (tell the user — these are NOT lost)

These are reachable **right now** via the **Scene tree panel** (dock titled "Scene",
left side). If the user says "I can't add an SDF", first confirm they're using these:

- **Add an SDF** — Scene tree panel → **"Add SDF"** button (dropdown menu of primitives).
  Also right-click in the tree → "Add SDF". Emits `add_primitive_requested` →
  `main_window._on_add_primitive` (works: adds to Document, re-renders, selects).
- **Boolean SDF operators** (Union / Intersect / Difference) — right-click selected
  nodes in the scene tree → Boolean menu (`scene_tree.py` ~216–218).
- **Extrude / Revolve** — right-click in scene tree (`scene_tree.py` ~247–248).
- **Draw an SDF** — Scene tree panel → **"Draw"** button (line 93–94). Emits
  `viewport_create_requested`. ⚠️ This arms a viewport draw tool that needs the
  in-viewport drawing interaction (see §3) — the button exists but the viewport-side
  tool may be a stub. VERIFY end-to-end.

**Action item for the fresh conversation:** actually launch the app and click each of
the above. Confirm which truly work vs. which are dead buttons. Don't assume.

---

## 3. What is genuinely MISSING (the parity work)

All of this lived in the deleted `app/viewport/viewport_widget.py`. Recover from git:
```
git show 03da5eb~1:app/viewport/viewport_widget.py        # the full old viewport
git show 03da5eb~1:app/viewport/renderer.py               # old SDFRenderer (gizmo vertices)
```
(`03da5eb` is the commit that deleted them; `~1` is the last version that had them.)

### 3a. In-viewport command panel buttons
Old panel was a centered-bottom `QFrame` overlay with a 2×2 grid:
`Move | PlanarCutter` / `SurfaceCutter | BoundaryRegion`.
- **Move** — DONE (current `_build_command_panel`, button at 0,0).
- **PlanarCutter** — button with a menu (Segment / Polyline / Bezier Polycurve);
  each calls `begin_boundary_cutter_tool("planar", kind)`. Disabled until a
  BoundaryRegion is selected (original behavior).
- **SurfaceCutter** — button with a menu (Sphere / Box / Cylinder / Cone);
  `begin_boundary_cutter_tool("surface", kind)`. Disabled until BoundaryRegion selected.
- **BoundaryRegion** — button, emits `signals.viewport_boundary_tool_requested`.
Old wiring lives around lines 1040–1140 of the old viewport (grep
`_build_command_panel`, `_build_planar_cutter_menu`, `_build_surface_cutter_menu`).

### 3b. In-viewport drawing tools (the "Draw SDF" flow)
- Drag-to-create primitives: emits `viewport_shape_drawn(kind, start, end, params)` →
  `main_window._on_viewport_shape_drawn` (EXISTS, main_window.py:637 — handles planar
  segment cutters, surface cutters, and `add_primitive_from_drag`).
- Point/click create: `viewport_point_shape_drawn`.
- Extrude/revolve by drawing a profile: `viewport_extrude_requested`,
  `viewport_revolve_requested`.
- These need the viewport-side **tool state machine**: arm tool → capture mouse
  start/end → project screen→world on the active grid plane → emit signal.
  Recover the projection + tool logic from the old viewport.

### 3c. Move/rotate/extrude GIZMOS + previews
- Old gizmos were **line geometry** rendered by the old `SDFRenderer`
  (`build_rotation_gizmo_vertices`, `_rotation_gizmo_state`, `_extrude_gizmo_state`).
  The QRhi renderer currently has NO line/overlay pipeline — it only draws the
  fullscreen SDF pass. **This is the one real new piece of infrastructure needed:**
  a second QRhi graphics pipeline that draws colored line lists (gizmos, drag
  previews) over the SDF pass, in the same `render()` (begin a second pass or draw
  after the SDF draw before `endPass`).
- **Move preview** was a temporary offset uniform on the selected object during drag,
  committed on mouse-release. (Current Move emits `viewport_move_requested` per
  mouse-move and commits every frame — works functionally but there's no gizmo and
  the "preview" is just the committed move re-rendering.)
- Screen-projection of a world point for gizmos can be done in the fragment shader
  the same way the corner axis gizmo is (see `axisGizmo` in the .glsl) — anchored at
  the object's projected screen position instead of a fixed corner. Or use the line
  pipeline. Pick one and keep it single-codebase.

### 3d. Camera / nav (status)
- Orbit (drag), zoom (wheel, proportional), grid, corner axis gizmo with X/Y/Z
  labels — DONE.
- **WASD + QE fly** — DONE (commit `1dea716`): W/S forward-back, A/D strafe, Q/E
  down-up (world Z), step ∝ distance, viewport takes StrongFocus. Click viewport to
  focus first. Tune step `0.06` if needed.

---

## 4. Viewport API surface main_window expects

`main_window.py` calls these on `self.viewport`. Verify each is REAL vs stub on
`QRhiViewportWidget` (grep them in `viewport.py`). A missing one = crash; a stub =
silent no-op feature:
```
set_scene_artifact, set_mode("sdf"/"lattice"), set_grid_visible, set_grid_spacing,
configure_grid, configure_default_grid, frame_default_grid, frame_box, grid_spacing,
reset_grid_spacing, set_background_color, set_snap_enabled, set_components_visible,
set_sdf_opacity, begin_move_tool, active_boundary_cutter_tool, has_scene_object_id,
paste_offset, reference_plane_label, set_boolean_preview, begin_revolve_tool,
begin_extrude_tool, begin_boundary_region_tool, begin_boundary_cutter_tool
```
Signals main_window listens to (in `app/signals.py`): `add_primitive_requested`,
`viewport_create_requested`, `viewport_shape_drawn`, `viewport_point_shape_drawn`,
`viewport_extrude_requested`, `viewport_revolve_requested`, `viewport_move_requested`,
`viewport_move_tool_requested`, `viewport_boundary_tool_requested`,
`viewport_scene_object_selected`, `sdf_op_requested`, …

---

## 5. Suggested order of work (smallest risk first)

1. **Audit pass (no code):** launch the app, click every Scene-tree button and
   every menu. Write down what works vs dead. Confirm the §4 API list — flag any
   method that's a silent stub. This replaces guessing.
2. **Command panel buttons (§3a):** add PlanarCutter/SurfaceCutter/BoundaryRegion to
   `_build_command_panel`. BoundaryRegion functional immediately; cutters disabled
   until a BoundaryRegion is selected (match original). Recover wiring from git.
3. **Boundary cutter tool (§3b):** port `begin_boundary_cutter_tool` + the
   screen→world projection so the cutter menus actually do something.
4. **Drawing tools (§3b):** port drag-to-create so the "Draw" button works
   end-to-end (`viewport_shape_drawn` handler already exists in main_window).
5. **QRhi line/overlay pipeline (§3c):** the new infra. Then move/rotate/extrude
   gizmos + drag previews on top of it.

Commit after each step. Keep `progress/switch_to_QRhiWidget_progress.md` updated.

---

## 6. Memory / context the next session should load

Memory files at `/home/edof/.claude/projects/-home-edof-repos-casoCAD/memory/`:
- `interpreter-backend-is-chosen-path.md`
- `qrhi-single-codebase-is-the-point.md`
- `be-accurate-verify-before-claiming.md`
- `use-sdf-not-csg-terminology.md`
- `exact-sdf-migration-state.md`
- `lag-is-glsl-compile-on-gui-thread.md`

Full prior transcript (if needed):
`/home/edof/.claude/projects/-home-edof-repos-casoCAD/bda546c6-2342-40cb-8da1-bf4786671320.jsonl`
