# Mesh Preview vs. Geometry Opacity (Design Invariant)

**Status:** adopted design rule (2026-07-11); extended with a high-opacity
occlusion regime (2026-07-21).

## The rule

The Meshing-workspace preview overlay's relationship to the geometry opacity
slider has two regimes, split at `MESH_PREVIEW_OCCLUDE_OPACITY = 0.9`
(`render/src/renderer.rs`):

- **Below 90% opacity:** mesh preview is fully **opacity-independent**.
  Opacity applies to display surfaces of the *geometry* only; mesh preview
  elements (nodes, bars, faces-as-wireframe) stay fully visible and drawn on
  top, regardless of how transparent the geometry is. This is the x-ray
  inspection mode described below.
- **At/above 90% opacity:** geometry surfaces write depth and mesh preview
  elements **depth-test against them**, so near-opaque geometry occludes the
  mesh preview elements behind it. This is a visual QA mode: it lets you see
  by eye how well the mesh approximates the true geometry surface.

The two regimes are intentionally different tools and both are deliberate.

## Why the low-opacity regime is opacity-independent

The two visualizations answer different questions:

- **Display surfaces** show *what the geometry is*. Opacity is a tool for
  looking through it.
- **Mesh preview** shows *what the mesher produced* — nodes, bars, faces,
  physics tags. It is an inspection overlay, not part of the geometry.

Because they are decoupled below the threshold, turning geometry opacity
down toward ~0 becomes a first-class inspection workflow: the geometry fades
to a ghost while the mesh lattice stays crisp on top of it. The user can see
interior mesh structure *inside* a solid without any dedicated "x-ray mesh
view" mode — the existing opacity slider already is that mode below 90%.
This interaction is deliberate and must be preserved.

## Why the high-opacity regime occludes

Above 90% opacity the geometry is meant to read as "essentially solid," so
letting it occlude the mesh preview answers a different, equally useful
question: does the mesh actually hug the geometry surface, or does it poke
through / fall short? Seeing the solid hide mesh elements that are outside
it (and reveal ones that are inside it) is a direct visual check of mesh
quality against the true boundary.

## How the pipeline implements both regimes

(`render/src/renderer.rs`)

1. **Surface (triangle) pipelines** — `surface_pipeline` (depth-write) /
   `surface_blend_pipeline` (no depth-write) — the choice between them now
   flips at `MESH_PREVIEW_OCCLUDE_OPACITY` (0.9) instead of the old ~1.0
   cutover, so surfaces start writing depth as soon as the occlusion regime
   kicks in.
2. **Thick-line pipeline** — `line_pipeline` (`shaders/line.wgsl`) has a
   sibling `line_pipeline_depth_test`; both hard-code `alpha = 1.0`, but only
   the depth-test variant reads the depth buffer (`Less`, no write). `render()`
   picks between them using the same `occlude_mesh_preview` flag.
3. **Point markers** (`shaders/point_marker.wgsl`, sphere impostors) have the
   analogous `point_pipeline` / `point_pipeline_depth_test` split.

Mesh preview elements (`app/src/meshing_panel.rs::preview_surfaces`) are
still emitted as **wire-only** `ViewportSurface` chunks with
`object_kind = "mesh_preview"`: 1D elements become segments and 2D/3D
element faces become their edge outlines — no chunk tagging or shader
changes were needed to add the occlusion regime, only pipeline selection in
`render()`.

## Filled mesh faces were dropped deliberately (2026-07-11)

Earlier versions (a carry-over from the Python casoCAD port) also
fan-triangulated face elements into shaded fills. That was removed on
purpose, for two reasons:

1. Wireframe-only is the useful default for FEA/CFD mesh inspection — the
   element structure is the information; shaded fills hide it.
2. Filled faces rode the surface pipeline and inherited the geometry
   opacity slider unconditionally; and a per-tag chunk containing any face
   element caused the renderer to drop the chunk's wire segments entirely
   (`set_scene` uses `wire_indices` only when a chunk has no triangles),
   hiding 1D bars sharing the tag.

Do not reintroduce filled mesh faces casually. If shaded faces ever come
back, they must follow the same two-regime opacity/occlusion behavior as
lines and points above — not ride the surface pipeline's blend-alpha
unconditionally.

## Rules for future work

1. Below `MESH_PREVIEW_OCCLUDE_OPACITY`, never route mesh-preview rendering
   through the geometry opacity uniform or depth buffer. If a new pipeline
   or shader is added for mesh visualization, it gets its own opacity
   control (if any) for that regime, defaulting to fully opaque and
   depth-test-free.
2. Never make mesh preview fade (alpha-blend) with the geometry — only
   occlusion (depth test) is allowed above the threshold. Fading would
   collapse the two regimes into one and break the low-opacity x-ray
   workflow's "always crisp" guarantee.
3. If a per-overlay visibility control is ever needed, it is the existing
   `Preview` checkbox in the Meshing panel (`show_preview`), not the opacity
   slider.
4. Other inspection overlays (boundary-region hover, selection gizmo) still
   follow the original decoupling rule in full — they have no occlusion
   regime and must not fade or occlude with geometry at any opacity.

## Pointers

- `app/src/meshing_panel.rs` — `preview_surfaces()` builds the per-tag
  overlay chunks (`object_kind = "mesh_preview"`).
- `app/src/viewport_panel.rs` — `set_mesh_preview()` /
  `mesh_preview_revision()`; overlay chunks are appended to the scene in
  `upload_scene`.
- `render/src/renderer.rs` — pipeline split; `MESH_PREVIEW_OCCLUDE_OPACITY`
  constant; opacity uniform at `surface_data[16]`; line path via
  `push_line_segment`.
- `render/src/shaders/line.wgsl` — line fragment shader returns
  `vec4(color, 1.0)`.
