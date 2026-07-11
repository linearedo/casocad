# Mesh Preview Is Opacity-Independent (Design Invariant)

**Status:** adopted design rule (2026-07-11). Originated as an accidental
property of the render pipeline split; explicitly promoted to an intentional
invariant because it turned out to be exactly the right UX.

## The rule

The Meshing-workspace preview overlay is **never affected by the geometry
opacity slider**. Opacity applies to display surfaces of the *geometry*
(the SDF objects) only. Mesh preview elements stay fully visible at any
opacity setting.

## Why this is the right behavior

The two visualizations answer different questions:

- **Display surfaces** show *what the geometry is*. Opacity is a tool for
  looking through it.
- **Mesh preview** shows *what the mesher produced* — nodes, bars, faces,
  physics tags. It is an inspection overlay, not part of the geometry.

Because they are decoupled, turning geometry opacity down to ~0 becomes a
first-class inspection workflow: the geometry fades to a ghost while the mesh
lattice stays crisp on top of it. The user can see interior mesh structure
*inside* a solid without any dedicated "x-ray mesh view" mode — the existing
opacity slider already is that mode. This interaction is deliberate and must
be preserved.

## How the pipeline currently guarantees it

The invariant falls out of the renderer's pipeline split
(`render/src/renderer.rs`):

1. **Surface (triangle) pipelines** — `surface_pipeline` /
   `surface_blend_pipeline` — read the global `opacity` uniform
   (`surface_data[16]`) and switch to alpha blending when the slider drops
   below ~1.0. Only geometry display surfaces are meant to ride this path.
2. **Thick-line pipeline** — `line_pipeline` with `shaders/line.wgsl` —
   hard-codes `alpha = 1.0` and draws without depth writes, on top of the
   surfaces. It never sees the opacity uniform.

Mesh preview elements (`app/src/meshing_panel.rs::preview_surfaces`) are
emitted as `ViewportSurface` chunks with `object_kind = "mesh_preview"`:
point elements become tiny crosses, 1D elements become segments, and 2D/3D
element faces get both fan triangles and wire outlines. Wire-only chunks go
through the line pipeline — hence the opacity immunity the user observed.

## Known gap (to fix, not to relitigate)

The immunity is currently complete only for **wire/point** content. A
per-tag preview chunk that contains any face element has triangles, so it is
drawn through the surface pipeline and **does** inherit the global opacity —
a violation of this rule. The renderer cannot tell chunks apart today
(`SurfaceChunk` drops `object_kind` in `set_scene`).

When mesh-face shading is worked on, the fix must respect the invariant:

- Tag chunks originating from `object_kind == "mesh_preview"` through
  `set_scene`, and
- draw their triangles with **opacity pinned to 1.0** (dedicated pipeline or
  a second uniform slot), independent of the slider.

## Rules for future work

1. Never route mesh-preview rendering through the geometry opacity uniform.
   If a new pipeline or shader is added for mesh visualization, it gets its
   own opacity control (if any), defaulting to fully opaque.
2. Never "fix" the mesh preview to fade with the geometry — the decoupling
   is the feature. If someone reports "mesh stays visible when opacity is
   0", the answer is *working as designed*.
3. If a per-overlay visibility control is ever needed, it is the existing
   `Preview` checkbox in the Meshing panel (`show_preview`), not the opacity
   slider.
4. The same reasoning extends to other inspection overlays (boundary-region
   hover, selection gizmo): overlays communicate tool state and must not
   fade with geometry.

## Pointers

- `app/src/meshing_panel.rs` — `preview_surfaces()` builds the per-tag
  overlay chunks (`object_kind = "mesh_preview"`).
- `app/src/viewport_panel.rs` — `set_mesh_preview()` /
  `mesh_preview_revision()`; overlay chunks are appended to the scene in
  `upload_scene`.
- `render/src/renderer.rs` — pipeline split; opacity uniform at
  `surface_data[16]`; line path via `push_line_segment`.
- `render/src/shaders/line.wgsl` — line fragment shader returns
  `vec4(color, 1.0)`.
