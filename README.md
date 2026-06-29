# casoCAD 0.1.2

casoCAD is an experimental SDF-based CAD and lattice-grid generator.

The application provides:

- A persistent scene with placed 1D segments, 2D profiles, and 3D primitives.
- Scene-tree creation, selection, renaming, editing, and deletion.
- Union, intersection, difference, and smooth-union CSG operations.
- Nested CSG graph editing: any operation result can feed another operation.
- Translate, rotate, and scale controls for moving and editing geometry.
- Extrude, revolve, and straight sweep from placed 2D sections.
- Versioned JSON scene open/save.
- A real-time ModernGL raymarched SDF preview.
- Coloured component/X-ray overlays for enclosed and subtracted SDF operands.
- Editable geometry dimensions and positions in SI meters.
- Orbit, pan, zoom, scene framing, an infinite adaptive reference grid, and a camera-axis gizmo.
- Chunked CPU lattice classification.
- One strictly uniform lattice spacing; refinement is disabled and `level` is always `0`.
- Apache Arrow IPC export for 2D and 3D lattices with JSON metadata.
- Refined per-edge SDF boundary crossings for directional ownership, normals,
  and dimension-aware boundary-region assignment while retaining a uniform
  solver lattice.
- Error-driven global `dx` reduction with maximum, mean, RMS, and
  95th-percentile boundary-approximation statistics.
- A standalone **Mesh Preview** action and a separate persistent Arrow export.
- Stable per-object colors in both the CAD raymarch view and lattice preview,
  including constituent volumes, subtractive boundaries, and tagged regions.
- Adjustable CAD surface opacity that reveals internal SDF objects in their
  stable colors without changing geometry or meshing.
- Constructive volume and boundary ownership, including subtracted obstacle
  surfaces without assigning cutter color to retained interior fluid.
- Per-object lattice filtering from the Scene selection.
- Built-in pipe, von Kármán, boolean, placed-tag, and benchmark examples.
- An in-application log panel.

## Run

Python 3.11 or newer and an OpenGL 3.3 capable desktop are required.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
./outbin/casocad
```

Drag in the viewport to orbit, right-drag to pan, and use the mouse wheel to
zoom. casoCAD uses XY as its ground plane and Z as vertical height. Use the
toolbar shape buttons or **Draw on Grid** in the Scene dock, then drag on the
`Z=0` XY reference grid to place and size a new 2D or 3D object. Select one
object and press **Move**, then drag on the same grid to reposition it. Exact
positions, workplanes, and dimensions remain editable in Properties.

Drawing and preview commit behavior:

- Simple drag-create tools may finalize on mouse release when one drag fully
  defines the object.
- Stateful tools such as extrude, revolve, move/rotate previews, boolean
  previews, and multi-point curves keep a transient preview active.
- Stateful previews are committed with **Enter**, so dragging updates the
  preview but does not necessarily finalize the operation.

For booleans, either select two independent nodes and use the toolbar, or
right-click one SDF object and choose **Boolean → Union with / Intersect with /
Subtract from this / Subtract this from** followed by the other object. The two
difference menus make operand order explicit.
The camera gyroscope labels the red X, green Y, and blue Z axes.
The CAD scene starts at 40% shell opacity so internal SDF objects are visible
immediately. Use the **Opacity** slider to adjust transparency or return to
100% opaque rendering. **Components** remains an explicit X-ray toggle.
Boolean operands are not rendered as separate solids: a boolean node appears
as its final result. Transparency reveals only independent top-level scene
objects.

Selecting a boolean result in lattice mode evaluates that boolean SDF over the
preview nodes and displays the result using the boolean object's own color.
When the current Fluid Domain root is used as a boolean operand, the new
boolean result automatically becomes the Fluid Domain root for subsequent
meshing.

Set one 2D or 3D result as the Fluid Domain, enable dimension-compatible tags,
configure the Mesher panel, and click **Mesh Preview**. A 2D domain uses placed
1D segments for boundary regions. A 3D domain uses placed 2D profiles or
owner-based `BoundaryRegion` objects. Selecting a primitive, CSG subtree, or
tag filters the lattice to its geometric or tag attribution. Use **Mesh and
Export .arrow** for the persistent solver file.

For a 2D domain, choose **Segment 1D** and place or edit the segment on the
domain boundary, or create a directional boundary SDF from a 2D owner. The
current affine placement represents straight segments; curved 1D embeddings
will require future arc or path placement types.

For a 3D domain, choose **Boundary Region** in the toolbar, move over the final
Fluid Domain surface, and click. casoCAD ray-picks the final SDF and creates an
enabled owner-based tag. Axis-aligned box faces and cylinder caps retain the
clicked face direction; curved hits select the complete final boundary
contribution of that owner.
While the tool is active, final boundary contributions use stable owner colors,
and the candidate region under the cursor is highlighted in yellow before
selection. Camera navigation remains available: left-drag orbits, right-drag
pans, and the mouse wheel zooms. A stationary left click selects the region.

The default `dx=0.08 m` model is deliberately small and exports quickly. Fine
spacing increases the node count cubically.

## Test

```bash
.venv/bin/pytest
```

The native end-to-end GUI workflow can also be checked with:

```bash
.venv/bin/python tests/gui_workflow_smoke.py
```

License: Apache-2.0
