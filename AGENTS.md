# AGENTS.md - casoCAD Contributor Instructions

Read this file before modifying casoCAD. Read `PLAN.md` when you need
architecture, product-scope, or roadmap context.

Not change AGENTS.md file nor the PLAN.md file if not explicitly requested.
Add always a confirmation step if user request to change them.

Do not touch/change:
- file that are not related to the user request
- function that have no relation with the scope of request

## Project Overview

casoCAD is a Signed Distance Function based CAD application for analysis
workflows.

The authoritative geometry model is an acyclic graph of 1D, 2D, and 3D SDF
objects. Fluid-domain roots are 2D or 3D; 1D SDFs define boundary regions for
2D domains.
The primary product goal is not generic mesh export or visual CAD alone. The
primary product goal is to author solver-ready analysis models and generate
case outputs for open-source solvers, primarily CFD solvers such as:

- OpenLB
- Palabos
- OpenFOAM
- SU2

Meshing is part of the product, not an external afterthought. casoCAD must own
the path from editable SDF geometry to solver-oriented discrete output.

Current and planned execution paths are:

- CAD visualization from `RenderIR` consumed by viewport backends
- CPU analysis evaluation from `to_numpy()`
- integrated meshing and classification
- typed simulation setup stored as normal Python state
- lattice meshing for OpenLB and Palabos
- future surface and volume meshing for OpenFOAM and SU2
- solver-specific case generators

Core product invariants:

- SDF geometry is the authoritative geometry kernel
- GPU rendering never defines solver geometry
- solver outputs derive from the same canonical SDF scene
- case generators consume the simulation setup and the appropriate mesher
  result
- geometry semantics, region semantics, and solver settings must remain
  separated in code

When requirements are ambiguous:

1. inspect the current code and tests
2. identify the geometry and solver-contract invariants involved
3. choose the least surprising behavior consistent with those invariants
4. update `PLAN.md` if the decision changes durable project direction
5. add or update regression tests for user-visible behavior

## Build And Test Commands

Use the repository launcher and local virtual environment.

Launch the application:

```bash
./outbin/casocad
```

Run the headless test suite:

```bash
.venv/bin/pytest -q
```

Run focused tests while developing:

```bash
.venv/bin/pytest -q tests/test_mesher.py
.venv/bin/pytest -q tests/test_arrow_roundtrip.py
.venv/bin/pytest -q tests/test_scene_document.py
```

Run GUI and rendering smoke tests when relevant:

```bash
.venv/bin/python tests/gui_workflow_smoke.py
.venv/bin/python tests/render_difference_smoke.py
.venv/bin/python tests/render_object_colors_smoke.py
.venv/bin/python tests/render_transparency_smoke.py
.venv/bin/python tests/render_added_sphere_cavity_smoke.py
.venv/bin/python tests/render_selected_1d_smoke.py
.venv/bin/python tests/render_selected_2d_smoke.py
.venv/bin/python tests/render_selected_boundary_regions_smoke.py
```

Repository hygiene checks:

```bash
git diff --check
```

Use focused tests first, then the full suite for completed work. For geometry,
mesher, export, renderer, or GUI workflow changes, run the relevant smoke tests
before considering the task done.

## Code Style Guidelines

Follow the existing style and keep changes narrow.

- Start Python modules with `from __future__ import annotations`
- Type-annotate all function and method signatures
- Use `numpy.typing.NDArray` for array contracts
- Prefer dataclasses for data-oriented domain objects
- Use `PascalCase` for classes, `snake_case` for functions and variables, and
  `SCREAMING_SNAKE_CASE` for constants
- Keep `core/` free of Qt, OpenGL, and GUI dependencies
- Keep `app/` responsible for widgets, signals, OpenGL, and worker wiring
- Keep `to_numpy()` as the authoritative SDF evaluation path for CPU analysis
  and meshing; viewport backends render the same scene through `RenderIR`
- Use `float64` for CPU analysis and meshing data, and convert to `float32`
  only at GPU upload boundaries
- Prefer vectorized NumPy operations over Python loops for lattice/node work
- Use comments only when mathematical or lifecycle reasoning is not obvious
- Use ASCII unless the file already requires otherwise
- Do not add runtime dependencies beyond the project's allowed stack without a
  deliberate project decision

Design guardrails:

- Do not treat bounding boxes as geometry semantics
- Do not let solver-specific assumptions leak into generic SDF classes
- Do not require one universal post-meshing representation for lattice and
  mesh-based solvers
- Do not treat Arrow as a mandatory layer between casoCAD and every solver
- Do not make the renderer depend on mesher output to define geometry
- Do not silently diverge CAD geometry, mesher geometry, and exported geometry

If you introduce a new geometric operation, it is incomplete until:

- `to_numpy()` is implemented
- parity or regression tests cover the behavior

If you introduce a new solver output capability, keep the split explicit:

- common simulation and boundary-condition state
- the appropriate lattice or mesh result
- solver-specific case generator

No universal intermediate file is required. Case generators may consume Python
result objects directly. Arrow is currently an optional serialization and
interchange format for lattice data.

## Testing Instructions

The minimum expectation for any non-trivial change is:

1. run the most relevant focused tests
2. run `.venv/bin/pytest -q`
3. run relevant GUI or rendering smoke tests when the change affects viewport,
   export workflow, or solver-facing meshing behavior
4. run `git diff --check`

Testing priorities by change type:

- SDF primitives, booleans, transforms, or scene logic:
  add or update headless tests
- mesher or export logic:
  add numerical assertions on classification, ownership, tags, metadata, and
  file round-trip behavior
- GUI workflow changes:
  run `tests/gui_workflow_smoke.py`
- rendering changes:
  run the relevant framebuffer smoke tests
- solver case-generator changes:
  add generator-level tests for emitted case structure and mapping semantics

Definition of done:

- geometry behavior is mathematically coherent
- CAD and CPU analysis paths agree on the same SDF scene
- meshing remains bounded in memory
- exported semantics remain stable or are intentionally versioned
- tests cover the regression or feature
- relevant smoke checks pass

If you could not run a required test, state that explicitly in your handoff.
