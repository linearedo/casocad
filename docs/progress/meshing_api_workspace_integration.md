# Meshing API Workspace Integration

## Goal

Implement the first vertical slice from `docs/sdf_mesher_integration.md` while
keeping meshing independent from the live CAD document.

The intended flow is:

```text
scene.json -> MeshableDomain -> user mesher script -> Arrow mesh artifact
```

## Package Boundary Decision

Update after cleanup: the old `core/mesher/` package has now been deleted. See
`progress/old_lattice_mesher_cleanup.md` for the detailed removal trace and
verification results.

The current split is:

```text
core/domain.py
  neutral current FluidDomain bridge for scene serialization and CAD logic

core/sdf_attribution.py
  neutral SDF owner attribution and picking helpers

core/boundary_selection.py
  neutral boundary-region selector masks used by CAD/boundary tests

core/meshing/
  new generic meshing integration boundary
  MeshableDomain runtime API
  generic Arrow mesh artifact schema

app/meshing/
  standalone Meshing Workspace
  future QRhi mesh-artifact viewer
```

Historical note from the first integration slice:

There are now two similarly named concepts that must stay separated during the
transition:

```text
core/mesher/
  current/legacy lattice preview implementation
  still used by FluidDomain, SceneDocument, serialization, the old Mesher panel,
  app/mesher_process.py, tests, examples, and viewport CPU picking

core/meshing/
  new generic meshing integration boundary
  MeshableDomain runtime API
  generic Arrow mesh artifact schema
```

Do not mix these layers.

That historical split is now resolved: non-mesher responsibilities moved to the
neutral modules above, and the old lattice package/UI/process path was removed.

Cleanup should happen in this order:

1. keep the new public API in `core/meshing/` - done
2. extract neutral responsibilities from `core/mesher/` - done
3. remove the old Mesher panel/process - done
4. delete `core/mesher/` after nothing imports it - done
5. replace the current `FluidDomain` bridge when the exact-SDF `Model`/`Domain`
   migration lands - still future work

The intended final split is:

```text
core/meshing/
  api.py       # MeshableDomain and scene/domain adapters
  artifact.py  # generic Arrow mesh artifact schema

app/meshing/
  workspace.py # import scene.json, run scripts, write artifacts
  viewer.py    # future QRhi mesh-artifact viewer
  renderer.py  # future QRhi raster renderer
  loader.py    # future async Arrow artifact loader
```

## Implemented

- Added `core.meshing.api.MeshableDomain`.
- Added `core.meshing.api.MeshableDomains`.
  - Supports iteration and integer indexing for compatibility.
  - User scripts should prefer string lookup by exact domain name or unique
    domain kind, for example `domains["fluid"]`.
  - Ambiguous kind lookup raises `KeyError`; use `domains.by_kind(kind)` when
    more than one domain has the same kind.
- Added `core.meshing.api.load_meshable_domains(scene_path)`.
- Current loader reads the existing saved `fluid_domain` from `scene.json` and
  exposes it as one `MeshableDomain` with `kind=("fluid",)`.
- Added `core.meshing.api.sdf_callable(node)` for `(N, 3)` world-coordinate batch
  SDF queries.
- Added SDF-backed boundary tag exposure:
  - direct SDF tag objects are exposed as `(tag.name, tag_sdf)`
  - selector-backed `BoundaryRegion` objects are converted through
    `surface_selector_volume()`
  - owner/directional-only boundary regions are intentionally skipped because
    they are not SDF callables by themselves
- Added `core.meshing.artifact.MeshArtifactWriter`.
- Added `core.meshing.artifact.read_mesh_artifact`.
- Arrow mesh artifact schema:

```text
element_type: string
vertices: list<fixed_size_list<float64, 3>>
tag_name: string
```

- Added standalone `app.meshing.workspace.MeshingWorkspace`.
- Added a top-level `Meshing -> Open Meshing Workspace...` menu action.
- Main window integration is intentionally minimal:
  - imports `MeshingWorkspace`
  - owns one optional workspace window instance
  - opens/raises the workspace from the menu
- Meshing workspace responsibilities:
  - import `scene.json`
  - build `MeshableDomain` objects
  - expose `domains`, `np`, and `emit(...)` to the script editor
  - stream emitted elements to an Arrow artifact
  - show a bounded preview table and output log

## Deliberate Scope Limits

- Existing lattice mesher and old mesher panel were removed in the cleanup
  pass after this first slice.
- No attempt to replace current `FluidDomain` yet; it now lives in
  `core/domain.py` as a neutral bridge.
- Future exact-SDF `Model`/`Domain` classes can feed the same
  `MeshableDomain` contract later.
- The meshing workspace does not mutate the live CAD scene.
- The first workspace preview is a table/log, not a full mesh viewport.
- Script execution is local and experimental; sandboxing/persistence of user
  scripts is not implemented in this slice.

## Tests Added

- `tests/test_mesh_api.py`
  - loads `scene.json`
  - verifies one meshable fluid domain
  - verifies `(N, 3)` SDF batch queries
  - verifies bad point shape rejection
- `tests/test_mesh_artifact.py`
  - verifies Arrow mesh artifact round-trip
  - verifies invalid vertex shape rejection

## Verification

Commands run:

```text
.venv/bin/pytest -q tests/test_mesh_api.py tests/test_mesh_artifact.py
.venv/bin/python -m py_compile core/meshing/api.py core/meshing/artifact.py app/meshing/workspace.py app/main_window.py
.venv/bin/pytest -q
git diff --check
```

Later verification after adding `MeshableDomains` lookup:

```text
.venv/bin/python -m py_compile core/meshing/api.py core/meshing/__init__.py app/meshing/workspace.py tests/test_mesh_api.py
.venv/bin/pytest -q tests/test_mesh_api.py tests/test_mesh_artifact.py tests/test_mesh_viewer_loader.py
.venv/bin/pytest -q
git diff --check
```

Results:

```text
focused: 7 passed
full: 147 passed, 3 skipped
```

Results:

```text
4 passed in focused mesh API/artifact tests
144 passed, 3 skipped in full test suite
git diff --check clean
```

## Next Work

- Add a real mesh visualization viewport inside the Meshing Workspace.
- Add chunk helper utilities for common grid iteration without forcing users to
  allocate all points at once.
- Decide whether emitted Arrow artifacts should default to a user-selected path
  instead of `/tmp/casocad_mesh_workspace.arrow`.
- When the exact-SDF `Model`/`Domain` migration lands, add a second loader path
  that maps those domains to `MeshableDomain` without changing external mesher
  scripts.
