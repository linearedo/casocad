# Old Lattice Mesher Cleanup

## Goal

Remove the old lattice preview mesher dependency from CAD/core code before
building the new Meshing Viewer.

The problem is not only the lattice algorithm itself. `core/mesher/` currently
also owns concepts that are not mesher concepts:

- `FluidDomain`
- boundary/domain tag validation
- SDF boundary attribution helpers
- viewport CPU picking attribution

That makes CAD depend on the old mesher package. The cleanup goal is:

```text
CAD/domain concepts -> neutral core modules
old lattice preview -> removed
new meshing API     -> core/meshing/
new meshing UI      -> app/meshing/
```

## Intended Package Boundaries

```text
core/domain.py
  FluidDomain and domain/tag validation needed by current scene serialization

core/sdf_attribution.py
  SDF owner attribution and picking helpers used by CAD/viewport/boundary logic

core/meshing/
  new generic meshing API and generic Arrow mesh artifact schema

app/meshing/
  new Meshing Workspace and future Mesh Viewer
```

The old `core/mesher/` package should be deleted after its non-mesher
responsibilities are moved.

## Cleanup Order

1. Move `FluidDomain` / `LatticeTag` out of `core/mesher/domain.py`.
2. Move owner-attribution helpers out of `core/mesher/classifier.py`.
3. Remove old lattice preview UI and process wiring:
   - `app/panels/mesher_panel.py`
   - `app/mesher_process.py`
   - old `signals.mesh_requested`, `mesh_progress`, `preview_ready`,
     `mesh_ready` paths where no longer used
4. Delete old lattice implementation:
   - `core/mesher/grid.py`
   - `core/mesher/mesher.py`
   - `core/mesher/resolution.py`
   - old lattice Arrow writer/reader if no longer referenced
5. Update imports/tests.
6. Run full test suite.

## Work Started

- Confirmed the worktree was clean before starting this cleanup.
- Confirmed old `core/mesher/` is still referenced by scene serialization,
  scene document logic, examples, old mesher panel/process, tests, and viewport
  CPU picking.
- Decided not to add the QRhi Meshing Viewer until this cleanup is complete.

## Work In Progress

Moved non-mesher responsibilities out of the old lattice package:

- Added `core/domain.py`.
  - Owns `FluidDomain`.
  - Owns `DomainTag`.
  - CAD/scene/serialization now import `FluidDomain` from `core.domain`.
- Added `core/sdf_attribution.py`.
  - Owns `boundary_owner_ids`.
  - Owns `pick_sdf_surface`.
  - Owns `pick_boundary_owner`.
  - Owns `evaluate_with_attribution`.
  - Owns `evaluate_volume_attribution`.
  - Viewport CPU picking now imports attribution from here.
- Added `core/boundary_selection.py`.
  - Owns `boundary_interval_mask`.
  - Owns `surface_split_selector_mask`.
  - Boundary patch tests now import these helpers from the neutral module.

Removed old lattice UI/process integration from CAD:

- Removed `MesherPanel` from `MainWindow`.
- Removed the right-side old `Mesher` dock.
- Removed `ExportPanel` from `MainWindow`.
- Removed the bottom old `Export` dock.
- Removed old lattice worker/process state from `MainWindow`.
- Removed old mesh/export signal connections from `MainWindow`.
- Removed old lattice process methods from `MainWindow`.
- Removed old lattice worker close guard from `MainWindow`.
- Removed old signals:
  - `mesh_ready`
  - `preview_ready`
  - `export_requested`
  - `mesh_requested`
  - `mesh_progress`

Deleted old app-side lattice files:

- `app/panels/mesher_panel.py`
- `app/panels/export_panel.py`
- `app/mesher_process.py`

Deleted old lattice package files:

- `core/mesher/__init__.py`
- `core/mesher/domain.py`
- `core/mesher/classifier.py`
- `core/mesher/grid.py`
- `core/mesher/mesher.py`
- `core/mesher/resolution.py`
- removed generated `core/mesher/__pycache__`
- removed the empty `core/mesher/` directory

Deleted old lattice Arrow IO:

- `core/io/arrow_writer.py`
- `core/io/arrow_reader.py`
- `core/io/__init__.py` now has no public exports

Removed remaining CAD/viewport lattice hooks:

- Removed the old CAD toolbar `SDF` / `Lattice` mode toggle.
- Removed the empty-scene reset that forced the viewport into `sdf` mode.
- Removed old selection bookkeeping that only existed to build lattice preview
  filters.
- Removed `_attribution_ids()` from `MainWindow`; CAD selection now only drives
  real scene selection and boundary-region highlighting.
- Removed lattice upload methods from `app/viewport/renderer_base.py`.
- Removed `set_lattice_filter` and `append_lattice_preview_chunk` no-op
  compatibility aliases from the QRhi CAD viewport.
- Removed the unused QRhi viewport `set_mode()` state.

Known non-code stale references:

- `docs/reach_feature_parity.md` still mentions the previous
  `set_mode("sdf"/"lattice")` renderer API. It is roadmap/context text, not
  active code. Leave it untouched unless a docs cleanup is requested.

Cleaned active-code language that still referred to the old lattice path:

- Updated `core/gpu_selector.py` parity comment to reference
  `core.boundary_selection.surface_split_selector_mask()`.
- Updated `SceneDocument.set_tag_enabled()` error text from tagging lattice
  nodes to tagging `FluidDomain` boundaries.
- Updated the new `core/meshing/api.py` boundary-region docstring so it no
  longer refers to a current lattice mesher.

At this point, the old lattice preview implementation is removed. Remaining
work is import cleanup, test adjustment, and verification.

## Verification

- Edited-module compile check passed:
  - `app/main_window.py`
  - `app/viewport/renderer_base.py`
  - `app/viewport/renderers/qrhi/viewport.py`
  - `core/domain.py`
  - `core/sdf_attribution.py`
  - `core/boundary_selection.py`
  - `core/scene.py`
  - `core/serialization.py`
  - `core/meshing/api.py`
  - `core/gpu_selector.py`
- Active-code stale-reference scan passed for deleted mesher modules, old
  lattice result types, old Arrow IO, old signals, old panels, and old viewport
  lattice hooks.
- Focused tests passed:
  - `tests/test_boundary_patches.py`
  - `tests/test_mesh_api.py`
  - `tests/test_mesh_artifact.py`
  - result: `54 passed`
- Full test suite passed:
  - result: `144 passed, 3 skipped`
- `git diff --check` passed.
- GUI smoke note:
  - The documented `tests/gui_workflow_smoke.py` file is not present in this
    checkout.
  - Ran an offscreen `MainWindow` construction check instead.
  - Result: passed. Qt reported `QRhiWidget: QRhi is not supported on this
    platform`, which is expected for the offscreen platform; the window still
    constructed and closed cleanly.
