# Boundary Region v2 — implementation log

> Plan: `design_docs/boundary_region_v2.md`. All seven phases landed
> 2026-07-02, suite green after each phase (287 tests at completion).

## What landed, per phase

1. **Core schema + classifier** — `BoundaryCut` + `BoundaryRegion.cuts`/`tag`
   (`core/boundary.py`); `core/boundary_region.py` with
   `boundary_region_mask` (on-surface ∧ owner-active ∧ patch scope ∧ cut
   signs), a generic owner-provenance walk (min/max attribution through all
   operators and transforms, ties allowed), scale-relative tolerances
   (`RELATIVE_SURFACE_TOLERANCE · owner diagonal`), and
   `sample_boundary_points` for diagnostics.
   Tests: `tests/test_boundary_region_classifier.py` (obstacle owns the cut
   surface, partition exactness, chain conjunction, 1D ghost extrusion,
   mm-scale tolerance, pyramid pickability).
2. **Document split** — `SceneDocument.split_boundary_region`: children
   partition and REPLACE the parent, ghosts deep-copied into chains (never
   scene objects), legacy single-selector parents converted via
   `_legacy_selector_cut` so old and new cuts compose, empty sides reported.
   Tests: `tests/test_boundary_split.py`.
3. **Serialization + migration** — new self-contained `cuts` record
   (`_ghost_to_record`/`_ghost_from_record`, leaf ghosts only); loader
   migrates legacy volume-selector records into chains one-way and drops
   orphaned `__boundary_selector_*` nodes; writer inlines any in-session
   legacy region so files are always the new format. Interval (2D curve)
   selectors stay legacy-readable per §9.
   Tests: `tests/test_boundary_serialization.py` incl. a hand-written legacy
   file fixture.
4. **Meshing API** — `MeshableBoundaryRegion` (`contains` = the same
   classifier, `owner_sdf`, `selector_sdf`, opaque `tag`) + name-keyed
   `MeshableBoundaryRegions` on `MeshableDomain`. Every region is callable
   now, including direction-only ones the old contract dropped;
   `boundary_tags` kept as the SDF-backed back-compat shim.
5. **Single Boundary Cutter** — one `BoundaryCutter` button (any knife
   shape); ghosts built on a scratch snapshot so they never touch the scene
   graph; segment keeps its half-plane semantics; the "region + object"
   tree path now routes to `split_boundary_region` too. Deleted:
   PlanarCutter/SurfaceCutter buttons, `_mark_internal_boundary_selector`,
   planar/surface kind helpers, `_try_create_boundary_selector_split`.
   Tests: `tests/test_boundary_cutter_ui.py`.
6. **Hover/select tool on QRhi** — `boundary_tool.py` (CPU ray pick against
   the live fluid root, scale-relative pick tolerances); generic
   whole-surface fallback patch in `_surface_patches_for_node` (every 3D
   leaf is hover-selectable — Pyramid, BoxFrame, tubes, generators);
   `_surface_patch_contains` now requires the point on the owner's zero set
   for generic patches; owner-chunk hover highlight via the existing
   selected-highlight path; click tags, Esc cancels, tools are mutually
   exclusive.
7. **Cleanup + docs** — legacy `add_boundary_selector_*` producers marked
   deprecated (kept only for 2D/interval flows and migration tests, §9);
   properties panel shows the region tag (editable, opaque) and the cut
   lineage; decision D10 added to `exact_sdf_decisions.md`.

## Deviations from the plan (deliberate)

- **Hover highlight = classifier-filtered root chunk overlay** (better than
  the planned thin-shell contouring): the Domain's committed display surface
  is filtered by `boundary_region_mask` for the hovered patch, recolored
  bright cyan, lifted slightly along normals, and drawn as an extra chunk
  (`main_window._boundary_patch_highlight_surface`). What lights up is
  literally `contains()`; no meshing per hover, works for every patch kind.
  (First attempt tinted the owner's chunk — invisible for operand owners,
  whose chunks are hidden when Components is off.)
- **`PatchTag`/`domain_surface_provenance`** left as specced in v2 §4 (WALL
  default for Domain provenance); the *region* `tag` is the opaque field.
  The taxonomy question stays open (§12).
- **No `"id"` field** in region records: this format reallocates object ids
  at load for all nodes; names are the stable identity.
- **"Remove last cut"** panel action skipped: children partition the parent,
  so un-cutting one region without its sibling would create overlap; undo
  (Ctrl+Z) restores the parent pair correctly.
- **Commit on mouse press** for the boundary tool (not click-vs-drag
  discrimination); orbit before arming the tool.
- **Hover pick keeps the pre-existing cut-surface priority**: a ray crossing
  both an outer face and an obstacle cut surface returns the obstacle
  (spec-§4 semantics, asserted by
  `test_pick_difference_cut_surface_is_attributed_to_obstacle_patch`) — the
  obstacle wall is otherwise unreachable from outside.

## UX round 2 (post-implementation, user feedback 2026-07-02)

- **Hover resolves EXISTING regions, cut-chain children included**: the hit
  point is classified against every region (`_hovered_boundary_region`, most
  specific match wins — longest chain, then patch scope). Hovering a disk
  child highlights the disk, the status bar names the region, and clicking
  SELECTS it in the tree instead of tagging a duplicate. Untagged patches
  keep the old click-to-tag behavior.
- **Arming the cutter highlights the region about to be cut** (new
  `boundary_cutter_armed` signal → cyan overlay of the selected region).
- **Live split preview while dragging the knife**: the selected region is
  recolored into its would-be children — cyan = inside, orange = outside —
  driven by the same classifier the commit uses
  (`_boundary_cut_preview_surfaces`). Knife ghosts for previews are built on
  an empty throwaway document (cheap per mouse-move).
- Overlay plumbing generalized: `show_boundary_patch_highlight` accepts one
  or many surfaces; Esc/commit/tool-switch all clear it.
- **A missed knife refuses the split** (user feedback): if either side of
  a cut selects no boundary points, `split_boundary_region` raises before
  mutating and the parent survives — no more empty-child + duplicate pairs.
  Validation uses the Domain display-mesh vertices (dense) plus the coarse
  grid fallback, so small-but-real cuts still pass.
- **Scene-panel selection lights regions up too**: selecting BoundaryRegions
  in the tree shows the same classifier overlay (regions have no scene chunk,
  so the per-object selection tint cannot show them), up to four at once,
  cleared on deselect and re-applied after every rebuild
  (`_update_selected_region_highlight` + the artifact-ready hook).

## Follow-ups (tracked in v2 §9)

- 2D domain parity (interval selectors → 2D ghosts), then delete the
  deprecated selector producers and interval record support.
- Patch-shell hover preview; per-domain-kind tag suggestion lists once solid
  domains get region UI.
- Region statistics for meshers (area estimate, sample generators).
