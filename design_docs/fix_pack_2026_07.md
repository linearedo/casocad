# Fix pack — July 2026

Five user-reported fixes, planned together. They touch different layers but
share files, so they are ordered to avoid rework. Each fix is one commit (or
a few small ones), independently shippable, with its own tests.

Status: **planned — not started**.

| # | Report | Real defect | Layer |
| --- | --- | --- | --- |
| 1 | Subtract a tiny 2D shape from a huge one → hole invisible | 2D contouring bails on holes + grid too coarse for small features | `surfaces` |
| 2 | Operand properties don't match the boolean output | `combine` clones operand profiles; source objects go stale | `kernel` + `app` |
| 3 | Add menu has fewer objects than Draw | `add_primitive` only builds 3D kinds | `kernel` + `app` |
| 4 | Polygon is click-only; want add/delete points in properties | Point lists in the panel are edit-only, fixed length | `app` |
| 5 | Bezier surface exists as a cutter knife but can't be drawn | Kernel supports it (`add_point_shape_from_world_points`); the app never exposes it as a tool | `app` |

Recommended order: **Fix 1 → Fix 2 → Fix 3+5 (together) → Fix 4.**
Fixes 1 and 2 are correctness bugs; 3/5 and 4 are feature gaps. 3 and 5
share the "one catalog of kinds" refactor, so they are one phase.

---

## Fix 1 — invisible holes in 2D boolean surfaces

### What actually happens

A 2D boolean is a single `Placed2D` whose profile is `Profile2D::Binary`.
`profile_outline` returns nothing for `Binary`, so display falls through
`placed_2d_outline` (`surfaces/src/profiles2d.rs:704`) to the contoured
path, and two independent defects hide the hole:

1. **Holes are refused outright.** `contoured_placed_2d_surface` calls
   `contour_rings_have_holes` (`profiles2d.rs:564`) and returns `None`
   whenever a ring is nested inside another — because
   `triangulate_simple_polygon` can't triangulate a ring set with holes.
   Every difference that punches a hole drops to
   `sampled_placed_2d_surface`, a filled-cell staircase.
2. **The grid can't see small features.** Both paths sample a uniform grid
   over the whole profile bounds: contoured is capped at
   `MAX_CONTOURED_2D_CELLS = 96` cells (64 at the default resolution 12),
   sampled at 48. A 100 m surface gives ≥ 1 m cells; a 0.2 m hole falls
   inside one cell and produces no crossing at all.

### Changes (all in `surfaces/src/profiles2d.rs`, zero-dep crate — hand-rolled)

1. **Hole-aware triangulation.** New `triangulate_rings_with_holes`:
   classify cleaned rings by even-odd containment depth (even = outer,
   odd = hole, deeper nesting = island recursion). For each outer ring,
   bridge its holes in with the standard keyhole cut (connect each hole's
   max-u vertex to a visible vertex of the outer ring, holes processed in
   descending max-u order), then reuse the existing
   `triangulate_simple_polygon` ear-clipper on the bridged polygon.
   Replace the `contour_rings_have_holes` bail with this; keep the sampled
   path only as the last-resort fallback.
2. **Feature-refined sampling grid.** `marching_squares_rings` already
   takes explicit `us`/`vs` coordinate arrays, so a non-uniform grid is
   legal. New helper `refined_grid_axes(profile, base_cells)`:
   - walk the `Profile2D` tree (`Binary` both sides, `Offset` accumulating
     displacement) collecting each leaf operand's bbox;
   - start from the uniform coarse axes; for every operand whose extent is
     smaller than ~2 coarse cells, splice fine lines (e.g. 32 across the
     padded operand bbox) into both axes;
   - cap total lines (e.g. 256/axis) and dedupe near-identical lines.
   Use it in `contoured_placed_2d_surface` (and drop the 96-cell cap in
   favor of the new total-line cap).
3. **Generalize `marching_squares_rings` to independent axis lengths** —
   it currently assumes `us.len() == vs.len()` (`resolution = us.len()-1`
   indexes both axes). Refinement makes the axes differ.
4. **Optional polish:** Newton-snap ring vertices onto the exact zero set
   before triangulation (the refinement loop already exists in
   `placed_outline_rings`, `profiles2d.rs:792`); share it. This also
   benefits the boundary-region highlight/cutter rings, which use the same
   sampling and have the same small-feature blindness — port
   `refined_grid_axes` into `placed_outline_rings` too.

### Tests (`surfaces/tests`)

- 200×200 rectangle minus r=0.5 circle at center → contoured surface is
  produced (not sampled fallback), no triangle covers the hole center,
  triangle area ≈ rect − circle within tolerance.
- Same at an off-center hole position; union of two distant small shapes;
  nested case (difference inside a difference).
- Regression: existing golden tests still pass (`cargo test -p caso-surfaces`).

---

## Fix 2 — boolean operand properties drift from the output

### What actually happens

For 1D/2D booleans, `SceneDocument::combine` (`kernel/src/scene.rs:729`)
**clones** the operand profiles into a `Profile2D::Binary` on the combined
`Placed2D`, and records the original objects in `sources`. Those source
objects stay alive, are listed as children in the scene tree
(`children()` returns `sources`), and are fully editable — but geometry is
built only from the cloned `Binary` profile. So:

- editing a source (properties, Move/Rotate) changes nothing in the output;
- editing/moving the combined object leaves the sources showing stale
  values — exactly the mismatch reported.

(3D `Operator` booleans reference children by id and are already live —
this fix is 1D/2D only.)

### Design: two-way sync, combined node ⇄ sources

The mapping is bijective by construction:

- `sources[0]` ⇄ `Binary.left`, sharing the combined node's
  origin/axes;
- `sources[1]` ⇄ `Binary.right = Offset { child, offset }`, with
  `source_origin = origin + axis_u·offset[0] + axis_v·offset[1]`;
- nested booleans recurse (a source that is itself combined has its own
  `Binary` + `sources`).

New kernel functions (in `scene.rs`, near `combine`):

- `sync_boolean_down(id)` — after the combined payload changes, rewrite
  each source's payload from the corresponding profile subtree +
  derived origin; recurse into sources that are booleans.
- `sync_boolean_up(source_id)` — after a source payload changes, find
  parents whose `sources` contain it (linear scan is fine), rebuild the
  parent's `Binary` subtree, parent origin := sources[0].origin, right
  offset := projected displacement; then recurse up and finish with a
  `sync_boolean_down` so siblings' derived origins stay consistent.
- 1D (`Profile1D::Binary`/`Offset`) gets the same pair.

Call sites in the app (each already knows the edited id):

- `PropertiesPanel::apply` (`app/src/properties_panel.rs:598`);
- Move/Rotate tool commits in `app/src/tools.rs` (verify which payloads
  they touch);
- `combine` itself runs `sync_boolean_down` once so sources are normalized
  from day one (fixes pre-existing scenes on first edit too).

Axis subtlety: `combine` only checks coplanarity, not axis alignment — a
second operand with rotated in-plane axes is silently re-interpreted in
the first operand's axes (its shape rotates). Down-sync writes the parent
axes into sources, which resolves the display mismatch; leave a test
documenting the behavior.

No file-format change: `profile` and `sources` are both already
serialized, so round-trip parity is untouched.

### Tests (`kernel/tests` + app unit tests)

- combine two rectangles → edit source B's center → parent `Binary` right
  subtree updated, `build_node` eval reflects it.
- move the combined object's origin → sources' origins follow.
- nested combine (3 operands) syncs through both levels.
- undo across a synced edit restores both parent and sources.

---

## Fix 3 — Add menu should offer every object kind

`PRIMITIVE_KINDS` (`app/src/lib.rs:37`) lists only the ten 3D kinds because
`SceneDocument::add_primitive` (`kernel/src/scene.rs:612`) errors on
anything else. The Draw menu meanwhile offers 2D and 1D kinds.

### Changes

1. **Kernel:** extend `add_primitive` with default-sized placed objects on
   the XY plane at the origin (all sizes × `scale`):
   - 2D: `rectangle`, `square`, `circle`, `rounded_rectangle`, `ellipse`,
     `regular_polygon` (6 sides), `polygon` (regular-pentagon points),
     `quadratic_bezier_surface` (closed 3-span blob, odd point count);
   - 1D: `segment`, `polyline` (3 points), `quadratic_bezier_curve`,
     `quadratic_bezier_polycurve` (5 points), all as
     `Placed1D`/`PlacedPolyline1D` like the point tools produce.
2. **One catalog, two menus.** Move the kind lists into a single catalog in
   `app/src/tools.rs` (label, kind, group: 3D/2D/1D, drawable-tool kind)
   consumed by both the Add and Draw menus, so they cannot drift again.
   Add menu gets the same 3D / 2D / 1D group headers as Draw.

### Tests

- kernel: `add_primitive` for every catalog kind succeeds, `build_node`
  succeeds, scene round-trips through serialization.
- app: catalog invariant test — every Draw kind is also an Add kind.

---

## Fix 4 — add/delete points in the properties panel

Point lists (`uv_point_list_ui`, `vec3_point_list_ui` in
`app/src/properties_panel.rs`) only edit existing points.

### Changes

1. New `PointListPolicy` passed by each caller:
   - `Polygon` — insert/delete single points, minimum 3;
   - `Polyline` — single points, minimum 2;
   - `QuadraticBezierPolycurve` / bezier tube / bezier surface — insert or
     delete one *span* (control + anchor pair) to keep the odd count;
     minimums 3 / 3 / 5;
   - `Segment`, `RegularPolygon`, `QuadraticBezierCurve`, plain
     `PolylineTube` points — fixed/exact-count kinds get `Fixed` for
     segment and bezier curve (no buttons), tube polylines allow single
     points, minimum 2.
2. UI: a small "✕" per row (disabled at minimum) and a "+" per row
   inserting after it; new points default to the midpoint of the adjacent
   segment (new bezier spans: midpoint anchor with control at the
   midpoint offset), so the shape doesn't jump. Buttons must work inside
   the virtualized scroll path too. Because the profile point vectors are
   mutated in place, switch those arms from `&mut [..]` to `&mut Vec<..>`.
3. Deletion may make a polygon self-intersecting — same policy as
   dragging a point into self-intersection today: allowed, the SDF eval
   defines the fill (no new validation).
4. Fix 2's `sync_boolean_up` call in `apply` covers point edits on
   boolean operands automatically.

### Tests

- extract the insert/delete span logic into pure helpers, unit-test the
  count invariants (odd counts stay odd, minimums enforced);
- extend `panel_renders_every_payload_kind` to click through the new
  buttons where the harness allows.

---

## Fix 5 — Bezier surface as a drawable object

The kernel already builds it (`add_point_shape_from_world_points`,
`kernel/src/scene.rs:1850`); the icon exists; only the app tools/menus
never expose it.

### Changes (`app/src/tools.rs` + catalog from Fix 3)

1. Add `("Bezier Surface", "quadratic_bezier_surface")` to the 2D
   point-placed kinds (`POINT_KINDS_2D` → catalog).
2. Add it to `needs_odd_points` (`tools.rs:85`) so the status line and
   Enter-commit guidance match the kernel's odd-count rule; mirror
   whatever the polycurve tool does for the live ghost at even counts
   (ghost builds through the same kernel path via `ghost_from_points`,
   so it works at odd counts for free).
3. Status-line text for the tool ("anchor, control, anchor… Enter
   commits").
4. Add-menu default shape comes from Fix 3's catalog work.

### Tests

- tools test: point sequence → commit creates a `Placed2D` with
  `Profile2D::QuadraticBezierSurface`; even count refuses with the odd-count
  message; ghost exists at odd counts.

---

## Verification per phase

- `cargo test --workspace` and `cargo clippy --workspace` clean.
- Manual viewport check per fix (native `cargo run -p caso-app`):
  1. 100 m square minus 0.2 m circle → visible hole, clean outline;
  2. combine → edit operand in tree → output updates; move output →
     operand rows follow;
  3. Add menu shows 3D/2D/1D groups, every entry creates a selectable
     object at the origin;
  4. polygon: add and delete corners from the panel;
  5. draw a bezier surface from the Draw menu, then subtract it.

## Risks / notes

- Fix 1's keyhole bridging must handle hole-touches-outer degeneracies;
  bail to the sampled path on failure (never panic) — behavior then no
  worse than today.
- Fix 2 changes edit semantics of source objects (from dead to live).
  If any script/golden relies on stale sources, tests will surface it.
- Fix 3 widens `add_primitive` beyond "3D primitive"; keep the error
  message for truly unknown kinds and update its wording.
- Parity rule: none of this touches golden-file behavior except where a
  test asserts today's hole-refusing fallback — update those goldens
  deliberately and note it in the commit.
