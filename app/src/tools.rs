//! Viewport tools: draw-on-grid creation (drag-sized and point-placed
//! kinds), Move, and Rotate — the port of casoCAD's grid interaction tools.
//! All tools act on the `Z = 0` XY reference grid, like the Python app.

use caso_kernel::boundary_ops::BoundaryPatchHit;
use caso_kernel::scene::{ObjectId, SceneDocument, ScenePayload};
use caso_kernel::sdf::node::{Node, RotationAxis};
use caso_kernel::sdf::primitives_2d::Profile2D;
use caso_kernel::sdf::solid_from_2d::RevolveAxis;
use caso_kernel::vec3::{vec3, Vec3};
use caso_render::OrbitCamera;
use caso_surfaces::profiles2d::profile_outline;
use eframe::egui;

use crate::boundary_tool;
use crate::dimensions::parse_dimension_entry;
use crate::gizmo::{self, GizmoKind};
use crate::state::AppState;

/// Drag-sized creation kinds (one drag fully defines the object).
pub const DRAG_KINDS_3D: [(&str, &str); 8] = [
    ("Box", "box"),
    ("Sphere", "sphere"),
    ("Cylinder", "cylinder"),
    ("Cone", "cone"),
    ("Capped Cone", "capped_cone"),
    ("Pyramid", "pyramid"),
    ("Box Frame", "box_frame"),
    ("Torus", "torus"),
];

pub const DRAG_KINDS_2D: [(&str, &str); 7] = [
    ("Rectangle", "rectangle"),
    ("Circle", "circle"),
    ("Square", "square"),
    ("Rounded Rectangle", "rounded_rectangle"),
    ("Ellipse", "ellipse"),
    ("Regular Polygon", "regular_polygon"),
    ("Polygon", "polygon"),
];

/// Point-placed kinds (click points, Enter commits). All 1D objects are
/// point tools — segments and bezier curves included.
pub const POINT_KINDS: [(&str, &str); 7] = [
    ("Segment 1D", "segment"),
    ("Polyline", "polyline"),
    ("Bezier Curve", "quadratic_bezier_curve"),
    ("Bezier Polycurve", "quadratic_bezier_polycurve"),
    ("Polyline Tube", "polyline_tube"),
    ("Bezier Tube", "quadratic_bezier_tube"),
    ("Polygon (points)", "polygon"),
];

/// Cutter knife kinds: (menu label, kind key). A smooth on-surface polyline
/// knife existed until 2026-07-12 and was removed as unproven — see
/// design_docs/boundary_cutter_exactness.md for its archived design record.
pub const KNIFE_KINDS: [(&str, &str); 3] = [
    ("Segment (drag)", "segment"),
    ("Polygon (points)", "polygon"),
    ("Bezier Surface (points)", "quadratic_bezier_surface"),
];

/// Most points a point-placed kind accepts (the kernel builder requires the
/// exact count for these kinds); None is unlimited.
fn point_capacity(kind: &str) -> Option<usize> {
    match kind {
        "segment" => Some(2),
        "quadratic_bezier_curve" => Some(3),
        _ => None,
    }
}

/// Kinds the kernel only accepts with an odd point count (chained
/// anchor-control-anchor bezier spans).
fn needs_odd_points(kind: &str) -> bool {
    matches!(kind, "quadratic_bezier_polycurve" | "quadratic_bezier_tube")
}

/// Fewest points that define a knife of the given kind.
fn knife_minimum_points(knife: &str) -> usize {
    if knife == "segment" {
        2
    } else {
        3
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    Select,
    CreateDrag(&'static str),
    CreatePoints(&'static str),
    Move,
    Rotate,
    /// Two-click distance annotation, drawn as a viewport overlay (input and
    /// state live in the viewport panel, like Select clicks).
    Measure,
    /// Hover the fluid boundary, click to create/select a region.
    BoundaryRegion,
    /// Split the selected region with a knife of the given kind.
    BoundaryCutter(&'static str),
    /// Place the revolve axis on the section's plane: Enter commits the
    /// dashed default axis, or click origin then direction to re-place it.
    RevolveAxisPick(ObjectId),
}

pub struct ToolState {
    pub kind: ToolKind,
    /// World-space start of the active drag on the grid plane.
    drag_start: Option<Vec3>,
    drag_current: Option<Vec3>,
    /// Screen anchor of the active drag (for overlay + rotate deltas).
    screen_start: Option<egui::Pos2>,
    /// Committed points of an active point tool.
    pub points: Vec<Vec3>,
    /// Revolve angle in degrees, parsed from the toolbar at Revolve-tool
    /// activation and consumed by the commit.
    pub revolve_angle: f64,
    /// Typed-dimension buffer, filled from keystrokes during a create drag.
    pub dimension_text: String,
    move_last: Option<Vec3>,
    rotate_last_x: Option<f32>,
    undo_pushed: bool,
    /// Esc aborted a gesture while the mouse button is still held: drag
    /// events are swallowed until the button is released.
    gesture_aborted: bool,
    /// Gizmo handle being dragged (Move arrows / Rotate rings).
    gizmo_axis: Option<RotationAxis>,
    /// Gizmo handle under the cursor (highlight only).
    gizmo_hover: Option<RotationAxis>,
    /// Previous drag parameter: axis t (Move) or ring angle (Rotate).
    gizmo_last: Option<f64>,
    /// Pivot frozen at gizmo drag start (the axis line / rotation center).
    gizmo_pivot: Option<Vec3>,
    /// Boundary tool: the candidate patch under the cursor.
    pub hover_hit: Option<BoundaryPatchHit>,
    /// Cutter: the knife ghost built from the current gesture.
    pub preview_ghost: Option<Node>,
    /// Create tools: real-geometry ghost of the object that would commit
    /// now, built by the same kernel builder as the commit path.
    pub create_ghost: Option<Node>,
    /// Change detector for the ghost inputs (start, end/last, point count,
    /// typed dimensions) so the ghost only rebuilds when the gesture moves.
    create_ghost_sig: Option<(Vec3, Vec3, usize, String)>,
    /// Bumped whenever hover/preview overlays must be rebuilt.
    pub overlay_revision: u64,
    /// Dense on-surface points from the display mesh (set by the viewport,
    /// used to validate splits).
    pub validation_points: Vec<Vec3>,
}

impl Default for ToolState {
    fn default() -> Self {
        Self {
            kind: ToolKind::Select,
            drag_start: None,
            drag_current: None,
            screen_start: None,
            points: Vec::new(),
            revolve_angle: 360.0,
            dimension_text: String::new(),
            move_last: None,
            rotate_last_x: None,
            undo_pushed: false,
            gesture_aborted: false,
            gizmo_axis: None,
            gizmo_hover: None,
            gizmo_last: None,
            gizmo_pivot: None,
            hover_hit: None,
            preview_ghost: None,
            create_ghost: None,
            create_ghost_sig: None,
            overlay_revision: 0,
            validation_points: Vec::new(),
        }
    }
}

/// World-space ray under a screen position.
pub fn screen_ray(
    camera: &OrbitCamera,
    pos: egui::Pos2,
    rect: egui::Rect,
    pixels_per_point: f32,
) -> (Vec3, Vec3) {
    let scale = pixels_per_point as f64;
    camera.screen_ray(
        (pos.x - rect.min.x) as f64 * scale,
        (pos.y - rect.min.y) as f64 * scale,
        rect.width() as f64 * scale,
        rect.height() as f64 * scale,
    )
}

/// Intersect the screen ray with the `Z = 0` grid plane.
pub fn grid_point(
    camera: &OrbitCamera,
    pos: egui::Pos2,
    rect: egui::Rect,
    pixels_per_point: f32,
) -> Option<Vec3> {
    let scale = pixels_per_point as f64;
    let (origin, direction) = camera.screen_ray(
        (pos.x - rect.min.x) as f64 * scale,
        (pos.y - rect.min.y) as f64 * scale,
        rect.width() as f64 * scale,
        rect.height() as f64 * scale,
    );
    if direction.z.abs() < 1e-12 {
        return None;
    }
    let t = -origin.z / direction.z;
    if t <= 0.0 {
        return None;
    }
    Some(origin + direction * t)
}

impl ToolState {
    pub fn set_tool(&mut self, kind: ToolKind, state: &mut AppState) {
        self.cancel(state);
        self.kind = kind;
        state.status = match kind {
            ToolKind::Select => "Select: click objects, drag orbits".to_string(),
            ToolKind::CreateDrag(create) => format!(
                "Draw {create}: drag on the grid — type dimensions (applied on release), Backspace edits, Esc cancels/exits"
            ),
            ToolKind::CreatePoints(create) => format!(
                "Place {create}: click points — Enter commits, Backspace removes last, Esc cancels/exits"
            ),
            ToolKind::Move => {
                "Move: drag the selection (arrows constrain) — Esc aborts the drag, again exits"
                    .to_string()
            }
            ToolKind::Rotate => {
                "Rotate: drag a ring or drag horizontally — Esc aborts the drag, again exits"
                    .to_string()
            }
            ToolKind::Measure => {
                "Measure: click two points (surfaces snap, grid otherwise) — Backspace/Esc removes the pending point, Delete clears all, Esc again exits".to_string()
            }
            ToolKind::BoundaryRegion => {
                "Boundary Region: hover the Fluid Domain surface, click to tag it".to_string()
            }
            ToolKind::BoundaryCutter(knife) => format!(
                "Cutter ({knife}): place the knife — Enter splits, Backspace removes last, Esc cancels/exits"
            ),
            ToolKind::RevolveAxisPick(_) => {
                "Revolve: dashed line is the axis — Enter revolves now, or click the axis origin then its direction (Esc exits)".to_string()
            }
        };
    }

    pub fn cancel(&mut self, state: &mut AppState) {
        if !self.points.is_empty() || self.drag_start.is_some() {
            state.status = "Tool cancelled".to_string();
        }
        self.drag_start = None;
        self.drag_current = None;
        self.screen_start = None;
        self.points.clear();
        self.dimension_text.clear();
        self.move_last = None;
        self.rotate_last_x = None;
        self.undo_pushed = false;
        self.gesture_aborted = false;
        self.gizmo_axis = None;
        self.gizmo_hover = None;
        self.gizmo_last = None;
        self.gizmo_pivot = None;
        if self.hover_hit.is_some() || self.preview_ghost.is_some() || self.create_ghost.is_some()
        {
            self.overlay_revision += 1;
        }
        self.hover_hit = None;
        self.preview_ghost = None;
        self.create_ghost = None;
        self.create_ghost_sig = None;
    }

    pub fn is_active(&self) -> bool {
        self.kind != ToolKind::Select
    }

    /// Uncommitted input the active tool is holding: placed points, typed
    /// dimensions, a live create drag, or a live Move/Rotate gesture. The
    /// single source of truth for the key grammar's "pending" concept.
    pub fn has_pending(&self) -> bool {
        match self.kind {
            ToolKind::Select | ToolKind::BoundaryRegion => false,
            ToolKind::CreateDrag(_) => {
                self.drag_start.is_some() || !self.dimension_text.is_empty()
            }
            ToolKind::CreatePoints(_)
            | ToolKind::Measure
            | ToolKind::BoundaryCutter(_)
            | ToolKind::RevolveAxisPick(_) => !self.points.is_empty(),
            ToolKind::Move | ToolKind::Rotate => {
                self.undo_pushed
                    || self.gizmo_axis.is_some()
                    || self.move_last.is_some()
                    || self.rotate_last_x.is_some()
            }
        }
    }

    /// Esc rung 1: drop all pending input, keep the tool armed. A live
    /// Move/Rotate gesture reverts through `AppState::abort_to_last_snapshot`
    /// (no redo entry). Returns true when something was cleared, so the
    /// caller's second rung (exit to Select) only fires on an idle tool.
    pub fn clear_pending(&mut self, state: &mut AppState) -> bool {
        if !self.has_pending() {
            return false;
        }
        match self.kind {
            ToolKind::Select | ToolKind::BoundaryRegion => false,
            ToolKind::CreateDrag(_) => {
                if self.drag_start.is_some() {
                    // The button may still be held: swallow the rest of
                    // the drag so releasing it cannot commit.
                    self.gesture_aborted = true;
                }
                self.drag_start = None;
                self.drag_current = None;
                self.screen_start = None;
                self.dimension_text.clear();
                self.set_create_ghost(None, None);
                state.status = "Draw cancelled".to_string();
                true
            }
            ToolKind::CreatePoints(_) => {
                self.points.clear();
                self.set_create_ghost(None, None);
                state.status = "Points cleared".to_string();
                true
            }
            ToolKind::Move | ToolKind::Rotate => {
                if self.undo_pushed {
                    state.abort_to_last_snapshot();
                }
                self.move_last = None;
                self.rotate_last_x = None;
                self.gizmo_axis = None;
                self.gizmo_last = None;
                self.gizmo_pivot = None;
                self.undo_pushed = false;
                self.gesture_aborted = true;
                state.status = "Gesture aborted".to_string();
                true
            }
            ToolKind::Measure => {
                self.points.clear();
                state.status = "Measure point cancelled".to_string();
                true
            }
            ToolKind::BoundaryCutter(_) => {
                self.points.clear();
                self.preview_ghost = None;
                self.overlay_revision += 1;
                state.status = "Knife cleared".to_string();
                true
            }
            ToolKind::RevolveAxisPick(_) => {
                self.points.clear();
                state.status =
                    "Axis origin cleared — Enter revolves around the default axis".to_string();
                true
            }
        }
    }

    /// Enter rung: commit whatever is pending (point-shape commit, cutter
    /// split). Returns true when the key was consumed — including a refused
    /// commit that reported an error and kept the points for adjustment.
    pub fn confirm_pending(&mut self, state: &mut AppState) -> bool {
        match self.kind {
            ToolKind::CreatePoints(_) => self.commit_point_shape(state),
            ToolKind::BoundaryCutter(_) => self.commit_cutter_split(state),
            // Enter always commits a revolve: with no points it uses the
            // default axis the dashed preview is showing.
            ToolKind::RevolveAxisPick(section) => self.commit_revolve(section, None, state),
            _ => false,
        }
    }

    /// Backspace rung: pop the last typed-dimension character (create
    /// drags), else the last placed point. Returns true when consumed.
    pub fn pop_pending(&mut self, state: &mut AppState) -> bool {
        match self.kind {
            ToolKind::CreateDrag(_) => self.dimension_text.pop().is_some(),
            ToolKind::CreatePoints(_) => {
                if self.points.pop().is_some() {
                    state.status = format!("{} point(s) — Enter commits", self.points.len());
                    true
                } else {
                    false
                }
            }
            ToolKind::Measure => {
                if self.points.pop().is_some() {
                    state.status = "Measure point removed".to_string();
                    true
                } else {
                    false
                }
            }
            ToolKind::BoundaryCutter(knife) => {
                if self.points.pop().is_some() {
                    state.status =
                        format!("{} knife point(s) — Enter splits", self.points.len());
                    self.refresh_cutter_ghost(knife, state);
                    true
                } else {
                    false
                }
            }
            ToolKind::RevolveAxisPick(_) => {
                if self.points.pop().is_some() {
                    state.status =
                        "Axis origin removed — Enter revolves around the default axis".to_string();
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Enter commit for a point tool: build the shape from the placed
    /// points. On failure the snapshot rolls back without a redo entry and
    /// the points stay so the user can adjust and retry.
    fn commit_point_shape(&mut self, state: &mut AppState) -> bool {
        let ToolKind::CreatePoints(kind) = self.kind else {
            return false;
        };
        if self.points.is_empty() {
            return false;
        }
        let points = std::mem::take(&mut self.points);
        state.push_undo();
        let result = state
            .document
            .add_point_shape_from_world_points(kind, &points, "xy");
        if let Some(id) = state.report(result, &format!("Created {kind}")) {
            state.select_only(id);
            self.set_create_ghost(None, None);
        } else {
            state.abort_to_last_snapshot();
            self.points = points;
        }
        true
    }

    /// Enter commit for the cutter: split the selected region with the
    /// knife built from the placed points. Refusals (missing fluid root or
    /// region, below-minimum points, empty cut) report on the status line
    /// and keep the points.
    fn commit_cutter_split(&mut self, state: &mut AppState) -> bool {
        let ToolKind::BoundaryCutter(knife) = self.kind else {
            return false;
        };
        if self.points.is_empty() {
            return false;
        }
        let Some(root) = boundary_tool::fluid_root_node(&state.document) else {
            state.status = "Cutter: set a Fluid Domain first".to_string();
            return true;
        };
        let Some(region_id) = state.selected_region else {
            state.status =
                "Cutter: select a boundary region first (Boundary Region tool)".to_string();
            return true;
        };
        let minimum = knife_minimum_points(knife);
        if self.points.len() < minimum {
            state.status = format!("{knife} knife needs at least {minimum} points");
            return true;
        }
        match boundary_tool::cutter_ghost(&root, knife, &self.points) {
            Ok(ghost) => {
                state.push_undo();
                let validation = std::mem::take(&mut self.validation_points);
                let result = state.document.split_boundary_region(
                    region_id,
                    &ghost.node,
                    Some(&validation),
                );
                self.validation_points = validation;
                match result {
                    Ok((inside_id, _outside_id)) => {
                        state.selected_region = Some(inside_id);
                        // Repeat the ghost warnings at commit so the user
                        // knows exactly what was stored.
                        state.status = if ghost.warnings.is_empty() {
                            "Region split".to_string()
                        } else {
                            format!("Region split — {}", ghost.warnings.join("; "))
                        };
                        self.points.clear();
                        self.preview_ghost = None;
                        self.overlay_revision += 1;
                    }
                    Err(error) => {
                        // Empty-cut refusal: the snapshot rolls back.
                        state.abort_to_last_snapshot();
                        state.status = error.to_string();
                    }
                }
            }
            Err(error) => state.status = error,
        }
        true
    }

    /// Rebuild the cutter preview ghost from the current points (dropped
    /// below the knife's minimum); builder warnings/errors reach the status
    /// line at preview time.
    fn refresh_cutter_ghost(&mut self, knife: &'static str, state: &mut AppState) {
        let root = boundary_tool::fluid_root_node(&state.document);
        self.preview_ghost = match root {
            Some(root) if self.points.len() >= knife_minimum_points(knife) => {
                match boundary_tool::cutter_ghost(&root, knife, &self.points) {
                    Ok(ghost) => {
                        // Warnings (planar slice on a curved boundary)
                        // surface at preview time.
                        if !ghost.warnings.is_empty() {
                            state.status = ghost.warnings.join("; ");
                        }
                        Some(ghost.node)
                    }
                    Err(error) => {
                        state.status = error;
                        None
                    }
                }
            }
            _ => None,
        };
        self.overlay_revision += 1;
    }

    /// Whether the active tool owns the primary mouse button (the Boundary
    /// Region hover tool keeps camera navigation available, like Python).
    pub fn blocks_camera(&self) -> bool {
        !matches!(
            self.kind,
            ToolKind::Select
                | ToolKind::Measure
                | ToolKind::BoundaryRegion
                | ToolKind::RevolveAxisPick(_)
        )
    }

    /// Handle viewport input for the active tool. Returns true when the tool
    /// consumed the primary-button interaction (so the caller must not orbit).
    #[allow(clippy::too_many_arguments)]
    pub fn handle_viewport(
        &mut self,
        response: &egui::Response,
        ui: &egui::Ui,
        camera: &OrbitCamera,
        rect: egui::Rect,
        pixels_per_point: f32,
        state: &mut AppState,
    ) -> bool {
        match self.kind {
            // Select and Measure clicks are handled by the viewport panel
            // (they pick against its display surfaces).
            ToolKind::Select | ToolKind::Measure => false,
            ToolKind::CreateDrag(kind) => {
                self.handle_create_drag(kind, response, ui, camera, rect, pixels_per_point, state)
            }
            ToolKind::CreatePoints(kind) => {
                self.handle_create_points(kind, response, ui, camera, rect, pixels_per_point, state)
            }
            ToolKind::Move => {
                self.handle_move(response, ui, camera, rect, pixels_per_point, state)
            }
            ToolKind::Rotate => {
                self.handle_rotate(response, ui, camera, rect, pixels_per_point, state)
            }
            ToolKind::BoundaryRegion => {
                self.handle_boundary_region(response, ui, camera, rect, pixels_per_point, state)
            }
            ToolKind::BoundaryCutter(knife) => self.handle_boundary_cutter(
                knife,
                response,
                ui,
                camera,
                rect,
                pixels_per_point,
                state,
            ),
            ToolKind::RevolveAxisPick(section) => self.handle_revolve_axis(
                section,
                response,
                ui,
                camera,
                rect,
                pixels_per_point,
                state,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_boundary_region(
        &mut self,
        response: &egui::Response,
        ui: &egui::Ui,
        camera: &OrbitCamera,
        rect: egui::Rect,
        pixels_per_point: f32,
        state: &mut AppState,
    ) -> bool {
        let Some(root) = boundary_tool::fluid_root_node(&state.document) else {
            state.status = "Boundary Region: set a Fluid Domain first".to_string();
            self.set_hover(None);
            // Camera stays usable while the tool cannot act.
            return false;
        };
        let pointer = ui.ctx().input(|input| input.pointer.latest_pos());
        let hit = pointer.filter(|pos| rect.contains(*pos)).and_then(|pos| {
            let (origin, direction) = screen_ray(camera, pos, rect, pixels_per_point);
            boundary_tool::pick_patch(&root, origin, direction)
        });
        self.set_hover(hit);
        if response.clicked() {
            if let Some(hit) = self.hover_hit.clone() {
                // Click selects an existing matching region, else creates one.
                let existing = state.document.boundary_regions.iter().find(|region| {
                    region.owner_object_id == hit.owner_object_id
                        && region.patch_id.as_deref() == Some(hit.patch_id.as_str())
                        && region.cuts.is_empty()
                });
                if let Some(region) = existing {
                    state.selected_region = Some(region.object_id);
                    state.status = format!("Selected region {}", region.name);
                } else {
                    state.push_undo();
                    let result = state.document.add_boundary_region(
                        hit.owner_object_id,
                        hit.outside_direction,
                        Some(&hit.patch_id),
                        Some(&hit.patch_type),
                    );
                    if let Some(id) = state.report(result, "Boundary region created") {
                        state.selected_region = Some(id);
                    }
                }
            }
        }
        // Left-drag still orbits while hovering (Python keeps navigation).
        false
    }

    /// Swap the create ghost, bumping the overlay revision only when the
    /// gesture signature actually changed (mirrors `set_hover`).
    fn set_create_ghost(
        &mut self,
        ghost: Option<Node>,
        sig: Option<(Vec3, Vec3, usize, String)>,
    ) {
        if self.create_ghost_sig == sig && self.create_ghost.is_some() == ghost.is_some() {
            return;
        }
        self.overlay_revision += 1;
        self.create_ghost = ghost;
        self.create_ghost_sig = sig;
    }

    fn set_hover(&mut self, hit: Option<BoundaryPatchHit>) {
        let signature = |candidate: &BoundaryPatchHit| {
            (candidate.owner_object_id, candidate.patch_id.clone())
        };
        if self.hover_hit.as_ref().map(&signature) != hit.as_ref().map(&signature) {
            self.overlay_revision += 1;
        }
        self.hover_hit = hit;
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_boundary_cutter(
        &mut self,
        knife: &'static str,
        response: &egui::Response,
        ui: &egui::Ui,
        camera: &OrbitCamera,
        rect: egui::Rect,
        pixels_per_point: f32,
        state: &mut AppState,
    ) -> bool {
        let Some(root) = boundary_tool::fluid_root_node(&state.document) else {
            state.status = "Cutter: set a Fluid Domain first".to_string();
            return false;
        };
        if state.selected_region.is_none() {
            state.status = "Cutter: select a boundary region first (Boundary Region tool)"
                .to_string();
            return false;
        }
        let pointer = ui.ctx().input(|input| input.pointer.latest_pos());
        let pick = |pos: egui::Pos2| -> Option<Vec3> {
            let (origin, direction) = screen_ray(camera, pos, rect, pixels_per_point);
            boundary_tool::pick_surface_point(&root, origin, direction)
                .or_else(|| grid_point(camera, pos, rect, pixels_per_point))
        };
        let mut points_changed = false;
        if knife == "segment" {
            if response.drag_started_by(egui::PointerButton::Primary) {
                if let Some(point) = pointer.and_then(pick) {
                    self.points = vec![point];
                    points_changed = true;
                }
            } else if response.dragged_by(egui::PointerButton::Primary) {
                if let Some(point) = pointer.and_then(pick) {
                    if self.points.len() == 1 {
                        self.points.push(point);
                    } else if let Some(last) = self.points.last_mut() {
                        *last = point;
                    }
                    points_changed = true;
                }
            }
        } else if response.clicked() {
            if let Some(point) = pointer.and_then(pick) {
                self.points.push(point);
                points_changed = true;
                state.status = format!("{} knife point(s) — Enter splits", self.points.len());
            }
        }
        // Rebuild the knife ghost (drives the cyan/orange split preview);
        // Enter/Esc/Backspace are handled by the grammar dispatcher.
        if points_changed {
            self.refresh_cutter_ghost(knife, state);
        }
        // Overlay: knife points + rubber band.
        let painter = ui.painter_at(rect);
        let accent = egui::Color32::from_rgb(255, 80, 200); // magenta knife
        let to_screen = |world: Vec3| project_to_screen(camera, world, rect, pixels_per_point);
        let mut screen_points = Vec::new();
        for point in &self.points {
            if let Some(pos) = to_screen(*point) {
                painter.circle_filled(pos, 3.0, accent);
                screen_points.push(pos);
            }
        }
        for pair in screen_points.windows(2) {
            painter.line_segment([pair[0], pair[1]], egui::Stroke::new(1.5, accent));
        }
        true
    }

    /// Revolve axis pick: the dashed preview always shows the axis the
    /// commit would use — default (section origin + V) with no points, the
    /// picked origin toward the cursor with one. Clicks snap to the section
    /// outline (vertices, edge midpoints, origin); the second click commits.
    #[allow(clippy::too_many_arguments)]
    fn handle_revolve_axis(
        &mut self,
        section: ObjectId,
        response: &egui::Response,
        ui: &egui::Ui,
        camera: &OrbitCamera,
        rect: egui::Rect,
        pixels_per_point: f32,
        state: &mut AppState,
    ) -> bool {
        let Some(frame) = section_frame(&state.document, section) else {
            state.status = "Revolve: the section is gone — tool exited".to_string();
            self.points.clear();
            self.kind = ToolKind::Select;
            return false;
        };
        let pointer = ui
            .ctx()
            .input(|input| input.pointer.latest_pos())
            .filter(|pos| rect.contains(*pos));
        let candidates: Vec<(egui::Pos2, Vec3)> = revolve_snap_points(&frame)
            .into_iter()
            .filter_map(|world| {
                project_to_screen(camera, world, rect, pixels_per_point)
                    .map(|pos| (pos, world))
            })
            .collect();
        let snap_hit = pointer.and_then(|pos| {
            snap_to_candidates(pos, &candidates, REVOLVE_SNAP_THRESHOLD_PX)
        });
        let cursor_point = snap_hit.or_else(|| {
            pointer.and_then(|pos| {
                let (origin, direction) = screen_ray(camera, pos, rect, pixels_per_point);
                ray_plane_point(origin, direction, frame.origin, frame.normal)
            })
        });
        let mut consumed = false;
        if response.clicked() {
            if let Some(point) = cursor_point {
                consumed = true;
                if self.points.is_empty() {
                    self.points.push(point);
                    state.status =
                        "Revolve: click the axis direction — Enter uses the V direction"
                            .to_string();
                } else {
                    let start = self.points[0];
                    let toward = point - start;
                    if toward.length() <= 1e-9 {
                        state.status =
                            "Revolve: the direction point must differ from the origin"
                                .to_string();
                    } else {
                        let unit = toward / toward.length();
                        let direction = snap_axis_direction(
                            unit,
                            frame.axis_u,
                            frame.axis_v,
                            REVOLVE_DIRECTION_SNAP_DEGREES,
                        );
                        self.commit_revolve(section, Some(direction), state);
                        return true;
                    }
                }
            }
        }
        // Preview: exactly the axis a commit right now would use.
        let (axis_origin, axis_direction) = match (self.points.first().copied(), cursor_point)
        {
            (Some(start), Some(cursor)) if (cursor - start).length() > 1e-9 => {
                let unit = (cursor - start) / (cursor - start).length();
                (
                    start,
                    snap_axis_direction(
                        unit,
                        frame.axis_u,
                        frame.axis_v,
                        REVOLVE_DIRECTION_SNAP_DEGREES,
                    ),
                )
            }
            (Some(start), _) => (start, frame.axis_v),
            (None, _) => (frame.origin, frame.axis_v),
        };
        let radial = radial_toward_centroid(
            axis_origin,
            axis_direction,
            frame.normal,
            outline_centroid(&frame),
        );
        let half_length = axis_half_length(
            frame.origin,
            frame.axis_u,
            frame.axis_v,
            frame.profile.bounds(),
            axis_origin,
        );
        let painter = ui.painter_at(rect);
        paint_revolve_axis(
            &painter,
            camera,
            rect,
            pixels_per_point,
            axis_origin,
            axis_direction,
            radial,
            half_length,
            REVOLVE_AXIS_COLOR,
        );
        // Cursor marker: filled when snapped to the outline, ring otherwise.
        if let Some(point) = cursor_point {
            if let Some(pos) = project_to_screen(camera, point, rect, pixels_per_point) {
                if snap_hit.is_some() {
                    painter.circle_filled(pos, 4.5, REVOLVE_AXIS_COLOR);
                } else {
                    painter.circle_stroke(
                        pos,
                        4.5,
                        egui::Stroke::new(1.5, REVOLVE_AXIS_COLOR),
                    );
                }
            }
        }
        consumed
    }

    /// Commit the revolve around the picked (or default) axis. The radial
    /// direction always points from the axis toward the profile centroid so
    /// the swept half-plane is the side holding the profile mass.
    fn commit_revolve(
        &mut self,
        section: ObjectId,
        direction: Option<Vec3>,
        state: &mut AppState,
    ) -> bool {
        let Some(frame) = section_frame(&state.document, section) else {
            state.status = "Revolve: the section is gone — tool exited".to_string();
            self.points.clear();
            self.kind = ToolKind::Select;
            return true;
        };
        let axis_origin = self.points.first().copied().unwrap_or(frame.origin);
        let axis_direction = direction.unwrap_or(frame.axis_v);
        let radial = radial_toward_centroid(
            axis_origin,
            axis_direction,
            frame.normal,
            outline_centroid(&frame),
        );
        state.push_undo();
        let result = state.document.solid_from_2d(
            section,
            "revolve",
            None,
            RevolveAxis::V,
            Some(axis_origin),
            Some(axis_direction),
            Some(radial),
            self.revolve_angle,
        );
        if let Some(id) = state.report(result, "Revolved") {
            state.select_only(id);
            self.points.clear();
            // Not `set_tool`: its cancel() would overwrite the status.
            self.kind = ToolKind::Select;
        } else {
            // Refused commit: roll back the snapshot, keep the picked point
            // so the user can adjust and retry.
            state.abort_to_last_snapshot();
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_create_drag(
        &mut self,
        kind: &'static str,
        response: &egui::Response,
        ui: &egui::Ui,
        camera: &OrbitCamera,
        rect: egui::Rect,
        pixels_per_point: f32,
        state: &mut AppState,
    ) -> bool {
        let pointer = ui.ctx().input(|input| input.pointer.latest_pos());
        self.collect_dimension_keys(ui);
        if self.gesture_aborted {
            // Esc aborted this drag: swallow events until the button lifts.
            if response.drag_stopped_by(egui::PointerButton::Primary)
                || !response.dragged_by(egui::PointerButton::Primary)
            {
                self.gesture_aborted = false;
            }
        } else if response.drag_started_by(egui::PointerButton::Primary) {
            if let Some(pos) = pointer {
                self.drag_start = grid_point(camera, pos, rect, pixels_per_point);
                self.drag_current = self.drag_start;
                self.screen_start = Some(pos);
            }
        } else if response.dragged_by(egui::PointerButton::Primary) {
            if let Some(pos) = pointer {
                if let Some(point) = grid_point(camera, pos, rect, pixels_per_point) {
                    self.drag_current = Some(point);
                }
            }
        } else if response.drag_stopped_by(egui::PointerButton::Primary) {
            if let (Some(start), Some(end)) = (self.drag_start, self.drag_current) {
                // Typed dimensions override the drag extents (full sizes, in
                // the working unit, e.g. "1 x 2").
                let (start, end) =
                    apply_typed_dimensions(start, end, &self.dimension_text, state.unit.factor);
                state.push_undo();
                let result =
                    state
                        .document
                        .add_primitive_from_drag(kind, start, end, state.unit.factor);
                if let Some(id) = state.report(result, &format!("Created {kind}")) {
                    state.select_only(id);
                }
            }
            self.drag_start = None;
            self.drag_current = None;
            self.screen_start = None;
            self.dimension_text.clear();
        }
        // Live geometry ghost: rebuild only when the gesture moved.
        if let (Some(start), Some(current)) = (self.drag_start, self.drag_current) {
            let (ghost_start, ghost_end) =
                apply_typed_dimensions(start, current, &self.dimension_text, state.unit.factor);
            let sig = (ghost_start, ghost_end, 0usize, self.dimension_text.clone());
            if self.create_ghost_sig.as_ref() != Some(&sig) {
                let ghost = ghost_from_drag(kind, ghost_start, ghost_end, state.unit.factor);
                self.set_create_ghost(ghost, Some(sig));
            }
        } else {
            self.set_create_ghost(None, None);
        }
        if let (Some(anchor), Some(pos)) = (self.screen_start, pointer) {
            let painter = ui.painter_at(rect);
            let accent = egui::Color32::from_rgb(74, 168, 255);
            // Screen-space rectangle only as fallback when no geometry
            // ghost could be built.
            if self.create_ghost.is_none() {
                painter.rect_stroke(
                    egui::Rect::from_two_pos(anchor, pos),
                    0.0,
                    egui::Stroke::new(1.0, accent),
                    egui::StrokeKind::Middle,
                );
            }
            if !self.dimension_text.is_empty() {
                painter.text(
                    pos + egui::vec2(6.0, 6.0),
                    egui::Align2::LEFT_TOP,
                    format!("{} {}", self.dimension_text, state.unit.key),
                    egui::FontId::monospace(12.0),
                    accent,
                );
            }
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_create_points(
        &mut self,
        kind: &'static str,
        response: &egui::Response,
        ui: &egui::Ui,
        camera: &OrbitCamera,
        rect: egui::Rect,
        pixels_per_point: f32,
        state: &mut AppState,
    ) -> bool {
        let pointer = ui.ctx().input(|input| input.pointer.latest_pos());
        // Exact-count kinds (segment, bezier curve) refuse extra clicks: the
        // kernel builder demands the exact count, so a surplus point could
        // only fail at Enter.
        let at_capacity =
            point_capacity(kind).is_some_and(|capacity| self.points.len() >= capacity);
        if response.clicked() {
            if let Some(pos) = pointer {
                if let Some(point) = grid_point(camera, pos, rect, pixels_per_point) {
                    if at_capacity {
                        state.status = format!(
                            "{kind} takes exactly {} points — Enter commits, Backspace edits",
                            point_capacity(kind).expect("checked"),
                        );
                    } else {
                        self.points.push(point);
                        let count = self.points.len();
                        // Odd-count kinds can't commit mid-span: say so
                        // instead of promising Enter.
                        state.status = if needs_odd_points(kind) && count.is_multiple_of(2) {
                            format!(
                                "{count} point(s) — click the span's anchor (odd count commits)"
                            )
                        } else {
                            format!("{count} point(s) — Enter commits")
                        };
                    }
                }
            }
        }
        // Enter/Esc/Backspace are handled by the grammar dispatcher.
        // Live geometry ghost from the committed points plus the cursor as
        // a tentative point, so the ghost rubber-bands while placing (the
        // cursor stops counting once the kind's capacity is placed).
        let cursor_point = pointer
            .filter(|_| !at_capacity)
            .filter(|pos| rect.contains(*pos))
            .and_then(|pos| grid_point(camera, pos, rect, pixels_per_point));
        let mut tentative = self.points.clone();
        if let Some(point) = cursor_point {
            tentative.push(point);
        }
        if tentative.is_empty() {
            self.set_create_ghost(None, None);
        } else {
            let sig = (
                tentative.first().copied().unwrap_or_default(),
                tentative.last().copied().unwrap_or_default(),
                tentative.len(),
                String::new(),
            );
            if self.create_ghost_sig.as_ref() != Some(&sig) {
                // Below the kind's minimum point count the builder errs and
                // the ghost stays off (the dot painter still shows). For
                // odd-count kinds (bezier polycurve/tube) the tentative
                // cursor point makes the count even — fall back to the
                // committed points so a valid ghost survives while placing.
                let ghost = ghost_from_points(kind, &tentative)
                    .or_else(|| ghost_from_points(kind, &self.points));
                self.set_create_ghost(ghost, Some(sig));
            }
        }
        // Overlay: committed points + rubber band to the cursor.
        let painter = ui.painter_at(rect);
        let accent = egui::Color32::from_rgb(74, 168, 255);
        let to_screen = |world: Vec3| project_to_screen(camera, world, rect, pixels_per_point);
        let mut screen_points = Vec::new();
        for point in &self.points {
            if let Some(pos) = to_screen(*point) {
                painter.circle_filled(pos, 3.0, accent);
                screen_points.push(pos);
            }
        }
        for pair in screen_points.windows(2) {
            painter.line_segment([pair[0], pair[1]], egui::Stroke::new(1.5, accent));
        }
        // No rubber band to the cursor once the kind's capacity is placed —
        // it would suggest another point can be added.
        if let (Some(last), Some(cursor)) = (screen_points.last(), pointer) {
            if rect.contains(cursor) && !at_capacity {
                painter.line_segment(
                    [*last, cursor],
                    egui::Stroke::new(1.0, accent.gamma_multiply(0.6)),
                );
            }
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_move(
        &mut self,
        response: &egui::Response,
        ui: &egui::Ui,
        camera: &OrbitCamera,
        rect: egui::Rect,
        pixels_per_point: f32,
        state: &mut AppState,
    ) -> bool {
        let pointer = ui.ctx().input(|input| input.pointer.latest_pos());
        self.update_gizmo_hover(GizmoKind::Move, response, pointer, camera, rect, pixels_per_point, state);
        if self.gesture_aborted {
            // Esc aborted this drag: swallow events until the button lifts.
            if response.drag_stopped_by(egui::PointerButton::Primary)
                || !response.dragged_by(egui::PointerButton::Primary)
            {
                self.gesture_aborted = false;
            }
        } else if response.drag_started_by(egui::PointerButton::Primary) {
            self.undo_pushed = false;
            let pivot = selection_pivot(state);
            if let (Some(axis), Some(pivot), Some(pos)) = (self.gizmo_hover, pivot, pointer) {
                // Constrained drag along the picked arrow's axis.
                let (origin, direction) = screen_ray(camera, pos, rect, pixels_per_point);
                self.gizmo_axis = Some(axis);
                self.gizmo_pivot = Some(pivot);
                self.gizmo_last = gizmo::axis_drag_parameter(
                    origin,
                    direction,
                    pivot,
                    gizmo::axis_direction(axis),
                );
            } else if let Some(pos) = pointer {
                self.move_last = grid_point(camera, pos, rect, pixels_per_point);
            }
        } else if response.dragged_by(egui::PointerButton::Primary) {
            if let (Some(axis), Some(pivot), Some(pos)) =
                (self.gizmo_axis, self.gizmo_pivot, pointer)
            {
                let (origin, direction) = screen_ray(camera, pos, rect, pixels_per_point);
                let along = gizmo::axis_direction(axis);
                if let Some(t) = gizmo::axis_drag_parameter(origin, direction, pivot, along) {
                    if let Some(last) = self.gizmo_last {
                        self.apply_move_delta(along * (t - last), state);
                    }
                    self.gizmo_last = Some(t);
                }
            } else if let (Some(last), Some(pos)) = (self.move_last, pointer) {
                if let Some(current) = grid_point(camera, pos, rect, pixels_per_point) {
                    self.apply_move_delta(current - last, state);
                    self.move_last = Some(current);
                }
            }
        } else if response.drag_stopped_by(egui::PointerButton::Primary) {
            self.move_last = None;
            self.gizmo_axis = None;
            self.gizmo_last = None;
            self.gizmo_pivot = None;
            self.undo_pushed = false;
        }
        // The gizmo follows the moved selection.
        if let Some(pivot) = selection_pivot(state) {
            gizmo::paint(
                GizmoKind::Move,
                &ui.painter_at(rect),
                camera,
                pivot,
                rect,
                pixels_per_point,
                self.gizmo_axis.or(self.gizmo_hover),
            );
        }
        true
    }

    fn apply_move_delta(&mut self, delta: Vec3, state: &mut AppState) {
        if delta.length() <= 0.0 || state.selection.is_empty() {
            return;
        }
        if !self.undo_pushed {
            state.push_undo();
            self.undo_pushed = true;
        }
        let ids = state.selection.clone();
        let mut new_selection = Vec::new();
        for id in ids {
            match state.document.move_object(id, delta) {
                Ok(moved) => new_selection.push(moved),
                Err(error) => state.status = error.to_string(),
            }
        }
        state.selection = new_selection;
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_rotate(
        &mut self,
        response: &egui::Response,
        ui: &egui::Ui,
        camera: &OrbitCamera,
        rect: egui::Rect,
        pixels_per_point: f32,
        state: &mut AppState,
    ) -> bool {
        let pointer = ui.ctx().input(|input| input.pointer.latest_pos());
        self.update_gizmo_hover(GizmoKind::Rotate, response, pointer, camera, rect, pixels_per_point, state);
        if self.gesture_aborted {
            // Esc aborted this drag: swallow events until the button lifts.
            if response.drag_stopped_by(egui::PointerButton::Primary)
                || !response.dragged_by(egui::PointerButton::Primary)
            {
                self.gesture_aborted = false;
            }
        } else if response.drag_started_by(egui::PointerButton::Primary) {
            self.undo_pushed = false;
            let pivot = selection_pivot(state);
            if let (Some(axis), Some(pivot), Some(pos)) = (self.gizmo_hover, pivot, pointer) {
                // Constrained rotation about the picked ring's axis; the
                // pivot freezes for the whole gesture.
                let (origin, direction) = screen_ray(camera, pos, rect, pixels_per_point);
                self.gizmo_axis = Some(axis);
                self.gizmo_pivot = Some(pivot);
                self.gizmo_last = gizmo::ring_drag_angle(origin, direction, pivot, axis);
            } else {
                self.rotate_last_x = Some(0.0);
            }
        } else if response.dragged_by(egui::PointerButton::Primary) {
            if let (Some(axis), Some(pivot), Some(pos)) =
                (self.gizmo_axis, self.gizmo_pivot, pointer)
            {
                let (origin, direction) = screen_ray(camera, pos, rect, pixels_per_point);
                if let Some(angle) = gizmo::ring_drag_angle(origin, direction, pivot, axis) {
                    if let Some(last) = self.gizmo_last {
                        let degrees = gizmo::wrap_angle(angle - last).to_degrees();
                        self.apply_rotation(axis, degrees, Some(pivot), state);
                    }
                    self.gizmo_last = Some(angle);
                }
            } else if self.rotate_last_x.is_some() {
                // Fallback: half a degree per pixel, about the world Z axis.
                let angle = (response.drag_delta().x as f64) * 0.5;
                self.apply_rotation(RotationAxis::Z, angle, None, state);
            }
        } else if response.drag_stopped_by(egui::PointerButton::Primary) {
            self.rotate_last_x = None;
            self.gizmo_axis = None;
            self.gizmo_last = None;
            self.gizmo_pivot = None;
            self.undo_pushed = false;
        }
        // Paint at the frozen pivot while dragging (stable rings).
        if let Some(pivot) = self.gizmo_pivot.or_else(|| selection_pivot(state)) {
            gizmo::paint(
                GizmoKind::Rotate,
                &ui.painter_at(rect),
                camera,
                pivot,
                rect,
                pixels_per_point,
                self.gizmo_axis.or(self.gizmo_hover),
            );
        }
        true
    }

    fn apply_rotation(
        &mut self,
        axis: RotationAxis,
        angle_degrees: f64,
        pivot: Option<Vec3>,
        state: &mut AppState,
    ) {
        if angle_degrees.abs() <= 0.0 || state.selection.is_empty() {
            return;
        }
        if !self.undo_pushed {
            state.push_undo();
            self.undo_pushed = true;
        }
        let ids = state.selection.clone();
        for id in ids {
            if let Err(error) = state.document.rotate_object(id, axis, angle_degrees, pivot) {
                state.status = error.to_string();
            }
        }
    }

    /// Refresh the hovered gizmo handle (skipped mid-drag so the armed
    /// handle stays highlighted).
    #[allow(clippy::too_many_arguments)]
    fn update_gizmo_hover(
        &mut self,
        kind: GizmoKind,
        response: &egui::Response,
        pointer: Option<egui::Pos2>,
        camera: &OrbitCamera,
        rect: egui::Rect,
        pixels_per_point: f32,
        state: &AppState,
    ) {
        if self.gizmo_axis.is_some() || response.dragged_by(egui::PointerButton::Primary) {
            return;
        }
        self.gizmo_hover = match (selection_pivot(state), pointer) {
            (Some(pivot), Some(pos)) if rect.contains(pos) => {
                gizmo::hit_test(kind, pos, camera, pivot, rect, pixels_per_point)
            }
            _ => None,
        };
    }

    /// Route digit/unit keystrokes into the typed-dimension buffer while a
    /// create drag is armed (Backspace/Esc editing lives in the grammar
    /// dispatcher: `pop_pending` / `clear_pending`).
    fn collect_dimension_keys(&mut self, ui: &egui::Ui) {
        ui.ctx().input(|input| {
            for event in &input.events {
                if let egui::Event::Text(text) = event {
                    for character in text.chars() {
                        if character.is_ascii_digit()
                            || ".,xX;mkcft'\" +-*/()".contains(character)
                        {
                            self.dimension_text.push(character);
                        }
                    }
                }
            }
        });
    }
}

/// Ghost of a drag-created primitive: run the drag through the SAME kernel
/// builder that commits on mouse-up, on a throwaway document, so the preview
/// can never drift from the committed result (any kind, 1D/2D/3D).
fn ghost_from_drag(kind: &str, start: Vec3, end: Vec3, scale: f64) -> Option<Node> {
    let mut scratch = SceneDocument::new();
    let id = scratch.add_primitive_from_drag(kind, start, end, scale).ok()?;
    scratch.build_node(id).ok()
}

/// Ghost of a point-placed shape (same throwaway-document invariant).
fn ghost_from_points(kind: &str, points: &[Vec3]) -> Option<Node> {
    let mut scratch = SceneDocument::new();
    let id = scratch
        .add_point_shape_from_world_points(kind, points, "xy")
        .ok()?;
    scratch.build_node(id).ok()
}

/// Typed-dimension override for a create drag: full sizes in the working
/// unit (e.g. "1 x 2") replace the drag extents about the drag center.
/// Shared by the commit branch and the live ghost.
fn apply_typed_dimensions(start: Vec3, end: Vec3, text: &str, unit_factor: f64) -> (Vec3, Vec3) {
    if text.is_empty() {
        return (start, end);
    }
    let Ok(values) = parse_dimension_entry(text, unit_factor) else {
        return (start, end);
    };
    let center = (start + end) * 0.5;
    let size_a = values.first().copied().unwrap_or(0.1) * unit_factor;
    let size_b = values.get(1).copied().unwrap_or(size_a / unit_factor) * unit_factor;
    let half = vec3(size_a * 0.5, size_b * 0.5, 0.0);
    (center - half, center + half)
}

/// Center of the combined bounding boxes of the selection (gizmo pivot).
fn selection_pivot(state: &AppState) -> Option<Vec3> {
    let mut low = vec3(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut high = vec3(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for id in &state.selection {
        let Ok(node) = state.document.build_node(*id) else {
            continue;
        };
        let Ok(bounds) = node.bounding_box() else {
            continue;
        };
        low = vec3(
            low.x.min(bounds.x_min),
            low.y.min(bounds.y_min),
            low.z.min(bounds.z_min),
        );
        high = vec3(
            high.x.max(bounds.x_max),
            high.y.max(bounds.y_max),
            high.z.max(bounds.z_max),
        );
    }
    if low.x > high.x {
        return None;
    }
    Some((low + high) * 0.5)
}

/// World → screen projection matching the WGSL vertex path.
pub fn project_to_screen(
    camera: &OrbitCamera,
    world: Vec3,
    rect: egui::Rect,
    pixels_per_point: f32,
) -> Option<egui::Pos2> {
    let basis = camera.basis();
    let relative = world - basis.position;
    let depth = relative.dot(basis.forward);
    if depth <= 1e-9 {
        return None;
    }
    // Exact inverse of `OrbitCamera::screen_ray`: suvx = r·focal/(2·depth).
    let scale = pixels_per_point as f64;
    let width = rect.width() as f64 * scale;
    let height = rect.height() as f64 * scale;
    let px =
        0.5 * width + height * relative.dot(basis.right) * camera.focal / (2.0 * depth);
    let py =
        0.5 * height - height * relative.dot(basis.up) * camera.focal / (2.0 * depth);
    Some(egui::pos2(
        rect.min.x + (px / scale) as f32,
        rect.min.y + (py / scale) as f32,
    ))
}

// ---------------------------------------------------------------------------
// Revolve axis pick: plane picking, snapping, and axis rendering

/// Revolve-axis orange, distinct from the accent blue, measure amber, and
/// knife magenta.
pub(crate) const REVOLVE_AXIS_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 140, 60);

/// Screen-space snap radius to the section outline, in points.
const REVOLVE_SNAP_THRESHOLD_PX: f32 = 12.0;

/// Direction clicks within this angle of the workplane U/V axes snap onto
/// them.
const REVOLVE_DIRECTION_SNAP_DEGREES: f64 = 4.0;

/// The section's workplane and profile, read fresh from the document every
/// frame so undo/redo mid-tool can never leave the tool acting on a stale
/// section.
struct SectionFrame {
    origin: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
    normal: Vec3,
    profile: Profile2D,
}

fn section_frame(document: &SceneDocument, id: ObjectId) -> Option<SectionFrame> {
    let object = document.object(id).ok()?;
    let ScenePayload::Placed2D {
        profile,
        origin,
        axis_u,
        axis_v,
        ..
    } = &object.payload
    else {
        return None;
    };
    let normal = axis_u.cross(*axis_v);
    let length = normal.length();
    if length <= 1e-12 {
        return None;
    }
    Some(SectionFrame {
        origin: *origin,
        axis_u: *axis_u,
        axis_v: *axis_v,
        normal: normal / length,
        profile: profile.clone(),
    })
}

/// Intersect a world ray with an arbitrary plane (the general form of
/// `grid_point`'s Z=0 case). Rejects near-parallel rays and hits behind the
/// ray origin.
pub fn ray_plane_point(
    origin: Vec3,
    direction: Vec3,
    plane_origin: Vec3,
    plane_normal: Vec3,
) -> Option<Vec3> {
    let denominator = direction.dot(plane_normal);
    if denominator.abs() < 1e-12 {
        return None;
    }
    let t = (plane_origin - origin).dot(plane_normal) / denominator;
    if t <= 0.0 {
        return None;
    }
    Some(origin + direction * t)
}

/// World-space snap targets on the section: outline vertices, outline edge
/// midpoints, and the workplane origin.
fn revolve_snap_points(frame: &SectionFrame) -> Vec<Vec3> {
    let outline = profile_outline(&frame.profile, 64);
    let lift = |u: f64, v: f64| frame.origin + frame.axis_u * u + frame.axis_v * v;
    let mut points: Vec<Vec3> = outline.iter().map(|point| lift(point[0], point[1])).collect();
    for index in 0..outline.len() {
        let a = outline[index];
        let b = outline[(index + 1) % outline.len()];
        points.push(lift((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5));
    }
    points.push(frame.origin);
    points
}

/// Nearest projected candidate within the pixel threshold.
pub fn snap_to_candidates(
    cursor: egui::Pos2,
    candidates: &[(egui::Pos2, Vec3)],
    threshold: f32,
) -> Option<Vec3> {
    let mut best: Option<(f32, Vec3)> = None;
    for (pos, world) in candidates {
        let distance = pos.distance(cursor);
        if distance <= threshold && best.is_none_or(|(nearest, _)| distance < nearest) {
            best = Some((distance, *world));
        }
    }
    best.map(|(_, world)| world)
}

/// Snap a unit direction onto ±axis_u/±axis_v when within tolerance, else
/// return it unchanged.
pub fn snap_axis_direction(
    direction: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
    tolerance_degrees: f64,
) -> Vec3 {
    let cos_tolerance = tolerance_degrees.to_radians().cos();
    for candidate in [axis_u, axis_v] {
        let length = candidate.length();
        if length <= 1e-12 {
            continue;
        }
        let unit = candidate / length;
        let alignment = direction.dot(unit);
        if alignment.abs() >= cos_tolerance {
            return unit * alignment.signum();
        }
    }
    direction
}

/// World-space centroid of the section outline: signed-area centroid of the
/// closed polygon, vertex mean when the area degenerates, workplane origin
/// when the outline is empty.
fn outline_centroid(frame: &SectionFrame) -> Vec3 {
    let outline = profile_outline(&frame.profile, 64);
    if outline.is_empty() {
        return frame.origin;
    }
    let mut doubled_area = 0.0;
    let mut weighted_u = 0.0;
    let mut weighted_v = 0.0;
    for index in 0..outline.len() {
        let a = outline[index];
        let b = outline[(index + 1) % outline.len()];
        let cross = a[0] * b[1] - b[0] * a[1];
        doubled_area += cross;
        weighted_u += (a[0] + b[0]) * cross;
        weighted_v += (a[1] + b[1]) * cross;
    }
    let (u, v) = if doubled_area.abs() > 1e-12 {
        (
            weighted_u / (3.0 * doubled_area),
            weighted_v / (3.0 * doubled_area),
        )
    } else {
        let count = outline.len() as f64;
        (
            outline.iter().map(|point| point[0]).sum::<f64>() / count,
            outline.iter().map(|point| point[1]).sum::<f64>() / count,
        )
    };
    frame.origin + frame.axis_u * u + frame.axis_v * v
}

/// In-plane unit perpendicular of the axis, oriented toward the profile
/// centroid — so the revolve's kept half-plane is always the side holding
/// the profile (a centroid exactly on the axis keeps the +perp side, where
/// either choice sweeps the same solid).
pub fn radial_toward_centroid(
    axis_origin: Vec3,
    axis_direction: Vec3,
    plane_normal: Vec3,
    centroid: Vec3,
) -> Vec3 {
    let perpendicular = plane_normal.cross(axis_direction);
    let length = perpendicular.length();
    if length <= 1e-12 {
        return plane_normal;
    }
    let perpendicular = perpendicular / length;
    if (centroid - axis_origin).dot(perpendicular) < 0.0 {
        -perpendicular
    } else {
        perpendicular
    }
}

/// Half-length of the drawn axis line: the farthest lifted profile-bounds
/// corner from the axis origin with margin, floored for tiny profiles.
pub fn axis_half_length(
    section_origin: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
    bounds: (f64, f64, f64, f64),
    axis_origin: Vec3,
) -> f64 {
    let (u_min, u_max, v_min, v_max) = bounds;
    let mut farthest: f64 = 0.0;
    for (u, v) in [(u_min, v_min), (u_min, v_max), (u_max, v_min), (u_max, v_max)] {
        let corner = section_origin + axis_u * u + axis_v * v;
        farthest = farthest.max((corner - axis_origin).length());
    }
    (farthest * 1.5).max(0.5)
}

/// Liang–Barsky clip of a screen segment to a rect (keeps the dash count
/// bounded when a projected endpoint lands far outside the viewport).
fn clip_segment_to_rect(
    a: egui::Pos2,
    b: egui::Pos2,
    rect: egui::Rect,
) -> Option<(egui::Pos2, egui::Pos2)> {
    let delta = b - a;
    let mut enter = 0.0f32;
    let mut exit = 1.0f32;
    for (direction, distance_low, distance_high) in [
        (delta.x, rect.min.x - a.x, rect.max.x - a.x),
        (delta.y, rect.min.y - a.y, rect.max.y - a.y),
    ] {
        if direction.abs() < 1e-9 {
            if distance_low > 0.0 || distance_high < 0.0 {
                return None;
            }
            continue;
        }
        let (mut low, mut high) = (distance_low / direction, distance_high / direction);
        if low > high {
            std::mem::swap(&mut low, &mut high);
        }
        enter = enter.max(low);
        exit = exit.min(high);
        if enter > exit {
            return None;
        }
    }
    Some((a + delta * enter, a + delta * exit))
}

/// Dashed world-space segment: clipped to the camera-front half-space (so a
/// long axis stays visible when one end is behind the camera), projected,
/// clipped to the viewport rect, then dashed in screen space.
pub fn paint_dashed_segment(
    painter: &egui::Painter,
    camera: &OrbitCamera,
    rect: egui::Rect,
    pixels_per_point: f32,
    a: Vec3,
    b: Vec3,
    stroke: egui::Stroke,
) {
    const NEAR: f64 = 1e-3;
    let basis = camera.basis();
    let depth = |point: Vec3| (point - basis.position).dot(basis.forward);
    let (depth_a, depth_b) = (depth(a), depth(b));
    if depth_a <= NEAR && depth_b <= NEAR {
        return;
    }
    let (mut a, mut b) = (a, b);
    if depth_a <= NEAR {
        a = a + (b - a) * ((NEAR - depth_a) / (depth_b - depth_a));
    } else if depth_b <= NEAR {
        b = b + (a - b) * ((NEAR - depth_b) / (depth_a - depth_b));
    }
    let (Some(a), Some(b)) = (
        project_to_screen(camera, a, rect, pixels_per_point),
        project_to_screen(camera, b, rect, pixels_per_point),
    ) else {
        return;
    };
    let Some((a, b)) = clip_segment_to_rect(a, b, rect) else {
        return;
    };
    const DASH: f32 = 8.0;
    const GAP: f32 = 5.0;
    let along = b - a;
    let length = along.length();
    if length < 1e-3 {
        return;
    }
    let unit = along / length;
    let mut travelled = 0.0;
    while travelled < length {
        let end = (travelled + DASH).min(length);
        painter.line_segment([a + unit * travelled, a + unit * end], stroke);
        travelled = end + GAP;
    }
}

/// The revolve-axis glyph, shared by the pick tool's live preview and the
/// selection overlay on committed revolves: dashed axis line with an
/// arrowhead at the +axis end, a solid "kept side" radial tick, and the
/// origin dot.
#[allow(clippy::too_many_arguments)]
pub fn paint_revolve_axis(
    painter: &egui::Painter,
    camera: &OrbitCamera,
    rect: egui::Rect,
    pixels_per_point: f32,
    origin: Vec3,
    axis: Vec3,
    radial: Vec3,
    half_length: f64,
    color: egui::Color32,
) {
    let stroke = egui::Stroke::new(1.5, color);
    paint_dashed_segment(
        painter,
        camera,
        rect,
        pixels_per_point,
        origin - axis * half_length,
        origin + axis * half_length,
        stroke,
    );
    let anchor = project_to_screen(camera, origin, rect, pixels_per_point);
    if let (Some(anchor), Some(tip)) = (
        anchor,
        project_to_screen(camera, origin + axis * half_length, rect, pixels_per_point),
    ) {
        let along = tip - anchor;
        if along.length() > 1e-3 {
            arrow_head(painter, tip, -along.normalized(), stroke);
        }
    }
    if let (Some(anchor), Some(tick)) = (
        anchor,
        project_to_screen(
            camera,
            origin + radial * (half_length * 0.25),
            rect,
            pixels_per_point,
        ),
    ) {
        painter.line_segment([anchor, tick], stroke);
        let along = tick - anchor;
        if along.length() > 1e-3 {
            arrow_head(painter, tick, -along.normalized(), stroke);
        }
    }
    if let Some(anchor) = anchor {
        painter.circle_filled(anchor, 3.5, color);
    }
}

/// Two wing strokes at a line tip; `direction` points inward along the line
/// (unit length). Shared by the measure dimension lines and the revolve
/// axis glyph.
pub(crate) fn arrow_head(
    painter: &egui::Painter,
    tip: egui::Pos2,
    direction: egui::Vec2,
    stroke: egui::Stroke,
) {
    const WING_LENGTH: f32 = 8.0;
    const WING_ANGLE: f32 = 0.45;
    for angle in [WING_ANGLE, -WING_ANGLE] {
        let (sin, cos) = angle.sin_cos();
        let wing = egui::vec2(
            direction.x * cos - direction.y * sin,
            direction.x * sin + direction.y * cos,
        );
        painter.line_segment([tip, tip + wing * WING_LENGTH], stroke);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use caso_kernel::roles::DomainKind;

    /// The ghost invariant: the preview node is byte-for-byte the node the
    /// commit path would produce, for every dimension category.
    #[test]
    fn ghost_matches_committed_node() {
        let start = vec3(-1.0, -0.5, 0.0);
        let end = vec3(2.0, 1.5, 0.0);
        for kind in ["box", "sphere", "torus", "circle", "rectangle", "segment"] {
            let ghost = ghost_from_drag(kind, start, end, 1.0)
                .unwrap_or_else(|| panic!("no ghost for {kind}"));
            let mut document = SceneDocument::new();
            let id = document
                .add_primitive_from_drag(kind, start, end, 1.0)
                .expect(kind);
            let committed = document.build_node(id).expect(kind);
            assert_eq!(
                format!("{ghost:?}"),
                format!("{committed:?}"),
                "ghost drifted from commit for {kind}"
            );
        }
    }

    #[test]
    fn ghost_from_points_matches_committed_node() {
        let points = [
            vec3(0.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 1.0, 0.0),
        ];
        let ghost = ghost_from_points("polygon", &points).expect("polygon ghost");
        let mut document = SceneDocument::new();
        let id = document
            .add_point_shape_from_world_points("polygon", &points, "xy")
            .expect("polygon");
        let committed = document.build_node(id).expect("polygon");
        assert_eq!(format!("{ghost:?}"), format!("{committed:?}"));
    }

    #[test]
    fn ghost_below_minimum_points_is_none() {
        assert!(ghost_from_points("polygon", &[vec3(0.0, 0.0, 0.0)]).is_none());
    }

    /// The click cap mirrors the kernel's exact-count rules: capacity
    /// points build, one more refuses.
    #[test]
    fn point_capacity_matches_kernel_exact_counts() {
        let square = [
            vec3(0.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
        ];
        for (kind, capacity) in [("segment", 2), ("quadratic_bezier_curve", 3)] {
            assert_eq!(point_capacity(kind), Some(capacity));
            assert!(ghost_from_points(kind, &square[..capacity]).is_some());
            assert!(ghost_from_points(kind, &square[..capacity + 1]).is_none());
        }
        assert_eq!(point_capacity("polyline"), None);
        assert!(ghost_from_points("polyline", &square).is_some());
    }

    /// The polycurve point tool: unlimited clicks, valid ghost at odd
    /// counts only, and Enter commits a multi-span curve.
    #[test]
    fn polycurve_ghost_and_capacity() {
        assert_eq!(point_capacity("quadratic_bezier_polycurve"), None);
        let points = [
            vec3(0.0, 0.0, 0.0),
            vec3(1.0, 1.0, 0.0),
            vec3(2.0, 0.0, 0.0),
            vec3(3.0, -1.0, 0.0),
            vec3(4.0, 0.0, 0.0),
        ];
        assert!(ghost_from_points("quadratic_bezier_polycurve", &points).is_some());
        assert!(
            ghost_from_points("quadratic_bezier_polycurve", &points[..4]).is_none(),
            "even count must refuse"
        );
        let mut state = AppState::new(SceneDocument::new());
        let mut tools = ToolState::default();
        tools.set_tool(
            ToolKind::CreatePoints("quadratic_bezier_polycurve"),
            &mut state,
        );
        tools.points = points.to_vec();
        assert!(tools.confirm_pending(&mut state));
        assert_eq!(state.document.roots.len(), 1);
        assert_eq!(state.selection, state.document.roots);
    }

    /// The universal grammar on a point tool: Backspace pops, Enter commits
    /// (tool stays armed, object selected), Esc on an idle tool is not
    /// consumed (so the caller's second rung exits to Select).
    #[test]
    fn point_tool_grammar_ladder() {
        let mut state = AppState::new(SceneDocument::new());
        let mut tools = ToolState::default();
        tools.set_tool(ToolKind::CreatePoints("polyline"), &mut state);
        tools.points = vec![
            vec3(0.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 1.0, 0.0),
        ];
        assert!(tools.pop_pending(&mut state));
        assert_eq!(tools.points.len(), 2);
        assert!(tools.confirm_pending(&mut state));
        assert_eq!(state.document.roots.len(), 1);
        assert_eq!(state.selection, state.document.roots);
        assert!(tools.points.is_empty(), "commit consumes the points");
        assert_eq!(tools.kind, ToolKind::CreatePoints("polyline"), "stays armed");
        assert!(!tools.confirm_pending(&mut state), "nothing pending");
        assert!(!tools.clear_pending(&mut state), "idle: Esc falls to rung 2");
        // A refused commit (below the kind's minimum) keeps the points and
        // leaves no redo entry.
        tools.points = vec![vec3(0.0, 0.0, 0.0)];
        assert!(tools.confirm_pending(&mut state), "refusal still consumes Enter");
        assert_eq!(tools.points.len(), 1, "points kept for adjustment");
        assert_eq!(state.document.roots.len(), 1, "document unchanged");
        assert!(!state.can_redo());
    }

    /// Backspace edits the typed-dimension buffer; Esc clears it (rung 1)
    /// and only an idle tool lets Esc fall through.
    #[test]
    fn create_drag_dimension_backspace_and_escape() {
        let mut state = AppState::new(SceneDocument::new());
        let mut tools = ToolState::default();
        tools.set_tool(ToolKind::CreateDrag("box"), &mut state);
        tools.dimension_text = "1x2".to_string();
        assert!(tools.has_pending());
        assert!(tools.pop_pending(&mut state));
        assert_eq!(tools.dimension_text, "1x");
        assert!(tools.clear_pending(&mut state));
        assert!(tools.dimension_text.is_empty());
        assert!(!tools.has_pending());
        assert!(!tools.clear_pending(&mut state));
    }

    /// Esc during a live Move gesture reverts the document with no redo
    /// entry and swallows the rest of the held drag.
    #[test]
    fn move_abort_reverts_document() {
        let mut state = AppState::new(SceneDocument::new());
        let id = state.document.add_primitive("box", 1.0).unwrap();
        state.select_only(id);
        let before = format!("{:?}", state.document.build_node(id).unwrap());
        let mut tools = ToolState::default();
        tools.set_tool(ToolKind::Move, &mut state);
        tools.apply_move_delta(vec3(1.0, 0.0, 0.0), &mut state);
        assert!(tools.has_pending(), "live gesture is pending input");
        let moved = state.selection[0];
        assert_ne!(
            before,
            format!("{:?}", state.document.build_node(moved).unwrap())
        );
        assert!(tools.clear_pending(&mut state));
        assert!(tools.gesture_aborted, "held drag must be swallowed");
        assert!(!state.can_redo(), "aborted gesture must not be redoable");
        let root = state.document.roots[0];
        assert_eq!(
            before,
            format!("{:?}", state.document.build_node(root).unwrap())
        );
        assert!(!tools.has_pending());
    }

    /// Enter on the cutter is consumed even when the split is refused, and
    /// the knife points survive so the user can adjust and retry.
    #[test]
    fn cutter_enter_below_minimum_keeps_points() {
        let mut state = AppState::new(SceneDocument::new());
        let id = state.document.add_primitive("box", 1.0).unwrap();
        state
            .document
            .set_domain_root(id, DomainKind::Fluid)
            .unwrap();
        let region = state.document.add_boundary_region(id, None, None, None).unwrap();
        state.selected_region = Some(region);
        let mut tools = ToolState::default();
        tools.set_tool(ToolKind::BoundaryCutter("polygon"), &mut state);
        tools.points = vec![vec3(0.5, 0.0, 0.0), vec3(0.0, 0.5, 0.0)];
        assert!(tools.confirm_pending(&mut state), "refusal still consumes Enter");
        assert_eq!(tools.points.len(), 2, "points kept");
        assert_eq!(state.document.boundary_regions.len(), 1, "no split");
        assert!(!state.can_undo(), "no snapshot for a refused split");
        // Esc rung 1 clears the knife; rung 2 is then the caller's.
        assert!(tools.clear_pending(&mut state));
        assert!(tools.points.is_empty());
        assert!(!tools.clear_pending(&mut state));
    }

    /// The Measure pending point lives in `points` and answers the grammar.
    #[test]
    fn measure_pending_pop() {
        let mut state = AppState::new(SceneDocument::new());
        let mut tools = ToolState::default();
        tools.set_tool(ToolKind::Measure, &mut state);
        tools.points.push(vec3(1.0, 2.0, 0.0));
        assert!(tools.has_pending());
        assert!(tools.pop_pending(&mut state));
        assert!(tools.points.is_empty());
        assert!(!tools.pop_pending(&mut state));
        assert!(!tools.has_pending());
    }

    #[test]
    fn ray_plane_point_hits_oblique_plane() {
        let plane_origin = vec3(1.0, 0.0, 0.0);
        let plane_normal = vec3(1.0, 1.0, 0.0);
        let hit = ray_plane_point(vec3(3.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0), plane_origin, plane_normal)
            .expect("hit");
        assert!((hit - vec3(1.0, 0.0, 0.0)).length() < 1e-12);
        // Parallel ray misses; a plane behind the ray origin misses.
        assert!(ray_plane_point(
            vec3(0.0, 0.0, 1.0),
            vec3(1.0, -1.0, 0.0),
            plane_origin,
            plane_normal
        )
        .is_none());
        assert!(ray_plane_point(
            vec3(3.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            plane_origin,
            plane_normal
        )
        .is_none());
    }

    #[test]
    fn snap_prefers_nearest_candidate_within_threshold() {
        let candidates = [
            (egui::pos2(20.0, 0.0), vec3(2.0, 0.0, 0.0)),
            (egui::pos2(8.0, 0.0), vec3(1.0, 0.0, 0.0)),
        ];
        assert_eq!(
            snap_to_candidates(egui::pos2(0.0, 0.0), &candidates, 12.0),
            Some(vec3(1.0, 0.0, 0.0))
        );
        assert_eq!(
            snap_to_candidates(egui::pos2(0.0, 40.0), &candidates, 12.0),
            None
        );
    }

    #[test]
    fn direction_snaps_to_workplane_axes() {
        let axis_u = vec3(1.0, 0.0, 0.0);
        let axis_v = vec3(0.0, 1.0, 0.0);
        let nearly_u = vec3(3.0f64.to_radians().cos(), 3.0f64.to_radians().sin(), 0.0);
        assert_eq!(snap_axis_direction(nearly_u, axis_u, axis_v, 4.0), axis_u);
        let off_u = vec3(10.0f64.to_radians().cos(), 10.0f64.to_radians().sin(), 0.0);
        assert_eq!(snap_axis_direction(off_u, axis_u, axis_v, 4.0), off_u);
        let nearly_negative_v =
            vec3(2.0f64.to_radians().sin(), -2.0f64.to_radians().cos(), 0.0);
        assert_eq!(
            snap_axis_direction(nearly_negative_v, axis_u, axis_v, 4.0),
            -axis_v
        );
    }

    /// The kept half-plane always holds the profile: the radial flips with
    /// the profile side and stays finite when the centroid sits on the axis.
    #[test]
    fn radial_points_toward_centroid() {
        let normal = vec3(0.0, 0.0, 1.0);
        let axis = vec3(0.0, 1.0, 0.0);
        let origin = vec3(0.0, 0.0, 0.0);
        let toward_positive_u =
            radial_toward_centroid(origin, axis, normal, vec3(0.5, 0.0, 0.0));
        assert!((toward_positive_u - vec3(1.0, 0.0, 0.0)).length() < 1e-12);
        let toward_negative_u =
            radial_toward_centroid(origin, axis, normal, vec3(-0.5, 0.2, 0.0));
        assert!((toward_negative_u - vec3(-1.0, 0.0, 0.0)).length() < 1e-12);
        let on_axis = radial_toward_centroid(origin, axis, normal, vec3(0.0, 3.0, 0.0));
        assert!((on_axis.length() - 1.0).abs() < 1e-12);
        assert!(on_axis.x.is_finite() && on_axis.y.is_finite() && on_axis.z.is_finite());
    }

    #[test]
    fn axis_half_length_covers_profile() {
        let mut state = AppState::new(SceneDocument::new());
        let id = state
            .document
            .add_primitive_from_drag(
                "rectangle",
                vec3(-0.3, -0.5, 0.0),
                vec3(0.3, 0.5, 0.0),
                1.0,
            )
            .unwrap();
        let frame = section_frame(&state.document, id).expect("section");
        let axis_origin = frame.origin + frame.axis_u * 0.3;
        let half = axis_half_length(
            frame.origin,
            frame.axis_u,
            frame.axis_v,
            frame.profile.bounds(),
            axis_origin,
        );
        // Farthest corner from (0.3, 0) is (-0.3, ±0.5): distance ~0.781.
        assert!(half >= (0.6f64.powi(2) + 0.5f64.powi(2)).sqrt());
    }

    /// Enter with no picked points commits the same revolve the old
    /// one-click button produced, but with the axis stored explicitly.
    #[test]
    fn enter_with_no_points_commits_default_axis() {
        let mut state = AppState::new(SceneDocument::new());
        let id = state
            .document
            .add_primitive_from_drag(
                "rectangle",
                vec3(-0.3, -0.5, 0.0),
                vec3(0.3, 0.5, 0.0),
                1.0,
            )
            .unwrap();
        let frame = section_frame(&state.document, id).expect("section");
        let mut tools = ToolState::default();
        tools.set_tool(ToolKind::RevolveAxisPick(id), &mut state);
        tools.revolve_angle = 360.0;
        assert!(tools.confirm_pending(&mut state));
        assert_eq!(tools.kind, ToolKind::Select, "commit exits the tool");
        assert_eq!(state.document.roots.len(), 1);
        let root = state.document.roots[0];
        assert_eq!(state.selection, vec![root]);
        let object = state.document.object(root).unwrap();
        let ScenePayload::Revolve {
            axis_origin,
            axis_direction,
            radial_direction,
            angle_degrees,
            ..
        } = &object.payload
        else {
            panic!("expected a revolve payload");
        };
        assert_eq!(*axis_origin, Some(frame.origin));
        assert_eq!(*axis_direction, Some(frame.axis_v));
        assert!(radial_direction.is_some());
        assert_eq!(*angle_degrees, 360.0);
        assert!(state.can_undo());
    }

    /// A picked origin + explicit direction land verbatim in the payload,
    /// with the radial oriented at the profile.
    #[test]
    fn commit_with_explicit_origin_and_direction() {
        let mut state = AppState::new(SceneDocument::new());
        let id = state
            .document
            .add_primitive_from_drag(
                "rectangle",
                vec3(-0.3, -0.5, 0.0),
                vec3(0.3, 0.5, 0.0),
                1.0,
            )
            .unwrap();
        let frame = section_frame(&state.document, id).expect("section");
        // Axis along the rectangle's left edge: classic solid cylinder.
        let picked_origin = frame.origin - frame.axis_u * 0.3;
        let mut tools = ToolState::default();
        tools.set_tool(ToolKind::RevolveAxisPick(id), &mut state);
        tools.points.push(picked_origin);
        assert!(tools.commit_revolve(id, Some(frame.axis_v), &mut state));
        let root = state.document.roots[0];
        let ScenePayload::Revolve {
            axis_origin,
            axis_direction,
            radial_direction,
            ..
        } = &state.document.object(root).unwrap().payload
        else {
            panic!("expected a revolve payload");
        };
        assert_eq!(*axis_origin, Some(picked_origin));
        assert_eq!(*axis_direction, Some(frame.axis_v));
        let radial = radial_direction.expect("radial derived");
        assert!(
            radial.dot(frame.axis_u) > 0.99,
            "radial must point from the axis toward the profile"
        );
        state.undo();
        assert_eq!(state.document.roots, vec![id], "one undo restores the section");
    }

    /// Deleting the section mid-tool exits cleanly instead of committing
    /// against a dead id.
    #[test]
    fn stale_section_exits_to_select() {
        let mut state = AppState::new(SceneDocument::new());
        let id = state
            .document
            .add_primitive_from_drag(
                "circle",
                vec3(0.2, -0.2, 0.0),
                vec3(0.6, 0.2, 0.0),
                1.0,
            )
            .unwrap();
        let mut tools = ToolState::default();
        tools.set_tool(ToolKind::RevolveAxisPick(id), &mut state);
        state.document.delete_many(&[id]);
        assert!(tools.confirm_pending(&mut state), "Enter is consumed");
        assert_eq!(tools.kind, ToolKind::Select);
        assert!(state.document.roots.is_empty(), "document unchanged");
    }

    /// Point-placed segment builds the same node as the drag path did.
    #[test]
    fn point_placed_segment_matches_drag_segment() {
        let start = vec3(0.0, 0.0, 0.0);
        let end = vec3(3.0, 4.0, 0.0);
        let from_points = ghost_from_points("segment", &[start, end]).expect("segment");
        let from_drag = ghost_from_drag("segment", start, end, 1.0).expect("segment");
        assert_eq!(format!("{from_points:?}"), format!("{from_drag:?}"));
        // Exactly two points: a third must refuse.
        assert!(ghost_from_points("segment", &[start, end, vec3(1.0, 1.0, 0.0)]).is_none());
    }
}
