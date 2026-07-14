//! Viewport tools: draw-on-grid creation (drag-sized and point-placed
//! kinds), Move, and Rotate — the port of casoCAD's grid interaction tools.
//! All tools act on the `Z = 0` XY reference grid, like the Python app.

use caso_kernel::boundary_ops::BoundaryPatchHit;
use caso_kernel::scene::SceneDocument;
use caso_kernel::sdf::node::{Node, RotationAxis};
use caso_kernel::vec3::{vec3, Vec3};
use caso_render::OrbitCamera;
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
pub const POINT_KINDS: [(&str, &str); 6] = [
    ("Segment 1D", "segment"),
    ("Polyline", "polyline"),
    ("Bezier Curve", "quadratic_bezier_curve"),
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
    /// Typed-dimension buffer, filled from keystrokes during a create drag.
    pub dimension_text: String,
    move_last: Option<Vec3>,
    rotate_last_x: Option<f32>,
    undo_pushed: bool,
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
            dimension_text: String::new(),
            move_last: None,
            rotate_last_x: None,
            undo_pushed: false,
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
            ToolKind::CreateDrag(create) => {
                format!("Draw {create}: drag on the grid (type dimensions, Enter applies)")
            }
            ToolKind::CreatePoints(create) => {
                format!("Place {create}: click points on the grid, Enter commits, Esc cancels")
            }
            ToolKind::Move => "Move: drag the selection on the grid".to_string(),
            ToolKind::Rotate => "Rotate: drag horizontally to spin about Z".to_string(),
            ToolKind::Measure => {
                "Measure: click two points (surfaces snap, grid otherwise) — Esc cancels a point, Delete clears all".to_string()
            }
            ToolKind::BoundaryRegion => {
                "Boundary Region: hover the Fluid Domain surface, click to tag it".to_string()
            }
            ToolKind::BoundaryCutter(knife) => format!(
                "Cutter ({knife}): select a region, place the knife, Enter splits"
            ),
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

    /// Whether the active tool owns the primary mouse button (the Boundary
    /// Region hover tool keeps camera navigation available, like Python).
    pub fn blocks_camera(&self) -> bool {
        !matches!(
            self.kind,
            ToolKind::Select | ToolKind::Measure | ToolKind::BoundaryRegion
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
        let (enter, escape, backspace) = ui.ctx().input(|input| {
            (
                input.key_pressed(egui::Key::Enter),
                input.key_pressed(egui::Key::Escape),
                input.key_pressed(egui::Key::Backspace),
            )
        });
        if backspace && !self.points.is_empty() {
            self.points.pop();
            points_changed = true;
        }
        if escape {
            self.cancel(state);
            return true;
        }
        // Rebuild the knife ghost (drives the cyan/orange split preview).
        let region_id = state.selected_region.expect("checked");
        let minimum = match knife {
            "segment" => 2,
            _ => 3,
        };
        if points_changed {
            self.preview_ghost = if self.points.len() >= minimum {
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
            } else {
                None
            };
            self.overlay_revision += 1;
        }
        if enter {
            let ghost = if self.points.len() >= minimum {
                boundary_tool::cutter_ghost(&root, knife, &self.points)
            } else {
                Err(format!("{knife} knife needs at least {minimum} points"))
            };
            match ghost {
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
                            // Empty-cut refusal: undo snapshot rolls back.
                            state.undo();
                            state.status = error.to_string();
                        }
                    }
                }
                Err(error) => state.status = error,
            }
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
        if response.drag_started_by(egui::PointerButton::Primary) {
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
        if response.clicked() {
            if let Some(pos) = pointer {
                if let Some(point) = grid_point(camera, pos, rect, pixels_per_point) {
                    self.points.push(point);
                    state.status = format!("{} point(s) — Enter commits", self.points.len());
                }
            }
        }
        let (enter, escape, backspace) = ui.ctx().input(|input| {
            (
                input.key_pressed(egui::Key::Enter),
                input.key_pressed(egui::Key::Escape),
                input.key_pressed(egui::Key::Backspace),
            )
        });
        if backspace {
            self.points.pop();
        }
        if escape {
            self.cancel(state);
        }
        if enter && !self.points.is_empty() {
            let points = std::mem::take(&mut self.points);
            state.push_undo();
            let result = state
                .document
                .add_point_shape_from_world_points(kind, &points, "xy");
            if let Some(id) = state.report(result, &format!("Created {kind}")) {
                state.select_only(id);
            } else {
                state.undo();
            }
        }
        // Live geometry ghost from the committed points plus the cursor as
        // a tentative point, so the ghost rubber-bands while placing.
        let cursor_point = pointer
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
                // exact-count kinds (segment, bezier curve) the tentative
                // cursor point can overshoot — fall back to the committed
                // points so the ghost survives until Enter.
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
        if let (Some(last), Some(cursor)) = (screen_points.last(), pointer) {
            if rect.contains(cursor) {
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
        if response.drag_started_by(egui::PointerButton::Primary) {
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
        if response.drag_started_by(egui::PointerButton::Primary) {
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
    /// create drag is armed.
    fn collect_dimension_keys(&mut self, ui: &egui::Ui) {
        ui.ctx().input(|input| {
            for event in &input.events {
                match event {
                    egui::Event::Text(text) => {
                        for character in text.chars() {
                            if character.is_ascii_digit()
                                || ".,xX;mkcft'\" +-*/()".contains(character)
                            {
                                self.dimension_text.push(character);
                            }
                        }
                    }
                    egui::Event::Key {
                        key: egui::Key::Backspace,
                        pressed: true,
                        ..
                    } => {
                        self.dimension_text.pop();
                    }
                    egui::Event::Key {
                        key: egui::Key::Escape,
                        pressed: true,
                        ..
                    } => {
                        self.dimension_text.clear();
                    }
                    _ => {}
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

#[cfg(test)]
mod tests {
    use super::*;

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
