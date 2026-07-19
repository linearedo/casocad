//! Properties panel: unit-aware editing of the selected object's placement
//! and dimensions. All lengths are stored in meters; the widgets display and
//! accept values in the current working unit (the model is never rescaled).

use caso_kernel::frame::Frame;
use caso_kernel::scene::{ObjectId, OperatorKind, SceneDocument, ScenePayload};
use caso_kernel::sdf::curtain::NormalCurtain;
use caso_kernel::sdf::node::RotationAxis;
use caso_kernel::sdf::primitives_1d::Profile1D;
use caso_kernel::sdf::primitives_2d::{Point2, Profile2D};
use caso_kernel::sdf::solid_from_2d::RevolveAxis;
use caso_kernel::sdf::tubes::CapStyle;
use caso_kernel::vec3::Vec3;
use eframe::egui;

use crate::scene_panel::payload_label;
use crate::state::AppState;

/// Smallest editable dimension in meters (matches positive-dimension guards).
const MIN_DIMENSION: f64 = 1e-6;

#[derive(Default)]
pub struct PropertiesPanel {
    /// One undo snapshot per continuous edit gesture (drag or typing burst).
    undo_group_open: bool,
}

/// A draggable length field in working units; returns true when edited.
fn length_field(
    ui: &mut egui::Ui,
    label: &str,
    value_meters: &mut f64,
    factor: f64,
    positive: bool,
) -> bool {
    let mut shown = *value_meters / factor;
    let range = if positive {
        (MIN_DIMENSION / factor)..=f64::INFINITY
    } else {
        f64::NEG_INFINITY..=f64::INFINITY
    };
    let changed = ui
        .horizontal(|ui| {
            ui.label(label);
            ui.add(
                egui::DragValue::new(&mut shown)
                    .speed(0.01)
                    .range(range)
                    .max_decimals(6),
            )
            .changed()
        })
        .inner;
    if changed {
        *value_meters = shown * factor;
    }
    changed
}

fn vec3_field(ui: &mut egui::Ui, label: &str, value: &mut Vec3, factor: f64) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        for (axis, component) in [
            ("x", &mut value.x),
            ("y", &mut value.y),
            ("z", &mut value.z),
        ] {
            let mut shown = *component / factor;
            if ui
                .add(
                    egui::DragValue::new(&mut shown)
                        .speed(0.01)
                        .prefix(format!("{axis} "))
                        .max_decimals(6),
                )
                .changed()
            {
                *component = shown * factor;
                changed = true;
            }
        }
    });
    changed
}

fn angle_field(ui: &mut egui::Ui, label: &str, degrees: &mut f64) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(
            egui::DragValue::new(degrees)
                .speed(0.5)
                .suffix("°")
                .max_decimals(3),
        )
        .changed()
    })
    .inner
}

/// A length field clamped to `0..=max_meters` (e.g. tube inner radius).
fn bounded_length_field(
    ui: &mut egui::Ui,
    label: &str,
    value_meters: &mut f64,
    factor: f64,
    max_meters: f64,
) -> bool {
    let mut shown = *value_meters / factor;
    let changed = ui
        .horizontal(|ui| {
            ui.label(label);
            ui.add(
                egui::DragValue::new(&mut shown)
                    .speed(0.01)
                    .range(0.0..=(max_meters / factor))
                    .max_decimals(6),
            )
            .changed()
        })
        .inner;
    if changed {
        *value_meters = (shown * factor).clamp(0.0, max_meters);
    }
    changed
}

/// Orientation editor: the frame shown and edited as three Euler angles in
/// degrees, with the raw axes displayed read-only underneath.
fn frame_field(ui: &mut egui::Ui, frame: &mut Frame) -> bool {
    let [mut rx, mut ry, mut rz] = frame.to_euler_degrees();
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Orientation");
        for (axis, angle) in [("Rx", &mut rx), ("Ry", &mut ry), ("Rz", &mut rz)] {
            changed |= ui
                .add(
                    egui::DragValue::new(angle)
                        .speed(0.5)
                        .prefix(format!("{axis} "))
                        .suffix("°")
                        .max_decimals(3),
                )
                .changed();
        }
    });
    if !frame.is_identity() || changed {
        weak_axis_row(ui, "u", frame.u);
        weak_axis_row(ui, "v", frame.v);
        weak_axis_row(ui, "w", frame.w);
    }
    if changed {
        *frame = Frame::from_euler_degrees(rx, ry, rz);
    }
    changed
}

fn weak_axis_row(ui: &mut egui::Ui, label: &str, axis: Vec3) {
    ui.weak(format!(
        "  {label} ({:.3}, {:.3}, {:.3})",
        axis.x, axis.y, axis.z
    ));
}

/// A profile point stored in workplane (u, v) coordinates, displayed and
/// edited as its final world x,y,z. Edits are projected back onto the sketch
/// plane, so the sketch stays planar by construction.
fn world_point_field(
    ui: &mut egui::Ui,
    label: &str,
    point: &mut Point2,
    origin: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
    factor: f64,
) -> bool {
    let mut world = origin + axis_u * point[0] + axis_v * point[1];
    if vec3_field(ui, label, &mut world, factor) {
        let offset = world - origin;
        *point = [offset.dot(axis_u), offset.dot(axis_v)];
        true
    } else {
        false
    }
}

/// An in-plane (u, v) displacement displayed and edited as a world vector.
fn world_offset_field(
    ui: &mut egui::Ui,
    label: &str,
    offset: &mut Point2,
    axis_u: Vec3,
    axis_v: Vec3,
    factor: f64,
) -> bool {
    let mut world = axis_u * offset[0] + axis_v * offset[1];
    if vec3_field(ui, label, &mut world, factor) {
        *offset = [world.dot(axis_u), world.dot(axis_v)];
        true
    } else {
        false
    }
}

fn point_label(index: usize, bezier: bool) -> String {
    if bezier {
        let role = if index % 2 == 1 { "control" } else { "anchor" };
        format!("P{index} ({role})")
    } else {
        format!("P{index}")
    }
}

/// Point lists longer than this render inside a virtualized scroll area:
/// laying out hundreds of drag widgets every frame makes the whole app
/// sluggish, so only the visible rows are built.
const INLINE_POINT_ROWS: usize = 10;

/// What structural edits a point list allows, dictated by the kernel's
/// validation rules for the profile kind that owns the points.
#[derive(Clone, Copy, PartialEq)]
enum PointListPolicy {
    /// Insert/delete one point at a time; `closed` picks midpoint wrap
    /// (polygon) versus tail extension (polyline, tube path).
    Single { min: usize, closed: bool },
    /// Bezier kinds (odd anchor-control-anchor chains): insert splits a
    /// span exactly (De Casteljau), delete removes a control+anchor pair;
    /// buttons live on anchor rows only.
    Spans { min: usize },
}

#[derive(Clone, Copy)]
enum PointEdit {
    InsertAfter(usize),
    Delete(usize),
}

/// (insert allowed, delete allowed) for the row at `index`.
fn point_row_buttons(policy: PointListPolicy, index: usize, len: usize) -> (bool, bool) {
    match policy {
        PointListPolicy::Single { min, .. } => (true, len > min),
        PointListPolicy::Spans { min } => {
            let anchor = index.is_multiple_of(2);
            (anchor && len >= 3, anchor && len >= min + 2)
        }
    }
}

/// Apply an insert/delete to the raw point list; `lerp(a, b, t)` is the
/// only geometry the helper needs, so (u, v) and world points share it.
/// Returns false when the policy forbids the edit (list left untouched).
fn apply_point_edit<T: Copy>(
    points: &mut Vec<T>,
    edit: PointEdit,
    policy: PointListPolicy,
    lerp: impl Fn(T, T, f64) -> T,
) -> bool {
    let len = points.len();
    match (policy, edit) {
        (PointListPolicy::Single { closed, .. }, PointEdit::InsertAfter(index)) if index < len => {
            let point = if index + 1 < len {
                lerp(points[index], points[index + 1], 0.5)
            } else if closed {
                lerp(points[index], points[0], 0.5)
            } else if len >= 2 {
                // Extend the open tail by mirroring the last segment.
                lerp(points[index - 1], points[index], 2.0)
            } else {
                return false;
            };
            points.insert(index + 1, point);
            true
        }
        (PointListPolicy::Single { min, .. }, PointEdit::Delete(index))
            if index < len && len > min =>
        {
            points.remove(index);
            true
        }
        (PointListPolicy::Spans { .. }, PointEdit::InsertAfter(index))
            if index < len && index.is_multiple_of(2) && len >= 3 =>
        {
            // Split the span starting at this anchor (the one ending here
            // for the last anchor) at t = 0.5 — the curve is unchanged.
            let start = if index + 2 < len { index } else { index - 2 };
            let (a, c, b) = (points[start], points[start + 1], points[start + 2]);
            let c1 = lerp(a, c, 0.5);
            let c2 = lerp(c, b, 0.5);
            let mid = lerp(c1, c2, 0.5);
            points[start + 1] = c1;
            points.insert(start + 2, mid);
            points.insert(start + 3, c2);
            true
        }
        (PointListPolicy::Spans { min }, PointEdit::Delete(index))
            if index < len && index.is_multiple_of(2) && len >= min + 2 =>
        {
            // Remove the anchor with its preceding control (following
            // control for the first anchor): one span disappears.
            if index == 0 {
                points.remove(1);
                points.remove(0);
            } else {
                points.remove(index);
                points.remove(index - 1);
            }
            true
        }
        _ => false,
    }
}

/// Insert/delete buttons for one row; returns the requested edit.
fn point_row_edit_buttons(
    ui: &mut egui::Ui,
    policy: PointListPolicy,
    index: usize,
    len: usize,
) -> Option<PointEdit> {
    let (insert, delete) = point_row_buttons(policy, index, len);
    let mut edit = None;
    if insert
        && ui
            .small_button("+")
            .on_hover_text("Insert a point after this one")
            .clicked()
    {
        edit = Some(PointEdit::InsertAfter(index));
    }
    if delete
        && ui
            .small_button("✕")
            .on_hover_text("Delete this point")
            .clicked()
    {
        edit = Some(PointEdit::Delete(index));
    }
    edit
}

fn uv_point_list_ui(
    ui: &mut egui::Ui,
    points: &mut Vec<Point2>,
    policy: PointListPolicy,
    origin: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
    factor: f64,
) -> bool {
    let bezier = matches!(policy, PointListPolicy::Spans { .. });
    let mut changed = false;
    let mut edit: Option<PointEdit> = None;
    let len = points.len();
    let mut row = |ui: &mut egui::Ui, index: usize, point: &mut Point2| {
        ui.horizontal(|ui| {
            changed |= world_point_field(
                ui,
                &point_label(index, bezier),
                point,
                origin,
                axis_u,
                axis_v,
                factor,
            );
            if let Some(requested) = point_row_edit_buttons(ui, policy, index, len) {
                edit = Some(requested);
            }
        });
    };
    if points.len() <= INLINE_POINT_ROWS {
        for (index, point) in points.iter_mut().enumerate() {
            row(ui, index, point);
        }
    } else {
        ui.weak(format!("Points: {}", points.len()));
        let row_height = ui.spacing().interact_size.y;
        egui::ScrollArea::vertical()
            .id_salt("uv_points")
            .max_height(row_height * INLINE_POINT_ROWS as f32)
            .show_rows(ui, row_height, points.len(), |ui, rows| {
                for index in rows {
                    let mut point = points[index];
                    row(ui, index, &mut point);
                    points[index] = point;
                }
            });
    }
    if let Some(edit) = edit {
        changed |= apply_point_edit(points, edit, policy, |a, b, t| {
            [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
        });
    }
    changed
}

fn vec3_point_list_ui(
    ui: &mut egui::Ui,
    points: &mut Vec<Vec3>,
    policy: PointListPolicy,
    factor: f64,
) -> bool {
    let bezier = matches!(policy, PointListPolicy::Spans { .. });
    let mut changed = false;
    let mut edit: Option<PointEdit> = None;
    let len = points.len();
    let mut row = |ui: &mut egui::Ui, index: usize, point: &mut Vec3| {
        ui.horizontal(|ui| {
            changed |= vec3_field(ui, &point_label(index, bezier), point, factor);
            if let Some(requested) = point_row_edit_buttons(ui, policy, index, len) {
                edit = Some(requested);
            }
        });
    };
    if points.len() <= INLINE_POINT_ROWS {
        for (index, point) in points.iter_mut().enumerate() {
            row(ui, index, point);
        }
    } else {
        ui.weak(format!("Points: {}", points.len()));
        let row_height = ui.spacing().interact_size.y;
        egui::ScrollArea::vertical()
            .id_salt("world_points")
            .max_height(row_height * INLINE_POINT_ROWS as f32)
            .show_rows(ui, row_height, points.len(), |ui, rows| {
                for index in rows {
                    let mut point = points[index];
                    row(ui, index, &mut point);
                    points[index] = point;
                }
            });
    }
    if let Some(edit) = edit {
        changed |= apply_point_edit(points, edit, policy, |a, b, t| a + (b - a) * t);
    }
    changed
}

/// All parameters of a placed 2D profile, points in world x,y,z. Recurses
/// into offset/boolean children with the child's effective plane origin so
/// nested points still display their final world position.
fn profile2d_ui(
    ui: &mut egui::Ui,
    profile: &mut Profile2D,
    origin: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
    factor: f64,
) -> bool {
    let mut changed = false;
    match profile {
        Profile2D::Polyline { points } => {
            changed |= uv_point_list_ui(
                ui,
                points,
                PointListPolicy::Single {
                    min: 2,
                    closed: false,
                },
                origin,
                axis_u,
                axis_v,
                factor,
            );
        }
        Profile2D::Polygon { points } => {
            changed |= uv_point_list_ui(
                ui,
                points,
                PointListPolicy::Single {
                    min: 3,
                    closed: true,
                },
                origin,
                axis_u,
                axis_v,
                factor,
            );
        }
        Profile2D::QuadraticBezierCurve { points }
        | Profile2D::QuadraticBezierSurface { points } => {
            changed |= uv_point_list_ui(
                ui,
                points,
                PointListPolicy::Spans { min: 3 },
                origin,
                axis_u,
                axis_v,
                factor,
            );
        }
        Profile2D::Circle { center, radius } => {
            changed |= world_point_field(ui, "Center", center, origin, axis_u, axis_v, factor);
            changed |= length_field(ui, "Radius", radius, factor, true);
        }
        Profile2D::Rectangle { center, half_size } => {
            changed |= world_point_field(ui, "Center", center, origin, axis_u, axis_v, factor);
            changed |= length_field(ui, "Half width", &mut half_size[0], factor, true);
            changed |= length_field(ui, "Half height", &mut half_size[1], factor, true);
        }
        Profile2D::Square { center, half_size } => {
            changed |= world_point_field(ui, "Center", center, origin, axis_u, axis_v, factor);
            changed |= length_field(ui, "Half size", half_size, factor, true);
        }
        Profile2D::RoundedRectangle {
            center,
            half_size,
            corner_radius,
        } => {
            changed |= world_point_field(ui, "Center", center, origin, axis_u, axis_v, factor);
            changed |= length_field(ui, "Half width", &mut half_size[0], factor, true);
            changed |= length_field(ui, "Half height", &mut half_size[1], factor, true);
            changed |= length_field(ui, "Corner radius", corner_radius, factor, true);
        }
        Profile2D::Ellipse { center, semi_axes } => {
            changed |= world_point_field(ui, "Center", center, origin, axis_u, axis_v, factor);
            changed |= length_field(ui, "Semi-axis A", &mut semi_axes[0], factor, true);
            changed |= length_field(ui, "Semi-axis B", &mut semi_axes[1], factor, true);
        }
        Profile2D::RegularPolygon {
            center,
            radius,
            side_count,
            rotation,
        } => {
            changed |= world_point_field(ui, "Center", center, origin, axis_u, axis_v, factor);
            changed |= length_field(ui, "Radius", radius, factor, true);
            ui.horizontal(|ui| {
                ui.label("Sides");
                changed |= ui
                    .add(egui::DragValue::new(side_count).speed(0.1).range(3..=360))
                    .changed();
            });
            let mut degrees = rotation.to_degrees();
            if angle_field(ui, "Rotation", &mut degrees) {
                *rotation = degrees.to_radians();
                changed = true;
            }
        }
        Profile2D::Offset { child, offset } => {
            changed |= world_offset_field(ui, "Offset", offset, axis_u, axis_v, factor);
            let child_origin = origin + axis_u * offset[0] + axis_v * offset[1];
            changed |= profile2d_ui(ui, child, child_origin, axis_u, axis_v, factor);
        }
        Profile2D::DistanceOffset { child, offset } => {
            changed |= length_field(ui, "Grow by", offset, factor, false);
            changed |= profile2d_ui(ui, child, origin, axis_u, axis_v, factor);
        }
        Profile2D::Binary {
            left,
            right,
            operation,
            ..
        } => {
            ui.weak(format!("Boolean: {}", operation.as_str()));
            ui.weak("Left:");
            changed |= ui
                .indent("left", |ui| {
                    profile2d_ui(ui, left, origin, axis_u, axis_v, factor)
                })
                .inner;
            ui.weak("Right:");
            changed |= ui
                .indent("right", |ui| {
                    profile2d_ui(ui, right, origin, axis_u, axis_v, factor)
                })
                .inner;
        }
    }
    changed
}

/// All parameters of a placed 1D profile; segments are shown as their two
/// world-space endpoints along the placement axis.
fn profile1d_ui(
    ui: &mut egui::Ui,
    profile: &mut Profile1D,
    origin: Vec3,
    axis_u: Vec3,
    factor: f64,
) -> bool {
    let mut changed = false;
    match profile {
        Profile1D::Segment {
            center,
            half_length,
        } => {
            let mut start = origin + axis_u * (*center - *half_length);
            let mut end = origin + axis_u * (*center + *half_length);
            let edited =
                vec3_field(ui, "Start", &mut start, factor) | vec3_field(ui, "End", &mut end, factor);
            if edited {
                let t0 = (start - origin).dot(axis_u);
                let t1 = (end - origin).dot(axis_u);
                *center = 0.5 * (t0 + t1);
                *half_length = (0.5 * (t1 - t0).abs()).max(MIN_DIMENSION);
                changed = true;
            }
        }
        Profile1D::Offset { child, offset } => {
            changed |= length_field(ui, "Offset", offset, factor, false);
            changed |= profile1d_ui(ui, child, origin + axis_u * *offset, axis_u, factor);
        }
        Profile1D::Binary {
            left,
            right,
            operation,
            ..
        } => {
            ui.weak(format!("Boolean: {}", operation.as_str()));
            ui.weak("Left:");
            changed |= ui
                .indent("left", |ui| profile1d_ui(ui, left, origin, axis_u, factor))
                .inner;
            ui.weak("Right:");
            changed |= ui
                .indent("right", |ui| profile1d_ui(ui, right, origin, axis_u, factor))
                .inner;
        }
    }
    changed
}

fn caps_field(ui: &mut egui::Ui, caps: &mut CapStyle) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Caps");
        egui::ComboBox::from_id_salt("tube_caps")
            .selected_text(caps.as_str())
            .show_ui(ui, |ui| {
                for option in [CapStyle::Round, CapStyle::Flat] {
                    changed |= ui.selectable_value(caps, option, option.as_str()).changed();
                }
            });
    });
    changed
}

/// Read-only row naming a referenced child object.
fn child_row(ui: &mut egui::Ui, document: &SceneDocument, label: &str, id: ObjectId) {
    let name = document
        .object(id)
        .map(|object| object.name.clone())
        .unwrap_or_else(|_| format!("object {id}"));
    ui.horizontal(|ui| {
        ui.weak(label);
        ui.label(name);
    });
}

impl PropertiesPanel {
    pub fn ui(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
        if let Some(region_id) = state.selected_region {
            self.region_ui(ui, state, region_id);
            ui.separator();
        }
        let Some(id) = state.selected_single() else {
            if state.selected_region.is_none() {
                ui.weak(if state.selection.is_empty() {
                    "No selection."
                } else {
                    "Multiple objects selected."
                });
            }
            return;
        };
        let Ok(object) = state.document.object(id) else {
            return;
        };
        let name = object.name.clone();
        let kind = payload_label(&object.payload);
        let mut payload = object.payload.clone();
        let factor = state.unit.factor;

        ui.strong(name);
        ui.horizontal(|ui| {
            ui.weak("Kind:");
            ui.label(kind);
            ui.weak(format!("(lengths in {})", state.unit.key));
        });
        if let Some(domain) = state.document.domain_kinds.get(&id) {
            ui.horizontal(|ui| {
                ui.weak("Domain:");
                ui.label(domain.to_string());
            });
        }
        ui.separator();

        let changed = Self::payload_ui(ui, &state.document, &mut payload, factor);
        if changed {
            if !self.undo_group_open {
                state.push_undo();
                self.undo_group_open = true;
            }
            self.apply(state, id, payload);
        } else if !ui.ctx().input(|input| input.pointer.any_down())
            && ui.ctx().memory(|memory| memory.focused().is_none())
        {
            self.undo_group_open = false;
        }
    }

    /// Properties of the selected BoundaryRegion: owner, patch scope, cut
    /// lineage, and the physics tag (with suggestions per domain kind).
    fn region_ui(&mut self, ui: &mut egui::Ui, state: &mut AppState, region_id: u32) {
        let Some(region) = state
            .document
            .boundary_regions
            .iter()
            .find(|region| region.object_id == region_id)
            .cloned()
        else {
            return;
        };
        ui.strong(&region.name);
        let owner_name = state
            .document
            .object(region.owner_object_id)
            .map(|object| object.name.clone())
            .unwrap_or_else(|_| format!("object {}", region.owner_object_id));
        ui.horizontal(|ui| {
            ui.weak("Owner:");
            ui.label(owner_name);
        });
        if let Some(patch) = &region.patch_id {
            ui.horizontal(|ui| {
                ui.weak("Patch:");
                ui.label(patch);
            });
        }
        if !region.cuts.is_empty() {
            ui.weak("Cut lineage:");
            for (index, cut) in region.cuts.iter().enumerate() {
                ui.label(format!(
                    "  {}. {} of {}",
                    index + 1,
                    cut.side.as_str(),
                    cut.ghost.name
                ));
            }
        }
        let mut tag = region.tag.clone().unwrap_or_default();
        let changed = ui
            .horizontal(|ui| {
                ui.label("Tag");
                ui.text_edit_singleline(&mut tag).changed()
            })
            .inner;
        ui.horizontal(|ui| {
            for suggestion in ["inlet", "outlet", "wall", "symmetry"] {
                if ui.small_button(suggestion).clicked() {
                    self.set_region_tag(state, region_id, Some(suggestion.to_string()));
                }
            }
        });
        if changed {
            self.set_region_tag(
                state,
                region_id,
                if tag.trim().is_empty() {
                    None
                } else {
                    Some(tag)
                },
            );
        }
    }

    fn set_region_tag(&mut self, state: &mut AppState, region_id: u32, tag: Option<String>) {
        if !self.undo_group_open {
            state.push_undo();
            self.undo_group_open = true;
        }
        if let Some(region) = state
            .document
            .boundary_regions
            .iter_mut()
            .find(|region| region.object_id == region_id)
        {
            region.tag = tag;
            state.document.mark_changed();
        }
    }

    fn apply(&mut self, state: &mut AppState, id: ObjectId, payload: ScenePayload) {
        if let Ok(object) = state.document.object_mut(id) {
            object.payload = payload;
            // Keep combined booleans and their operand objects in lockstep
            // whichever side was edited.
            state.document.resync_boolean_chains(id);
            state.document.mark_changed();
            state.status = "Edited".to_string();
        }
    }

    fn payload_ui(
        ui: &mut egui::Ui,
        document: &SceneDocument,
        payload: &mut ScenePayload,
        factor: f64,
    ) -> bool {
        let mut changed = false;
        match payload {
            ScenePayload::Sphere(sphere) => {
                changed |= vec3_field(ui, "Center", &mut sphere.center, factor);
                changed |= length_field(ui, "Radius", &mut sphere.radius, factor, true);
            }
            ScenePayload::Box3(shape) => {
                changed |= vec3_field(ui, "Center", &mut shape.center, factor);
                changed |= vec3_field(ui, "Half size", &mut shape.half_size, factor);
                changed |= frame_field(ui, &mut shape.frame);
            }
            ScenePayload::Cylinder(shape) => {
                changed |= vec3_field(ui, "Center", &mut shape.center, factor);
                changed |= length_field(ui, "Radius", &mut shape.radius, factor, true);
                changed |= length_field(ui, "Half height", &mut shape.half_height, factor, true);
                changed |= frame_field(ui, &mut shape.frame);
            }
            ScenePayload::Cone(shape) => {
                changed |= vec3_field(ui, "Center", &mut shape.center, factor);
                changed |= length_field(ui, "Radius", &mut shape.radius, factor, true);
                changed |= length_field(ui, "Half height", &mut shape.half_height, factor, true);
                changed |= frame_field(ui, &mut shape.frame);
            }
            ScenePayload::CappedCone(shape) => {
                changed |= vec3_field(ui, "Center", &mut shape.center, factor);
                changed |= length_field(ui, "Radius A", &mut shape.radius_a, factor, true);
                changed |= length_field(ui, "Radius B", &mut shape.radius_b, factor, true);
                changed |= length_field(ui, "Half height", &mut shape.half_height, factor, true);
                changed |= frame_field(ui, &mut shape.frame);
            }
            ScenePayload::Pyramid(shape) => {
                changed |= vec3_field(ui, "Center", &mut shape.center, factor);
                changed |=
                    length_field(ui, "Base half size", &mut shape.base_half_size, factor, true);
                changed |= length_field(ui, "Half height", &mut shape.half_height, factor, true);
                changed |= frame_field(ui, &mut shape.frame);
            }
            ScenePayload::BoxFrame(shape) => {
                changed |= vec3_field(ui, "Center", &mut shape.center, factor);
                changed |= vec3_field(ui, "Half size", &mut shape.half_size, factor);
                changed |= length_field(ui, "Thickness", &mut shape.thickness, factor, true);
                changed |= frame_field(ui, &mut shape.frame);
            }
            ScenePayload::Torus(shape) => {
                changed |= vec3_field(ui, "Center", &mut shape.center, factor);
                changed |= length_field(ui, "Major radius", &mut shape.major_radius, factor, true);
                changed |= length_field(ui, "Minor radius", &mut shape.minor_radius, factor, true);
                changed |= frame_field(ui, &mut shape.frame);
            }
            ScenePayload::PolylineTube(tube) => {
                changed |= length_field(ui, "Radius", &mut tube.radius, factor, true);
                let max_inner = (tube.radius - MIN_DIMENSION).max(0.0);
                changed |=
                    bounded_length_field(ui, "Inner radius", &mut tube.inner_radius, factor, max_inner);
                changed |= caps_field(ui, &mut tube.caps);
                changed |= vec3_point_list_ui(
                    ui,
                    &mut tube.points,
                    PointListPolicy::Single {
                        min: 2,
                        closed: false,
                    },
                    factor,
                );
            }
            ScenePayload::QuadraticBezierTube(tube) => {
                changed |= length_field(ui, "Radius", &mut tube.radius, factor, true);
                let max_inner = (tube.radius - MIN_DIMENSION).max(0.0);
                changed |=
                    bounded_length_field(ui, "Inner radius", &mut tube.inner_radius, factor, max_inner);
                changed |= caps_field(ui, &mut tube.caps);
                changed |= vec3_point_list_ui(
                    ui,
                    &mut tube.points,
                    PointListPolicy::Spans { min: 3 },
                    factor,
                );
            }
            ScenePayload::NormalCurtain(curtain) => {
                let mut extent = curtain.extent;
                if length_field(ui, "Extent", &mut extent, factor, true) {
                    // Binormals are derived at construction, so edits go
                    // through the validating constructor.
                    if let Ok(rebuilt) = NormalCurtain::new(
                        curtain.points.clone(),
                        curtain.normals.clone(),
                        extent,
                    ) {
                        *curtain = rebuilt;
                        changed = true;
                    }
                }
                let curtain_row = |ui: &mut egui::Ui, index: usize| {
                    let (point, normal) = (curtain.points[index], curtain.normals[index]);
                    ui.weak(format!(
                        "P{index} ({:.3}, {:.3}, {:.3})  n ({:.2}, {:.2}, {:.2})",
                        point.x / factor,
                        point.y / factor,
                        point.z / factor,
                        normal.x,
                        normal.y,
                        normal.z
                    ));
                };
                if curtain.points.len() <= INLINE_POINT_ROWS {
                    for index in 0..curtain.points.len() {
                        curtain_row(ui, index);
                    }
                } else {
                    ui.weak(format!("Points: {}", curtain.points.len()));
                    let row_height = ui.text_style_height(&egui::TextStyle::Body);
                    egui::ScrollArea::vertical()
                        .id_salt("curtain_points")
                        .max_height(row_height * INLINE_POINT_ROWS as f32)
                        .show_rows(ui, row_height, curtain.points.len(), |ui, rows| {
                            for index in rows {
                                curtain_row(ui, index);
                            }
                        });
                }
            }
            ScenePayload::Placed2D {
                profile,
                origin,
                axis_u,
                axis_v,
                ..
            }
            | ScenePayload::PlacedPolyline1D {
                profile,
                origin,
                axis_u,
                axis_v,
            } => {
                changed |= vec3_field(ui, "Origin", origin, factor);
                weak_axis_row(ui, "Plane normal", axis_u.cross(*axis_v));
                changed |= profile2d_ui(ui, profile, *origin, *axis_u, *axis_v, factor);
            }
            ScenePayload::Placed1D {
                profile,
                origin,
                axis_u,
                ..
            } => {
                changed |= vec3_field(ui, "Origin", origin, factor);
                weak_axis_row(ui, "Direction", *axis_u);
                changed |= profile1d_ui(ui, profile, *origin, *axis_u, factor);
            }
            ScenePayload::Translate { child, offset } => {
                changed |= vec3_field(ui, "Offset", offset, factor);
                child_row(ui, document, "Child:", *child);
            }
            ScenePayload::Rotate {
                child,
                axis,
                angle_degrees,
            } => {
                ui.horizontal(|ui| {
                    ui.label("Axis");
                    egui::ComboBox::from_id_salt("rotate_axis")
                        .selected_text(axis.as_str())
                        .show_ui(ui, |ui| {
                            for option in [RotationAxis::X, RotationAxis::Y, RotationAxis::Z] {
                                changed |=
                                    ui.selectable_value(axis, option, option.as_str()).changed();
                            }
                        });
                });
                changed |= angle_field(ui, "Angle", angle_degrees);
                child_row(ui, document, "Child:", *child);
            }
            ScenePayload::Scale {
                child,
                factor: scale,
            } => {
                ui.horizontal(|ui| {
                    ui.label("Factor");
                    changed |= ui
                        .add(
                            egui::DragValue::new(scale)
                                .speed(0.01)
                                .range(1e-6..=f64::INFINITY)
                                .max_decimals(6),
                        )
                        .changed();
                });
                child_row(ui, document, "Child:", *child);
            }
            ScenePayload::Extrude {
                section,
                height,
                center_offset,
            } => {
                changed |= length_field(ui, "Height", height, factor, true);
                changed |= length_field(ui, "Center offset", center_offset, factor, false);
                child_row(ui, document, "Section:", *section);
            }
            ScenePayload::Revolve {
                section,
                axis,
                axis_origin,
                axis_direction,
                radial_direction,
                angle_degrees,
            } => {
                ui.horizontal(|ui| {
                    ui.label("Axis");
                    egui::ComboBox::from_id_salt("revolve_axis")
                        .selected_text(axis.as_str())
                        .show_ui(ui, |ui| {
                            for option in [RevolveAxis::U, RevolveAxis::V] {
                                changed |=
                                    ui.selectable_value(axis, option, option.as_str()).changed();
                            }
                        });
                });
                changed |= angle_field(ui, "Angle", angle_degrees);
                match axis_origin {
                    Some(point) => changed |= vec3_field(ui, "Axis origin", point, factor),
                    None => {
                        ui.weak("Axis origin: default");
                    }
                }
                match axis_direction {
                    Some(direction) => changed |= vec3_field(ui, "Axis direction", direction, 1.0),
                    None => {
                        ui.weak("Axis direction: default");
                    }
                }
                match radial_direction {
                    Some(direction) => {
                        changed |= vec3_field(ui, "Radial direction", direction, 1.0)
                    }
                    None => {
                        ui.weak("Radial direction: default");
                    }
                }
                child_row(ui, document, "Section:", *section);
            }
            ScenePayload::Operator { kind, left, right } => {
                ui.horizontal(|ui| {
                    ui.label("Operation");
                    egui::ComboBox::from_id_salt("operator_kind")
                        .selected_text(kind.as_str())
                        .show_ui(ui, |ui| {
                            for option in [
                                OperatorKind::Union,
                                OperatorKind::Intersection,
                                OperatorKind::Difference,
                                OperatorKind::Xor,
                            ] {
                                changed |=
                                    ui.selectable_value(kind, option, option.as_str()).changed();
                            }
                        });
                });
                child_row(ui, document, "Left:", *left);
                child_row(ui, document, "Right:", *right);
            }
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;
    use caso_kernel::scene::SceneDocument;
    use caso_kernel::sdf::primitives_3d::{
        Box3, BoxFrame, CappedCone, Cone, Cylinder, Pyramid, Sphere, Torus,
    };
    use caso_kernel::sdf::solid_from_2d::RevolveAxis;
    use caso_kernel::sdf::tubes::{PolylineTube, QuadraticBezierTube};
    use caso_kernel::vec3::vec3;

    /// Renders the panel once for every payload kind (frames, tubes, placed
    /// profiles, transforms, operators, curtains) to catch runtime panics
    /// the type checker cannot see.
    #[test]
    fn panel_renders_every_payload_kind() {
        let mut document = SceneDocument::new();
        let frame = Frame::from_euler_degrees(30.0, -20.0, 45.0);
        let x = vec3(1.0, 0.0, 0.0);
        let y = vec3(0.0, 1.0, 0.0);
        let origin = vec3(0.5, 0.5, 0.0);
        let mut ids = Vec::new();
        let add = |document: &mut SceneDocument, name: &str, payload: ScenePayload| {
            document.insert_object(name, payload).expect(name)
        };

        let sphere = add(
            &mut document,
            "sphere",
            ScenePayload::Sphere(Sphere {
                center: origin,
                radius: 1.0,
            }),
        );
        ids.push(sphere);
        let box3 = add(
            &mut document,
            "box",
            ScenePayload::Box3(Box3 {
                center: origin,
                half_size: vec3(1.0, 2.0, 3.0),
                frame,
            }),
        );
        ids.push(box3);
        ids.push(add(
            &mut document,
            "cylinder",
            ScenePayload::Cylinder(Cylinder {
                center: origin,
                radius: 1.0,
                half_height: 2.0,
                frame,
            }),
        ));
        ids.push(add(
            &mut document,
            "cone",
            ScenePayload::Cone(Cone {
                center: origin,
                radius: 1.0,
                half_height: 2.0,
                frame,
            }),
        ));
        ids.push(add(
            &mut document,
            "capped_cone",
            ScenePayload::CappedCone(CappedCone {
                center: origin,
                radius_a: 1.0,
                radius_b: 0.5,
                half_height: 2.0,
                frame,
            }),
        ));
        ids.push(add(
            &mut document,
            "pyramid",
            ScenePayload::Pyramid(Pyramid {
                center: origin,
                base_half_size: 1.0,
                half_height: 2.0,
                frame,
            }),
        ));
        ids.push(add(
            &mut document,
            "box_frame",
            ScenePayload::BoxFrame(BoxFrame {
                center: origin,
                half_size: vec3(1.0, 2.0, 3.0),
                thickness: 0.1,
                frame,
            }),
        ));
        ids.push(add(
            &mut document,
            "torus",
            ScenePayload::Torus(Torus {
                center: origin,
                major_radius: 2.0,
                minor_radius: 0.5,
                frame,
            }),
        ));
        ids.push(add(
            &mut document,
            "polyline_tube",
            ScenePayload::PolylineTube(
                PolylineTube::new(
                    vec![origin, vec3(1.0, 1.0, 1.0), vec3(2.0, 0.0, 1.0)],
                    0.2,
                    0.05,
                    CapStyle::Flat,
                )
                .expect("tube"),
            ),
        ));
        ids.push(add(
            &mut document,
            "bezier_tube",
            ScenePayload::QuadraticBezierTube(
                QuadraticBezierTube::new(
                    vec![origin, vec3(1.0, 1.0, 1.0), vec3(2.0, 0.0, 1.0)],
                    0.2,
                    0.05,
                    CapStyle::Round,
                )
                .expect("tube"),
            ),
        ));
        ids.push(add(
            &mut document,
            "curtain",
            ScenePayload::NormalCurtain(
                NormalCurtain::new(
                    vec![origin, vec3(1.0, 0.0, 0.0), vec3(2.0, 1.0, 0.0)],
                    vec![vec3(0.0, 0.0, 1.0); 3],
                    0.5,
                )
                .expect("curtain"),
            ),
        ));
        let polygon = add(
            &mut document,
            "polygon",
            ScenePayload::Placed2D {
                profile: Profile2D::polygon(vec![[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
                    .expect("polygon"),
                origin,
                axis_u: x,
                axis_v: y,
                sources: Vec::new(),
            },
        );
        ids.push(polygon);
        ids.push(add(
            &mut document,
            "boolean_2d",
            ScenePayload::Placed2D {
                profile: Profile2D::Binary {
                    left: Box::new(Profile2D::Circle {
                        center: [0.0, 0.0],
                        radius: 1.0,
                    }),
                    right: Box::new(Profile2D::Offset {
                        child: Box::new(Profile2D::RegularPolygon {
                            center: [0.0, 0.0],
                            radius: 0.5,
                            side_count: 6,
                            rotation: 0.3,
                        }),
                        offset: [1.5, 0.0],
                    }),
                    operation: caso_kernel::sdf::primitives_1d::BooleanOp1D::Difference,
                    smoothing: 0.1,
                },
                origin,
                axis_u: x,
                axis_v: y,
                sources: Vec::new(),
            },
        ));
        ids.push(add(
            &mut document,
            "polyline_1d",
            ScenePayload::PlacedPolyline1D {
                profile: Profile2D::polyline(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
                    .expect("polyline"),
                origin,
                axis_u: x,
                axis_v: y,
            },
        ));
        ids.push(add(
            &mut document,
            "bezier_curve",
            ScenePayload::PlacedPolyline1D {
                profile: Profile2D::quadratic_bezier_curve(vec![
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                ])
                .expect("bezier"),
                origin,
                axis_u: x,
                axis_v: y,
            },
        ));
        ids.push(add(
            &mut document,
            "segment",
            ScenePayload::Placed1D {
                profile: Profile1D::segment(0.5, 1.0).expect("segment"),
                origin,
                axis_u: x,
                sources: Vec::new(),
            },
        ));
        ids.push(add(
            &mut document,
            "translate",
            ScenePayload::Translate {
                child: sphere,
                offset: vec3(1.0, 0.0, 0.0),
            },
        ));
        ids.push(add(
            &mut document,
            "rotate",
            ScenePayload::Rotate {
                child: sphere,
                axis: RotationAxis::Z,
                angle_degrees: 30.0,
            },
        ));
        ids.push(add(
            &mut document,
            "scale",
            ScenePayload::Scale {
                child: sphere,
                factor: 2.0,
            },
        ));
        ids.push(add(
            &mut document,
            "extrude",
            ScenePayload::Extrude {
                section: polygon,
                height: 1.0,
                center_offset: 0.0,
            },
        ));
        ids.push(add(
            &mut document,
            "revolve_defaults",
            ScenePayload::Revolve {
                section: polygon,
                axis: RevolveAxis::V,
                axis_origin: None,
                axis_direction: None,
                radial_direction: None,
                angle_degrees: 180.0,
            },
        ));
        ids.push(add(
            &mut document,
            "revolve_explicit",
            ScenePayload::Revolve {
                section: polygon,
                axis: RevolveAxis::U,
                axis_origin: Some(origin),
                axis_direction: Some(vec3(0.0, 0.0, 1.0)),
                radial_direction: Some(x),
                angle_degrees: 90.0,
            },
        ));
        ids.push(add(
            &mut document,
            "operator",
            ScenePayload::Operator {
                kind: OperatorKind::Difference,
                left: sphere,
                right: box3,
            },
        ));

        let ctx = egui::Context::default();
        for id in ids {
            let mut state = AppState::new(document.clone());
            state.selection = vec![id];
            let mut panel = PropertiesPanel::default();
            let _ = ctx.run_ui(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| panel.ui(ui, &mut state));
            });
        }
    }

    #[test]
    fn world_point_round_trip_matches_creation_click() {
        // A polygon vertex clicked at world (1.4, 2.2, 0) on the xy plane
        // with origin (1, 2, 0) is stored as uv (0.4, 0.2); the panel must
        // show the original world x,y,z again.
        let origin = vec3(1.0, 2.0, 0.0);
        let axis_u = vec3(1.0, 0.0, 0.0);
        let axis_v = vec3(0.0, 1.0, 0.0);
        let clicked = vec3(1.4, 2.2, 0.0);
        let offset = clicked - origin;
        let uv = [offset.dot(axis_u), offset.dot(axis_v)];
        let shown = origin + axis_u * uv[0] + axis_v * uv[1];
        assert!((shown - clicked).length() < 1e-12);
    }

    /// Rendering the panel with no input must never mutate the document:
    /// a spurious edit would bump the version and force a full surface
    /// rebuild every frame.
    #[test]
    fn idle_panel_render_keeps_document_version_stable() {
        let mut document = SceneDocument::default_scene().expect("scene");
        let points: Vec<[f64; 2]> = (0..200)
            .map(|i| {
                let a = i as f64 / 200.0 * std::f64::consts::TAU;
                [a.cos() * (2.0 + 0.3 * (7.0 * a).sin()), a.sin() * 2.0]
            })
            .collect();
        let polygon = document
            .insert_object(
                "probe_polygon",
                ScenePayload::Placed2D {
                    profile: Profile2D::polygon(points).expect("polygon"),
                    origin: vec3(0.0, 0.0, 0.0),
                    axis_u: vec3(1.0, 0.0, 0.0),
                    axis_v: vec3(0.0, 1.0, 0.0),
                    sources: Vec::new(),
                },
            )
            .expect("insert");
        let ids: Vec<_> = document.live_ids().into_iter().chain([polygon]).collect();
        let ctx = egui::Context::default();
        for id in ids {
            let mut state = AppState::new(document.clone());
            state.selection = vec![id];
            let mut panel = PropertiesPanel::default();
            let before = state.document.version;
            for _ in 0..3 {
                let _ = ctx.run_ui(egui::RawInput::default(), |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| panel.ui(ui, &mut state));
                });
            }
            assert_eq!(
                state.document.version, before,
                "panel render alone bumped document version for object {id}"
            );
        }
    }

    fn lerp2(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
        [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
    }

    #[test]
    fn polygon_point_edits_respect_the_minimum() {
        let policy = PointListPolicy::Single {
            min: 3,
            closed: true,
        };
        let mut points = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]];
        // Insert after the last corner wraps to the first: midpoint (1, 1).
        assert!(apply_point_edit(&mut points, PointEdit::InsertAfter(2), policy, lerp2));
        assert_eq!(points, vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 1.0]]);
        assert!(apply_point_edit(&mut points, PointEdit::Delete(3), policy, lerp2));
        // At the minimum of three corners, delete refuses.
        assert!(!apply_point_edit(&mut points, PointEdit::Delete(0), policy, lerp2));
        assert_eq!(points.len(), 3);
    }

    #[test]
    fn open_polyline_insert_extends_the_tail() {
        let policy = PointListPolicy::Single {
            min: 2,
            closed: false,
        };
        let mut points = vec![[0.0, 0.0], [1.0, 1.0]];
        assert!(apply_point_edit(&mut points, PointEdit::InsertAfter(1), policy, lerp2));
        // Tail insert mirrors the last segment instead of wrapping.
        assert_eq!(points, vec![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]);
    }

    #[test]
    fn bezier_span_insert_splits_without_changing_the_curve() {
        let policy = PointListPolicy::Spans { min: 3 };
        let mut points = vec![[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]];
        assert!(apply_point_edit(&mut points, PointEdit::InsertAfter(0), policy, lerp2));
        // De Casteljau split at t = 0.5: the midpoint anchor lies ON the
        // original curve at (1, 1), and the count stays odd.
        assert_eq!(
            points,
            vec![[0.0, 0.0], [0.5, 1.0], [1.0, 1.0], [1.5, 1.0], [2.0, 0.0]]
        );
        // The last anchor splits the span that ends at it.
        assert!(apply_point_edit(&mut points, PointEdit::InsertAfter(4), policy, lerp2));
        assert_eq!(points.len(), 7);
        assert_eq!(points.len() % 2, 1, "span edits must keep the odd count");
    }

    #[test]
    fn bezier_span_delete_keeps_odd_count_and_minimum() {
        let policy = PointListPolicy::Spans { min: 3 };
        let mut points = vec![[0.0, 0.0], [0.5, 1.0], [1.0, 1.0], [1.5, 1.0], [2.0, 0.0]];
        // Control rows never edit; anchors drop a control+anchor pair.
        assert!(!apply_point_edit(&mut points, PointEdit::Delete(1), policy, lerp2));
        assert!(apply_point_edit(&mut points, PointEdit::Delete(2), policy, lerp2));
        assert_eq!(points, vec![[0.0, 0.0], [1.5, 1.0], [2.0, 0.0]]);
        // A single span is the floor: delete refuses.
        assert!(!apply_point_edit(&mut points, PointEdit::Delete(0), policy, lerp2));
        assert_eq!(points.len(), 3);
    }

    #[test]
    fn point_row_buttons_follow_the_policy() {
        let spans = PointListPolicy::Spans { min: 3 };
        assert_eq!(point_row_buttons(spans, 0, 5), (true, true));
        assert_eq!(point_row_buttons(spans, 1, 5), (false, false), "control row");
        assert_eq!(point_row_buttons(spans, 2, 3), (true, false), "at the span floor");
        let polygon = PointListPolicy::Single {
            min: 3,
            closed: true,
        };
        assert_eq!(point_row_buttons(polygon, 0, 3), (true, false));
        assert_eq!(point_row_buttons(polygon, 2, 4), (true, true));
    }
}
