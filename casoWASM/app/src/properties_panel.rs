//! Properties panel: unit-aware editing of the selected object's placement
//! and dimensions. All lengths are stored in meters; the widgets display and
//! accept values in the current working unit (the model is never rescaled).

use caso_kernel::scene::{ObjectId, ScenePayload};
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
        ui.separator();

        let changed = Self::payload_ui(ui, &mut payload, factor);
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
            state.document.mark_changed();
            state.status = "Edited".to_string();
        }
    }

    fn payload_ui(ui: &mut egui::Ui, payload: &mut ScenePayload, factor: f64) -> bool {
        let mut changed = false;
        match payload {
            ScenePayload::Sphere(sphere) => {
                changed |= vec3_field(ui, "Center", &mut sphere.center, factor);
                changed |= length_field(ui, "Radius", &mut sphere.radius, factor, true);
            }
            ScenePayload::Box3(shape) => {
                changed |= vec3_field(ui, "Center", &mut shape.center, factor);
                changed |= vec3_field(ui, "Half size", &mut shape.half_size, factor);
            }
            ScenePayload::Cylinder(shape) => {
                changed |= vec3_field(ui, "Center", &mut shape.center, factor);
                changed |= length_field(ui, "Radius", &mut shape.radius, factor, true);
                changed |= length_field(ui, "Half height", &mut shape.half_height, factor, true);
            }
            ScenePayload::Cone(shape) => {
                changed |= vec3_field(ui, "Center", &mut shape.center, factor);
                changed |= length_field(ui, "Radius", &mut shape.radius, factor, true);
                changed |= length_field(ui, "Half height", &mut shape.half_height, factor, true);
            }
            ScenePayload::CappedCone(shape) => {
                changed |= vec3_field(ui, "Center", &mut shape.center, factor);
                changed |= length_field(ui, "Radius A", &mut shape.radius_a, factor, true);
                changed |= length_field(ui, "Radius B", &mut shape.radius_b, factor, true);
                changed |= length_field(ui, "Half height", &mut shape.half_height, factor, true);
            }
            ScenePayload::Pyramid(shape) => {
                changed |= vec3_field(ui, "Center", &mut shape.center, factor);
                changed |=
                    length_field(ui, "Base half size", &mut shape.base_half_size, factor, true);
                changed |= length_field(ui, "Half height", &mut shape.half_height, factor, true);
            }
            ScenePayload::BoxFrame(shape) => {
                changed |= vec3_field(ui, "Center", &mut shape.center, factor);
                changed |= vec3_field(ui, "Half size", &mut shape.half_size, factor);
                changed |= length_field(ui, "Thickness", &mut shape.thickness, factor, true);
            }
            ScenePayload::Torus(shape) => {
                changed |= vec3_field(ui, "Center", &mut shape.center, factor);
                changed |= length_field(ui, "Major radius", &mut shape.major_radius, factor, true);
                changed |= length_field(ui, "Minor radius", &mut shape.minor_radius, factor, true);
            }
            ScenePayload::PolylineTube(tube) => {
                changed |= length_field(ui, "Radius", &mut tube.radius, factor, true);
                for (index, point) in tube.points.iter_mut().enumerate() {
                    changed |= vec3_field(ui, &format!("P{index}"), point, factor);
                }
            }
            ScenePayload::QuadraticBezierTube(tube) => {
                changed |= length_field(ui, "Radius", &mut tube.radius, factor, true);
                for (index, point) in tube.points.iter_mut().enumerate() {
                    changed |= vec3_field(ui, &format!("P{index}"), point, factor);
                }
            }
            ScenePayload::Placed2D { origin, .. }
            | ScenePayload::PlacedPolyline1D { origin, .. }
            | ScenePayload::Placed1D { origin, .. } => {
                changed |= vec3_field(ui, "Origin", origin, factor);
                ui.weak("Profile parameters: edit via viewport tools.");
            }
            ScenePayload::Translate { offset, .. } => {
                changed |= vec3_field(ui, "Offset", offset, factor);
            }
            ScenePayload::Rotate { angle_degrees, .. } => {
                changed |= angle_field(ui, "Angle", angle_degrees);
            }
            ScenePayload::Scale { factor: scale, .. } => {
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
            }
            ScenePayload::Extrude {
                height,
                center_offset,
                ..
            } => {
                changed |= length_field(ui, "Height", height, factor, true);
                changed |= length_field(ui, "Center offset", center_offset, factor, false);
            }
            ScenePayload::Revolve { angle_degrees, .. } => {
                changed |= angle_field(ui, "Angle", angle_degrees);
            }
            ScenePayload::Operator { .. } | ScenePayload::NormalCurtain(_) => {
                ui.weak("No editable parameters.");
            }
        }
        changed
    }
}
