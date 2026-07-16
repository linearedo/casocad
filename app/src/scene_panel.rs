//! Scene tree panel: hierarchical view of the document (roots first, operator
//! and transform children nested), selection, rename, delete, and domain-root
//! actions — the port of the Qt Scene dock's tree.

use caso_kernel::roles::DomainKind;
use caso_kernel::scene::{ObjectId, ScenePayload, TagRef};
use eframe::egui;

use crate::state::AppState;
use crate::theme;

/// Pending inline rename: (object id, text buffer).
#[derive(Default)]
pub struct ScenePanel {
    rename: Option<(ObjectId, String)>,
    region_rename: Option<(u32, String)>,
    /// Grab keyboard focus on the first frame the editor shows. Requesting
    /// focus every frame would mask `lost_focus`, so Enter could never commit.
    rename_focus: bool,
}

pub fn payload_label(payload: &ScenePayload) -> &'static str {
    match payload {
        ScenePayload::Sphere(_) => "Sphere",
        ScenePayload::Box3(_) => "Box",
        ScenePayload::Cylinder(_) => "Cylinder",
        ScenePayload::Cone(_) => "Cone",
        ScenePayload::CappedCone(_) => "Capped Cone",
        ScenePayload::Pyramid(_) => "Pyramid",
        ScenePayload::BoxFrame(_) => "Box Frame",
        ScenePayload::Torus(_) => "Torus",
        ScenePayload::PolylineTube(_) => "Polyline Tube",
        ScenePayload::QuadraticBezierTube(_) => "Bezier Tube",
        ScenePayload::NormalCurtain(_) => "Normal Curtain",
        ScenePayload::Placed2D { .. } => "2D Section",
        ScenePayload::PlacedPolyline1D { .. } => "1D Polyline",
        ScenePayload::Placed1D { .. } => "1D Segment",
        ScenePayload::Operator { kind, .. } => match kind.as_str() {
            "union" => "Union",
            "intersection" => "Intersection",
            "difference" => "Difference",
            _ => "Operator",
        },
        ScenePayload::Translate { .. } => "Translate",
        ScenePayload::Rotate { .. } => "Rotate",
        ScenePayload::Scale { .. } => "Scale",
        ScenePayload::Extrude { .. } => "Extrude",
        ScenePayload::Revolve { .. } => "Revolve",
    }
}

impl ScenePanel {
    /// One frame of the inline rename editor. Returns `None` while editing,
    /// `Some(None)` on cancel, `Some(Some(text))` on Enter.
    fn rename_editor(
        ui: &mut egui::Ui,
        buffer: &mut String,
        take_focus: &mut bool,
    ) -> Option<Option<String>> {
        let response = ui.text_edit_singleline(buffer);
        if *take_focus {
            response.request_focus();
            *take_focus = false;
        }
        if response.lost_focus() {
            if ui.input(|input| input.key_pressed(egui::Key::Enter)) {
                return Some(Some(buffer.trim().to_string()));
            }
            return Some(None);
        }
        if ui.input(|input| input.key_pressed(egui::Key::Escape)) {
            return Some(None);
        }
        None
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let roots: Vec<ObjectId> = state
                    .document
                    .roots
                    .iter()
                    .copied()
                    .filter(|id| state.document.object(*id).is_ok())
                    .collect();
                for root in roots {
                    self.node_ui(ui, state, root, true);
                }
                if !state.document.boundary_regions.is_empty() {
                    ui.separator();
                    ui.strong("Boundary Regions");
                    self.regions_ui(ui, state);
                }
            });
    }

    fn regions_ui(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
        let regions: Vec<(u32, String, Option<String>)> = state
            .document
            .boundary_regions
            .iter()
            .map(|region| (region.object_id, region.name.clone(), region.tag.clone()))
            .collect();
        for (region_id, name, tag) in regions {
            // Inline rename takes over the row until Enter/Esc.
            if let Some((rename_id, buffer)) = &mut self.region_rename {
                if *rename_id == region_id {
                    if let Some(outcome) =
                        Self::rename_editor(ui, buffer, &mut self.rename_focus)
                    {
                        self.region_rename = None;
                        if let Some(text) = outcome {
                            if !text.is_empty() && text != name {
                                state.push_undo();
                                for region in state.document.boundary_regions.iter_mut() {
                                    if region.object_id == region_id {
                                        region.name = text.clone();
                                    }
                                }
                                state.document.mark_changed();
                                state.status = format!("Renamed region to {text}");
                            }
                        }
                    }
                    continue;
                }
            }

            let selected = state.selected_region == Some(region_id);
            let mut title = name.clone();
            if let Some(tag) = &tag {
                title.push_str(&format!("  [{tag}]"));
            }
            let response = ui.selectable_label(selected, title);
            if response.clicked() {
                state.selected_region = if selected { None } else { Some(region_id) };
            }
            if response.double_clicked() {
                self.region_rename = Some((region_id, name.clone()));
                self.rename_focus = true;
            }
            response.context_menu(|ui| {
                if ui.button("Rename").clicked() {
                    self.region_rename = Some((region_id, name.clone()));
                    self.rename_focus = true;
                    ui.close();
                }
                if ui.button("Delete region").clicked() {
                    state.push_undo();
                    state
                        .document
                        .boundary_regions
                        .retain(|region| region.object_id != region_id);
                    if let Some(fluid) = state.document.fluid_domain.as_mut() {
                        fluid
                            .tags
                            .retain(|tag| *tag != TagRef::Region(region_id));
                    }
                    state.document.mark_changed();
                    state.retain_live_selection();
                    state.status = format!("Deleted region {name}");
                    ui.close();
                }
            });
        }
    }

    fn node_ui(&mut self, ui: &mut egui::Ui, state: &mut AppState, id: ObjectId, top_level: bool) {
        let Ok(object) = state.document.object(id) else {
            return;
        };
        let children = object.payload.children();
        let kind = payload_label(&object.payload);
        let name = object.name.clone();
        let selected = state.selection.contains(&id);
        let domain = state.document.domain_kinds.get(&id).copied();
        let is_fluid_root = state
            .document
            .fluid_domain
            .as_ref()
            .is_some_and(|fluid| fluid.root == id);

        // Inline rename takes over the row until Enter/Esc.
        if let Some((rename_id, buffer)) = &mut self.rename {
            if *rename_id == id {
                if let Some(outcome) = Self::rename_editor(ui, buffer, &mut self.rename_focus) {
                    self.rename = None;
                    if let Some(text) = outcome {
                        if !text.is_empty() && text != name {
                            state.push_undo();
                            let result = state.document.rename(id, text);
                            state.report(result, "Renamed");
                        }
                    }
                }
                for child in children {
                    ui.indent(id, |ui| self.node_ui(ui, state, child, false));
                }
                return;
            }
        }

        let mut title = format!("{name}  ·  {kind}");
        if is_fluid_root {
            title.push_str("  [Fluid Domain]");
        } else if let Some(kind) = domain {
            title.push_str(&format!("  [{} Domain]", kind.as_str()));
        }
        let text = if top_level {
            egui::RichText::new(title).color(theme::TEXT_COLOR)
        } else {
            egui::RichText::new(title).weak()
        };
        let response = ui.selectable_label(selected, text);
        if response.clicked() {
            if ui.input(|input| input.modifiers.ctrl) {
                state.toggle_select(id);
            } else {
                state.select_only(id);
            }
        }
        if response.double_clicked() {
            self.rename = Some((id, name.clone()));
            self.rename_focus = true;
        }
        response.context_menu(|ui| {
            if ui.button("Rename").clicked() {
                self.rename = Some((id, name.clone()));
                self.rename_focus = true;
                ui.close();
            }
            if ui.button("Delete").clicked() {
                state.push_undo();
                let removed = state.document.delete(id);
                state.retain_live_selection();
                state.status = format!("Deleted {removed} node(s)");
                ui.close();
            }
            ui.separator();
            // Domain marks are legal on nested objects too (a subtracted
            // solid stays a live domain inside the difference).
            if ui.button("Set Fluid Domain").clicked() {
                state.push_undo();
                let result = state.document.set_domain_root(id, DomainKind::Fluid);
                state.report(result, "Fluid domain set");
                ui.close();
            }
            if ui.button("Set Solid Domain").clicked() {
                state.push_undo();
                let result = state.document.set_domain_root(id, DomainKind::Solid);
                state.report(result, "Solid domain set");
                ui.close();
            }
            if (domain.is_some() || is_fluid_root) && ui.button("Unset Domain").clicked() {
                state.push_undo();
                state.document.unset_domain_root(id);
                state.status = "Domain unset".to_string();
                ui.close();
            }
        });

        for child in children {
            ui.indent(id, |ui| self.node_ui(ui, state, child, false));
        }
    }
}
