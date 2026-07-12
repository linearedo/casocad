//! caso-app — the casoWASM application: egui UI around the wgpu viewport.
//! One codebase for native desktop and the browser.

#![forbid(unsafe_code)]

mod boundary_tool;
mod dimensions;
mod gizmo;
mod meshing_panel;
mod properties_panel;
mod scene_panel;
mod script_runner;
mod state;
mod theme;
mod tools;
mod viewport_panel;

use caso_kernel::model::compile_model;
use caso_kernel::roles::Domain;
use caso_kernel::scene::{SceneDocument, ScenePayload};
use caso_kernel::sdf::solid_from_2d::RevolveAxis;
use caso_kernel::vec3::vec3;
use eframe::egui;

use dimensions::{parse_dimension_entry, parse_scalar_entry};
use meshing_panel::MeshingPanel;
use properties_panel::PropertiesPanel;
use scene_panel::ScenePanel;
use state::{AppState, LENGTH_UNITS};
use tools::{
    ToolKind, ToolState, DRAG_KINDS_1D, DRAG_KINDS_2D, DRAG_KINDS_3D, KNIFE_KINDS, POINT_KINDS,
};
use viewport_panel::ViewportPanel;

/// (menu label, `SceneDocument::add_primitive` kind key)
const PRIMITIVE_KINDS: [(&str, &str); 10] = [
    ("Sphere", "sphere"),
    ("Box", "box"),
    ("Cylinder", "cylinder"),
    ("Cone", "cone"),
    ("Capped Cone", "capped_cone"),
    ("Pyramid", "pyramid"),
    ("Box Frame", "box_frame"),
    ("Torus", "torus"),
    ("Polyline Tube", "polyline_tube"),
    ("Bezier Tube", "quadratic_bezier_tube"),
];

/// (button label, SDF operator key) — Difference is first − second.
const SDF_OPERATORS: [(&str, &str); 3] = [
    ("Union", "union"),
    ("Intersect", "intersection"),
    ("Subtract", "difference"),
];

const DISJOINTNESS_RESOLUTION: usize = 32;

/// The casoCAD wordmark (vector version of the Python app's title-bar logo),
/// rasterized by the SVG loader at the on-screen pixel size — crisp at any
/// DPI. Displayed at this height; width follows the SVG's 1572:216 aspect.
const WORDMARK_HEIGHT_POINTS: f32 = 22.0;
const WORDMARK_ASPECT: f32 = 1572.0 / 216.0;

/// Browser entry point: attaches the app to the `casowasm_canvas` element.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub async fn start() -> Result<(), wasm_bindgen::JsValue> {
    use eframe::wasm_bindgen::JsCast;
    let document = web_sys::window()
        .ok_or("no window")?
        .document()
        .ok_or("no document")?;
    let canvas = document
        .get_element_by_id("casowasm_canvas")
        .ok_or("missing #casowasm_canvas")?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;
    eframe::WebRunner::new()
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(|creation_context| Ok(Box::new(CasoApp::new(creation_context)))),
        )
        .await
}

#[derive(PartialEq, Clone, Copy)]
enum LeftTab {
    Scene,
    Properties,
    Meshing,
}

pub struct CasoApp {
    state: AppState,
    viewport: ViewportPanel,
    tools: ToolState,
    scene_panel: ScenePanel,
    properties_panel: PropertiesPanel,
    meshing_panel: MeshingPanel,
    left_tab: LeftTab,
    log: Vec<String>,
    show_log: bool,
    /// Copy buffer: a document snapshot plus the copied root ids.
    clipboard: Option<(SceneDocument, Vec<caso_kernel::scene::ObjectId>)>,
    extrude_height_text: String,
    revolve_angle_text: String,
    #[cfg(not(target_arch = "wasm32"))]
    scene_path: String,
}

impl CasoApp {
    pub fn new(creation_context: &eframe::CreationContext<'_>) -> Self {
        egui_extras::install_image_loaders(&creation_context.egui_ctx);
        theme::apply(&creation_context.egui_ctx);
        let document = SceneDocument::default_scene().expect("default scene");
        let mut state = AppState::new(document);
        state.status = "casoWASM — von Kármán demo scene".to_string();
        Self {
            state,
            viewport: ViewportPanel::default(),
            tools: ToolState::default(),
            scene_panel: ScenePanel::default(),
            properties_panel: PropertiesPanel::default(),
            meshing_panel: MeshingPanel::default(),
            left_tab: LeftTab::Scene,
            log: Vec::new(),
            show_log: false,
            clipboard: None,
            extrude_height_text: "1".to_string(),
            revolve_angle_text: "360".to_string(),
            #[cfg(not(target_arch = "wasm32"))]
            scene_path: "scene.json".to_string(),
        }
    }

    fn log_status(&mut self) {
        if !self.state.status.is_empty()
            && self.log.last().map(String::as_str) != Some(self.state.status.as_str())
        {
            self.log.push(self.state.status.clone());
            if self.log.len() > 500 {
                self.log.remove(0);
            }
        }
    }

    fn tool_button(&mut self, ui: &mut egui::Ui, label: &str, kind: ToolKind) {
        if ui
            .selectable_label(self.tools.kind == kind, label)
            .clicked()
        {
            if self.tools.kind == kind {
                self.tools.set_tool(ToolKind::Select, &mut self.state);
            } else {
                self.tools.set_tool(kind, &mut self.state);
            }
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.add(
                egui::Image::new(egui::include_image!("../assets/casocad-wordmark.svg"))
                    .fit_to_exact_size(egui::vec2(
                        WORDMARK_HEIGHT_POINTS * WORDMARK_ASPECT,
                        WORDMARK_HEIGHT_POINTS,
                    )),
            );
            ui.separator();

            ui.menu_button("Add", |ui| {
                for (label, kind) in PRIMITIVE_KINDS {
                    if ui.button(label).clicked() {
                        self.state.push_undo();
                        let scale = self.state.unit.factor;
                        let result = self.state.document.add_primitive(kind, scale);
                        if let Some(id) = self.state.report(result, &format!("Added {label}")) {
                            self.state.select_only(id);
                        }
                        ui.close();
                    }
                }
            });
            ui.menu_button("Draw", |ui| {
                ui.label(egui::RichText::new("3D (drag on grid)").weak());
                for (label, kind) in DRAG_KINDS_3D {
                    if ui.button(label).clicked() {
                        self.tools
                            .set_tool(ToolKind::CreateDrag(kind), &mut self.state);
                        ui.close();
                    }
                }
                ui.separator();
                ui.label(egui::RichText::new("2D sections").weak());
                for (label, kind) in DRAG_KINDS_2D {
                    if ui.button(label).clicked() {
                        self.tools
                            .set_tool(ToolKind::CreateDrag(kind), &mut self.state);
                        ui.close();
                    }
                }
                ui.separator();
                ui.label(egui::RichText::new("1D").weak());
                for (label, kind) in DRAG_KINDS_1D {
                    if ui.button(label).clicked() {
                        self.tools
                            .set_tool(ToolKind::CreateDrag(kind), &mut self.state);
                        ui.close();
                    }
                }
                ui.separator();
                ui.label(egui::RichText::new("Point tools (Enter commits)").weak());
                for (label, kind) in POINT_KINDS {
                    if ui.button(label).clicked() {
                        self.tools
                            .set_tool(ToolKind::CreatePoints(kind), &mut self.state);
                        ui.close();
                    }
                }
            });
            self.tool_button(ui, "Move", ToolKind::Move);
            self.tool_button(ui, "Rotate", ToolKind::Rotate);
            ui.separator();

            self.tool_button(ui, "Boundary Region", ToolKind::BoundaryRegion);
            ui.menu_button("Cutter", |ui| {
                if self.state.selected_region.is_none() {
                    ui.weak("Select a boundary region first.");
                }
                for (label, kind) in KNIFE_KINDS {
                    if ui.button(label).clicked() {
                        self.tools
                            .set_tool(ToolKind::BoundaryCutter(kind), &mut self.state);
                        ui.close();
                    }
                }
            });
            ui.separator();

            // SDF operators need exactly two selected operands.
            let two = self.state.selection.len() == 2;
            for (label, operation) in SDF_OPERATORS {
                if ui
                    .add_enabled(two, egui::Button::new(label))
                    .on_hover_text("Select exactly two SDF nodes (Subtract: first − second)")
                    .clicked()
                {
                    let (first, second) = (self.state.selection[0], self.state.selection[1]);
                    self.state.push_undo();
                    let result = self.state.document.combine(first, second, operation);
                    if let Some(id) = self.state.report(result, label) {
                        self.state.select_only(id);
                    }
                }
            }
            self.solid_from_2d_controls(ui);
            ui.separator();

            let undo = ui
                .add_enabled(self.state.can_undo(), egui::Button::new("Undo"))
                .clicked();
            let redo = ui
                .add_enabled(self.state.can_redo(), egui::Button::new("Redo"))
                .clicked();
            if undo {
                self.state.undo();
            }
            if redo {
                self.state.redo();
            }
            ui.separator();

            // Working unit: display-only — rescales camera and grid, never
            // the committed geometry (model stays in meters).
            let current = self.state.unit;
            egui::ComboBox::from_id_salt("working_unit")
                .selected_text(current.key)
                .width(52.0)
                .show_ui(ui, |ui| {
                    for unit in LENGTH_UNITS {
                        if ui
                            .selectable_label(unit.key == current.key, unit.label)
                            .clicked()
                            && unit.key != current.key
                        {
                            self.state.unit = unit;
                            self.viewport.set_working_unit(unit.factor);
                            self.state.status = format!("Working unit: {}", unit.label);
                        }
                    }
                });

            ui.checkbox(&mut self.viewport.options.show_grid, "Grid");
            ui.add(
                egui::Slider::new(&mut self.viewport.options.opacity, 0.05..=1.0).text("Opacity"),
            );
            ui.separator();

            if ui.button("Validate").on_hover_text("Compile the Model: exactness grammar, preconditions, Domain disjointness").clicked() {
                self.validate_domains();
            }
            ui.toggle_value(&mut self.show_log, "Log");
            ui.separator();

            #[cfg(not(target_arch = "wasm32"))]
            self.file_controls(ui);
            #[cfg(target_arch = "wasm32")]
            self.file_controls_web(ui);
        });
    }

    /// Extrude / Revolve from a selected placed 2D section.
    fn solid_from_2d_controls(&mut self, ui: &mut egui::Ui) {
        let section = self.state.selected_single().filter(|id| {
            matches!(
                self.state.document.object(*id).map(|object| &object.payload),
                Ok(ScenePayload::Placed2D { .. })
            )
        });
        let enabled = section.is_some();
        ui.menu_button("Solid From 2D", |ui| {
            if !enabled {
                ui.weak("Select one placed 2D section.");
                return;
            }
            let section = section.expect("checked");
            ui.horizontal(|ui| {
                ui.label("Height");
                ui.add(
                    egui::TextEdit::singleline(&mut self.extrude_height_text).desired_width(60.0),
                );
                ui.label(self.state.unit.key);
                if ui.button("Extrude").clicked() {
                    match parse_dimension_entry(&self.extrude_height_text, self.state.unit.factor)
                    {
                        Ok(values) => {
                            let height = values[0] * self.state.unit.factor;
                            self.state.push_undo();
                            let result = self.state.document.solid_from_2d(
                                section,
                                "extrude",
                                Some(height),
                                RevolveAxis::V,
                                None,
                                None,
                                None,
                                360.0,
                            );
                            if let Some(id) = self.state.report(result, "Extruded") {
                                self.state.select_only(id);
                            }
                        }
                        Err(error) => self.state.status = error,
                    }
                    ui.close();
                }
            });
            ui.horizontal(|ui| {
                ui.label("Angle");
                ui.add(
                    egui::TextEdit::singleline(&mut self.revolve_angle_text).desired_width(60.0),
                );
                ui.label("°");
                if ui.button("Revolve").clicked() {
                    match parse_scalar_entry(&self.revolve_angle_text) {
                        Ok(angle) => {
                            self.state.push_undo();
                            let result = self.state.document.solid_from_2d(
                                section,
                                "revolve",
                                None,
                                RevolveAxis::V,
                                None,
                                None,
                                None,
                                angle,
                            );
                            if let Some(id) = self.state.report(result, "Revolved") {
                                self.state.select_only(id);
                            }
                        }
                        Err(error) => self.state.status = error,
                    }
                    ui.close();
                }
            });
        });
    }

    /// Compile the document's named Domains against the exact-SDF contract.
    fn validate_domains(&mut self) {
        let mut domains = Vec::new();
        let mut entries: Vec<_> = self
            .state
            .document
            .domain_kinds
            .iter()
            .map(|(id, kind)| (*id, *kind))
            .collect();
        if let Some(fluid) = &self.state.document.fluid_domain {
            if !entries.iter().any(|(id, _)| *id == fluid.root) {
                entries.push((fluid.root, caso_kernel::roles::DomainKind::Fluid));
            }
        }
        if entries.is_empty() {
            self.state.status = "Validate: no Domains set".to_string();
            return;
        }
        for (id, kind) in entries {
            let name = match self.state.document.object(id) {
                Ok(object) => object.name.clone(),
                Err(error) => {
                    self.state.status = error.to_string();
                    return;
                }
            };
            let region = match self.state.document.build_node(id) {
                Ok(node) => node,
                Err(error) => {
                    self.state.status = error.to_string();
                    return;
                }
            };
            match Domain::new(name, kind, region) {
                Ok(domain) => domains.push(domain),
                Err(error) => {
                    self.state.status = format!("Validate: {error}");
                    return;
                }
            }
        }
        let count = domains.len();
        match caso_kernel::model::Model::new(domains)
            .and_then(|model| compile_model(&model, DISJOINTNESS_RESOLUTION))
        {
            Ok(()) => {
                self.state.status = format!("Validate: {count} Domain(s) compile cleanly");
            }
            Err(error) => self.state.status = format!("Validate: {error}"),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn file_controls(&mut self, ui: &mut egui::Ui) {
        ui.add(
            egui::TextEdit::singleline(&mut self.scene_path)
                .desired_width(160.0)
                .hint_text("scene.json"),
        );
        if ui.button("Save").clicked() {
            match caso_kernel::serialization::save_scene_to_string(&self.state.document)
                .map_err(|error| error.to_string())
                .and_then(|text| {
                    std::fs::write(&self.scene_path, text).map_err(|error| error.to_string())
                }) {
                Ok(()) => self.state.status = format!("Saved {}", self.scene_path),
                Err(error) => self.state.status = error,
            }
        }
        if ui.button("Load").clicked() {
            match std::fs::read_to_string(&self.scene_path)
                .map_err(|error| error.to_string())
                .and_then(|text| {
                    caso_kernel::serialization::load_scene_from_str(&text)
                        .map_err(|error| error.to_string())
                }) {
                Ok(document) => {
                    self.state.push_undo();
                    self.state.document = document;
                    self.state.document.mark_changed();
                    self.state.retain_live_selection();
                    self.viewport.request_frame_all();
                    self.state.status = format!("Loaded {}", self.scene_path);
                }
                Err(error) => self.state.status = error,
            }
        }
        ui.separator();
    }

    /// Web save/load: download the scene.json / open a picked file.
    #[cfg(target_arch = "wasm32")]
    fn file_controls_web(&mut self, ui: &mut egui::Ui) {
        if ui.button("Save").clicked() {
            match caso_kernel::serialization::save_scene_to_string(&self.state.document) {
                Ok(text) => {
                    if let Err(error) = web_download("scene.json", &text) {
                        self.state.status = format!("Download failed: {error:?}");
                    } else {
                        self.state.status = "Downloaded scene.json".to_string();
                    }
                }
                Err(error) => self.state.status = error.to_string(),
            }
        }
        ui.separator();
    }

    fn shortcuts(&mut self, ui: &egui::Ui) {
        // Skip document shortcuts while a text field owns the keyboard or a
        // create tool is collecting typed dimensions/points.
        if ui.ctx().memory(|memory| memory.focused().is_some()) {
            return;
        }
        let create_tool_active = matches!(
            self.tools.kind,
            ToolKind::CreateDrag(_) | ToolKind::CreatePoints(_)
        );
        let (undo, redo, delete, duplicate, copy, paste, escape) = ui.ctx().input(|input| {
            (
                input.modifiers.ctrl && input.key_pressed(egui::Key::Z) && !input.modifiers.shift,
                input.modifiers.ctrl
                    && (input.key_pressed(egui::Key::Y)
                        || (input.modifiers.shift && input.key_pressed(egui::Key::Z))),
                input.key_pressed(egui::Key::Delete),
                input.modifiers.ctrl && input.key_pressed(egui::Key::D),
                input.modifiers.ctrl && input.key_pressed(egui::Key::C),
                input.modifiers.ctrl && input.key_pressed(egui::Key::V),
                input.key_pressed(egui::Key::Escape),
            )
        });
        if escape && self.tools.is_active() && !create_tool_active {
            self.tools.set_tool(ToolKind::Select, &mut self.state);
        }
        if create_tool_active {
            // Esc inside create tools is handled by the tool itself; a second
            // Esc (nothing pending) leaves the tool.
            if escape && self.tools.points.is_empty() && self.tools.dimension_text.is_empty() {
                self.tools.set_tool(ToolKind::Select, &mut self.state);
            }
            return;
        }
        if undo {
            self.state.undo();
        }
        if redo {
            self.state.redo();
        }
        if delete && !self.state.selection.is_empty() {
            self.state.push_undo();
            let ids = self.state.selection.clone();
            let removed = self.state.document.delete_many(&ids);
            self.state.retain_live_selection();
            self.state.status = format!("Deleted {removed} node(s)");
        }
        if copy && !self.state.selection.is_empty() {
            self.clipboard = Some((self.state.document.snapshot(), self.state.selection.clone()));
            self.state.status = format!("Copied {} node(s)", self.state.selection.len());
        }
        if paste {
            if let Some((source, ids)) = self.clipboard.clone() {
                self.state.push_undo();
                let offset = vec3(0.1, 0.1, 0.0) * self.state.unit.factor;
                let mut pasted = Vec::new();
                for id in ids {
                    match self.state.document.import_subtree(&source, id, offset) {
                        Ok(new_id) => pasted.push(new_id),
                        Err(error) => self.state.status = error.to_string(),
                    }
                }
                if !pasted.is_empty() {
                    self.state.selection = pasted;
                    self.state.status = "Pasted".to_string();
                }
            }
        }
        if duplicate && !self.state.selection.is_empty() {
            self.state.push_undo();
            let ids = self.state.selection.clone();
            let offset = vec3(0.1, 0.1, 0.0) * self.state.unit.factor;
            let result = self.state.document.duplicate_nodes(&ids, offset);
            if let Some(pasted) = self.state.report(result, "Duplicated") {
                self.state.selection = pasted;
            }
        }
    }
}

/// Trigger a browser download of raw bytes as `filename`.
#[cfg(target_arch = "wasm32")]
pub(crate) fn web_download_bytes(
    filename: &str,
    bytes: &[u8],
) -> Result<(), wasm_bindgen::JsValue> {
    use eframe::wasm_bindgen::JsCast;
    let document = web_sys::window()
        .ok_or("no window")?
        .document()
        .ok_or("no document")?;
    let anchor: web_sys::HtmlAnchorElement = document.create_element("a")?.dyn_into()?;
    let array = js_sys::Uint8Array::from(bytes);
    let parts = js_sys::Array::new();
    parts.push(&array.buffer());
    let blob = web_sys::Blob::new_with_buffer_source_sequence(&parts)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)?;
    anchor.set_href(&url);
    anchor.set_download(filename);
    anchor.click();
    let _ = web_sys::Url::revoke_object_url(&url);
    Ok(())
}

/// Trigger a browser download of `contents` as `filename`.
#[cfg(target_arch = "wasm32")]
fn web_download(filename: &str, contents: &str) -> Result<(), wasm_bindgen::JsValue> {
    use eframe::wasm_bindgen::JsCast;
    let document = web_sys::window()
        .ok_or("no window")?
        .document()
        .ok_or("no document")?;
    let anchor: web_sys::HtmlAnchorElement =
        document.create_element("a")?.dyn_into()?;
    let encoded = js_sys::encode_uri_component(contents);
    anchor.set_href(&format!("data:application/json;charset=utf-8,{encoded}"));
    anchor.set_download(filename);
    anchor.click();
    Ok(())
}

impl eframe::App for CasoApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let Some(render_state) = frame.wgpu_render_state().cloned() else {
            return;
        };
        self.shortcuts(ui);
        egui::Panel::top("toolbar").show(ui, |ui| self.toolbar(ui));
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.state.status);
            });
        });
        if self.show_log {
            egui::Panel::bottom("log")
                .resizable(true)
                .default_size(120.0)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            for line in &self.log {
                                ui.monospace(line);
                            }
                        });
                });
        }
        egui::Panel::left("dock")
            .resizable(true)
            .default_size(260.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.left_tab, LeftTab::Scene, "Scene");
                    ui.selectable_value(&mut self.left_tab, LeftTab::Properties, "Properties");
                    ui.selectable_value(&mut self.left_tab, LeftTab::Meshing, "Meshing");
                });
                ui.separator();
                match self.left_tab {
                    LeftTab::Scene => self.scene_panel.ui(ui, &mut self.state),
                    LeftTab::Properties => self.properties_panel.ui(ui, &mut self.state),
                    LeftTab::Meshing => self.meshing_panel.ui(ui, &mut self.state),
                }
            });
        self.viewport.set_selection(self.state.selected_single());
        if self.viewport.mesh_preview_revision() != self.meshing_panel.preview_revision {
            let surfaces = self.meshing_panel.preview_surfaces();
            let points = self.meshing_panel.preview_points();
            self.viewport
                .set_mesh_preview(self.meshing_panel.preview_revision, surfaces, points);
        }
        self.viewport
            .ui(ui, &mut self.state, &mut self.tools, &render_state);
        self.log_status();
    }
}
