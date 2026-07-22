//! Meshing Controls: Rhai refinement controls, native meshing, preview, and
//! Arrow/SU2 export.

use std::collections::{BTreeMap, BTreeSet};

use caso_kernel::meshing::meshable_domains_from_document;
use caso_kernel::scene::SceneDocument;
use caso_kernel::vec3::{vec3, Vec3};
use caso_meshing::{
    convert::{MeshConverter, CONVERTERS},
    element_wire_edges,
    quality::{analyze, QualityMetric, QualityReport},
    read_mesh_ir, write_mesh_ir, MeshIr,
};
use caso_surfaces::types::mesh_tag_color;
use caso_surfaces::{SurfaceStatus, ViewportSurface, ViewportSurfaceKey};
use eframe::egui;

use crate::script_runner::run_native_mesher;
use crate::state::AppState;

#[cfg(target_arch = "wasm32")]
use std::{cell::RefCell, rc::Rc};

/// A mesh file delivered by the async browser picker: (filename, contents).
#[cfg(target_arch = "wasm32")]
type PickedFile = Rc<RefCell<Option<(String, Vec<u8>)>>>;

const QUALITY_BANDS: usize = 32;

#[derive(Default)]
struct SurfaceBuffers {
    vertices: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<u32>,
    wire_indices: Vec<u32>,
}

pub struct MeshingPanel {
    pub mesh: MeshIr,
    pub show_preview: bool,
    /// Bumped when `mesh` changes (viewport rebuilds the preview overlay).
    pub preview_revision: u64,
    /// Per preview entity distance to the nearest Domain boundary.
    boundary_distances: BTreeMap<(String, u64), f64>,
    max_boundary_distance: f64,
    boundary_range: f64,
    inspector_active: bool,
    show_quality: bool,
    show_boundary_tags: bool,
    quality_metric: QualityMetric,
    mesh_revision: u64,
    quality_cache: BTreeMap<(u64, QualityMetric), QualityReport>,
    selected_tags: BTreeSet<u64>,
    z_min: f64,
    z_max: f64,
    z_lower: f64,
    z_upper: f64,
    /// Filename stem for browser downloads (empty uses the export default).
    #[cfg(target_arch = "wasm32")]
    download_name: String,
    /// File handed back by the async browser picker, consumed on the next
    /// frame (wasm is single-threaded, so `Rc<RefCell>` suffices).
    #[cfg(target_arch = "wasm32")]
    picked: PickedFile,
}

impl Default for MeshingPanel {
    fn default() -> Self {
        Self {
            mesh: MeshIr::default(),
            show_preview: true,
            preview_revision: 0,
            boundary_distances: BTreeMap::new(),
            max_boundary_distance: 0.0,
            boundary_range: 0.0,
            inspector_active: false,
            show_quality: true,
            show_boundary_tags: false,
            quality_metric: QualityMetric::ScaledJacobian,
            mesh_revision: 0,
            quality_cache: BTreeMap::new(),
            selected_tags: BTreeSet::new(),
            z_min: 0.0,
            z_max: 0.0,
            z_lower: 0.0,
            z_upper: 0.0,
            #[cfg(target_arch = "wasm32")]
            download_name: String::new(),
            #[cfg(target_arch = "wasm32")]
            picked: PickedFile::default(),
        }
    }
}

impl MeshingPanel {
    pub fn ui(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
        self.take_picked(state);
        ui.horizontal(|ui| {
            if ui.button("Generate Mesh").clicked() {
                let script = state.document.meshing.control_script.clone();
                match run_native_mesher(&state.document, &script) {
                    Ok(output) => {
                        state.status = format!(
                            "Native mesh: {} domain(s), {} point(s), {} cell(s)",
                            output.statistics.domains,
                            output.statistics.points,
                            output.statistics.cells
                        );
                        self.mesh = output.mesh;
                        self.mesh_replaced();
                        self.update_boundary_distances(&state.document);
                        self.preview_revision += 1;
                    }
                    Err(error) => state.status = format!("Meshing failed: {error}"),
                }
            }
            if ui
                .checkbox(&mut self.show_preview, "Preview")
                .on_hover_text("Show MeshIR topology in the viewport")
                .changed()
            {
                self.preview_revision += 1;
            }
            if ui
                .checkbox(&mut self.inspector_active, "Inspector")
                .on_hover_text("Open the mesh quality and boundary-tag inspector")
                .changed()
            {
                self.preview_revision += 1;
            }
            if self.mesh.entity_count() > 0 {
                ui.weak(format!("{} MeshIR entities", self.mesh.entity_count()));
            }
        });
        ui.horizontal(|ui| {
            #[cfg(target_arch = "wasm32")]
            ui.add(
                egui::TextEdit::singleline(&mut self.download_name)
                    .desired_width(140.0)
                    .hint_text("mesh"),
            );
            let export = ui
                .add_enabled(
                    self.mesh.entity_count() > 0,
                    egui::Button::new("Export .arrow"),
                )
                .clicked();
            if export {
                self.export(state);
            }
            ui.add_enabled_ui(self.mesh.entity_count() > 0, |ui| {
                ui.menu_button("Convert", |ui| {
                    for converter in CONVERTERS {
                        if ui.button(converter.label).clicked() {
                            self.convert(state, converter);
                            ui.close();
                        }
                    }
                });
            });
            if ui.button("Load .arrow…").clicked() {
                self.load(ui, state);
            }
        });
        ui.separator();
        ui.label("Meshing Controls");
        ui.weak("Empty script uses deterministic defaults. Scripts may mutate controls only.");
        ui.collapsing("Mesher options", |ui| {
            let options = &mut state.document.meshing.options;
            let mut changed = false;
            changed |= ui
                .add(
                    egui::DragValue::new(&mut options.cells_2d)
                        .range(1..=4096)
                        .prefix("2D cells: "),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut options.cells_3d)
                        .range(1..=512)
                        .prefix("3D cells: "),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut options.minimum_cross_cells)
                        .range(1..=128)
                        .prefix("Minimum across: "),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut options.max_cells)
                        .range(1..=1_000_000)
                        .prefix("Cell limit: "),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut options.max_adaptive_levels)
                        .range(0..=12)
                        .prefix("Adaptive levels: "),
                )
                .changed();
            if changed {
                state.document.mark_changed();
            }
        });
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if ui
                    .add(
                        egui::TextEdit::multiline(&mut state.document.meshing.control_script)
                            .code_editor()
                            .desired_width(f32::INFINITY)
                            .desired_rows(24),
                    )
                    .changed()
                {
                    state.document.mark_changed();
                }
            });
    }

    pub fn inspector_active(&self) -> bool {
        self.inspector_active
    }

    /// Bottom inspector shown by the application only while the Meshing tab
    /// is active. Its controls only bump the overlay revision; quality is
    /// cached against the independent mesh revision.
    pub fn inspector_ui(&mut self, ui: &mut egui::Ui, state: &AppState) {
        if self.mesh.entity_count() == 0 {
            ui.weak("Generate a native mesh or load a MeshIR artifact to inspect it.");
            return;
        }
        let mut changed = false;
        ui.horizontal(|ui| {
            changed |= ui.checkbox(&mut self.show_quality, "Quality").changed();
            changed |= ui
                .checkbox(&mut self.show_boundary_tags, "Boundary Tags")
                .changed();
            if self.max_boundary_distance > 0.0 {
                changed |= ui
                    .add(
                        egui::Slider::new(
                            &mut self.boundary_range,
                            0.0..=self.max_boundary_distance,
                        )
                        .text("Boundary distance"),
                    )
                    .on_hover_text(
                        "Preview only mesh topology within this distance of a Domain boundary",
                    )
                    .changed();
            }
        });
        if self.top_dimension() == Some(3) && (self.show_quality || self.show_boundary_tags) {
            let factor = state.unit.factor;
            let min = self.z_min / factor;
            let max = self.z_max / factor;
            let mut lower = self.z_lower / factor;
            let mut upper = self.z_upper / factor;
            changed |= ui
                .add(
                    egui::Slider::new(&mut lower, min..=upper)
                        .text(format!("Z lower ({})", state.unit.key)),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut upper, lower..=max)
                        .text(format!("Z upper ({})", state.unit.key)),
                )
                .changed();
            self.z_lower = lower * factor;
            self.z_upper = upper * factor;
        }
        if self.show_quality {
            ui.separator();
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("mesh_quality_metric")
                    .selected_text(self.quality_metric.label())
                    .show_ui(ui, |ui| {
                        for metric in QualityMetric::ALL {
                            changed |= ui
                                .selectable_value(&mut self.quality_metric, metric, metric.label())
                                .changed();
                        }
                    });
                quality_legend(ui);
            });
            let visible_ids: BTreeSet<u64> = self
                .mesh
                .cells
                .iter()
                .filter(|cell| {
                    self.cell_visible(cell.id)
                        && (!self.show_boundary_tags || self.cell_has_selected_boundary(cell.id))
                })
                .map(|cell| cell.id)
                .collect();
            let key = (self.mesh_revision, self.quality_metric);
            self.quality_report();
            let report = &self.quality_cache[&key];
            let visible: Vec<_> = report
                .cells
                .iter()
                .filter(|entry| visible_ids.contains(&entry.cell_id))
                .collect();
            let scored: Vec<_> = visible
                .iter()
                .filter_map(|entry| entry.score.map(|score| (entry.cell_id, score)))
                .collect();
            let minimum = scored.iter().map(|(_, score)| *score).reduce(f64::min);
            let maximum = scored.iter().map(|(_, score)| *score).reduce(f64::max);
            let mean = (!scored.is_empty())
                .then(|| scored.iter().map(|(_, score)| *score).sum::<f64>() / scored.len() as f64);
            let worst = scored.iter().min_by(|a, b| a.1.total_cmp(&b.1));
            ui.horizontal_wrapped(|ui| {
                ui.label(format!("Visible {}/{}", visible.len(), report.cells.len()));
                ui.separator();
                ui.label(format!("Min {}", format_score(minimum)));
                ui.label(format!("Mean {}", format_score(mean)));
                ui.label(format!("Max {}", format_score(maximum)));
                ui.label(format!(
                    "Worst ID {}",
                    worst.map_or("N/A".to_string(), |(id, _)| id.to_string())
                ));
                ui.label(format!(
                    "N/A {}",
                    visible.iter().filter(|entry| entry.score.is_none()).count()
                ));
            });
        }
        if self.show_boundary_tags {
            ui.separator();
            let tags = self.assigned_boundary_tags();
            ui.horizontal(|ui| {
                if ui.button("All").clicked() {
                    self.selected_tags = tags.iter().map(|(id, _)| *id).collect();
                    changed = true;
                }
                if ui.button("None").clicked() {
                    self.selected_tags.clear();
                    changed = true;
                }
                ui.weak(format!("{} assigned tag(s)", tags.len()));
            });
            egui::ScrollArea::horizontal().show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    for (id, name) in tags {
                        let mut selected = self.selected_tags.contains(&id);
                        if ui.checkbox(&mut selected, name).changed() {
                            if selected {
                                self.selected_tags.insert(id);
                            } else {
                                self.selected_tags.remove(&id);
                            }
                            changed = true;
                        }
                    }
                });
            });
        }
        if changed {
            self.preview_revision += 1;
        }
    }

    fn mesh_replaced(&mut self) {
        self.mesh_revision = self.mesh_revision.wrapping_add(1);
        self.quality_cache.clear();
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;
        for point in &self.mesh.points {
            z_min = z_min.min(point.position[2]);
            z_max = z_max.max(point.position[2]);
        }
        if !z_min.is_finite() {
            z_min = 0.0;
            z_max = 0.0;
        }
        self.z_min = z_min;
        self.z_max = z_max;
        self.z_lower = z_min;
        self.z_upper = z_max;
        self.selected_tags = self
            .assigned_boundary_tags()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        self.quality_cache.insert(
            (self.mesh_revision, self.quality_metric),
            analyze(&self.mesh, self.quality_metric),
        );
    }

    fn quality_report(&mut self) -> &QualityReport {
        self.quality_cache
            .entry((self.mesh_revision, self.quality_metric))
            .or_insert_with(|| analyze(&self.mesh, self.quality_metric))
    }

    fn top_dimension(&self) -> Option<u8> {
        self.mesh
            .cells
            .iter()
            .filter_map(|cell| mesh_dimension(&cell.type_name))
            .max()
    }

    fn cell_visible(&self, cell_id: u64) -> bool {
        let Some(cell) = self.mesh.cells.iter().find(|cell| cell.id == cell_id) else {
            return false;
        };
        self.is_shown("cell", cell_id) && self.ids_in_z(&cell.point_ids)
    }

    fn cell_has_selected_boundary(&self, cell_id: u64) -> bool {
        if self.top_dimension() == Some(3) {
            self.mesh.faces.iter().any(|face| {
                (face.owner_cell_id == Some(cell_id) || face.neighbor_cell_id == Some(cell_id))
                    && entity_has_selected_tag(&face.tag_ids, &self.selected_tags)
                    && self.is_shown("face", face.id)
                    && self.ids_in_z(&face.point_ids)
            })
        } else {
            self.mesh.edges.iter().any(|edge| {
                (edge.owner_cell_id == Some(cell_id) || edge.neighbor_cell_id == Some(cell_id))
                    && entity_has_selected_tag(&edge.tag_ids, &self.selected_tags)
                    && self.is_shown("edge", edge.id)
            })
        }
    }

    fn ids_in_z(&self, ids: &[u64]) -> bool {
        if self.top_dimension() != Some(3) {
            return true;
        }
        ids.iter().all(|id| {
            self.mesh.point(*id).is_some_and(|point| {
                point.position[2] >= self.z_lower && point.position[2] <= self.z_upper
            })
        })
    }

    fn assigned_boundary_tags(&self) -> Vec<(u64, String)> {
        let ids: BTreeSet<u64> = if self.top_dimension() == Some(3) {
            self.mesh
                .faces
                .iter()
                .flat_map(|face| face.tag_ids.iter().copied())
                .collect()
        } else {
            self.mesh
                .edges
                .iter()
                .flat_map(|edge| edge.tag_ids.iter().copied())
                .collect()
        };
        ids.into_iter()
            .filter_map(|id| self.mesh.tag_name(id).map(|name| (id, name.to_string())))
            .collect()
    }

    fn update_boundary_distances(&mut self, document: &SceneDocument) {
        self.boundary_distances.clear();
        self.max_boundary_distance = 0.0;
        let Ok(domains) = meshable_domains_from_document(document) else {
            self.boundary_range = 0.0;
            return;
        };
        let points: Vec<Vec3> = self
            .mesh
            .points
            .iter()
            .map(|point| vec3(point.position[0], point.position[1], point.position[2]))
            .collect();
        let mut nearest = vec![f64::INFINITY; points.len()];
        for domain in domains.iter() {
            for (slot, sdf) in nearest.iter_mut().zip(domain.domain_sdf(&points)) {
                *slot = slot.min(sdf.abs());
            }
        }
        let by_point: BTreeMap<u64, f64> = self
            .mesh
            .points
            .iter()
            .map(|point| point.id)
            .zip(nearest)
            .collect();

        let mut rows: Vec<(&'static str, u64, Vec<u64>)> = Vec::new();
        rows.extend(
            self.mesh
                .points
                .iter()
                .map(|point| ("point", point.id, vec![point.id])),
        );
        rows.extend(
            self.mesh
                .edges
                .iter()
                .map(|edge| ("edge", edge.id, edge.point_ids.clone())),
        );
        rows.extend(
            self.mesh
                .faces
                .iter()
                .map(|face| ("face", face.id, face.point_ids.clone())),
        );
        rows.extend(
            self.mesh
                .cells
                .iter()
                .map(|cell| ("cell", cell.id, cell.point_ids.clone())),
        );
        for (kind, id, point_ids) in rows {
            self.insert_distance(kind, id, &point_ids, &by_point);
        }
        if self.boundary_range <= 0.0 || self.boundary_range > self.max_boundary_distance {
            self.boundary_range = self.max_boundary_distance;
        }
    }

    fn insert_distance(
        &mut self,
        kind: &str,
        id: u64,
        point_ids: &[u64],
        by_point: &BTreeMap<u64, f64>,
    ) {
        let distance = point_ids
            .iter()
            .filter_map(|point_id| by_point.get(point_id))
            .copied()
            .fold(f64::INFINITY, f64::min);
        let distance = if distance.is_finite() { distance } else { 0.0 };
        self.boundary_distances
            .insert((kind.to_string(), id), distance);
        self.max_boundary_distance = self.max_boundary_distance.max(distance);
    }

    fn is_shown(&self, kind: &str, id: u64) -> bool {
        self.boundary_distances
            .get(&(kind.to_string(), id))
            .is_none_or(|distance| *distance <= self.boundary_range)
    }

    fn export(&mut self, state: &mut AppState) {
        let metadata = serde_json::json!({
            "source": "casowasm",
            "schema": "casocad.mesh_ir.v1",
            "point_count": self.mesh.points.len(),
            "edge_count": self.mesh.edges.len(),
            "face_count": self.mesh.faces.len(),
            "cell_count": self.mesh.cells.len(),
        });
        match write_mesh_ir(&self.mesh, &metadata) {
            Ok(bytes) => self.deliver(state, "mesh.arrow", bytes),
            Err(error) => state.status = format!("Arrow export failed: {error}"),
        }
    }

    fn convert(&mut self, state: &mut AppState, converter: &MeshConverter) {
        match (converter.write)(&self.mesh) {
            Ok(bytes) => {
                let name = format!("mesh.{}", converter.extension);
                self.deliver(state, &name, bytes);
            }
            Err(error) => state.status = format!("{} conversion failed: {error}", converter.label),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn deliver(&mut self, state: &mut AppState, default_name: &str, bytes: Vec<u8>) {
        let Some(path) = mesh_dialog().set_file_name(default_name).save_file() else {
            return;
        };
        match std::fs::write(&path, bytes) {
            Ok(()) => state.status = format!("Exported {}", path.display()),
            Err(error) => state.status = format!("Export failed: {error}"),
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn deliver(&mut self, state: &mut AppState, default_name: &str, bytes: Vec<u8>) {
        let name = download_name(&self.download_name, default_name);
        match crate::web_download_bytes(&name, &bytes) {
            Ok(()) => state.status = format!("Downloaded {name}"),
            Err(error) => state.status = format!("Download failed: {error:?}"),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load(&mut self, _ui: &egui::Ui, state: &mut AppState) {
        let Some(path) = mesh_dialog().pick_file() else {
            return;
        };
        let name = path.display().to_string();
        match std::fs::read(&path) {
            Ok(bytes) => self.apply_loaded(state, &name, &bytes),
            Err(error) => state.status = format!("Arrow load failed ({name}): {error}"),
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn load(&mut self, ui: &egui::Ui, _state: &mut AppState) {
        let picked = self.picked.clone();
        let ctx = ui.ctx().clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Some(file) = rfd::AsyncFileDialog::new()
                .add_filter("Arrow mesh", &["arrow"])
                .pick_file()
                .await
            {
                let bytes = file.read().await;
                *picked.borrow_mut() = Some((file.file_name(), bytes));
                ctx.request_repaint();
            }
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn take_picked(&mut self, _state: &mut AppState) {}

    #[cfg(target_arch = "wasm32")]
    fn take_picked(&mut self, state: &mut AppState) {
        let Some((name, bytes)) = self.picked.borrow_mut().take() else {
            return;
        };
        self.apply_loaded(state, &name, &bytes);
    }

    /// The loaded mesh replaces the panel mesh as-is: it is not validated
    /// against the current scene geometry.
    fn apply_loaded(&mut self, state: &mut AppState, name: &str, bytes: &[u8]) {
        match read_mesh_ir(bytes) {
            Ok((mesh, _metadata)) => {
                state.status = format!(
                    "Loaded {name}: {} point(s), {} cell(s)",
                    mesh.points.len(),
                    mesh.cells.len()
                );
                self.mesh = mesh;
                self.mesh_replaced();
                self.update_boundary_distances(&state.document);
                self.preview_revision += 1;
            }
            Err(error) => state.status = format!("Arrow load failed ({name}): {error}"),
        }
    }

    pub fn preview_points(&self) -> Vec<f32> {
        Vec::new()
    }

    pub fn preview_surfaces(&self) -> Vec<ViewportSurface> {
        if !self.show_preview || self.mesh.entity_count() == 0 {
            return Vec::new();
        }
        if !self.inspector_active || (!self.show_quality && !self.show_boundary_tags) {
            return self.full_mesh_surfaces();
        }
        match (self.show_quality, self.show_boundary_tags) {
            (true, true) => self.quality_boundary_surfaces(),
            (true, false) => self.quality_surfaces(),
            (false, true) => self.tag_surfaces(true),
            (false, false) => self.full_mesh_surfaces(),
        }
    }

    fn full_mesh_surfaces(&self) -> Vec<ViewportSurface> {
        let positions: BTreeMap<u64, [f64; 3]> = self
            .mesh
            .points
            .iter()
            .map(|point| (point.id, point.position))
            .collect();
        let mut groups: BTreeMap<String, SurfaceBuffers> = BTreeMap::new();
        if self.mesh.faces.is_empty() {
            for edge in &self.mesh.edges {
                if self.is_shown("edge", edge.id) {
                    append_element_wire(
                        groups.entry(edge_label(&self.mesh, edge)).or_default(),
                        &positions,
                        &edge.type_name,
                        &edge.point_ids,
                    );
                }
            }
        } else {
            let edges: BTreeMap<u64, &caso_meshing::MeshEdge> =
                self.mesh.edges.iter().map(|edge| (edge.id, edge)).collect();
            let mut selected: BTreeMap<u64, (bool, String)> = BTreeMap::new();
            for face in &self.mesh.faces {
                if !self.is_shown("face", face.id) {
                    continue;
                }
                let label = face_label(&self.mesh, face);
                let tagged = face
                    .tag_ids
                    .first()
                    .and_then(|id| self.mesh.tag_name(*id))
                    .is_some();
                for edge_id in &face.edge_ids {
                    selected
                        .entry(*edge_id)
                        .and_modify(|(old_tagged, old_label)| {
                            if tagged && !*old_tagged {
                                *old_tagged = true;
                                *old_label = label.clone();
                            }
                        })
                        .or_insert_with(|| (tagged, label.clone()));
                }
            }
            for (edge_id, (_, label)) in selected {
                let Some(edge) = edges.get(&edge_id) else {
                    continue;
                };
                append_element_wire(
                    groups.entry(label).or_default(),
                    &positions,
                    &edge.type_name,
                    &edge.point_ids,
                );
            }
        }
        groups
            .into_iter()
            .enumerate()
            .filter_map(|(index, (label, buffers))| {
                surface_from_buffers(
                    buffers,
                    mesh_tag_color(tag_color_id(&label)),
                    self.preview_revision,
                    200 + index as u32,
                )
            })
            .collect()
    }

    fn quality_surfaces(&self) -> Vec<ViewportSurface> {
        let Some(report) = self
            .quality_cache
            .get(&(self.mesh_revision, self.quality_metric))
        else {
            return Vec::new();
        };
        let positions: BTreeMap<u64, [f64; 3]> = self
            .mesh
            .points
            .iter()
            .map(|p| (p.id, p.position))
            .collect();
        let score: BTreeMap<u64, Option<f64>> = report
            .cells
            .iter()
            .map(|entry| (entry.cell_id, entry.score))
            .collect();
        let mut groups: Vec<SurfaceBuffers> = (0..=QUALITY_BANDS)
            .map(|_| SurfaceBuffers::default())
            .collect();

        if report.top_dimension == Some(2) {
            let mut edge_bands: BTreeMap<(u64, u64), (usize, u64, u64)> = BTreeMap::new();
            for cell in self
                .mesh
                .cells
                .iter()
                .filter(|cell| score.contains_key(&cell.id) && self.cell_visible(cell.id))
            {
                let Some(ids) = cell_corner_ids(&cell.type_name, &cell.point_ids) else {
                    continue;
                };
                let band = quality_band(score[&cell.id]);
                append_polygon(&mut groups[band], &positions, ids, true, false);
                for i in 0..ids.len() {
                    let (a, b) = (ids[i], ids[(i + 1) % ids.len()]);
                    let key = if a < b { (a, b) } else { (b, a) };
                    edge_bands
                        .entry(key)
                        .and_modify(|entry| entry.0 = worse_band(entry.0, band))
                        .or_insert((band, a, b));
                }
            }
            for (_, (band, a, b)) in edge_bands {
                append_line(&mut groups[band], &positions, a, b);
            }
        } else if report.top_dimension == Some(3) {
            let selected: BTreeSet<u64> = report
                .cells
                .iter()
                .filter(|entry| self.cell_visible(entry.cell_id))
                .map(|entry| entry.cell_id)
                .collect();
            for face in &self.mesh.faces {
                let owner = face.owner_cell_id.filter(|id| selected.contains(id));
                let neighbor = face.neighbor_cell_id.filter(|id| selected.contains(id));
                if owner.is_some() == neighbor.is_some()
                    || !self.is_shown("face", face.id)
                    || !self.ids_in_z(&face.point_ids)
                {
                    continue;
                }
                let cell_id = owner.or(neighbor).expect("one selected cell");
                let band = quality_band(score.get(&cell_id).copied().flatten());
                if let Some(ids) = face_corner_ids(&face.type_name, &face.point_ids) {
                    append_polygon(&mut groups[band], &positions, ids, true, false);
                }
            }
            let edges: BTreeMap<u64, &caso_meshing::MeshEdge> =
                self.mesh.edges.iter().map(|edge| (edge.id, edge)).collect();
            let mut edge_bands: BTreeMap<u64, usize> = BTreeMap::new();
            for cell in self
                .mesh
                .cells
                .iter()
                .filter(|cell| selected.contains(&cell.id))
            {
                let band = quality_band(score[&cell.id]);
                for edge_id in &cell.edge_ids {
                    edge_bands
                        .entry(*edge_id)
                        .and_modify(|old| *old = worse_band(*old, band))
                        .or_insert(band);
                }
            }
            for (edge_id, band) in edge_bands {
                let Some(edge) = edges.get(&edge_id) else {
                    continue;
                };
                if self.is_shown("edge", edge.id) && self.ids_in_z(&edge.point_ids) {
                    if let Some((a, b)) = edge_endpoints(&edge.type_name, &edge.point_ids) {
                        append_line(&mut groups[band], &positions, a, b);
                    }
                }
            }
        }
        quality_group_surfaces(groups, self.preview_revision)
    }

    fn quality_boundary_surfaces(&self) -> Vec<ViewportSurface> {
        let Some(report) = self
            .quality_cache
            .get(&(self.mesh_revision, self.quality_metric))
        else {
            return Vec::new();
        };
        let positions: BTreeMap<u64, [f64; 3]> = self
            .mesh
            .points
            .iter()
            .map(|point| (point.id, point.position))
            .collect();
        let scores: BTreeMap<u64, Option<f64>> = report
            .cells
            .iter()
            .map(|entry| (entry.cell_id, entry.score))
            .collect();
        let mut groups: Vec<SurfaceBuffers> = (0..=QUALITY_BANDS)
            .map(|_| SurfaceBuffers::default())
            .collect();

        if report.top_dimension == Some(3) {
            for face in &self.mesh.faces {
                if !entity_has_selected_tag(&face.tag_ids, &self.selected_tags)
                    || !self.is_shown("face", face.id)
                    || !self.ids_in_z(&face.point_ids)
                {
                    continue;
                }
                let Some(cell_id) = face
                    .owner_cell_id
                    .filter(|id| scores.contains_key(id))
                    .or_else(|| face.neighbor_cell_id.filter(|id| scores.contains_key(id)))
                else {
                    continue;
                };
                let band = quality_band(scores[&cell_id]);
                if let Some(ids) = face_corner_ids(&face.type_name, &face.point_ids) {
                    append_polygon(&mut groups[band], &positions, ids, true, true);
                }
            }
        } else if report.top_dimension == Some(2) {
            for edge in &self.mesh.edges {
                if !entity_has_selected_tag(&edge.tag_ids, &self.selected_tags)
                    || !self.is_shown("edge", edge.id)
                {
                    continue;
                }
                let Some(cell_id) = edge
                    .owner_cell_id
                    .filter(|id| scores.contains_key(id))
                    .or_else(|| edge.neighbor_cell_id.filter(|id| scores.contains_key(id)))
                else {
                    continue;
                };
                append_element_wire(
                    &mut groups[quality_band(scores[&cell_id])],
                    &positions,
                    &edge.type_name,
                    &edge.point_ids,
                );
            }
        }
        quality_group_surfaces(groups, self.preview_revision)
    }

    fn tag_surfaces(&self, fill_faces: bool) -> Vec<ViewportSurface> {
        let positions: BTreeMap<u64, [f64; 3]> = self
            .mesh
            .points
            .iter()
            .map(|p| (p.id, p.position))
            .collect();
        let mut groups: BTreeMap<u64, SurfaceBuffers> = BTreeMap::new();
        if self.top_dimension() == Some(3) {
            for face in &self.mesh.faces {
                let Some(tag) = face
                    .tag_ids
                    .iter()
                    .copied()
                    .find(|id| self.selected_tags.contains(id))
                else {
                    continue;
                };
                if !self.is_shown("face", face.id) || !self.ids_in_z(&face.point_ids) {
                    continue;
                }
                if let Some(ids) = face_corner_ids(&face.type_name, &face.point_ids) {
                    append_polygon(
                        groups.entry(tag).or_default(),
                        &positions,
                        ids,
                        fill_faces,
                        true,
                    );
                }
            }
        } else {
            for edge in &self.mesh.edges {
                let Some(tag) = edge
                    .tag_ids
                    .iter()
                    .copied()
                    .find(|id| self.selected_tags.contains(id))
                else {
                    continue;
                };
                if !self.is_shown("edge", edge.id) {
                    continue;
                }
                if let Some((a, b)) = edge_endpoints(&edge.type_name, &edge.point_ids) {
                    append_line(groups.entry(tag).or_default(), &positions, a, b);
                }
            }
        }
        groups
            .into_iter()
            .enumerate()
            .filter_map(|(index, (tag, buffers))| {
                let label = self.mesh.tag_name(tag).unwrap_or("boundary");
                surface_from_buffers(
                    buffers,
                    mesh_tag_color(tag_color_id(label)),
                    self.preview_revision,
                    QUALITY_BANDS as u32 + 1 + index as u32,
                )
            })
            .collect()
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn mesh_dialog() -> rfd::FileDialog {
    rfd::FileDialog::new().add_filter("Arrow mesh", &["arrow"])
}

/// The download filename: trimmed user entry with `.arrow` appended when
/// missing; `mesh.arrow` when empty.
#[cfg(target_arch = "wasm32")]
fn download_name(raw: &str, default_name: &str) -> String {
    let name = raw.trim();
    if name.is_empty() {
        default_name.to_string()
    } else if default_name
        .rsplit_once('.')
        .is_some_and(|(_, extension)| name.ends_with(&format!(".{extension}")))
    {
        name.to_string()
    } else {
        let extension = default_name
            .rsplit_once('.')
            .map_or("", |(_, extension)| extension);
        format!("{name}.{extension}")
    }
}

fn append_line(buffers: &mut SurfaceBuffers, points: &BTreeMap<u64, [f64; 3]>, a: u64, b: u64) {
    let (Some(a), Some(b)) = (points.get(&a), points.get(&b)) else {
        return;
    };
    let base = buffers.vertices.len() as u32;
    buffers.vertices.extend([to_f32(*a), to_f32(*b)]);
    buffers.normals.extend([[0.0, 0.0, 1.0]; 2]);
    buffers.wire_indices.extend([base, base + 1]);
}

fn append_element_wire(
    buffers: &mut SurfaceBuffers,
    points: &BTreeMap<u64, [f64; 3]>,
    type_name: &str,
    point_ids: &[u64],
) {
    for (a, b) in element_wire_edges(type_name, point_ids) {
        append_line(buffers, points, a, b);
    }
}

fn append_polygon(
    buffers: &mut SurfaceBuffers,
    points: &BTreeMap<u64, [f64; 3]>,
    ids: &[u64],
    fill: bool,
    outline: bool,
) {
    let p: Vec<[f64; 3]> = ids
        .iter()
        .filter_map(|id| points.get(id).copied())
        .collect();
    if p.len() != ids.len() || p.len() < 3 {
        return;
    }
    let normal = polygon_normal(&p);
    let base = buffers.vertices.len() as u32;
    buffers.vertices.extend(p.iter().copied().map(to_f32));
    buffers.normals.extend(std::iter::repeat_n(normal, p.len()));
    if fill {
        for i in 1..p.len() - 1 {
            buffers
                .indices
                .extend([base, base + i as u32, base + i as u32 + 1]);
        }
    }
    if outline {
        for i in 0..p.len() {
            buffers
                .wire_indices
                .extend([base + i as u32, base + ((i + 1) % p.len()) as u32]);
        }
    }
}

fn edge_label(mesh: &MeshIr, edge: &caso_meshing::MeshEdge) -> String {
    edge.tag_ids
        .first()
        .and_then(|id| mesh.tag_name(*id))
        .map(str::to_string)
        .or_else(|| cell_zone_label(mesh, edge.owner_cell_id))
        .unwrap_or_else(|| "mesh".to_string())
}

fn face_label(mesh: &MeshIr, face: &caso_meshing::MeshFace) -> String {
    face.tag_ids
        .first()
        .and_then(|id| mesh.tag_name(*id))
        .map(str::to_string)
        .or_else(|| cell_zone_label(mesh, face.owner_cell_id))
        .unwrap_or_else(|| "mesh".to_string())
}

fn cell_zone_label(mesh: &MeshIr, cell_id: Option<u64>) -> Option<String> {
    let cell = mesh.cells.iter().find(|cell| Some(cell.id) == cell_id)?;
    cell.zone_id
        .and_then(|zone_id| mesh.zone_name(zone_id))
        .map(str::to_string)
}

fn surface_from_buffers(
    buffers: SurfaceBuffers,
    color: [f32; 3],
    revision: u64,
    index: u32,
) -> Option<ViewportSurface> {
    if buffers.vertices.is_empty() {
        return None;
    }
    let (bounds_min, bounds_max) = f32_bounds(&buffers.vertices);
    Some(ViewportSurface {
        key: ViewportSurfaceKey {
            object_id: u32::MAX - 10 - index,
            scene_revision: revision,
            resolution: 0,
        },
        object_kind: "mesh_inspector".to_string(),
        status: SurfaceStatus::Ready,
        vertices: buffers.vertices,
        normals: buffers.normals,
        indices: buffers.indices,
        wire_indices: buffers.wire_indices,
        color,
        alpha: 1.0,
        bounds_min,
        bounds_max,
        message: String::new(),
    })
}

fn cell_corner_ids<'a>(type_name: &str, ids: &'a [u64]) -> Option<&'a [u64]> {
    let count = match type_name {
        "tri3" | "tri6" => 3,
        "quad4" | "quad8" | "quad9" => 4,
        "tet4" | "tet10" => 4,
        "hex8" | "hex20" | "hex27" => 8,
        "prism6" | "prism15" => 6,
        "pyramid5" | "pyramid13" => 5,
        "polygon" | "polyhedron" => ids.len(),
        _ => return None,
    };
    ids.get(..count)
}

fn face_corner_ids<'a>(type_name: &str, ids: &'a [u64]) -> Option<&'a [u64]> {
    let count = match type_name {
        "tri3" | "tri6" => 3,
        "quad4" | "quad8" | "quad9" => 4,
        "polygon" => ids.len(),
        _ => return None,
    };
    ids.get(..count)
}

fn edge_endpoints(type_name: &str, ids: &[u64]) -> Option<(u64, u64)> {
    match type_name {
        "edge2" if ids.len() == 2 => Some((ids[0], ids[1])),
        "edge3" if ids.len() == 3 => Some((ids[0], ids[1])),
        _ => None,
    }
}

fn mesh_dimension(type_name: &str) -> Option<u8> {
    Some(match type_name {
        "point1" => 0,
        "edge2" | "edge3" => 1,
        "tri3" | "tri6" | "quad4" | "quad8" | "quad9" | "polygon" => 2,
        "tet4" | "tet10" | "hex8" | "hex20" | "hex27" | "prism6" | "prism15" | "pyramid5"
        | "pyramid13" | "polyhedron" => 3,
        _ => return None,
    })
}

fn quality_band(score: Option<f64>) -> usize {
    score.map_or(QUALITY_BANDS, |score| {
        (score.clamp(0.0, 1.0) * (QUALITY_BANDS - 1) as f64).floor() as usize
    })
}

fn entity_has_selected_tag(tag_ids: &[u64], selected_tags: &BTreeSet<u64>) -> bool {
    tag_ids.iter().any(|id| selected_tags.contains(id))
}

fn quality_group_surfaces(groups: Vec<SurfaceBuffers>, revision: u64) -> Vec<ViewportSurface> {
    groups
        .into_iter()
        .enumerate()
        .filter_map(|(band, buffers)| {
            let color = if band == QUALITY_BANDS {
                [0.45, 0.47, 0.5]
            } else {
                quality_color((band as f64 + 0.5) / QUALITY_BANDS as f64)
            };
            surface_from_buffers(buffers, color, revision, band as u32)
        })
        .collect()
}

fn worse_band(a: usize, b: usize) -> usize {
    if a == QUALITY_BANDS || b == QUALITY_BANDS {
        QUALITY_BANDS
    } else {
        a.min(b)
    }
}

fn quality_color(score: f64) -> [f32; 3] {
    if score <= 0.5 {
        [1.0, (score * 2.0) as f32, 0.0]
    } else {
        [(2.0 - score * 2.0) as f32, 1.0, 0.0]
    }
}

fn polygon_normal(points: &[[f64; 3]]) -> [f32; 3] {
    let mut n = [0.0; 3];
    for i in 0..points.len() {
        let (a, b) = (points[i], points[(i + 1) % points.len()]);
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if length <= 1.0e-14 {
        [0.0, 0.0, 1.0]
    } else {
        [
            (n[0] / length) as f32,
            (n[1] / length) as f32,
            (n[2] / length) as f32,
        ]
    }
}

fn to_f32(point: [f64; 3]) -> [f32; 3] {
    [point[0] as f32, point[1] as f32, point[2] as f32]
}

fn format_score(score: Option<f64>) -> String {
    score.map_or_else(|| "N/A".to_string(), |score| format!("{score:.3}"))
}

fn quality_legend(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(150.0, 14.0), egui::Sense::hover());
    for band in 0..QUALITY_BANDS {
        let x0 = egui::lerp(rect.x_range(), band as f32 / QUALITY_BANDS as f32);
        let x1 = egui::lerp(rect.x_range(), (band + 1) as f32 / QUALITY_BANDS as f32);
        let color = quality_color((band as f64 + 0.5) / QUALITY_BANDS as f64);
        ui.painter().rect_filled(
            egui::Rect::from_min_max(egui::pos2(x0, rect.top()), egui::pos2(x1, rect.bottom())),
            0.0,
            egui::Color32::from_rgb((color[0] * 255.0) as u8, (color[1] * 255.0) as u8, 0),
        );
    }
}

fn f32_bounds(vertices: &[[f32; 3]]) -> ([f64; 3], [f64; 3]) {
    let mut bounds_min = [f64::INFINITY; 3];
    let mut bounds_max = [f64::NEG_INFINITY; 3];
    for vertex in vertices {
        for axis in 0..3 {
            bounds_min[axis] = bounds_min[axis].min(vertex[axis] as f64);
            bounds_max[axis] = bounds_max[axis].max(vertex[axis] as f64);
        }
    }
    if vertices.is_empty() {
        return ([0.0; 3], [0.0; 3]);
    }
    (bounds_min, bounds_max)
}

fn tag_color_id(tag: &str) -> u32 {
    let mut hash: u32 = 2_166_136_261;
    for byte in tag.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    (hash % 60_000).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panel_with_mesh(mesh: MeshIr) -> MeshingPanel {
        let mut panel = MeshingPanel {
            mesh,
            ..MeshingPanel::default()
        };
        panel.mesh_replaced();
        panel.inspector_active = true;
        panel
    }

    #[test]
    fn quality_2d_fills_cells_and_deduplicates_outlines() {
        let mut builder = caso_meshing::MeshIrBuilder::new();
        let zone = builder.zone("zone", "fluid");
        let p0 = builder.point(0.0, 0.0, 0.0).expect("point");
        let p1 = builder.point(1.0, 0.0, 0.0).expect("point");
        let p2 = builder.point(1.0, 1.0, 0.0).expect("point");
        let p3 = builder.point(0.0, 1.0, 0.0).expect("point");
        let p4 = builder.point(2.0, 0.0, 0.0).expect("point");
        let p5 = builder.point(2.0, 1.0, 0.0).expect("point");
        builder
            .cell("quad4", vec![p0, p1, p2, p3], zone)
            .expect("cell");
        builder
            .cell("quad4", vec![p1, p4, p5, p2], zone)
            .expect("cell");
        let panel = panel_with_mesh(builder.build().expect("mesh"));

        let surfaces = panel.preview_surfaces();
        assert!(surfaces.len() <= QUALITY_BANDS + 1);
        assert_eq!(surfaces.len(), 1);
        assert_eq!(surfaces[0].wire_indices.len() / 2, 7);
        assert_eq!(surfaces[0].indices.len() / 3, 4);
    }

    #[test]
    fn strict_z_filter_and_3d_skinning_remove_internal_face() {
        let mut builder = caso_meshing::MeshIrBuilder::new();
        let zone = builder.zone("zone", "fluid");
        let mut point = |x, y, z| builder.point(x, y, z).unwrap();
        let p000 = point(0., 0., 0.);
        let p100 = point(1., 0., 0.);
        let p200 = point(2., 0., 0.);
        let p010 = point(0., 1., 0.);
        let p110 = point(1., 1., 0.);
        let p210 = point(2., 1., 0.);
        let p001 = point(0., 0., 1.);
        let p101 = point(1., 0., 1.);
        let p201 = point(2., 0., 1.);
        let p011 = point(0., 1., 1.);
        let p111 = point(1., 1., 1.);
        let p211 = point(2., 1., 1.);
        builder
            .cell(
                "hex8",
                vec![p000, p100, p110, p010, p001, p101, p111, p011],
                zone,
            )
            .unwrap();
        builder
            .cell(
                "hex8",
                vec![p100, p200, p210, p110, p101, p201, p211, p111],
                zone,
            )
            .unwrap();
        let mut panel = panel_with_mesh(builder.build().unwrap());
        let surfaces = panel.preview_surfaces();
        assert_eq!(
            surfaces.iter().map(|s| s.indices.len() / 3).sum::<usize>(),
            20
        );
        assert_eq!(
            surfaces
                .iter()
                .map(|s| s.wire_indices.len() / 2)
                .sum::<usize>(),
            20
        );

        panel.z_lower = 0.1;
        assert!(
            panel.preview_surfaces().is_empty(),
            "crossing cells are excluded"
        );
        panel.z_lower = 0.0;
        assert!(
            !panel.preview_surfaces().is_empty(),
            "endpoints are inclusive"
        );
    }

    #[test]
    fn tag_multi_selection_emits_only_selected_edges() {
        let mut builder = caso_meshing::MeshIrBuilder::new();
        let zone = builder.zone("zone", "fluid");
        let inlet = builder.tag("inlet", "boundary");
        let wall = builder.tag("wall", "boundary");
        let p0 = builder.point(0., 0., 0.).unwrap();
        let p1 = builder.point(1., 0., 0.).unwrap();
        let p2 = builder.point(1., 1., 0.).unwrap();
        let p3 = builder.point(0., 1., 0.).unwrap();
        builder.cell("quad4", vec![p0, p1, p2, p3], zone).unwrap();
        builder.tag_edge(vec![p0, p1], inlet);
        builder.tag_edge(vec![p1, p2], wall);
        let mut panel = panel_with_mesh(builder.build().unwrap());
        panel.show_quality = false;
        panel.show_boundary_tags = true;
        panel.selected_tags = BTreeSet::from([inlet]);
        let surfaces = panel.preview_surfaces();
        assert_eq!(surfaces.len(), 1);
        assert_eq!(surfaces[0].wire_indices.len() / 2, 1);
        assert_eq!(surfaces[0].color, mesh_tag_color(tag_color_id("inlet")));
    }

    #[test]
    fn inspector_is_opt_in_and_quality_recolors_only_tagged_boundaries() {
        let mut builder = caso_meshing::MeshIrBuilder::new();
        let zone = builder.zone("zone", "fluid");
        let inlet = builder.tag("inlet", "boundary");
        let ids: Vec<u64> = [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]
            .into_iter()
            .map(|p| builder.point(p[0], p[1], p[2]).unwrap())
            .collect();
        builder.cell("quad4", ids.clone(), zone).unwrap();
        builder.tag_edge(ids[..2].to_vec(), inlet);
        let mesh = builder.build().unwrap();

        let mut panel = MeshingPanel {
            mesh,
            ..MeshingPanel::default()
        };
        panel.mesh_replaced();
        assert!(!panel.inspector_active);
        assert!(panel.preview_points().is_empty());
        assert!(panel
            .preview_surfaces()
            .iter()
            .all(|surface| surface.indices.is_empty()));

        panel.inspector_active = true;
        panel.show_boundary_tags = true;
        let surfaces = panel.preview_surfaces();
        assert_eq!(surfaces.len(), 1);
        assert_eq!(surfaces[0].wire_indices.len() / 2, 1);
        assert!(surfaces[0].indices.is_empty());
        assert_eq!(
            surfaces[0].color,
            quality_color((QUALITY_BANDS as f64 - 0.5) / QUALITY_BANDS as f64)
        );
    }

    #[test]
    fn quality_surface_batches_are_bounded() {
        assert_eq!(quality_band(None), QUALITY_BANDS);
        assert_eq!(quality_band(Some(0.0)), 0);
        assert_eq!(quality_band(Some(1.0)), QUALITY_BANDS - 1);
        assert_eq!(worse_band(4, 19), 4);
        assert_eq!(worse_band(4, QUALITY_BANDS), QUALITY_BANDS);
    }
}
