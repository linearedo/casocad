//! The Meshing workspace: Rhai MeshIR builder script editor, run/preview,
//! and Arrow artifact export.

use std::collections::BTreeMap;

use caso_kernel::meshing::meshable_domains_from_document;
use caso_kernel::scene::SceneDocument;
use caso_kernel::vec3::{vec3, Vec3};
use caso_meshing::{element_wire_edges, write_mesh_ir, MeshIr};
use caso_surfaces::types::mesh_tag_color;
use caso_surfaces::{SurfaceStatus, ViewportSurface, ViewportSurfaceKey};
use eframe::egui;

use crate::script_runner::{run_mesher_script, EXAMPLE_SCRIPT};
use crate::state::AppState;

type WireBuffers = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>);

pub struct MeshingPanel {
    script: String,
    pub mesh: MeshIr,
    pub show_preview: bool,
    /// Bumped when `mesh` changes (viewport rebuilds the preview overlay).
    pub preview_revision: u64,
    /// Per preview entity distance to the nearest Domain boundary.
    boundary_distances: BTreeMap<(String, u64), f64>,
    max_boundary_distance: f64,
    boundary_range: f64,
    #[cfg(not(target_arch = "wasm32"))]
    export_path: String,
}

impl Default for MeshingPanel {
    fn default() -> Self {
        Self {
            script: EXAMPLE_SCRIPT.to_string(),
            mesh: MeshIr::default(),
            show_preview: true,
            preview_revision: 0,
            boundary_distances: BTreeMap::new(),
            max_boundary_distance: 0.0,
            boundary_range: 0.0,
            #[cfg(not(target_arch = "wasm32"))]
            export_path: "mesh.arrow".to_string(),
        }
    }
}

impl MeshingPanel {
    pub fn ui(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
        ui.horizontal(|ui| {
            if ui.button("Run Script").clicked() {
                match run_mesher_script(&state.document, &self.script) {
                    Ok(mesh) => {
                        state.status = format!(
                            "Mesher script: {} point(s), {} cell(s)",
                            mesh.points.len(),
                            mesh.cells.len()
                        );
                        self.mesh = mesh;
                        self.update_boundary_distances(&state.document);
                        self.preview_revision += 1;
                    }
                    Err(error) => state.status = format!("Mesher script error: {error}"),
                }
            }
            if ui
                .checkbox(&mut self.show_preview, "Preview")
                .on_hover_text("Show MeshIR topology in the viewport")
                .changed()
            {
                self.preview_revision += 1;
            }
            if self.mesh.entity_count() > 0 {
                ui.weak(format!("{} MeshIR entities", self.mesh.entity_count()));
            }
        });
        if self.mesh.entity_count() > 0 && self.max_boundary_distance > 0.0 {
            let slider = ui
                .add(
                    egui::Slider::new(&mut self.boundary_range, 0.0..=self.max_boundary_distance)
                        .text("Boundary distance"),
                )
                .on_hover_text(
                    "Preview only mesh topology within this distance of a Domain boundary",
                );
            if slider.changed() {
                self.preview_revision += 1;
            }
        }
        ui.horizontal(|ui| {
            #[cfg(not(target_arch = "wasm32"))]
            {
                ui.add(
                    egui::TextEdit::singleline(&mut self.export_path)
                        .desired_width(140.0)
                        .hint_text("mesh.arrow"),
                );
            }
            let export = ui
                .add_enabled(
                    self.mesh.entity_count() > 0,
                    egui::Button::new("Mesh and Export .arrow"),
                )
                .clicked();
            if export {
                self.export(state);
            }
        });
        ui.separator();
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut self.script)
                        .code_editor()
                        .desired_width(f32::INFINITY)
                        .desired_rows(24),
                );
            });
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
            Ok(bytes) => self.deliver(state, bytes),
            Err(error) => state.status = format!("Arrow export failed: {error}"),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn deliver(&mut self, state: &mut AppState, bytes: Vec<u8>) {
        match std::fs::write(&self.export_path, bytes) {
            Ok(()) => state.status = format!("Exported {}", self.export_path),
            Err(error) => state.status = error.to_string(),
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn deliver(&mut self, state: &mut AppState, bytes: Vec<u8>) {
        match crate::web_download_bytes("mesh.arrow", &bytes) {
            Ok(()) => state.status = "Downloaded mesh.arrow".to_string(),
            Err(error) => state.status = format!("Download failed: {error:?}"),
        }
    }

    pub fn preview_points(&self) -> Vec<f32> {
        if !self.show_preview {
            return Vec::new();
        }
        let mut points = Vec::new();
        for point in &self.mesh.points {
            if !self.is_shown("point", point.id) {
                continue;
            }
            let color = mesh_tag_color(tag_color_id(
                point
                    .tag_ids
                    .first()
                    .and_then(|id| self.mesh.tag_name(*id))
                    .unwrap_or("mesh_points"),
            ));
            points.extend([
                point.position[0] as f32,
                point.position[1] as f32,
                point.position[2] as f32,
            ]);
            points.extend(color);
        }
        points
    }

    pub fn preview_surfaces(&self) -> Vec<ViewportSurface> {
        if !self.show_preview || self.mesh.entity_count() == 0 {
            return Vec::new();
        }
        let point_positions: BTreeMap<u64, [f64; 3]> = self
            .mesh
            .points
            .iter()
            .map(|point| (point.id, point.position))
            .collect();
        let mut groups: BTreeMap<String, WireBuffers> = BTreeMap::new();

        if self.mesh.faces.is_empty() {
            for edge in &self.mesh.edges {
                if self.is_shown("edge", edge.id) {
                    let label = edge_label(&self.mesh, edge);
                    append_wire(
                        groups.entry(label).or_default(),
                        &point_positions,
                        &edge.type_name,
                        &edge.point_ids,
                    );
                }
            }
        } else {
            for face in &self.mesh.faces {
                if self.is_shown("face", face.id) {
                    let label = face_label(&self.mesh, face);
                    append_wire(
                        groups.entry(label).or_default(),
                        &point_positions,
                        &face.type_name,
                        &face.point_ids,
                    );
                }
            }
        }

        let mut surfaces = Vec::new();
        for (tag_index, (label, (vertices, normals, wire_indices))) in
            groups.into_iter().enumerate()
        {
            if vertices.is_empty() {
                continue;
            }
            let (bounds_min, bounds_max) = f32_bounds(&vertices);
            surfaces.push(ViewportSurface {
                key: ViewportSurfaceKey {
                    object_id: u32::MAX - 10 - tag_index as u32,
                    scene_revision: self.preview_revision,
                    resolution: 0,
                },
                object_kind: "mesh_preview".to_string(),
                status: SurfaceStatus::Ready,
                vertices,
                normals,
                indices: Vec::new(),
                wire_indices,
                color: mesh_tag_color(tag_color_id(&label)),
                alpha: 1.0,
                bounds_min,
                bounds_max,
                message: String::new(),
            });
        }
        surfaces
    }
}

fn append_wire(
    buffers: &mut WireBuffers,
    points: &BTreeMap<u64, [f64; 3]>,
    type_name: &str,
    point_ids: &[u64],
) {
    let (vertices, normals, wire_indices) = buffers;
    for (a, b) in element_wire_edges(type_name, point_ids) {
        let (Some(a), Some(b)) = (points.get(&a), points.get(&b)) else {
            continue;
        };
        let base = vertices.len() as u32;
        vertices.push([a[0] as f32, a[1] as f32, a[2] as f32]);
        vertices.push([b[0] as f32, b[1] as f32, b[2] as f32]);
        normals.push([0.0, 0.0, 1.0]);
        normals.push([0.0, 0.0, 1.0]);
        wire_indices.extend([base, base + 1]);
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
