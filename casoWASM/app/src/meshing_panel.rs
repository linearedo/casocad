//! The Meshing workspace: Rhai mesher-script editor, run/preview, and Arrow
//! artifact export — the port of casoCAD's Meshing workspace page (with Rhai
//! replacing the Python subprocess, so it also runs in the browser).

use caso_meshing::{write_mesh_artifact, MeshElement};
use caso_surfaces::types::object_color;
use caso_surfaces::{SurfaceStatus, ViewportSurface, ViewportSurfaceKey};
use eframe::egui;

use crate::script_runner::{run_mesher_script, EXAMPLE_SCRIPT};
use crate::state::AppState;

pub struct MeshingPanel {
    script: String,
    pub elements: Vec<MeshElement>,
    pub show_preview: bool,
    /// Bumped when `elements` change (viewport rebuilds the preview overlay).
    pub preview_revision: u64,
    #[cfg(not(target_arch = "wasm32"))]
    export_path: String,
}

impl Default for MeshingPanel {
    fn default() -> Self {
        Self {
            script: EXAMPLE_SCRIPT.to_string(),
            elements: Vec::new(),
            show_preview: true,
            preview_revision: 0,
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
                    Ok(elements) => {
                        state.status = format!("Mesher script: {} element(s)", elements.len());
                        self.elements = elements;
                        self.preview_revision += 1;
                    }
                    Err(error) => state.status = format!("Mesher script error: {error}"),
                }
            }
            if ui
                .checkbox(&mut self.show_preview, "Preview")
                .on_hover_text("Show emitted elements in the viewport")
                .changed()
            {
                self.preview_revision += 1;
            }
            if !self.elements.is_empty() {
                ui.weak(format!("{} element(s)", self.elements.len()));
            }
        });
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
                    !self.elements.is_empty(),
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

    fn export(&mut self, state: &mut AppState) {
        let metadata = serde_json::json!({
            "source": "casowasm",
            "element_count": self.elements.len(),
        });
        match write_mesh_artifact(&self.elements, &metadata) {
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

    /// Browser: download the artifact bytes as mesh.arrow.
    #[cfg(target_arch = "wasm32")]
    fn deliver(&mut self, state: &mut AppState, bytes: Vec<u8>) {
        match crate::web_download_bytes("mesh.arrow", &bytes) {
            Ok(()) => state.status = "Downloaded mesh.arrow".to_string(),
            Err(error) => state.status = format!("Download failed: {error:?}"),
        }
    }

    /// The emitted point elements as flat instance data (xyz + rgb per
    /// point) for the renderer's sphere-impostor markers.
    pub fn preview_points(&self) -> Vec<f32> {
        if !self.show_preview {
            return Vec::new();
        }
        let mut points = Vec::new();
        for element in &self.elements {
            if element.vertices.len() != 1 {
                continue;
            }
            let point = element.vertices[0];
            let color = object_color(tag_color_id(&element.tag_name));
            points.extend([point[0] as f32, point[1] as f32, point[2] as f32]);
            points.extend(color);
        }
        points
    }

    /// The emitted elements as wire-only viewport preview surfaces (face
    /// outlines and segments), colored stably per tag.
    pub fn preview_surfaces(&self) -> Vec<ViewportSurface> {
        if !self.show_preview || self.elements.is_empty() {
            return Vec::new();
        }
        // Group by tag so each physics tag keeps one stable color.
        let mut tags: Vec<&str> = self
            .elements
            .iter()
            .map(|element| element.tag_name.as_str())
            .collect();
        tags.sort_unstable();
        tags.dedup();
        let mut surfaces = Vec::new();
        for (tag_index, tag) in tags.iter().enumerate() {
            let mut vertices: Vec<[f32; 3]> = Vec::new();
            let mut normals: Vec<[f32; 3]> = Vec::new();
            let mut wire_indices: Vec<u32> = Vec::new();
            let color = object_color(tag_color_id(tag));
            for element in self
                .elements
                .iter()
                .filter(|element| element.tag_name == **tag)
            {
                let base = vertices.len() as u32;
                match element.vertices.len() {
                    // Points are drawn separately as sphere impostors
                    // (`preview_points`), not as wire geometry.
                    0 | 1 => continue,
                    2 => {
                        for point in &element.vertices {
                            vertices
                                .push([point[0] as f32, point[1] as f32, point[2] as f32]);
                            normals.push([0.0, 0.0, 1.0]);
                        }
                        wire_indices.extend([base, base + 1]);
                    }
                    _ => {
                        // Face elements: wire outline only (no filled
                        // triangles by design — see design_docs/
                        // mesh_preview_opacity_independence.md).
                        for point in &element.vertices {
                            vertices
                                .push([point[0] as f32, point[1] as f32, point[2] as f32]);
                            normals.push([0.0, 0.0, 1.0]);
                        }
                        for index in 0..element.vertices.len() as u32 {
                            wire_indices.extend([
                                base + index,
                                base + (index + 1) % element.vertices.len() as u32,
                            ]);
                        }
                    }
                }
            }
            let mut bounds_min = [f64::INFINITY; 3];
            let mut bounds_max = [f64::NEG_INFINITY; 3];
            for vertex in &vertices {
                for axis in 0..3 {
                    bounds_min[axis] = bounds_min[axis].min(vertex[axis] as f64);
                    bounds_max[axis] = bounds_max[axis].max(vertex[axis] as f64);
                }
            }
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
                color,
                alpha: 1.0,
                bounds_min,
                bounds_max,
                message: String::new(),
            });
        }
        surfaces
    }
}

fn tag_color_id(tag: &str) -> u32 {
    // FNV-1a over the tag name → stable per-tag hue.
    let mut hash: u32 = 2_166_136_261;
    for byte in tag.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    (hash % 60_000).max(1)
}

