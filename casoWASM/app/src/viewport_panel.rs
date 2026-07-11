//! The central 3D viewport panel: renders the scene into an offscreen wgpu
//! texture (own render pass with depth) and shows it as an egui image, with
//! orbit / pan / zoom / reference-view input matching casoCAD.

use caso_kernel::scene::SceneDocument;
use caso_render::{OrbitCamera, RenderOptions, ViewportRenderer};
use caso_surfaces::ViewportSurfaceCache;
use eframe::egui;
use eframe::egui_wgpu::{wgpu, RenderState};

use crate::state::AppState;
use crate::tools::ToolState;

/// Progressive refinement ladder (coarse first paint, then quality tiers).
const REFINEMENT_TIERS: [u32; 3] = [12, 64, 96];

pub struct ViewportPanel {
    pub camera: OrbitCamera,
    pub options: RenderOptions,
    renderer: Option<ViewportRenderer>,
    texture_id: Option<egui::TextureId>,
    caches: Vec<(u32, ViewportSurfaceCache)>,
    pending_tiers: Vec<u32>,
    scene_version: u64,
    framed_once: bool,
    /// Latest built surface scene (base colors, no highlight applied).
    base_scene: Option<caso_surfaces::ViewportSurfaceScene>,
    selection: Option<u32>,
    applied_selection: Option<u32>,
    /// Boundary-tool overlay surfaces (candidate/selected/split preview).
    overlays: Vec<caso_surfaces::ViewportSurface>,
    overlay_signature: (u64, u64, Option<u32>),
    /// Meshing-workspace preview surfaces (per-tag lattice/mesh elements).
    mesh_overlays: Vec<caso_surfaces::ViewportSurface>,
    mesh_preview_revision: u64,
    upload_pending: bool,
}

impl Default for ViewportPanel {
    fn default() -> Self {
        Self {
            camera: OrbitCamera::default(),
            options: RenderOptions::default(),
            renderer: None,
            texture_id: None,
            caches: REFINEMENT_TIERS
                .iter()
                .map(|tier| (*tier, ViewportSurfaceCache::new(*tier)))
                .collect(),
            pending_tiers: REFINEMENT_TIERS.to_vec(),
            scene_version: u64::MAX,
            framed_once: false,
            base_scene: None,
            selection: None,
            applied_selection: None,
            overlays: Vec::new(),
            overlay_signature: (u64::MAX, u64::MAX, None),
            mesh_overlays: Vec::new(),
            mesh_preview_revision: u64::MAX,
            upload_pending: false,
        }
    }
}

impl ViewportPanel {
    pub fn mark_scene_changed(&mut self) {
        self.pending_tiers = REFINEMENT_TIERS.to_vec();
    }

    /// Highlighted object (single selection), applied on top of the built
    /// surfaces without rebuilding them.
    pub fn set_selection(&mut self, selection: Option<u32>) {
        self.selection = selection;
    }

    /// Reframe the camera on the scene on the next surface refresh.
    pub fn request_frame_all(&mut self) {
        self.framed_once = false;
        self.mark_scene_changed();
    }

    pub fn mesh_preview_revision(&self) -> u64 {
        self.mesh_preview_revision
    }

    /// Replace the meshing-workspace preview overlay.
    pub fn set_mesh_preview(
        &mut self,
        revision: u64,
        surfaces: Vec<caso_surfaces::ViewportSurface>,
    ) {
        self.mesh_preview_revision = revision;
        self.mesh_overlays = surfaces;
        self.upload_pending = true;
    }

    /// Working-unit switch: rescale the camera and snap the grid to one unit;
    /// committed geometry is never rescaled (model stays in meters).
    pub fn set_working_unit(&mut self, factor: f64) {
        self.camera.view_scale = factor;
        self.options.grid_spacing = factor as f32;
        self.request_frame_all();
    }

    fn upload_scene(&mut self, render_state: &RenderState) {
        let Some(base) = &self.base_scene else {
            return;
        };
        let renderer = self
            .renderer
            .get_or_insert_with(|| ViewportRenderer::new(&render_state.device));
        let mut scene = match self.selection {
            Some(object_id) => base.with_selected_highlight(object_id),
            None => base.clone(),
        };
        scene.surfaces.extend(self.overlays.iter().cloned());
        scene.surfaces.extend(self.mesh_overlays.iter().cloned());
        renderer.set_scene(&render_state.device, &render_state.queue, &scene);
        self.applied_selection = self.selection;
        self.upload_pending = false;
    }

    /// Rebuild the boundary highlight overlays (yellow hover candidate,
    /// cyan selected region, cyan/orange split preview) when their inputs
    /// change; also hands the cutter its dense validation points.
    fn refresh_boundary_overlays(
        &mut self,
        state: &AppState,
        tools: &mut ToolState,
        render_state: &RenderState,
    ) {
        let signature = (
            self.scene_version,
            tools.overlay_revision,
            state.selected_region,
        );
        if signature == self.overlay_signature {
            return;
        }
        self.overlay_signature = signature;
        self.overlays.clear();
        let Some(base) = self.base_scene.clone() else {
            self.upload_scene(render_state);
            return;
        };
        let Some(root) = crate::boundary_tool::fluid_root_node(&state.document) else {
            tools.validation_points.clear();
            self.upload_scene(render_state);
            return;
        };
        tools.validation_points = crate::boundary_tool::validation_points(&base);

        let selected = state.selected_region.and_then(|region_id| {
            state
                .document
                .boundary_regions
                .iter()
                .find(|region| region.object_id == region_id)
        });
        if let (Some(region), Some(ghost)) = (selected, &tools.preview_ghost) {
            // Split preview supersedes the plain selection highlight.
            let (inside, outside) =
                crate::boundary_tool::split_preview_children(region, ghost);
            if let Some(surface) = crate::boundary_tool::region_highlight_surface(
                &root,
                &inside,
                &base,
                crate::boundary_tool::PREVIEW_INSIDE_COLOR,
                u32::MAX - 4,
            ) {
                self.overlays.push(surface);
            }
            if let Some(surface) = crate::boundary_tool::region_highlight_surface(
                &root,
                &outside,
                &base,
                crate::boundary_tool::PREVIEW_OUTSIDE_COLOR,
                u32::MAX - 5,
            ) {
                self.overlays.push(surface);
            }
        } else if let Some(region) = selected {
            if let Some(surface) = crate::boundary_tool::region_highlight_surface(
                &root,
                region,
                &base,
                crate::boundary_tool::SELECTED_COLOR,
                u32::MAX - 2,
            ) {
                self.overlays.push(surface);
            }
        }
        if let Some(hit) = &tools.hover_hit {
            let candidate = crate::boundary_tool::candidate_region(hit);
            if let Some(surface) = crate::boundary_tool::region_highlight_surface(
                &root,
                &candidate,
                &base,
                crate::boundary_tool::CANDIDATE_COLOR,
                u32::MAX - 3,
            ) {
                self.overlays.push(surface);
            }
        }
        self.upload_scene(render_state);
    }

    /// Build the next pending refinement tier and upload it.
    fn refresh_surfaces(&mut self, document: &SceneDocument, render_state: &RenderState) {
        if document.version != self.scene_version {
            self.scene_version = document.version;
            self.pending_tiers = REFINEMENT_TIERS.to_vec();
        }
        let Some(tier) = self.pending_tiers.first().copied() else {
            return;
        };
        self.pending_tiers.remove(0);
        // Visible top-level components (never internal selector nodes).
        let mut components = Vec::new();
        for root in &document.roots {
            if let Ok(object) = document.object(*root) {
                if SceneDocument::is_internal_scene_node(&object.name) {
                    continue;
                }
                if let Ok(node) = document.build_node(*root) {
                    components.push(node);
                }
            }
        }
        let cache = self
            .caches
            .iter_mut()
            .find(|(cache_tier, _)| *cache_tier == tier)
            .map(|(_, cache)| cache)
            .expect("tier cache");
        let scene =
            caso_surfaces::build_viewport_surface_scene(&components, document.version, cache);
        self.base_scene = Some(scene);
        self.upload_scene(render_state);
        // Force the boundary overlays to re-filter against the new surfaces.
        self.overlay_signature = (u64::MAX, u64::MAX, None);

        if !self.framed_once {
            self.framed_once = true;
            if let Some(first) = components.first() {
                if let Ok(bounds) = first.bounding_box() {
                    self.camera.frame_box(&bounds);
                }
            }
        }
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        state: &mut AppState,
        tools: &mut ToolState,
        render_state: &RenderState,
    ) {
        self.refresh_surfaces(&state.document, render_state);
        if self.selection != self.applied_selection || self.upload_pending {
            self.upload_scene(render_state);
        }
        let available = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());
        let pixels_per_point = ui.ctx().pixels_per_point();
        let width = (rect.width() * pixels_per_point).round().max(1.0) as u32;
        let height = (rect.height() * pixels_per_point).round().max(1.0) as u32;

        // Active tools consume the primary button; camera keeps the rest
        // (the Boundary Region hover tool leaves navigation available).
        let tool_consumed = tools.blocks_camera();

        // Input: left-drag orbit, right/middle-drag pan, wheel zoom.
        let drag = response.drag_delta();
        if !tool_consumed && response.dragged_by(egui::PointerButton::Primary) {
            self.camera
                .orbit(drag.x as f64 * pixels_per_point as f64, drag.y as f64 * pixels_per_point as f64);
        } else if response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            self.camera.pan(
                drag.x as f64 * pixels_per_point as f64,
                drag.y as f64 * pixels_per_point as f64,
                height as f64,
            );
        }
        if response.hovered() {
            let scroll = ui.ctx().input(|input| input.smooth_scroll_delta().y);
            if scroll.abs() > 0.0 {
                self.camera.zoom_by(scroll as f64);
            }
        }
        // View keys stay off while a tool consumes typed input (digits are
        // dimensions during a create drag, not view shortcuts).
        if response.hovered() && !tool_consumed {
            ui.ctx().input(|input| {
                for (key, yaw, pitch) in [
                    (egui::Key::Num1, 35.0_f64, 28.0_f64),
                    (egui::Key::Num2, 90.0, 89.5),
                    (egui::Key::Num3, -90.0, 0.0),
                    (egui::Key::Num4, 0.0, 0.0),
                ] {
                    if input.key_pressed(key) {
                        self.camera.yaw = yaw.to_radians();
                        self.camera.pitch = pitch.to_radians();
                    }
                }
                if input.key_pressed(egui::Key::Home) {
                    self.framed_once = false;
                    self.mark_scene_changed();
                }
                // WASD fly: move the orbit target in the view plane.
                let basis = self.camera.basis();
                let step = self.camera.fly_step();
                for (key, direction) in [
                    (egui::Key::W, basis.forward),
                    (egui::Key::S, basis.forward * -1.0),
                    (egui::Key::D, basis.right),
                    (egui::Key::A, basis.right * -1.0),
                ] {
                    if input.key_down(key) {
                        self.camera.target += direction * step;
                        ui.ctx().request_repaint();
                    }
                }
            });
        }

        let renderer = self
            .renderer
            .get_or_insert_with(|| ViewportRenderer::new(&render_state.device));
        let resized = renderer.size() != (width, height);
        let view = renderer.resize(&render_state.device, width, height);
        renderer.render(
            &render_state.device,
            &render_state.queue,
            &self.camera,
            &self.options,
            &[],
        );
        if resized || self.texture_id.is_none() {
            let mut egui_renderer = render_state.renderer.write();
            if let Some(old) = self.texture_id.take() {
                egui_renderer.free_texture(&old);
            }
            self.texture_id = Some(egui_renderer.register_native_texture(
                &render_state.device,
                &view,
                wgpu::FilterMode::Linear,
            ));
        }
        if let Some(texture_id) = self.texture_id {
            ui.painter().image(
                texture_id,
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
        // Tool input + ghost overlays go on top of the rendered image.
        tools.handle_viewport(&response, ui, &self.camera, rect, pixels_per_point, state);
        self.refresh_boundary_overlays(state, tools, render_state);
        // Keep painting while refinement tiers are pending.
        if !self.pending_tiers.is_empty() {
            ui.ctx().request_repaint();
        }
    }
}
