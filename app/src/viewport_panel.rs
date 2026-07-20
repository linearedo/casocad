//! The central 3D viewport panel: renders the scene into an offscreen wgpu
//! texture (own render pass with depth) and shows it as an egui image, with
//! orbit / pan / zoom / reference-view input matching casoCAD.

use caso_kernel::scene::{SceneDocument, ScenePayload};
use caso_kernel::sdf::node::Shape;
use caso_kernel::vec3::{vec3, Vec3};
use caso_render::{OrbitCamera, RenderOptions, ViewportRenderer};
use caso_surfaces::ViewportSurfaceCache;
use eframe::egui;
use eframe::egui_wgpu::{wgpu, RenderState};

use crate::state::AppState;
use crate::tools::{ToolKind, ToolState};

/// Wire objects (1D) are picked in screen space within this radius (points).
const WIRE_PICK_RADIUS: f32 = 8.0;

/// Tint of the create-tool geometry ghost (the accent blue 74,168,255).
const CREATE_GHOST_COLOR: [f32; 3] = [0.29, 0.66, 1.0];

/// Measure-annotation amber, distinct from the accent blue and axis colors.
const MEASURE_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 200, 80);

/// Progressive refinement ladder (coarse first paint, then quality tiers).
const REFINEMENT_TIERS: [u32; 3] = [12, 64, 96];

/// Reference views, matching the Python view panel: label, shortcut key,
/// grid plane (0=XY, 1=XZ, 2=YZ) and camera yaw/pitch in degrees.
const REFERENCE_VIEWS: [(&str, egui::Key, u32, f64, f64); 4] = [
    ("3D", egui::Key::Num1, 0, 35.0, 28.0),
    ("{x,y}", egui::Key::Num2, 0, -90.0, 89.5),
    ("{x,z}", egui::Key::Num3, 1, -90.0, 0.0),
    ("{y,z}", egui::Key::Num4, 2, 0.0, 0.0),
];

/// Duration of the camera flight to a reference view, in seconds.
const VIEW_FLIGHT_SECONDS: f64 = 0.26;

/// Axis triad overlay (bottom-left): world axes with their colors.
const TRIAD_AXES: [(&str, Vec3, egui::Color32); 3] = [
    ("X", vec3(1.0, 0.0, 0.0), egui::Color32::from_rgb(255, 86, 65)),
    ("Y", vec3(0.0, 1.0, 0.0), egui::Color32::from_rgb(85, 235, 105)),
    ("Z", vec3(0.0, 0.0, 1.0), egui::Color32::from_rgb(92, 145, 255)),
];

/// In-flight smoothstep camera flight to a reference view.
struct ViewFlight {
    start_time: f64,
    start_yaw: f64,
    start_pitch: f64,
    delta_yaw: f64,
    delta_pitch: f64,
}

pub struct ViewportPanel {
    pub camera: OrbitCamera,
    pub options: RenderOptions,
    renderer: Option<ViewportRenderer>,
    texture_id: Option<egui::TextureId>,
    caches: Vec<(u32, ViewportSurfaceCache)>,
    pending_tiers: Vec<u32>,
    scene_version: u64,
    /// egui clock time of the last edit-triggered rebuild, and its cost:
    /// together they throttle rebuilds during continuous drags so an
    /// expensive scene cannot pin every frame on re-contouring.
    last_rebuild_time: f64,
    last_build_ms: f64,
    framed_once: bool,
    /// Latest built surface scene (base colors, no highlight applied).
    base_scene: Option<caso_surfaces::ViewportSurfaceScene>,
    selection: Option<u32>,
    applied_selection: Option<u32>,
    /// Boundary-tool overlay surfaces (candidate/selected/split preview).
    overlays: Vec<caso_surfaces::ViewportSurface>,
    /// (scene version, overlay revision, selected region, zoom bucket):
    /// the zoom bucket quantizes world-per-pixel in ~10% steps so the
    /// pixel-sized highlight ribbons rebuild on zoom, not every frame.
    overlay_signature: (u64, u64, Option<u32>, i64),
    /// Meshing-workspace preview surfaces (per-tag lattice/mesh elements).
    mesh_overlays: Vec<caso_surfaces::ViewportSurface>,
    /// Meshing-workspace point elements (xyz + rgb per point), drawn as
    /// sphere-impostor markers.
    mesh_points: Vec<f32>,
    mesh_preview_revision: u64,
    upload_pending: bool,
    view_flight: Option<ViewFlight>,
    /// Draw the bounding-extent tripod on each selected object.
    pub show_bounds: bool,
    /// Committed measure annotations (world points, meters). Session-only —
    /// never part of the scene document. The pending first point of a pair
    /// lives in `ToolState::points` like every other point tool.
    measurements: Vec<(Vec3, Vec3)>,
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
            last_rebuild_time: f64::NEG_INFINITY,
            last_build_ms: 0.0,
            framed_once: false,
            base_scene: None,
            selection: None,
            applied_selection: None,
            overlays: Vec::new(),
            overlay_signature: (u64::MAX, u64::MAX, None, i64::MAX),
            mesh_overlays: Vec::new(),
            mesh_points: Vec::new(),
            mesh_preview_revision: u64::MAX,
            upload_pending: false,
            view_flight: None,
            show_bounds: false,
            measurements: Vec::new(),
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
        points: Vec<f32>,
    ) {
        self.mesh_preview_revision = revision;
        self.mesh_overlays = surfaces;
        self.mesh_points = points;
        self.upload_pending = true;
    }

    /// Working-unit switch: rescale the camera and snap the grid to one unit;
    /// committed geometry is never rescaled (model stays in meters).
    pub fn set_working_unit(&mut self, factor: f64) {
        self.camera.view_scale = factor;
        self.options.grid_spacing = factor as f32;
        self.request_frame_all();
    }

    /// Switch the visualization grid plane and fly the camera to the view
    /// (matching the Python `set_reference_view`). Drawing tools stay on XY.
    fn fly_to_reference_view(&mut self, plane: u32, yaw_deg: f64, pitch_deg: f64, now: f64) {
        self.options.grid_plane = plane;
        let tau = std::f64::consts::TAU;
        let delta_yaw = (yaw_deg.to_radians() - self.camera.yaw + tau / 2.0).rem_euclid(tau)
            - tau / 2.0;
        let delta_pitch = pitch_deg.to_radians() - self.camera.pitch;
        if delta_yaw.abs().max(delta_pitch.abs()) <= 1.0e-6 {
            return;
        }
        self.view_flight = Some(ViewFlight {
            start_time: now,
            start_yaw: self.camera.yaw,
            start_pitch: self.camera.pitch,
            delta_yaw,
            delta_pitch,
        });
    }

    /// Advance the in-flight reference-view animation (smoothstep easing).
    fn advance_view_flight(&mut self, ctx: &egui::Context, now: f64) {
        let Some(flight) = &self.view_flight else {
            return;
        };
        let t = ((now - flight.start_time) / VIEW_FLIGHT_SECONDS).clamp(0.0, 1.0);
        let eased = t * t * (3.0 - 2.0 * t);
        self.camera.yaw = flight.start_yaw + flight.delta_yaw * eased;
        self.camera.pitch = flight.start_pitch + flight.delta_pitch * eased;
        if t >= 1.0 {
            self.view_flight = None;
        } else {
            ctx.request_repaint();
        }
    }

    /// Bottom-left axis triad: world X/Y/Z projected with the camera basis,
    /// back-to-front with depth-dimmed alpha (ported from `_OrientationWidget`).
    fn paint_orientation_triad(&self, ui: &egui::Ui, rect: egui::Rect) {
        let painter = ui.painter_at(rect);
        let size = 96.0;
        let widget_rect = egui::Rect::from_min_size(
            egui::pos2(rect.min.x + 10.0, rect.max.y - 10.0 - size),
            egui::vec2(size, size),
        );
        painter.rect_filled(
            widget_rect,
            egui::CornerRadius::same(6),
            egui::Color32::from_rgba_unmultiplied(6, 10, 16, 145),
        );
        let basis = self.camera.basis();
        let origin = widget_rect.min + egui::vec2(38.0, 58.0);
        let mut axes: Vec<(&str, f32, f32, f64, egui::Color32)> = TRIAD_AXES
            .into_iter()
            .map(|(label, axis, color)| {
                (
                    label,
                    axis.dot(basis.right) as f32,
                    -(axis.dot(basis.up)) as f32,
                    axis.dot(basis.forward),
                    color,
                )
            })
            .collect();
        axes.sort_by(|a, b| a.3.total_cmp(&b.3));
        for (label, dx, dy, depth, color) in axes {
            let alpha = if depth > 0.0 { 115 } else { 235 };
            let color =
                egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha);
            let end = origin + egui::vec2(dx * 30.0, dy * 30.0);
            painter.line_segment([origin, end], egui::Stroke::new(4.0, color));
            painter.circle_filled(end, 4.5, color);
            painter.text(
                end + egui::vec2(6.0, -5.0),
                egui::Align2::LEFT_BOTTOM,
                label,
                egui::FontId::proportional(12.0),
                color,
            );
        }
    }

    /// Bounding-extent tripod for each selected object: one edge per axis of
    /// the world AABB, anchored at the minimum corner, labeled with its
    /// extent in the working unit. Informational overlay only — bounding
    /// boxes are never treated as geometry.
    fn paint_bounds_tripods(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        pixels_per_point: f32,
        state: &AppState,
    ) {
        let painter = ui.painter_at(rect);
        for id in &state.selection {
            let Ok(node) = state.document.build_node(*id) else {
                continue;
            };
            let Ok(bounds) = node.bounding_box() else {
                continue;
            };
            let corner = vec3(bounds.x_min, bounds.y_min, bounds.z_min);
            let Some(anchor) =
                crate::tools::project_to_screen(&self.camera, corner, rect, pixels_per_point)
            else {
                continue;
            };
            let edges = [
                (vec3(bounds.x_max, bounds.y_min, bounds.z_min), bounds.x_max - bounds.x_min),
                (vec3(bounds.x_min, bounds.y_max, bounds.z_min), bounds.y_max - bounds.y_min),
                (vec3(bounds.x_min, bounds.y_min, bounds.z_max), bounds.z_max - bounds.z_min),
            ];
            for (index, ((_, _, color), (end_world, extent))) in
                TRIAD_AXES.into_iter().zip(edges).enumerate()
            {
                let text = crate::dimensions::format_length(extent, &state.unit);
                // Degenerate axes (2D/1D objects) get the label only,
                // stacked at the anchor so they never overlap each other.
                if extent <= 1e-9 {
                    let pos = anchor + egui::vec2(8.0, -6.0 - 14.0 * index as f32);
                    label_with_backdrop(&painter, pos, &text, color);
                    continue;
                }
                let Some(end) = crate::tools::project_to_screen(
                    &self.camera,
                    end_world,
                    rect,
                    pixels_per_point,
                ) else {
                    continue;
                };
                painter.line_segment([anchor, end], egui::Stroke::new(2.0, color));
                painter.circle_filled(end, 3.0, color);
                let mid = anchor + (end - anchor) * 0.5;
                label_with_backdrop(&painter, mid + egui::vec2(6.0, -6.0), &text, color);
            }
        }
    }

    /// Dashed revolve-axis glyph for each selected revolved solid — the same
    /// rendering the pick tool previews, so the axis stays inspectable after
    /// commit (`Revolve::axis_frame` is the single source of truth).
    fn paint_revolve_axes(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        pixels_per_point: f32,
        state: &AppState,
    ) {
        let painter = ui.painter_at(rect);
        for id in &state.selection {
            let Ok(object) = state.document.object(*id) else {
                continue;
            };
            if !matches!(object.payload, ScenePayload::Revolve { .. }) {
                continue;
            }
            let Ok(node) = state.document.build_node(*id) else {
                continue;
            };
            let Shape::Revolve(revolve) = &node.shape else {
                continue;
            };
            let Ok(frame) = revolve.axis_frame() else {
                continue;
            };
            let section = revolve.section2d();
            let half_length = crate::tools::axis_half_length(
                section.origin,
                section.axis_u,
                section.axis_v,
                section.profile.bounds(),
                frame.origin,
            );
            crate::tools::paint_revolve_axis(
                &painter,
                &self.camera,
                rect,
                pixels_per_point,
                frame.origin,
                frame.axis,
                frame.radial,
                half_length,
                crate::tools::REVOLVE_AXIS_COLOR,
            );
        }
    }

    /// Bottom-center reference-view buttons: 3D / {x,y} / {x,z} / {y,z}.
    fn view_panel_ui(&mut self, ctx: &egui::Context, rect: egui::Rect, now: f64) {
        egui::Area::new(egui::Id::new("viewport_view_panel"))
            .order(egui::Order::Foreground)
            .pivot(egui::Align2::CENTER_BOTTOM)
            .fixed_pos(egui::pos2(rect.center().x, rect.max.y - 10.0))
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(egui::Color32::from_rgba_unmultiplied(6, 10, 16, 170))
                    .stroke(egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgba_unmultiplied(120, 210, 255, 110),
                    ))
                    .corner_radius(egui::CornerRadius::same(4))
                    .inner_margin(egui::Margin::same(4))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            for (index, (label, _, plane, yaw, pitch)) in
                                REFERENCE_VIEWS.iter().enumerate()
                            {
                                let button = ui
                                    .button(*label)
                                    .on_hover_text(format!("{label} view (key {})", index + 1));
                                if button.clicked() {
                                    self.fly_to_reference_view(*plane, *yaw, *pitch, now);
                                }
                            }
                        });
                    });
            });
    }

    /// Upload the base mesh (+ selection highlight) to its own GPU chunks.
    /// Call only when the document actually rebuilds or selection changes —
    /// never from zoom or overlay-only updates, see `upload_overlays` and
    /// `upload_mesh_overlays`.
    fn upload_base_scene(&mut self, render_state: &RenderState) {
        let Some(base) = &self.base_scene else {
            return;
        };
        let renderer = self
            .renderer
            .get_or_insert_with(|| ViewportRenderer::new(&render_state.device));
        let scene = match self.selection {
            Some(object_id) => base.with_selected_highlight(object_id),
            None => base.clone(),
        };
        renderer.set_scene(&render_state.device, &render_state.queue, &scene);
        self.applied_selection = self.selection;
    }

    /// Upload boundary-tool overlays (highlight ribbons, create-ghost) to
    /// their own GPU chunks, independent of the base mesh AND of the mesh
    /// preview. This is the path `refresh_boundary_overlays`'s zoom bucket
    /// drives, so it must exclude `mesh_overlays` — the meshing-workspace
    /// preview can be arbitrarily large (every mesh element), and re-zooming
    /// must never re-upload it.
    fn upload_overlays(&mut self, render_state: &RenderState) {
        let renderer = self
            .renderer
            .get_or_insert_with(|| ViewportRenderer::new(&render_state.device));
        renderer.set_overlays(&render_state.device, &render_state.queue, &self.overlays);
    }

    /// Upload the meshing-workspace preview (surfaces + points) to its own
    /// GPU chunks. Driven solely by `mesh_preview_revision` changes
    /// (`upload_pending`), never by zoom or boundary-overlay churn.
    fn upload_mesh_overlays(&mut self, render_state: &RenderState) {
        let renderer = self
            .renderer
            .get_or_insert_with(|| ViewportRenderer::new(&render_state.device));
        renderer.set_mesh_overlays(&render_state.device, &render_state.queue, &self.mesh_overlays);
        renderer.set_points(&render_state.device, &render_state.queue, &self.mesh_points);
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
        rect: egui::Rect,
        pixels_per_point: f32,
    ) {
        // Camera-only zoom proxy (same formula as OrbitCamera::pan),
        // quantized in ~10% steps: the pixel-sized ribbons follow zoom and
        // window resizes without rebuilding on every orbit frame. The exact
        // per-workplane pixel size is measured below at build time.
        let physical_height = (rect.height() * pixels_per_point).max(1.0) as f64;
        let world_per_pixel =
            2.0 * self.camera.distance / (self.camera.focal * physical_height);
        let zoom_bucket = (world_per_pixel.max(1.0e-12).ln() * 10.0).round() as i64;
        let signature = (
            self.scene_version,
            tools.overlay_revision,
            state.selected_region,
            zoom_bucket,
        );
        if signature == self.overlay_signature {
            return;
        }
        self.overlay_signature = signature;
        self.overlays.clear();
        // Create-tool ghost: tessellate the live-preview node through the
        // regular surface router (3D fill / 2D outline / 1D wires) — works
        // with or without a fluid domain. Only the translucency is
        // dimension-gated; construction is kind-agnostic.
        if let Some(ghost) = &tools.create_ghost {
            let key = caso_surfaces::ViewportSurfaceKey {
                object_id: u32::MAX - 6,
                scene_revision: tools.overlay_revision,
                resolution: caso_surfaces::types::DEFAULT_RESOLUTION,
            };
            let mut surface = caso_surfaces::build_viewport_surface(ghost, key);
            surface.color = CREATE_GHOST_COLOR;
            if ghost.dimension() == 3 {
                surface.alpha = 0.35;
            }
            if surface.has_geometry() {
                self.overlays.push(surface);
            }
        }
        let Some(base) = self.base_scene.clone() else {
            self.upload_overlays(render_state);
            return;
        };
        // Every marked domain (fluid, solid, …) exposes its boundary to the
        // region tools; each overlay classifies against ITS domain's root.
        let roots = crate::boundary_tool::domain_root_nodes(&state.document);
        if roots.is_empty() {
            tools.validation_points.clear();
            self.upload_overlays(render_state);
            return;
        }
        let selected = state.selected_region.and_then(|region_id| {
            state
                .document
                .boundary_regions
                .iter()
                .find(|region| region.object_id == region_id)
        });
        let selected_root = selected
            .and_then(|region| crate::boundary_tool::region_root_node(&state.document, region));
        // Validation points serve the cutter, which operates on the selected
        // region's domain; default to the first marked domain.
        let validation_root = selected_root.as_ref().unwrap_or(&roots[0].1);
        tools.validation_points =
            crate::boundary_tool::validation_points_for(validation_root, &base);

        // World length of one screen pixel on each root's workplane: makes
        // the 2D highlight ribbon a constant apparent width at any zoom.
        let camera = self.camera;
        let pixel_size_for = |root: &caso_kernel::sdf::node::Node| {
            crate::tools::workplane_pixel_radius(
                &camera,
                rect.center(),
                rect,
                pixels_per_point,
                root,
                1.0,
            )
        };
        if let (Some(region), Some(root), Some(ghost)) =
            (selected, selected_root.as_ref(), &tools.preview_ghost)
        {
            // Split preview supersedes the plain selection highlight.
            let (inside, outside) =
                crate::boundary_tool::split_preview_children(region, ghost);
            let pixel_size = pixel_size_for(root);
            if let Some(surface) = crate::boundary_tool::region_highlight_surface(
                root,
                &inside,
                &base,
                crate::boundary_tool::PREVIEW_INSIDE_COLOR,
                u32::MAX - 4,
                pixel_size,
            ) {
                self.overlays.push(surface);
            }
            if let Some(surface) = crate::boundary_tool::region_highlight_surface(
                root,
                &outside,
                &base,
                crate::boundary_tool::PREVIEW_OUTSIDE_COLOR,
                u32::MAX - 5,
                pixel_size,
            ) {
                self.overlays.push(surface);
            }
        } else if let (Some(region), Some(root)) = (selected, selected_root.as_ref()) {
            if let Some(surface) = crate::boundary_tool::region_highlight_surface(
                root,
                region,
                &base,
                crate::boundary_tool::SELECTED_COLOR,
                u32::MAX - 2,
                pixel_size_for(root),
            ) {
                self.overlays.push(surface);
            }
        }
        if let Some(hit) = &tools.hover_hit {
            let hover_root = tools
                .hover_domain
                .and_then(|id| roots.iter().find(|(root_id, _)| *root_id == id))
                .map(|(_, node)| node)
                .unwrap_or(&roots[0].1);
            let candidate = crate::boundary_tool::candidate_region(hit);
            if let Some(surface) = crate::boundary_tool::region_highlight_surface(
                hover_root,
                &candidate,
                &base,
                crate::boundary_tool::CANDIDATE_COLOR,
                u32::MAX - 3,
                pixel_size_for(hover_root),
            ) {
                self.overlays.push(surface);
            }
        }
        self.upload_overlays(render_state);
    }

    /// The scene object under the screen position: nearest ray/triangle hit
    /// on the built display surfaces, else the nearest wire segment (1D
    /// objects) within a small screen-space radius.
    fn pick_object(
        &self,
        pos: egui::Pos2,
        rect: egui::Rect,
        pixels_per_point: f32,
    ) -> Option<u32> {
        let base = self.base_scene.as_ref()?;
        let (origin, direction) = crate::tools::screen_ray(&self.camera, pos, rect, pixels_per_point);
        let mut nearest: Option<(f64, u32)> = None;
        for surface in &base.surfaces {
            for triangle in surface.indices.chunks_exact(3) {
                let a = vertex_point(surface, triangle[0]);
                let b = vertex_point(surface, triangle[1]);
                let c = vertex_point(surface, triangle[2]);
                if let Some(t) = ray_triangle_hit(origin, direction, a, b, c) {
                    if nearest.is_none_or(|(best, _)| t < best) {
                        nearest = Some((t, surface.key.object_id));
                    }
                }
            }
        }
        if let Some((_, object_id)) = nearest {
            return Some(object_id);
        }
        // Wire fallback: 1D objects have no triangles, only line segments.
        let mut nearest_wire: Option<(f32, u32)> = None;
        for surface in &base.surfaces {
            if !surface.indices.is_empty() {
                continue;
            }
            for segment in surface.wire_indices.chunks_exact(2) {
                let a = crate::tools::project_to_screen(
                    &self.camera,
                    vertex_point(surface, segment[0]),
                    rect,
                    pixels_per_point,
                );
                let b = crate::tools::project_to_screen(
                    &self.camera,
                    vertex_point(surface, segment[1]),
                    rect,
                    pixels_per_point,
                );
                let (Some(a), Some(b)) = (a, b) else {
                    continue;
                };
                let distance = crate::gizmo::point_segment_distance(pos, a, b);
                if distance <= WIRE_PICK_RADIUS
                    && nearest_wire.is_none_or(|(best, _)| distance < best)
                {
                    nearest_wire = Some((distance, surface.key.object_id));
                }
            }
        }
        nearest_wire.map(|(_, object_id)| object_id)
    }

    /// Select-tool click: pick under the cursor (ctrl toggles, empty clears).
    fn handle_select_click(
        &mut self,
        response: &egui::Response,
        ui: &egui::Ui,
        rect: egui::Rect,
        pixels_per_point: f32,
        state: &mut AppState,
    ) {
        if !response.clicked() {
            return;
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        let toggle = ui.ctx().input(|input| input.modifiers.command);
        match self.pick_object(pos, rect, pixels_per_point) {
            Some(object_id) => {
                if toggle {
                    state.toggle_select(object_id);
                } else {
                    state.select_only(object_id);
                }
                if let Ok(object) = state.document.object(object_id) {
                    state.status = format!("Selected {}", object.name);
                }
            }
            None => {
                if !toggle && !state.selection.is_empty() {
                    state.selection.clear();
                    state.status = "Selection cleared".to_string();
                }
            }
        }
    }

    /// Where a measure click lands: the nearest display-surface hit under
    /// the cursor, falling back to the Z=0 grid plane.
    fn measure_point(
        &self,
        pos: egui::Pos2,
        rect: egui::Rect,
        pixels_per_point: f32,
    ) -> Option<Vec3> {
        if let Some(base) = self.base_scene.as_ref() {
            let (origin, direction) =
                crate::tools::screen_ray(&self.camera, pos, rect, pixels_per_point);
            let mut nearest: Option<f64> = None;
            for surface in &base.surfaces {
                for triangle in surface.indices.chunks_exact(3) {
                    let a = vertex_point(surface, triangle[0]);
                    let b = vertex_point(surface, triangle[1]);
                    let c = vertex_point(surface, triangle[2]);
                    if let Some(t) = ray_triangle_hit(origin, direction, a, b, c) {
                        if nearest.is_none_or(|best| t < best) {
                            nearest = Some(t);
                        }
                    }
                }
            }
            if let Some(t) = nearest {
                return Some(origin + direction * t);
            }
        }
        crate::tools::grid_point(&self.camera, pos, rect, pixels_per_point)
    }

    /// Measure-tool click: first click arms the pending point (held in
    /// `tools.points` like every point tool), the second commits an
    /// annotation pair.
    fn handle_measure_click(
        &mut self,
        response: &egui::Response,
        rect: egui::Rect,
        pixels_per_point: f32,
        state: &mut AppState,
        tools: &mut ToolState,
    ) {
        if !response.clicked() {
            return;
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        let Some(point) = self.measure_point(pos, rect, pixels_per_point) else {
            return;
        };
        match tools.points.pop() {
            None => {
                tools.points.push(point);
                state.status =
                    "Measure: click the second point — Backspace/Esc removes it".to_string();
            }
            Some(start) => {
                let distance = (point - start).length();
                self.measurements.push((start, point));
                state.status = format!(
                    "Measured {}",
                    crate::dimensions::format_length(distance, &state.unit)
                );
            }
        }
    }

    /// Draw every committed measurement (they persist across tool switches)
    /// plus, while the Measure tool is active, the pending point marker and
    /// its live rubber band to the snapped cursor point.
    fn paint_measurements(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        pixels_per_point: f32,
        state: &AppState,
        pending: Option<Vec3>,
    ) {
        if self.measurements.is_empty() && pending.is_none() {
            return;
        }
        let painter = ui.painter_at(rect);
        for (start, end) in &self.measurements {
            self.paint_dimension_line(&painter, rect, pixels_per_point, state, *start, *end);
        }
        let Some(pending) = pending else {
            return;
        };
        if let Some(anchor) =
            crate::tools::project_to_screen(&self.camera, pending, rect, pixels_per_point)
        {
            painter.circle_filled(anchor, 3.5, MEASURE_COLOR);
        }
        let hover = ui
            .ctx()
            .input(|input| input.pointer.latest_pos())
            .filter(|pos| rect.contains(*pos))
            .and_then(|pos| self.measure_point(pos, rect, pixels_per_point));
        // A zero-length rubber band (cursor still on the first click) would
        // only paint a degenerate "0" label on top of the marker.
        if let Some(current) = hover.filter(|point| (*point - pending).length() > 1e-12) {
            self.paint_dimension_line(&painter, rect, pixels_per_point, state, pending, current);
        }
    }

    /// One arrowed dimension line: segment, inward arrowheads at both ends,
    /// distance label at the midpoint.
    fn paint_dimension_line(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        pixels_per_point: f32,
        state: &AppState,
        start: Vec3,
        end: Vec3,
    ) {
        let a = crate::tools::project_to_screen(&self.camera, start, rect, pixels_per_point);
        let b = crate::tools::project_to_screen(&self.camera, end, rect, pixels_per_point);
        let (Some(a), Some(b)) = (a, b) else {
            return;
        };
        let stroke = egui::Stroke::new(1.5, MEASURE_COLOR);
        painter.line_segment([a, b], stroke);
        let along = b - a;
        if along.length() > 1e-3 {
            let inward = along.normalized();
            crate::tools::arrow_head(painter, a, inward, stroke);
            crate::tools::arrow_head(painter, b, -inward, stroke);
        }
        let text = crate::dimensions::format_length((end - start).length(), &state.unit);
        // Label offset perpendicular to the line so it never sits on it.
        let normal = egui::vec2(-along.y, along.x).normalized();
        let mid = a + along * 0.5 + normal * 10.0;
        label_with_backdrop(painter, mid, &text, MEASURE_COLOR);
    }

    /// Cost of the most recent surface rebuild, for the status-bar readout.
    pub fn surface_build_ms(&self) -> f64 {
        self.last_build_ms
    }

    /// Drop all committed measurements (Delete while the Measure tool is
    /// active).
    pub fn clear_measurements(&mut self) -> usize {
        let count = self.measurements.len();
        self.measurements.clear();
        count
    }

    /// Build the next pending refinement tier and upload it.
    ///
    /// While the pointer is held down (gizmo drags, panel value drags) an
    /// edit-triggered rebuild is deferred until the last rebuild's cost times
    /// a backoff factor has elapsed, so heavy scenes update at a bounded rate
    /// instead of re-contouring every frame; cheap scenes stay per-frame.
    /// The deferred rebuild happens automatically on a later frame because
    /// the version mismatch persists.
    fn refresh_surfaces(
        &mut self,
        document: &SceneDocument,
        render_state: &RenderState,
        now: f64,
        editing: bool,
    ) {
        const REBUILD_BACKOFF: f64 = 2.5;
        const MAX_REBUILD_INTERVAL_S: f64 = 0.3;
        if document.version != self.scene_version {
            let interval =
                (self.last_build_ms / 1000.0 * REBUILD_BACKOFF).min(MAX_REBUILD_INTERVAL_S);
            if editing && now - self.last_rebuild_time < interval {
                return;
            }
            self.scene_version = document.version;
            self.pending_tiers = REFINEMENT_TIERS.to_vec();
            self.last_rebuild_time = now;
        }
        let Some(tier) = self.pending_tiers.first().copied() else {
            return;
        };
        self.pending_tiers.remove(0);
        // Visible top-level components. A domain exposed as a root while
        // nested in another chain already renders inside that chain, so
        // only primary roots are meshed (no duplicate surfaces).
        let mut components = Vec::new();
        for root in document.primary_roots() {
            if let Ok(node) = document.build_node(root) {
                components.push(node);
            }
        }
        // Subtracted 2D solid domains: their sheet is exactly the hole the
        // containing chain renders, so draw it separately. 3D subtracted
        // solids stay omitted — their wall already renders as the cavity
        // surface of the difference.
        for root in document.subtracted_domain_roots() {
            if !matches!(document.dimension_of(root), Ok(2)) {
                continue;
            }
            if let Ok(node) = document.embedded_node(root) {
                components.push(node);
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
        self.last_build_ms = scene.build_ms;
        self.base_scene = Some(scene);
        self.upload_base_scene(render_state);
        // Force the boundary overlays to re-filter against the new surfaces.
        self.overlay_signature = (u64::MAX, u64::MAX, None, i64::MAX);

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
        let now = ui.input(|input| input.time);
        let editing = ui.ctx().input(|input| input.pointer.any_down());
        self.refresh_surfaces(&state.document, render_state, now, editing);
        // Keep frames coming while a rebuild is deferred or refinement tiers
        // remain, so throttled drags and background refinement both complete
        // without waiting for the next input event.
        if state.document.version != self.scene_version || !self.pending_tiers.is_empty() {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(30));
        }
        if self.selection != self.applied_selection {
            self.upload_base_scene(render_state);
        }
        if self.upload_pending {
            self.upload_mesh_overlays(render_state);
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
            self.view_flight = None; // manual orbit overrides a view flight
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
        let now = ui.ctx().input(|input| input.time);
        // View keys stay off while a tool consumes typed input (digits are
        // dimensions during a create drag, not view shortcuts).
        if response.hovered() && !tool_consumed {
            for (_, key, plane, yaw, pitch) in REFERENCE_VIEWS {
                if ui.ctx().input(|input| input.key_pressed(key)) {
                    self.fly_to_reference_view(plane, yaw, pitch, now);
                }
            }
            ui.ctx().input(|input| {
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

        self.advance_view_flight(ui.ctx(), now);

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
        // Live cursor readout for the status bar: the world point under the
        // pointer (surface hit or grid plane), None off-viewport.
        state.cursor_world = ui
            .ctx()
            .input(|input| input.pointer.latest_pos())
            .filter(|pos| rect.contains(*pos))
            .and_then(|pos| self.measure_point(pos, rect, pixels_per_point));
        // Navigation overlays: axis triad (bottom-left) + view buttons
        // (bottom-center).
        self.paint_orientation_triad(ui, rect);
        self.view_panel_ui(ui.ctx(), rect, now);
        // Tool input + ghost overlays go on top of the rendered image.
        tools.handle_viewport(&response, ui, &self.camera, rect, pixels_per_point, state);
        if tools.kind == ToolKind::Select {
            self.handle_select_click(&response, ui, rect, pixels_per_point, state);
        }
        if tools.kind == ToolKind::Measure {
            self.handle_measure_click(&response, rect, pixels_per_point, state, tools);
        }
        // Annotation overlays: measurements persist across tool switches;
        // the bounds tripod follows the selection while toggled on. The
        // pending point only shows while the Measure tool is active.
        let measure_pending = if tools.kind == ToolKind::Measure {
            tools.points.first().copied()
        } else {
            None
        };
        self.paint_measurements(ui, rect, pixels_per_point, state, measure_pending);
        if self.show_bounds {
            self.paint_bounds_tripods(ui, rect, pixels_per_point, state);
        }
        self.paint_revolve_axes(ui, rect, pixels_per_point, state);
        self.refresh_boundary_overlays(state, tools, render_state, rect, pixels_per_point);
        // Keep painting while refinement tiers are pending.
        if !self.pending_tiers.is_empty() {
            ui.ctx().request_repaint();
        }
    }
}

/// Monospace overlay label on a dark backdrop (the triad's palette) so the
/// text stays readable over any geometry.
fn label_with_backdrop(
    painter: &egui::Painter,
    pos: egui::Pos2,
    text: &str,
    color: egui::Color32,
) {
    let galley = painter.layout_no_wrap(text.to_string(), egui::FontId::monospace(11.0), color);
    let rect = egui::Rect::from_min_size(pos, galley.size()).expand(3.0);
    painter.rect_filled(
        rect,
        egui::CornerRadius::same(3),
        egui::Color32::from_rgba_unmultiplied(6, 10, 16, 190),
    );
    painter.galley(pos, galley, color);
}

fn vertex_point(surface: &caso_surfaces::ViewportSurface, index: u32) -> Vec3 {
    let vertex = surface.vertices[index as usize];
    vec3(vertex[0] as f64, vertex[1] as f64, vertex[2] as f64)
}

/// Möller–Trumbore ray/triangle intersection; returns the ray parameter of
/// a front hit (either winding — display surfaces are viewed from both sides).
fn ray_triangle_hit(origin: Vec3, direction: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f64> {
    const EPSILON: f64 = 1e-12;
    let edge_ab = b - a;
    let edge_ac = c - a;
    let p = direction.cross(edge_ac);
    let determinant = edge_ab.dot(p);
    if determinant.abs() < EPSILON {
        return None;
    }
    let inverse = 1.0 / determinant;
    let s = origin - a;
    let u = s.dot(p) * inverse;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(edge_ab);
    let v = direction.dot(q) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = edge_ac.dot(q) * inverse;
    (t > EPSILON).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reference-view switch sets the visualization grid plane and flies
    /// the camera onto the target yaw/pitch by the end of the flight.
    #[test]
    fn reference_view_switch_sets_plane_and_lands_on_view() {
        let mut panel = ViewportPanel::default();
        let (_, _, plane, yaw, pitch) = REFERENCE_VIEWS[2]; // {x,z}
        panel.fly_to_reference_view(plane, yaw, pitch, 10.0);
        assert_eq!(panel.options.grid_plane, 1);
        assert!(panel.view_flight.is_some());

        let ctx = egui::Context::default();
        panel.advance_view_flight(&ctx, 10.0 + 2.0 * VIEW_FLIGHT_SECONDS);
        assert!(panel.view_flight.is_none());
        assert!((panel.camera.yaw - yaw.to_radians()).abs() < 1e-9);
        assert!((panel.camera.pitch - pitch.to_radians()).abs() < 1e-9);

        // Returning to 3D restores the XY grid plane.
        let (_, _, plane, yaw, pitch) = REFERENCE_VIEWS[0];
        panel.fly_to_reference_view(plane, yaw, pitch, 20.0);
        assert_eq!(panel.options.grid_plane, 0);
    }

    /// Clicking the framed default scene picks its root object; clicking
    /// empty space picks nothing.
    #[test]
    fn pick_object_hits_scene_surface() {
        let document = SceneDocument::default_scene().expect("default scene");
        let root = document.roots[0];
        let node = document.build_node(root).expect("root node");
        let bounds = node.bounding_box().expect("bounds");

        let mut panel = ViewportPanel::default();
        let mut cache = ViewportSurfaceCache::new(24);
        panel.base_scene = Some(caso_surfaces::build_viewport_surface_scene(
            &[node],
            document.version,
            &mut cache,
        ));
        panel.camera.frame_box(&bounds);

        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0));
        let center = vec3(
            (bounds.x_min + bounds.x_max) * 0.5,
            (bounds.y_min + bounds.y_max) * 0.5,
            (bounds.z_min + bounds.z_max) * 0.5,
        );
        let on_object = crate::tools::project_to_screen(&panel.camera, center, rect, 1.0)
            .expect("center projects");
        assert_eq!(panel.pick_object(on_object, rect, 1.0), Some(root));
        assert_eq!(panel.pick_object(egui::pos2(2.0, 2.0), rect, 1.0), None);
    }
}
