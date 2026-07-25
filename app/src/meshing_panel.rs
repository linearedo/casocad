//! Arrow-native mesh generation, loading, querying, and preview.

use std::collections::BTreeSet;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
use std::sync::Arc;

use caso_kernel::meshing::meshable_domains_from_document;
use caso_meshing::convert::{MeshConverter, CONVERTERS};
use caso_meshing::quality::QualityMetric;
#[cfg(not(target_arch = "wasm32"))]
use caso_meshing::NativeFileStorage;
use caso_meshing::{
    EntityKind, Interval, JobControl, MeshArtifact, MeshFile, MeshQuery, MeshingOutput,
};
use eframe::egui;

use crate::state::AppState;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{self, Receiver};

#[cfg(target_arch = "wasm32")]
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

#[cfg(target_arch = "wasm32")]
type PickedFile = Rc<RefCell<Option<(String, Vec<u8>)>>>;

#[cfg(not(target_arch = "wasm32"))]
enum JobMessage {
    Progress(caso_meshing::MeshingProgress),
    Finished(Result<MeshingOutput, String>),
}

pub struct MeshingPanel {
    mesh: Option<Arc<MeshFile>>,
    renderer: Option<caso_meshing::MeshRendererCache>,
    focus_deferred: bool,
    pub show_preview: bool,
    pub preview_revision: u64,
    inspector_active: bool,
    show_quality: bool,
    show_boundary_tags: bool,
    quality_metric: QualityMetric,
    selected_tags: BTreeSet<u64>,
    z_lower: f64,
    z_upper: f64,
    job_control: Option<JobControl>,
    #[cfg(not(target_arch = "wasm32"))]
    job: Option<Receiver<JobMessage>>,
    #[cfg(not(target_arch = "wasm32"))]
    generation_path: Option<PathBuf>,
    #[cfg(target_arch = "wasm32")]
    download_name: String,
    #[cfg(target_arch = "wasm32")]
    picked: PickedFile,
    #[cfg(target_arch = "wasm32")]
    generated: Rc<RefCell<Option<Result<MeshingOutput, String>>>>,
    #[cfg(target_arch = "wasm32")]
    worker_progress: Rc<RefCell<Option<caso_meshing::MeshingProgress>>>,
    #[cfg(target_arch = "wasm32")]
    worker: Option<web_sys::Worker>,
    #[cfg(target_arch = "wasm32")]
    worker_callback: Option<wasm_bindgen::closure::Closure<dyn FnMut(web_sys::MessageEvent)>>,
    #[cfg(target_arch = "wasm32")]
    worker_error_callback: Option<wasm_bindgen::closure::Closure<dyn FnMut(web_sys::ErrorEvent)>>,
    #[cfg(target_arch = "wasm32")]
    worker_ready: Rc<Cell<bool>>,
    #[cfg(target_arch = "wasm32")]
    worker_started_at: Option<f64>,
}

impl Default for MeshingPanel {
    fn default() -> Self {
        Self {
            mesh: None,
            renderer: None,
            focus_deferred: false,
            show_preview: true,
            preview_revision: 0,
            inspector_active: false,
            show_quality: true,
            show_boundary_tags: false,
            quality_metric: QualityMetric::ScaledJacobian,
            selected_tags: BTreeSet::new(),
            z_lower: f64::NEG_INFINITY,
            z_upper: f64::INFINITY,
            job_control: None,
            #[cfg(not(target_arch = "wasm32"))]
            job: None,
            #[cfg(not(target_arch = "wasm32"))]
            generation_path: None,
            #[cfg(target_arch = "wasm32")]
            download_name: String::new(),
            #[cfg(target_arch = "wasm32")]
            picked: PickedFile::default(),
            #[cfg(target_arch = "wasm32")]
            generated: Rc::new(RefCell::new(None)),
            #[cfg(target_arch = "wasm32")]
            worker_progress: Rc::new(RefCell::new(None)),
            #[cfg(target_arch = "wasm32")]
            worker: None,
            #[cfg(target_arch = "wasm32")]
            worker_callback: None,
            #[cfg(target_arch = "wasm32")]
            worker_error_callback: None,
            #[cfg(target_arch = "wasm32")]
            worker_ready: Rc::new(Cell::new(false)),
            #[cfg(target_arch = "wasm32")]
            worker_started_at: None,
        }
    }
}

impl MeshingPanel {
    pub fn ui(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
        self.poll(state);
        ui.horizontal_wrapped(|ui| {
            egui::ComboBox::from_id_salt("mesh_algorithm")
                .selected_text(
                    caso_meshing::descriptors()
                        .iter()
                        .find(|descriptor| descriptor.id == state.document.meshing.algorithm_id)
                        .map_or(state.document.meshing.algorithm_id.as_str(), |descriptor| {
                            descriptor.label
                        }),
                )
                .show_ui(ui, |ui| {
                    for descriptor in caso_meshing::descriptors() {
                        if ui
                            .selectable_value(
                                &mut state.document.meshing.algorithm_id,
                                descriptor.id.into(),
                                descriptor.label,
                            )
                            .changed()
                        {
                            state.document.mark_changed();
                        }
                    }
                });
            if ui
                .checkbox(&mut self.show_preview, "Preview")
                .on_hover_text("Show query-selected Arrow mesh tiles in the viewport")
                .changed()
            {
                self.preview_revision = self.preview_revision.wrapping_add(1);
            }
            if ui
                .checkbox(&mut self.inspector_active, "Inspector")
                .changed()
            {
                self.preview_revision = self.preview_revision.wrapping_add(1);
            }
        });

        let mut changed = false;
        ui.horizontal(|ui| {
            changed |= ui
                .add(
                    egui::DragValue::new(&mut state.document.meshing.element_min_size)
                        .range(1.0e-9..=state.document.meshing.element_max_size)
                        .speed(0.001)
                        .prefix("Element Min: "),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut state.document.meshing.element_max_size)
                        .range(state.document.meshing.element_min_size..=1.0e9)
                        .speed(0.001)
                        .prefix("Element Max: "),
                )
                .changed();
        });
        #[cfg(target_arch = "wasm32")]
        {
            changed |= ui
                .add(
                    egui::Slider::new(&mut state.document.meshing.wasm_file_cap_mib, 64..=512)
                        .text("WASM file cap (MiB)"),
                )
                .changed();
        }
        if changed {
            state.document.mark_changed();
        }

        ui.horizontal_wrapped(|ui| {
            if self.job_control.is_none() {
                if ui.button("Generate").clicked() {
                    self.generate(state, ui.ctx().clone());
                }
            } else if ui.button("Cancel").clicked() {
                self.cancel_generation(state);
            }
            if ui.button("Load .casomesh.arrow…").clicked() {
                self.load(ui, state);
            }
            ui.add_enabled_ui(self.mesh.is_some(), |ui| {
                if ui.button("Export Arrow…").clicked() {
                    self.export(state);
                }
                ui.menu_button("Convert", |ui| {
                    for converter in CONVERTERS {
                        if ui.button(converter.label).clicked() {
                            self.convert(state, converter);
                            ui.close();
                        }
                    }
                });
                if ui.button("Full Audit").clicked() {
                    match self
                        .mesh
                        .as_ref()
                        .expect("button is enabled only with a mesh")
                        .full_audit(&JobControl::default())
                    {
                        Ok(report) => {
                            state.status = format!(
                                "Full mesh audit passed: {} batches, {} entities",
                                report.exact_batches, report.entities
                            )
                        }
                        Err(error) => state.status = format!("Full mesh audit failed: {error}"),
                    }
                }
            });
        });
        #[cfg(not(target_arch = "wasm32"))]
        ui.horizontal_wrapped(|ui| {
            ui.weak(
                self.generation_path
                    .as_ref()
                    .map_or("Output: choose on first generation".into(), |path| {
                        format!("Output: {}", path.display())
                    }),
            );
            if ui.button("Change Output…").clicked() {
                if let Some(path) = mesh_dialog()
                    .set_file_name("mesh.casomesh.arrow")
                    .save_file()
                {
                    self.generation_path = Some(path);
                }
            }
        });
        #[cfg(target_arch = "wasm32")]
        ui.add(
            egui::TextEdit::singleline(&mut self.download_name)
                .desired_width(160.0)
                .hint_text("mesh"),
        );

        if let Some(mesh) = &self.mesh {
            let counts = &mesh.manifest().counts;
            ui.weak(format!(
                "{}D Arrow mesh — {} points, {} cells, {} tiles",
                mesh.manifest().dimension,
                counts.points,
                counts.cells,
                mesh.entity_batches(caso_meshing::RowKind::Cell)
                    .filter_map(|entry| entry.spatial_node_id)
                    .collect::<BTreeSet<_>>()
                    .len()
            ));
        }
        ui.separator();
        ui.label("Rhai Meshing Controls");
        ui.weak("Scripts can create typed refinement and boundary-layer controls only.");
        if ui
            .add(
                egui::TextEdit::multiline(&mut state.document.meshing.control_script)
                    .code_editor()
                    .desired_width(f32::INFINITY)
                    .desired_rows(12),
            )
            .changed()
        {
            state.document.mark_changed();
        }
    }

    pub fn inspector_active(&self) -> bool {
        self.inspector_active
    }

    pub fn inspector_ui(&mut self, ui: &mut egui::Ui, _state: &AppState) {
        let Some(mesh) = &self.mesh else {
            ui.weak("Generate or load a .casomesh.arrow file to inspect it.");
            return;
        };
        let mut changed = false;
        ui.horizontal(|ui| {
            changed |= ui.checkbox(&mut self.show_quality, "Quality").changed();
            changed |= ui
                .checkbox(&mut self.show_boundary_tags, "Boundary Tags")
                .changed();
            if self.show_quality {
                egui::ComboBox::from_id_salt("mesh_quality_metric")
                    .selected_text(self.quality_metric.label())
                    .show_ui(ui, |ui| {
                        for metric in QualityMetric::ALL {
                            changed |= ui
                                .selectable_value(&mut self.quality_metric, metric, metric.label())
                                .changed();
                        }
                    });
            }
        });
        if mesh.manifest().dimension == 3 {
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.z_lower)
                        .speed(0.01)
                        .prefix("Z min: "),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.z_upper)
                        .speed(0.01)
                        .prefix("Z max: "),
                )
                .changed();
        }
        if self.show_boundary_tags {
            let tags = assigned_tags(mesh);
            ui.horizontal_wrapped(|ui| {
                if ui.button("All").clicked() {
                    self.selected_tags = tags.iter().map(|(id, _)| *id).collect();
                    changed = true;
                }
                if ui.button("None").clicked() {
                    self.selected_tags.clear();
                    changed = true;
                }
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
        }
        if changed {
            self.preview_revision = self.preview_revision.wrapping_add(1);
        }
    }

    pub fn update_mesh_focus(&mut self, focus: [f64; 3]) {
        if self.show_preview {
            if let Some(renderer) = &mut self.renderer {
                renderer.update_lod_focus(focus);
            }
        }
    }

    pub fn set_focus_deferred(&mut self, deferred: bool) {
        if deferred && !self.focus_deferred {
            if let Some(renderer) = &mut self.renderer {
                renderer.defer_lod_view();
            }
        }
        self.focus_deferred = deferred;
    }

    pub fn prepare_preview_frame(&mut self) -> Option<caso_meshing::IncrementalLodPreparation> {
        let Some(mesh) = &self.mesh else {
            return None;
        };
        let entity_kind = if self.show_preview {
            preview_entity_kind(
                mesh.manifest().dimension,
                self.show_boundary_tags,
                mesh.manifest().counts.faces,
            )
        } else {
            EntityKind::Cell
        };
        let query = MeshQuery {
            entity_kind,
            z: Interval::new(self.z_lower, self.z_upper),
            tag_ids: if self.show_boundary_tags && entity_kind != EntityKind::Cell {
                self.selected_tags.clone()
            } else {
                BTreeSet::new()
            },
            display_limit: usize::MAX,
            ..MeshQuery::default()
        };
        let Some(renderer) = &mut self.renderer else {
            return None;
        };
        if !self.show_preview {
            renderer.clear_lod_view();
        }
        renderer
            .prepare_lod_incremental(query, &BTreeSet::new(), &BTreeSet::new(), 1.0)
            .ok()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn generate(&mut self, state: &mut AppState, ctx: egui::Context) {
        let domains = match meshable_domains_from_document(&state.document) {
            Ok(domains) => domains,
            Err(error) => {
                state.status = format!("Meshing failed: {error}");
                return;
            }
        };
        let controls = match crate::meshing_controls::compile_control_script(
            &domains,
            &state.document.meshing.control_script,
        ) {
            Ok(controls) => controls,
            Err(error) => {
                state.status = format!("Meshing controls failed: {error}");
                return;
            }
        };
        let Some(path) = reuse_or_choose_path(&mut self.generation_path, || {
            mesh_dialog()
                .set_file_name("mesh.casomesh.arrow")
                .save_file()
        }) else {
            return;
        };
        let (sender, receiver) = mpsc::channel();
        let progress_sender = sender.clone();
        let control = JobControl::default().with_progress(move |progress| {
            let _ = progress_sender.send(JobMessage::Progress(progress));
            ctx.request_repaint();
        });
        let request = caso_meshing::MeshingRequest {
            domains,
            algorithm_id: state.document.meshing.algorithm_id.clone(),
            element_min_size: state.document.meshing.element_min_size,
            element_max_size: state.document.meshing.element_max_size,
            controls,
            limits: caso_meshing::GenerationLimits::default(),
            job_control: control.clone(),
        };
        std::thread::spawn(move || {
            let result = NativeFileStorage::new(&path)
                .and_then(|storage| caso_meshing::run_meshing(request, storage))
                .map_err(|error| error.to_string());
            let _ = sender.send(JobMessage::Finished(result));
        });
        self.job_control = Some(control);
        self.job = Some(receiver);
        state.status = "Generating Arrow mesh…".into();
    }

    #[cfg(target_arch = "wasm32")]
    fn generate(&mut self, state: &mut AppState, ctx: egui::Context) {
        use wasm_bindgen::JsCast;
        let domains = match meshable_domains_from_document(&state.document) {
            Ok(domains) => domains,
            Err(error) => {
                state.status = format!("Meshing failed: {error}");
                return;
            }
        };
        let _ = domains;
        let scene = match caso_kernel::serialization::save_scene_to_string(&state.document) {
            Ok(scene) => scene,
            Err(error) => {
                state.status = format!("Meshing failed: {error}");
                return;
            }
        };
        let options = web_sys::WorkerOptions::new();
        options.set_type(web_sys::WorkerType::Module);
        let worker = match web_sys::Worker::new_with_options("mesh_worker.js", &options) {
            Ok(worker) => worker,
            Err(error) => {
                state.status = format!("Could not start mesh worker: {error:?}");
                return;
            }
        };
        let request = serde_json::json!({
            "scene": scene,
            "cap_mib": state.document.meshing.wasm_file_cap_mib,
        })
        .to_string();
        let generated = self.generated.clone();
        let worker_progress = self.worker_progress.clone();
        let worker_ready = self.worker_ready.clone();
        let callback_worker = worker.clone();
        let callback_generated = generated.clone();
        let callback_ctx = ctx.clone();
        let callback = wasm_bindgen::closure::Closure::<dyn FnMut(web_sys::MessageEvent)>::new(
            move |event: web_sys::MessageEvent| {
                let data = event.data();
                let get = |key: &str| {
                    js_sys::Reflect::get(&data, &wasm_bindgen::JsValue::from_str(key))
                        .unwrap_or(wasm_bindgen::JsValue::UNDEFINED)
                };
                let kind = get("kind").as_string().unwrap_or_default();
                let number = |key: &str| get(key).as_f64().unwrap_or(0.0) as u64;
                let result = match kind.as_str() {
                    "ready" => {
                        if !worker_ready.replace(true) {
                            if let Err(error) = callback_worker
                                .post_message(&wasm_bindgen::JsValue::from_str(&request))
                            {
                                *callback_generated.borrow_mut() = Some(Err(format!(
                                    "Could not send mesh worker request: {error:?}"
                                )));
                            }
                        }
                        callback_ctx.request_repaint();
                        return;
                    }
                    "progress" => {
                        *worker_progress.borrow_mut() = Some(caso_meshing::MeshingProgress {
                            completed_chunks: number("completed_chunks"),
                            cells_committed: number("cells_committed"),
                            active_bytes: number("active_bytes"),
                        });
                        callback_ctx.request_repaint();
                        return;
                    }
                    "complete" => {
                        let bytes = js_sys::Uint8Array::new(&get("bytes")).to_vec();
                        Ok(MeshingOutput {
                            artifact: MeshArtifact::Memory(Arc::from(bytes)),
                            statistics: caso_meshing::MeshingStatistics {
                                domains: number("domains"),
                                chunks: number("chunks"),
                                points: number("points"),
                                cells: number("cells"),
                                committed_batches: number("batches"),
                                peak_active_bytes: number("peak_bytes"),
                                elapsed_millis: number("elapsed_millis"),
                            },
                        })
                    }
                    "error" => Err(get("error")
                        .as_string()
                        .unwrap_or_else(|| "mesh worker failed".into())),
                    _ => Err(format!(
                        "mesh worker sent unknown message kind: {}",
                        if kind.is_empty() { "<missing>" } else { &kind }
                    )),
                };
                *generated.borrow_mut() = Some(result);
                callback_ctx.request_repaint();
            },
        );
        let error_generated = self.generated.clone();
        let error_ctx = ctx.clone();
        let error_callback = wasm_bindgen::closure::Closure::<dyn FnMut(web_sys::ErrorEvent)>::new(
            move |event: web_sys::ErrorEvent| {
                let message = event.message();
                *error_generated.borrow_mut() = Some(Err(if message.is_empty() {
                    "mesh worker failed to load".into()
                } else {
                    message
                }));
                error_ctx.request_repaint();
            },
        );
        worker.set_onmessage(Some(callback.as_ref().unchecked_ref()));
        worker.set_onerror(Some(error_callback.as_ref().unchecked_ref()));
        self.worker_ready.set(false);
        self.worker_started_at = Some(js_sys::Date::now());
        self.worker = Some(worker);
        self.worker_callback = Some(callback);
        self.worker_error_callback = Some(error_callback);
        self.job_control = Some(JobControl::default());
        ctx.request_repaint_after(std::time::Duration::from_secs(60));
        state.status = "Starting mesh worker…".into();
    }

    fn cancel_generation(&mut self, state: &mut AppState) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(control) = &self.job_control {
            control.cancel();
            state.status = "Cancelling mesh generation…".into();
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.cleanup_wasm_worker();
            state.status = "Mesh generation cancelled".into();
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn poll(&mut self, state: &mut AppState) {
        let Some(receiver) = &self.job else {
            return;
        };
        let messages: Vec<_> = receiver.try_iter().collect();
        for message in messages {
            match message {
                JobMessage::Progress(progress) => {
                    state.status = format!(
                        "Meshing chunk {} — {} cells",
                        progress.completed_chunks, progress.cells_committed
                    );
                }
                JobMessage::Finished(result) => {
                    self.job = None;
                    self.job_control = None;
                    match result {
                        Ok(output) => self.install_output(state, output),
                        Err(error) => state.status = format!("Meshing failed: {error}"),
                    }
                    break;
                }
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn poll(&mut self, state: &mut AppState) {
        if self.worker_ready.get() && self.worker_started_at.take().is_some() {
            state.status = "Generating Arrow mesh…".into();
        }
        if let Some(progress) = self.worker_progress.borrow_mut().take() {
            state.status = format!(
                "Meshing chunk {} — {} cells",
                progress.completed_chunks, progress.cells_committed
            );
        }
        let generated = { self.generated.borrow_mut().take() };
        if let Some(result) = generated {
            self.cleanup_wasm_worker();
            match result {
                Ok(output) => self.install_output(state, output),
                Err(error) => state.status = format!("Meshing failed: {error}"),
            }
        } else if self
            .worker_started_at
            .is_some_and(|started_at| js_sys::Date::now() - started_at >= 60_000.0)
        {
            self.cleanup_wasm_worker();
            state.status = "Meshing failed: mesh worker did not start within 60 seconds".into();
        }
        let picked = { self.picked.borrow_mut().take() };
        if let Some((name, bytes)) = picked {
            self.apply_wasm_bytes(state, &name, bytes);
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn cleanup_wasm_worker(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.set_onmessage(None);
            worker.set_onerror(None);
            worker.terminate();
        }
        self.worker_callback = None;
        self.worker_error_callback = None;
        self.worker_ready.set(false);
        self.worker_started_at = None;
        *self.worker_progress.borrow_mut() = None;
        *self.generated.borrow_mut() = None;
        self.job_control = None;
    }

    fn install_output(&mut self, state: &mut AppState, output: MeshingOutput) {
        let opened = match output.artifact {
            #[cfg(not(target_arch = "wasm32"))]
            MeshArtifact::Native(path) => MeshFile::open_native(path),
            MeshArtifact::Memory(bytes) => MeshFile::from_memory(bytes),
        };
        match opened {
            Ok(mesh) => {
                state.status = format!(
                    "Mesh complete: {} chunks, {} points, {} cells",
                    output.statistics.chunks,
                    mesh.manifest().counts.points,
                    mesh.manifest().counts.cells
                );
                self.install_mesh(mesh);
            }
            Err(error) => state.status = format!("Generated mesh failed validation: {error}"),
        }
    }

    fn install_mesh(&mut self, mesh: MeshFile) {
        let mesh = Arc::new(mesh);
        self.selected_tags = assigned_tags(&mesh).into_iter().map(|(id, _)| id).collect();
        self.z_lower = mesh.manifest().bounds.min[2];
        self.z_upper = mesh.manifest().bounds.max[2];
        self.renderer = Some(caso_meshing::MeshRendererCache::new(
            mesh.clone(),
            caso_meshing::RendererBudgets::default(),
        ));
        self.mesh = Some(mesh);
        self.preview_revision = self.preview_revision.wrapping_add(1);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load(&mut self, _ui: &egui::Ui, state: &mut AppState) {
        let Some(path) = mesh_dialog().pick_file() else {
            return;
        };
        match MeshFile::open_native(&path) {
            Ok(mesh) => {
                state.status = format!(
                    "Loaded {}: {} points, {} cells",
                    path.display(),
                    mesh.manifest().counts.points,
                    mesh.manifest().counts.cells
                );
                self.install_mesh(mesh);
            }
            Err(error) => state.status = format!("Arrow load failed: {error}"),
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn load(&mut self, ui: &egui::Ui, _state: &mut AppState) {
        let picked = self.picked.clone();
        let ctx = ui.ctx().clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Some(file) = rfd::AsyncFileDialog::new()
                .add_filter("casoCAD Arrow mesh", &["arrow"])
                .pick_file()
                .await
            {
                *picked.borrow_mut() = Some((file.file_name(), file.read().await));
                ctx.request_repaint();
            }
        });
    }

    #[cfg(target_arch = "wasm32")]
    fn apply_wasm_bytes(&mut self, state: &mut AppState, name: &str, bytes: Vec<u8>) {
        match MeshFile::from_wasm_bytes(Arc::from(bytes)) {
            Ok(mesh) => {
                state.status = format!(
                    "Loaded {name}: {} points, {} cells",
                    mesh.manifest().counts.points,
                    mesh.manifest().counts.cells
                );
                self.install_mesh(mesh);
            }
            Err(error) => state.status = format!("Arrow load failed ({name}): {error}"),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn export(&mut self, state: &mut AppState) {
        let Some(mesh) = &self.mesh else {
            return;
        };
        let Some(path) = mesh_dialog()
            .set_file_name("mesh.casomesh.arrow")
            .save_file()
        else {
            return;
        };
        let result = if let Some(source) = mesh.source_path() {
            std::fs::copy(source, &path).map(|_| ())
        } else {
            std::fs::write(&path, mesh.bytes())
        };
        match result {
            Ok(()) => state.status = format!("Exported {}", path.display()),
            Err(error) => state.status = format!("Export failed: {error}"),
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn export(&mut self, state: &mut AppState) {
        let Some(mesh) = &self.mesh else {
            return;
        };
        let name = download_name(&self.download_name, "mesh.casomesh.arrow");
        match crate::web_download_bytes(&name, mesh.bytes()) {
            Ok(()) => state.status = format!("Downloaded {name}"),
            Err(error) => state.status = format!("Download failed: {error:?}"),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn convert(&mut self, state: &mut AppState, converter: &MeshConverter) {
        let Some(mesh) = &self.mesh else {
            return;
        };
        let Some(path) = rfd::FileDialog::new()
            .add_filter(converter.label, &[converter.extension])
            .set_file_name(format!("mesh.{}", converter.extension))
            .save_file()
        else {
            return;
        };
        let result = std::fs::File::create(&path)
            .map(std::io::BufWriter::new)
            .map_err(caso_meshing::MeshError::from)
            .and_then(|mut output| {
                caso_meshing::convert::write_to(converter.id, mesh, &mut output)?;
                std::io::Write::flush(&mut output).map_err(Into::into)
            });
        match result {
            Ok(()) => state.status = format!("Exported {}", path.display()),
            Err(error) => {
                let _ = std::fs::remove_file(&path);
                state.status = format!("{} conversion failed: {error}", converter.label);
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn convert(&mut self, state: &mut AppState, converter: &MeshConverter) {
        let Some(mesh) = &self.mesh else {
            return;
        };
        let bytes = match (converter.write)(mesh) {
            Ok(bytes) => bytes,
            Err(error) => {
                state.status = format!("{} conversion failed: {error}", converter.label);
                return;
            }
        };
        let default = format!("mesh.{}", converter.extension);
        let name = download_name(&self.download_name, &default);
        match crate::web_download_bytes(&name, &bytes) {
            Ok(()) => state.status = format!("Downloaded {name}"),
            Err(error) => state.status = format!("Download failed: {error:?}"),
        }
    }
}

fn preview_entity_kind(dimension: u8, show_boundary_tags: bool, faces: u64) -> EntityKind {
    match (dimension, show_boundary_tags, faces) {
        (2, true, _) => EntityKind::Edge,
        (3, true, 1..) => EntityKind::Face,
        _ => EntityKind::Cell,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn mesh_dialog() -> rfd::FileDialog {
    rfd::FileDialog::new().add_filter("casoCAD Arrow mesh", &["arrow"])
}

#[cfg(not(target_arch = "wasm32"))]
fn reuse_or_choose_path(
    saved: &mut Option<PathBuf>,
    choose: impl FnOnce() -> Option<PathBuf>,
) -> Option<PathBuf> {
    saved.clone().or_else(|| {
        *saved = choose();
        saved.clone()
    })
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[test]
fn mesh_generation_reuses_the_first_selected_path() {
    let mut saved = None;
    let mut choices = 0;
    for _ in 0..2 {
        assert_eq!(
            reuse_or_choose_path(&mut saved, || {
                choices += 1;
                Some("chosen.casomesh.arrow".into())
            }),
            Some("chosen.casomesh.arrow".into())
        );
    }
    assert_eq!(choices, 1);
}

#[test]
fn preview_query_kind_preserves_2d_and_switches_3d_only_for_boundary_faces() {
    assert_eq!(preview_entity_kind(2, false, 0), EntityKind::Cell);
    assert_eq!(preview_entity_kind(2, true, 0), EntityKind::Edge);
    assert_eq!(preview_entity_kind(3, false, 4), EntityKind::Cell);
    assert_eq!(preview_entity_kind(3, true, 4), EntityKind::Face);
    assert_eq!(preview_entity_kind(3, true, 0), EntityKind::Cell);
}

#[cfg(target_arch = "wasm32")]
fn download_name(raw: &str, default_name: &str) -> String {
    let raw = raw.trim();
    if raw.is_empty() {
        default_name.into()
    } else if raw.contains('.') {
        raw.into()
    } else {
        format!(
            "{raw}.{}",
            default_name.split_once('.').map_or("", |(_, ext)| ext)
        )
    }
}

fn assigned_tags(mesh: &MeshFile) -> Vec<(u64, String)> {
    let mut tags = BTreeSet::new();
    for kind in [caso_meshing::RowKind::Edge, caso_meshing::RowKind::Face] {
        for entry in mesh.entity_batches(kind) {
            tags.extend(entry.tag_ids.iter().copied());
        }
    }
    tags.into_iter()
        .filter_map(|id| mesh.catalog_name("tag", id).map(|name| (id, name.into())))
        .collect()
}
