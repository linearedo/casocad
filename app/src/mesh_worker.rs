use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use caso_meshing::convert;
use caso_meshing::{
    EntityKind, Interval, JobControl, LodTargetSelection, MemoryArtifact, MemoryStorage,
    MeshArtifact, MeshAuditCursor, MeshAuditProgress, MeshAuditReport, MeshFile, MeshManifest,
    MeshQuery, MeshQueryCursor, MeshQueryStatistics, MeshRenderStyle, MeshRendererCache,
    MeshStorage, MeshTileKey, MeshingPhase, MeshingProgress, MeshingStatistics, QueryBudget,
    QueryMeasures, QueryProgress, QueryStatisticsAccumulator, RendererBudgets, TagFilter, TagScope,
};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

pub const WEB_MAX_ARTIFACT_MIB: u16 = 256;
const WEB_MIN_ARTIFACT_MIB: u16 = 64;
const FILE_READ_CHUNK_BYTES: usize = 4 * 1024 * 1024;
pub const PREVIEW_PACKET_BYTES: usize = 1024 * 1024;
const PREVIEW_PACKET_LINES: usize = PREVIEW_PACKET_BYTES / (9 * size_of::<f32>());

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkerCommand {
    Generate {
        session_id: u64,
        scene: String,
        cap_mib: u16,
    },
    Preview {
        session_id: u64,
        revision: u64,
        focus: [f64; 3],
        query: BrowserQuery,
        style: MeshRenderStyle,
    },
    AnalysisStart {
        session_id: u64,
        request_id: u64,
        query: BrowserQuery,
    },
    AnalysisStep {
        session_id: u64,
        request_id: u64,
    },
    AuditStart {
        session_id: u64,
        request_id: u64,
    },
    AuditStep {
        session_id: u64,
        request_id: u64,
    },
    ExportArrow {
        session_id: u64,
        request_id: u64,
    },
    Convert {
        session_id: u64,
        request_id: u64,
        converter_id: String,
    },
}

impl WorkerCommand {
    fn session_id(&self) -> u64 {
        match *self {
            Self::Generate { session_id, .. }
            | Self::Preview { session_id, .. }
            | Self::AnalysisStart { session_id, .. }
            | Self::AnalysisStep { session_id, .. }
            | Self::AuditStart { session_id, .. }
            | Self::AuditStep { session_id, .. }
            | Self::ExportArrow { session_id, .. }
            | Self::Convert { session_id, .. } => session_id,
        }
    }

    fn request_id(&self) -> Option<u64> {
        match self {
            Self::AnalysisStart { request_id, .. }
            | Self::AnalysisStep { request_id, .. }
            | Self::AuditStart { request_id, .. }
            | Self::AuditStep { request_id, .. }
            | Self::ExportArrow { request_id, .. }
            | Self::Convert { request_id, .. } => Some(*request_id),
            Self::Generate { .. } | Self::Preview { .. } => None,
        }
    }

    fn operation(&self) -> &'static str {
        match self {
            Self::Generate { .. } => "generation",
            Self::Preview { .. } => "preview",
            Self::AnalysisStart { .. } | Self::AnalysisStep { .. } => "analysis",
            Self::AuditStart { .. } | Self::AuditStep { .. } => "audit",
            Self::ExportArrow { .. } => "export",
            Self::Convert { .. } => "conversion",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkerResponse {
    Ready,
    Progress {
        session_id: u64,
        progress: MeshingProgress,
    },
    Installed {
        session_id: u64,
        name: String,
        summary: Box<BrowserMeshSummary>,
    },
    PreviewState {
        session_id: u64,
        revision: u64,
        selection: LodTargetSelection,
        more: bool,
    },
    Analysis {
        session_id: u64,
        request_id: u64,
        progress: QueryProgress,
        statistics: Option<MeshQueryStatistics>,
    },
    Audit {
        session_id: u64,
        request_id: u64,
        progress: MeshAuditProgress,
        report: Option<MeshAuditReport>,
    },
    Error {
        session_id: u64,
        request_id: Option<u64>,
        operation: String,
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserMeshSummary {
    pub manifest: MeshManifest,
    pub statistics: MeshingStatistics,
    pub artifact_bytes: u64,
    pub tile_count: usize,
    pub tags: Vec<(u64, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserQuery {
    pub x: BrowserInterval,
    pub y: BrowserInterval,
    pub z: BrowserInterval,
    pub entity_kind: EntityKind,
    pub zone_ids: Vec<u64>,
    pub tag_ids: Vec<u64>,
    pub tag_scope: Option<TagScope>,
    pub measures: QueryMeasures,
    pub boundary_distance: Option<BrowserInterval>,
    pub display_limit: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BrowserInterval {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl From<Interval> for BrowserInterval {
    fn from(interval: Interval) -> Self {
        Self {
            min: interval.min.is_finite().then_some(interval.min),
            max: interval.max.is_finite().then_some(interval.max),
        }
    }
}

impl BrowserInterval {
    fn into_interval(self) -> Interval {
        Interval::new(
            self.min.unwrap_or(f64::NEG_INFINITY),
            self.max.unwrap_or(f64::INFINITY),
        )
    }
}

impl From<&MeshQuery> for BrowserQuery {
    fn from(query: &MeshQuery) -> Self {
        Self {
            x: query.x.into(),
            y: query.y.into(),
            z: query.z.into(),
            entity_kind: query.entity_kind,
            zone_ids: query.zone_ids.iter().copied().collect(),
            tag_ids: query
                .tag_filter
                .as_ref()
                .map_or_else(Vec::new, |filter| filter.ids.iter().copied().collect()),
            tag_scope: query.tag_filter.as_ref().map(|filter| filter.scope),
            measures: query.measures,
            boundary_distance: query.boundary_distance.map(Into::into),
            display_limit: query.display_limit,
        }
    }
}

impl BrowserQuery {
    fn into_query(self) -> MeshQuery {
        MeshQuery {
            x: self.x.into_interval(),
            y: self.y.into_interval(),
            z: self.z.into_interval(),
            entity_kind: self.entity_kind,
            zone_ids: self.zone_ids.into_iter().collect(),
            tag_filter: self
                .tag_scope
                .map(|scope| TagFilter::any(self.tag_ids, scope)),
            measures: self.measures,
            boundary_distance: self.boundary_distance.map(BrowserInterval::into_interval),
            display_limit: self.display_limit,
            ..MeshQuery::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewPacketMeta {
    pub session_id: u64,
    pub revision: u64,
    pub generation: u64,
    pub key: MeshTileKey,
    pub complete: bool,
}

struct PendingPreview {
    revision: u64,
    generation: u64,
    key: MeshTileKey,
    lines: Arc<[caso_meshing::RenderLine]>,
    offset: usize,
}

struct AnalysisState {
    request_id: u64,
    cursor: MeshQueryCursor,
    accumulator: QueryStatisticsAccumulator,
}

struct AuditState {
    request_id: u64,
    cursor: MeshAuditCursor,
}

#[derive(Default)]
struct WorkerState {
    session_id: u64,
    cap_bytes: usize,
    mesh: Option<Arc<MeshFile>>,
    renderer: Option<MeshRendererCache>,
    preview_focus: Option<[f64; 3]>,
    preview_selection: Option<LodTargetSelection>,
    pending_preview: Option<PendingPreview>,
    analysis: Option<AnalysisState>,
    audit: Option<AuditState>,
}

pub fn install(scope: web_sys::DedicatedWorkerGlobalScope) {
    let state = Rc::new(RefCell::new(WorkerState::default()));
    let callback_scope = scope.clone();
    let callback_state = state.clone();
    let callback =
        Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |event: web_sys::MessageEvent| {
            let failure = if let Some(text) = event.data().as_string() {
                match serde_json::from_str::<WorkerCommand>(&text) {
                    Ok(command) => {
                        let session_id = command.session_id();
                        let request_id = command.request_id();
                        let operation = command.operation();
                        callback_state
                            .borrow_mut()
                            .handle_command(&callback_scope, command)
                            .err()
                            .map(|error| (session_id, request_id, operation, error))
                    }
                    Err(error) => Some((
                        callback_state.borrow().session_id,
                        None,
                        "protocol",
                        error.to_string(),
                    )),
                }
            } else {
                let result = callback_state
                    .borrow_mut()
                    .handle_load(&callback_scope, &event.data());
                let session_id = callback_state.borrow().session_id;
                result.err().map(|error| (session_id, None, "load", error))
            };
            if let Some((session_id, request_id, operation, error)) = failure {
                post_json(
                    &callback_scope,
                    &WorkerResponse::Error {
                        session_id,
                        request_id,
                        operation: operation.into(),
                        error,
                    },
                );
            }
        });
    scope.set_onmessage(Some(callback.as_ref().unchecked_ref()));
    callback.forget();
    post_json(&scope, &WorkerResponse::Ready);
}

impl WorkerState {
    fn handle_command(
        &mut self,
        scope: &web_sys::DedicatedWorkerGlobalScope,
        command: WorkerCommand,
    ) -> Result<(), String> {
        if !matches!(command, WorkerCommand::Generate { .. })
            && command.session_id() != self.session_id
        {
            return Ok(());
        }
        match command {
            WorkerCommand::Generate {
                session_id,
                scene,
                cap_mib,
            } => self.generate(scope, session_id, &scene, cap_mib),
            WorkerCommand::Preview {
                revision,
                focus,
                query,
                style,
                ..
            } => self.preview(scope, revision, focus, query.into_query(), style),
            WorkerCommand::AnalysisStart {
                request_id, query, ..
            } => self.start_analysis(scope, request_id, query.into_query()),
            WorkerCommand::AnalysisStep { request_id, .. } => self.step_analysis(scope, request_id),
            WorkerCommand::AuditStart { request_id, .. } => self.start_audit(scope, request_id),
            WorkerCommand::AuditStep { request_id, .. } => self.step_audit(scope, request_id),
            WorkerCommand::ExportArrow { request_id, .. } => {
                let mesh = self.mesh()?;
                let blob = bytes_blob(mesh.bytes())?;
                post_blob(scope, self.session_id, request_id, blob);
                Ok(())
            }
            WorkerCommand::Convert {
                request_id,
                converter_id,
                ..
            } => self.convert(scope, request_id, &converter_id),
        }
    }

    fn generate(
        &mut self,
        scope: &web_sys::DedicatedWorkerGlobalScope,
        session_id: u64,
        scene: &str,
        cap_mib: u16,
    ) -> Result<(), String> {
        let cap_bytes = validated_cap(cap_mib)?;
        self.reset(session_id, cap_bytes);
        let document = caso_kernel::serialization::load_scene_from_str(scene)
            .map_err(|error| error.to_string())?;
        let domains = caso_kernel::meshing::meshable_domains_from_document(&document)
            .map_err(|error| error.to_string())?;
        let controls = crate::meshing_controls::compile_control_script(
            &domains,
            &document.meshing.control_script,
        )?;
        let progress_scope = scope.clone();
        let last_post = Rc::new(std::cell::Cell::new(-1000.0));
        let last_phase = Rc::new(std::cell::Cell::new(MeshingPhase::Generating));
        let progress_last_post = last_post.clone();
        let progress_last_phase = last_phase.clone();
        let control = JobControl::default().with_progress(move |progress| {
            let now = js_sys::Date::now();
            if progress.phase != progress_last_phase.get()
                || now - progress_last_post.get() >= 100.0
            {
                progress_last_phase.set(progress.phase);
                progress_last_post.set(now);
                post_json(
                    &progress_scope,
                    &WorkerResponse::Progress {
                        session_id,
                        progress,
                    },
                );
            }
        });
        let output = caso_meshing::run_meshing(
            caso_meshing::MeshingRequest {
                domains,
                algorithm_id: document.meshing.algorithm_id.clone(),
                element_min_size: document.meshing.element_min_size,
                element_max_size: document.meshing.element_max_size,
                controls,
                limits: caso_meshing::GenerationLimits::default(),
                job_control: control,
            },
            MemoryStorage::new(cap_bytes).map_err(|error| error.to_string())?,
        )
        .map_err(|error| web_error(error.to_string(), cap_mib))?;
        let MeshArtifact::Memory(artifact) = output.artifact;
        self.install_mesh(scope, "Generated mesh", artifact, output.statistics)
    }

    fn handle_load(
        &mut self,
        scope: &web_sys::DedicatedWorkerGlobalScope,
        data: &JsValue,
    ) -> Result<(), String> {
        let get = |key: &str| {
            js_sys::Reflect::get(data, &JsValue::from_str(key))
                .map_err(|error| format!("could not read load request {key}: {error:?}"))
        };
        if get("kind")?.as_string().as_deref() != Some("load") {
            return Err("worker request must be JSON text or a load object".into());
        }
        let session_id = get("session_id")?
            .as_f64()
            .ok_or_else(|| "load request has no session_id".to_string())?
            as u64;
        let cap_mib = get("cap_mib")?
            .as_f64()
            .ok_or_else(|| "load request has no cap_mib".to_string())? as u16;
        let name = get("name")?
            .as_string()
            .unwrap_or_else(|| "mesh.casomesh.arrow".into());
        let file = get("file")?
            .dyn_into::<web_sys::File>()
            .map_err(|_| "load request file is not a browser File".to_string())?;
        let cap_bytes = validated_cap(cap_mib)?;
        self.reset(session_id, cap_bytes);
        let artifact = read_file(&file, cap_bytes)?;
        self.install_mesh(scope, &name, artifact, MeshingStatistics::default())
    }

    fn install_mesh(
        &mut self,
        scope: &web_sys::DedicatedWorkerGlobalScope,
        name: &str,
        artifact: MemoryArtifact,
        mut statistics: MeshingStatistics,
    ) -> Result<(), String> {
        let artifact_bytes = artifact.len() as u64;
        let mesh = Arc::new(MeshFile::from_memory(artifact).map_err(|error| error.to_string())?);
        if statistics.points == 0 {
            statistics.points = mesh.manifest().counts.points;
            statistics.cells = mesh.manifest().counts.cells;
            statistics.chunks = mesh
                .entity_batches(caso_meshing::RowKind::Cell)
                .filter_map(|entry| entry.spatial_node_id)
                .collect::<BTreeSet<_>>()
                .len() as u64;
        }
        let summary = BrowserMeshSummary {
            manifest: mesh.manifest().clone(),
            statistics,
            artifact_bytes,
            tile_count: mesh
                .entity_batches(caso_meshing::RowKind::Cell)
                .filter_map(|entry| entry.spatial_node_id)
                .collect::<BTreeSet<_>>()
                .len(),
            tags: assigned_tags(&mesh),
        };
        self.renderer = Some(MeshRendererCache::new(
            mesh.clone(),
            RendererBudgets {
                decoded_bytes: 64 * 1024 * 1024,
                gpu_bytes: 64 * 1024 * 1024,
            },
        ));
        self.mesh = Some(mesh);
        post_json(
            scope,
            &WorkerResponse::Installed {
                session_id: self.session_id,
                name: name.into(),
                summary: Box::new(summary),
            },
        );
        Ok(())
    }

    fn preview(
        &mut self,
        scope: &web_sys::DedicatedWorkerGlobalScope,
        revision: u64,
        focus: [f64; 3],
        query: MeshQuery,
        style: MeshRenderStyle,
    ) -> Result<(), String> {
        let renderer = self
            .renderer
            .as_mut()
            .ok_or_else(|| "no mesh is installed".to_string())?;
        if self.preview_focus != Some(focus)
            || self
                .pending_preview
                .as_ref()
                .is_some_and(|pending| pending.revision != revision)
        {
            renderer.defer_lod_view();
            self.pending_preview = None;
            self.preview_focus = Some(focus);
        }
        renderer.update_lod_focus(focus);
        let mut selection = None;
        let mut more = self.pending_preview.is_some();
        if self.pending_preview.is_none() {
            let update = renderer
                .prepare_lod_incremental_styled(
                    query,
                    style,
                    &BTreeSet::new(),
                    &BTreeSet::new(),
                    1.0,
                )
                .map_err(|error| error.to_string())?;
            selection = Some(update.selection.clone());
            self.preview_selection = Some(update.selection.clone());
            more = update.stats.pending_tiles != 0;
            if let Some(tile) = update.prepared.into_iter().next() {
                self.pending_preview = Some(PendingPreview {
                    revision,
                    generation: tile.generation,
                    key: tile.key,
                    lines: tile.lines,
                    offset: 0,
                });
                more = true;
            }
        }
        let selection = selection
            .or_else(|| self.preview_selection.clone())
            .unwrap_or(LodTargetSelection {
                generation: 0,
                tiles: Vec::new(),
            });
        post_json(
            scope,
            &WorkerResponse::PreviewState {
                session_id: self.session_id,
                revision,
                selection,
                more,
            },
        );
        if let Some(pending) = &mut self.pending_preview {
            let end = (pending.offset + PREVIEW_PACKET_LINES).min(pending.lines.len());
            let mut floats = Vec::with_capacity((end - pending.offset) * 9);
            for line in &pending.lines[pending.offset..end] {
                floats.extend(caso_render::mesh_line_instance(line));
            }
            pending.offset = end;
            let complete = pending.offset == pending.lines.len();
            let meta = PreviewPacketMeta {
                session_id: self.session_id,
                revision,
                generation: pending.generation,
                key: pending.key,
                complete,
            };
            post_preview_packet(scope, &meta, &floats);
            if complete {
                self.pending_preview = None;
            }
        }
        Ok(())
    }

    fn start_analysis(
        &mut self,
        scope: &web_sys::DedicatedWorkerGlobalScope,
        request_id: u64,
        query: MeshQuery,
    ) -> Result<(), String> {
        let mesh = self.mesh()?.clone();
        let service = caso_meshing::MeshQueryService::new(mesh.clone());
        let plan = service.plan(query).map_err(|error| error.to_string())?;
        let progress = QueryProgress {
            candidate_rows: plan.candidate_rows,
            candidate_batches: plan.candidate_batches(),
            ..QueryProgress::default()
        };
        let quality_metric = plan.measures.quality;
        self.analysis = Some(AnalysisState {
            request_id,
            cursor: service.cursor(plan),
            accumulator: QueryStatisticsAccumulator::with_quality_metric(
                mesh.manifest().counts.cells,
                quality_metric,
            ),
        });
        post_json(
            scope,
            &WorkerResponse::Analysis {
                session_id: self.session_id,
                request_id,
                progress,
                statistics: None,
            },
        );
        Ok(())
    }

    fn step_analysis(
        &mut self,
        scope: &web_sys::DedicatedWorkerGlobalScope,
        request_id: u64,
    ) -> Result<(), String> {
        let Some(mut analysis) = self.analysis.take() else {
            return Ok(());
        };
        if analysis.request_id != request_id {
            self.analysis = Some(analysis);
            return Ok(());
        }
        let step = analysis
            .cursor
            .step(QueryBudget::new(4_096, Duration::from_millis(4)))
            .map_err(|error| error.to_string())?;
        analysis.accumulator.extend(step.rows);
        let statistics = step
            .progress
            .complete
            .then(|| analysis.accumulator.finish(step.progress));
        let complete = statistics.is_some();
        post_json(
            scope,
            &WorkerResponse::Analysis {
                session_id: self.session_id,
                request_id,
                progress: step.progress,
                statistics,
            },
        );
        if !complete {
            self.analysis = Some(analysis);
        }
        Ok(())
    }

    fn start_audit(
        &mut self,
        scope: &web_sys::DedicatedWorkerGlobalScope,
        request_id: u64,
    ) -> Result<(), String> {
        let cursor = self.mesh()?.audit_cursor();
        let progress = cursor.progress();
        self.audit = Some(AuditState { request_id, cursor });
        post_json(
            scope,
            &WorkerResponse::Audit {
                session_id: self.session_id,
                request_id,
                progress,
                report: None,
            },
        );
        Ok(())
    }

    fn step_audit(
        &mut self,
        scope: &web_sys::DedicatedWorkerGlobalScope,
        request_id: u64,
    ) -> Result<(), String> {
        let Some(mut audit) = self.audit.take() else {
            return Ok(());
        };
        if audit.request_id != request_id {
            self.audit = Some(audit);
            return Ok(());
        }
        let step = self
            .mesh()?
            .audit_step(&mut audit.cursor, 1, &JobControl::default())
            .map_err(|error| error.to_string())?;
        post_json(
            scope,
            &WorkerResponse::Audit {
                session_id: self.session_id,
                request_id,
                progress: step.progress,
                report: step.report,
            },
        );
        if step.report.is_none() {
            self.audit = Some(audit);
        }
        Ok(())
    }

    fn convert(
        &self,
        scope: &web_sys::DedicatedWorkerGlobalScope,
        request_id: u64,
        converter_id: &str,
    ) -> Result<(), String> {
        let mut storage = MemoryStorage::new(self.cap_bytes).map_err(|error| error.to_string())?;
        let mut output = storage.begin().map_err(|error| error.to_string())?;
        convert::write_to(converter_id, self.mesh()?, &mut output)
            .map_err(|error| web_error(error.to_string(), self.cap_bytes / (1024 * 1024)))?;
        let MeshArtifact::Memory(bytes) =
            storage.publish(output).map_err(|error| error.to_string())?;
        post_blob(
            scope,
            self.session_id,
            request_id,
            bytes_blob(bytes.as_ref())?,
        );
        Ok(())
    }

    fn mesh(&self) -> Result<&Arc<MeshFile>, String> {
        self.mesh
            .as_ref()
            .ok_or_else(|| "no mesh is installed".into())
    }

    fn reset(&mut self, session_id: u64, cap_bytes: usize) {
        *self = Self {
            session_id,
            cap_bytes,
            ..Self::default()
        };
    }
}

fn validated_cap(cap_mib: u16) -> Result<usize, String> {
    if !(WEB_MIN_ARTIFACT_MIB..=WEB_MAX_ARTIFACT_MIB).contains(&cap_mib) {
        return Err(format!(
            "web mesh cap must be {WEB_MIN_ARTIFACT_MIB}–{WEB_MAX_ARTIFACT_MIB} MiB"
        ));
    }
    Ok(usize::from(cap_mib) * 1024 * 1024)
}

fn read_file(file: &web_sys::File, cap: usize) -> Result<MemoryArtifact, String> {
    let size = file.size();
    if !size.is_finite() || size < 0.0 || size > cap as f64 {
        return Err(format!(
            "mesh file is {:.1} MiB; the web artifact limit is {} MiB",
            size / 1024.0 / 1024.0,
            cap / 1024 / 1024
        ));
    }
    let size = size as usize;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| "browser could not reserve memory for the mesh file".to_string())?;
    bytes.resize(size, 0);
    let reader = web_sys::FileReaderSync::new()
        .map_err(|error| format!("could not create worker file reader: {error:?}"))?;
    for start in (0..size).step_by(FILE_READ_CHUNK_BYTES) {
        let end = (start + FILE_READ_CHUNK_BYTES).min(size);
        let blob = file
            .slice_with_f64_and_f64(start as f64, end as f64)
            .map_err(|error| format!("could not slice mesh file: {error:?}"))?;
        let buffer = reader
            .read_as_array_buffer(&blob)
            .map_err(|error| format!("could not read mesh file: {error:?}"))?;
        js_sys::Uint8Array::new(&buffer).copy_to(&mut bytes[start..end]);
    }
    Ok(MemoryArtifact::from_vec(bytes))
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

fn bytes_blob(bytes: &[u8]) -> Result<web_sys::Blob, String> {
    let parts = js_sys::Array::new();
    for chunk in bytes.chunks(FILE_READ_CHUNK_BYTES) {
        parts.push(&js_sys::Uint8Array::from(chunk));
    }
    web_sys::Blob::new_with_u8_array_sequence(parts.as_ref())
        .map_err(|error| format!("could not build download Blob: {error:?}"))
}

fn post_json(scope: &web_sys::DedicatedWorkerGlobalScope, response: &WorkerResponse) {
    if let Ok(text) = serde_json::to_string(response) {
        let _ = scope.post_message(&JsValue::from_str(&text));
    }
}

fn post_preview_packet(
    scope: &web_sys::DedicatedWorkerGlobalScope,
    meta: &PreviewPacketMeta,
    floats: &[f32],
) {
    let message = js_sys::Object::new();
    let array = js_sys::Float32Array::from(floats);
    let _ = js_sys::Reflect::set(
        &message,
        &JsValue::from_str("kind"),
        &JsValue::from_str("preview_packet"),
    );
    let _ = js_sys::Reflect::set(
        &message,
        &JsValue::from_str("meta"),
        &JsValue::from_str(&serde_json::to_string(meta).unwrap_or_default()),
    );
    let _ = js_sys::Reflect::set(&message, &JsValue::from_str("floats"), array.as_ref());
    let transfer = js_sys::Array::of1(&array.buffer());
    let _ = scope.post_message_with_transfer(message.as_ref(), transfer.as_ref());
}

fn post_blob(
    scope: &web_sys::DedicatedWorkerGlobalScope,
    session_id: u64,
    request_id: u64,
    blob: web_sys::Blob,
) {
    let message = js_sys::Object::new();
    for (key, value) in [
        ("kind", JsValue::from_str("download")),
        ("session_id", JsValue::from_f64(session_id as f64)),
        ("request_id", JsValue::from_f64(request_id as f64)),
        ("blob", blob.into()),
    ] {
        let _ = js_sys::Reflect::set(&message, &JsValue::from_str(key), &value);
    }
    let _ = scope.post_message(message.as_ref());
}

fn web_error(error: String, cap_mib: impl std::fmt::Display) -> String {
    if error.contains("configured memory mesh cap exceeded") {
        format!(
            "mesh exceeds the web {cap_mib} MiB artifact limit; increase element size or use the native build"
        )
    } else {
        error
    }
}
