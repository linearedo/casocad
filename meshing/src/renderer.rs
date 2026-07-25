use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{self, Receiver, Sender};

use arrow_array::{Array, LargeListArray, UInt64Array};
use web_time::Instant;

use crate::query::{EntityKind, MeshQuery, MeshQueryService, SelectedEntity};
use crate::schema::{element_dimension, Bounds3, RowKind};
use crate::{MeshFile, MeshResult};

pub const LOD_EXPAND_PIXELS: f32 = 256.0;
pub const LOD_COLLAPSE_PIXELS: f32 = 192.0;
const LINE_INSTANCE_BYTES: usize = 9 * std::mem::size_of::<f32>();
const FOCUS_TILE_LIMIT: usize = 4;
#[cfg(not(target_arch = "wasm32"))]
const MAX_WORKER_REQUESTS: usize = 4;

#[cfg(not(target_arch = "wasm32"))]
fn worker_generation_matches(generation: &AtomicU64, request_generation: u64) -> bool {
    request_generation == generation.load(Ordering::Acquire)
}

/// Camera state needed for spatial-node visibility and screen-space LOD.
/// The matrix follows wgpu clip coordinates: x/y in -w..w, z in 0..w.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshView {
    pub clip_from_world: [f32; 16],
    pub viewport: [u32; 2],
}

impl MeshView {
    pub fn new(clip_from_world: [f32; 16], width: u32, height: u32) -> Self {
        Self {
            clip_from_world,
            viewport: [width.max(1), height.max(1)],
        }
    }

    pub(crate) fn projected_pixels(self, bounds: Bounds3) -> Option<f32> {
        let corners = bounds_corners(bounds).map(|point| self.project(point));
        if corners.iter().all(|point| point[3] <= 0.0)
            || corners.iter().all(|point| point[0] < -point[3])
            || corners.iter().all(|point| point[0] > point[3])
            || corners.iter().all(|point| point[1] < -point[3])
            || corners.iter().all(|point| point[1] > point[3])
            || corners.iter().all(|point| point[2] < 0.0)
            || corners.iter().all(|point| point[2] > point[3])
        {
            return None;
        }
        // A node crossing the camera/near plane must be refined
        // conservatively; perspective division cannot bound it reliably.
        if corners
            .iter()
            .any(|point| point[3] <= f64::EPSILON || point[2] < 0.0)
        {
            return Some(f32::INFINITY);
        }
        let mut min = [f64::INFINITY; 2];
        let mut max = [f64::NEG_INFINITY; 2];
        for point in corners {
            for axis in 0..2 {
                let ndc = (point[axis] / point[3]).clamp(-1.0, 1.0);
                min[axis] = min[axis].min(ndc);
                max[axis] = max[axis].max(ndc);
            }
        }
        Some(
            ((max[0] - min[0]) * 0.5 * f64::from(self.viewport[0]))
                .max((max[1] - min[1]) * 0.5 * f64::from(self.viewport[1])) as f32,
        )
    }

    fn project(self, point: [f64; 3]) -> [f64; 4] {
        let matrix = self.clip_from_world.map(f64::from);
        [
            matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12],
            matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13],
            matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14],
            matrix[3] * point[0] + matrix[7] * point[1] + matrix[11] * point[2] + matrix[15],
        ]
    }
}

fn bounds_corners(bounds: Bounds3) -> [[f64; 3]; 8] {
    std::array::from_fn(|index| {
        std::array::from_fn(|axis| {
            if index & (1 << axis) == 0 {
                bounds.min[axis]
            } else {
                bounds.max[axis]
            }
        })
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RendererBudgets {
    pub decoded_bytes: usize,
    pub gpu_bytes: usize,
}

impl Default for RendererBudgets {
    fn default() -> Self {
        Self {
            decoded_bytes: 128 * 1024 * 1024,
            gpu_bytes: 256 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderLine {
    pub edge_id: u64,
    pub a: [f32; 3],
    pub b: [f32; 3],
    pub color_id: u64,
    pub opacity: f32,
    pub highlighted: bool,
    pub selected: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MeshTileDetail {
    Preview,
    Exact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeshTileKey {
    pub node_id: u64,
    pub detail: MeshTileDetail,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LodTargetSelection {
    pub generation: u64,
    pub tiles: Vec<MeshTileKey>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreparedMeshTile {
    pub generation: u64,
    pub key: MeshTileKey,
    pub lines: Arc<[RenderLine]>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MeshPreparationStats {
    pub generation: u64,
    pub selection_ms: f32,
    pub decode_ms: f32,
    pub decode_p95_ms: f32,
    pub line_build_ms: f32,
    pub decoded_tiles: usize,
    pub resident_tiles: usize,
    pub pending_tiles: usize,
    pub worker_active: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IncrementalLodPreparation {
    pub selection: LodTargetSelection,
    pub prepared: Vec<PreparedMeshTile>,
    pub stats: MeshPreparationStats,
}

#[derive(Debug, Clone)]
struct CachedTile {
    last_used: u64,
    bytes: usize,
    entities: Vec<SelectedEntity>,
}

#[derive(Debug, Clone)]
struct LineTile {
    last_used: u64,
    bytes: usize,
    lines: Arc<[RenderLine]>,
}

#[derive(Debug)]
struct DecodedTile {
    entities: Vec<SelectedEntity>,
    lines: Arc<[RenderLine]>,
    decode_ms: Option<f32>,
    line_build_ms: f32,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
struct DecodeRequest {
    generation: u64,
    key: MeshTileKey,
    entities: Option<Vec<SelectedEntity>>,
    query: MeshQuery,
    selected_ids: BTreeSet<u64>,
    highlighted_ids: BTreeSet<u64>,
    opacity: f32,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
struct DecodeResponse {
    generation: u64,
    key: MeshTileKey,
    result: Option<MeshResult<DecodedTile>>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
struct DecodeWorker {
    requests: Sender<DecodeRequest>,
    responses: Receiver<DecodeResponse>,
}

#[cfg(not(target_arch = "wasm32"))]
impl DecodeWorker {
    fn new(file: Arc<MeshFile>, generation: Arc<AtomicU64>) -> std::io::Result<Self> {
        let (request_tx, request_rx) = mpsc::channel::<DecodeRequest>();
        let (response_tx, response_rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("casocad-mesh-preview".into())
            .spawn(move || {
                let service = MeshQueryService::new(file);
                while let Ok(request) = request_rx.recv() {
                    let response = if worker_generation_matches(&generation, request.generation) {
                        run_decode_request(&service, request)
                    } else {
                        DecodeResponse {
                            generation: request.generation,
                            key: request.key,
                            result: None,
                        }
                    };
                    if response_tx.send(response).is_err() {
                        break;
                    }
                }
            })?;
        Ok(Self {
            requests: request_tx,
            responses: response_rx,
        })
    }
}

#[derive(Debug)]
pub struct MeshRendererCache {
    service: MeshQueryService,
    budgets: RendererBudgets,
    tick: u64,
    decoded_used: usize,
    gpu_used: usize,
    decoded: BTreeMap<MeshTileKey, CachedTile>,
    gpu: BTreeMap<MeshTileKey, LineTile>,
    refined: BTreeSet<u64>,
    generation: u64,
    target: BTreeSet<MeshTileKey>,
    pending: VecDeque<MeshTileKey>,
    announced: BTreeSet<MeshTileKey>,
    focus: Option<[f64; 3]>,
    query: Option<MeshQuery>,
    selected_ids: BTreeSet<u64>,
    highlighted_ids: BTreeSet<u64>,
    opacity: f32,
    selection_ms: f32,
    decode_ms: f32,
    line_build_ms: f32,
    decode_samples_ms: VecDeque<f32>,
    #[cfg(not(target_arch = "wasm32"))]
    worker: Option<DecodeWorker>,
    #[cfg(not(target_arch = "wasm32"))]
    worker_in_flight: BTreeSet<(u64, MeshTileKey)>,
    #[cfg(not(target_arch = "wasm32"))]
    worker_generation: Arc<AtomicU64>,
    #[cfg(not(target_arch = "wasm32"))]
    worker_retry_available: bool,
}

impl MeshRendererCache {
    pub fn new(file: Arc<MeshFile>, budgets: RendererBudgets) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let worker_generation = Arc::new(AtomicU64::new(0));
        #[cfg(not(target_arch = "wasm32"))]
        let worker = DecodeWorker::new(file.clone(), worker_generation.clone()).ok();
        Self {
            service: MeshQueryService::new(file),
            budgets,
            tick: 0,
            decoded_used: 0,
            gpu_used: 0,
            decoded: BTreeMap::new(),
            gpu: BTreeMap::new(),
            refined: BTreeSet::new(),
            generation: 0,
            target: BTreeSet::new(),
            pending: VecDeque::new(),
            announced: BTreeSet::new(),
            focus: None,
            query: None,
            selected_ids: BTreeSet::new(),
            highlighted_ids: BTreeSet::new(),
            opacity: 1.0,
            selection_ms: 0.0,
            decode_ms: 0.0,
            line_build_ms: 0.0,
            decode_samples_ms: VecDeque::new(),
            #[cfg(not(target_arch = "wasm32"))]
            worker,
            #[cfg(not(target_arch = "wasm32"))]
            worker_in_flight: BTreeSet::new(),
            #[cfg(not(target_arch = "wasm32"))]
            worker_generation,
            #[cfg(not(target_arch = "wasm32"))]
            worker_retry_available: true,
        }
    }

    /// Update the camera-dependent octree selection. The returned generation
    /// changes only when the visible node/detail target changes.
    pub fn update_lod_view(&mut self, view: MeshView) -> Option<LodTargetSelection> {
        let started = Instant::now();
        self.focus = None;
        let selection = self
            .service
            .mesh_file()
            .select_lod_nodes(
                view,
                LOD_EXPAND_PIXELS,
                LOD_COLLAPSE_PIXELS,
                &mut self.refined,
            )
            .into_iter()
            .map(|(node_id, exact)| MeshTileKey {
                node_id,
                detail: if exact {
                    MeshTileDetail::Exact
                } else {
                    MeshTileDetail::Preview
                },
            })
            .collect();
        self.selection_ms = elapsed_ms(started);
        self.set_target(selection)
    }

    /// Select a bounded exact neighborhood around the camera focus. Rotation,
    /// zoom, and viewport size are intentionally absent from this key.
    pub fn update_lod_focus(&mut self, focus: [f64; 3]) -> Option<LodTargetSelection> {
        if self.focus == Some(focus) {
            return None;
        }
        self.focus = Some(focus);
        let started = Instant::now();
        self.refined.clear();
        let selection = self
            .service
            .mesh_file()
            .nearest_leaf_nodes(focus, FOCUS_TILE_LIMIT)
            .into_iter()
            .map(|node_id| MeshTileKey {
                node_id,
                detail: MeshTileDetail::Exact,
            })
            .collect();
        self.selection_ms = elapsed_ms(started);
        self.set_target(selection)
    }

    pub fn clear_lod_view(&mut self) -> Option<LodTargetSelection> {
        self.focus = None;
        self.refined.clear();
        self.set_target(BTreeSet::new())
    }

    /// Invalidate preparation for a camera view that will be superseded
    /// before selection resumes. Decoded entities and line caches stay hot.
    pub fn defer_lod_view(&mut self) {
        self.advance_generation();
        self.reset_pending();
    }

    fn set_target(&mut self, target: BTreeSet<MeshTileKey>) -> Option<LodTargetSelection> {
        if target == self.target {
            return None;
        }
        self.advance_generation();
        self.target = target;
        self.reset_pending();
        Some(self.selection())
    }

    /// Prepare missing spatial tiles. Cached target tiles are returned by Arc
    /// so a new GPU generation can reuse them without decode.
    pub fn prepare_lod_incremental(
        &mut self,
        query: MeshQuery,
        selected_ids: &BTreeSet<u64>,
        highlighted_ids: &BTreeSet<u64>,
        opacity: f32,
    ) -> MeshResult<IncrementalLodPreparation> {
        self.tick = self.tick.wrapping_add(1);
        let opacity = opacity.clamp(0.0, 1.0);
        let query_changed = self.query.as_ref() != Some(&query);
        if query_changed
            || &self.selected_ids != selected_ids
            || &self.highlighted_ids != highlighted_ids
            || self.opacity != opacity
        {
            self.query = Some(query.clone());
            self.selected_ids = selected_ids.clone();
            self.highlighted_ids = highlighted_ids.clone();
            self.opacity = opacity;
            self.advance_generation();
            if query_changed {
                self.decoded.clear();
                self.decoded_used = 0;
            }
            self.gpu.clear();
            self.gpu_used = 0;
            self.reset_pending();
        }

        let mut prepared = Vec::new();
        for key in self.target.iter().copied() {
            if self.announced.insert(key) {
                if let Some(tile) = self.gpu.get_mut(&key) {
                    tile.last_used = self.tick;
                    prepared.push(PreparedMeshTile {
                        generation: self.generation,
                        key,
                        lines: tile.lines.clone(),
                    });
                }
            }
        }

        self.decode_ms = 0.0;
        self.line_build_ms = 0.0;

        #[cfg(not(target_arch = "wasm32"))]
        let (responses, worker_failed) = self.drain_worker_responses();
        #[cfg(not(target_arch = "wasm32"))]
        for response in responses {
            if response.generation == self.generation && self.target.contains(&response.key) {
                if let Some(result) = response.result {
                    let tile = match result {
                        Ok(tile) => tile,
                        Err(error) => {
                            if worker_failed {
                                self.reset_pending();
                            }
                            return Err(error);
                        }
                    };
                    self.accept_tile(response.key, tile, &mut prepared);
                }
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        if worker_failed {
            self.reset_pending();
        }

        #[cfg(not(target_arch = "wasm32"))]
        while self.worker_in_flight.len() < MAX_WORKER_REQUESTS {
            let Some(key) = self.pending.pop_front() else {
                break;
            };
            let entities = self.take_decoded(key);
            let request = DecodeRequest {
                generation: self.generation,
                key,
                entities,
                query: query.clone(),
                selected_ids: selected_ids.clone(),
                highlighted_ids: highlighted_ids.clone(),
                opacity,
            };
            if let Err(request) = self.send_worker_request(request) {
                let response = run_decode_request(&self.service, *request);
                let tile = response
                    .result
                    .expect("synchronous decode cannot be canceled")?;
                self.accept_tile(response.key, tile, &mut prepared);
                self.pending.retain(|pending| *pending != response.key);
                break;
            }
        }

        #[cfg(target_arch = "wasm32")]
        if let Some(key) = self.pending.front().copied() {
            let entities = self.take_decoded(key);
            self.pending.pop_front();
            let tile = if let Some(entities) = entities {
                build_decoded_tile(entities, selected_ids, highlighted_ids, opacity, None)
            } else {
                decode_and_build(
                    &self.service,
                    key,
                    &query,
                    selected_ids,
                    highlighted_ids,
                    opacity,
                )?
            };
            self.accept_tile(key, tile, &mut prepared);
        }

        self.evict_decoded();
        self.evict_gpu();
        Ok(IncrementalLodPreparation {
            selection: self.selection(),
            prepared,
            stats: self.stats(),
        })
    }

    fn advance_generation(&mut self) {
        self.generation = self
            .generation
            .checked_add(1)
            .expect("mesh preview generation overflowed");
        #[cfg(not(target_arch = "wasm32"))]
        self.worker_generation
            .store(self.generation, Ordering::Release);
    }

    pub fn decoded_bytes(&self) -> usize {
        self.decoded_used
    }

    pub fn gpu_bytes(&self) -> usize {
        self.gpu_used
    }

    pub fn stats(&self) -> MeshPreparationStats {
        MeshPreparationStats {
            generation: self.generation,
            selection_ms: self.selection_ms,
            decode_ms: self.decode_ms,
            decode_p95_ms: percentile_95(&self.decode_samples_ms),
            line_build_ms: self.line_build_ms,
            decoded_tiles: self.decoded.len(),
            resident_tiles: self.gpu.len(),
            pending_tiles: self
                .target
                .iter()
                .filter(|key| !self.gpu.contains_key(key))
                .count(),
            worker_active: {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.worker.is_some()
                }
                #[cfg(target_arch = "wasm32")]
                {
                    false
                }
            },
        }
    }

    pub fn selection(&self) -> LodTargetSelection {
        LodTargetSelection {
            generation: self.generation,
            tiles: self.target.iter().copied().collect(),
        }
    }

    fn reset_pending(&mut self) {
        self.pending = self
            .target
            .iter()
            .filter(|key| !self.gpu.contains_key(key))
            .copied()
            .collect();
        self.announced.clear();
    }

    fn insert_decoded(&mut self, key: MeshTileKey, entities: Vec<SelectedEntity>) {
        let bytes = decoded_bytes(&entities);
        if let Some(old) = self.decoded.remove(&key) {
            self.decoded_used = self.decoded_used.saturating_sub(old.bytes);
        }
        self.decoded.insert(
            key,
            CachedTile {
                last_used: self.tick,
                bytes,
                entities,
            },
        );
        self.decoded_used += bytes;
    }

    fn take_decoded(&mut self, key: MeshTileKey) -> Option<Vec<SelectedEntity>> {
        self.decoded.remove(&key).map(|tile| {
            self.decoded_used = self.decoded_used.saturating_sub(tile.bytes);
            tile.entities
        })
    }

    fn insert_lines(&mut self, key: MeshTileKey, lines: Arc<[RenderLine]>) {
        let bytes = lines.len() * LINE_INSTANCE_BYTES;
        if let Some(old) = self.gpu.remove(&key) {
            self.gpu_used = self.gpu_used.saturating_sub(old.bytes);
        }
        self.gpu.insert(
            key,
            LineTile {
                last_used: self.tick,
                bytes,
                lines,
            },
        );
        self.gpu_used += bytes;
    }

    fn record_decode_sample(&mut self, sample: f32) {
        const SAMPLE_COUNT: usize = 128;
        if self.decode_samples_ms.len() == SAMPLE_COUNT {
            self.decode_samples_ms.pop_front();
        }
        self.decode_samples_ms.push_back(sample);
    }

    fn accept_tile(
        &mut self,
        key: MeshTileKey,
        tile: DecodedTile,
        prepared: &mut Vec<PreparedMeshTile>,
    ) {
        self.decode_ms += tile.decode_ms.unwrap_or(0.0);
        self.line_build_ms += tile.line_build_ms;
        if let Some(decode_ms) = tile.decode_ms {
            self.record_decode_sample(decode_ms);
        }
        self.insert_decoded(key, tile.entities);
        self.insert_lines(key, tile.lines.clone());
        self.announced.insert(key);
        prepared.push(PreparedMeshTile {
            generation: self.generation,
            key,
            lines: tile.lines,
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn drain_worker_responses(&mut self) -> (Vec<DecodeResponse>, bool) {
        use std::sync::mpsc::TryRecvError;

        let mut responses = Vec::new();
        let mut worker_failed = false;
        loop {
            let Some(worker) = &self.worker else {
                break;
            };
            match worker.responses.try_recv() {
                Ok(response) => {
                    self.worker_in_flight
                        .remove(&(response.generation, response.key));
                    responses.push(response);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.worker = None;
                    self.worker_in_flight.clear();
                    worker_failed = true;
                    break;
                }
            }
        }
        (responses, worker_failed)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn send_worker_request(
        &mut self,
        mut request: DecodeRequest,
    ) -> Result<(), Box<DecodeRequest>> {
        loop {
            if self.worker.is_none() {
                if !self.worker_retry_available {
                    return Err(Box::new(request));
                }
                self.worker_retry_available = false;
                self.worker = DecodeWorker::new(
                    self.service.mesh_file().clone(),
                    self.worker_generation.clone(),
                )
                .ok();
                if self.worker.is_none() {
                    return Err(Box::new(request));
                }
            }
            let generation = request.generation;
            let key = request.key;
            match self
                .worker
                .as_ref()
                .expect("worker was created above")
                .requests
                .send(request)
            {
                Ok(()) => {
                    self.pending.retain(|pending| *pending != key);
                    self.worker_in_flight.insert((generation, key));
                    return Ok(());
                }
                Err(error) => {
                    request = error.0;
                    self.worker = None;
                    self.worker_in_flight.clear();
                    self.reset_pending();
                }
            }
        }
    }

    fn protected(&self, key: &MeshTileKey) -> bool {
        self.target.contains(key) || self.pending.contains(key)
    }

    fn evict_decoded(&mut self) {
        while self.decoded_used > self.budgets.decoded_bytes {
            let candidate = self
                .decoded
                .iter()
                .filter(|(tile, _)| !self.protected(tile))
                .min_by_key(|(_, value)| value.last_used)
                .map(|(tile, _)| *tile);
            let Some(tile) = candidate else {
                break;
            };
            if let Some(entry) = self.decoded.remove(&tile) {
                self.decoded_used = self.decoded_used.saturating_sub(entry.bytes);
            }
        }
    }

    fn evict_gpu(&mut self) {
        while self.gpu_used > self.budgets.gpu_bytes {
            let candidate = self
                .gpu
                .iter()
                .filter(|(tile, _)| !self.protected(tile))
                .min_by_key(|(_, value)| value.last_used)
                .map(|(tile, _)| *tile);
            let Some(tile) = candidate else {
                break;
            };
            if let Some(entry) = self.gpu.remove(&tile) {
                self.gpu_used = self.gpu_used.saturating_sub(entry.bytes);
            }
        }
    }
}

fn decode_and_build(
    service: &MeshQueryService,
    key: MeshTileKey,
    query: &MeshQuery,
    selected_ids: &BTreeSet<u64>,
    highlighted_ids: &BTreeSet<u64>,
    opacity: f32,
) -> MeshResult<DecodedTile> {
    let started = Instant::now();
    let entities = load_tile(service, key, query)?;
    let decode_ms = elapsed_ms(started);
    Ok(build_decoded_tile(
        entities,
        selected_ids,
        highlighted_ids,
        opacity,
        Some(decode_ms),
    ))
}

fn build_decoded_tile(
    entities: Vec<SelectedEntity>,
    selected_ids: &BTreeSet<u64>,
    highlighted_ids: &BTreeSet<u64>,
    opacity: f32,
    decode_ms: Option<f32>,
) -> DecodedTile {
    let started = Instant::now();
    let lines = build_lines(&entities, selected_ids, highlighted_ids, opacity).into();
    DecodedTile {
        entities,
        lines,
        decode_ms,
        line_build_ms: elapsed_ms(started),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_decode_request(service: &MeshQueryService, request: DecodeRequest) -> DecodeResponse {
    let result = if let Some(entities) = request.entities {
        Ok(build_decoded_tile(
            entities,
            &request.selected_ids,
            &request.highlighted_ids,
            request.opacity,
            None,
        ))
    } else {
        decode_and_build(
            service,
            request.key,
            &request.query,
            &request.selected_ids,
            &request.highlighted_ids,
            request.opacity,
        )
    };
    DecodeResponse {
        generation: request.generation,
        key: request.key,
        result: Some(result),
    }
}

fn load_tile(
    service: &MeshQueryService,
    key: MeshTileKey,
    query: &MeshQuery,
) -> MeshResult<Vec<SelectedEntity>> {
    match key.detail {
        MeshTileDetail::Preview => load_preview(service.mesh_file(), key.node_id, query),
        MeshTileDetail::Exact => {
            let nodes = BTreeSet::from([key.node_id]);
            Ok(service
                .execute_selected_nodes(query.clone(), &nodes)?
                .render_tiles
                .into_iter()
                .find(|tile| tile.tile_id == key.node_id)
                .map_or_else(Vec::new, |tile| tile.entities))
        }
    }
}

fn decoded_bytes(entities: &[SelectedEntity]) -> usize {
    entities
        .iter()
        .map(|entity| {
            std::mem::size_of::<SelectedEntity>()
                + entity.points.len() * std::mem::size_of::<[f64; 3]>()
                + (entity.point_ids.len()
                    + entity.edge_ids.len()
                    + entity.face_ids.len()
                    + entity.tag_ids.len())
                    * std::mem::size_of::<u64>()
        })
        .sum()
}

fn elapsed_ms(started: Instant) -> f32 {
    started.elapsed().as_secs_f32() * 1_000.0
}

fn percentile_95(samples: &VecDeque<f32>) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.iter().copied().collect::<Vec<_>>();
    sorted.sort_by(f32::total_cmp);
    sorted[(sorted.len() * 95).div_ceil(100).saturating_sub(1)]
}

fn load_preview(file: &MeshFile, node: u64, query: &MeshQuery) -> MeshResult<Vec<SelectedEntity>> {
    let mut points = BTreeMap::new();
    for entry in file.tile_batches(node, RowKind::PreviewPoint) {
        let batch = file.batch_view(entry.batch_index)?;
        let ids = batch.u64s("entity_id")?;
        let x = batch.f64s("x")?;
        let y = batch.f64s("y")?;
        let z = batch.f64s("z")?;
        for row in 0..batch.len() {
            points.insert(ids.value(row), [x.value(row), y.value(row), z.value(row)]);
        }
    }
    let mut result = Vec::new();
    for entry in file.tile_batches(node, RowKind::PreviewElement) {
        let batch = file.batch_view(entry.batch_index)?;
        let ids = batch.u64s("entity_id")?;
        let types = batch.strings("element_type")?;
        let connectivity = batch.lists("point_ids")?;
        let tags = batch.lists("tag_ids")?;
        let zones = batch.u64s("zone_id")?;
        let boundary = batch.bools("boundary")?;
        for row in 0..batch.len() {
            let element_type = types.value(row);
            let is_boundary = !boundary.is_null(row) && boundary.value(row);
            let dimension = element_dimension(element_type).unwrap_or(0);
            let wanted = match query.entity_kind {
                EntityKind::Point => false,
                EntityKind::Edge => is_boundary && dimension == 1,
                EntityKind::Face => is_boundary && dimension == 2,
                EntityKind::Cell => !is_boundary && dimension == file.manifest().dimension,
            };
            if !wanted
                || query
                    .element_type
                    .as_ref()
                    .is_some_and(|wanted| wanted != element_type)
            {
                continue;
            }
            let point_ids = preview_list(connectivity, row)?;
            let geometry = point_ids
                .iter()
                .map(|id| points.get(id).copied())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    crate::MeshError::InvalidFile(
                        "preview element references a missing preview point".into(),
                    )
                })?;
            let tag_ids = preview_list(tags, row)?;
            let zone_id = (!zones.is_null(row)).then(|| zones.value(row));
            if geometry.iter().any(|point| {
                !query.x.contains(point[0])
                    || !query.y.contains(point[1])
                    || !query.z.contains(point[2])
            }) || (!query.tag_ids.is_empty()
                && query.tag_ids.iter().any(|tag| !tag_ids.contains(tag)))
                || (!query.zone_ids.is_empty()
                    && zone_id.is_none_or(|zone| !query.zone_ids.contains(&zone)))
            {
                continue;
            }
            let quality = query.quality.and_then(|filter| {
                crate::quality::quality_score(element_type, &geometry, filter.metric)
                    .filter(|score| filter.interval.contains(*score))
            });
            if query.quality.is_some() && quality.is_none() {
                continue;
            }
            result.push(SelectedEntity {
                id: ids.value(row),
                kind: query.entity_kind,
                tile_id: node,
                element_type: element_type.into(),
                point_ids,
                points: geometry,
                edge_ids: Vec::new(),
                face_ids: Vec::new(),
                tag_ids,
                zone_id,
                source_id: None,
                source_object_id: None,
                boundary: is_boundary,
                boundary_distance: None,
                quality,
            });
            if result.len() >= query.display_limit {
                return Ok(result);
            }
        }
    }
    Ok(result)
}

fn preview_list(array: &LargeListArray, row: usize) -> MeshResult<Vec<u64>> {
    let values = array.value(row);
    values
        .as_any()
        .downcast_ref::<UInt64Array>()
        .map(|values| values.values().to_vec())
        .ok_or_else(|| crate::MeshError::InvalidFile("preview list must contain u64".into()))
}

fn build_lines(
    entities: &[SelectedEntity],
    selected_ids: &BTreeSet<u64>,
    highlighted_ids: &BTreeSet<u64>,
    opacity: f32,
) -> Vec<RenderLine> {
    let mut lines = BTreeMap::<u64, RenderLine>::new();
    let mut volume_lines = BTreeMap::<(u64, u64), RenderLine>::new();
    for entity in entities {
        let color_id = entity
            .tag_ids
            .first()
            .copied()
            .or(entity.source_object_id)
            .or(entity.zone_id)
            .unwrap_or(0);
        let selected = selected_ids.contains(&entity.id);
        let highlighted = highlighted_ids.contains(&entity.id);
        if let Some(edges) = volume_edges(&entity.element_type) {
            for &(a, b) in edges {
                let Some((&a_id, &b_id)) = entity.point_ids.get(a).zip(entity.point_ids.get(b))
                else {
                    continue;
                };
                let Some((&a_point, &b_point)) = entity.points.get(a).zip(entity.points.get(b))
                else {
                    continue;
                };
                let key = if a_id < b_id {
                    (a_id, b_id)
                } else {
                    (b_id, a_id)
                };
                let candidate = line(
                    synthetic_volume_edge_id(key),
                    a_point,
                    b_point,
                    color_id,
                    opacity,
                    highlighted,
                    selected,
                );
                volume_lines
                    .entry(key)
                    .and_modify(|line| {
                        line.highlighted |= highlighted;
                        line.selected |= selected;
                    })
                    .or_insert(candidate);
            }
            continue;
        }
        match entity.element_type.as_str() {
            "edge2" | "edge3" if entity.points.len() >= 2 => {
                lines.entry(entity.id).or_insert_with(|| {
                    line(
                        entity.id,
                        entity.points[0],
                        entity.points[1],
                        color_id,
                        opacity,
                        highlighted,
                        selected,
                    )
                });
            }
            "tri3" | "tri6" if entity.points.len() >= 3 => {
                for (index, (a, b)) in [(0, 1), (1, 2), (2, 0)].into_iter().enumerate() {
                    let edge_id = entity
                        .edge_ids
                        .get(index)
                        .copied()
                        .unwrap_or_else(|| synthetic_edge_id(entity.id, index));
                    let candidate = line(
                        edge_id,
                        entity.points[a],
                        entity.points[b],
                        color_id,
                        opacity,
                        highlighted,
                        selected,
                    );
                    lines
                        .entry(edge_id)
                        .and_modify(|line| {
                            line.highlighted |= highlighted;
                            line.selected |= selected;
                        })
                        .or_insert(candidate);
                }
            }
            "quad4" | "quad8" | "quad9" if entity.points.len() >= 4 => {
                for (index, (a, b)) in [(0, 1), (1, 2), (2, 3), (3, 0)].into_iter().enumerate() {
                    let edge_id = entity
                        .edge_ids
                        .get(index)
                        .copied()
                        .unwrap_or_else(|| synthetic_edge_id(entity.id, index));
                    let candidate = line(
                        edge_id,
                        entity.points[a],
                        entity.points[b],
                        color_id,
                        opacity,
                        highlighted,
                        selected,
                    );
                    lines
                        .entry(edge_id)
                        .and_modify(|line| {
                            line.highlighted |= highlighted;
                            line.selected |= selected;
                        })
                        .or_insert(candidate);
                }
            }
            _ => {}
        }
    }
    let mut result = lines.into_values().collect::<Vec<_>>();
    result.extend(volume_lines.into_values());
    result
}

fn volume_edges(element_type: &str) -> Option<&'static [(usize, usize)]> {
    const TET: &[(usize, usize)] = &[(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)];
    const PYRAMID: &[(usize, usize)] = &[
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
    ];
    const PRISM: &[(usize, usize)] = &[
        (0, 1),
        (1, 2),
        (2, 0),
        (3, 4),
        (4, 5),
        (5, 3),
        (0, 3),
        (1, 4),
        (2, 5),
    ];
    const HEX: &[(usize, usize)] = &[
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    match element_type {
        "tet4" | "tet10" => Some(TET),
        "pyramid5" | "pyramid13" => Some(PYRAMID),
        "prism6" | "prism15" => Some(PRISM),
        "hex8" | "hex20" | "hex27" => Some(HEX),
        _ => None,
    }
}

fn line(
    edge_id: u64,
    a: [f64; 3],
    b: [f64; 3],
    color_id: u64,
    opacity: f32,
    highlighted: bool,
    selected: bool,
) -> RenderLine {
    RenderLine {
        edge_id,
        a: a.map(|value| value as f32),
        b: b.map(|value| value as f32),
        color_id,
        opacity,
        highlighted,
        selected,
    }
}

fn synthetic_edge_id(entity_id: u64, local_edge: usize) -> u64 {
    entity_id.rotate_left(17) ^ local_edge as u64 ^ (1u64 << 63)
}

fn synthetic_volume_edge_id((a, b): (u64, u64)) -> u64 {
    a.rotate_left(17) ^ b.rotate_right(13) ^ (1u64 << 62)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];

    #[test]
    fn mesh_view_culls_outside_nodes_and_sizes_visible_nodes() {
        let view = MeshView::new(IDENTITY, 200, 100);
        let visible = Bounds3 {
            min: [-0.5, -0.5, 0.25],
            max: [0.5, 0.5, 0.75],
        };
        assert_eq!(view.projected_pixels(visible), Some(100.0));

        let outside = Bounds3 {
            min: [2.0, -0.5, 0.25],
            max: [3.0, 0.5, 0.75],
        };
        assert_eq!(view.projected_pixels(outside), None);
    }

    #[test]
    fn mesh_view_refines_near_plane_intersections_conservatively() {
        let view = MeshView::new(IDENTITY, 200, 100);
        let crossing = Bounds3 {
            min: [-0.5, -0.5, -0.25],
            max: [0.5, 0.5, 0.25],
        };
        assert_eq!(view.projected_pixels(crossing), Some(f32::INFINITY));

        let behind = Bounds3 {
            min: [-0.5, -0.5, -0.75],
            max: [0.5, 0.5, -0.25],
        };
        assert_eq!(view.projected_pixels(behind), None);
    }

    fn volume_entity(id: u64, element_type: &str, point_ids: Vec<u64>) -> SelectedEntity {
        let points = point_ids
            .iter()
            .map(|point_id| [*point_id as f64, 0.0, 0.0])
            .collect();
        SelectedEntity {
            id,
            kind: EntityKind::Cell,
            tile_id: 1,
            element_type: element_type.into(),
            point_ids,
            points,
            edge_ids: Vec::new(),
            face_ids: Vec::new(),
            tag_ids: Vec::new(),
            zone_id: None,
            source_id: None,
            source_object_id: None,
            boundary: false,
            boundary_distance: None,
            quality: None,
        }
    }

    #[test]
    fn volume_families_draw_only_corner_edges() {
        for (element_type, point_count, corner_count, edge_count) in [
            ("tet4", 4, 4, 6),
            ("tet10", 10, 4, 6),
            ("pyramid5", 5, 5, 8),
            ("pyramid13", 13, 5, 8),
            ("prism6", 6, 6, 9),
            ("prism15", 15, 6, 9),
            ("hex8", 8, 8, 12),
            ("hex20", 20, 8, 12),
            ("hex27", 27, 8, 12),
        ] {
            let point_ids = (100..100 + point_count).collect::<Vec<_>>();
            let lines = build_lines(
                &[volume_entity(1, element_type, point_ids)],
                &BTreeSet::new(),
                &BTreeSet::new(),
                1.0,
            );
            assert_eq!(lines.len(), edge_count, "{element_type}");
            assert!(
                lines.iter().all(|line| {
                    line.a[0] < (100 + corner_count) as f32
                        && line.b[0] < (100 + corner_count) as f32
                }),
                "{element_type} used a higher-order point"
            );
        }
    }

    #[test]
    fn shared_volume_edges_are_deduplicated_and_combine_selection_state() {
        let first = volume_entity(10, "tet4", vec![1, 2, 3, 4]);
        let second = volume_entity(20, "tet4", vec![1, 2, 5, 6]);
        let lines = build_lines(
            &[first, second],
            &BTreeSet::from([10]),
            &BTreeSet::from([20]),
            1.0,
        );
        assert_eq!(lines.len(), 11);
        let shared = lines
            .iter()
            .find(|line| {
                [line.a[0], line.b[0]] == [1.0, 2.0] || [line.a[0], line.b[0]] == [2.0, 1.0]
            })
            .expect("shared edge");
        assert!(shared.selected);
        assert!(shared.highlighted);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn test_renderer_cache() -> MeshRendererCache {
        use std::sync::OnceLock;

        use caso_kernel::meshing::meshable_domains_from_document;
        use caso_kernel::roles::DomainKind;
        use caso_kernel::scene::SceneDocument;
        use caso_kernel::vec3::vec3;

        use crate::{
            ControlSet, GenerationLimits, JobControl, MemoryStorage, MeshArtifact, MeshingRequest,
        };

        static FILE: OnceLock<Arc<MeshFile>> = OnceLock::new();
        let file = FILE
            .get_or_init(|| {
                let mut document = SceneDocument::new();
                let rectangle = document
                    .add_primitive_from_drag(
                        "rectangle",
                        vec3(0.0, 0.0, 0.0),
                        vec3(2.0, 1.0, 0.0),
                        1.0,
                    )
                    .expect("rectangle");
                document
                    .set_domain_root(rectangle, DomainKind::Fluid)
                    .expect("domain");
                let output = crate::run_meshing(
                    MeshingRequest {
                        domains: meshable_domains_from_document(&document).expect("meshable"),
                        algorithm_id: "uniform_2d".into(),
                        element_min_size: 0.1,
                        element_max_size: 0.25,
                        controls: ControlSet::default(),
                        limits: GenerationLimits::default(),
                        job_control: JobControl::default(),
                    },
                    MemoryStorage::new(16 * 1024 * 1024).expect("storage"),
                )
                .expect("mesh");
                let MeshArtifact::Memory(bytes) = output.artifact else {
                    panic!("expected memory mesh");
                };
                Arc::new(MeshFile::from_memory(bytes).expect("test mesh"))
            })
            .clone();
        let mut renderer = MeshRendererCache::new(file, RendererBudgets::default());
        renderer.generation = 1;
        renderer.worker_generation.store(1, Ordering::Release);
        renderer.query = Some(MeshQuery::default());
        renderer
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn native_worker_dispatch_is_bounded_to_four_requests() {
        let mut renderer = test_renderer_cache();
        let keys = (1..=6)
            .map(|node_id| MeshTileKey {
                node_id,
                detail: MeshTileDetail::Preview,
            })
            .collect::<BTreeSet<_>>();
        renderer.target = keys.clone();
        renderer.pending = keys.iter().copied().collect();

        let (request_tx, request_rx) = mpsc::channel();
        let (_response_tx, response_rx) = mpsc::channel();
        renderer.worker = Some(DecodeWorker {
            requests: request_tx,
            responses: response_rx,
        });
        renderer
            .prepare_lod_incremental(
                MeshQuery::default(),
                &BTreeSet::new(),
                &BTreeSet::new(),
                1.0,
            )
            .expect("dispatch");

        assert_eq!(renderer.worker_in_flight.len(), MAX_WORKER_REQUESTS);
        assert_eq!(request_rx.try_iter().count(), MAX_WORKER_REQUESTS);
        assert_eq!(renderer.pending.len(), 2);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn native_worker_drains_all_ready_responses_together() {
        let mut renderer = test_renderer_cache();
        let keys = [11, 12].map(|node_id| MeshTileKey {
            node_id,
            detail: MeshTileDetail::Preview,
        });
        renderer.target = keys.into_iter().collect();

        let (request_tx, _request_rx) = mpsc::channel();
        let (response_tx, response_rx) = mpsc::channel();
        renderer.worker = Some(DecodeWorker {
            requests: request_tx,
            responses: response_rx,
        });
        for (key, decode_ms, line_build_ms) in [(keys[0], 1.0, 2.0), (keys[1], 3.0, 4.0)] {
            renderer.worker_in_flight.insert((1, key));
            response_tx
                .send(DecodeResponse {
                    generation: 1,
                    key,
                    result: Some(Ok(DecodedTile {
                        entities: Vec::new(),
                        lines: Arc::from([]),
                        decode_ms: Some(decode_ms),
                        line_build_ms,
                    })),
                })
                .expect("ready response");
        }
        drop(response_tx);

        let update = renderer
            .prepare_lod_incremental(
                MeshQuery::default(),
                &BTreeSet::new(),
                &BTreeSet::new(),
                1.0,
            )
            .expect("drain");
        assert_eq!(update.prepared.len(), 2);
        assert_eq!(update.stats.decode_ms, 4.0);
        assert_eq!(update.stats.line_build_ms, 6.0);
        assert!(renderer.worker_in_flight.is_empty());
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn stale_worker_generation_is_rejected_before_decode() {
        let generation = AtomicU64::new(7);
        assert!(worker_generation_matches(&generation, 7));
        generation.store(8, Ordering::Release);
        assert!(!worker_generation_matches(&generation, 7));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn worker_failure_falls_back_and_keeps_remaining_requests_pending() {
        let mut renderer = test_renderer_cache();
        let root = MeshTileKey {
            node_id: renderer.service.mesh_file().manifest().spatial_root,
            detail: MeshTileDetail::Preview,
        };
        let later = MeshTileKey {
            node_id: u64::MAX,
            detail: MeshTileDetail::Preview,
        };
        renderer.target = [root, later].into_iter().collect();
        renderer.pending = [root, later].into_iter().collect();

        let (request_tx, request_rx) = mpsc::channel();
        drop(request_rx);
        let (_response_tx, response_rx) = mpsc::channel();
        renderer.worker = Some(DecodeWorker {
            requests: request_tx,
            responses: response_rx,
        });
        renderer.worker_retry_available = false;

        let update = renderer
            .prepare_lod_incremental(
                MeshQuery::default(),
                &BTreeSet::new(),
                &BTreeSet::new(),
                1.0,
            )
            .expect("synchronous fallback");
        assert_eq!(update.prepared.len(), 1);
        assert!(renderer.gpu.contains_key(&root));
        assert_eq!(renderer.pending, VecDeque::from([later]));
    }
}
