use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use arrow_array::{Array, LargeListArray, UInt64Array};
use serde::{Deserialize, Serialize};
use web_time::Instant;

use crate::error::{MeshError, MeshResult};
use crate::quality::{
    corner_count, polyhedron_quality_score, quality_score, quality_score_exact, side_indices,
    QualityMetric,
};
use crate::schema::{element_dimension, Bounds3, RowKind};
use crate::{BatchView, MeshFile};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    pub min: f64,
    pub max: f64,
}

impl Interval {
    pub const ALL: Self = Self {
        min: f64::NEG_INFINITY,
        max: f64::INFINITY,
    };

    pub const fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }

    pub fn contains(self, value: f64) -> bool {
        value >= self.min && value <= self.max
    }

    fn validate(self, name: &str) -> MeshResult<()> {
        if self.min.is_nan() || self.max.is_nan() || self.min > self.max {
            return Err(MeshError::InvalidInput(format!(
                "{name} interval must be ordered and non-NaN"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EntityKind {
    Point,
    Edge,
    Face,
    Cell,
}

impl EntityKind {
    pub const fn row_kind(self) -> RowKind {
        match self {
            Self::Point => RowKind::Point,
            Self::Edge => RowKind::Edge,
            Self::Face => RowKind::Face,
            Self::Cell => RowKind::Cell,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityFilter {
    pub metric: QualityMetric,
    pub interval: Interval,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryMeasures {
    pub quality: Option<QualityMetric>,
    pub boundary_distance: bool,
    pub adjacent_boundary_tags: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagMatch {
    Any,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TagScope {
    Entity,
    AdjacentBoundary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagFilter {
    pub ids: BTreeSet<u64>,
    pub matching: TagMatch,
    pub scope: TagScope,
}

impl TagFilter {
    pub fn any(ids: impl IntoIterator<Item = u64>, scope: TagScope) -> Self {
        Self {
            ids: ids.into_iter().collect(),
            matching: TagMatch::Any,
            scope,
        }
    }

    pub fn all(ids: impl IntoIterator<Item = u64>, scope: TagScope) -> Self {
        Self {
            ids: ids.into_iter().collect(),
            matching: TagMatch::All,
            scope,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshQuery {
    pub x: Interval,
    pub y: Interval,
    pub z: Interval,
    pub entity_kind: EntityKind,
    pub element_type: Option<String>,
    pub zone_ids: BTreeSet<u64>,
    /// Compatibility shorthand for `All + Entity`. New callers should use
    /// `tag_filter`, which can also select adjacent boundary tags.
    pub tag_ids: BTreeSet<u64>,
    pub tag_filter: Option<TagFilter>,
    pub measures: QueryMeasures,
    pub boundary_distance: Option<Interval>,
    pub quality: Option<QualityFilter>,
    pub formula: Option<TypedFormula>,
    pub display_limit: usize,
}

impl Default for MeshQuery {
    fn default() -> Self {
        Self {
            x: Interval::ALL,
            y: Interval::ALL,
            z: Interval::ALL,
            entity_kind: EntityKind::Cell,
            element_type: None,
            zone_ids: BTreeSet::new(),
            tag_ids: BTreeSet::new(),
            tag_filter: None,
            measures: QueryMeasures::default(),
            boundary_distance: None,
            quality: None,
            formula: None,
            display_limit: 1_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectedEntity {
    pub id: u64,
    pub kind: EntityKind,
    pub tile_id: u64,
    pub element_type: String,
    pub point_ids: Vec<u64>,
    pub points: Vec<[f64; 3]>,
    pub edge_ids: Vec<u64>,
    pub face_ids: Vec<u64>,
    pub tag_ids: Vec<u64>,
    pub zone_id: Option<u64>,
    pub source_id: Option<u64>,
    pub source_object_id: Option<u64>,
    pub owner_cell_id: Option<u64>,
    pub neighbor_cell_id: Option<u64>,
    pub boundary: bool,
    pub boundary_distance: Option<f64>,
    pub quality: Option<f64>,
    pub adjacent_boundary_tag_ids: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TileRenderData {
    pub tile_id: u64,
    pub entities: Vec<SelectedEntity>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshQueryResult {
    pub total_matching_count: u64,
    pub displayed_count: usize,
    pub selected_entity_ids: Vec<u64>,
    pub render_tiles: Vec<TileRenderData>,
    pub progress: QueryProgress,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct QueryProgress {
    pub scanned_rows: u64,
    pub candidate_rows: u64,
    pub completed_batches: usize,
    pub candidate_batches: usize,
    pub complete: bool,
}

impl QueryProgress {
    pub fn fraction(self) -> f32 {
        if self.complete {
            1.0
        } else if self.candidate_rows == 0 {
            0.0
        } else {
            (self.scanned_rows as f64 / self.candidate_rows as f64).clamp(0.0, 1.0) as f32
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryBudget {
    pub max_rows: usize,
    pub max_time: Duration,
}

impl QueryBudget {
    pub const fn new(max_rows: usize, max_time: Duration) -> Self {
        Self { max_rows, max_time }
    }
}

impl Default for QueryBudget {
    fn default() -> Self {
        Self {
            max_rows: 8_192,
            max_time: Duration::from_millis(8),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueryStep {
    pub rows: Vec<SelectedEntity>,
    pub progress: QueryProgress,
}

#[derive(Debug, Clone)]
pub struct MeshQueryPlan {
    pub query: MeshQuery,
    pub measures: QueryMeasures,
    pub candidate_rows: u64,
    batch_indices: Vec<usize>,
}

impl MeshQueryPlan {
    pub fn candidate_batches(&self) -> usize {
        self.batch_indices.len()
    }
}

#[derive(Debug, Clone, Default)]
pub struct QueryCancellation(Arc<AtomicBool>);

impl QueryCancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Debug)]
pub struct MeshQueryService {
    file: Arc<MeshFile>,
    topology: Arc<OnceLock<Arc<TopologyIndex>>>,
    boundary: Arc<OnceLock<Arc<Mutex<BoundaryIndex>>>>,
}

impl MeshQueryService {
    pub fn new(file: Arc<MeshFile>) -> Self {
        Self {
            file,
            topology: Arc::new(OnceLock::new()),
            boundary: Arc::new(OnceLock::new()),
        }
    }

    pub fn mesh_file(&self) -> &Arc<MeshFile> {
        &self.file
    }

    pub fn plan(&self, query: MeshQuery) -> MeshResult<MeshQueryPlan> {
        self.plan_nodes(query, None)
    }

    fn plan_nodes(
        &self,
        query: MeshQuery,
        selected_nodes: Option<&BTreeSet<u64>>,
    ) -> MeshResult<MeshQueryPlan> {
        validate_query(&query)?;
        let measures = required_measures(&query)?;
        let query_bounds = Bounds3 {
            min: [query.x.min, query.y.min, query.z.min],
            max: [query.x.max, query.y.max, query.z.max],
        };
        let mut candidate_tiles = self.file.candidate_leaf_tiles(query_bounds);
        if let Some(selected) = selected_nodes {
            candidate_tiles.retain(|tile| selected.contains(tile));
        }
        let entity_tags =
            effective_tag_filter(&query).filter(|filter| filter.scope == TagScope::Entity);
        let mut batch_indices = Vec::new();
        let mut candidate_rows = 0u64;
        for entry in self.file.entity_batches(query.entity_kind.row_kind()) {
            let pruned = entry
                .spatial_node_id
                .is_none_or(|tile_id| !candidate_tiles.contains(&tile_id))
                || entry
                    .bounds
                    .is_some_and(|bounds| !bounds.intersects(query_bounds))
                || query
                    .element_type
                    .as_ref()
                    .is_some_and(|value| !entry.element_types.iter().any(|found| found == value))
                || (!query.zone_ids.is_empty()
                    && entry.zone_ids.iter().all(|id| !query.zone_ids.contains(id)))
                || entity_tags.as_ref().is_some_and(|filter| {
                    !tags_match(&entry.tag_ids, &filter.ids, filter.matching)
                });
            if !pruned {
                batch_indices.push(entry.batch_index);
                candidate_rows = candidate_rows.saturating_add(entry.rows as u64);
            }
        }
        Ok(MeshQueryPlan {
            query,
            measures,
            candidate_rows,
            batch_indices,
        })
    }

    pub fn cursor(&self, plan: MeshQueryPlan) -> MeshQueryCursor {
        MeshQueryCursor::new(
            self.file.clone(),
            self.topology.clone(),
            self.boundary.clone(),
            plan,
            QueryCancellation::default(),
        )
    }

    pub fn cursor_with_cancellation(
        &self,
        plan: MeshQueryPlan,
        cancellation: QueryCancellation,
    ) -> MeshQueryCursor {
        MeshQueryCursor::new(
            self.file.clone(),
            self.topology.clone(),
            self.boundary.clone(),
            plan,
            cancellation,
        )
    }

    pub fn execute(&self, query: MeshQuery) -> MeshResult<MeshQueryResult> {
        self.execute_nodes(query, None)
    }

    pub(crate) fn execute_selected_nodes(
        &self,
        query: MeshQuery,
        nodes: &BTreeSet<u64>,
    ) -> MeshResult<MeshQueryResult> {
        self.execute_nodes(query, Some(nodes))
    }

    fn execute_nodes(
        &self,
        query: MeshQuery,
        selected_nodes: Option<&BTreeSet<u64>>,
    ) -> MeshResult<MeshQueryResult> {
        let plan = self.plan_nodes(query, selected_nodes)?;
        let display_limit = plan.query.display_limit;
        let kind = plan.query.entity_kind;
        let mut cursor = self.cursor(plan);
        let mut total = 0u64;
        let mut heap = BinaryHeap::<(u64, u64)>::new();
        let mut selected = BTreeMap::<u64, SelectedEntity>::new();
        loop {
            let step = cursor.step(QueryBudget {
                max_rows: 65_536,
                max_time: Duration::MAX,
            })?;
            for entity in step.rows {
                total += 1;
                if display_limit == 0 {
                    continue;
                }
                let hash = stable_entity_hash(kind, entity.id);
                if heap.len() < display_limit {
                    heap.push((hash, entity.id));
                    selected.insert(entity.id, entity);
                } else if heap.peek().is_some_and(|&(largest_hash, largest_id)| {
                    (hash, entity.id) < (largest_hash, largest_id)
                }) {
                    let (_, removed) = heap.pop().expect("heap is non-empty");
                    selected.remove(&removed);
                    heap.push((hash, entity.id));
                    selected.insert(entity.id, entity);
                }
            }
            if step.progress.complete {
                break;
            }
        }

        let mut ranked: Vec<_> = heap.into_vec();
        ranked.sort_unstable();
        let selected_entity_ids: Vec<u64> = ranked.iter().map(|(_, id)| *id).collect();
        let mut tiles = BTreeMap::<u64, Vec<SelectedEntity>>::new();
        for (_, id) in ranked {
            if let Some(entity) = selected.remove(&id) {
                tiles.entry(entity.tile_id).or_default().push(entity);
            }
        }
        let render_tiles = tiles
            .into_iter()
            .map(|(tile_id, entities)| TileRenderData { tile_id, entities })
            .collect::<Vec<_>>();
        Ok(MeshQueryResult {
            total_matching_count: total,
            displayed_count: selected_entity_ids.len(),
            selected_entity_ids,
            render_tiles,
            progress: cursor.progress(),
        })
    }

    pub fn statistics(&self, query: MeshQuery) -> MeshResult<MeshQueryStatistics> {
        if query.entity_kind != EntityKind::Cell {
            return Err(MeshError::InvalidInput(
                "mesh query statistics require cell entities".into(),
            ));
        }
        let plan = self.plan(query)?;
        let total_cells = self.file.manifest().counts.cells;
        let mut cursor = self.cursor(plan);
        let mut accumulator = QueryStatisticsAccumulator::new(total_cells);
        loop {
            let step = cursor.step(QueryBudget {
                max_rows: 65_536,
                max_time: Duration::MAX,
            })?;
            accumulator.extend(step.rows);
            if step.progress.complete {
                return Ok(accumulator.finish(step.progress));
            }
        }
    }
}

#[derive(Debug)]
pub struct MeshQueryCursor {
    file: Arc<MeshFile>,
    plan: MeshQueryPlan,
    cancellation: QueryCancellation,
    next_batch: usize,
    batch: Option<BatchView>,
    row: usize,
    tile_id: Option<u64>,
    points: BTreeMap<u64, [f64; 3]>,
    scanned_rows: u64,
    completed_batches: usize,
    boundary: Option<Arc<Mutex<BoundaryIndex>>>,
    topology_cache: Arc<OnceLock<Arc<TopologyIndex>>>,
    topology: Option<Arc<TopologyIndex>>,
}

impl MeshQueryCursor {
    fn new(
        file: Arc<MeshFile>,
        topology_cache: Arc<OnceLock<Arc<TopologyIndex>>>,
        boundary_cache: Arc<OnceLock<Arc<Mutex<BoundaryIndex>>>>,
        plan: MeshQueryPlan,
        cancellation: QueryCancellation,
    ) -> Self {
        let boundary_needed = plan.measures.boundary_distance
            || plan.measures.adjacent_boundary_tags
            || effective_tag_filter(&plan.query)
                .is_some_and(|filter| filter.scope == TagScope::AdjacentBoundary);
        let boundary = boundary_needed.then(|| {
            boundary_cache
                .get_or_init(|| Arc::new(Mutex::new(BoundaryIndex::new(file.clone()))))
                .clone()
        });
        Self {
            file,
            plan,
            cancellation,
            next_batch: 0,
            batch: None,
            row: 0,
            tile_id: None,
            points: BTreeMap::new(),
            scanned_rows: 0,
            completed_batches: 0,
            boundary,
            topology_cache,
            topology: None,
        }
    }

    pub fn query(&self) -> &MeshQuery {
        &self.plan.query
    }

    pub fn measures(&self) -> QueryMeasures {
        self.plan.measures
    }

    pub fn cancellation(&self) -> QueryCancellation {
        self.cancellation.clone()
    }

    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    pub fn progress(&self) -> QueryProgress {
        QueryProgress {
            scanned_rows: self.scanned_rows,
            candidate_rows: self.plan.candidate_rows,
            completed_batches: self.completed_batches,
            candidate_batches: self.plan.batch_indices.len(),
            complete: self.batch.is_none() && self.next_batch == self.plan.batch_indices.len(),
        }
    }

    pub fn step(&mut self, budget: QueryBudget) -> MeshResult<QueryStep> {
        if self.cancellation.is_cancelled() {
            return Err(MeshError::Cancelled);
        }
        let started = Instant::now();
        let max_rows = budget.max_rows.max(1);
        let mut scanned = 0usize;
        let mut rows = Vec::new();
        while scanned < max_rows {
            if scanned != 0 && started.elapsed() >= budget.max_time {
                break;
            }
            if self.batch.is_none() {
                let Some(&batch_index) = self.plan.batch_indices.get(self.next_batch) else {
                    break;
                };
                let batch = self.file.batch_view(batch_index)?;
                let tile_id = batch.directory_entry().spatial_node_id.ok_or_else(|| {
                    MeshError::InvalidFile(
                        "entity batch has no owning tile in its directory".into(),
                    )
                })?;
                if self.tile_id != Some(tile_id) {
                    self.points = load_points(&self.file, tile_id)?;
                    self.tile_id = Some(tile_id);
                }
                self.batch = Some(batch);
                self.row = 0;
                self.next_batch += 1;
            }
            let batch = self.batch.as_ref().expect("loaded above").clone();
            if self.row == batch.len() {
                self.batch = None;
                self.completed_batches += 1;
                continue;
            }
            let row = self.row;
            self.row += 1;
            self.scanned_rows += 1;
            scanned += 1;
            if scanned.is_multiple_of(1_024) && self.cancellation.is_cancelled() {
                return Err(MeshError::Cancelled);
            }
            if !row_passes_cheap(
                &self.plan.query,
                self.file.manifest().dimension,
                &batch,
                row,
            )? {
                continue;
            }
            let tile_id = self.tile_id.expect("batch has a tile");
            let Some(mut entity) = entity_from_row(
                &self.file,
                &batch,
                row,
                self.plan.query.entity_kind,
                tile_id,
                &self.points,
            )?
            else {
                continue;
            };
            if !entity.points.iter().all(|point| {
                self.plan.query.x.contains(point[0])
                    && self.plan.query.y.contains(point[1])
                    && self.plan.query.z.contains(point[2])
            }) {
                continue;
            }

            let adjacent_tags_needed = self.plan.measures.adjacent_boundary_tags
                || effective_tag_filter(&self.plan.query)
                    .is_some_and(|filter| filter.scope == TagScope::AdjacentBoundary);
            if adjacent_tags_needed {
                entity.adjacent_boundary_tag_ids = self
                    .boundary
                    .as_ref()
                    .expect("planned boundary index")
                    .lock()
                    .map_err(|_| MeshError::InvalidFile("boundary index lock was poisoned".into()))?
                    .adjacent_tags(&entity, &self.cancellation)?;
            }
            if let Some(filter) = effective_tag_filter(&self.plan.query) {
                let values = match filter.scope {
                    TagScope::Entity => &entity.tag_ids,
                    TagScope::AdjacentBoundary => &entity.adjacent_boundary_tag_ids,
                };
                if !tags_match(values, &filter.ids, filter.matching) {
                    continue;
                }
            }
            if self.plan.measures.boundary_distance {
                entity.boundary_distance = self
                    .boundary
                    .as_ref()
                    .expect("planned boundary index")
                    .lock()
                    .map_err(|_| MeshError::InvalidFile("boundary index lock was poisoned".into()))?
                    .entity_distance(&entity, &self.cancellation)?;
                if self.plan.query.boundary_distance.is_some_and(|interval| {
                    entity
                        .boundary_distance
                        .is_none_or(|distance| !interval.contains(distance))
                }) {
                    continue;
                }
            }
            if let Some(metric) = self.plan.measures.quality {
                entity.quality = if entity.kind == EntityKind::Cell {
                    if entity.element_type == "polyhedron" {
                        self.ensure_topology()?.cell_quality(entity.id, metric)
                    } else {
                        let neighbors = if metric == QualityMetric::Orthogonality {
                            self.ensure_topology()?.neighbor_centers(
                                entity.id,
                                &entity.element_type,
                                &entity.point_ids,
                            )
                        } else {
                            BTreeMap::new()
                        };
                        quality_score_exact(
                            &entity.element_type,
                            &entity.point_ids,
                            &entity.points,
                            metric,
                            &neighbors,
                        )
                    }
                } else if entity.boundary {
                    self.ensure_topology()?.boundary_quality(&entity, metric)
                } else {
                    quality_score(&entity.element_type, &entity.points, metric)
                };
                if self.plan.query.quality.is_some_and(|filter| {
                    entity
                        .quality
                        .is_none_or(|quality| !filter.interval.contains(quality))
                }) {
                    continue;
                }
            }
            if let Some(formula) = &self.plan.query.formula {
                let context = FormulaContext {
                    centroid: centroid(&entity.points),
                    dimension: element_dimension(&entity.element_type).unwrap_or(0),
                    quality_metric: self.plan.measures.quality,
                    entity: &entity,
                    file: &self.file,
                };
                if !formula.evaluate(&context)? {
                    continue;
                }
            }
            rows.push(entity);
        }
        if self
            .batch
            .as_ref()
            .is_some_and(|batch| self.row == batch.len())
        {
            self.batch = None;
            self.completed_batches += 1;
        }
        Ok(QueryStep {
            rows,
            progress: self.progress(),
        })
    }

    fn ensure_topology(&mut self) -> MeshResult<&TopologyIndex> {
        if self.topology.is_none() {
            let topology = if let Some(topology) = self.topology_cache.get() {
                topology.clone()
            } else {
                let topology = Arc::new(TopologyIndex::build(&self.file, &self.cancellation)?);
                let _ = self.topology_cache.set(topology.clone());
                self.topology_cache.get().cloned().unwrap_or(topology)
            };
            self.topology = Some(topology);
        }
        Ok(self.topology.as_deref().expect("inserted above"))
    }
}

fn row_passes_cheap(
    query: &MeshQuery,
    top_dimension: u8,
    batch: &BatchView,
    row: usize,
) -> MeshResult<bool> {
    let ids = batch.u64s("entity_id")?;
    if ids.is_null(row) {
        return Ok(false);
    }
    if query.entity_kind == EntityKind::Point {
        let ghosts = batch.bools("ghost")?;
        if ghosts.is_null(row) || ghosts.value(row) {
            return Ok(false);
        }
    } else {
        let types = batch.strings("element_type")?;
        if types.is_null(row) {
            return Err(MeshError::InvalidFile(
                "entity row has no element type".into(),
            ));
        }
        if query.entity_kind == EntityKind::Cell
            && element_dimension(types.value(row)) != Some(top_dimension)
        {
            return Ok(false);
        }
        if query
            .element_type
            .as_ref()
            .is_some_and(|wanted| wanted != types.value(row))
        {
            return Ok(false);
        }
    }
    let zones = batch.u64s("zone_id")?;
    if !query.zone_ids.is_empty()
        && (zones.is_null(row) || !query.zone_ids.contains(&zones.value(row)))
    {
        return Ok(false);
    }
    if let Some(filter) =
        effective_tag_filter(query).filter(|filter| filter.scope == TagScope::Entity)
    {
        let tags = list_values(batch.lists("tag_ids")?, row)?;
        if !tags_match(&tags, &filter.ids, filter.matching) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn required_measures(query: &MeshQuery) -> MeshResult<QueryMeasures> {
    let mut measures = query.measures;
    measures.boundary_distance |= query.boundary_distance.is_some();
    measures.adjacent_boundary_tags |= effective_tag_filter(query)
        .is_some_and(|filter| filter.scope == TagScope::AdjacentBoundary);
    if let Some(filter) = query.quality {
        if measures
            .quality
            .is_some_and(|requested| requested != filter.metric)
        {
            return Err(MeshError::InvalidInput(
                "one query cannot request two quality metrics".into(),
            ));
        }
        measures.quality = Some(filter.metric);
    }
    if let Some(formula) = &query.formula {
        formula.apply_requirements(&mut measures)?;
    }
    Ok(measures)
}

fn effective_tag_filter(query: &MeshQuery) -> Option<TagFilter> {
    query.tag_filter.clone().or_else(|| {
        (!query.tag_ids.is_empty())
            .then(|| TagFilter::all(query.tag_ids.iter().copied(), TagScope::Entity))
    })
}

fn tags_match(values: &[u64], wanted: &BTreeSet<u64>, matching: TagMatch) -> bool {
    match matching {
        TagMatch::Any => wanted.iter().any(|id| values.contains(id)),
        TagMatch::All => wanted.iter().all(|id| values.contains(id)),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshQueryStatistics {
    pub total_cells: u64,
    pub filtered_cells: u64,
    pub supported: u64,
    pub unsupported: u64,
    pub minimum: Option<f64>,
    pub mean: Option<f64>,
    pub maximum: Option<f64>,
    pub worst_cell_id: Option<u64>,
    pub maximum_boundary_distance: Option<f64>,
    pub progress: QueryProgress,
}

#[derive(Debug, Clone)]
pub struct QueryStatisticsAccumulator {
    total_cells: u64,
    filtered_cells: u64,
    supported: u64,
    unsupported: u64,
    minimum: Option<f64>,
    mean: f64,
    maximum: Option<f64>,
    worst_cell_id: Option<u64>,
    maximum_boundary_distance: Option<f64>,
}

impl QueryStatisticsAccumulator {
    pub fn new(total_cells: u64) -> Self {
        Self {
            total_cells,
            filtered_cells: 0,
            supported: 0,
            unsupported: 0,
            minimum: None,
            mean: 0.0,
            maximum: None,
            worst_cell_id: None,
            maximum_boundary_distance: None,
        }
    }

    pub fn push(&mut self, entity: SelectedEntity) {
        if entity.kind != EntityKind::Cell {
            return;
        }
        self.filtered_cells += 1;
        if let Some(distance) = entity.boundary_distance.filter(|value| value.is_finite()) {
            self.maximum_boundary_distance = Some(
                self.maximum_boundary_distance
                    .map_or(distance, |old| old.max(distance)),
            );
        }
        let Some(quality) = entity.quality else {
            self.unsupported += 1;
            return;
        };
        self.supported += 1;
        self.mean += (quality - self.mean) / self.supported as f64;
        self.maximum = Some(self.maximum.map_or(quality, |old| old.max(quality)));
        match self.minimum {
            None => {
                self.minimum = Some(quality);
                self.worst_cell_id = Some(entity.id);
            }
            Some(old)
                if quality < old
                    || (quality == old && self.worst_cell_id.is_none_or(|id| entity.id < id)) =>
            {
                self.minimum = Some(quality);
                self.worst_cell_id = Some(entity.id);
            }
            _ => {}
        }
    }

    pub fn extend(&mut self, entities: impl IntoIterator<Item = SelectedEntity>) {
        for entity in entities {
            self.push(entity);
        }
    }

    pub fn finish(&self, progress: QueryProgress) -> MeshQueryStatistics {
        MeshQueryStatistics {
            total_cells: self.total_cells,
            filtered_cells: self.filtered_cells,
            supported: self.supported,
            unsupported: self.unsupported,
            minimum: self.minimum,
            mean: (self.supported != 0).then_some(self.mean),
            maximum: self.maximum,
            worst_cell_id: self.worst_cell_id,
            maximum_boundary_distance: self.maximum_boundary_distance,
            progress,
        }
    }
}

fn validate_query(query: &MeshQuery) -> MeshResult<()> {
    query.x.validate("x")?;
    query.y.validate("y")?;
    query.z.validate("z")?;
    if let Some(interval) = query.boundary_distance {
        interval.validate("boundary distance")?;
        if interval.min < 0.0 {
            return Err(MeshError::InvalidInput(
                "boundary distance cannot be negative".into(),
            ));
        }
    }
    if let Some(quality) = query.quality {
        quality.interval.validate("quality")?;
    }
    Ok(())
}

fn entity_from_row(
    file: &MeshFile,
    batch: &BatchView,
    row: usize,
    kind: EntityKind,
    tile_id: u64,
    points: &BTreeMap<u64, [f64; 3]>,
) -> MeshResult<Option<SelectedEntity>> {
    let ids = batch.u64s("entity_id")?;
    if ids.is_null(row) {
        return Ok(None);
    }
    if kind == EntityKind::Point {
        let ghosts = batch.bools("ghost")?;
        if ghosts.is_null(row) || ghosts.value(row) {
            return Ok(None);
        }
    }
    let element_types = batch.strings("element_type")?;
    let element_type = if kind == EntityKind::Point {
        "point1".to_string()
    } else if element_types.is_null(row) {
        return Err(MeshError::InvalidFile(
            "entity row has no element type".into(),
        ));
    } else {
        element_types.value(row).to_string()
    };
    let point_ids = if kind == EntityKind::Point {
        vec![ids.value(row)]
    } else {
        list_values(batch.lists("point_ids")?, row)?
    };
    let geometry: Option<Vec<_>> = point_ids.iter().map(|id| points.get(id).copied()).collect();
    let Some(geometry) = geometry else {
        return Err(MeshError::InvalidFile(format!(
            "{kind:?} {} references a point absent from tile {tile_id}",
            ids.value(row)
        )));
    };
    let tag_ids = list_values(batch.lists("tag_ids")?, row)?;
    let edge_ids = list_values(batch.lists("edge_ids")?, row)?;
    let face_ids = list_values(batch.lists("face_ids")?, row)?;
    let zones = batch.u64s("zone_id")?;
    let sources = batch.u64s("source_id")?;
    let source_id = (!sources.is_null(row)).then(|| sources.value(row));
    let owners = batch.u64s("owner_cell_id")?;
    let neighbors = batch.u64s("neighbor_cell_id")?;
    let boundary_flags = batch.bools("boundary")?;
    let boundary = !boundary_flags.is_null(row) && boundary_flags.value(row);
    Ok(Some(SelectedEntity {
        id: ids.value(row),
        kind,
        tile_id,
        element_type,
        point_ids,
        points: geometry,
        edge_ids,
        face_ids,
        tag_ids,
        zone_id: (!zones.is_null(row)).then(|| zones.value(row)),
        source_id,
        source_object_id: source_id.and_then(|id| file.catalog_source_object("source", id)),
        owner_cell_id: (!owners.is_null(row)).then(|| owners.value(row)),
        neighbor_cell_id: (!neighbors.is_null(row)).then(|| neighbors.value(row)),
        boundary,
        boundary_distance: None,
        quality: None,
        adjacent_boundary_tag_ids: Vec::new(),
    }))
}

fn load_points(file: &MeshFile, tile_id: u64) -> MeshResult<BTreeMap<u64, [f64; 3]>> {
    let mut points = BTreeMap::new();
    for entry in file.tile_batches(tile_id, RowKind::Point) {
        let batch = file.batch_view(entry.batch_index)?;
        let ids = batch.u64s("entity_id")?;
        let x = batch.f64s("x")?;
        let y = batch.f64s("y")?;
        let z = batch.f64s("z")?;
        for row in 0..batch.len() {
            points.insert(ids.value(row), [x.value(row), y.value(row), z.value(row)]);
        }
    }
    Ok(points)
}

fn list_values(array: &LargeListArray, row: usize) -> MeshResult<Vec<u64>> {
    let values = array.value(row);
    values
        .as_any()
        .downcast_ref::<UInt64Array>()
        .map(|values| values.values().to_vec())
        .ok_or_else(|| MeshError::InvalidFile("list column must contain u64 values".into()))
}

fn centroid(points: &[[f64; 3]]) -> [f64; 3] {
    let mut result = [0.0; 3];
    for point in points {
        for axis in 0..3 {
            result[axis] += point[axis] / points.len() as f64;
        }
    }
    result
}

fn point_segment_distance(point: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = std::array::from_fn::<_, 3, _>(|axis| b[axis] - a[axis]);
    let ap = std::array::from_fn::<_, 3, _>(|axis| point[axis] - a[axis]);
    let denominator: f64 = ab.iter().map(|value| value * value).sum();
    let t = if denominator <= f64::EPSILON {
        0.0
    } else {
        (0..3).map(|axis| ap[axis] * ab[axis]).sum::<f64>() / denominator
    }
    .clamp(0.0, 1.0);
    (0..3)
        .map(|axis| (point[axis] - (a[axis] + t * ab[axis])).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[derive(Debug, Clone)]
struct BoundaryPrimitive {
    id: u64,
    point_ids: Vec<u64>,
    points: Vec<[f64; 3]>,
    tag_ids: Vec<u64>,
    owner_cell_id: Option<u64>,
    neighbor_cell_id: Option<u64>,
}

#[derive(Debug, Clone)]
struct BoundaryTileMeta {
    id: u64,
    bounds: Bounds3,
}

/// Lazy exact-Arrow boundary geometry index. Its representation is private so
/// cache and spatial-search details can change without affecting query users.
#[derive(Debug)]
pub struct BoundaryIndex {
    file: Arc<MeshFile>,
    kind: RowKind,
    tiles: Vec<BoundaryTileMeta>,
    cache: BTreeMap<u64, Arc<Vec<BoundaryPrimitive>>>,
    lru: VecDeque<u64>,
    cache_tiles: usize,
    adjacency_built: bool,
    tags_by_cell: BTreeMap<u64, BTreeSet<u64>>,
    tags_by_entity: BTreeMap<u64, BTreeSet<u64>>,
    tags_by_signature: BTreeMap<Vec<u64>, BTreeSet<u64>>,
    cancellation: QueryCancellation,
}

impl BoundaryIndex {
    pub fn new(file: Arc<MeshFile>) -> Self {
        let kind = if file.manifest().dimension == 3 {
            RowKind::Face
        } else {
            RowKind::Edge
        };
        let mut by_tile = BTreeMap::<u64, Bounds3>::new();
        for entry in file.entity_batches(kind) {
            if let (Some(id), Some(bounds)) = (entry.spatial_node_id, entry.bounds) {
                by_tile
                    .entry(id)
                    .and_modify(|old| *old = old.union(bounds))
                    .or_insert(bounds);
            }
        }
        Self {
            file,
            kind,
            tiles: by_tile
                .into_iter()
                .map(|(id, bounds)| BoundaryTileMeta { id, bounds })
                .collect(),
            cache: BTreeMap::new(),
            lru: VecDeque::new(),
            cache_tiles: 16,
            adjacency_built: false,
            tags_by_cell: BTreeMap::new(),
            tags_by_entity: BTreeMap::new(),
            tags_by_signature: BTreeMap::new(),
            cancellation: QueryCancellation::default(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    pub fn cancellation(&self) -> QueryCancellation {
        self.cancellation.clone()
    }

    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    pub fn distance(&mut self, point: [f64; 3]) -> MeshResult<Option<f64>> {
        let cancellation = self.cancellation.clone();
        self.distance_with_cancellation(point, &cancellation)
    }

    fn distance_with_cancellation(
        &mut self,
        point: [f64; 3],
        cancellation: &QueryCancellation,
    ) -> MeshResult<Option<f64>> {
        if self.cancellation.is_cancelled() || cancellation.is_cancelled() {
            return Err(MeshError::Cancelled);
        }
        let mut ordered = self
            .tiles
            .iter()
            .map(|tile| (point_bounds_distance(point, tile.bounds), tile.id))
            .collect::<Vec<_>>();
        ordered.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut best = f64::INFINITY;
        let mut found = false;
        for (lower_bound, tile) in ordered {
            if self.cancellation.is_cancelled() || cancellation.is_cancelled() {
                return Err(MeshError::Cancelled);
            }
            if lower_bound >= best {
                break;
            }
            for primitive in self.load_tile(tile)?.iter() {
                let distance = if self.kind == RowKind::Edge {
                    primitive
                        .points
                        .get(0..2)
                        .map(|points| point_segment_distance(point, points[0], points[1]))
                } else {
                    point_polygon_distance(point, &primitive.points)
                };
                if let Some(distance) = distance {
                    found = true;
                    best = best.min(distance);
                }
            }
        }
        Ok(found.then_some(best))
    }

    fn entity_distance(
        &mut self,
        entity: &SelectedEntity,
        cancellation: &QueryCancellation,
    ) -> MeshResult<Option<f64>> {
        let count = corner_count(&entity.element_type).unwrap_or(entity.points.len());
        let mut result: Option<f64> = None;
        for &point in entity.points.iter().take(count) {
            if let Some(distance) = self.distance_with_cancellation(point, cancellation)? {
                result = Some(result.map_or(distance, |old| old.min(distance)));
            }
        }
        Ok(result)
    }

    fn adjacent_tags(
        &mut self,
        entity: &SelectedEntity,
        cancellation: &QueryCancellation,
    ) -> MeshResult<Vec<u64>> {
        self.ensure_adjacency(cancellation)?;
        if entity.kind != EntityKind::Cell {
            return Ok(entity.tag_ids.clone());
        }
        let mut tags = BTreeSet::new();
        for id in entity.edge_ids.iter().chain(&entity.face_ids) {
            if let Some(found) = self.tags_by_entity.get(id) {
                tags.extend(found);
            }
        }
        let corner_count = corner_count(&entity.element_type).unwrap_or(entity.point_ids.len());
        if let (Some(ids), Some(sides)) = (
            entity.point_ids.get(..corner_count),
            side_indices(&entity.element_type, corner_count),
        ) {
            for side in sides {
                let mut signature = side.into_iter().map(|index| ids[index]).collect::<Vec<_>>();
                signature.sort_unstable();
                if let Some(found) = self.tags_by_signature.get(&signature) {
                    tags.extend(found);
                }
            }
        }
        if let Some(found) = self.tags_by_cell.get(&entity.id) {
            tags.extend(found);
        }
        Ok(tags.into_iter().collect())
    }

    fn ensure_adjacency(&mut self, cancellation: &QueryCancellation) -> MeshResult<()> {
        if self.adjacency_built {
            return Ok(());
        }
        let tiles = self.tiles.iter().map(|tile| tile.id).collect::<Vec<_>>();
        for tile in tiles {
            if self.cancellation.is_cancelled() || cancellation.is_cancelled() {
                return Err(MeshError::Cancelled);
            }
            for primitive in self.load_tile(tile)?.iter() {
                self.tags_by_entity
                    .entry(primitive.id)
                    .or_default()
                    .extend(&primitive.tag_ids);
                for cell in [primitive.owner_cell_id, primitive.neighbor_cell_id]
                    .into_iter()
                    .flatten()
                {
                    self.tags_by_cell
                        .entry(cell)
                        .or_default()
                        .extend(&primitive.tag_ids);
                }
                let mut signature = primitive.point_ids.clone();
                signature.sort_unstable();
                self.tags_by_signature
                    .entry(signature)
                    .or_default()
                    .extend(&primitive.tag_ids);
            }
        }
        self.adjacency_built = true;
        Ok(())
    }

    fn load_tile(&mut self, tile: u64) -> MeshResult<Arc<Vec<BoundaryPrimitive>>> {
        if let Some(value) = self.cache.get(&tile).cloned() {
            self.touch(tile);
            return Ok(value);
        }
        let points = load_points(&self.file, tile)?;
        let mut primitives = Vec::new();
        for entry in self.file.tile_batches(tile, self.kind) {
            let batch = self.file.batch_view(entry.batch_index)?;
            let ids = batch.u64s("entity_id")?;
            let types = batch.strings("element_type")?;
            let connectivity = batch.lists("point_ids")?;
            let tags = batch.lists("tag_ids")?;
            let boundary = batch.bools("boundary")?;
            let owners = batch.u64s("owner_cell_id")?;
            let neighbors = batch.u64s("neighbor_cell_id")?;
            for row in 0..batch.len() {
                if boundary.is_null(row) || !boundary.value(row) {
                    continue;
                }
                let point_ids = list_values(connectivity, row)?;
                let corner_count = match types.value(row) {
                    "edge2" | "edge3" => 2,
                    "polygon" => point_ids.len(),
                    element_type => corner_count(element_type).unwrap_or(point_ids.len()),
                };
                let point_ids = point_ids
                    .get(..corner_count)
                    .ok_or_else(|| {
                        MeshError::InvalidFile(format!(
                            "boundary entity {} has insufficient corner connectivity",
                            ids.value(row)
                        ))
                    })?
                    .to_vec();
                let geometry = point_ids
                    .iter()
                    .map(|id| points.get(id).copied())
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        MeshError::InvalidFile(format!(
                            "boundary entity {} references a point absent from tile {tile}",
                            ids.value(row)
                        ))
                    })?;
                primitives.push(BoundaryPrimitive {
                    id: ids.value(row),
                    point_ids,
                    points: geometry,
                    tag_ids: list_values(tags, row)?,
                    owner_cell_id: (!owners.is_null(row)).then(|| owners.value(row)),
                    neighbor_cell_id: (!neighbors.is_null(row)).then(|| neighbors.value(row)),
                });
            }
        }
        let primitives = Arc::new(primitives);
        self.cache.insert(tile, primitives.clone());
        self.touch(tile);
        while self.cache.len() > self.cache_tiles {
            if let Some(oldest) = self.lru.pop_front() {
                self.cache.remove(&oldest);
            }
        }
        Ok(primitives)
    }

    fn touch(&mut self, tile: u64) {
        self.lru.retain(|id| *id != tile);
        self.lru.push_back(tile);
    }
}

#[derive(Debug, Clone)]
struct IndexedCell {
    id: u64,
    element_type: String,
    point_ids: Vec<u64>,
    points: Vec<[f64; 3]>,
    face_ids: Vec<u64>,
    center: [f64; 3],
}

#[derive(Debug, Clone)]
struct IndexedFace {
    point_ids: Vec<u64>,
    points: Vec<[f64; 3]>,
}

#[derive(Debug)]
struct TopologyIndex {
    cells: BTreeMap<u64, IndexedCell>,
    faces: BTreeMap<u64, IndexedFace>,
    side_cells: BTreeMap<Vec<u64>, Vec<u64>>,
}

impl TopologyIndex {
    fn build(file: &MeshFile, cancellation: &QueryCancellation) -> MeshResult<Self> {
        let mut faces = BTreeMap::new();
        let mut current_tile = None;
        let mut points = BTreeMap::new();
        for entry in file.entity_batches(RowKind::Face) {
            if cancellation.is_cancelled() {
                return Err(MeshError::Cancelled);
            }
            let tile = entry
                .spatial_node_id
                .ok_or_else(|| MeshError::InvalidFile("face batch has no owning tile".into()))?;
            if current_tile != Some(tile) {
                points = load_points(file, tile)?;
                current_tile = Some(tile);
            }
            let batch = file.batch_view(entry.batch_index)?;
            for row in 0..batch.len() {
                let Some(entity) =
                    entity_from_row(file, &batch, row, EntityKind::Face, tile, &points)?
                else {
                    continue;
                };
                let count = match entity.element_type.as_str() {
                    "polygon" => entity.point_ids.len(),
                    _ => corner_count(&entity.element_type).unwrap_or(entity.point_ids.len()),
                };
                faces.insert(
                    entity.id,
                    IndexedFace {
                        point_ids: entity
                            .point_ids
                            .get(..count)
                            .unwrap_or(&entity.point_ids)
                            .into(),
                        points: entity.points.get(..count).unwrap_or(&entity.points).into(),
                    },
                );
            }
        }
        let mut cells = BTreeMap::new();
        let mut side_cells = BTreeMap::<Vec<u64>, Vec<u64>>::new();
        current_tile = None;
        points.clear();
        for entry in file.entity_batches(RowKind::Cell) {
            if cancellation.is_cancelled() {
                return Err(MeshError::Cancelled);
            }
            let tile = entry
                .spatial_node_id
                .ok_or_else(|| MeshError::InvalidFile("cell batch has no owning tile".into()))?;
            if current_tile != Some(tile) {
                points = load_points(file, tile)?;
                current_tile = Some(tile);
            }
            let batch = file.batch_view(entry.batch_index)?;
            for row in 0..batch.len() {
                let Some(entity) =
                    entity_from_row(file, &batch, row, EntityKind::Cell, tile, &points)?
                else {
                    continue;
                };
                if element_dimension(&entity.element_type) != Some(file.manifest().dimension) {
                    continue;
                }
                let count = corner_count(&entity.element_type).unwrap_or(entity.point_ids.len());
                let ids = entity.point_ids.get(..count).ok_or_else(|| {
                    MeshError::InvalidFile(format!(
                        "cell {} has insufficient corner connectivity",
                        entity.id
                    ))
                })?;
                if let Some(sides) = side_indices(&entity.element_type, count) {
                    for side in sides {
                        let mut signature =
                            side.into_iter().map(|index| ids[index]).collect::<Vec<_>>();
                        signature.sort_unstable();
                        side_cells.entry(signature).or_default().push(entity.id);
                    }
                } else if entity.element_type == "polyhedron" {
                    for face_id in &entity.face_ids {
                        if let Some(face) = faces.get(face_id) {
                            let mut signature = face.point_ids.clone();
                            signature.sort_unstable();
                            side_cells.entry(signature).or_default().push(entity.id);
                        }
                    }
                }
                let cell = IndexedCell {
                    id: entity.id,
                    element_type: entity.element_type,
                    point_ids: entity.point_ids,
                    face_ids: entity.face_ids,
                    center: centroid(
                        entity
                            .points
                            .get(..count)
                            .unwrap_or(entity.points.as_slice()),
                    ),
                    points: entity.points,
                };
                cells.insert(cell.id, cell);
            }
        }
        for ids in side_cells.values_mut() {
            ids.sort_unstable();
            ids.dedup();
        }
        Ok(Self {
            cells,
            faces,
            side_cells,
        })
    }

    fn neighbor_centers(
        &self,
        cell_id: u64,
        element_type: &str,
        point_ids: &[u64],
    ) -> BTreeMap<Vec<u64>, [f64; 3]> {
        let count = corner_count(element_type).unwrap_or(point_ids.len());
        let Some(ids) = point_ids.get(..count) else {
            return BTreeMap::new();
        };
        side_indices(element_type, count)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|side| {
                let mut signature = side.into_iter().map(|index| ids[index]).collect::<Vec<_>>();
                signature.sort_unstable();
                let neighbor = self
                    .side_cells
                    .get(&signature)?
                    .iter()
                    .copied()
                    .find(|id| *id != cell_id)?;
                Some((signature, self.cells.get(&neighbor)?.center))
            })
            .collect()
    }

    fn boundary_quality(&self, entity: &SelectedEntity, metric: QualityMetric) -> Option<f64> {
        let mut ids = [entity.owner_cell_id, entity.neighbor_cell_id]
            .into_iter()
            .flatten()
            .filter(|id| self.cells.contains_key(id))
            .collect::<Vec<_>>();
        let mut signature = entity.point_ids.clone();
        signature.sort_unstable();
        ids.extend(
            self.side_cells
                .get(&signature)
                .into_iter()
                .flatten()
                .copied(),
        );
        ids.sort_unstable();
        ids.dedup();
        let scores = ids
            .into_iter()
            .map(|id| self.cell_quality(id, metric))
            .collect::<Vec<_>>();
        if scores.is_empty() || scores.iter().any(Option::is_none) {
            None
        } else {
            scores.into_iter().flatten().reduce(f64::min)
        }
    }

    fn cell_quality(&self, id: u64, metric: QualityMetric) -> Option<f64> {
        let cell = self.cells.get(&id)?;
        if cell.element_type == "polyhedron" {
            let faces = cell
                .face_ids
                .iter()
                .map(|id| {
                    let face = self.faces.get(id)?;
                    Some((face.point_ids.clone(), face.points.clone()))
                })
                .collect::<Option<Vec<_>>>()?;
            let neighbors = faces
                .iter()
                .filter_map(|(signature, _)| {
                    let mut signature = signature.clone();
                    signature.sort_unstable();
                    let neighbor = self
                        .side_cells
                        .get(&signature)?
                        .iter()
                        .copied()
                        .find(|other| *other != id)?;
                    Some((signature, self.cells.get(&neighbor)?.center))
                })
                .collect();
            return polyhedron_quality_score(&cell.points, &faces, metric, &neighbors);
        }
        let neighbors = if metric == QualityMetric::Orthogonality {
            self.neighbor_centers(id, &cell.element_type, &cell.point_ids)
        } else {
            BTreeMap::new()
        };
        quality_score_exact(
            &cell.element_type,
            &cell.point_ids,
            &cell.points,
            metric,
            &neighbors,
        )
    }
}

fn point_bounds_distance(point: [f64; 3], bounds: Bounds3) -> f64 {
    (0..3)
        .map(|axis| {
            if point[axis] < bounds.min[axis] {
                bounds.min[axis] - point[axis]
            } else if point[axis] > bounds.max[axis] {
                point[axis] - bounds.max[axis]
            } else {
                0.0
            }
        })
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

fn point_polygon_distance(point: [f64; 3], polygon: &[[f64; 3]]) -> Option<f64> {
    let [a, rest @ ..] = polygon else {
        return None;
    };
    if rest.len() < 2 {
        return None;
    }
    (1..polygon.len() - 1)
        .map(|index| point_triangle_distance(point, *a, polygon[index], polygon[index + 1]))
        .reduce(f64::min)
}

fn point_triangle_distance(point: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let sub = |x: [f64; 3], y: [f64; 3]| std::array::from_fn(|axis| x[axis] - y[axis]);
    let dot = |x: [f64; 3], y: [f64; 3]| (0..3).map(|axis| x[axis] * y[axis]).sum::<f64>();
    let distance = |x: [f64; 3], y: [f64; 3]| dot(sub(x, y), sub(x, y)).sqrt();
    let ab = sub(b, a);
    let ac = sub(c, a);
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    if dot(cross, cross) <= f64::EPSILON {
        return point_segment_distance(point, a, b)
            .min(point_segment_distance(point, b, c))
            .min(point_segment_distance(point, c, a));
    }
    let ap = sub(point, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return distance(point, a);
    }
    let bp = sub(point, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return distance(point, b);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let projection = std::array::from_fn(|axis| a[axis] + v * ab[axis]);
        return distance(point, projection);
    }
    let cp = sub(point, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return distance(point, c);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let projection = std::array::from_fn(|axis| a[axis] + w * ac[axis]);
        return distance(point, projection);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let edge = sub(c, b);
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let projection = std::array::from_fn(|axis| b[axis] + w * edge[axis]);
        return distance(point, projection);
    }
    let denominator = 1.0 / (va + vb + vc);
    let v = vb * denominator;
    let w = vc * denominator;
    let projection = std::array::from_fn(|axis| a[axis] + ab[axis] * v + ac[axis] * w);
    distance(point, projection)
}

fn stable_entity_hash(kind: EntityKind, id: u64) -> u64 {
    let kind = match kind {
        EntityKind::Point => 1u64,
        EntityKind::Edge => 2,
        EntityKind::Face => 3,
        EntityKind::Cell => 4,
    };
    let mut value = id ^ kind.wrapping_mul(0x9e3779b97f4a7c15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedFormula {
    expression: BoolExpr,
}

impl TypedFormula {
    pub fn parse(source: &str) -> MeshResult<Self> {
        let mut parser = Parser::new(source)?;
        let expression = parser.parse_or()?;
        if parser.peek() != &Token::End {
            return Err(MeshError::InvalidInput(
                "unexpected token after formula expression".into(),
            ));
        }
        Ok(Self { expression })
    }

    fn evaluate(&self, context: &FormulaContext<'_>) -> MeshResult<bool> {
        self.expression.evaluate(context)
    }

    fn apply_requirements(&self, measures: &mut QueryMeasures) -> MeshResult<()> {
        self.expression.apply_requirements(measures)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum BoolExpr {
    Literal(bool),
    Not(Box<Self>),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Compare(ValueExpr, CompareOp, ValueExpr),
    HasTag(TagArgument),
    IsSupported(Option<QualityMetric>),
}

impl BoolExpr {
    fn apply_requirements(&self, measures: &mut QueryMeasures) -> MeshResult<()> {
        match self {
            Self::Not(value) => value.apply_requirements(measures),
            Self::And(a, b) | Self::Or(a, b) => {
                a.apply_requirements(measures)?;
                b.apply_requirements(measures)
            }
            Self::Compare(a, _, b) => {
                a.apply_requirements(measures)?;
                b.apply_requirements(measures)
            }
            Self::IsSupported(metric) => {
                let metric = (*metric)
                    .or(measures.quality)
                    .unwrap_or(QualityMetric::ScaledJacobian);
                require_quality(measures, metric)
            }
            Self::Literal(_) | Self::HasTag(_) => Ok(()),
        }
    }

    fn evaluate(&self, context: &FormulaContext<'_>) -> MeshResult<bool> {
        Ok(match self {
            Self::Literal(value) => *value,
            Self::Not(value) => !value.evaluate(context)?,
            Self::And(a, b) => a.evaluate(context)? && b.evaluate(context)?,
            Self::Or(a, b) => a.evaluate(context)? || b.evaluate(context)?,
            Self::Compare(a, operator, b) => {
                operator.evaluate(a.evaluate(context)?, b.evaluate(context)?)?
            }
            Self::HasTag(TagArgument::Id(id)) => context.entity.tag_ids.contains(id),
            Self::HasTag(TagArgument::Name(name)) => context
                .entity
                .tag_ids
                .iter()
                .any(|id| context.file.catalog_name("tag", *id) == Some(name.as_str())),
            Self::IsSupported(metric) => metric.or(context.quality_metric).is_some_and(|metric| {
                if context.quality_metric == Some(metric) {
                    context.entity.quality.is_some()
                } else {
                    quality_score(&context.entity.element_type, &context.entity.points, metric)
                        .is_some()
                }
            }),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ValueExpr {
    Number(f64),
    Unsigned(u64),
    String(String),
    Field(FieldName),
}

impl ValueExpr {
    fn apply_requirements(&self, measures: &mut QueryMeasures) -> MeshResult<()> {
        match self {
            Self::Field(FieldName::Quality) => {
                let metric = measures.quality.unwrap_or(QualityMetric::ScaledJacobian);
                require_quality(measures, metric)
            }
            Self::Field(FieldName::BoundaryDistance) => {
                measures.boundary_distance = true;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn evaluate(&self, context: &FormulaContext<'_>) -> MeshResult<Value> {
        Ok(match self {
            Self::Number(value) => Value::Number(*value),
            Self::Unsigned(value) => Value::Unsigned(*value),
            Self::String(value) => Value::String(value.clone()),
            Self::Field(field) => match field {
                FieldName::Id => Value::Unsigned(context.entity.id),
                FieldName::X => Value::Number(context.centroid[0]),
                FieldName::Y => Value::Number(context.centroid[1]),
                FieldName::Z => Value::Number(context.centroid[2]),
                FieldName::TileId => Value::Unsigned(context.entity.tile_id),
                FieldName::ZoneId => context
                    .entity
                    .zone_id
                    .map(Value::Unsigned)
                    .unwrap_or(Value::Null),
                FieldName::Dimension => Value::Unsigned(u64::from(context.dimension)),
                FieldName::Quality => context
                    .entity
                    .quality
                    .map(Value::Number)
                    .unwrap_or(Value::Null),
                FieldName::BoundaryDistance => context
                    .entity
                    .boundary_distance
                    .map(Value::Number)
                    .unwrap_or(Value::Null),
                FieldName::ElementType => Value::String(context.entity.element_type.clone()),
            },
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldName {
    Id,
    X,
    Y,
    Z,
    TileId,
    ZoneId,
    Dimension,
    Quality,
    BoundaryDistance,
    ElementType,
}

#[derive(Debug, Clone)]
enum Value {
    Number(f64),
    Unsigned(u64),
    String(String),
    Null,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CompareOp {
    fn evaluate(self, left: Value, right: Value) -> MeshResult<bool> {
        match (left, right) {
            (Value::Null, Value::Null) => Ok(matches!(self, Self::Eq | Self::Le | Self::Ge)),
            (Value::Null, _) | (_, Value::Null) => Ok(matches!(self, Self::Ne)),
            (Value::Number(a), Value::Number(b)) => Ok(match self {
                Self::Eq => a == b,
                Self::Ne => a != b,
                Self::Lt => a < b,
                Self::Le => a <= b,
                Self::Gt => a > b,
                Self::Ge => a >= b,
            }),
            (Value::Unsigned(a), Value::Unsigned(b)) => Ok(match self {
                Self::Eq => a == b,
                Self::Ne => a != b,
                Self::Lt => a < b,
                Self::Le => a <= b,
                Self::Gt => a > b,
                Self::Ge => a >= b,
            }),
            (Value::Unsigned(a), Value::Number(b)) => {
                self.evaluate(Value::Number(a as f64), Value::Number(b))
            }
            (Value::Number(a), Value::Unsigned(b)) => {
                self.evaluate(Value::Number(a), Value::Number(b as f64))
            }
            (Value::String(a), Value::String(b)) => Ok(match self {
                Self::Eq => a == b,
                Self::Ne => a != b,
                Self::Lt => a < b,
                Self::Le => a <= b,
                Self::Gt => a > b,
                Self::Ge => a >= b,
            }),
            _ => Err(MeshError::InvalidInput(
                "formula comparison operands have different types".into(),
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TagArgument {
    Id(u64),
    Name(String),
}

struct FormulaContext<'a> {
    entity: &'a SelectedEntity,
    centroid: [f64; 3],
    dimension: u8,
    quality_metric: Option<QualityMetric>,
    file: &'a MeshFile,
}

fn require_quality(measures: &mut QueryMeasures, metric: QualityMetric) -> MeshResult<()> {
    if measures
        .quality
        .is_some_and(|requested| requested != metric)
    {
        return Err(MeshError::InvalidInput(
            "one query cannot request two quality metrics".into(),
        ));
    }
    measures.quality = Some(metric);
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Identifier(String),
    Number(NumericLiteral),
    String(String),
    True,
    False,
    LeftParen,
    RightParen,
    Comma,
    Not,
    And,
    Or,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    End,
}

#[derive(Debug, Clone, PartialEq)]
struct NumericLiteral {
    value: f64,
    unsigned: Option<u64>,
}

struct Parser {
    tokens: Vec<Token>,
    index: usize,
}

impl Parser {
    fn new(source: &str) -> MeshResult<Self> {
        Ok(Self {
            tokens: tokenize(source)?,
            index: 0,
        })
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.index]
    }

    fn take(&mut self) -> Token {
        let token = self.tokens[self.index].clone();
        self.index += 1;
        token
    }

    fn parse_or(&mut self) -> MeshResult<BoolExpr> {
        let mut expression = self.parse_and()?;
        while self.peek() == &Token::Or {
            self.take();
            expression = BoolExpr::Or(Box::new(expression), Box::new(self.parse_and()?));
        }
        Ok(expression)
    }

    fn parse_and(&mut self) -> MeshResult<BoolExpr> {
        let mut expression = self.parse_not()?;
        while self.peek() == &Token::And {
            self.take();
            expression = BoolExpr::And(Box::new(expression), Box::new(self.parse_not()?));
        }
        Ok(expression)
    }

    fn parse_not(&mut self) -> MeshResult<BoolExpr> {
        if self.peek() == &Token::Not {
            self.take();
            return Ok(BoolExpr::Not(Box::new(self.parse_not()?)));
        }
        self.parse_primary_bool()
    }

    fn parse_primary_bool(&mut self) -> MeshResult<BoolExpr> {
        if self.peek() == &Token::LeftParen {
            self.take();
            let expression = self.parse_or()?;
            self.expect(Token::RightParen)?;
            return Ok(expression);
        }
        if self.peek() == &Token::True {
            self.take();
            return Ok(BoolExpr::Literal(true));
        }
        if self.peek() == &Token::False {
            self.take();
            return Ok(BoolExpr::Literal(false));
        }
        if let Token::Identifier(name) = self.peek().clone() {
            if name == "has_tag" {
                self.take();
                self.expect(Token::LeftParen)?;
                let argument = match self.take() {
                    Token::Number(NumericLiteral {
                        unsigned: Some(value),
                        ..
                    }) => TagArgument::Id(value),
                    Token::String(value) => TagArgument::Name(value),
                    _ => {
                        return Err(MeshError::InvalidInput(
                            "has_tag() expects an integer ID or string name".into(),
                        ))
                    }
                };
                self.expect(Token::RightParen)?;
                return Ok(BoolExpr::HasTag(argument));
            }
            if name == "is_supported" {
                self.take();
                self.expect(Token::LeftParen)?;
                let metric = if self.peek() == &Token::RightParen {
                    None
                } else {
                    let Token::String(value) = self.take() else {
                        return Err(MeshError::InvalidInput(
                            "is_supported() expects an optional metric string".into(),
                        ));
                    };
                    Some(QualityMetric::parse(&value).ok_or_else(|| {
                        MeshError::InvalidInput(format!("unknown quality metric {value:?}"))
                    })?)
                };
                self.expect(Token::RightParen)?;
                return Ok(BoolExpr::IsSupported(metric));
            }
        }
        let left = self.parse_value()?;
        let operator = match self.take() {
            Token::Eq => CompareOp::Eq,
            Token::Ne => CompareOp::Ne,
            Token::Lt => CompareOp::Lt,
            Token::Le => CompareOp::Le,
            Token::Gt => CompareOp::Gt,
            Token::Ge => CompareOp::Ge,
            _ => {
                return Err(MeshError::InvalidInput(
                    "formula value must be followed by a comparison operator".into(),
                ))
            }
        };
        let right = self.parse_value()?;
        Ok(BoolExpr::Compare(left, operator, right))
    }

    fn parse_value(&mut self) -> MeshResult<ValueExpr> {
        match self.take() {
            Token::Number(value) => Ok(value
                .unsigned
                .map(ValueExpr::Unsigned)
                .unwrap_or(ValueExpr::Number(value.value))),
            Token::String(value) => Ok(ValueExpr::String(value)),
            Token::Identifier(name) => Ok(ValueExpr::Field(match name.as_str() {
                "id" => FieldName::Id,
                "x" => FieldName::X,
                "y" => FieldName::Y,
                "z" => FieldName::Z,
                "tile_id" => FieldName::TileId,
                "zone_id" => FieldName::ZoneId,
                "dimension" => FieldName::Dimension,
                "quality" => FieldName::Quality,
                "boundary_distance" => FieldName::BoundaryDistance,
                "element_type" => FieldName::ElementType,
                _ => {
                    return Err(MeshError::InvalidInput(format!(
                        "unknown formula field {name:?}"
                    )))
                }
            })),
            _ => Err(MeshError::InvalidInput(
                "expected a formula field or literal".into(),
            )),
        }
    }

    fn expect(&mut self, token: Token) -> MeshResult<()> {
        if self.peek() == &token {
            self.take();
            Ok(())
        } else {
            Err(MeshError::InvalidInput(format!(
                "expected {token:?}, found {:?}",
                self.peek()
            )))
        }
    }
}

fn tokenize(source: &str) -> MeshResult<Vec<Token>> {
    let chars: Vec<char> = source.chars().collect();
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        let character = chars[index];
        if character.is_whitespace() {
            index += 1;
            continue;
        }
        let pair = chars.get(index + 1).map(|next| [character, *next]);
        if let Some(token) = match pair {
            Some(['&', '&']) => Some(Token::And),
            Some(['|', '|']) => Some(Token::Or),
            Some(['=', '=']) => Some(Token::Eq),
            Some(['!', '=']) => Some(Token::Ne),
            Some(['<', '=']) => Some(Token::Le),
            Some(['>', '=']) => Some(Token::Ge),
            _ => None,
        } {
            tokens.push(token);
            index += 2;
            continue;
        }
        match character {
            '(' => tokens.push(Token::LeftParen),
            ')' => tokens.push(Token::RightParen),
            ',' => tokens.push(Token::Comma),
            '!' => tokens.push(Token::Not),
            '<' => tokens.push(Token::Lt),
            '>' => tokens.push(Token::Gt),
            '"' | '\'' => {
                let quote = character;
                index += 1;
                let start = index;
                while index < chars.len() && chars[index] != quote {
                    index += 1;
                }
                if index == chars.len() {
                    return Err(MeshError::InvalidInput(
                        "unterminated formula string".into(),
                    ));
                }
                tokens.push(Token::String(chars[start..index].iter().collect()));
            }
            c if c.is_ascii_digit()
                || (c == '-'
                    && chars
                        .get(index + 1)
                        .is_some_and(|next| next.is_ascii_digit())) =>
            {
                let start = index;
                index += 1;
                while chars.get(index).is_some_and(|value| {
                    value.is_ascii_digit() || matches!(value, '.' | 'e' | 'E' | '+' | '-')
                }) {
                    index += 1;
                }
                let text: String = chars[start..index].iter().collect();
                let number: f64 = text.parse().map_err(|_| {
                    MeshError::InvalidInput(format!("invalid formula number {text:?}"))
                })?;
                if !number.is_finite() {
                    return Err(MeshError::InvalidInput(format!(
                        "formula number {text:?} is not finite"
                    )));
                }
                let unsigned = (!text.starts_with('-') && !text.contains(['.', 'e', 'E']))
                    .then(|| text.parse::<u64>().ok())
                    .flatten();
                tokens.push(Token::Number(NumericLiteral {
                    value: number,
                    unsigned,
                }));
                continue;
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = index;
                index += 1;
                while chars
                    .get(index)
                    .is_some_and(|value| value.is_ascii_alphanumeric() || *value == '_')
                {
                    index += 1;
                }
                let identifier: String = chars[start..index].iter().collect();
                tokens.push(match identifier.as_str() {
                    "true" => Token::True,
                    "false" => Token::False,
                    "and" => Token::And,
                    "or" => Token::Or,
                    "not" => Token::Not,
                    _ => Token::Identifier(identifier),
                });
                continue;
            }
            _ => {
                return Err(MeshError::InvalidInput(format!(
                    "unexpected formula character {character:?}"
                )))
            }
        }
        index += 1;
    }
    tokens.push(Token::End);
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formula_keeps_large_unsigned_literals_exact() {
        let formula = TypedFormula::parse("id == 9007199254740993").expect("formula");
        assert!(matches!(
            formula.expression,
            BoolExpr::Compare(
                ValueExpr::Field(FieldName::Id),
                CompareOp::Eq,
                ValueExpr::Unsigned(9_007_199_254_740_993)
            )
        ));
        assert!(TypedFormula::parse("has_tag(18446744073709551615)").is_ok());
        assert!(TypedFormula::parse("quality >= 0").is_ok());
    }

    #[test]
    fn segment_and_polygon_distances_are_exact_and_inclusive() {
        assert!(
            (point_segment_distance([0.5, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]) - 1.0).abs()
                < 1.0e-12
        );
        let face = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        assert!((point_polygon_distance([0.5, 0.5, 2.0], &face).unwrap() - 2.0).abs() < 1.0e-12);
        assert_eq!(point_polygon_distance([0.5, 0.5, 0.0], &face), Some(0.0));
        assert!(Interval::new(0.0, 2.0).contains(2.0));
    }

    #[test]
    fn tag_matching_obeys_any_all_and_empty_set_semantics() {
        let values = [1, 2];
        assert!(tags_match(&values, &BTreeSet::from([2, 3]), TagMatch::Any));
        assert!(!tags_match(&values, &BTreeSet::from([2, 3]), TagMatch::All));
        assert!(!tags_match(&values, &BTreeSet::new(), TagMatch::Any));
        assert!(tags_match(&values, &BTreeSet::new(), TagMatch::All));
    }
}
