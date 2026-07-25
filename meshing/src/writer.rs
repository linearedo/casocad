use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

use arrow_ipc::writer::FileWriter;
use web_time::Instant;

use crate::algorithm::{
    CatalogEntry, MeshCatalog, MeshSink, MeshingContext, MeshingRequest, MeshingStatistics,
};
use crate::chunk::{ChunkElement, MeshChunk};
use crate::error::{MeshError, MeshResult};
use crate::row::{decoded_size, rows_to_batch, MeshRow};
use crate::schema::{
    mesh_schema, BatchDirectoryEntry, BatchRange, Bounds3, MeshCounts, MeshManifest, RowKind,
    MAX_BATCH_ROWS, MESH_SCHEMA_NAME, MESH_SCHEMA_VERSION,
};
use crate::storage::{MeshArtifact, MeshStorage};

const PREVIEW_PER_CLASS: usize = 256;

#[derive(Debug, Clone)]
pub struct MeshingOutput {
    pub artifact: MeshArtifact,
    pub statistics: MeshingStatistics,
}

#[derive(Debug, Clone)]
struct PreviewEntity {
    hash: u64,
    id: u64,
    element_type: String,
    points: Vec<[f64; 3]>,
    tag_ids: Vec<u64>,
    zone_id: Option<u64>,
    boundary: bool,
}

#[derive(Debug, Clone)]
struct LeafCandidate {
    chunk_id: u32,
    bounds: Bounds3,
    preview: Vec<PreviewEntity>,
}

#[derive(Debug, Clone)]
struct SpatialNode {
    id: u64,
    parent: Option<u64>,
    children: Vec<u64>,
    chunks: Vec<u32>,
    level: u32,
    bounds: Bounds3,
    preview: Vec<PreviewEntity>,
}

struct MeshArtifactWriter<W: Write> {
    writer: FileWriter<W>,
    dimension: u8,
    batch_index: usize,
    exact_start: usize,
    directory: Vec<BatchDirectoryEntry>,
    leaves: Vec<LeafCandidate>,
    counts: MeshCounts,
    bounds: Bounds3,
    next_chunk_id: u32,
    cells_by_source: BTreeMap<u64, u64>,
    limits: crate::algorithm::GenerationLimits,
    peak_active_bytes: u64,
}

impl<W: Write> MeshArtifactWriter<W> {
    fn new(
        output: W,
        dimension: u8,
        catalog: &[CatalogEntry],
        limits: crate::algorithm::GenerationLimits,
    ) -> MeshResult<Self> {
        let writer = FileWriter::try_new(output, mesh_schema().as_ref())?;
        let mut this = Self {
            writer,
            dimension,
            batch_index: 0,
            exact_start: 0,
            directory: Vec::new(),
            leaves: Vec::new(),
            counts: MeshCounts::default(),
            bounds: Bounds3::EMPTY,
            next_chunk_id: 1,
            cells_by_source: BTreeMap::new(),
            limits,
            peak_active_bytes: 0,
        };
        for entries in catalog.chunks(MAX_BATCH_ROWS) {
            let rows = entries.iter().map(catalog_row).collect::<Vec<_>>();
            this.write_batch(rows, None, None)?;
            this.counts.catalog += entries.len() as u64;
        }
        this.exact_start = this.batch_index;
        Ok(this)
    }

    fn write_batch(
        &mut self,
        rows: Vec<MeshRow>,
        spatial_node_id: Option<u64>,
        bounds: Option<Bounds3>,
    ) -> MeshResult<()> {
        let kind = rows
            .first()
            .map(|row| row.row_kind)
            .ok_or_else(|| MeshError::InvalidInput("cannot write an empty mesh batch".into()))?;
        let bytes = decoded_size(&rows);
        let element_types = rows
            .iter()
            .filter_map(|row| row.element_type.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let zone_ids = rows
            .iter()
            .filter_map(|row| row.zone_id)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let tag_ids = rows
            .iter()
            .flat_map(|row| row.tag_ids.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        self.writer.write(&rows_to_batch(&rows)?)?;
        self.directory.push(BatchDirectoryEntry {
            batch_index: self.batch_index,
            row_kind: kind,
            spatial_node_id,
            bounds,
            rows: rows.len(),
            decoded_bytes: bytes,
            element_types,
            zone_ids,
            tag_ids,
        });
        self.batch_index += 1;
        Ok(())
    }

    fn finish(
        mut self,
        generator_id: &str,
        settings: serde_json::Value,
    ) -> MeshResult<(W, MeshingStatistics)> {
        let exact_end = self.batch_index;
        if self.leaves.is_empty() {
            return Err(MeshError::InvalidInput(
                "meshing produced no spatial chunks".into(),
            ));
        }
        let (nodes, chunk_nodes, root) = build_spatial_tree(self.dimension, &self.leaves)?;
        for entry in &mut self.directory {
            if entry.row_kind.is_exact() {
                let chunk = entry.spatial_node_id.ok_or_else(|| {
                    MeshError::InvalidInput("exact batch is missing its chunk ID".into())
                })? as u32;
                entry.spatial_node_id = chunk_nodes.get(&chunk).copied();
            }
        }

        let preview_start = self.batch_index;
        for node in &nodes {
            self.write_preview(node)?;
        }
        let preview_end = self.batch_index;

        let spatial_start = self.batch_index;
        for chunk in nodes.chunks(MAX_BATCH_ROWS) {
            let rows = chunk.iter().map(spatial_row).collect::<Vec<_>>();
            self.write_batch(rows, None, Some(self.bounds))?;
        }
        self.counts.spatial_nodes = nodes.len() as u64;
        let spatial_end = self.batch_index;

        let directory_start = self.batch_index;
        let entries = self.directory.clone();
        for chunk in entries.chunks(MAX_BATCH_ROWS) {
            let rows = chunk.iter().map(directory_row).collect::<Vec<_>>();
            self.writer.write(&rows_to_batch(&rows)?)?;
            self.batch_index += 1;
            self.counts.directory_rows += rows.len() as u64;
        }
        let directory_end = self.batch_index;

        let manifest = MeshManifest {
            schema_name: MESH_SCHEMA_NAME.into(),
            schema_version: MESH_SCHEMA_VERSION,
            dimension: self.dimension,
            coordinate_system: "world_cartesian_meters".into(),
            counts: self.counts.clone(),
            generator_id: generator_id.into(),
            settings,
            bounds: self.bounds,
            spatial_root: root,
            catalog_batches: BatchRange::new(0, self.exact_start),
            exact_batches: BatchRange::new(self.exact_start, exact_end),
            preview_batches: BatchRange::new(preview_start, preview_end),
            spatial_batches: BatchRange::new(spatial_start, spatial_end),
            directory_batches: BatchRange::new(directory_start, directory_end),
        };
        let mut row = MeshRow::new(RowKind::Manifest);
        row.schema_version = Some(MESH_SCHEMA_VERSION);
        row.metadata = Some(serde_json::to_string(&manifest).map_err(|error| {
            MeshError::InvalidInput(format!("could not encode mesh manifest: {error}"))
        })?);
        row.counts = counts_list(&manifest.counts);
        self.writer.write(&rows_to_batch(&[row])?)?;
        self.batch_index += 1;
        let output = self.writer.into_inner()?;
        Ok((
            output,
            MeshingStatistics {
                chunks: self.leaves.len() as u64,
                points: self.counts.points,
                cells: self.counts.cells,
                committed_batches: self.batch_index as u64,
                peak_active_bytes: self.peak_active_bytes,
                ..MeshingStatistics::default()
            },
        ))
    }

    fn write_preview(&mut self, node: &SpatialNode) -> MeshResult<()> {
        if node.preview.is_empty() {
            return Ok(());
        }
        let mut point_rows = Vec::new();
        let mut element_rows = Vec::new();
        let mut ordinal = 1u64;
        let mut element_ordinal = 1u64;
        for entity in &node.preview {
            let mut point_ids = Vec::with_capacity(entity.points.len());
            for point in &entity.points {
                let id = preview_id(0xf, node.id, ordinal)?;
                ordinal += 1;
                point_ids.push(id);
                let mut row = MeshRow::new(RowKind::PreviewPoint);
                row.entity_id = Some(id);
                row.spatial_node_id = Some(node.id);
                row.position = Some(*point);
                point_rows.push(row);
            }
            let mut row = MeshRow::new(RowKind::PreviewElement);
            row.entity_id = Some(preview_id(0xe, node.id, element_ordinal)?);
            element_ordinal += 1;
            row.spatial_node_id = Some(node.id);
            row.element_type = Some(entity.element_type.clone());
            row.point_ids = point_ids;
            row.tag_ids = entity.tag_ids.clone();
            row.zone_id = entity.zone_id;
            row.boundary = Some(entity.boundary);
            element_rows.push(row);
        }
        for rows in point_rows.chunks(MAX_BATCH_ROWS) {
            self.write_batch(rows.to_vec(), Some(node.id), Some(node.bounds))?;
            self.counts.preview_points += rows.len() as u64;
        }
        for rows in element_rows.chunks(MAX_BATCH_ROWS) {
            self.write_batch(rows.to_vec(), Some(node.id), Some(node.bounds))?;
            self.counts.preview_elements += rows.len() as u64;
        }
        Ok(())
    }
}

impl<W: Write> MeshSink for MeshArtifactWriter<W> {
    fn allocate_chunk_id(&mut self) -> MeshResult<u32> {
        let id = self.next_chunk_id;
        self.next_chunk_id = self
            .next_chunk_id
            .checked_add(1)
            .ok_or_else(|| MeshError::LimitExceeded("mesh chunk ID space exhausted".into()))?;
        if u64::from(id) > self.limits.max_chunks {
            return Err(MeshError::LimitExceeded(format!(
                "mesh exceeds the configured {} chunk limit",
                self.limits.max_chunks
            )));
        }
        Ok(id)
    }

    fn emit(&mut self, chunk: MeshChunk) -> MeshResult<()> {
        chunk.validate(self.dimension)?;
        if self.leaves.iter().any(|leaf| leaf.chunk_id == chunk.id) {
            return Err(MeshError::InvalidInput(format!(
                "chunk {} was emitted more than once",
                chunk.id
            )));
        }
        if self.counts.cells.saturating_add(chunk.cells.len() as u64) > self.limits.max_cells {
            return Err(MeshError::LimitExceeded(format!(
                "mesh exceeds the configured {} cell limit",
                self.limits.max_cells
            )));
        }
        let active_bytes = chunk.decoded_bytes() as u64;
        self.peak_active_bytes = self.peak_active_bytes.max(active_bytes);
        self.bounds = self.bounds.union(chunk.bounds);
        let preview = sample_chunk(&chunk);
        let chunk_id = u64::from(chunk.id);

        if !chunk.points.is_empty() {
            let rows = chunk
                .points
                .iter()
                .map(|point| point_row(&chunk, point))
                .collect();
            self.write_batch(rows, Some(chunk_id), Some(chunk.bounds))?;
            self.counts.points += chunk
                .points
                .iter()
                .filter(|point| !point.is_ghost_in(chunk.id))
                .count() as u64;
        }
        for (kind, elements) in [
            (RowKind::Edge, &chunk.edges),
            (RowKind::Face, &chunk.faces),
            (RowKind::Cell, &chunk.cells),
        ] {
            if elements.is_empty() {
                continue;
            }
            let rows = elements
                .iter()
                .map(|element| element_row(kind, chunk.id, element))
                .collect();
            self.write_batch(rows, Some(chunk_id), Some(chunk.bounds))?;
        }
        self.counts.edges += chunk.edges.len() as u64;
        self.counts.faces += chunk.faces.len() as u64;
        self.counts.cells += chunk.cells.len() as u64;
        for source in chunk.cells.iter().filter_map(|cell| cell.source_id) {
            *self.cells_by_source.entry(source).or_default() += 1;
        }
        self.leaves.push(LeafCandidate {
            chunk_id: chunk.id,
            bounds: chunk.bounds,
            preview,
        });
        Ok(())
    }
}

pub fn run_meshing<S: MeshStorage>(
    request: MeshingRequest,
    mut storage: S,
) -> MeshResult<MeshingOutput> {
    let started = Instant::now();
    let dimension = request.validate()?;
    let algorithm = crate::registry::algorithm(&request.algorithm_id).ok_or_else(|| {
        MeshError::InvalidInput(format!(
            "mesh algorithm {:?} is not compiled in; available: {}",
            request.algorithm_id,
            crate::registry::descriptors()
                .iter()
                .map(|descriptor| descriptor.id)
                .collect::<Vec<_>>()
                .join(", ")
        ))
    })?;
    let descriptor = algorithm.descriptor();
    if !descriptor.supports_dimension(dimension) {
        return Err(MeshError::UnsupportedDimension {
            domain: request
                .domains
                .iter()
                .next()
                .map(|domain| domain.name.clone())
                .unwrap_or_default(),
            dimension,
        });
    }
    if !descriptor.capabilities.refinement && !request.controls.refinements.is_empty() {
        return Err(MeshError::Capability(format!(
            "algorithm {:?} does not support refinement controls",
            descriptor.id
        )));
    }
    if !descriptor.capabilities.boundary_layers && !request.controls.boundary_layers.is_empty() {
        return Err(MeshError::Capability(format!(
            "algorithm {:?} does not support boundary-layer controls",
            descriptor.id
        )));
    }

    let catalog = MeshCatalog::from_domains(&request.domains, descriptor.id);
    let output = storage.begin()?;
    let mut writer = MeshArtifactWriter::new(output, dimension, catalog.entries(), request.limits)?;
    let context = MeshingContext {
        domains: &request.domains,
        element_min_size: request.element_min_size,
        element_max_size: request.element_max_size,
        controls: &request.controls,
        job_control: &request.job_control,
        limits: request.limits,
        catalog: &catalog,
    };
    request.job_control.check()?;
    let generated = algorithm.generate(&context, &mut writer)?;
    request.job_control.check()?;
    for domain in request.domains.iter() {
        let source = catalog.domain(&domain.name)?.source;
        if writer.cells_by_source.get(&source).copied().unwrap_or(0) == 0 {
            return Err(empty_domain_error(domain));
        }
    }
    let settings = serde_json::json!({
        "element_min_size": request.element_min_size,
        "element_max_size": request.element_max_size,
        "controls": request.controls.metadata(),
    });
    let (output, mut statistics) = writer.finish(descriptor.id, settings)?;
    let artifact = storage.publish(output)?;
    statistics.domains = generated.domains.max(request.domains.len() as u64);
    statistics.peak_active_bytes = statistics
        .peak_active_bytes
        .max(generated.peak_active_bytes);
    statistics.elapsed_millis = started.elapsed().as_millis() as u64;
    Ok(MeshingOutput {
        artifact,
        statistics,
    })
}

fn empty_domain_error(domain: &caso_kernel::meshing::MeshableDomain) -> MeshError {
    use caso_kernel::vec3::vec3;

    let bounds = &domain.bounds;
    let center = vec3(
        (bounds.x_min + bounds.x_max) * 0.5,
        (bounds.y_min + bounds.y_max) * 0.5,
        (bounds.z_min + bounds.z_max) * 0.5,
    );
    let samples = [
        center,
        vec3(bounds.x_min, bounds.y_min, bounds.z_min),
        vec3(bounds.x_max, bounds.y_max, bounds.z_max),
    ];
    MeshError::InvalidInput(format!(
        "declared domain {:?} produced zero cells; bounds=({}, {}, {})..({}, {}, {}), sampled_sdf={:?}",
        domain.name,
        bounds.x_min,
        bounds.y_min,
        bounds.z_min,
        bounds.x_max,
        bounds.y_max,
        bounds.z_max,
        domain.domain_sdf(&samples)
    ))
}

fn catalog_row(entry: &CatalogEntry) -> MeshRow {
    let mut row = MeshRow::new(RowKind::Catalog);
    row.entity_id = Some(entry.id);
    row.catalog_kind = Some(entry.catalog_kind.as_str().into());
    row.name = Some(entry.name.clone());
    row.kind = Some(entry.kind.clone());
    row.dimension = entry.dimension;
    row.coordinate_system = Some("world_cartesian_meters".into());
    row.source_object_id = entry.source_object_id;
    row.source_region_id = entry.source_region_id;
    row
}

fn point_row(chunk: &MeshChunk, point: &crate::chunk::ChunkPoint) -> MeshRow {
    let mut row = MeshRow::new(RowKind::Point);
    row.entity_id = Some(point.id.raw());
    row.spatial_node_id = Some(u64::from(chunk.id));
    row.owner_chunk_id = Some(u64::from(point.owner_chunk_id));
    row.ghost = Some(point.is_ghost_in(chunk.id));
    row.classification = Some(point.classification.clone());
    row.position = Some(point.position);
    row.tag_ids = point.tag_ids.clone();
    row.boundary = Some(point.classification == "boundary");
    row
}

fn element_row(kind: RowKind, chunk_id: u32, element: &ChunkElement) -> MeshRow {
    let mut row = MeshRow::new(kind);
    row.entity_id = Some(element.id.raw());
    row.spatial_node_id = Some(u64::from(chunk_id));
    row.element_type = Some(element.element_type.clone());
    row.point_ids = element.point_ids.iter().map(|id| id.raw()).collect();
    row.edge_ids = element.edge_ids.iter().map(|id| id.raw()).collect();
    row.face_ids = element.face_ids.iter().map(|id| id.raw()).collect();
    row.tag_ids = element.tag_ids.clone();
    row.owner_cell_id = element.owner_cell_id.map(|id| id.raw());
    row.neighbor_cell_id = element.neighbor_cell_id.map(|id| id.raw());
    row.zone_id = element.zone_id;
    row.source_id = element.source_id;
    row.boundary = Some(element.boundary);
    row
}

fn directory_row(entry: &BatchDirectoryEntry) -> MeshRow {
    let mut row = MeshRow::new(RowKind::BatchDirectory);
    row.batch_index = Some(entry.batch_index as u64);
    row.kind = Some(entry.row_kind.as_str().into());
    row.spatial_node_id = entry.spatial_node_id;
    row.bounds = entry.bounds;
    row.rows = Some(entry.rows as u64);
    row.decoded_bytes = Some(entry.decoded_bytes as u64);
    row.element_types = entry.element_types.clone();
    row.zone_ids = entry.zone_ids.clone();
    row.tag_ids = entry.tag_ids.clone();
    row
}

fn spatial_row(node: &SpatialNode) -> MeshRow {
    let mut row = MeshRow::new(RowKind::SpatialNode);
    row.entity_id = Some(node.id);
    row.parent_id = node.parent;
    row.child_ids = node.children.clone();
    row.chunk_ids = node.chunks.iter().map(|id| u64::from(*id)).collect();
    row.level = Some(node.level);
    row.bounds = Some(node.bounds);
    row
}

fn counts_list(counts: &MeshCounts) -> Vec<u64> {
    vec![
        counts.catalog,
        counts.points,
        counts.edges,
        counts.faces,
        counts.cells,
        counts.preview_points,
        counts.preview_elements,
        counts.spatial_nodes,
        counts.directory_rows,
    ]
}

fn sample_chunk(chunk: &MeshChunk) -> Vec<PreviewEntity> {
    let points = chunk
        .points
        .iter()
        .map(|point| (point.id, point.position))
        .collect::<BTreeMap<_, _>>();
    let boundary = chunk
        .edges
        .iter()
        .chain(&chunk.faces)
        .filter(|entity| entity.boundary);
    let interior = chunk.cells.iter();
    sample_entities(boundary, &points, PREVIEW_PER_CLASS)
        .into_iter()
        .chain(sample_entities(interior, &points, PREVIEW_PER_CLASS))
        .collect()
}

fn sample_entities<'a>(
    elements: impl Iterator<Item = &'a ChunkElement>,
    points: &BTreeMap<crate::chunk::MeshId, [f64; 3]>,
    limit: usize,
) -> Vec<PreviewEntity> {
    let mut values = elements
        .filter_map(|element| {
            let geometry = element
                .point_ids
                .iter()
                .map(|id| points.get(id).copied())
                .collect::<Option<Vec<_>>>()?;
            Some(PreviewEntity {
                hash: stable_hash(element.id.raw()),
                id: element.id.raw(),
                element_type: element.element_type.clone(),
                points: geometry,
                tag_ids: element.tag_ids.clone(),
                zone_id: element.zone_id,
                boundary: element.boundary,
            })
        })
        .collect::<Vec<_>>();
    values.sort_by_key(|entity| (entity.hash, entity.id));
    values.truncate(limit);
    values
}

fn build_spatial_tree(
    dimension: u8,
    leaves: &[LeafCandidate],
) -> MeshResult<(Vec<SpatialNode>, BTreeMap<u32, u64>, u64)> {
    let mut builder = TreeBuilder {
        dimension,
        leaves,
        nodes: Vec::new(),
        chunk_nodes: BTreeMap::new(),
        next_id: leaves
            .iter()
            .map(|leaf| u64::from(leaf.chunk_id))
            .max()
            .unwrap_or(0)
            + 1,
    };
    let indices = (0..leaves.len()).collect::<Vec<_>>();
    let root = builder.build(&indices, 0, None)?;
    let parents = builder
        .nodes
        .iter()
        .flat_map(|node| node.children.iter().map(move |child| (*child, node.id)))
        .collect::<BTreeMap<_, _>>();
    for node in &mut builder.nodes {
        node.parent = parents.get(&node.id).copied();
    }
    builder.nodes.sort_by_key(|node| node.id);
    Ok((builder.nodes, builder.chunk_nodes, root))
}

struct TreeBuilder<'a> {
    dimension: u8,
    leaves: &'a [LeafCandidate],
    nodes: Vec<SpatialNode>,
    chunk_nodes: BTreeMap<u32, u64>,
    next_id: u64,
}

impl TreeBuilder<'_> {
    fn build(
        &mut self,
        indices: &[usize],
        level: u32,
        forced_bounds: Option<Bounds3>,
    ) -> MeshResult<u64> {
        let bounds = forced_bounds.unwrap_or_else(|| {
            indices.iter().fold(Bounds3::EMPTY, |bounds, index| {
                bounds.union(self.leaves[*index].bounds)
            })
        });
        if indices.len() == 1 {
            return Ok(self.leaf(indices[0], level));
        }
        if indices.len() <= 8 {
            return Ok(self.parent_of_leaves(indices, level, bounds));
        }
        let center = bounds.center();
        let bucket_count = 1usize << self.dimension;
        let mut buckets = vec![Vec::new(); bucket_count];
        for index in indices {
            let point = self.leaves[*index].bounds.center();
            let mut bucket = 0usize;
            for axis in 0..usize::from(self.dimension) {
                bucket |= usize::from(point[axis] >= center[axis]) << axis;
            }
            buckets[bucket].push(*index);
        }
        let nonempty = buckets.iter().filter(|bucket| !bucket.is_empty()).count();
        if nonempty <= 1 {
            return Ok(self.parent_of_leaves(indices, level, bounds));
        }
        let mut children = Vec::new();
        for bucket in buckets.into_iter().filter(|bucket| !bucket.is_empty()) {
            children.push(self.build(&bucket, level + 1, None)?);
        }
        let id = self.allocate_id();
        let preview = bottom_k(
            children
                .iter()
                .filter_map(|child| self.nodes.iter().find(|node| node.id == *child))
                .flat_map(|node| node.preview.iter().cloned()),
        );
        self.nodes.push(SpatialNode {
            id,
            parent: None,
            children,
            chunks: Vec::new(),
            level,
            bounds,
            preview,
        });
        Ok(id)
    }

    fn leaf(&mut self, index: usize, level: u32) -> u64 {
        let leaf = &self.leaves[index];
        let id = u64::from(leaf.chunk_id);
        self.chunk_nodes.insert(leaf.chunk_id, id);
        self.nodes.push(SpatialNode {
            id,
            parent: None,
            children: Vec::new(),
            chunks: vec![leaf.chunk_id],
            level,
            bounds: leaf.bounds,
            preview: leaf.preview.clone(),
        });
        id
    }

    fn parent_of_leaves(&mut self, indices: &[usize], level: u32, bounds: Bounds3) -> u64 {
        let children = indices
            .iter()
            .map(|index| self.leaf(*index, level + 1))
            .collect::<Vec<_>>();
        let id = self.allocate_id();
        let preview = bottom_k(
            indices
                .iter()
                .flat_map(|index| self.leaves[*index].preview.iter().cloned()),
        );
        self.nodes.push(SpatialNode {
            id,
            parent: None,
            children,
            chunks: Vec::new(),
            level,
            bounds,
            preview,
        });
        id
    }

    fn allocate_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

fn bottom_k(values: impl Iterator<Item = PreviewEntity>) -> Vec<PreviewEntity> {
    let mut boundary = values.collect::<Vec<_>>();
    boundary.sort_by_key(|entity| (entity.boundary, entity.hash, entity.id));
    let mut tagged = boundary
        .iter()
        .filter(|entity| entity.boundary)
        .take(PREVIEW_PER_CLASS)
        .cloned()
        .collect::<Vec<_>>();
    tagged.extend(
        boundary
            .into_iter()
            .filter(|entity| !entity.boundary)
            .take(PREVIEW_PER_CLASS),
    );
    tagged
}

fn stable_hash(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn preview_id(namespace: u64, node: u64, ordinal: u64) -> MeshResult<u64> {
    if node >= (1 << 28) || ordinal >= (1 << 32) {
        return Err(MeshError::LimitExceeded(
            "preview ID space exhausted".into(),
        ));
    }
    Ok((namespace << 60) | (node << 32) | ordinal)
}
