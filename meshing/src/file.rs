use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::{Cursor, Read, Seek};
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, Float64Array, LargeListArray, RecordBatch, StringArray, UInt32Array,
    UInt64Array,
};
use arrow_ipc::reader::FileReader;
use arrow_schema::SchemaRef;

use crate::algorithm::{JobControl, MeshingProgress};
use crate::error::{MeshError, MeshResult};
use crate::renderer::MeshView;
use crate::schema::{
    element_dimension, expected_points, mesh_schema, BatchDirectoryEntry, BatchRange, Bounds3,
    MeshCounts, MeshManifest, RowKind, MAX_BATCH_BYTES, MAX_BATCH_ROWS, MESH_SCHEMA_NAME,
    MESH_SCHEMA_VERSION,
};

trait ReadSeek: Read + Seek {}
impl<T: Read + Seek> ReadSeek for T {}

type CatalogNames = BTreeMap<(String, u64), String>;
type CatalogSources = BTreeMap<(String, u64), u64>;

pub struct MeshReadSession {
    reader: FileReader<Box<dyn ReadSeek>>,
}

impl fmt::Debug for MeshReadSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MeshReadSession")
            .field("batches", &self.reader.num_batches())
            .finish()
    }
}

impl MeshReadSession {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn native(path: impl AsRef<Path>) -> MeshResult<Self> {
        Self::new(Box::new(std::fs::File::open(path)?))
    }

    pub fn memory(bytes: Arc<[u8]>) -> MeshResult<Self> {
        Self::new(Box::new(Cursor::new(bytes)))
    }

    fn new(reader: Box<dyn ReadSeek>) -> MeshResult<Self> {
        Ok(Self {
            reader: FileReader::try_new(reader, None)?,
        })
    }

    pub fn schema(&self) -> SchemaRef {
        self.reader.schema()
    }

    pub fn num_batches(&self) -> usize {
        self.reader.num_batches()
    }

    pub fn read_batch(&mut self, index: usize) -> MeshResult<RecordBatch> {
        self.reader.set_index(index)?;
        self.reader
            .next()
            .transpose()?
            .ok_or_else(|| MeshError::InvalidFile(format!("Arrow batch {index} is missing")))
    }
}

#[derive(Debug, Clone)]
enum MeshSource {
    #[cfg(not(target_arch = "wasm32"))]
    Native(PathBuf),
    Memory(Arc<[u8]>),
}

impl MeshSource {
    fn session(&self) -> MeshResult<MeshReadSession> {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Self::Native(path) => MeshReadSession::native(path),
            Self::Memory(bytes) => MeshReadSession::memory(bytes.clone()),
        }
    }
}

#[derive(Debug, Clone)]
struct SpatialNodeMeta {
    parent: Option<u64>,
    children: Vec<u64>,
    chunks: Vec<u64>,
    level: u32,
    bounds: Bounds3,
}

#[derive(Debug)]
pub struct MeshFile {
    source: MeshSource,
    manifest: MeshManifest,
    directory: Vec<BatchDirectoryEntry>,
    catalog_names: BTreeMap<(String, u64), String>,
    catalog_source_objects: BTreeMap<(String, u64), u64>,
    spatial_nodes: BTreeMap<u64, SpatialNodeMeta>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshStorageKind {
    NativeFile,
    Memory,
}

#[derive(Debug, Clone)]
pub struct BatchView {
    entry: BatchDirectoryEntry,
    batch: RecordBatch,
}

impl BatchView {
    pub fn directory_entry(&self) -> &BatchDirectoryEntry {
        &self.entry
    }

    pub fn record_batch(&self) -> &RecordBatch {
        &self.batch
    }

    pub fn row_kind(&self) -> RowKind {
        self.entry.row_kind
    }

    pub fn len(&self) -> usize {
        self.batch.num_rows()
    }

    pub fn is_empty(&self) -> bool {
        self.batch.num_rows() == 0
    }

    pub(crate) fn strings(&self, name: &str) -> MeshResult<&StringArray> {
        downcast(&self.batch, name)
    }

    pub(crate) fn u64s(&self, name: &str) -> MeshResult<&UInt64Array> {
        downcast(&self.batch, name)
    }

    pub(crate) fn f64s(&self, name: &str) -> MeshResult<&Float64Array> {
        downcast(&self.batch, name)
    }

    pub(crate) fn bools(&self, name: &str) -> MeshResult<&BooleanArray> {
        downcast(&self.batch, name)
    }

    pub(crate) fn lists(&self, name: &str) -> MeshResult<&LargeListArray> {
        downcast(&self.batch, name)
    }
}

fn downcast<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> MeshResult<&'a T> {
    batch
        .column_by_name(name)
        .and_then(|array| array.as_any().downcast_ref::<T>())
        .ok_or_else(|| MeshError::InvalidFile(format!("missing or invalid Arrow column {name:?}")))
}

impl MeshFile {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_native(path: impl AsRef<Path>) -> MeshResult<Self> {
        Self::open(MeshSource::Native(path.as_ref().to_path_buf()))
    }

    pub fn from_memory(bytes: Arc<[u8]>) -> MeshResult<Self> {
        Self::open(MeshSource::Memory(bytes))
    }

    pub fn from_wasm_bytes(bytes: Arc<[u8]>) -> MeshResult<Self> {
        Self::from_memory(bytes)
    }

    fn open(source: MeshSource) -> MeshResult<Self> {
        let mut session = source.session()?;
        validate_schema(&session.schema())?;
        let count = session.num_batches();
        if count < 5 {
            return Err(MeshError::InvalidFile(
                "Arrow v3 mesh is missing required batch sections".into(),
            ));
        }
        let manifest_batch = session.read_batch(count - 1)?;
        if batch_kind(&manifest_batch)? != RowKind::Manifest || manifest_batch.num_rows() != 1 {
            return Err(MeshError::InvalidFile(
                "the final Arrow batch must be one manifest row".into(),
            ));
        }
        let metadata = downcast::<StringArray>(&manifest_batch, "metadata")?;
        if metadata.is_null(0) {
            return Err(MeshError::InvalidFile(
                "manifest metadata is missing".into(),
            ));
        }
        let manifest: MeshManifest = serde_json::from_str(metadata.value(0)).map_err(|error| {
            MeshError::InvalidFile(format!("manifest metadata is malformed: {error}"))
        })?;
        validate_manifest(&manifest, count)?;

        let directory = read_directory(&mut session, &manifest)?;
        validate_directory(&directory, &manifest)?;
        let (catalog_names, catalog_source_objects) =
            read_catalog(&mut session, &manifest.catalog_batches)?;
        let spatial_nodes = read_spatial_nodes(&mut session, &manifest.spatial_batches)?;
        validate_spatial_tree(&spatial_nodes, &manifest, &directory)?;
        Ok(Self {
            source,
            manifest,
            directory,
            catalog_names,
            catalog_source_objects,
            spatial_nodes,
        })
    }

    pub fn manifest(&self) -> &MeshManifest {
        &self.manifest
    }

    pub fn batch_view(&self, batch_index: usize) -> MeshResult<BatchView> {
        let entry = self
            .directory
            .iter()
            .find(|entry| entry.batch_index == batch_index)
            .cloned()
            .ok_or_else(|| {
                MeshError::InvalidInput(format!(
                    "batch {batch_index} is not present in the v3 directory"
                ))
            })?;
        let mut session = self.source.session()?;
        let batch = session.read_batch(batch_index)?;
        validate_decoded_batch(&batch, &entry)?;
        Ok(BatchView { entry, batch })
    }

    pub fn entity_batches(&self, kind: RowKind) -> impl Iterator<Item = &BatchDirectoryEntry> {
        self.directory
            .iter()
            .filter(move |entry| entry.row_kind == kind)
    }

    pub fn tile_batches(
        &self,
        spatial_node_id: u64,
        kind: RowKind,
    ) -> impl Iterator<Item = &BatchDirectoryEntry> {
        self.directory.iter().filter(move |entry| {
            entry.row_kind == kind && entry.spatial_node_id == Some(spatial_node_id)
        })
    }

    pub fn catalog_name(&self, catalog_kind: &str, id: u64) -> Option<&str> {
        self.catalog_names
            .get(&(catalog_kind.to_string(), id))
            .map(String::as_str)
    }

    pub fn catalog_source_object(&self, catalog_kind: &str, id: u64) -> Option<u64> {
        self.catalog_source_objects
            .get(&(catalog_kind.to_string(), id))
            .copied()
    }

    pub fn source_path(&self) -> Option<&Path> {
        match &self.source {
            #[cfg(not(target_arch = "wasm32"))]
            MeshSource::Native(path) => Some(path),
            MeshSource::Memory(_) => None,
        }
    }

    pub fn memory_bytes(&self) -> Option<&[u8]> {
        match &self.source {
            #[cfg(not(target_arch = "wasm32"))]
            MeshSource::Native(_) => None,
            MeshSource::Memory(bytes) => Some(bytes),
        }
    }

    pub fn bytes(&self) -> &[u8] {
        self.memory_bytes()
            .expect("native MeshFile has no RAM-sized byte slice; use source_path")
    }

    pub fn storage_kind(&self) -> MeshStorageKind {
        match &self.source {
            #[cfg(not(target_arch = "wasm32"))]
            MeshSource::Native(_) => MeshStorageKind::NativeFile,
            MeshSource::Memory(_) => MeshStorageKind::Memory,
        }
    }

    pub(crate) fn candidate_leaf_tiles(&self, bounds: Bounds3) -> BTreeSet<u64> {
        let mut result = BTreeSet::new();
        let mut stack = vec![self.manifest.spatial_root];
        while let Some(id) = stack.pop() {
            let Some(node) = self.spatial_nodes.get(&id) else {
                continue;
            };
            if !node.bounds.intersects(bounds) {
                continue;
            }
            if node.children.is_empty() {
                result.insert(id);
            } else {
                stack.extend(node.children.iter().copied());
            }
        }
        result
    }

    pub(crate) fn select_lod_nodes(
        &self,
        view: MeshView,
        expand_pixels: f32,
        collapse_pixels: f32,
        refined: &mut BTreeSet<u64>,
    ) -> Vec<(u64, bool)> {
        let mut selected = Vec::new();
        let mut stack = vec![self.manifest.spatial_root];
        while let Some(id) = stack.pop() {
            let node = &self.spatial_nodes[&id];
            let Some(pixels) = view.projected_pixels(node.bounds) else {
                continue;
            };
            let refine = if refined.contains(&id) {
                pixels >= collapse_pixels
            } else {
                pixels > expand_pixels
            };
            if refine {
                refined.insert(id);
            } else {
                refined.remove(&id);
            }
            if !node.children.is_empty() && refine {
                stack.extend(node.children.iter().rev().copied());
            } else {
                selected.push((id, node.children.is_empty() && refine));
            }
        }
        selected
    }

    pub(crate) fn nearest_leaf_nodes(&self, focus: [f64; 3], limit: usize) -> Vec<u64> {
        // ponytail: O(spatial nodes); use a tree nearest-neighbor walk if
        // focus selection itself becomes measurable on million-chunk files.
        let mut nearest = Vec::with_capacity(limit + 1);
        for (id, node) in self
            .spatial_nodes
            .iter()
            .filter(|(_, node)| node.children.is_empty())
        {
            let bounds_distance = point_bounds_distance_squared(focus, node.bounds);
            let center_distance = point_distance_squared(focus, node.bounds.center());
            nearest.push((bounds_distance, center_distance, *id));
            nearest.sort_by(|a, b| {
                a.0.total_cmp(&b.0)
                    .then_with(|| a.1.total_cmp(&b.1))
                    .then_with(|| a.2.cmp(&b.2))
            });
            nearest.truncate(limit);
        }
        nearest.into_iter().map(|(_, _, id)| id).collect()
    }

    pub fn full_audit(&self, control: &JobControl) -> MeshResult<MeshAuditReport> {
        let mut ids = BTreeSet::new();
        let mut owners = BTreeMap::<u64, (u64, [f64; 3])>::new();
        let mut counts = MeshCounts::default();
        let leaves = self
            .spatial_nodes
            .iter()
            .filter(|(_, node)| node.children.is_empty())
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        for (completed, leaf) in leaves.iter().enumerate() {
            control.check()?;
            let mut local_points = BTreeSet::new();
            for entry in self.tile_batches(*leaf, RowKind::Point) {
                let batch = self.batch_view(entry.batch_index)?;
                let point_ids = batch.u64s("entity_id")?;
                let owner_chunks = batch.u64s("owner_chunk_id")?;
                let ghosts = batch.bools("ghost")?;
                let x = batch.f64s("x")?;
                let y = batch.f64s("y")?;
                let z = batch.f64s("z")?;
                for row in 0..batch.len() {
                    let id = point_ids.value(row);
                    local_points.insert(id);
                    let position = [x.value(row), y.value(row), z.value(row)];
                    if !ghosts.value(row) {
                        if !ids.insert(id)
                            || owners
                                .insert(id, (owner_chunks.value(row), position))
                                .is_some()
                        {
                            return Err(MeshError::InvalidFile(format!(
                                "point {id} has duplicate owner rows"
                            )));
                        }
                        counts.points += 1;
                    } else if let Some((owner, expected)) = owners.get(&id) {
                        if *owner != owner_chunks.value(row) || *expected != position {
                            return Err(MeshError::InvalidFile(format!(
                                "ghost point {id} disagrees with its owner"
                            )));
                        }
                    }
                }
            }
            for kind in [RowKind::Edge, RowKind::Face, RowKind::Cell] {
                for entry in self.tile_batches(*leaf, kind) {
                    let batch = self.batch_view(entry.batch_index)?;
                    let entity_ids = batch.u64s("entity_id")?;
                    let connectivity = batch.lists("point_ids")?;
                    for row in 0..batch.len() {
                        let id = entity_ids.value(row);
                        if !ids.insert(id) {
                            return Err(MeshError::InvalidFile(format!(
                                "entity ID {id} is not globally unique"
                            )));
                        }
                        if list_u64(connectivity, row)?
                            .iter()
                            .any(|point| !local_points.contains(point))
                        {
                            return Err(MeshError::InvalidFile(format!(
                                "entity {id} references a point absent from its exact chunk"
                            )));
                        }
                        match kind {
                            RowKind::Edge => counts.edges += 1,
                            RowKind::Face => counts.faces += 1,
                            RowKind::Cell => counts.cells += 1,
                            _ => unreachable!(),
                        }
                    }
                }
            }
            control.report(MeshingProgress {
                completed_chunks: completed as u64 + 1,
                cells_committed: counts.cells,
                active_bytes: 0,
            });
        }
        if counts.points != self.manifest.counts.points
            || counts.edges != self.manifest.counts.edges
            || counts.faces != self.manifest.counts.faces
            || counts.cells != self.manifest.counts.cells
        {
            return Err(MeshError::InvalidFile(format!(
                "full audit counts {counts:?} disagree with manifest {:?}",
                self.manifest.counts
            )));
        }
        Ok(MeshAuditReport {
            exact_batches: self
                .directory
                .iter()
                .filter(|entry| entry.row_kind.is_exact())
                .count() as u64,
            entities: counts.entity_count(),
        })
    }
}

fn point_bounds_distance_squared(point: [f64; 3], bounds: Bounds3) -> f64 {
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
        .map(|distance| distance * distance)
        .sum()
}

fn point_distance_squared(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|axis| (a[axis] - b[axis]).powi(2)).sum()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshAuditReport {
    pub exact_batches: u64,
    pub entities: u64,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn validate_metadata_path(path: &Path) -> MeshResult<()> {
    MeshFile::open_native(path).map(|_| ())
}

fn validate_schema(schema: &SchemaRef) -> MeshResult<()> {
    let name = schema.metadata().get("casocad.schema").map(String::as_str);
    let version = schema
        .metadata()
        .get("casocad.schema_version")
        .and_then(|value| value.parse::<u32>().ok());
    if version != Some(MESH_SCHEMA_VERSION) {
        return Err(MeshError::UnsupportedFileVersion(version.unwrap_or(0)));
    }
    if name != Some(MESH_SCHEMA_NAME) || schema.fields() != mesh_schema().fields() {
        return Err(MeshError::InvalidFile(format!(
            "Arrow schema does not match {MESH_SCHEMA_NAME}"
        )));
    }
    Ok(())
}

fn validate_manifest(manifest: &MeshManifest, batch_count: usize) -> MeshResult<()> {
    if manifest.schema_name != MESH_SCHEMA_NAME
        || manifest.schema_version != MESH_SCHEMA_VERSION
        || !matches!(manifest.dimension, 2 | 3)
        || manifest.coordinate_system != "world_cartesian_meters"
        || !manifest.bounds.is_valid()
    {
        return Err(MeshError::InvalidFile(
            "manifest schema identity, dimension, coordinates, or bounds are invalid".into(),
        ));
    }
    let ranges = [
        &manifest.catalog_batches,
        &manifest.exact_batches,
        &manifest.preview_batches,
        &manifest.spatial_batches,
        &manifest.directory_batches,
    ];
    if ranges[0].start != 0
        || ranges.windows(2).any(|pair| pair[0].end != pair[1].start)
        || ranges.last().expect("five ranges").end + 1 != batch_count
        || ranges.iter().any(|range| range.start > range.end)
        || manifest.catalog_batches.start == manifest.catalog_batches.end
        || manifest.exact_batches.start == manifest.exact_batches.end
        || manifest.spatial_batches.start == manifest.spatial_batches.end
        || manifest.directory_batches.start == manifest.directory_batches.end
    {
        return Err(MeshError::InvalidFile(
            "manifest batch ranges are not ordered, complete, and non-overlapping".into(),
        ));
    }
    Ok(())
}

fn read_directory(
    session: &mut MeshReadSession,
    manifest: &MeshManifest,
) -> MeshResult<Vec<BatchDirectoryEntry>> {
    let mut entries = Vec::new();
    for index in manifest.directory_batches.as_range() {
        let batch = session.read_batch(index)?;
        if batch_kind(&batch)? != RowKind::BatchDirectory {
            return Err(MeshError::InvalidFile(
                "directory range contains a non-directory batch".into(),
            ));
        }
        let indices = downcast::<UInt64Array>(&batch, "batch_index")?;
        let kinds = downcast::<StringArray>(&batch, "kind")?;
        let nodes = downcast::<UInt64Array>(&batch, "spatial_node_id")?;
        let rows = downcast::<UInt64Array>(&batch, "rows")?;
        let bytes = downcast::<UInt64Array>(&batch, "decoded_bytes")?;
        let element_types = downcast::<LargeListArray>(&batch, "element_types")?;
        let zone_ids = downcast::<LargeListArray>(&batch, "zone_ids")?;
        let tag_ids = downcast::<LargeListArray>(&batch, "tag_ids")?;
        for row in 0..batch.num_rows() {
            entries.push(BatchDirectoryEntry {
                batch_index: indices.value(row) as usize,
                row_kind: RowKind::parse(kinds.value(row)).ok_or_else(|| {
                    MeshError::InvalidFile("directory contains an unknown row kind".into())
                })?,
                spatial_node_id: (!nodes.is_null(row)).then(|| nodes.value(row)),
                bounds: row_bounds(&batch, row)?,
                rows: rows.value(row) as usize,
                decoded_bytes: bytes.value(row) as usize,
                element_types: list_strings(element_types, row)?,
                zone_ids: list_u64(zone_ids, row)?,
                tag_ids: list_u64(tag_ids, row)?,
            });
        }
    }
    Ok(entries)
}

fn validate_directory(
    directory: &[BatchDirectoryEntry],
    manifest: &MeshManifest,
) -> MeshResult<()> {
    let expected = manifest.directory_batches.start;
    if directory.len() != expected {
        return Err(MeshError::InvalidFile(format!(
            "directory has {} rows for {expected} indexed batches",
            directory.len()
        )));
    }
    let mut seen = BTreeSet::new();
    for entry in directory {
        if !seen.insert(entry.batch_index)
            || entry.batch_index >= expected
            || entry.rows == 0
            || entry.rows > MAX_BATCH_ROWS
            || entry.decoded_bytes > MAX_BATCH_BYTES
        {
            return Err(MeshError::InvalidFile(
                "directory batch index, row count, or decoded size is invalid".into(),
            ));
        }
        let valid_range = match entry.row_kind {
            RowKind::Catalog => &manifest.catalog_batches,
            kind if kind.is_exact() => &manifest.exact_batches,
            RowKind::PreviewPoint | RowKind::PreviewElement => &manifest.preview_batches,
            RowKind::SpatialNode => &manifest.spatial_batches,
            _ => {
                return Err(MeshError::InvalidFile(
                    "directory references a forbidden row kind".into(),
                ))
            }
        };
        if !valid_range.contains(entry.batch_index) {
            return Err(MeshError::InvalidFile(
                "directory row kind disagrees with manifest ranges".into(),
            ));
        }
        if (entry.row_kind.is_exact()
            || matches!(
                entry.row_kind,
                RowKind::PreviewPoint | RowKind::PreviewElement
            ))
            && (entry.spatial_node_id.is_none() || entry.bounds.is_none())
        {
            return Err(MeshError::InvalidFile(
                "spatial data batch lacks node ID or bounds".into(),
            ));
        }
    }
    Ok(())
}

fn read_catalog(
    session: &mut MeshReadSession,
    range: &BatchRange,
) -> MeshResult<(CatalogNames, CatalogSources)> {
    let mut names = BTreeMap::new();
    let mut sources = BTreeMap::new();
    for index in range.as_range() {
        let batch = session.read_batch(index)?;
        if batch_kind(&batch)? != RowKind::Catalog {
            return Err(MeshError::InvalidFile(
                "catalog range contains a non-catalog batch".into(),
            ));
        }
        let ids = downcast::<UInt64Array>(&batch, "entity_id")?;
        let kinds = downcast::<StringArray>(&batch, "catalog_kind")?;
        let labels = downcast::<StringArray>(&batch, "name")?;
        let source_objects = downcast::<UInt64Array>(&batch, "source_object_id")?;
        for row in 0..batch.num_rows() {
            if ids.is_null(row) || kinds.is_null(row) || labels.is_null(row) {
                return Err(MeshError::InvalidFile(
                    "catalog row lacks kind, ID, or name".into(),
                ));
            }
            let key = (kinds.value(row).to_string(), ids.value(row));
            if names
                .insert(key.clone(), labels.value(row).into())
                .is_some()
            {
                return Err(MeshError::InvalidFile(
                    "catalog IDs must be unique within each kind".into(),
                ));
            }
            if !source_objects.is_null(row) {
                sources.insert(key, source_objects.value(row));
            }
        }
    }
    Ok((names, sources))
}

fn read_spatial_nodes(
    session: &mut MeshReadSession,
    range: &BatchRange,
) -> MeshResult<BTreeMap<u64, SpatialNodeMeta>> {
    let mut nodes = BTreeMap::new();
    for index in range.as_range() {
        let batch = session.read_batch(index)?;
        if batch_kind(&batch)? != RowKind::SpatialNode {
            return Err(MeshError::InvalidFile(
                "spatial range contains a non-spatial batch".into(),
            ));
        }
        let ids = downcast::<UInt64Array>(&batch, "entity_id")?;
        let parents = downcast::<UInt64Array>(&batch, "parent_id")?;
        let children = downcast::<LargeListArray>(&batch, "child_ids")?;
        let chunks = downcast::<LargeListArray>(&batch, "chunk_ids")?;
        let levels = downcast::<UInt32Array>(&batch, "level")?;
        for row in 0..batch.num_rows() {
            let id = ids.value(row);
            let value = SpatialNodeMeta {
                parent: (!parents.is_null(row)).then(|| parents.value(row)),
                children: list_u64(children, row)?,
                chunks: list_u64(chunks, row)?,
                level: levels.value(row),
                bounds: row_bounds(&batch, row)?
                    .ok_or_else(|| MeshError::InvalidFile("spatial node has no bounds".into()))?,
            };
            if nodes.insert(id, value).is_some() {
                return Err(MeshError::InvalidFile(
                    "spatial node IDs must be unique".into(),
                ));
            }
        }
    }
    Ok(nodes)
}

fn validate_spatial_tree(
    nodes: &BTreeMap<u64, SpatialNodeMeta>,
    manifest: &MeshManifest,
    directory: &[BatchDirectoryEntry],
) -> MeshResult<()> {
    if nodes.len() as u64 != manifest.counts.spatial_nodes
        || !nodes.contains_key(&manifest.spatial_root)
    {
        return Err(MeshError::InvalidFile(
            "spatial tree count or root is invalid".into(),
        ));
    }
    let mut reached = BTreeSet::new();
    let mut stack = vec![manifest.spatial_root];
    while let Some(id) = stack.pop() {
        if !reached.insert(id) {
            return Err(MeshError::InvalidFile(
                "spatial tree contains a cycle or shared child".into(),
            ));
        }
        let node = nodes.get(&id).ok_or_else(|| {
            MeshError::InvalidFile("spatial tree references a missing node".into())
        })?;
        if !node.bounds.is_valid()
            || (!node.children.is_empty() && !node.chunks.is_empty())
            || (node.children.is_empty() && node.chunks.len() != 1)
        {
            return Err(MeshError::InvalidFile(
                "spatial node bounds or leaf payload is invalid".into(),
            ));
        }
        for child in &node.children {
            let child_node = nodes.get(child).ok_or_else(|| {
                MeshError::InvalidFile("spatial tree references a missing child".into())
            })?;
            if child_node.parent != Some(id)
                || child_node.level <= node.level
                || !node.bounds.contains(child_node.bounds.min)
                || !node.bounds.contains(child_node.bounds.max)
            {
                return Err(MeshError::InvalidFile(
                    "spatial child has invalid parent, level, or bounds".into(),
                ));
            }
            stack.push(*child);
        }
    }
    if reached.len() != nodes.len() {
        return Err(MeshError::InvalidFile(
            "spatial tree contains unreachable nodes".into(),
        ));
    }
    for entry in directory.iter().filter(|entry| {
        entry.row_kind.is_exact()
            || matches!(
                entry.row_kind,
                RowKind::PreviewPoint | RowKind::PreviewElement
            )
    }) {
        if !nodes.contains_key(&entry.spatial_node_id.expect("validated")) {
            return Err(MeshError::InvalidFile(
                "directory references an unknown spatial node".into(),
            ));
        }
    }
    Ok(())
}

fn validate_decoded_batch(batch: &RecordBatch, entry: &BatchDirectoryEntry) -> MeshResult<()> {
    if batch_kind(batch)? != entry.row_kind || batch.num_rows() != entry.rows {
        return Err(MeshError::InvalidFile(format!(
            "decoded batch {} disagrees with its directory row",
            entry.batch_index
        )));
    }
    if entry.row_kind == RowKind::Point || entry.row_kind == RowKind::PreviewPoint {
        let x = downcast::<Float64Array>(batch, "x")?;
        let y = downcast::<Float64Array>(batch, "y")?;
        let z = downcast::<Float64Array>(batch, "z")?;
        for row in 0..batch.num_rows() {
            if x.is_null(row)
                || y.is_null(row)
                || z.is_null(row)
                || [x.value(row), y.value(row), z.value(row)]
                    .into_iter()
                    .any(|value| !value.is_finite())
            {
                return Err(MeshError::InvalidFile(
                    "point batch contains null or non-finite coordinates".into(),
                ));
            }
        }
    }
    if matches!(
        entry.row_kind,
        RowKind::Edge | RowKind::Face | RowKind::Cell | RowKind::PreviewElement
    ) {
        let types = downcast::<StringArray>(batch, "element_type")?;
        let points = downcast::<LargeListArray>(batch, "point_ids")?;
        for row in 0..batch.num_rows() {
            if types.is_null(row) {
                return Err(MeshError::InvalidFile(
                    "element batch has a null element type".into(),
                ));
            }
            let count = list_u64(points, row)?.len();
            if expected_points(types.value(row)).is_none_or(|expected| !expected.contains(&count)) {
                return Err(MeshError::InvalidFile(format!(
                    "{} row has invalid arity {count}",
                    types.value(row)
                )));
            }
            if entry.row_kind == RowKind::Cell && element_dimension(types.value(row)).is_none() {
                return Err(MeshError::InvalidFile("unknown cell element type".into()));
            }
        }
    }
    Ok(())
}

fn batch_kind(batch: &RecordBatch) -> MeshResult<RowKind> {
    let kinds = downcast::<StringArray>(batch, "row_kind")?;
    if batch.num_rows() == 0 || kinds.is_null(0) {
        return Err(MeshError::InvalidFile(
            "record batches must contain at least one row kind".into(),
        ));
    }
    let kind = RowKind::parse(kinds.value(0))
        .ok_or_else(|| MeshError::InvalidFile("unknown row kind".into()))?;
    if (0..batch.num_rows()).any(|row| kinds.is_null(row) || kinds.value(row) != kind.as_str()) {
        return Err(MeshError::InvalidFile(
            "record batch mixes row kinds".into(),
        ));
    }
    Ok(kind)
}

fn row_bounds(batch: &RecordBatch, row: usize) -> MeshResult<Option<Bounds3>> {
    let arrays = [
        downcast::<Float64Array>(batch, "x_min")?,
        downcast::<Float64Array>(batch, "x_max")?,
        downcast::<Float64Array>(batch, "y_min")?,
        downcast::<Float64Array>(batch, "y_max")?,
        downcast::<Float64Array>(batch, "z_min")?,
        downcast::<Float64Array>(batch, "z_max")?,
    ];
    if arrays.iter().all(|array| array.is_null(row)) {
        return Ok(None);
    }
    if arrays.iter().any(|array| array.is_null(row)) {
        return Err(MeshError::InvalidFile(
            "bounds columns must be all null or all populated".into(),
        ));
    }
    let bounds = Bounds3 {
        min: [
            arrays[0].value(row),
            arrays[2].value(row),
            arrays[4].value(row),
        ],
        max: [
            arrays[1].value(row),
            arrays[3].value(row),
            arrays[5].value(row),
        ],
    };
    if !bounds.is_valid() {
        return Err(MeshError::InvalidFile("row bounds are invalid".into()));
    }
    Ok(Some(bounds))
}

fn list_u64(array: &LargeListArray, row: usize) -> MeshResult<Vec<u64>> {
    let values = array.value(row);
    values
        .as_any()
        .downcast_ref::<UInt64Array>()
        .map(|values| values.values().to_vec())
        .ok_or_else(|| MeshError::InvalidFile("list column must contain u64 values".into()))
}

fn list_strings(array: &LargeListArray, row: usize) -> MeshResult<Vec<String>> {
    let values = array.value(row);
    values
        .as_any()
        .downcast_ref::<StringArray>()
        .map(|values| values.iter().flatten().map(str::to_string).collect())
        .ok_or_else(|| MeshError::InvalidFile("list column must contain strings".into()))
}
