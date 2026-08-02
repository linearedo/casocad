use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use caso_kernel::meshing::{MeshableDomain, MeshableDomainSpace, MeshableDomains};
use serde::{Deserialize, Serialize};

use crate::chunk::{MeshChunk, MeshChunkBuilder};
use crate::controls::ControlSet;
use crate::error::{MeshError, MeshResult};
use crate::schema::Bounds3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshAlgorithmCapabilities {
    pub refinement: bool,
    pub boundary_layers: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshAlgorithmDescriptor {
    pub id: &'static str,
    pub label: &'static str,
    pub dimensions: &'static [u8],
    pub capabilities: MeshAlgorithmCapabilities,
}

impl MeshAlgorithmDescriptor {
    pub fn supports_dimension(&self, dimension: u8) -> bool {
        self.dimensions.contains(&dimension)
    }
}

pub trait MeshAlgorithm: Sync {
    fn descriptor(&self) -> &'static MeshAlgorithmDescriptor;

    fn generate(
        &self,
        context: &MeshingContext<'_>,
        sink: &mut dyn MeshSink,
    ) -> MeshResult<MeshingStatistics>;
}

pub trait MeshSink {
    fn allocate_chunk_id(&mut self) -> MeshResult<u32>;
    fn emit(&mut self, chunk: MeshChunk) -> MeshResult<()>;

    fn chunk_builder(&mut self, bounds: Bounds3) -> MeshResult<MeshChunkBuilder> {
        MeshChunkBuilder::new(self.allocate_chunk_id()?, bounds)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogKind {
    Zone,
    Tag,
    Source,
    Provenance,
}

impl CatalogKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Zone => "zone",
            Self::Tag => "tag",
            Self::Source => "source",
            Self::Provenance => "provenance",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogEntry {
    pub catalog_kind: CatalogKind,
    pub id: u64,
    pub name: String,
    pub kind: String,
    pub dimension: Option<u8>,
    pub source_object_id: Option<u64>,
    pub source_region_id: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainCatalogIds {
    pub zone: u64,
    pub source: u64,
    pub wall_tag: u64,
}

#[derive(Debug, Clone)]
pub struct MeshCatalog {
    entries: Vec<CatalogEntry>,
    domains: BTreeMap<String, DomainCatalogIds>,
    tags: BTreeMap<(String, String), u64>,
}

impl MeshCatalog {
    pub fn from_domains(domains: &MeshableDomains, generator: &str) -> Self {
        let mut entries = Vec::new();
        let mut domain_ids = BTreeMap::new();
        let mut tags = BTreeMap::new();
        let mut next_tag = 1u64;
        for (index, domain) in domains.iter().enumerate() {
            let zone = index as u64 + 1;
            let source = zone;
            let source_object_id = domain.source_object_id.map(u64::from);
            entries.push(CatalogEntry {
                catalog_kind: CatalogKind::Zone,
                id: zone,
                name: domain.name.clone(),
                kind: domain.kind.as_str().into(),
                dimension: Some(domain.dimension),
                source_object_id,
                source_region_id: None,
            });
            entries.push(CatalogEntry {
                catalog_kind: CatalogKind::Source,
                id: source,
                name: domain.name.clone(),
                kind: "sdf_domain".into(),
                dimension: Some(domain.dimension),
                source_object_id,
                source_region_id: None,
            });
            for (region_index, region) in domain.boundary_regions.iter().enumerate() {
                let id = next_tag;
                next_tag += 1;
                tags.insert((domain.name.clone(), region.name.clone()), id);
                entries.push(CatalogEntry {
                    catalog_kind: CatalogKind::Tag,
                    id,
                    name: region.name.clone(),
                    kind: region.tag.clone().unwrap_or_else(|| "boundary".into()),
                    dimension: Some(domain.dimension.saturating_sub(1)),
                    source_object_id: Some(u64::from(region.owner_object_id)),
                    source_region_id: Some(region_index as u64 + 1),
                });
            }
            let wall_tag = next_tag;
            next_tag += 1;
            entries.push(CatalogEntry {
                catalog_kind: CatalogKind::Tag,
                id: wall_tag,
                name: format!("{}_wall", domain.name),
                kind: "wall".into(),
                dimension: Some(domain.dimension.saturating_sub(1)),
                source_object_id,
                source_region_id: None,
            });
            domain_ids.insert(
                domain.name.clone(),
                DomainCatalogIds {
                    zone,
                    source,
                    wall_tag,
                },
            );
        }
        entries.push(CatalogEntry {
            catalog_kind: CatalogKind::Provenance,
            id: 1,
            name: generator.into(),
            kind: "generator".into(),
            dimension: domains.iter().map(|domain| domain.dimension).max(),
            source_object_id: None,
            source_region_id: None,
        });
        Self {
            entries,
            domains: domain_ids,
            tags,
        }
    }

    pub fn entries(&self) -> &[CatalogEntry] {
        &self.entries
    }

    pub fn domain(&self, name: &str) -> MeshResult<DomainCatalogIds> {
        self.domains.get(name).copied().ok_or_else(|| {
            MeshError::InvalidInput(format!("no catalog entry for meshing domain {name:?}"))
        })
    }

    pub fn boundary_tag(&self, domain: &str, region: &str) -> Option<u64> {
        self.tags
            .get(&(domain.to_string(), region.to_string()))
            .copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationLimits {
    pub max_cells: u64,
    pub max_chunks: u64,
    pub target_chunk_bytes: usize,
}

impl Default for GenerationLimits {
    fn default() -> Self {
        Self {
            max_cells: 100_000_000,
            max_chunks: 1_000_000,
            target_chunk_bytes: 16 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeshingPhase {
    #[default]
    Generating,
    BuildingSpatialIndex,
    WritingPreviews,
    Finalizing,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshingProgress {
    pub phase: MeshingPhase,
    pub phase_completed: u64,
    pub phase_total: u64,
    pub completed_chunks: u64,
    pub cells_committed: u64,
    pub active_bytes: u64,
}

#[derive(Clone, Default)]
pub struct JobControl {
    cancelled: Arc<AtomicBool>,
    progress: Option<Arc<ProgressCallback>>,
}

#[cfg(not(target_arch = "wasm32"))]
type ProgressCallback = dyn Fn(MeshingProgress) + Send + Sync;
#[cfg(target_arch = "wasm32")]
type ProgressCallback = dyn Fn(MeshingProgress);

impl fmt::Debug for JobControl {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("JobControl")
            .field("cancelled", &self.is_cancelled())
            .field("has_progress_callback", &self.progress.is_some())
            .finish()
    }
}

impl JobControl {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_progress(
        mut self,
        callback: impl Fn(MeshingProgress) + Send + Sync + 'static,
    ) -> Self {
        self.progress = Some(Arc::new(callback));
        self
    }

    #[cfg(target_arch = "wasm32")]
    pub fn with_progress(mut self, callback: impl Fn(MeshingProgress) + 'static) -> Self {
        self.progress = Some(Arc::new(callback));
        self
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub fn check(&self) -> MeshResult<()> {
        if self.is_cancelled() {
            Err(MeshError::Cancelled)
        } else {
            Ok(())
        }
    }

    pub fn report(&self, progress: MeshingProgress) {
        if let Some(callback) = &self.progress {
            callback(progress);
        }
    }
}

#[derive(Debug)]
pub struct MeshingContext<'a> {
    pub domains: &'a MeshableDomains,
    pub target_size: f64,
    pub controls: &'a ControlSet,
    pub job_control: &'a JobControl,
    pub limits: GenerationLimits,
    pub catalog: &'a MeshCatalog,
}

impl MeshingContext<'_> {
    pub fn mesh_space(&self, domain: &MeshableDomain) -> MeshResult<Option<MeshableDomainSpace>> {
        if domain.dimension != 2 {
            return Ok(None);
        }
        domain
            .mesh_space()
            .map(Some)
            .map_err(|error| MeshError::InvalidInput(error.to_string()))
    }

    pub fn check(&self) -> MeshResult<()> {
        self.job_control.check()
    }
}

#[derive(Debug, Clone)]
pub struct MeshingRequest {
    pub domains: MeshableDomains,
    pub algorithm_id: String,
    pub controls: ControlSet,
    pub limits: GenerationLimits,
    pub job_control: JobControl,
}

impl MeshingRequest {
    pub fn validate(&self) -> MeshResult<u8> {
        if self.domains.is_empty() {
            return Err(MeshError::InvalidInput(
                "meshing requires at least one declared domain".into(),
            ));
        }
        self.controls
            .validate(&self.domains)
            .map_err(MeshError::InvalidInput)?;
        if self.limits.max_cells == 0
            || self.limits.max_chunks == 0
            || self.limits.target_chunk_bytes == 0
        {
            return Err(MeshError::InvalidInput(
                "meshing limits must be positive".into(),
            ));
        }
        let dimension = self.domains.iter().next().expect("nonempty").dimension;
        if self
            .domains
            .iter()
            .any(|domain| domain.dimension != dimension)
        {
            return Err(MeshError::InvalidInput(
                "one mesh artifact cannot mix domain dimensions".into(),
            ));
        }
        Ok(dimension)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshingStatistics {
    pub domains: u64,
    pub chunks: u64,
    pub points: u64,
    pub cells: u64,
    pub committed_batches: u64,
    pub peak_active_bytes: u64,
    pub elapsed_millis: u64,
    #[serde(default)]
    pub quality_passes: u64,
    #[serde(default)]
    pub quality_termination: QualityTermination,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityTermination {
    #[default]
    NotRun,
    Converged,
    MaxCells,
    MemoryBudget,
    IterationLimit,
}

impl fmt::Display for QualityTermination {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NotRun => "not run",
            Self::Converged => "converged",
            Self::MaxCells => "max cells",
            Self::MemoryBudget => "memory budget",
            Self::IterationLimit => "iteration limit",
        })
    }
}
