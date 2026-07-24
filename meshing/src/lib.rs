//! Shared incremental meshing and Arrow IPC v3 storage for casoCAD.

#![deny(unsafe_code)]

mod advancing_front;
mod algorithm;
mod chunk;
mod error;
mod file;
mod query;
mod registry;
mod renderer;
mod row;
mod schema;
mod storage;
mod uniform;
mod writer;

pub mod controls;
pub mod convert;
pub mod quality;

pub use algorithm::{
    CatalogEntry, CatalogKind, DomainCatalogIds, GenerationLimits, JobControl, MeshAlgorithm,
    MeshAlgorithmCapabilities, MeshAlgorithmDescriptor, MeshCatalog, MeshSink, MeshingContext,
    MeshingProgress, MeshingRequest, MeshingStatistics,
};
pub use chunk::{ChunkElement, ChunkPoint, MeshChunk, MeshChunkBuilder, MeshId};
pub use controls::{BoundaryLayerControl, ControlRegion, ControlSet, RefinementControl};
pub use error::{MeshError, MeshResult};
pub use file::{BatchView, MeshAuditReport, MeshFile, MeshReadSession, MeshStorageKind};
pub use query::{
    EntityKind, Interval, MeshQuery, MeshQueryResult, MeshQueryService, QualityFilter,
    SelectedEntity, TypedFormula,
};
pub use registry::{algorithm, descriptors};
pub use renderer::{
    IncrementalLodPreparation, LodTargetSelection, MeshPreparationStats, MeshRendererCache,
    MeshTileDetail, MeshTileKey, MeshView, PreparedMeshTile, RenderLine, RendererBudgets,
    LOD_COLLAPSE_PIXELS, LOD_EXPAND_PIXELS,
};
pub use schema::{
    arrow_schema, BatchDirectoryEntry, BatchRange, Bounds3, MeshCounts, MeshManifest, RowKind,
    MESH_FILE_EXTENSION, MESH_SCHEMA_NAME, MESH_SCHEMA_VERSION,
};
#[cfg(not(target_arch = "wasm32"))]
pub use storage::NativeFileStorage;
pub use storage::{CappedBuffer, MemoryStorage, MeshArtifact, MeshStorage};
pub use writer::{run_meshing, MeshingOutput};

pub use caso_kernel;
