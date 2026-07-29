//! Shared incremental meshing and Arrow IPC v3 storage for casoCAD.

#![deny(unsafe_code)]

mod advancing_front;
#[path = "advancing_front_2D.rs"]
mod advancing_front_2d;
#[path = "advancing_front_3D.rs"]
mod advancing_front_3d;
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
mod writer;

pub mod controls;
pub mod convert;
pub mod quality;

pub use algorithm::{
    CatalogEntry, CatalogKind, DomainCatalogIds, GenerationLimits, JobControl, MeshAlgorithm,
    MeshAlgorithmCapabilities, MeshAlgorithmDescriptor, MeshCatalog, MeshSink, MeshingContext,
    MeshingPhase, MeshingProgress, MeshingRequest, MeshingStatistics,
};
pub use chunk::{ChunkElement, ChunkPoint, MeshChunk, MeshChunkBuilder, MeshId};
pub use controls::{BoundaryLayerControl, ControlRegion, ControlSet, RefinementControl};
pub use error::{MeshError, MeshResult};
pub use file::{
    BatchView, MeshAuditCursor, MeshAuditProgress, MeshAuditReport, MeshAuditStep, MeshFile,
    MeshReadSession, MeshStorageKind,
};
pub use query::{
    BoundaryIndex, EntityKind, Interval, MeshQuery, MeshQueryCursor, MeshQueryPlan,
    MeshQueryResult, MeshQueryService, MeshQueryStatistics, QualityFilter, QueryBudget,
    QueryCancellation, QueryMeasures, QueryProgress, QueryStatisticsAccumulator, QueryStep,
    SelectedEntity, TagFilter, TagMatch, TagScope, TypedFormula,
};
pub use registry::{algorithm, descriptors};
pub use renderer::{
    quality_band, quality_color, IncrementalLodPreparation, LodTargetSelection,
    MeshPreparationStats, MeshRenderStyle, MeshRendererCache, MeshTileDetail, MeshTileKey,
    MeshView, PreparedMeshTile, RenderLine, RenderLineColor, RendererBudgets, LOD_COLLAPSE_PIXELS,
    LOD_EXPAND_PIXELS, QUALITY_BANDS,
};
pub use schema::{
    arrow_schema, BatchDirectoryEntry, BatchRange, Bounds3, MeshCounts, MeshManifest, RowKind,
    MESH_FILE_EXTENSION, MESH_SCHEMA_NAME, MESH_SCHEMA_VERSION,
};
#[cfg(not(target_arch = "wasm32"))]
pub use storage::NativeFileStorage;
pub use storage::{CappedBuffer, MemoryArtifact, MemoryStorage, MeshArtifact, MeshStorage};
pub use writer::{run_meshing, MeshingOutput};

pub use caso_kernel;
