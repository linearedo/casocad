use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use serde::{Deserialize, Serialize};

pub const MESH_SCHEMA_VERSION: u32 = 3;
pub const MESH_SCHEMA_NAME: &str = "casocad.casomesh.arrow.v3";
pub const MESH_FILE_EXTENSION: &str = "casomesh.arrow";
pub const MAX_BATCH_ROWS: usize = 65_536;
pub const MAX_BATCH_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowKind {
    Catalog,
    Point,
    Edge,
    Face,
    Cell,
    PreviewPoint,
    PreviewElement,
    SpatialNode,
    BatchDirectory,
    Manifest,
}

impl RowKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Catalog => "catalog",
            Self::Point => "point",
            Self::Edge => "edge",
            Self::Face => "face",
            Self::Cell => "cell",
            Self::PreviewPoint => "preview_point",
            Self::PreviewElement => "preview_element",
            Self::SpatialNode => "spatial_node",
            Self::BatchDirectory => "batch_directory",
            Self::Manifest => "manifest",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "catalog" => Self::Catalog,
            "point" => Self::Point,
            "edge" => Self::Edge,
            "face" => Self::Face,
            "cell" => Self::Cell,
            "preview_point" => Self::PreviewPoint,
            "preview_element" => Self::PreviewElement,
            "spatial_node" => Self::SpatialNode,
            "batch_directory" => Self::BatchDirectory,
            "manifest" => Self::Manifest,
            _ => return None,
        })
    }

    pub const fn is_exact(self) -> bool {
        matches!(self, Self::Point | Self::Edge | Self::Face | Self::Cell)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bounds3 {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Bounds3 {
    pub const EMPTY: Self = Self {
        min: [f64::INFINITY; 3],
        max: [f64::NEG_INFINITY; 3],
    };

    pub fn from_points(points: impl IntoIterator<Item = [f64; 3]>) -> Self {
        let mut bounds = Self::EMPTY;
        for point in points {
            bounds.include(point);
        }
        bounds
    }

    pub fn include(&mut self, point: [f64; 3]) {
        for (axis, value) in point.into_iter().enumerate() {
            self.min[axis] = self.min[axis].min(value);
            self.max[axis] = self.max[axis].max(value);
        }
    }

    pub fn union(mut self, other: Self) -> Self {
        if other.is_valid() {
            self.include(other.min);
            self.include(other.max);
        }
        self
    }

    pub fn is_valid(self) -> bool {
        self.min.into_iter().chain(self.max).all(f64::is_finite)
            && (0..3).all(|axis| self.min[axis] <= self.max[axis])
    }

    pub fn intersects(self, other: Self) -> bool {
        (0..3).all(|axis| {
            self.min[axis] <= other.max[axis] + coordinate_tolerance(self, other, axis)
                && self.max[axis] >= other.min[axis] - coordinate_tolerance(self, other, axis)
        })
    }

    pub fn contains(self, point: [f64; 3]) -> bool {
        (0..3).all(|axis| {
            let scale = self.min[axis]
                .abs()
                .max(self.max[axis].abs())
                .max(point[axis].abs())
                .max(1.0);
            let tolerance = scale * f64::EPSILON * 16.0;
            point[axis] >= self.min[axis] - tolerance && point[axis] <= self.max[axis] + tolerance
        })
    }

    pub fn expanded(self, distance: f64) -> Self {
        Self {
            min: self.min.map(|value| value - distance),
            max: self.max.map(|value| value + distance),
        }
    }

    pub fn center(self) -> [f64; 3] {
        std::array::from_fn(|axis| (self.min[axis] + self.max[axis]) * 0.5)
    }
}

fn coordinate_tolerance(a: Bounds3, b: Bounds3, axis: usize) -> f64 {
    [a.min[axis], a.max[axis], b.min[axis], b.max[axis]]
        .into_iter()
        .filter(|value| value.is_finite())
        .map(f64::abs)
        .fold(1.0, f64::max)
        * f64::EPSILON
        * 16.0
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshCounts {
    pub catalog: u64,
    pub points: u64,
    pub edges: u64,
    pub faces: u64,
    pub cells: u64,
    pub preview_points: u64,
    pub preview_elements: u64,
    pub spatial_nodes: u64,
    pub directory_rows: u64,
}

impl MeshCounts {
    pub fn entity_count(&self) -> u64 {
        self.points + self.edges + self.faces + self.cells
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchRange {
    pub start: usize,
    pub end: usize,
}

impl BatchRange {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn as_range(&self) -> Range<usize> {
        self.start..self.end
    }

    pub fn contains(&self, index: usize) -> bool {
        self.start <= index && index < self.end
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchDirectoryEntry {
    pub batch_index: usize,
    pub row_kind: RowKind,
    pub spatial_node_id: Option<u64>,
    pub bounds: Option<Bounds3>,
    pub rows: usize,
    pub decoded_bytes: usize,
    #[serde(default)]
    pub element_types: Vec<String>,
    #[serde(default)]
    pub zone_ids: Vec<u64>,
    #[serde(default)]
    pub tag_ids: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshManifest {
    pub schema_name: String,
    pub schema_version: u32,
    pub dimension: u8,
    pub coordinate_system: String,
    pub counts: MeshCounts,
    pub generator_id: String,
    pub settings: serde_json::Value,
    pub bounds: Bounds3,
    pub spatial_root: u64,
    pub catalog_batches: BatchRange,
    pub exact_batches: BatchRange,
    pub preview_batches: BatchRange,
    pub spatial_batches: BatchRange,
    pub directory_batches: BatchRange,
}

pub(crate) fn mesh_schema() -> SchemaRef {
    let list_u64 = || DataType::LargeList(Arc::new(Field::new("item", DataType::UInt64, true)));
    let list_utf8 = || DataType::LargeList(Arc::new(Field::new("item", DataType::Utf8, true)));
    let mut metadata = HashMap::new();
    metadata.insert("casocad.schema".to_string(), MESH_SCHEMA_NAME.to_string());
    metadata.insert(
        "casocad.schema_version".to_string(),
        MESH_SCHEMA_VERSION.to_string(),
    );
    metadata.insert("casocad.compression".to_string(), "none".to_string());
    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("row_kind", DataType::Utf8, false),
            Field::new("entity_id", DataType::UInt64, true),
            Field::new("spatial_node_id", DataType::UInt64, true),
            Field::new("owner_chunk_id", DataType::UInt64, true),
            Field::new("ghost", DataType::Boolean, true),
            Field::new("classification", DataType::Utf8, true),
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("z", DataType::Float64, true),
            Field::new("element_type", DataType::Utf8, true),
            Field::new("point_ids", list_u64(), true),
            Field::new("edge_ids", list_u64(), true),
            Field::new("face_ids", list_u64(), true),
            Field::new("tag_ids", list_u64(), true),
            Field::new("owner_cell_id", DataType::UInt64, true),
            Field::new("neighbor_cell_id", DataType::UInt64, true),
            Field::new("zone_id", DataType::UInt64, true),
            Field::new("source_id", DataType::UInt64, true),
            Field::new("boundary", DataType::Boolean, true),
            Field::new("catalog_kind", DataType::Utf8, true),
            Field::new("name", DataType::Utf8, true),
            Field::new("kind", DataType::Utf8, true),
            Field::new("dimension", DataType::UInt8, true),
            Field::new("coordinate_system", DataType::Utf8, true),
            Field::new("source_object_id", DataType::UInt64, true),
            Field::new("source_region_id", DataType::UInt64, true),
            Field::new("parent_id", DataType::UInt64, true),
            Field::new("child_ids", list_u64(), true),
            Field::new("chunk_ids", list_u64(), true),
            Field::new("level", DataType::UInt32, true),
            Field::new("x_min", DataType::Float64, true),
            Field::new("x_max", DataType::Float64, true),
            Field::new("y_min", DataType::Float64, true),
            Field::new("y_max", DataType::Float64, true),
            Field::new("z_min", DataType::Float64, true),
            Field::new("z_max", DataType::Float64, true),
            Field::new("batch_index", DataType::UInt64, true),
            Field::new("rows", DataType::UInt64, true),
            Field::new("decoded_bytes", DataType::UInt64, true),
            Field::new("element_types", list_utf8(), true),
            Field::new("zone_ids", list_u64(), true),
            Field::new("schema_version", DataType::UInt32, true),
            Field::new("counts", list_u64(), true),
            Field::new("metadata", DataType::Utf8, true),
        ],
        metadata,
    ))
}

pub fn arrow_schema() -> SchemaRef {
    mesh_schema()
}

pub(crate) fn element_dimension(element_type: &str) -> Option<u8> {
    Some(match element_type {
        "point1" => 0,
        "edge2" | "edge3" => 1,
        "tri3" | "tri6" | "quad4" | "quad8" | "quad9" | "polygon" => 2,
        "tet4" | "tet10" | "hex8" | "hex20" | "hex27" | "prism6" | "prism15" | "pyramid5"
        | "pyramid13" | "polyhedron" => 3,
        _ => return None,
    })
}

pub(crate) fn expected_points(element_type: &str) -> Option<Range<usize>> {
    let count = match element_type {
        "point1" => 1,
        "edge2" => 2,
        "edge3" | "tri3" => 3,
        "tri6" | "prism6" => 6,
        "quad4" | "tet4" => 4,
        "quad8" | "hex8" => 8,
        "quad9" => 9,
        "tet10" => 10,
        "pyramid5" => 5,
        "pyramid13" => 13,
        "prism15" => 15,
        "hex20" => 20,
        "hex27" => 27,
        "polygon" | "polyhedron" => return Some(3..usize::MAX),
        _ => return None,
    };
    Some(count..count + 1)
}
