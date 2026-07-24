use std::sync::Arc;

use arrow_array::builder::{
    BooleanBuilder, Float64Builder, LargeListBuilder, StringBuilder, UInt32Builder, UInt64Builder,
    UInt8Builder,
};
use arrow_array::{ArrayRef, RecordBatch};

use crate::error::{MeshError, MeshResult};
use crate::schema::{mesh_schema, Bounds3, RowKind, MAX_BATCH_ROWS};

#[derive(Debug, Clone)]
pub(crate) struct MeshRow {
    pub row_kind: RowKind,
    pub entity_id: Option<u64>,
    pub spatial_node_id: Option<u64>,
    pub owner_chunk_id: Option<u64>,
    pub ghost: Option<bool>,
    pub classification: Option<String>,
    pub position: Option<[f64; 3]>,
    pub element_type: Option<String>,
    pub point_ids: Vec<u64>,
    pub edge_ids: Vec<u64>,
    pub face_ids: Vec<u64>,
    pub tag_ids: Vec<u64>,
    pub owner_cell_id: Option<u64>,
    pub neighbor_cell_id: Option<u64>,
    pub zone_id: Option<u64>,
    pub source_id: Option<u64>,
    pub boundary: Option<bool>,
    pub catalog_kind: Option<String>,
    pub name: Option<String>,
    pub kind: Option<String>,
    pub dimension: Option<u8>,
    pub coordinate_system: Option<String>,
    pub source_object_id: Option<u64>,
    pub source_region_id: Option<u64>,
    pub parent_id: Option<u64>,
    pub child_ids: Vec<u64>,
    pub chunk_ids: Vec<u64>,
    pub level: Option<u32>,
    pub bounds: Option<Bounds3>,
    pub batch_index: Option<u64>,
    pub rows: Option<u64>,
    pub decoded_bytes: Option<u64>,
    pub element_types: Vec<String>,
    pub zone_ids: Vec<u64>,
    pub schema_version: Option<u32>,
    pub counts: Vec<u64>,
    pub metadata: Option<String>,
}

impl MeshRow {
    pub fn new(row_kind: RowKind) -> Self {
        Self {
            row_kind,
            entity_id: None,
            spatial_node_id: None,
            owner_chunk_id: None,
            ghost: None,
            classification: None,
            position: None,
            element_type: None,
            point_ids: Vec::new(),
            edge_ids: Vec::new(),
            face_ids: Vec::new(),
            tag_ids: Vec::new(),
            owner_cell_id: None,
            neighbor_cell_id: None,
            zone_id: None,
            source_id: None,
            boundary: None,
            catalog_kind: None,
            name: None,
            kind: None,
            dimension: None,
            coordinate_system: None,
            source_object_id: None,
            source_region_id: None,
            parent_id: None,
            child_ids: Vec::new(),
            chunk_ids: Vec::new(),
            level: None,
            bounds: None,
            batch_index: None,
            rows: None,
            decoded_bytes: None,
            element_types: Vec::new(),
            zone_ids: Vec::new(),
            schema_version: None,
            counts: Vec::new(),
            metadata: None,
        }
    }
}

fn strings(values: impl Iterator<Item = Option<String>>, len: usize) -> ArrayRef {
    let mut builder = StringBuilder::with_capacity(len, len * 8);
    for value in values {
        builder.append_option(value);
    }
    Arc::new(builder.finish())
}

fn u64s(values: impl Iterator<Item = Option<u64>>, len: usize) -> ArrayRef {
    let mut builder = UInt64Builder::with_capacity(len);
    for value in values {
        builder.append_option(value);
    }
    Arc::new(builder.finish())
}

fn u32s(values: impl Iterator<Item = Option<u32>>, len: usize) -> ArrayRef {
    let mut builder = UInt32Builder::with_capacity(len);
    for value in values {
        builder.append_option(value);
    }
    Arc::new(builder.finish())
}

fn u8s(values: impl Iterator<Item = Option<u8>>, len: usize) -> ArrayRef {
    let mut builder = UInt8Builder::with_capacity(len);
    for value in values {
        builder.append_option(value);
    }
    Arc::new(builder.finish())
}

fn f64s(values: impl Iterator<Item = Option<f64>>, len: usize) -> ArrayRef {
    let mut builder = Float64Builder::with_capacity(len);
    for value in values {
        builder.append_option(value);
    }
    Arc::new(builder.finish())
}

fn bools(values: impl Iterator<Item = Option<bool>>, len: usize) -> ArrayRef {
    let mut builder = BooleanBuilder::with_capacity(len);
    for value in values {
        builder.append_option(value);
    }
    Arc::new(builder.finish())
}

fn list_u64(values: impl Iterator<Item = Vec<u64>>, len: usize) -> ArrayRef {
    let mut builder = LargeListBuilder::with_capacity(UInt64Builder::new(), len);
    for values in values {
        builder.values().append_slice(&values);
        builder.append(true);
    }
    Arc::new(builder.finish())
}

fn list_strings(values: impl Iterator<Item = Vec<String>>, len: usize) -> ArrayRef {
    let mut builder = LargeListBuilder::with_capacity(StringBuilder::new(), len);
    for values in values {
        for value in values {
            builder.values().append_value(value);
        }
        builder.append(true);
    }
    Arc::new(builder.finish())
}

pub(crate) fn rows_to_batch(rows: &[MeshRow]) -> MeshResult<RecordBatch> {
    if rows.is_empty() {
        return Err(MeshError::InvalidInput(
            "Arrow record batches must not be empty".into(),
        ));
    }
    if rows.len() > MAX_BATCH_ROWS {
        return Err(MeshError::LimitExceeded(format!(
            "record batch has {} rows; maximum is {MAX_BATCH_ROWS}",
            rows.len()
        )));
    }
    let kind = rows[0].row_kind;
    if rows.iter().any(|row| row.row_kind != kind) {
        return Err(MeshError::InvalidInput(
            "each Arrow record batch must contain one row kind".into(),
        ));
    }
    let n = rows.len();
    let arrays: Vec<ArrayRef> = vec![
        strings(rows.iter().map(|row| Some(row.row_kind.as_str().into())), n),
        u64s(rows.iter().map(|row| row.entity_id), n),
        u64s(rows.iter().map(|row| row.spatial_node_id), n),
        u64s(rows.iter().map(|row| row.owner_chunk_id), n),
        bools(rows.iter().map(|row| row.ghost), n),
        strings(rows.iter().map(|row| row.classification.clone()), n),
        f64s(rows.iter().map(|row| row.position.map(|p| p[0])), n),
        f64s(rows.iter().map(|row| row.position.map(|p| p[1])), n),
        f64s(rows.iter().map(|row| row.position.map(|p| p[2])), n),
        strings(rows.iter().map(|row| row.element_type.clone()), n),
        list_u64(rows.iter().map(|row| row.point_ids.clone()), n),
        list_u64(rows.iter().map(|row| row.edge_ids.clone()), n),
        list_u64(rows.iter().map(|row| row.face_ids.clone()), n),
        list_u64(rows.iter().map(|row| row.tag_ids.clone()), n),
        u64s(rows.iter().map(|row| row.owner_cell_id), n),
        u64s(rows.iter().map(|row| row.neighbor_cell_id), n),
        u64s(rows.iter().map(|row| row.zone_id), n),
        u64s(rows.iter().map(|row| row.source_id), n),
        bools(rows.iter().map(|row| row.boundary), n),
        strings(rows.iter().map(|row| row.catalog_kind.clone()), n),
        strings(rows.iter().map(|row| row.name.clone()), n),
        strings(rows.iter().map(|row| row.kind.clone()), n),
        u8s(rows.iter().map(|row| row.dimension), n),
        strings(rows.iter().map(|row| row.coordinate_system.clone()), n),
        u64s(rows.iter().map(|row| row.source_object_id), n),
        u64s(rows.iter().map(|row| row.source_region_id), n),
        u64s(rows.iter().map(|row| row.parent_id), n),
        list_u64(rows.iter().map(|row| row.child_ids.clone()), n),
        list_u64(rows.iter().map(|row| row.chunk_ids.clone()), n),
        u32s(rows.iter().map(|row| row.level), n),
        f64s(rows.iter().map(|row| row.bounds.map(|b| b.min[0])), n),
        f64s(rows.iter().map(|row| row.bounds.map(|b| b.max[0])), n),
        f64s(rows.iter().map(|row| row.bounds.map(|b| b.min[1])), n),
        f64s(rows.iter().map(|row| row.bounds.map(|b| b.max[1])), n),
        f64s(rows.iter().map(|row| row.bounds.map(|b| b.min[2])), n),
        f64s(rows.iter().map(|row| row.bounds.map(|b| b.max[2])), n),
        u64s(rows.iter().map(|row| row.batch_index), n),
        u64s(rows.iter().map(|row| row.rows), n),
        u64s(rows.iter().map(|row| row.decoded_bytes), n),
        list_strings(rows.iter().map(|row| row.element_types.clone()), n),
        list_u64(rows.iter().map(|row| row.zone_ids.clone()), n),
        u32s(rows.iter().map(|row| row.schema_version), n),
        list_u64(rows.iter().map(|row| row.counts.clone()), n),
        strings(rows.iter().map(|row| row.metadata.clone()), n),
    ];
    RecordBatch::try_new(mesh_schema(), arrays).map_err(Into::into)
}

pub(crate) fn decoded_size(rows: &[MeshRow]) -> usize {
    rows.iter()
        .map(|row| {
            192 + row.classification.as_ref().map_or(0, String::len)
                + row.element_type.as_ref().map_or(0, String::len)
                + row.name.as_ref().map_or(0, String::len)
                + row.kind.as_ref().map_or(0, String::len)
                + row.metadata.as_ref().map_or(0, String::len)
                + (row.point_ids.len()
                    + row.edge_ids.len()
                    + row.face_ids.len()
                    + row.tag_ids.len()
                    + row.child_ids.len()
                    + row.chunk_ids.len()
                    + row.zone_ids.len()
                    + row.counts.len())
                    * 8
                + row.element_types.iter().map(String::len).sum::<usize>()
        })
        .sum()
}
