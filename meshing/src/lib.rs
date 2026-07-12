//! caso-meshing — FEA/CFD mesh artifacts for casoWASM.
//!
//! The Arrow IPC mesh artifact ports `core/meshing/artifact.py`: rows of
//! (`element_type`, `vertices`, `tag_name`) where `vertices` is a list of
//! xyz float64 triples, plus a JSON `metadata` entry in the schema metadata.
//! Files written here round-trip against Python's `read_mesh_artifact`.
//! ("Meshing" = solver discretization; viewport surfaces live elsewhere.)

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{FixedSizeListBuilder, Float64Builder, ListBuilder, StringBuilder};
use arrow_array::{Array, FixedSizeListArray, ListArray, RecordBatch, StringArray};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema};

pub use caso_kernel;

/// One mesh element: its type, flat vertex list, and physics tag.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshElement {
    pub element_type: String,
    /// (N, 3) world-space vertices in meters, f64 like all analysis data.
    pub vertices: Vec<[f64; 3]>,
    pub tag_name: String,
}

fn artifact_schema(metadata_json: &str) -> Schema {
    let vertex = Field::new("item", DataType::Float64, true);
    let vertex_type = DataType::FixedSizeList(Arc::new(vertex), 3);
    let vertices_item = Field::new("item", vertex_type.clone(), true);
    let schema = Schema::new(vec![
        Field::new("element_type", DataType::Utf8, false),
        Field::new("vertices", DataType::List(Arc::new(vertices_item)), false),
        Field::new("tag_name", DataType::Utf8, false),
    ]);
    let mut map = HashMap::new();
    map.insert("metadata".to_string(), metadata_json.to_string());
    schema.with_metadata(map)
}

/// Serialize mesh elements to Arrow IPC file bytes (the .arrow artifact).
/// `metadata` must be a JSON object; it is stored compact and key-sorted,
/// matching the Python writer.
pub fn write_mesh_artifact(
    elements: &[MeshElement],
    metadata: &serde_json::Value,
) -> Result<Vec<u8>, String> {
    let compact = to_sorted_compact_json(metadata);
    let schema = Arc::new(artifact_schema(&compact));

    let mut element_type = StringBuilder::new();
    let mut tag_name = StringBuilder::new();
    let vertex_builder = FixedSizeListBuilder::new(Float64Builder::new(), 3);
    let mut vertices = ListBuilder::new(vertex_builder);
    for element in elements {
        element_type.append_value(&element.element_type);
        tag_name.append_value(&element.tag_name);
        for vertex in &element.vertices {
            for component in vertex {
                vertices.values().values().append_value(*component);
            }
            vertices.values().append(true);
        }
        vertices.append(true);
    }
    let element_type: StringArray = element_type.finish();
    let vertices: ListArray = vertices.finish();
    let tag_name: StringArray = tag_name.finish();
    // Builders emit nullable item fields; recast to the declared schema.
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("element_type", DataType::Utf8, false),
            Field::new("vertices", vertices.data_type().clone(), false),
            Field::new("tag_name", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(element_type),
            Arc::new(vertices),
            Arc::new(tag_name),
        ],
    )
    .map_err(|error| error.to_string())?;

    let mut bytes = Vec::new();
    {
        let mut writer =
            FileWriter::try_new(&mut bytes, &schema).map_err(|error| error.to_string())?;
        let coerced = RecordBatch::try_new(schema.clone(), batch.columns().to_vec())
            .unwrap_or(batch);
        writer.write(&coerced).map_err(|error| error.to_string())?;
        writer.finish().map_err(|error| error.to_string())?;
    }
    Ok(bytes)
}

/// Read an Arrow IPC mesh artifact back into elements + metadata.
pub fn read_mesh_artifact(
    bytes: &[u8],
) -> Result<(Vec<MeshElement>, serde_json::Value), String> {
    let cursor = std::io::Cursor::new(bytes);
    let reader = FileReader::try_new(cursor, None).map_err(|error| error.to_string())?;
    let metadata_json = reader
        .schema()
        .metadata()
        .get("metadata")
        .cloned()
        .unwrap_or_else(|| "{}".to_string());
    let metadata: serde_json::Value =
        serde_json::from_str(&metadata_json).map_err(|error| error.to_string())?;
    let mut elements = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|error| error.to_string())?;
        let element_type = batch
            .column_by_name("element_type")
            .and_then(|array| array.as_any().downcast_ref::<StringArray>())
            .ok_or("missing element_type column")?;
        let tag_name = batch
            .column_by_name("tag_name")
            .and_then(|array| array.as_any().downcast_ref::<StringArray>())
            .ok_or("missing tag_name column")?;
        let vertices = batch
            .column_by_name("vertices")
            .and_then(|array| array.as_any().downcast_ref::<ListArray>())
            .ok_or("missing vertices column")?;
        for row in 0..batch.num_rows() {
            let row_vertices = vertices.value(row);
            let fixed = row_vertices
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or("vertices rows must be fixed-size xyz lists")?;
            let mut points = Vec::with_capacity(fixed.len());
            for index in 0..fixed.len() {
                let triple = fixed.value(index);
                let values = triple
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .ok_or("vertex components must be float64")?;
                points.push([values.value(0), values.value(1), values.value(2)]);
            }
            elements.push(MeshElement {
                element_type: element_type.value(row).to_string(),
                vertices: points,
                tag_name: tag_name.value(row).to_string(),
            });
        }
    }
    Ok((elements, metadata))
}

/// Compact, key-sorted JSON — byte-compatible with Python's
/// `json.dumps(..., separators=(",", ":"), sort_keys=True)`.
fn to_sorted_compact_json(value: &serde_json::Value) -> String {
    fn write(value: &serde_json::Value, out: &mut String) {
        match value {
            serde_json::Value::Object(map) => {
                out.push('{');
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                for (index, key) in keys.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    out.push_str(&serde_json::to_string(key).expect("string"));
                    out.push(':');
                    write(&map[*key], out);
                }
                out.push('}');
            }
            serde_json::Value::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    write(item, out);
                }
                out.push(']');
            }
            other => out.push_str(&other.to_string()),
        }
    }
    let mut out = String::new();
    write(value, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Port of `test_mesh_artifact_round_trip`.
    #[test]
    fn artifact_round_trip() {
        let elements = vec![
            MeshElement {
                element_type: "point".to_string(),
                vertices: vec![[0.0, 0.0, 0.0]],
                tag_name: "fluid_internal".to_string(),
            },
            MeshElement {
                element_type: "triangle".to_string(),
                vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                tag_name: "wall".to_string(),
            },
        ];
        let bytes =
            write_mesh_artifact(&elements, &json!({"source": "test"})).expect("write");
        let (read, metadata) = read_mesh_artifact(&bytes).expect("read");
        assert_eq!(metadata, json!({"source": "test"}));
        assert_eq!(read, elements);
        assert_eq!(read[1].vertices[2], [0.0, 1.0, 0.0]);
    }
}
