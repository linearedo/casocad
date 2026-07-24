use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write;

use arrow_array::Array;

use crate::{MeshError, MeshFile, MeshResult, RowKind};

pub fn write(mesh: &MeshFile) -> MeshResult<Vec<u8>> {
    let mut bytes = Vec::new();
    write_to(mesh, &mut bytes)?;
    Ok(bytes)
}

pub fn write_to(mesh: &MeshFile, output: &mut impl Write) -> MeshResult<()> {
    let dimension = mesh.manifest().dimension;
    let mut point_numbers = BTreeMap::<u64, usize>::new();
    writeln!(output, "NDIME= {dimension}")?;
    writeln!(output, "NPOIN= {}", mesh.manifest().counts.points)?;
    for entry in mesh.entity_batches(RowKind::Point) {
        let batch = mesh.batch_view(entry.batch_index)?;
        let ids = batch.record_batch().column_by_name("entity_id").unwrap();
        let ids = ids
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .unwrap();
        let ghosts = batch.record_batch().column_by_name("ghost").unwrap();
        let ghosts = ghosts
            .as_any()
            .downcast_ref::<arrow_array::BooleanArray>()
            .unwrap();
        let x = batch.record_batch().column_by_name("x").unwrap();
        let x = x
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .unwrap();
        let y = batch.record_batch().column_by_name("y").unwrap();
        let y = y
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .unwrap();
        let z = batch.record_batch().column_by_name("z").unwrap();
        let z = z
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .unwrap();
        for row in 0..batch.len() {
            if ghosts.value(row) {
                continue;
            }
            let id = ids.value(row);
            let point_number = point_numbers.len();
            if point_numbers.insert(id, point_number).is_some() {
                return Err(MeshError::InvalidFile(format!(
                    "point {id} has more than one owner row"
                )));
            }
            if dimension == 2 {
                writeln!(
                    output,
                    "{:.17e} {:.17e} {}",
                    x.value(row),
                    y.value(row),
                    point_number
                )?;
            } else {
                writeln!(
                    output,
                    "{:.17e} {:.17e} {:.17e} {}",
                    x.value(row),
                    y.value(row),
                    z.value(row),
                    point_number
                )?;
            }
        }
    }

    writeln!(output, "NELEM= {}", mesh.manifest().counts.cells)?;
    let mut element_number = 0_usize;
    for entry in mesh.entity_batches(RowKind::Cell) {
        let batch = mesh.batch_view(entry.batch_index)?;
        let types = batch.record_batch().column_by_name("element_type").unwrap();
        let types = types
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap();
        let point_ids = batch.record_batch().column_by_name("point_ids").unwrap();
        let point_ids = point_ids
            .as_any()
            .downcast_ref::<arrow_array::LargeListArray>()
            .unwrap();
        for row in 0..batch.len() {
            let code = su2_code(types.value(row)).ok_or_else(|| {
                MeshError::InvalidInput(format!(
                    "SU2 export does not support {:?}",
                    types.value(row)
                ))
            })?;
            let values = point_ids.value(row);
            let values = values
                .as_any()
                .downcast_ref::<arrow_array::UInt64Array>()
                .unwrap();
            let mut line = code.to_string();
            for id in values.values() {
                let number = point_numbers.get(id).ok_or_else(|| {
                    MeshError::InvalidFile(format!("element references missing point {id}"))
                })?;
                let _ = write!(line, " {number}");
            }
            let _ = write!(line, " {element_number}");
            writeln!(output, "{line}")?;
            element_number += 1;
        }
    }

    let boundary_kind = if dimension == 2 {
        RowKind::Edge
    } else {
        RowKind::Face
    };
    let mut markers = BTreeMap::<u64, u64>::new();
    for entry in mesh.entity_batches(boundary_kind) {
        let batch = mesh.batch_view(entry.batch_index)?;
        let boundary = batch.record_batch().column_by_name("boundary").unwrap();
        let boundary = boundary
            .as_any()
            .downcast_ref::<arrow_array::BooleanArray>()
            .unwrap();
        let tags = batch.record_batch().column_by_name("tag_ids").unwrap();
        let tags = tags
            .as_any()
            .downcast_ref::<arrow_array::LargeListArray>()
            .unwrap();
        for row in 0..batch.len() {
            if boundary.is_null(row) || !boundary.value(row) {
                continue;
            }
            let tag_values = tags.value(row);
            let tag_values = tag_values
                .as_any()
                .downcast_ref::<arrow_array::UInt64Array>()
                .unwrap();
            let tag = tag_values.values().first().copied().unwrap_or(0);
            *markers.entry(tag).or_default() += 1;
        }
    }
    writeln!(output, "NMARK= {}", markers.len())?;
    for (tag, count) in markers {
        let label = mesh.catalog_name("tag", tag).unwrap_or("boundary");
        writeln!(output, "MARKER_TAG= {label}")?;
        writeln!(output, "MARKER_ELEMS= {count}")?;
        for entry in mesh.entity_batches(boundary_kind) {
            let batch = mesh.batch_view(entry.batch_index)?;
            let boundary = batch.record_batch().column_by_name("boundary").unwrap();
            let boundary = boundary
                .as_any()
                .downcast_ref::<arrow_array::BooleanArray>()
                .unwrap();
            let tags = batch.record_batch().column_by_name("tag_ids").unwrap();
            let tags = tags
                .as_any()
                .downcast_ref::<arrow_array::LargeListArray>()
                .unwrap();
            let types = batch.record_batch().column_by_name("element_type").unwrap();
            let types = types
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .unwrap();
            let point_ids = batch.record_batch().column_by_name("point_ids").unwrap();
            let point_ids = point_ids
                .as_any()
                .downcast_ref::<arrow_array::LargeListArray>()
                .unwrap();
            for row in 0..batch.len() {
                if boundary.is_null(row) || !boundary.value(row) {
                    continue;
                }
                let tag_values = tags.value(row);
                let tag_values = tag_values
                    .as_any()
                    .downcast_ref::<arrow_array::UInt64Array>()
                    .unwrap();
                if tag_values.values().first().copied().unwrap_or(0) != tag {
                    continue;
                }
                let code = su2_code(types.value(row)).ok_or_else(|| {
                    MeshError::InvalidInput(format!(
                        "SU2 export does not support boundary type {:?}",
                        types.value(row)
                    ))
                })?;
                write!(output, "{code}")?;
                let values = point_ids.value(row);
                let values = values
                    .as_any()
                    .downcast_ref::<arrow_array::UInt64Array>()
                    .unwrap();
                for id in values.values() {
                    let number = point_numbers.get(id).ok_or_else(|| {
                        MeshError::InvalidFile(format!(
                            "boundary element references missing point {id}"
                        ))
                    })?;
                    write!(output, " {number}")?;
                }
                writeln!(output)?;
            }
        }
    }
    Ok(())
}

fn su2_code(element_type: &str) -> Option<u8> {
    Some(match element_type {
        "edge2" | "edge3" => 3,
        "tri3" | "tri6" => 5,
        "quad4" | "quad8" | "quad9" => 9,
        "tet4" | "tet10" => 10,
        "hex8" | "hex20" | "hex27" => 12,
        "prism6" | "prism15" => 13,
        "pyramid5" | "pyramid13" => 14,
        _ => return None,
    })
}
