//! MeshIR exporters to solver mesh formats.

mod su2;

use std::collections::BTreeMap;

use crate::{MeshIr, MeshPoint};

pub struct MeshConverter {
    pub id: &'static str,
    pub label: &'static str,
    pub extension: &'static str,
    pub write: fn(&MeshIr) -> Result<Vec<u8>, String>,
}

pub const CONVERTERS: &[MeshConverter] = &[MeshConverter {
    id: "su2",
    label: "SU2",
    extension: "su2",
    write: su2::write,
}];

pub fn converter(id: &str) -> Option<&'static MeshConverter> {
    CONVERTERS.iter().find(|converter| converter.id == id)
}

pub(crate) fn normalized_mesh(mesh: &MeshIr) -> Result<MeshIr, String> {
    let mut mesh = mesh.clone();
    mesh.derive_topology()?;
    mesh.validate()?;
    Ok(mesh)
}

pub(crate) fn mesh_cell_dimension(mesh: &MeshIr, label: &str) -> Result<u8, String> {
    let mut dimension = None;
    for cell in &mesh.cells {
        let cell_dimension = crate::element_dimension(&cell.type_name)
            .ok_or_else(|| format!("unknown cell type {:?}", cell.type_name))?;
        if cell_dimension < 2 {
            return Err(format!("{label} export requires 2D or 3D cells"));
        }
        match dimension {
            Some(dimension) if dimension != cell_dimension => {
                return Err(format!(
                    "{label} export does not support mixed {dimension}D/{cell_dimension}D cell meshes"
                ));
            }
            None => dimension = Some(cell_dimension),
            _ => {}
        }
    }
    dimension.ok_or_else(|| format!("{label} export requires at least one cell"))
}

pub(crate) fn point_indices(points: &[MeshPoint]) -> BTreeMap<u64, usize> {
    points
        .iter()
        .enumerate()
        .map(|(index, point)| (point.id, index))
        .collect()
}

pub(crate) fn point_index(indices: &BTreeMap<u64, usize>, point_id: u64) -> Result<usize, String> {
    indices
        .get(&point_id)
        .copied()
        .ok_or_else(|| format!("missing point {point_id}"))
}

pub(crate) fn marker_name(mesh: &MeshIr, tag_ids: &[u64]) -> String {
    tag_ids
        .first()
        .and_then(|tag_id| mesh.tag_name(*tag_id))
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or("boundary")
        .to_string()
}
