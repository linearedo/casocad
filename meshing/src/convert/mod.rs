//! Streaming exporters from finalized Arrow batches.

mod su2;

use std::io::Write;

use crate::{MeshFile, MeshResult};

pub struct MeshConverter {
    pub id: &'static str,
    pub label: &'static str,
    pub extension: &'static str,
    pub write: fn(&MeshFile) -> MeshResult<Vec<u8>>,
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

pub fn write_to(converter_id: &str, mesh: &MeshFile, output: &mut impl Write) -> MeshResult<()> {
    match converter_id {
        "su2" => su2::write_to(mesh, output),
        _ => Err(crate::MeshError::InvalidInput(format!(
            "unknown mesh converter {converter_id:?}"
        ))),
    }
}
