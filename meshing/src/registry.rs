use crate::algorithm::{MeshAlgorithm, MeshAlgorithmDescriptor};
use crate::distmesh::{DISTMESH, DISTMESH_DESCRIPTOR};

static DESCRIPTORS: [MeshAlgorithmDescriptor; 1] = [DISTMESH_DESCRIPTOR];

pub fn descriptors() -> &'static [MeshAlgorithmDescriptor] {
    &DESCRIPTORS
}

pub fn algorithm(id: &str) -> Option<&'static dyn MeshAlgorithm> {
    match id {
        "distmesh" => Some(&DISTMESH),
        _ => None,
    }
}
