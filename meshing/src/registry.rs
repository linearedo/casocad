use crate::advancing_front::{ADVANCING_FRONT, ADVANCING_FRONT_DESCRIPTOR};
use crate::algorithm::{MeshAlgorithm, MeshAlgorithmDescriptor};

static DESCRIPTORS: [MeshAlgorithmDescriptor; 1] = [ADVANCING_FRONT_DESCRIPTOR];

pub fn descriptors() -> &'static [MeshAlgorithmDescriptor] {
    &DESCRIPTORS
}

pub fn algorithm(id: &str) -> Option<&'static dyn MeshAlgorithm> {
    match id {
        "advancing_front" => Some(&ADVANCING_FRONT),
        _ => None,
    }
}
