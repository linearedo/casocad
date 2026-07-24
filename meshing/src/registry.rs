use crate::advancing_front::{ADVANCING_FRONT, ADVANCING_FRONT_DESCRIPTOR};
use crate::algorithm::{MeshAlgorithm, MeshAlgorithmDescriptor};
use crate::uniform::{UNIFORM_2D, UNIFORM_2D_DESCRIPTOR};

static DESCRIPTORS: [MeshAlgorithmDescriptor; 2] =
    [ADVANCING_FRONT_DESCRIPTOR, UNIFORM_2D_DESCRIPTOR];

pub fn descriptors() -> &'static [MeshAlgorithmDescriptor] {
    &DESCRIPTORS
}

pub fn algorithm(id: &str) -> Option<&'static dyn MeshAlgorithm> {
    match id {
        "advancing_front" => Some(&ADVANCING_FRONT),
        "uniform_2d" => Some(&UNIFORM_2D),
        _ => None,
    }
}
