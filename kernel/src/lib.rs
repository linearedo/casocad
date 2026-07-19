//! caso-kernel — the exact signed-distance-field geometry kernel of casoWASM.
//!
//! Pure geometry: no GPU, no UI, no external dependencies. This crate is the
//! Rust port of casoCAD's `core/` package; `Node::eval_point` / `Node::eval`
//! are the authoritative evaluation path (the `to_numpy()` analog), always in
//! f64. Terminology: boolean operations are *SDF operators* (never "CSG");
//! "meshing" is reserved for FEA/CFD discretization.

#![forbid(unsafe_code)]

pub mod bbox;
pub mod boundary;
pub mod boundary_ops;
pub mod boundary_paths;
pub mod differential;
pub mod error;
pub mod frame;
pub mod meshing;
pub mod model;
pub mod preconditions;
pub mod roles;
pub mod scene;
pub mod sdf;
pub mod serialization;
pub mod vec3;

pub use bbox::BoundingBox3D;
pub use error::{GeometryError, GeometryResult};
pub use frame::Frame;
pub use sdf::node::{Node, Shape};
pub use vec3::{vec3, Vec3};
