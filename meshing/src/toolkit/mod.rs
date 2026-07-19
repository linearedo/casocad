//! Mesher-facing toolkit (`design_docs/meshing_toolkit.md`): exact tagged 2D
//! boundary loops and the analytic sizing field. Interior-exactness contract:
//! no positive field value is ever consumed as a distance.

pub mod loops2d;
pub mod sizing;

pub use loops2d::{boundary_loops, BoundaryLoop, BoundarySpan};
pub use sizing::{SizingBand, SizingField, SizingSpec};
