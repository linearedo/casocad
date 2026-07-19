//! caso-surfaces — CPU display-surface builders for casoWASM.
//!
//! Port of `app/viewport/surface_*.py`: analytic primitive tessellation,
//! exact boolean clipping (Strategy A), dual contouring (Strategy B), and the
//! per-object surface cache. These are *display surfaces* — the word
//! "meshing" is reserved for FEA/CFD discretization. No GPU dependencies:
//! output is plain vertex/normal/index buffers ready for upload.

#![forbid(unsafe_code)]

pub mod boundary_outline;
pub mod builders;
pub mod clipping;
pub mod contouring;
pub mod geomops;
pub mod profiles2d;
pub mod types;

pub use builders::{build_viewport_surface, build_viewport_surface_scene, ViewportSurfaceCache};
pub use types::{SurfaceStatus, ViewportSurface, ViewportSurfaceKey, ViewportSurfaceScene};
