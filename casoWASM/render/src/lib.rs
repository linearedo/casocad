//! caso-render — the wgpu viewport renderer for casoWASM.
//!
//! One codebase for native (Vulkan/Metal/DX12/GL) and web (WebGPU/WebGL2):
//! backend choice belongs to wgpu, never to this code. Three fixed WGSL
//! shader programs (surface, analytic grid/axes, screen-space thick lines) —
//! no per-scene shader generation.

#![forbid(unsafe_code)]

pub mod camera;
pub mod renderer;

pub use camera::{OrbitCamera, DEFAULT_VIEW_DISTANCE};
pub use renderer::{RenderOptions, ViewportRenderer, DEFAULT_BACKGROUND, TARGET_FORMAT};
