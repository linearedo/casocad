//! Viewport-surface value types and the empty/failed/colour fallbacks.
//! Ported from `app/viewport/surface_types.py`. These are display surfaces —
//! never "meshes" (that word is reserved for FEA/CFD discretization).

use caso_kernel::sdf::node::Node;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceStatus {
    Ready,
    Outline,
    Empty,
    Failed,
}

pub const DEFAULT_RESOLUTION: u32 = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ViewportSurfaceKey {
    pub object_id: u32,
    pub scene_revision: u64,
    pub resolution: u32,
}

/// One object's display surface: f32 buffers ready for GPU upload.
#[derive(Debug, Clone, PartialEq)]
pub struct ViewportSurface {
    pub key: ViewportSurfaceKey,
    pub object_kind: String,
    pub status: SurfaceStatus,
    /// Flat xyz triples.
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub wire_indices: Vec<u32>,
    pub color: [f32; 3],
    /// 1.0 = opaque; below 1.0 the renderer draws the surface blended
    /// (ghost previews). Wire-only surfaces ignore it (lines stay opaque).
    pub alpha: f32,
    pub bounds_min: [f64; 3],
    pub bounds_max: [f64; 3],
    pub message: String,
}

impl ViewportSurface {
    pub fn has_geometry(&self) -> bool {
        !self.vertices.is_empty() && (!self.indices.is_empty() || !self.wire_indices.is_empty())
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ViewportSurfaceScene {
    pub revision: u64,
    pub surfaces: Vec<ViewportSurface>,
    pub build_ms: f64,
    pub primary_object_ids: Vec<u32>,
}

impl ViewportSurfaceScene {
    pub fn with_components_visible(&self, visible: bool) -> ViewportSurfaceScene {
        if visible || self.primary_object_ids.is_empty() {
            return self.clone();
        }
        let primary: Vec<ViewportSurface> = self
            .surfaces
            .iter()
            .filter(|surface| self.primary_object_ids.contains(&surface.key.object_id))
            .cloned()
            .collect();
        if primary.len() == self.surfaces.len() {
            return self.clone();
        }
        ViewportSurfaceScene {
            surfaces: primary,
            ..self.clone()
        }
    }

    /// Brighten the selected object's surface (`c*1.35 + 0.16`).
    pub fn with_selected_highlight(&self, object_id: u32) -> ViewportSurfaceScene {
        if object_id == 0 {
            return self.clone();
        }
        let mut changed = false;
        let surfaces = self
            .surfaces
            .iter()
            .map(|surface| {
                if surface.key.object_id != object_id {
                    return surface.clone();
                }
                changed = true;
                let mut highlighted = surface.clone();
                highlighted.color = [
                    (surface.color[0] * 1.35 + 0.16).min(1.0),
                    (surface.color[1] * 1.35 + 0.16).min(1.0),
                    (surface.color[2] * 1.35 + 0.16).min(1.0),
                ];
                highlighted
            })
            .collect();
        if !changed {
            return self.clone();
        }
        ViewportSurfaceScene {
            surfaces,
            ..self.clone()
        }
    }

    pub fn has_geometry(&self) -> bool {
        self.surfaces.iter().any(ViewportSurface::has_geometry)
    }

    pub fn vertex_count(&self) -> usize {
        self.surfaces.iter().map(ViewportSurface::vertex_count).sum()
    }

    pub fn triangle_count(&self) -> usize {
        self.surfaces.iter().map(ViewportSurface::triangle_count).sum()
    }
}

/// Stable per-object color: Knuth-hash hue, fixed saturation/value.
pub fn object_color(object_id: u32) -> [f32; 3] {
    let value = (object_id as u64).wrapping_mul(2_654_435_761) & 0xFFFF_FFFF;
    let hue = (value % 360) as f64 / 360.0;
    hsv_to_rgb(hue, 0.48, 0.92)
}

/// Stable color for a mesh-preview tag: same hue hashing as
/// [`object_color`] but saturated and dark, so mesh wires and markers stand
/// out against the pastel object surfaces they overlay.
pub fn mesh_tag_color(tag_id: u32) -> [f32; 3] {
    let value = (tag_id as u64).wrapping_mul(2_654_435_761) & 0xFFFF_FFFF;
    let hue = (value % 360) as f64 / 360.0;
    hsv_to_rgb(hue, 0.95, 0.65)
}

fn hsv_to_rgb(hue: f64, saturation: f64, value: f64) -> [f32; 3] {
    let h = (hue % 1.0) * 6.0;
    let c = value * saturation;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let m = value - c;
    let rgb = if h < 1.0 {
        (c, x, 0.0)
    } else if h < 2.0 {
        (x, c, 0.0)
    } else if h < 3.0 {
        (0.0, c, x)
    } else if h < 4.0 {
        (0.0, x, c)
    } else if h < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    [
        (rgb.0 + m) as f32,
        (rgb.1 + m) as f32,
        (rgb.2 + m) as f32,
    ]
}

fn safe_node_bounds(node: &Node) -> ([f64; 3], [f64; 3]) {
    match node.bounding_box() {
        Ok(bounds) => (
            [bounds.x_min, bounds.y_min, bounds.z_min],
            [bounds.x_max, bounds.y_max, bounds.z_max],
        ),
        Err(_) => ([0.0; 3], [0.0; 3]),
    }
}

pub fn empty_surface(
    node: &Node,
    key: ViewportSurfaceKey,
    color: [f32; 3],
    message: impl Into<String>,
) -> ViewportSurface {
    let (bounds_min, bounds_max) = safe_node_bounds(node);
    ViewportSurface {
        key,
        object_kind: node.kind().to_string(),
        status: SurfaceStatus::Empty,
        vertices: Vec::new(),
        normals: Vec::new(),
        indices: Vec::new(),
        wire_indices: Vec::new(),
        color,
        alpha: 1.0,
        bounds_min,
        bounds_max,
        message: message.into(),
    }
}

pub fn failed_surface(
    node: &Node,
    key: ViewportSurfaceKey,
    color: [f32; 3],
    message: impl Into<String>,
) -> ViewportSurface {
    let mut surface = empty_surface(node, key, color, message);
    surface.status = SurfaceStatus::Failed;
    surface
}
