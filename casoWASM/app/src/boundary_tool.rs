//! Boundary-region viewport support: the hover/select tool, the cutter, and
//! the classifier-filtered highlight overlays (yellow candidate, cyan
//! committed selection, cyan/orange split preview) — the port of casoCAD's
//! BoundaryRegion tool and BoundaryCutter.

use caso_kernel::boundary::{BoundaryCut, BoundaryRegion, CutSide};
use caso_kernel::boundary_ops::{
    boundary_region_mask, pick_boundary_patch, pick_sdf_surface, sdf_normal, BoundaryPatchHit,
};
use caso_kernel::boundary_paths::{smooth_polyline_knife, straight_knife};
use caso_kernel::scene::SceneDocument;
use caso_kernel::sdf::node::Node;
use caso_kernel::vec3::{vec3, Vec3};
use caso_surfaces::{SurfaceStatus, ViewportSurface, ViewportSurfaceKey, ViewportSurfaceScene};

pub const CANDIDATE_COLOR: [f32; 3] = [1.0, 0.9, 0.2]; // yellow
pub const SELECTED_COLOR: [f32; 3] = [0.2, 0.95, 1.0]; // cyan
pub const PREVIEW_INSIDE_COLOR: [f32; 3] = [0.2, 0.95, 1.0]; // cyan
pub const PREVIEW_OUTSIDE_COLOR: [f32; 3] = [1.0, 0.55, 0.15]; // orange

const PICK_TOLERANCE: f64 = 0.0008;
const PICK_TRAVEL: f64 = 100.0;

/// Build the fluid-domain root node, if a fluid domain is set.
pub fn fluid_root_node(document: &SceneDocument) -> Option<Node> {
    let fluid = document.fluid_domain.as_ref()?;
    document.build_node(fluid.root).ok()
}

/// Ray-pick the domain boundary and return the analytic patch hit.
pub fn pick_patch(root: &Node, origin: Vec3, direction: Vec3) -> Option<BoundaryPatchHit> {
    pick_boundary_patch(root, origin, direction, PICK_TOLERANCE, PICK_TRAVEL)
}

/// Ray-pick the boundary surface point (for cutter knife points).
pub fn pick_surface_point(root: &Node, origin: Vec3, direction: Vec3) -> Option<Vec3> {
    pick_sdf_surface(root, origin, direction, PICK_TOLERANCE, PICK_TRAVEL)
}

/// Surface normal at a picked point (for the straight knife plane).
pub fn surface_normal(root: &Node, point: Vec3) -> Vec3 {
    let diagonal = root
        .bounding_box()
        .map(|bounds| bounds.diagonal())
        .unwrap_or(1.0);
    sdf_normal(root, point, (diagonal * 1.0e-5).max(1.0e-7))
}

/// A transient region describing a hover candidate (never stored).
pub fn candidate_region(hit: &BoundaryPatchHit) -> BoundaryRegion {
    let mut region = BoundaryRegion::new("candidate", u32::MAX, hit.owner_object_id);
    region.patch_id = Some(hit.patch_id.clone());
    region.patch_type = Some(hit.patch_type.clone());
    region.outside_direction = hit.outside_direction;
    region
}

/// The classifier-filtered highlight: keep the base-surface triangles whose
/// vertices are all in the region, lifted along their normals by
/// diagonal·1e-3 so the overlay never z-fights the surface.
pub fn region_highlight_surface(
    root: &Node,
    region: &BoundaryRegion,
    scene: &ViewportSurfaceScene,
    color: [f32; 3],
    overlay_id: u32,
) -> Option<ViewportSurface> {
    let diagonal = root
        .bounding_box()
        .map(|bounds| bounds.diagonal())
        .unwrap_or(1.0);
    let lift = diagonal * 1.0e-3;

    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for surface in &scene.surfaces {
        if !scene.primary_object_ids.contains(&surface.key.object_id) {
            continue;
        }
        if surface.vertices.is_empty() || surface.indices.is_empty() {
            continue;
        }
        let points: Vec<Vec3> = surface
            .vertices
            .iter()
            .map(|v| vec3(v[0] as f64, v[1] as f64, v[2] as f64))
            .collect();
        let Ok(mask) = boundary_region_mask(root, region, &points, None) else {
            continue;
        };
        let base = vertices.len() as u32;
        let mut remap = vec![u32::MAX; points.len()];
        for triangle in surface.indices.chunks_exact(3) {
            let (a, b, c) = (
                triangle[0] as usize,
                triangle[1] as usize,
                triangle[2] as usize,
            );
            if !(mask[a] && mask[b] && mask[c]) {
                continue;
            }
            for &vertex in &[a, b, c] {
                if remap[vertex] == u32::MAX {
                    remap[vertex] = base + (vertices.len() as u32 - base);
                    let normal = surface.normals.get(vertex).copied().unwrap_or([0.0; 3]);
                    let position = surface.vertices[vertex];
                    vertices.push([
                        position[0] + normal[0] * lift as f32,
                        position[1] + normal[1] * lift as f32,
                        position[2] + normal[2] * lift as f32,
                    ]);
                    normals.push(normal);
                }
                indices.push(remap[vertex]);
            }
        }
    }
    if indices.is_empty() {
        return None;
    }
    let mut bounds_min = [f64::INFINITY; 3];
    let mut bounds_max = [f64::NEG_INFINITY; 3];
    for vertex in &vertices {
        for axis in 0..3 {
            bounds_min[axis] = bounds_min[axis].min(vertex[axis] as f64);
            bounds_max[axis] = bounds_max[axis].max(vertex[axis] as f64);
        }
    }
    Some(ViewportSurface {
        key: ViewportSurfaceKey {
            object_id: overlay_id,
            scene_revision: scene.revision,
            resolution: 0,
        },
        object_kind: "boundary_highlight".to_string(),
        status: SurfaceStatus::Ready,
        vertices,
        normals,
        indices,
        wire_indices: Vec::new(),
        color,
        bounds_min,
        bounds_max,
        message: String::new(),
    })
}

/// Build the ghost knife node for a cutter gesture (never a scene object).
pub fn cutter_ghost(
    root: &Node,
    kind: &str,
    points: &[Vec3],
) -> Result<Node, String> {
    match kind {
        "segment" => {
            if points.len() < 2 {
                return Err("segment knife needs two points".to_string());
            }
            let normal = surface_normal(root, points[0]);
            straight_knife(root, points[0], *points.last().expect("nonempty"), normal)
                .map_err(|error| error.to_string())
        }
        "smooth_polyline" => {
            smooth_polyline_knife(root, points).map_err(|error| error.to_string())
        }
        // Point-collected knives classify by their filled area: polyline
        // closes into a polygon, the bezier polycurve into its surface.
        "polygon" | "quadratic_bezier_surface" => {
            let mut scratch = SceneDocument::new();
            let handle = scratch
                .add_point_shape_from_world_points(kind, points, "xy")
                .map_err(|error| error.to_string())?;
            scratch.build_node(handle).map_err(|error| error.to_string())
        }
        other => Err(format!("unknown knife kind: {other}")),
    }
}

/// The two child regions a knife would produce (for the split preview).
pub fn split_preview_children(
    parent: &BoundaryRegion,
    ghost: &Node,
) -> (BoundaryRegion, BoundaryRegion) {
    let make = |side: CutSide| {
        let mut child = parent.clone();
        child.name = format!("{} preview {}", parent.name, side.as_str());
        let mut knife = ghost.clone();
        knife.object_id = 0;
        child.cuts.push(BoundaryCut { side, ghost: knife });
        child
    };
    (make(CutSide::Inside), make(CutSide::Outside))
}

/// Dense on-surface validation points from the display mesh (so
/// small-but-real cuts aren't rejected by coarse sampling).
pub fn validation_points(scene: &ViewportSurfaceScene) -> Vec<Vec3> {
    let mut points = Vec::new();
    for surface in &scene.surfaces {
        if !scene.primary_object_ids.contains(&surface.key.object_id) {
            continue;
        }
        points.extend(
            surface
                .vertices
                .iter()
                .map(|v| vec3(v[0] as f64, v[1] as f64, v[2] as f64)),
        );
    }
    points
}
