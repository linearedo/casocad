//! Boundary-region viewport support: the hover/select tool, the cutter, and
//! the classifier-filtered highlight overlays (yellow candidate, cyan
//! committed selection, cyan/orange split preview) — the port of casoCAD's
//! BoundaryRegion tool and BoundaryCutter.

use caso_kernel::boundary::{BoundaryCut, BoundaryRegion, CutSide};
use caso_kernel::boundary_ops::{
    boundary_region_base_mask, cut_volume, pick_boundary_patch, pick_sdf_surface, sdf_normal,
    BoundaryPatchHit,
};
use caso_kernel::boundary_paths::{
    stencil_knife, straight_knife, KNIFE_CURVATURE_WARNING_ALIGNMENT,
};
use caso_kernel::scene::SceneDocument;
use caso_kernel::sdf::node::Node;
use caso_kernel::vec3::{vec3, Vec3};
use caso_surfaces::clipping::{clip_mesh_to_sdf, tessellate_for_clip, OperandMesh};
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

/// The pre-lift highlight mesh for one region: base-surface triangles that
/// pass classifier criteria 1-3 (on-boundary, owner provenance, patch scope
/// — all mesh-aligned) are kept whole, then clipped exactly against each cut
/// volume in chain order, with seam vertices root-found onto the knife's
/// zero set. The seam therefore lies on the true cut regardless of the
/// display-surface tessellation density.
///
/// The overlay follows the exact `<= 0` zero set; the per-point classifier
/// keeps its scale-relative `tol` band for robust membership, so highlight
/// and mask can disagree only within an O(tol) band around the seam —
/// invisible at lift scale.
pub fn region_highlight_mesh(
    root: &Node,
    region: &BoundaryRegion,
    scene: &ViewportSurfaceScene,
) -> Option<OperandMesh> {
    let diagonal = root
        .bounding_box()
        .map(|bounds| bounds.diagonal())
        .unwrap_or(1.0);
    // Cut volumes once per call; a failed cut_volume hides the highlight
    // (matching the classifier's error behavior) rather than silently
    // showing the uncut region.
    let mut cut_volumes: Vec<(Node, CutSide)> = Vec::with_capacity(region.cuts.len());
    for cut in &region.cuts {
        let volume = cut_volume(root, cut).ok()?;
        cut_volumes.push((volume, cut.side));
    }
    let target_edge = diagonal / 64.0;
    let eps_value = (diagonal * 1.0e-4).max(1.0e-6);
    let eps = vec3(eps_value, eps_value, eps_value);

    let mut combined = OperandMesh::default();
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
        let Ok(mask) = boundary_region_base_mask(root, region, &points, None) else {
            continue;
        };
        // Whole-triangle filter on the mesh-aligned criteria, compacted into
        // an f64 operand mesh (UNLIFTED — root-finding must land on the true
        // zero set; lifting happens after clipping).
        let mut remap = vec![u32::MAX; points.len()];
        let mut mesh = OperandMesh::default();
        for triangle in surface.indices.chunks_exact(3) {
            let (a, b, c) = (
                triangle[0] as usize,
                triangle[1] as usize,
                triangle[2] as usize,
            );
            if !(mask[a] && mask[b] && mask[c]) {
                continue;
            }
            let mut mapped = [0u32; 3];
            for (slot, &vertex) in [a, b, c].iter().enumerate() {
                if remap[vertex] == u32::MAX {
                    remap[vertex] = mesh.vertices.len() as u32;
                    mesh.vertices.push(points[vertex]);
                    let normal = surface.normals.get(vertex).copied().unwrap_or([0.0; 3]);
                    mesh.normals.push(vec3(
                        normal[0] as f64,
                        normal[1] as f64,
                        normal[2] as f64,
                    ));
                }
                mapped[slot] = remap[vertex];
            }
            mesh.triangles.push(mapped);
        }
        if mesh.triangles.is_empty() {
            continue;
        }
        // Criterion 4, exact: clip along each cut's zero set in chain order
        // (later clips run on the already-clipped mesh, so the chain is an
        // exact conjunction). Both split-preview children run this identical
        // deterministic pipeline with opposite keep_inside, which is what
        // makes their shared seam bitwise-identical and crack-free — never
        // perturb or parallelize the two calls independently.
        if !cut_volumes.is_empty() {
            for (volume, side) in &cut_volumes {
                mesh = tessellate_for_clip(mesh, volume, target_edge, 6);
                mesh = clip_mesh_to_sdf(&mesh, volume, *side == CutSide::Inside, eps);
                if mesh.triangles.is_empty() {
                    break;
                }
            }
            if mesh.triangles.is_empty() {
                continue;
            }
        }
        let offset = combined.vertices.len() as u32;
        combined.vertices.extend(mesh.vertices);
        combined.normals.extend(mesh.normals);
        combined.triangles.extend(
            mesh.triangles
                .into_iter()
                .map(|tri| [tri[0] + offset, tri[1] + offset, tri[2] + offset]),
        );
    }
    if combined.triangles.is_empty() {
        return None;
    }
    // Drop vertices no surviving triangle references (the clip keeps all its
    // input vertices), remapping indices.
    let mut remap = vec![u32::MAX; combined.vertices.len()];
    let mut compact = OperandMesh::default();
    for tri in &combined.triangles {
        let mut mapped = [0u32; 3];
        for (slot, index) in tri.iter().enumerate() {
            let index = *index as usize;
            if remap[index] == u32::MAX {
                remap[index] = compact.vertices.len() as u32;
                compact.vertices.push(combined.vertices[index]);
                compact.normals.push(combined.normals[index]);
            }
            mapped[slot] = remap[index];
        }
        compact.triangles.push(mapped);
    }
    Some(compact)
}

/// The classifier-filtered highlight: `region_highlight_mesh` lifted along
/// its normals by diagonal·1e-3 so the overlay never z-fights the surface.
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
    let mesh = region_highlight_mesh(root, region, scene)?;

    let vertices: Vec<[f32; 3]> = mesh
        .vertices
        .iter()
        .zip(&mesh.normals)
        .map(|(position, normal)| {
            let lifted = *position + *normal * lift;
            [lifted.x as f32, lifted.y as f32, lifted.z as f32]
        })
        .collect();
    let normals: Vec<[f32; 3]> = mesh
        .normals
        .iter()
        .map(|n| [n.x as f32, n.y as f32, n.z as f32])
        .collect();
    let indices: Vec<u32> = mesh.triangles.iter().flatten().copied().collect();
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
        alpha: 1.0,
        bounds_min,
        bounds_max,
        message: String::new(),
    })
}

/// A knife ghost plus the warnings its construction raised. The ghost is
/// never a scene object; warnings must reach the status line at preview AND
/// commit so the user always knows what will be (was) stored.
#[derive(Debug)]
pub struct KnifeGhost {
    pub node: Node,
    pub warnings: Vec<String>,
}



/// Build the ghost knife node for a cutter gesture (never a scene object).
pub fn cutter_ghost(root: &Node, kind: &str, points: &[Vec3]) -> Result<KnifeGhost, String> {
    match kind {
        "segment" => {
            if points.len() < 2 {
                return Err("segment knife needs two points".to_string());
            }
            let start = points[0];
            let end = *points.last().expect("nonempty");
            // Mean of the endpoint normals: order-independent (swapping the
            // clicks yields the same plane), unlike orienting by the first
            // click alone.
            let normal_start = surface_normal(root, start);
            let normal_end = surface_normal(root, end);
            let mean = normal_start + normal_end;
            if mean.length() <= 1.0e-6 {
                return Err(
                    "segment endpoints have opposing surface normals — no meaningful \
                     cutting plane exists; cut across a flatter part of the boundary"
                        .to_string(),
                );
            }
            let mut warnings = Vec::new();
            if normal_start.dot(normal_end) < KNIFE_CURVATURE_WARNING_ALIGNMENT {
                warnings.push(
                    "segment cuts are planar slices; on curved boundaries the cut \
                     follows the plane–surface intersection"
                        .to_string(),
                );
            }
            let node = straight_knife(root, start, end, mean)
                .map_err(|error| error.to_string())?;
            Ok(KnifeGhost { node, warnings })
        }
        // Point-collected knives: planar stencils on the mean-click-normal
        // plane, extruded one-sidedly so only the clicked sheet is cut.
        "polygon" | "quadratic_bezier_surface" => stencil_knife(root, kind, points)
            .map(|(node, curved)| {
                let mut warnings = Vec::new();
                if curved {
                    warnings.push(
                        "point stencils are planar; on curved boundaries the cut \
                         follows the stencil–surface intersection"
                            .to_string(),
                    );
                }
                KnifeGhost { node, warnings }
            })
            .map_err(|error| error.to_string()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use caso_kernel::frame::IDENTITY_FRAME;
    use caso_kernel::sdf::node::Shape;
    use caso_kernel::sdf::primitives_3d::Sphere;
    use caso_surfaces::{build_viewport_surface_scene, ViewportSurfaceCache};

    /// Default von Kármán scene: Difference(flow box 1.6×0.7×0.45 half-size,
    /// cylinder r=0.24 through Z) — the -X face is flat and uncut.
    fn fixture() -> (Node, ViewportSurfaceScene) {
        let document = SceneDocument::default_scene().expect("default scene");
        let root = fluid_root_node(&document).expect("fluid root");
        let mut cache = ViewportSurfaceCache::default();
        let scene = build_viewport_surface_scene(std::slice::from_ref(&root), 1, &mut cache);
        (root, scene)
    }

    fn minus_x_face_region(root: &Node) -> BoundaryRegion {
        // The ray must miss the cylinder obstacle (r=0.24): its cut-surface
        // patch would win the pick outright.
        let hit = pick_patch(root, vec3(-5.0, 0.5, 0.02), vec3(1.0, 0.0, 0.0))
            .expect("-X face hit");
        assert!(hit.patch_id.ends_with("-X"), "unexpected patch {}", hit.patch_id);
        candidate_region(&hit)
    }

    /// Vertical segment at y=0 across the -X face: the knife volume is the
    /// half-space y >= 0 (within its spans), so Inside = upper half.
    fn face_segment_ghost(root: &Node) -> Node {
        cutter_ghost(
            root,
            "segment",
            &[vec3(-1.6, 0.0, -0.4), vec3(-1.6, 0.0, 0.4)],
        )
        .expect("segment ghost")
        .node
    }

    fn diagonal(root: &Node) -> f64 {
        root.bounding_box().expect("bounds").diagonal()
    }

    fn mesh_area(mesh: &OperandMesh) -> f64 {
        mesh.triangles
            .iter()
            .map(|tri| {
                let a = mesh.vertices[tri[0] as usize];
                let b = mesh.vertices[tri[1] as usize];
                let c = mesh.vertices[tri[2] as usize];
                (b - a).cross(c - a).length() * 0.5
            })
            .sum()
    }

    #[test]
    fn seam_vertices_lie_exactly_on_the_knife_zero_set() {
        let (root, scene) = fixture();
        let ghost = face_segment_ghost(&root);
        let (inside, _outside) =
            split_preview_children(&minus_x_face_region(&root), &ghost);
        let volume =
            cut_volume(&root, inside.cuts.last().expect("cut")).expect("cut volume");
        let mesh = region_highlight_mesh(&root, &inside, &scene).expect("highlight");
        let diag = diagonal(&root);
        let mut seam_vertices = 0usize;
        for vertex in &mesh.vertices {
            let value = volume.eval_point(*vertex).abs();
            // No vertex may sit "near but not on" the knife: anything inside
            // the near band must be a root-found seam vertex on the zero set.
            if value < 1.0e-4 * diag {
                assert!(
                    value <= 1.0e-9 * diag,
                    "vertex {vertex:?} is near the knife but off its zero set ({value})"
                );
                seam_vertices += 1;
            }
        }
        assert!(seam_vertices >= 2, "the cut must produce seam vertices");
    }

    #[test]
    fn preview_triangles_stay_strictly_on_their_side() {
        let (root, scene) = fixture();
        let ghost = face_segment_ghost(&root);
        let (inside, outside) =
            split_preview_children(&minus_x_face_region(&root), &ghost);
        let volume =
            cut_volume(&root, inside.cuts.last().expect("cut")).expect("cut volume");
        let diag = diagonal(&root);
        let slack = 1.0e-9 * diag;
        for (region, keep_negative) in [(&inside, true), (&outside, false)] {
            let mesh = region_highlight_mesh(&root, region, &scene).expect("highlight");
            for tri in &mesh.triangles {
                let centroid = (mesh.vertices[tri[0] as usize]
                    + mesh.vertices[tri[1] as usize]
                    + mesh.vertices[tri[2] as usize])
                    * (1.0 / 3.0);
                let value = volume.eval_point(centroid);
                if keep_negative {
                    assert!(value <= slack, "inside triangle leaked to {value}");
                } else {
                    assert!(value >= -slack, "outside triangle leaked to {value}");
                }
            }
        }
    }

    #[test]
    fn split_previews_partition_the_region_area() {
        let (root, scene) = fixture();
        let region = minus_x_face_region(&root);
        let ghost = face_segment_ghost(&root);
        let (inside, outside) = split_preview_children(&region, &ghost);
        let parent_area =
            mesh_area(&region_highlight_mesh(&root, &region, &scene).expect("parent"));
        let inside_area =
            mesh_area(&region_highlight_mesh(&root, &inside, &scene).expect("inside"));
        let outside_area =
            mesh_area(&region_highlight_mesh(&root, &outside, &scene).expect("outside"));
        assert!(parent_area > 0.0);
        let relative_gap =
            ((inside_area + outside_area) - parent_area).abs() / parent_area;
        assert!(
            relative_gap <= 1.0e-6,
            "children must partition the parent crack-free (gap {relative_gap})"
        );
    }

    #[test]
    fn hover_candidate_keeps_the_plain_triangle_filter() {
        let (root, scene) = fixture();
        let region = minus_x_face_region(&root);
        let mesh = region_highlight_mesh(&root, &region, &scene).expect("highlight");
        // Cut-free path: output vertices are display-mesh vertices, all of
        // which pass the base mask (no clipping ran).
        let mask = boundary_region_base_mask(&root, &region, &mesh.vertices, None)
            .expect("mask");
        assert!(mask.iter().all(|hit| *hit));
        assert!(!mesh.triangles.is_empty());
    }

    #[test]
    fn segment_knife_is_click_order_independent() {
        let (root, _scene) = fixture();
        let first = vec3(-1.6, 0.0, -0.4);
        let second = vec3(-1.6, 0.0, 0.4);
        let forward = cutter_ghost(
            &root,
            "segment", &[first, second])
            .expect("forward")
            .node;
        let backward = cutter_ghost(
            &root,
            "segment", &[second, first])
            .expect("backward")
            .node;
        let cut = |ghost: Node| BoundaryCut {
            side: CutSide::Inside,
            ghost,
        };
        let volume_forward = cut_volume(&root, &cut(forward)).expect("volume");
        let volume_backward = cut_volume(&root, &cut(backward)).expect("volume");
        // Same plane: the sign fields agree everywhere up to one global flip
        // (the Inside/Outside labels of the two children may swap, but the
        // partition itself is identical).
        let probes = [
            vec3(-1.6, 0.3, 0.1),
            vec3(-1.6, -0.3, 0.1),
            vec3(-1.6, 0.5, -0.3),
            vec3(-1.6, -0.5, -0.3),
        ];
        let signs: Vec<(bool, bool)> = probes
            .iter()
            .map(|p| {
                (
                    volume_forward.eval_point(*p) <= 0.0,
                    volume_backward.eval_point(*p) <= 0.0,
                )
            })
            .collect();
        let all_equal = signs.iter().all(|(f, b)| f == b);
        let all_flipped = signs.iter().all(|(f, b)| f != b);
        assert!(
            all_equal || all_flipped,
            "click order changed the partition: {signs:?}"
        );
    }

    #[test]
    fn segment_knife_warns_on_curved_boundaries() {
        let sphere = Node::with_id(
            "s",
            1,
            Shape::Sphere(Sphere::new(Vec3::ZERO, 1.0).expect("sphere")),
        );
        let ghost = cutter_ghost(
            &sphere,
            "segment",
            &[vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)],
        )
        .expect("ghost on curved surface");
        assert!(
            !ghost.warnings.is_empty(),
            "misaligned endpoint normals must raise the planar-slice warning"
        );
    }

    #[test]
    fn segment_knife_refuses_opposing_normals() {
        let sphere = Node::with_id(
            "s",
            1,
            Shape::Sphere(Sphere::new(Vec3::ZERO, 1.0).expect("sphere")),
        );
        let error = cutter_ghost(
            &sphere,
            "segment",
            &[vec3(1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0)],
        )
        .expect_err("antipodal endpoints have no cutting plane");
        assert!(error.contains("opposing"), "unexpected error: {error}");
    }

    #[test]
    fn flat_face_segment_raises_no_warning() {
        let (root, _scene) = fixture();
        let ghost = cutter_ghost(
            &root,
            "segment",
            &[vec3(-1.6, 0.0, -0.4), vec3(-1.6, 0.0, 0.4)],
        )
        .expect("ghost");
        assert!(ghost.warnings.is_empty());
        // Unused warning suppression for IDENTITY_FRAME import parity.
        let _ = IDENTITY_FRAME;
    }
}
