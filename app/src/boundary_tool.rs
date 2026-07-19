//! Boundary-region viewport support: the hover/select tool, the cutter, and
//! the classifier-filtered highlight overlays (yellow candidate, cyan
//! committed selection, cyan/orange split preview) — the port of casoCAD's
//! BoundaryRegion tool and BoundaryCutter.

use caso_kernel::boundary::{BoundaryCut, BoundaryRegion, CutSide};
use caso_kernel::boundary_ops::{
    boundary_region_base_mask, cut_volume, pick_boundary_patch,
    pick_boundary_patch_with_radius, pick_outline_point_with_radius, pick_sdf_surface,
    placed_2d_root, sdf_normal, surface_patches_for_root, BoundaryPatchHit,
    BoundarySurfacePatch,
};
use caso_kernel::boundary_paths::{
    point_knife, stencil_knife, straight_knife, workplane_normal,
    KNIFE_CURVATURE_WARNING_ALIGNMENT,
};
use caso_kernel::scene::SceneDocument;
use caso_kernel::sdf::node::Node;
use caso_kernel::vec3::{vec3, Vec3};
use caso_surfaces::boundary_outline::curve_patch_arcs;
use caso_surfaces::clipping::{clip_mesh_to_sdf, tessellate_for_clip, OperandMesh};
use caso_surfaces::profiles2d::placed_outline_rings;
use caso_surfaces::{SurfaceStatus, ViewportSurface, ViewportSurfaceKey, ViewportSurfaceScene};

/// Ring sampling density for 2D outline work (highlight ribbon, knife
/// crossing census, dense validation points).
const OUTLINE_RING_RESOLUTION: usize = 192;

pub const CANDIDATE_COLOR: [f32; 3] = [1.0, 0.9, 0.2]; // yellow
pub const SELECTED_COLOR: [f32; 3] = [0.2, 0.95, 1.0]; // cyan
pub const PREVIEW_INSIDE_COLOR: [f32; 3] = [0.2, 0.95, 1.0]; // cyan
pub const PREVIEW_OUTSIDE_COLOR: [f32; 3] = [1.0, 0.55, 0.15]; // orange

const PICK_TOLERANCE: f64 = 0.0008;
const PICK_TRAVEL: f64 = 100.0;

/// Build the fluid-domain root node, if a fluid domain is set. Production
/// code resolves roots per marked domain (`domain_root_nodes`,
/// `region_root_node`); this stays for the test fixtures.
#[allow(dead_code)]
pub fn fluid_root_node(document: &SceneDocument) -> Option<Node> {
    let fluid = document.fluid_domain.as_ref()?;
    document.build_node(fluid.root).ok()
}

/// Every marked domain's built root, fluid first — the boundaries the
/// Boundary Region tool picks against. Regions work on ALL domain kinds
/// (fluid, solid, future ones), not only fluid.
pub fn domain_root_nodes(document: &SceneDocument) -> Vec<(u32, Node)> {
    document
        .marked_domain_roots()
        .into_iter()
        .filter_map(|id| document.build_node(id).ok().map(|node| (id, node)))
        .collect()
}

/// The built root of the domain a region tags (the fluid domain for
/// regions from older files, which carry no explicit domain).
pub fn region_root_node(document: &SceneDocument, region: &BoundaryRegion) -> Option<Node> {
    let root = document.region_domain_root(region)?;
    document.build_node(root).ok()
}

/// The built root for the currently selected region, if any.
pub fn selected_region_root(document: &SceneDocument, selected: Option<u32>) -> Option<Node> {
    let region_id = selected?;
    let region = document
        .boundary_regions
        .iter()
        .find(|region| region.object_id == region_id)?;
    region_root_node(document, region)
}

/// Ray-pick the domain boundary with the default pick radius. Production
/// picks go through `pick_patch_with_radius` (screen-derived radius); this
/// stays for the test fixtures.
#[allow(dead_code)]
pub fn pick_patch(root: &Node, origin: Vec3, direction: Vec3) -> Option<BoundaryPatchHit> {
    pick_boundary_patch(root, origin, direction, PICK_TOLERANCE, PICK_TRAVEL)
}

/// `pick_patch` with a world-space 2D pick radius (screen-derived by the
/// viewport; `None` keeps the kernel's scale-relative default).
pub fn pick_patch_with_radius(
    root: &Node,
    origin: Vec3,
    direction: Vec3,
    curve_pick_radius: Option<f64>,
) -> Option<BoundaryPatchHit> {
    pick_boundary_patch_with_radius(
        root,
        origin,
        direction,
        PICK_TOLERANCE,
        PICK_TRAVEL,
        curve_pick_radius,
    )
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

/// Ribbon half-width and plane lift of the 2D highlight, in screen pixels
/// (applied to the world size of one pixel on the workplane). One knob for
/// the highlight's apparent thickness at any zoom.
const RIBBON_HALF_WIDTH_PIXELS: f64 = 1.0;
const RIBBON_LIFT_PIXELS: f64 = 1.0;

/// The classifier-filtered highlight: `region_highlight_mesh` lifted along
/// its normals by diagonal·1e-3 so the overlay never z-fights the surface.
/// `pixel_size` is the world length of one screen pixel on the domain's
/// workplane (2D domains only): when given, the ribbon has constant
/// apparent width; without it (degenerate camera, tests) the ribbon falls
/// back to scaling with the tagged patch's owner.
pub fn region_highlight_surface(
    root: &Node,
    region: &BoundaryRegion,
    scene: &ViewportSurfaceScene,
    color: [f32; 3],
    overlay_id: u32,
    pixel_size: Option<f64>,
) -> Option<ViewportSurface> {
    if root.dimension() == 2 {
        // 2D domains: the region is an arc of the outline curve — a lifted
        // ribbon, not a lifted triangle filter.
        return region_highlight_ribbon(root, region, scene, color, overlay_id, pixel_size);
    }
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

/// The crossing of a cut volume's zero set on the chord (p, q), by
/// bisection. Both split-preview children run this identical call for the
/// same chord and volume, so their shared split vertex is bitwise-identical
/// — the 1D analog of the 3D seam root-finding. Never perturb the two
/// calls independently.
fn bisect_crossing(volume: &Node, p: Vec3, q: Vec3) -> Vec3 {
    let sign_at_start = volume.eval_point(p) <= 0.0;
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        let value = volume.eval_point(p + (q - p) * mid);
        if (value <= 0.0) == sign_at_start {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    p + (q - p) * (0.5 * (lo + hi))
}

/// The curve patch a region tags, when it still resolves on this root.
fn region_curve_patch(root: &Node, region: &BoundaryRegion) -> Option<BoundarySurfacePatch> {
    let patch_id = region.patch_id.as_deref()?;
    surface_patches_for_root(root).into_iter().find(|patch| {
        patch.owner_object_id == region.owner_object_id
            && patch.patch_id == patch_id
            && patch.curve.is_some()
    })
}

/// The polylines the ribbon draws for a region: patch-exact arcs when the
/// region tags a curve patch (corners exact, ends root-found on the boolean
/// junctions — see `curve_patch_arcs`), else the legacy classifier-masked
/// outline-ring segments (regions without a resolvable patch). Every arc is
/// open: consecutive pairs only, closure is explicit.
fn region_highlight_arcs(
    root: &Node,
    region: &BoundaryRegion,
    patch: Option<&BoundarySurfacePatch>,
) -> Vec<Vec<Vec3>> {
    if let Some(patch) = patch {
        return curve_patch_arcs(root, patch, OUTLINE_RING_RESOLUTION);
    }
    let Some(placed) = placed_2d_root(root) else {
        return Vec::new();
    };
    let mut arcs = Vec::new();
    for ring in placed_outline_rings(placed, OUTLINE_RING_RESOLUTION) {
        if ring.len() < 2 {
            continue;
        }
        let Ok(mask) = boundary_region_base_mask(root, region, &ring, None) else {
            continue;
        };
        for i in 0..ring.len() {
            let j = (i + 1) % ring.len();
            if mask[i] && mask[j] {
                arcs.push(vec![ring[i], ring[j]]);
            }
        }
    }
    arcs
}

/// The 2D analog of `region_highlight_mesh` + lift: the arcs of the domain
/// outline that belong to the region, drawn as a thin triangle ribbon
/// centered on the outline and lifted off the sheet plane on BOTH sides
/// (the filled sheet would z-fight a coplanar ribbon, and the domain must
/// read correctly from either side of its plane). Arcs come from
/// `region_highlight_arcs`; each segment is then shortened exactly at each
/// cut crossing.
fn region_highlight_ribbon(
    root: &Node,
    region: &BoundaryRegion,
    scene: &ViewportSurfaceScene,
    color: [f32; 3],
    overlay_id: u32,
    pixel_size: Option<f64>,
) -> Option<ViewportSurface> {
    let placed = placed_2d_root(root)?;
    let plane_normal = placed.normal();
    let diagonal = root
        .bounding_box()
        .map(|bounds| bounds.diagonal())
        .unwrap_or(1.0);
    let step = (diagonal * 1.0e-5).max(1.0e-9);
    let patch = region_curve_patch(root, region);
    // Constant apparent width: size the ribbon by the world length of one
    // screen pixel on the workplane. Without a pixel size (degenerate
    // camera, tests), fall back to the feature the region tags (the operand
    // that owns the patch) — never the whole domain, so editing an
    // unrelated operand cannot fatten the highlight of an unchanged one.
    let (half_width, lift) = match pixel_size {
        Some(pixel) => (pixel * RIBBON_HALF_WIDTH_PIXELS, pixel * RIBBON_LIFT_PIXELS),
        None => {
            let feature_diagonal = patch
                .as_ref()
                .and_then(|patch| patch.owner.bounding_box().ok())
                .map(|bounds| bounds.diagonal())
                .unwrap_or(diagonal);
            (feature_diagonal * 4.0e-3, feature_diagonal * 1.5e-3)
        }
    };
    // Cut volumes once per call; a failed cut_volume hides the highlight
    // (matching the classifier's error behavior), as in the 3D path.
    let mut cut_volumes: Vec<(Node, CutSide)> = Vec::with_capacity(region.cuts.len());
    for cut in &region.cuts {
        let volume = cut_volume(root, cut).ok()?;
        cut_volumes.push((volume, cut.side));
    }
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for arc in region_highlight_arcs(root, region, patch.as_ref()) {
        for pair in arc.windows(2) {
            let (mut p, mut q) = (pair[0], pair[1]);
            let mut dropped = false;
            for (volume, side) in &cut_volumes {
                let keep = |value: f64| match side {
                    CutSide::Inside => value <= 0.0,
                    CutSide::Outside => value >= 0.0,
                };
                let keep_p = keep(volume.eval_point(p));
                let keep_q = keep(volume.eval_point(q));
                match (keep_p, keep_q) {
                    (true, true) => {}
                    (false, false) => {
                        dropped = true;
                        break;
                    }
                    _ => {
                        let crossing = bisect_crossing(volume, p, q);
                        if keep_p {
                            q = crossing;
                        } else {
                            p = crossing;
                        }
                    }
                }
            }
            let chord = q - p;
            if dropped || chord.length() <= diagonal * 1.0e-12 {
                continue;
            }
            // Per-segment outward direction (perpendicular to the chord in
            // the plane, signed by the domain gradient): exact on straight
            // edges, where per-vertex gradients would skew at corners.
            let direction = chord * (1.0 / chord.length());
            let mut outward = direction.cross(plane_normal);
            if outward.dot(sdf_normal(root, (p + q) * 0.5, step)) < 0.0 {
                outward = -outward;
            }
            for lift_sign in [1.0f64, -1.0] {
                let shift = plane_normal * (lift * lift_sign);
                let normal = plane_normal * lift_sign;
                let base = vertices.len() as u32;
                for corner in [
                    p + outward * half_width + shift,
                    p - outward * half_width + shift,
                    q + outward * half_width + shift,
                    q - outward * half_width + shift,
                ] {
                    vertices.push([corner.x as f32, corner.y as f32, corner.z as f32]);
                    normals.push([normal.x as f32, normal.y as f32, normal.z as f32]);
                }
                indices.extend_from_slice(&[base, base + 2, base + 3, base, base + 3, base + 1]);
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
    if root.dimension() == 2 {
        return cutter_ghost_2d(root, kind, points);
    }
    match kind {
        "point" => Err("point knife works on 2D domains — use a segment or stencil".to_string()),
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

/// 2D-domain knives. The segment knife is the same `straight_knife` as 3D
/// but oriented by the WORKPLANE normal: the outline gradient is in-plane,
/// and `side_axis = gradient × line` would come out perpendicular to the
/// sheet — a knife that classifies by height above the plane, degenerate
/// for curve points. With the plane normal, `side_axis` is the in-plane
/// perpendicular and the ghost is a half-plane bounded by the click line.
fn cutter_ghost_2d(root: &Node, kind: &str, points: &[Vec3]) -> Result<KnifeGhost, String> {
    let plane_normal = workplane_normal(root)
        .ok_or_else(|| "2D domain root is not a placed profile".to_string())?;
    let node = match kind {
        "segment" => {
            if points.len() < 2 {
                return Err("segment knife needs two points".to_string());
            }
            straight_knife(root, points[0], *points.last().expect("nonempty"), plane_normal)
                .map_err(|error| error.to_string())?
        }
        "point" => {
            if points.is_empty() {
                return Err("point knife needs one point on the outline".to_string());
            }
            point_knife(root, points[0]).map_err(|error| error.to_string())?
        }
        other => return Err(format!("{other} knife is not available on 2D domains")),
    };
    // Crossing census on the outline: a line crosses a closed curve an even
    // number of times — exactly two is a clean two-arc split; more means
    // each side will hold several arcs (non-convex outline). A knife whose
    // zero line runs ALONG an edge is degenerate and refused.
    let placed = placed_2d_root(root)
        .ok_or_else(|| "2D domain root is not a placed profile".to_string())?;
    let diagonal = root
        .bounding_box()
        .map(|bounds| bounds.diagonal())
        .unwrap_or(1.0);
    let zero_band = diagonal * 1.0e-7;
    let mut crossings = 0usize;
    for ring in placed_outline_rings(placed, OUTLINE_RING_RESOLUTION) {
        for i in 0..ring.len() {
            let j = (i + 1) % ring.len();
            let value_i = node.eval_point(ring[i]);
            let value_j = node.eval_point(ring[j]);
            if value_i.abs() <= zero_band && value_j.abs() <= zero_band {
                return Err(
                    "the cut line runs along the outline — pick points on different edges"
                        .to_string(),
                );
            }
            if (value_i <= 0.0) != (value_j <= 0.0) {
                crossings += 1;
            }
        }
    }
    let mut warnings = Vec::new();
    if crossings > 2 {
        warnings.push(format!(
            "cut line crosses the outline at {crossings} points; each side will contain \
             multiple arcs"
        ));
    }
    Ok(KnifeGhost { node, warnings })
}

/// Cutter click pick: the outline-snapped workplane point for 2D domains,
/// the sphere-traced surface point for 3D. `snap_radius` is the world-space
/// 2D snap distance (screen-derived by the viewport; `None` keeps the
/// kernel's forgiving default).
pub fn pick_cut_point(
    root: &Node,
    origin: Vec3,
    direction: Vec3,
    snap_radius: Option<f64>,
) -> Option<Vec3> {
    if root.dimension() == 2 {
        pick_outline_point_with_radius(root, origin, direction, snap_radius)
    } else {
        pick_surface_point(root, origin, direction)
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

/// Dense on-boundary validation points: outline-ring vertices for 2D
/// domains (they lie ON the outline; the display fill's vertices generally
/// do not), display-mesh vertices for 3D.
pub fn validation_points_for(root: &Node, scene: &ViewportSurfaceScene) -> Vec<Vec3> {
    if root.dimension() == 2 {
        if let Some(placed) = placed_2d_root(root) {
            return placed_outline_rings(placed, OUTLINE_RING_RESOLUTION)
                .into_iter()
                .flatten()
                .collect();
        }
    }
    validation_points(scene)
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

    /// Default von Kármán scene: Difference(flow box half-size 2.25×1.5×0.5
    /// at (2.25, 0, 0.5), Y-axis cylinder r=0.15 at (1.8, 0, 0.5)) — the -X
    /// face (x = 0) is flat and uncut.
    fn fixture() -> (Node, ViewportSurfaceScene) {
        let document = SceneDocument::default_scene().expect("default scene");
        let root = fluid_root_node(&document).expect("fluid root");
        let mut cache = ViewportSurfaceCache::default();
        let scene = build_viewport_surface_scene(std::slice::from_ref(&root), 1, &mut cache);
        (root, scene)
    }

    fn minus_x_face_region(root: &Node) -> BoundaryRegion {
        // The ray must miss the cylinder obstacle (y span ±0.7): its
        // cut-surface patch would win the pick outright.
        let hit = pick_patch(root, vec3(-5.0, 1.0, 0.5), vec3(1.0, 0.0, 0.0))
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
            &[vec3(0.0, 0.0, 0.1), vec3(0.0, 0.0, 0.9)],
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
        let first = vec3(0.0, 0.0, 0.1);
        let second = vec3(0.0, 0.0, 0.9);
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
            vec3(0.0, 0.3, 0.6),
            vec3(0.0, -0.3, 0.6),
            vec3(0.0, 0.5, 0.2),
            vec3(0.0, -0.5, 0.2),
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
            &[vec3(0.0, 0.0, 0.1), vec3(0.0, 0.0, 0.9)],
        )
        .expect("ghost");
        assert!(ghost.warnings.is_empty());
        // Unused warning suppression for IDENTITY_FRAME import parity.
        let _ = IDENTITY_FRAME;
    }

    // --- 2D domains (design_docs/boundary_region_2d.md) ------------------

    use caso_kernel::boundary_ops::surface_patches_for_root;
    use caso_kernel::sdf::placed::PlacedSdf2D;
    use caso_kernel::sdf::primitives_1d::BooleanOp1D;
    use caso_kernel::sdf::primitives_2d::Profile2D;

    /// Planar flow case on z = 0: flowbox rectangle (half 2×1) minus a
    /// circle obstacle (center (0.5, 0), r 0.3), merged like coplanar 2D
    /// scene booleans (one placed node, operand nodes in `sources`).
    fn fixture_2d() -> (Node, ViewportSurfaceScene) {
        fixture_2d_scaled(1.0)
    }

    /// `fixture_2d` with the flowbox rectangle enlarged by `scale`; the
    /// circle obstacle stays the same.
    fn fixture_2d_scaled(scale: f64) -> (Node, ViewportSurfaceScene) {
        let rect_profile = Profile2D::Rectangle {
            center: [0.0, 0.0],
            half_size: [2.0 * scale, 1.0 * scale],
        };
        let circle_profile = Profile2D::Circle {
            center: [0.5, 0.0],
            radius: 0.3,
        };
        let axis_u = vec3(1.0, 0.0, 0.0);
        let axis_v = vec3(0.0, 1.0, 0.0);
        let rect = Node::with_id(
            "flowbox",
            10,
            Shape::PlacedSdf2D(
                PlacedSdf2D::new(rect_profile.clone(), Vec3::ZERO, axis_u, axis_v, Vec::new())
                    .expect("rect"),
            ),
        );
        let circle = Node::with_id(
            "obstacle",
            11,
            Shape::PlacedSdf2D(
                PlacedSdf2D::new(circle_profile.clone(), Vec3::ZERO, axis_u, axis_v, Vec::new())
                    .expect("circle"),
            ),
        );
        let merged = Profile2D::Binary {
            left: Box::new(rect_profile),
            right: Box::new(Profile2D::Offset {
                child: Box::new(circle_profile),
                offset: [0.0, 0.0],
            }),
            operation: BooleanOp1D::Difference,
            smoothing: 0.1,
        };
        let root = Node::with_id(
            "domain",
            12,
            Shape::PlacedSdf2D(
                PlacedSdf2D::new(merged, Vec3::ZERO, axis_u, axis_v, vec![rect, circle])
                    .expect("merged"),
            ),
        );
        let mut cache = ViewportSurfaceCache::default();
        let scene = build_viewport_surface_scene(std::slice::from_ref(&root), 1, &mut cache);
        (root, scene)
    }

    fn left_edge_region(root: &Node) -> BoundaryRegion {
        let patch = surface_patches_for_root(root)
            .into_iter()
            .find(|patch| patch.patch_id == "flowbox.-U")
            .expect("left edge patch");
        let mut region = BoundaryRegion::new("left", 100, patch.owner_object_id);
        region.patch_id = Some(patch.patch_id.clone());
        region.patch_type = Some(patch.patch_type.clone());
        region.outside_direction = patch.outside_direction;
        region
    }

    #[test]
    fn ribbon_highlights_the_edge_region_only() {
        let (root, scene) = fixture_2d();
        let region = left_edge_region(&root);
        let surface = region_highlight_surface(&root, &region, &scene, CANDIDATE_COLOR, 1, None)
            .expect("ribbon");
        assert!(!surface.indices.is_empty());
        // Every ribbon vertex hugs the left edge x = -2, y in [-1, 1],
        // lifted off the z = 0 plane by less than the ribbon scale.
        for vertex in &surface.vertices {
            assert!((vertex[0] as f64 + 2.0).abs() < 0.05, "off-edge x {}", vertex[0]);
            assert!((vertex[1] as f64).abs() < 1.0 + 0.05, "off-edge y {}", vertex[1]);
            assert!((vertex[2] as f64).abs() < 0.05, "off-plane z {}", vertex[2]);
            assert!(vertex[2] != 0.0, "ribbon must be lifted off the sheet plane");
        }
    }

    #[test]
    fn split_preview_ribbons_share_the_exact_split_vertex() {
        let (root, scene) = fixture_2d();
        let region = left_edge_region(&root);
        let ghost = cutter_ghost(&root, "point", &[vec3(-2.0, 0.25, 0.0)])
            .expect("point knife")
            .node;
        let (inside, outside) = split_preview_children(&region, &ghost);
        let inside_surface =
            region_highlight_surface(&root, &inside, &scene, PREVIEW_INSIDE_COLOR, 2, None)
                .expect("inside ribbon");
        let outside_surface =
            region_highlight_surface(&root, &outside, &scene, PREVIEW_OUTSIDE_COLOR, 3, None)
                .expect("outside ribbon");
        // The ribbon offsets along the edge's outward normal (±x) and the
        // plane normal (±z), so y-coordinates are pure arc parameters: the
        // two children must meet at a bitwise-identical split y.
        let ys = |surface: &ViewportSurface| -> Vec<f32> {
            surface.vertices.iter().map(|v| v[1]).collect()
        };
        let inside_ys = ys(&inside_surface);
        let outside_ys = ys(&outside_surface);
        let max = |values: &[f32]| values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = |values: &[f32]| values.iter().cloned().fold(f32::INFINITY, f32::min);
        let shared_touch = max(&inside_ys) == min(&outside_ys)
            || max(&outside_ys) == min(&inside_ys);
        assert!(shared_touch, "children must meet at one bitwise-identical split vertex");
    }

    #[test]
    fn segment_through_the_hole_warns_about_multiple_arcs() {
        let (root, _scene) = fixture_2d();
        // The line y = 0 crosses the rectangle twice AND the circle twice.
        let ghost = cutter_ghost(
            &root,
            "segment",
            &[vec3(-2.0, 0.0, 0.0), vec3(2.0, 0.0, 0.0)],
        )
        .expect("ghost");
        assert!(
            ghost.warnings.iter().any(|warning| warning.contains("4 points")),
            "expected the multi-arc warning, got {:?}",
            ghost.warnings
        );
    }

    #[test]
    fn clean_two_arc_segment_raises_no_warning() {
        let (root, _scene) = fixture_2d();
        // A chord across the top-left corner misses the obstacle: exactly
        // two crossings.
        let ghost = cutter_ghost(
            &root,
            "segment",
            &[vec3(-2.0, 0.5, 0.0), vec3(-0.5, 1.0, 0.0)],
        )
        .expect("ghost");
        assert!(ghost.warnings.is_empty(), "unexpected: {:?}", ghost.warnings);
    }

    #[test]
    fn collinear_segment_on_one_edge_is_refused() {
        let (root, _scene) = fixture_2d();
        let error = cutter_ghost(
            &root,
            "segment",
            &[vec3(-2.0, -0.5, 0.0), vec3(-2.0, 0.5, 0.0)],
        )
        .expect_err("collinear cut refused");
        assert!(error.contains("along the outline"), "unexpected error: {error}");
    }

    #[test]
    fn point_knife_is_refused_on_3d_domains() {
        let (root, _scene) = fixture();
        let error = cutter_ghost(&root, "point", &[vec3(0.0, 0.0, 0.5)])
            .expect_err("3D root refused");
        assert!(error.contains("2D"), "unexpected error: {error}");
    }

    /// Ribbon width of a region's highlight: the first quad's first two
    /// vertices are `p ± outward * half_width`.
    fn ribbon_width(
        root: &Node,
        scene: &ViewportSurfaceScene,
        patch_id: &str,
        pixel_size: Option<f64>,
    ) -> f64 {
        let patch = surface_patches_for_root(root)
            .into_iter()
            .find(|patch| patch.patch_id == patch_id)
            .unwrap_or_else(|| panic!("patch {patch_id} exists"));
        let mut region = BoundaryRegion::new("region", 100, patch.owner_object_id);
        region.patch_id = Some(patch.patch_id.clone());
        region.patch_type = Some(patch.patch_type.clone());
        region.outside_direction = patch.outside_direction;
        let surface =
            region_highlight_ribbon(root, &region, scene, [1.0, 0.0, 0.0], 999, pixel_size)
                .expect("ribbon");
        let a = surface.vertices[0];
        let b = surface.vertices[1];
        vec3(
            (a[0] - b[0]) as f64,
            (a[1] - b[1]) as f64,
            (a[2] - b[2]) as f64,
        )
        .length()
    }

    #[test]
    fn highlight_width_follows_the_patch_owner_not_the_whole_domain() {
        // No pixel size (degenerate camera): the fallback scales with the
        // tagged patch's owner, never the whole domain.
        let (small, small_scene) = fixture_2d();
        let (large, large_scene) = fixture_2d_scaled(3.0);
        // The circle is unchanged, so its highlight must not grow with the
        // flowbox (it used to scale with the whole root's diagonal).
        let hole = "cut_surface.obstacle.outline";
        let width_small = ribbon_width(&small, &small_scene, hole, None);
        let width_large = ribbon_width(&large, &large_scene, hole, None);
        assert!(
            ((width_large - width_small) / width_small).abs() < 1.0e-5,
            "circle ribbon width changed with the flowbox: {width_small} -> {width_large}"
        );
        // A flowbox edge is owned by the flowbox: its ribbon still scales.
        let edge_small = ribbon_width(&small, &small_scene, "flowbox.-U", None);
        let edge_large = ribbon_width(&large, &large_scene, "flowbox.-U", None);
        assert!(
            edge_large > edge_small * 2.0,
            "flowbox edge ribbon should scale with the flowbox: {edge_small} -> {edge_large}"
        );
    }

    #[test]
    fn highlight_width_is_pixel_constant_across_features_and_scales() {
        let (small, small_scene) = fixture_2d();
        let (large, large_scene) = fixture_2d_scaled(3.0);
        let pixel = Some(0.01);
        let expected = 2.0 * 0.01 * RIBBON_HALF_WIDTH_PIXELS;
        for (label, width) in [
            (
                "circle/small",
                ribbon_width(&small, &small_scene, "cut_surface.obstacle.outline", pixel),
            ),
            (
                "edge/small",
                ribbon_width(&small, &small_scene, "flowbox.-U", pixel),
            ),
            (
                "circle/large",
                ribbon_width(&large, &large_scene, "cut_surface.obstacle.outline", pixel),
            ),
            (
                "edge/large",
                ribbon_width(&large, &large_scene, "flowbox.-U", pixel),
            ),
        ] {
            // Tolerance covers the f32 vertex storage (~1e-6 absolute at the
            // large fixture's coordinates); the old behavior was off by 3-10x.
            assert!(
                ((width - expected) / expected).abs() < 1.0e-3,
                "{label}: ribbon width {width} != {expected}"
            );
        }
    }
}
