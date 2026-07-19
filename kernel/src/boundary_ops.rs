//! Boundary-region classification, surface attribution, and analytic patch
//! picking — the ports of `core/boundary_region.py`, `core/sdf_attribution.py`
//! and the classification/picking half of `core/boundary_patches.py`.
//!
//! A world point belongs to a region iff (boundary_region_v2 §2):
//! 1. it lies on the Domain boundary            (|f_root| <= tolerance)
//! 2. the region's owner leaf is the active operand there (provenance —
//!    for Subtract the OBSTACLE owns the cut surface)
//! 3. it is within the analytic patch scope, when the region has one
//! 4. every cut in the chain is satisfied       (ghost sign vs side)
//!
//! Only sign tests of known SDFs — exact, cheap, tessellation-independent.
//! Tolerances are scale-relative (owner extent), never absolute meters.

use crate::boundary::{BoundaryCut, BoundaryRegion, CutSide};
use crate::error::{GeometryError, GeometryResult};
use crate::frame::Frame;
use crate::sdf::node::{Node, RotationAxis, Shape};
use crate::sdf::placed::PlacedSdf2D;
use crate::sdf::primitives_1d::{BooleanOp1D, Profile1D};
use crate::sdf::primitives_2d::{Point2, Profile2D};
use crate::sdf::primitives_3d::{Box3, Cylinder, Sphere};
use crate::sdf::solid_from_2d::Extrude;
use crate::vec3::{vec3, Vec3};

/// On-surface band as a fraction of the owner's bounding-box diagonal.
pub const RELATIVE_SURFACE_TOLERANCE: f64 = 1.0e-3;
pub const DIRECTION_ALIGNMENT_MINIMUM: f64 = 0.95;
pub const PATCH_TOLERANCE: f64 = 1.5e-3;
/// Default 2D hover pick radius (fraction of the domain diagonal) when the
/// caller supplies no screen-derived radius. Deliberately tight — hover must
/// not trigger from far away in a CAD viewport.
pub const CURVE_PATCH_PICK_TOLERANCE: f64 = 0.01;
/// Cutter-click snap radius (fraction of the domain diagonal): knife
/// endpoints stay forgiving — clicks near the outline snap onto it, far
/// clicks keep the raw in-plane point.
pub const OUTLINE_SNAP_TOLERANCE: f64 = 0.05;

// ---------------------------------------------------------------------------
// sdf_attribution
// ---------------------------------------------------------------------------

/// Object IDs that attribution may assign to the final boundary (leaves).
pub fn boundary_owner_ids(node: &Node) -> Vec<u32> {
    let mut ids = Vec::new();
    collect_leaf_ids(node, &mut ids);
    ids
}

fn collect_leaf_ids(node: &Node, ids: &mut Vec<u32>) {
    // Recurse only through operators and transforms: any other node is a
    // provenance leaf that owns its whole surface, matching `walk_owner` and
    // `evaluate_with_attribution` (generators keep their section internal).
    match &node.shape {
        Shape::Union(op) | Shape::Intersection(op) | Shape::Difference(op) | Shape::Xor(op) => {
            collect_leaf_ids(&op.left, ids);
            collect_leaf_ids(&op.right, ids);
        }
        Shape::Translate { child, .. }
        | Shape::Scale { child, .. }
        | Shape::Rotate { child, .. } => collect_leaf_ids(child, ids),
        _ => {
            if !ids.contains(&node.object_id) {
                ids.push(node.object_id);
            }
        }
    }
}

/// Ray-march an SDF and return the first visible surface point
/// (`pick_sdf_surface`).
pub fn pick_sdf_surface(
    root: &Node,
    ray_origin: Vec3,
    ray_direction: Vec3,
    hit_tolerance: f64,
    maximum_travel: f64,
) -> Option<Vec3> {
    let mut travel = 0.0;
    for _ in 0..160 {
        let point = ray_origin + ray_direction * travel;
        let value = root.eval_point(point);
        if value.abs() < hit_tolerance {
            return Some(point);
        }
        travel += value.abs().max(0.0002);
        if travel > maximum_travel {
            break;
        }
    }
    None
}

/// SDF gradient direction via central differences (`_sdf_normal`).
pub fn sdf_normal(root: &Node, point: Vec3, step: f64) -> Vec3 {
    let gradient = vec3(
        root.eval_point(point + vec3(step, 0.0, 0.0)) - root.eval_point(point - vec3(step, 0.0, 0.0)),
        root.eval_point(point + vec3(0.0, step, 0.0)) - root.eval_point(point - vec3(0.0, step, 0.0)),
        root.eval_point(point + vec3(0.0, 0.0, step)) - root.eval_point(point - vec3(0.0, 0.0, step)),
    );
    gradient * (1.0 / gradient.length().max(1.0e-12))
}

/// Evaluate the SDF at one point and identify the controlling leaf
/// (`evaluate_with_attribution`, point form).
pub fn evaluate_with_attribution(node: &Node, point: Vec3) -> (f64, u32) {
    match &node.shape {
        Shape::Translate { child, offset } => evaluate_with_attribution(child, point - *offset),
        Shape::Scale { child, factor } => {
            let (distance, id) = evaluate_with_attribution(child, point * (1.0 / *factor));
            (distance * factor, id)
        }
        Shape::Rotate {
            child,
            axis,
            angle_degrees,
        } => evaluate_with_attribution(child, rotate_local(point, *axis, *angle_degrees)),
        Shape::Union(operands) => {
            let (left, left_id) = evaluate_with_attribution(&operands.left, point);
            let (right, right_id) = evaluate_with_attribution(&operands.right, point);
            if left <= right {
                (left, left_id)
            } else {
                (right, right_id)
            }
        }
        Shape::Intersection(operands) => {
            let (left, left_id) = evaluate_with_attribution(&operands.left, point);
            let (right, right_id) = evaluate_with_attribution(&operands.right, point);
            if left >= right {
                (left, left_id)
            } else {
                (right, right_id)
            }
        }
        Shape::Difference(operands) => {
            let (left, left_id) = evaluate_with_attribution(&operands.left, point);
            let (right, right_id) = evaluate_with_attribution(&operands.right, point);
            if left >= -right {
                (left, left_id)
            } else {
                (-right, right_id)
            }
        }
        Shape::Xor(operands) => {
            let (left, left_id) = evaluate_with_attribution(&operands.left, point);
            let (right, right_id) = evaluate_with_attribution(&operands.right, point);
            let minimum = left.min(right);
            let negative_maximum = -left.max(right);
            let choose_left = if minimum >= negative_maximum {
                left <= right
            } else {
                left >= right
            };
            (
                minimum.max(negative_maximum),
                if choose_left { left_id } else { right_id },
            )
        }
        _ => (node.eval_point(point), node.object_id),
    }
}

/// Ray-march the final SDF and return (hit point, controlling owner id,
/// surface normal) — `pick_boundary_owner`.
pub fn pick_boundary_owner(
    root: &Node,
    ray_origin: Vec3,
    ray_direction: Vec3,
    hit_tolerance: f64,
    maximum_travel: f64,
) -> Option<(Vec3, u32, Vec3)> {
    let point = pick_sdf_surface(root, ray_origin, ray_direction, hit_tolerance, maximum_travel)?;
    let (_distance, owner_id) = evaluate_with_attribution(root, point);
    let normal = sdf_normal(root, point, hit_tolerance);
    Some((point, owner_id, normal))
}

fn rotate_local(point: Vec3, axis: RotationAxis, angle_degrees: f64) -> Vec3 {
    let angle = angle_degrees.to_radians();
    let (s, c) = (angle.sin(), angle.cos());
    match axis {
        RotationAxis::X => vec3(point.x, c * point.y + s * point.z, -s * point.y + c * point.z),
        RotationAxis::Y => vec3(c * point.x - s * point.z, point.y, s * point.x + c * point.z),
        RotationAxis::Z => vec3(c * point.x + s * point.y, -s * point.x + c * point.y, point.z),
    }
}

// ---------------------------------------------------------------------------
// owner-activity walk (classifier criterion 2)
// ---------------------------------------------------------------------------

fn walk_owner(
    node: &Node,
    owner_object_id: u32,
    points: &[Vec3],
    tie: f64,
) -> (Vec<f64>, Vec<bool>) {
    match &node.shape {
        Shape::Translate { child, offset } => {
            let moved: Vec<Vec3> = points.iter().map(|p| *p - *offset).collect();
            walk_owner(child, owner_object_id, &moved, tie)
        }
        Shape::Scale { child, factor } => {
            let scaled: Vec<Vec3> = points.iter().map(|p| *p * (1.0 / *factor)).collect();
            let (values, mask) = walk_owner(child, owner_object_id, &scaled, tie / *factor);
            (values.into_iter().map(|v| v * *factor).collect(), mask)
        }
        Shape::Rotate {
            child,
            axis,
            angle_degrees,
        } => {
            let rotated: Vec<Vec3> = points
                .iter()
                .map(|p| rotate_local(*p, *axis, *angle_degrees))
                .collect();
            walk_owner(child, owner_object_id, &rotated, tie)
        }
        Shape::Union(operands) => {
            let (lv, lm) = walk_owner(&operands.left, owner_object_id, points, tie);
            let (rv, rm) = walk_owner(&operands.right, owner_object_id, points, tie);
            let mut values = Vec::with_capacity(points.len());
            let mut mask = Vec::with_capacity(points.len());
            for i in 0..points.len() {
                values.push(lv[i].min(rv[i]));
                mask.push((lm[i] && lv[i] <= rv[i] + tie) || (rm[i] && rv[i] <= lv[i] + tie));
            }
            (values, mask)
        }
        Shape::Intersection(operands) => {
            let (lv, lm) = walk_owner(&operands.left, owner_object_id, points, tie);
            let (rv, rm) = walk_owner(&operands.right, owner_object_id, points, tie);
            let mut values = Vec::with_capacity(points.len());
            let mut mask = Vec::with_capacity(points.len());
            for i in 0..points.len() {
                values.push(lv[i].max(rv[i]));
                mask.push((lm[i] && lv[i] >= rv[i] - tie) || (rm[i] && rv[i] >= lv[i] - tie));
            }
            (values, mask)
        }
        Shape::Difference(operands) => {
            let (lv, lm) = walk_owner(&operands.left, owner_object_id, points, tie);
            let (rv, rm) = walk_owner(&operands.right, owner_object_id, points, tie);
            let mut values = Vec::with_capacity(points.len());
            let mut mask = Vec::with_capacity(points.len());
            for i in 0..points.len() {
                values.push(lv[i].max(-rv[i]));
                mask.push((lm[i] && lv[i] >= -rv[i] - tie) || (rm[i] && -rv[i] >= lv[i] - tie));
            }
            (values, mask)
        }
        Shape::Xor(operands) => {
            let (lv, lm) = walk_owner(&operands.left, owner_object_id, points, tie);
            let (rv, rm) = walk_owner(&operands.right, owner_object_id, points, tie);
            let mut values = Vec::with_capacity(points.len());
            let mut mask = Vec::with_capacity(points.len());
            for i in 0..points.len() {
                let inner = lv[i].min(rv[i]);
                let outer = -lv[i].max(rv[i]);
                values.push(inner.max(outer));
                let inner_active = inner >= outer - tie;
                let outer_active = outer >= inner - tie;
                let inner_mask = (lm[i] && lv[i] <= rv[i] + tie) || (rm[i] && rv[i] <= lv[i] + tie);
                let outer_mask = (lm[i] && lv[i] >= rv[i] - tie) || (rm[i] && rv[i] >= lv[i] - tie);
                mask.push((inner_active && inner_mask) || (outer_active && outer_mask));
            }
            (values, mask)
        }
        _ => {
            // Any non-operator, non-transform node is a provenance leaf:
            // generators (Extrude/Revolve), tubes, placed nodes own their
            // whole surface.
            let values: Vec<f64> = points.iter().map(|p| node.eval_point(*p)).collect();
            let active = node.object_id == owner_object_id;
            (values, vec![active; points.len()])
        }
    }
}

/// Which points have `owner_object_id` as the active operand (provenance).
pub fn owner_active_mask(
    root: &Node,
    owner_object_id: u32,
    points: &[Vec3],
    tie_tolerance: f64,
) -> Vec<bool> {
    walk_owner(root, owner_object_id, points, tie_tolerance).1
}

// ---------------------------------------------------------------------------
// classifier (criteria 1-4)
// ---------------------------------------------------------------------------

pub fn find_node_by_object_id(root: &Node, object_id: u32) -> Option<&Node> {
    if root.object_id == object_id {
        return Some(root);
    }
    root.children()
        .into_iter()
        .find_map(|child| find_node_by_object_id(child, object_id))
}

fn bounding_diagonal(node: &Node) -> f64 {
    node.bounding_box().map(|b| b.diagonal()).unwrap_or(0.0)
}

/// Scale-relative on-surface band: owner extent when resolvable, else the
/// Domain extent (`region_tolerance`).
pub fn region_tolerance(root: &Node, region: &BoundaryRegion) -> f64 {
    let reference =
        find_node_by_object_id(root, region.owner_object_id).unwrap_or(root);
    let mut diagonal = bounding_diagonal(reference);
    if !diagonal.is_finite() || diagonal <= 0.0 {
        diagonal = bounding_diagonal(root).max(1.0e-9);
    }
    RELATIVE_SURFACE_TOLERANCE * diagonal
}

/// The classification field of one cut: a 3D ghost as-is, a lower-dim ghost
/// extruded through the scene (`cut_volume`).
pub fn cut_volume(root: &Node, cut: &BoundaryCut) -> GeometryResult<Node> {
    if cut.ghost.dimension() == 3 {
        return Ok(cut.ghost.clone());
    }
    surface_selector_volume(root, &cut.ghost)?.ok_or_else(|| {
        GeometryError::new(format!(
            "boundary cut ghost {:?} cannot be converted to a classification volume",
            cut.ghost.name
        ))
    })
}

/// Coarse near-boundary point cloud + its band width
/// (`sample_boundary_points`) — diagnostics only, never membership.
pub fn sample_boundary_points(root: &Node, resolution: usize) -> GeometryResult<(Vec<Vec3>, f64)> {
    let bounds = root.bounding_box()?;
    let steps = resolution.max(2);
    let lin = |lo: f64, hi: f64, i: usize| lo + (hi - lo) * (i as f64) / ((steps - 1) as f64);
    let cell = ((bounds.x_max - bounds.x_min) / (steps - 1) as f64)
        .max((bounds.y_max - bounds.y_min) / (steps - 1) as f64)
        .max((bounds.z_max - bounds.z_min) / (steps - 1) as f64)
        .max(1.0e-12);
    let band = 0.75 * cell;
    let mut kept = Vec::new();
    for i in 0..steps {
        for j in 0..steps {
            for k in 0..steps {
                let point = vec3(
                    lin(bounds.x_min, bounds.x_max, i),
                    lin(bounds.y_min, bounds.y_max, j),
                    lin(bounds.z_min, bounds.z_max, k),
                );
                if root.eval_point(point).abs() <= band {
                    kept.push(point);
                }
            }
        }
    }
    Ok((kept, band))
}

/// Membership under criteria 1-3 only (on the Domain boundary, owner
/// provenance, analytic patch/direction scope) — the mesh-aligned part of the
/// classifier, ignoring the region's cut chain. Highlight overlays filter
/// whole display triangles with this mask and then clip the survivors
/// exactly against each cut volume.
pub fn boundary_region_base_mask(
    root: &Node,
    region: &BoundaryRegion,
    points: &[Vec3],
    tolerance: Option<f64>,
) -> GeometryResult<Vec<bool>> {
    let tol = tolerance.unwrap_or_else(|| region_tolerance(root, region));

    // 1. on the Domain boundary
    let mut mask: Vec<bool> = points
        .iter()
        .map(|p| root.eval_point(*p).abs() <= tol)
        .collect();

    // 2. owner leaf active (provenance)
    let owner_mask = owner_active_mask(root, region.owner_object_id, points, tol);
    for (m, owner) in mask.iter_mut().zip(owner_mask) {
        *m = *m && owner;
    }

    // 3. analytic patch scope
    if region.patch_id.is_some() {
        let scope = boundary_region_scope_mask(root, region, points, tol)?;
        for (m, in_scope) in mask.iter_mut().zip(scope) {
            *m = *m && in_scope;
        }
    } else if let Some(direction_index) = region.outside_direction {
        let owner = find_node_by_object_id(root, region.owner_object_id);
        if let Some(direction) =
            owner.and_then(|node| owner_outside_direction_vector(node, direction_index))
        {
            let step = (tol * 0.5).max(1.0e-9);
            for (m, point) in mask.iter_mut().zip(points) {
                if *m {
                    let normal = sdf_normal(root, *point, step);
                    *m = normal.dot(direction) >= DIRECTION_ALIGNMENT_MINIMUM;
                }
            }
        }
    }
    Ok(mask)
}

/// Exact membership of `points` in `region` (`boundary_region_mask`):
/// criteria 1-3 (`boundary_region_base_mask`) ∧ the cut chain (criterion 4).
pub fn boundary_region_mask(
    root: &Node,
    region: &BoundaryRegion,
    points: &[Vec3],
    tolerance: Option<f64>,
) -> GeometryResult<Vec<bool>> {
    let tol = tolerance.unwrap_or_else(|| region_tolerance(root, region));
    let mut mask = boundary_region_base_mask(root, region, points, Some(tol))?;

    // 4. the cut chain (conjunction)
    for cut in &region.cuts {
        let volume = cut_volume(root, cut)?;
        for (m, point) in mask.iter_mut().zip(points) {
            if *m {
                let inside = volume.eval_point(*point) <= tol;
                *m = match cut.side {
                    CutSide::Inside => inside,
                    CutSide::Outside => !inside,
                };
            }
        }
    }
    Ok(mask)
}

/// `owner_outside_direction_vector` from `core/boundary_direction.py`.
pub fn owner_outside_direction_vector(owner: &Node, direction: u8) -> Option<Vec3> {
    let directions = owner_outside_direction_vectors(owner);
    directions.get(direction as usize).copied()
}

fn frame_directions(frame: &Frame) -> Vec<Vec3> {
    vec![
        -frame.u, frame.u, -frame.v, frame.v, -frame.w, frame.w,
    ]
}

fn owner_outside_direction_vectors(owner: &Node) -> Vec<Vec3> {
    match &owner.shape {
        Shape::PlacedSdf2D(placed) => vec![
            -placed.axis_u,
            placed.axis_u,
            -placed.axis_v,
            placed.axis_v,
        ],
        Shape::Box3(shape) => frame_directions(&shape.frame),
        Shape::BoxFrame(shape) => frame_directions(&shape.frame),
        Shape::CappedCone(shape) => frame_directions(&shape.frame),
        Shape::Cone(shape) => frame_directions(&shape.frame),
        Shape::Cylinder(shape) => frame_directions(&shape.frame),
        Shape::Pyramid(shape) => frame_directions(&shape.frame),
        Shape::Torus(shape) => frame_directions(&shape.frame),
        _ => vec![
            vec3(-1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, -1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 0.0, 1.0),
        ],
    }
}

// ---------------------------------------------------------------------------
// analytic surface patches + ray pick (`core/boundary_patches.py` subset)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct BoundarySurfacePatch {
    pub owner_object_id: u32,
    pub patch_id: String,
    pub patch_type: String,
    /// Path from the root to the owner is not retained; the owner node is
    /// cloned (matching the Python dataclass holding a node reference).
    /// For curve patches (2D domains) this is the boolean *operand* whose
    /// outline the patch belongs to, while `owner_object_id` stays the
    /// merged root's id — the single provenance leaf a 2D domain has.
    pub owner: Node,
    pub normal: Option<Vec3>,
    pub outside_direction: Option<u8>,
    pub normal_sign: f64,
    /// Present only for 2D domains: the patch is a piece of the outline
    /// curve, not a surface.
    pub curve: Option<CurvePatchKind>,
}

/// One nameable piece of a 2D domain's outline curve — the curve analog of a
/// box face (2D domain → 1D boundary, one dimension down from 3D → 2D).
#[derive(Debug, Clone, PartialEq)]
pub enum CurvePatchKind {
    /// A straight edge between two world points with its in-plane outward
    /// normal (rectangle edge, polygon segment).
    Edge {
        start: Vec3,
        end: Vec3,
        outward: Vec3,
    },
    /// The operand's whole outline (circles, beziers, non-decomposable
    /// profiles) — the curve analog of the generic whole-surface patch.
    Outline,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryPatchHit {
    pub point: Vec3,
    pub owner_object_id: u32,
    pub patch_id: String,
    pub patch_type: String,
    pub normal: Vec3,
    pub outside_direction: Option<u8>,
}

fn patch_id(name: &str, cut_surface: bool) -> String {
    if cut_surface {
        format!("cut_surface.{name}")
    } else {
        name.to_string()
    }
}

pub fn surface_patches_for_root(root: &Node) -> Vec<BoundarySurfacePatch> {
    let mut patches = Vec::new();
    surface_patches_for_node(root, false, 1.0, &mut patches);
    patches
}

fn surface_patches_for_node(
    node: &Node,
    cut_surface: bool,
    normal_sign: f64,
    out: &mut Vec<BoundarySurfacePatch>,
) {
    match &node.shape {
        Shape::Translate { child, .. }
        | Shape::Scale { child, .. }
        | Shape::Rotate { child, .. } => {
            surface_patches_for_node(child, cut_surface, normal_sign, out)
        }
        Shape::Difference(operands) => {
            surface_patches_for_node(&operands.left, cut_surface, normal_sign, out);
            surface_patches_for_node(&operands.right, true, -normal_sign, out);
        }
        Shape::Union(operands) | Shape::Intersection(operands) | Shape::Xor(operands) => {
            surface_patches_for_node(&operands.left, cut_surface, normal_sign, out);
            surface_patches_for_node(&operands.right, cut_surface, normal_sign, out);
        }
        Shape::Box3(shape) => {
            let names = ["-X", "+X", "-Y", "+Y", "-Z", "+Z"];
            let normals = frame_directions(&shape.frame);
            for (index, (name, normal)) in names.iter().zip(normals).enumerate() {
                out.push(BoundarySurfacePatch {
                    owner_object_id: node.object_id,
                    patch_id: patch_id(name, cut_surface),
                    patch_type: if cut_surface { "cut_surface" } else { "face" }.to_string(),
                    owner: node.clone(),
                    normal: Some(normal),
                    outside_direction: Some(index as u8),
                    normal_sign,
                    curve: None,
                });
            }
        }
        Shape::Cylinder(_) | Shape::Cone(_) | Shape::CappedCone(_) => {
            let frame = cylinder_like_frame(node).expect("cylinder-like frame");
            out.push(BoundarySurfacePatch {
                owner_object_id: node.object_id,
                patch_id: patch_id("side_wall", cut_surface),
                patch_type: if cut_surface { "cut_surface" } else { "side_wall" }.to_string(),
                owner: node.clone(),
                normal: None,
                outside_direction: None,
                normal_sign,
                curve: None,
            });
            out.push(BoundarySurfacePatch {
                owner_object_id: node.object_id,
                patch_id: patch_id("-Z_cap", cut_surface),
                patch_type: if cut_surface { "cut_surface" } else { "cap" }.to_string(),
                owner: node.clone(),
                normal: Some(-frame.w),
                outside_direction: Some(4),
                normal_sign,
                curve: None,
            });
            out.push(BoundarySurfacePatch {
                owner_object_id: node.object_id,
                patch_id: patch_id("+Z_cap", cut_surface),
                patch_type: if cut_surface { "cut_surface" } else { "cap" }.to_string(),
                owner: node.clone(),
                normal: Some(frame.w),
                outside_direction: Some(5),
                normal_sign,
                curve: None,
            });
        }
        Shape::PlacedSdf2D(_) => {
            if node.object_id == 0 {
                return;
            }
            // 2D domain root: the boundary is the outline curve. Coplanar 2D
            // booleans merge into ONE placed node (the single provenance
            // leaf), keeping the operand nodes in `sources` — so the patch
            // walk recurses the source tree, not the scene operators.
            curve_patches_for_placed(node, node, cut_surface, normal_sign, out);
        }
        _ => {
            if node.object_id == 0 || node.dimension() != 3 {
                return;
            }
            // Generic fallback (boundary_region_v2 §7): every 3D provenance
            // leaf owns at least its whole surface.
            out.push(BoundarySurfacePatch {
                owner_object_id: node.object_id,
                patch_id: patch_id("surface", cut_surface),
                patch_type: if cut_surface { "cut_surface" } else { "surface" }.to_string(),
                owner: node.clone(),
                normal: None,
                outside_direction: None,
                normal_sign,
                curve: None,
            });
        }
    }
}

/// Curve patches of one operand in a 2D domain's source tree. `owner` stays
/// the merged root (its id is the provenance leaf id every region stores);
/// `node` is the operand whose outline is decomposed. A Difference right
/// operand becomes cut-surface patches with negated normal sign, exactly
/// like the 3D Difference arm above.
fn curve_patches_for_placed(
    owner: &Node,
    node: &Node,
    cut_surface: bool,
    normal_sign: f64,
    out: &mut Vec<BoundarySurfacePatch>,
) {
    let Shape::PlacedSdf2D(placed) = &node.shape else {
        return;
    };
    if placed.sources.len() == 2 {
        if let Profile2D::Binary { operation, .. } = &placed.profile {
            match operation {
                BooleanOp1D::Difference => {
                    curve_patches_for_placed(owner, &placed.sources[0], cut_surface, normal_sign, out);
                    curve_patches_for_placed(owner, &placed.sources[1], true, -normal_sign, out);
                }
                _ => {
                    curve_patches_for_placed(owner, &placed.sources[0], cut_surface, normal_sign, out);
                    curve_patches_for_placed(owner, &placed.sources[1], cut_surface, normal_sign, out);
                }
            }
            return;
        }
    }
    // Leaf operand: straight-edge profiles decompose into named edges
    // (the curve analog of box → 6 faces); everything else is one whole
    // outline patch.
    let prefix = if node.object_id == owner.object_id {
        String::new()
    } else {
        format!("{}.", node.name)
    };
    let Shape::PlacedSdf2D(owner_placed) = &owner.shape else {
        return;
    };
    let in_plane = |point: Point2| placed.origin + placed.axis_u * point[0] + placed.axis_v * point[1];
    let edges: Option<Vec<ProfileEdge>> = match &placed.profile {
        Profile2D::Rectangle { center, half_size } => Some(rectangle_edges(*center, *half_size)),
        Profile2D::Square { center, half_size } => {
            Some(rectangle_edges(*center, [*half_size, *half_size]))
        }
        Profile2D::Polygon { points } => Some(polygon_edges(points)),
        _ => None,
    };
    match edges {
        Some(edges) => {
            for (name, start, end, outward2) in edges {
                let outward = placed.axis_u * outward2[0] + placed.axis_v * outward2[1];
                // outside_direction indexes the OWNER's ±axis_u/±axis_v
                // (the PlacedSdf2D arm of `owner_outside_direction_vectors`).
                let owner_axes = [
                    -owner_placed.axis_u,
                    owner_placed.axis_u,
                    -owner_placed.axis_v,
                    owner_placed.axis_v,
                ];
                let outside_direction = if cut_surface {
                    None
                } else {
                    owner_axes
                        .iter()
                        .position(|axis| axis.dot(outward) > 0.999)
                        .map(|index| index as u8)
                };
                out.push(BoundarySurfacePatch {
                    owner_object_id: owner.object_id,
                    patch_id: patch_id(&format!("{prefix}{name}"), cut_surface),
                    patch_type: if cut_surface { "cut_surface" } else { "edge" }.to_string(),
                    owner: node.clone(),
                    normal: Some(outward),
                    outside_direction,
                    normal_sign,
                    curve: Some(CurvePatchKind::Edge {
                        start: in_plane(start),
                        end: in_plane(end),
                        outward,
                    }),
                });
            }
        }
        None => {
            out.push(BoundarySurfacePatch {
                owner_object_id: owner.object_id,
                patch_id: patch_id(&format!("{prefix}outline"), cut_surface),
                patch_type: if cut_surface { "cut_surface" } else { "outline" }.to_string(),
                owner: node.clone(),
                normal: None,
                outside_direction: None,
                normal_sign,
                curve: Some(CurvePatchKind::Outline),
            });
        }
    }
}

/// One straight profile edge: (name, start, end, outward normal), all in
/// profile (u, v) coordinates.
type ProfileEdge = (String, Point2, Point2, [f64; 2]);

/// The four edges of an axis-aligned rectangle in profile coordinates:
/// (name, start, end, outward normal).
fn rectangle_edges(
    center: Point2,
    half_size: Point2,
) -> Vec<ProfileEdge> {
    let (cx, cy) = (center[0], center[1]);
    let (hx, hy) = (half_size[0], half_size[1]);
    vec![
        (
            "-U".to_string(),
            [cx - hx, cy - hy],
            [cx - hx, cy + hy],
            [-1.0, 0.0],
        ),
        (
            "+U".to_string(),
            [cx + hx, cy - hy],
            [cx + hx, cy + hy],
            [1.0, 0.0],
        ),
        (
            "-V".to_string(),
            [cx - hx, cy - hy],
            [cx + hx, cy - hy],
            [0.0, -1.0],
        ),
        (
            "+V".to_string(),
            [cx - hx, cy + hy],
            [cx + hx, cy + hy],
            [0.0, 1.0],
        ),
    ]
}

/// The closed polygon's edges with outward normals from its winding.
fn polygon_edges(points: &[Point2]) -> Vec<ProfileEdge> {
    let count = points.len();
    let signed_area: f64 = (0..count)
        .map(|i| {
            let a = points[i];
            let b = points[(i + 1) % count];
            a[0] * b[1] - b[0] * a[1]
        })
        .sum();
    let winding = if signed_area >= 0.0 { 1.0 } else { -1.0 };
    (0..count)
        .map(|i| {
            let a = points[i];
            let b = points[(i + 1) % count];
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let length = (dx * dx + dy * dy).sqrt().max(1.0e-12);
            // CCW winding: outward = edge direction rotated -90 degrees.
            let outward = [winding * dy / length, -winding * dx / length];
            (format!("edge_{i}"), a, b, outward)
        })
        .collect()
}

fn cylinder_like_frame(node: &Node) -> Option<&Frame> {
    match &node.shape {
        Shape::Cylinder(shape) => Some(&shape.frame),
        Shape::Cone(shape) => Some(&shape.frame),
        Shape::CappedCone(shape) => Some(&shape.frame),
        _ => None,
    }
}

fn cylinder_like_radius(node: &Node) -> Option<f64> {
    match &node.shape {
        Shape::Cylinder(shape) => Some(shape.radius),
        Shape::Cone(shape) => Some(shape.radius),
        Shape::CappedCone(shape) => Some(shape.radius_a.max(shape.radius_b)),
        _ => None,
    }
}

fn cylinder_like_half_height(node: &Node) -> Option<f64> {
    match &node.shape {
        Shape::Cylinder(shape) => Some(shape.half_height),
        Shape::Cone(shape) => Some(shape.half_height),
        Shape::CappedCone(shape) => Some(shape.half_height),
        _ => None,
    }
}

fn cylinder_like_center(node: &Node) -> Option<Vec3> {
    match &node.shape {
        Shape::Cylinder(shape) => Some(shape.center),
        Shape::Cone(shape) => Some(shape.center),
        Shape::CappedCone(shape) => Some(shape.center),
        _ => None,
    }
}

fn oriented_local(point: Vec3, center: Vec3, frame: &Frame) -> Vec3 {
    let relative = point - center;
    vec3(
        relative.dot(frame.u),
        relative.dot(frame.v),
        relative.dot(frame.w),
    )
}

fn normal_alignment(first: Vec3, second: Vec3) -> f64 {
    let a = first * (1.0 / first.length().max(1.0e-12));
    let b = second * (1.0 / second.length().max(1.0e-12));
    a.dot(b)
}

fn box_face_axis(face: &str) -> Option<(usize, f64)> {
    let index = match face {
        "-X" | "+X" => 0,
        "-Y" | "+Y" => 1,
        "-Z" | "+Z" => 2,
        _ => return None,
    };
    Some((index, if face.starts_with('-') { -1.0 } else { 1.0 }))
}

fn frame_axis(frame: &Frame, index: usize) -> Vec3 {
    match index {
        0 => frame.u,
        1 => frame.v,
        _ => frame.w,
    }
}

fn half_size_component(half: Vec3, index: usize) -> f64 {
    match index {
        0 => half.x,
        1 => half.y,
        _ => half.z,
    }
}

fn patch_face(patch: &BoundarySurfacePatch) -> &str {
    patch.patch_id.rsplit('.').next().unwrap_or("")
}

/// Analytic ray intersections with one patch (`_surface_patch_ray_points`).
fn surface_patch_ray_points(
    patch: &BoundarySurfacePatch,
    ray_origin: Vec3,
    ray_direction: Vec3,
) -> Vec<(f64, Vec3)> {
    match &patch.owner.shape {
        Shape::Box3(shape) => {
            let Some((axis_index, sign)) = box_face_axis(patch_face(patch)) else {
                return Vec::new();
            };
            let axis = frame_axis(&shape.frame, axis_index);
            let normal = axis * sign;
            let point_on_plane =
                shape.center + axis * (sign * half_size_component(shape.half_size, axis_index));
            let denominator = ray_direction.dot(normal);
            if denominator.abs() <= 1.0e-12 {
                return Vec::new();
            }
            let travel = (point_on_plane - ray_origin).dot(normal) / denominator;
            if travel < 0.0 {
                return Vec::new();
            }
            vec![(travel, ray_origin + ray_direction * travel)]
        }
        Shape::Cylinder(_) | Shape::Cone(_) | Shape::CappedCone(_) => {
            let owner = &patch.owner;
            let radius = cylinder_like_radius(owner).expect("radius");
            let half_height = cylinder_like_half_height(owner).expect("half height");
            let center = cylinder_like_center(owner).expect("center");
            let frame = cylinder_like_frame(owner).expect("frame");
            let origin_local = oriented_local(ray_origin, center, frame);
            let direction_local = vec3(
                ray_direction.dot(frame.u),
                ray_direction.dot(frame.v),
                ray_direction.dot(frame.w),
            );
            let face = patch_face(patch);
            if face == "side_wall" {
                let a = direction_local.x * direction_local.x
                    + direction_local.y * direction_local.y;
                if a <= 1.0e-12 {
                    return Vec::new();
                }
                let b = 2.0
                    * (origin_local.x * direction_local.x + origin_local.y * direction_local.y);
                let c = origin_local.x * origin_local.x + origin_local.y * origin_local.y
                    - radius * radius;
                let discriminant = b * b - 4.0 * a * c;
                if discriminant < 0.0 {
                    return Vec::new();
                }
                let root = discriminant.sqrt();
                let mut result = Vec::new();
                for travel in [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)] {
                    let z = origin_local.z + travel * direction_local.z;
                    if travel >= 0.0 && z.abs() <= half_height + PATCH_TOLERANCE {
                        result.push((travel, ray_origin + ray_direction * travel));
                    }
                }
                result.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
                return result;
            }
            let cap_sign = match face {
                "-Z_cap" => -1.0,
                "+Z_cap" => 1.0,
                _ => return Vec::new(),
            };
            if direction_local.z.abs() <= 1.0e-12 {
                return Vec::new();
            }
            let travel = (cap_sign * half_height - origin_local.z) / direction_local.z;
            if travel < 0.0 {
                return Vec::new();
            }
            let point_local = origin_local + direction_local * travel;
            if point_local.x * point_local.x + point_local.y * point_local.y
                > (radius + PATCH_TOLERANCE) * (radius + PATCH_TOLERANCE)
            {
                return Vec::new();
            }
            let point = center
                + frame.u * point_local.x
                + frame.v * point_local.y
                + frame.w * point_local.z;
            vec![(travel, point)]
        }
        Shape::Sphere(shape) => {
            let relative = ray_origin - shape.center;
            let b = 2.0 * relative.dot(ray_direction);
            let c = relative.dot(relative) - shape.radius * shape.radius;
            let discriminant = b * b - 4.0 * c;
            if discriminant < 0.0 {
                return Vec::new();
            }
            let root = discriminant.sqrt();
            let mut result = Vec::new();
            for travel in [(-b - root) * 0.5, (-b + root) * 0.5] {
                if travel >= 0.0 {
                    result.push((travel, ray_origin + ray_direction * travel));
                }
            }
            result.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
            result
        }
        _ => Vec::new(),
    }
}

/// `_surface_patch_contains`.
fn surface_patch_contains(patch: &BoundarySurfacePatch, point: Vec3, tolerance: f64) -> bool {
    match &patch.owner.shape {
        Shape::Box3(shape) => {
            let local = oriented_local(point, shape.center, &shape.frame);
            let Some((index, sign)) = box_face_axis(patch_face(patch)) else {
                return false;
            };
            let local_components = [local.x, local.y, local.z];
            let half_components = [
                shape.half_size.x,
                shape.half_size.y,
                shape.half_size.z,
            ];
            if (local_components[index] - sign * half_components[index]).abs()
                > (4.0 * tolerance).max(PATCH_TOLERANCE)
            {
                return false;
            }
            (0..3).filter(|axis| *axis != index).all(|axis| {
                local_components[axis].abs() <= half_components[axis] + PATCH_TOLERANCE
            })
        }
        Shape::Cylinder(_) | Shape::Cone(_) | Shape::CappedCone(_) => {
            let owner = &patch.owner;
            let center = cylinder_like_center(owner).expect("center");
            let frame = cylinder_like_frame(owner).expect("frame");
            let half_height = cylinder_like_half_height(owner).expect("half height");
            let local = oriented_local(point, center, frame);
            match patch_face(patch) {
                "side_wall" => (local.z.abs() - half_height).abs() > PATCH_TOLERANCE,
                "-Z_cap" => (local.z + half_height).abs() <= (4.0 * tolerance).max(PATCH_TOLERANCE),
                "+Z_cap" => (local.z - half_height).abs() <= (4.0 * tolerance).max(PATCH_TOLERANCE),
                _ => false,
            }
        }
        _ => {
            // Generic whole-surface patch: the point lies on the owner's own
            // zero set (also correct for cut surfaces).
            patch.owner.eval_point(point).abs() <= (4.0 * tolerance).max(PATCH_TOLERANCE)
        }
    }
}

/// `_surface_patch_normal`.
fn surface_patch_normal(patch: &BoundarySurfacePatch, point: Vec3) -> Option<Vec3> {
    let normal = match patch.normal {
        Some(normal) => normal,
        None => sdf_normal(&patch.owner, point, PATCH_TOLERANCE),
    };
    let length = normal.length();
    if length <= 1.0e-12 {
        return None;
    }
    Some(normal * (patch.normal_sign / length))
}

fn surface_patch_hit(
    root: &Node,
    patch: &BoundarySurfacePatch,
    point: Vec3,
    final_normal: Option<Vec3>,
    tolerance: f64,
) -> Option<BoundaryPatchHit> {
    if patch.owner.eval_point(point).abs() > (4.0 * tolerance).max(PATCH_TOLERANCE) {
        return None;
    }
    if !surface_patch_contains(patch, point, tolerance) {
        return None;
    }
    if root.eval_point(point).abs() > tolerance.max(PATCH_TOLERANCE) {
        return None;
    }
    let final_normal = final_normal.unwrap_or_else(|| sdf_normal(root, point, tolerance));
    let patch_normal = surface_patch_normal(patch, point)?;
    if normal_alignment(patch_normal, final_normal) < 0.82 {
        return None;
    }
    Some(BoundaryPatchHit {
        point,
        owner_object_id: patch.owner_object_id,
        patch_id: patch.patch_id.clone(),
        patch_type: patch.patch_type.clone(),
        normal: patch_normal,
        outside_direction: patch.outside_direction,
    })
}

/// Analytic-first boundary pick (`pick_boundary_patch`, 3D path). 2D picks
/// use the default scale-relative radius; viewports that know their pixel
/// size should call `pick_boundary_patch_with_radius`.
pub fn pick_boundary_patch(
    root: &Node,
    ray_origin: Vec3,
    ray_direction: Vec3,
    hit_tolerance: f64,
    maximum_travel: f64,
) -> Option<BoundaryPatchHit> {
    pick_boundary_patch_with_radius(
        root,
        ray_origin,
        ray_direction,
        hit_tolerance,
        maximum_travel,
        None,
    )
}

/// `pick_boundary_patch` with an explicit world-space pick radius for the
/// 2D curve pick (typically derived from a few screen pixels at the
/// workplane). Ignored on 3D roots, where the ray itself is the precision.
pub fn pick_boundary_patch_with_radius(
    root: &Node,
    ray_origin: Vec3,
    ray_direction: Vec3,
    hit_tolerance: f64,
    maximum_travel: f64,
    curve_pick_radius: Option<f64>,
) -> Option<BoundaryPatchHit> {
    let length = ray_direction.length();
    if length <= 1.0e-12 {
        return None;
    }
    let direction = ray_direction * (1.0 / length);
    if root.dimension() == 2 {
        // 2D path: the boundary is the outline curve on the domain
        // workplane — analytic ray ∩ plane, then snap to the nearest curve
        // patch (the curve analog of the analytic face pick below).
        return pick_curve_patch(root, ray_origin, direction, curve_pick_radius);
    }
    let patches = surface_patches_for_root(root);

    // Analytic candidates, nearest first; a cut surface wins outright.
    let mut ray_points: Vec<(f64, &BoundarySurfacePatch, Vec3)> = Vec::new();
    for patch in &patches {
        for (travel, point) in surface_patch_ray_points(patch, ray_origin, direction) {
            if travel >= 0.0 {
                ray_points.push((travel, patch, point));
            }
        }
    }
    ray_points.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
    let mut first_hit: Option<BoundaryPatchHit> = None;
    for (_travel, patch, point) in &ray_points {
        if let Some(hit) = surface_patch_hit(root, patch, *point, None, hit_tolerance) {
            if hit.patch_type == "cut_surface" {
                return Some(hit);
            }
            if first_hit.is_none() {
                first_hit = Some(hit);
            }
        }
    }
    if first_hit.is_some() {
        return first_hit;
    }

    // Sphere-trace fallback for generic leaves.
    let point = pick_sdf_surface(root, ray_origin, direction, hit_tolerance, maximum_travel)?;
    let normal = sdf_normal(root, point, hit_tolerance);
    patches
        .iter()
        .filter_map(|patch| surface_patch_hit(root, patch, point, Some(normal), hit_tolerance))
        .max_by(|a, b| {
            normal_alignment(a.normal, normal)
                .partial_cmp(&normal_alignment(b.normal, normal))
                .expect("finite")
        })
}

/// The placed profile of a legal 2D domain root (coplanar booleans merge
/// into one placed node; transforms are refused for non-3D objects, so the
/// root of a 2D domain is always the placed node itself).
pub fn placed_2d_root(root: &Node) -> Option<&PlacedSdf2D> {
    match &root.shape {
        Shape::PlacedSdf2D(placed) => Some(placed),
        _ => None,
    }
}

/// Ray ∩ the 2D domain's workplane, in world space.
fn workplane_hit(placed: &PlacedSdf2D, ray_origin: Vec3, direction: Vec3) -> Option<Vec3> {
    let normal = placed.normal();
    let denominator = direction.dot(normal);
    if denominator.abs() <= 1.0e-12 {
        return None;
    }
    let travel = (placed.origin - ray_origin).dot(normal) / denominator;
    if travel < 0.0 {
        return None;
    }
    Some(ray_origin + direction * travel)
}

/// Newton-project an in-plane point onto the zero set of `field` (a placed
/// 2D node: its gradient is in-plane, so the iteration stays on the plane).
fn snap_to_outline(field: &Node, start: Vec3, step: f64, zero_band: f64) -> Vec3 {
    let mut point = start;
    for _ in 0..12 {
        let value = field.eval_point(point);
        if value.abs() <= zero_band {
            break;
        }
        point = point - sdf_normal(field, point, step) * value;
    }
    point
}

/// The 2D boundary pick: nearest curve patch to the ray's workplane hit,
/// within `pick_radius` (default `CURVE_PATCH_PICK_TOLERANCE` of the domain
/// diagonal). The overall nearest patch wins; a cut surface (subtracted
/// operand's outline) is preferred only when its distance ties the nearest
/// regular patch within the surface band — the 2D reading of the 3D
/// "coincident cut surface wins" rule, without letting a far cut patch
/// steal the hover from a clearly closer edge.
fn pick_curve_patch(
    root: &Node,
    ray_origin: Vec3,
    direction: Vec3,
    pick_radius: Option<f64>,
) -> Option<BoundaryPatchHit> {
    let placed = placed_2d_root(root)?;
    let plane_point = workplane_hit(placed, ray_origin, direction)?;
    let diagonal = bounding_diagonal(root).max(1.0e-9);
    let pick_tolerance = pick_radius.unwrap_or(CURVE_PATCH_PICK_TOLERANCE * diagonal);
    let surface_tolerance = RELATIVE_SURFACE_TOLERANCE * diagonal;
    let step = (diagonal * 1.0e-5).max(1.0e-9);
    let mut best: Option<(f64, BoundaryPatchHit)> = None;
    let mut best_cut: Option<(f64, BoundaryPatchHit)> = None;
    for patch in surface_patches_for_root(root) {
        let Some(curve) = &patch.curve else {
            continue;
        };
        let snapped = match curve {
            CurvePatchKind::Edge { start, end, .. } => {
                let axis = *end - *start;
                let along = ((plane_point - *start).dot(axis)
                    / axis.dot(axis).max(1.0e-24))
                .clamp(0.0, 1.0);
                *start + axis * along
            }
            CurvePatchKind::Outline => {
                snap_to_outline(&patch.owner, plane_point, step, 1.0e-12 * diagonal)
            }
        };
        let distance = (plane_point - snapped).length();
        if distance > pick_tolerance {
            continue;
        }
        // The snapped point must lie on the operand's outline AND still be
        // part of the final domain boundary (not swallowed by another
        // operand — e.g. the part of an edge inside a subtracted hole).
        if patch.owner.eval_point(snapped).abs() > surface_tolerance {
            continue;
        }
        if root.eval_point(snapped).abs() > surface_tolerance {
            continue;
        }
        let hit = BoundaryPatchHit {
            point: snapped,
            owner_object_id: patch.owner_object_id,
            patch_id: patch.patch_id.clone(),
            patch_type: patch.patch_type.clone(),
            normal: sdf_normal(root, snapped, step),
            outside_direction: patch.outside_direction,
        };
        let slot = if patch.patch_type == "cut_surface" {
            &mut best_cut
        } else {
            &mut best
        };
        if slot.as_ref().map(|(d, _)| distance < *d).unwrap_or(true) {
            *slot = Some((distance, hit));
        }
    }
    match (best_cut, best) {
        (Some((cut_distance, cut_hit)), Some((distance, hit))) => {
            if cut_distance <= distance + surface_tolerance {
                Some(cut_hit)
            } else {
                Some(hit)
            }
        }
        (cut, regular) => cut.or(regular).map(|(_, hit)| hit),
    }
}

/// Cutter clicks on a 2D domain: ray ∩ workplane, snapped onto the domain
/// outline when the in-plane hit is within the snap radius of it (the
/// raw in-plane point otherwise, so knife endpoints stay forgiving).
pub fn pick_outline_point(root: &Node, ray_origin: Vec3, ray_direction: Vec3) -> Option<Vec3> {
    pick_outline_point_with_radius(root, ray_origin, ray_direction, None)
}

/// `pick_outline_point` with an explicit world-space snap radius (default
/// `OUTLINE_SNAP_TOLERANCE` of the domain diagonal).
pub fn pick_outline_point_with_radius(
    root: &Node,
    ray_origin: Vec3,
    ray_direction: Vec3,
    snap_radius: Option<f64>,
) -> Option<Vec3> {
    let placed = placed_2d_root(root)?;
    let length = ray_direction.length();
    if length <= 1.0e-12 {
        return None;
    }
    let plane_point = workplane_hit(placed, ray_origin, ray_direction * (1.0 / length))?;
    let diagonal = bounding_diagonal(root).max(1.0e-9);
    let step = (diagonal * 1.0e-5).max(1.0e-9);
    let snapped = snap_to_outline(root, plane_point, step, 1.0e-12 * diagonal);
    if (plane_point - snapped).length() <= snap_radius.unwrap_or(OUTLINE_SNAP_TOLERANCE * diagonal)
        && root.eval_point(snapped).abs() <= RELATIVE_SURFACE_TOLERANCE * diagonal
    {
        Some(snapped)
    } else {
        Some(plane_point)
    }
}

// ---------------------------------------------------------------------------
// selector volumes + patch scope (classification fields)
// ---------------------------------------------------------------------------

fn root_span(root: &Node) -> f64 {
    root.bounding_box()
        .map(|bounds| {
            (bounds.x_max - bounds.x_min)
                .max(bounds.y_max - bounds.y_min)
                .max(bounds.z_max - bounds.z_min)
                .max(1.0)
        })
        .unwrap_or(1.0)
}

fn orthonormal_completion(axis_u: Vec3) -> (Vec3, Vec3) {
    let mut reference = vec3(0.0, 0.0, 1.0);
    if axis_u.dot(reference).abs() > 0.9 {
        reference = vec3(0.0, 1.0, 0.0);
    }
    let mut axis_v = reference.cross(axis_u);
    axis_v = axis_v * (1.0 / axis_v.length().max(1.0e-12));
    let mut axis_w = axis_u.cross(axis_v);
    axis_w = axis_w * (1.0 / axis_w.length().max(1.0e-12));
    (axis_v, axis_w)
}

/// The 3D SDF field used to classify boundary subregions
/// (`surface_selector_volume`, without region scoping).
pub fn surface_selector_volume(root: &Node, selector: &Node) -> GeometryResult<Option<Node>> {
    if selector.dimension() == 3 {
        return Ok(Some(selector.clone()));
    }
    let span = root_span(root);
    match &selector.shape {
        Shape::PlacedSdf2D(placed) => {
            let mut section = placed.clone();
            section.sources = Vec::new();
            let section_node = Node {
                name: format!("{}_selector_section", selector.name),
                object_id: 0,
                shape: Shape::PlacedSdf2D(section),
            };
            Ok(Some(Node {
                name: format!("{}_extruded_selector", selector.name),
                object_id: 0,
                shape: Shape::Extrude(Extrude::new(section_node, span * 4.0, 0.0)?),
            }))
        }
        Shape::PlacedPolyline1D(placed) => {
            let band_profile = Profile2D::distance_offset(
                placed.profile.clone(),
                PATCH_TOLERANCE.max(PATCH_TOLERANCE),
            )?;
            let section = PlacedSdf2D::new(
                band_profile,
                placed.origin,
                placed.axis_u,
                placed.axis_v,
                Vec::new(),
            )?;
            let section_node = Node {
                name: format!("{}_selector_section", selector.name),
                object_id: 0,
                shape: Shape::PlacedSdf2D(section),
            };
            Ok(Some(Node {
                name: format!("{}_extruded_selector", selector.name),
                object_id: 0,
                shape: Shape::Extrude(Extrude::new(section_node, span * 4.0, 0.0)?),
            }))
        }
        Shape::PlacedSdf1D(placed) => {
            let Profile1D::Segment {
                center: profile_center,
                half_length,
            } = placed.profile
            else {
                return Ok(None);
            };
            let lower = profile_center - half_length;
            let upper = profile_center + half_length;
            let half = (0.5 * (upper - lower)).max(PATCH_TOLERANCE);
            let center_offset = 0.5 * (upper + lower);
            let mut axis_u = placed.axis_u;
            axis_u = axis_u * (1.0 / axis_u.length().max(1.0e-12));
            let (axis_v, axis_w) = orthonormal_completion(axis_u);
            let center = placed.origin + axis_u * center_offset;
            Ok(Some(Node {
                name: format!("{}_extruded_selector", selector.name),
                object_id: 0,
                shape: Shape::Box3(Box3::new(
                    center,
                    vec3(half, span * 2.0, span * 2.0),
                    Frame {
                        u: axis_u,
                        v: axis_v,
                        w: axis_w,
                    },
                )?),
            }))
        }
        _ => Ok(None),
    }
}

/// The analytic patch a region tags, when it still resolves on this root.
fn region_patch(root: &Node, region: &BoundaryRegion) -> Option<BoundarySurfacePatch> {
    let patch_id_value = region.patch_id.as_deref()?;
    surface_patches_for_root(root).into_iter().find(|patch| {
        patch.owner_object_id == region.owner_object_id && patch.patch_id == patch_id_value
    })
}

/// `_region_patch_scope_volume` — the thin volume that limits a region to
/// its analytic patch.
pub fn region_patch_scope_volume(
    root: &Node,
    region: &BoundaryRegion,
    thickness: f64,
) -> Option<Node> {
    let patch = region_patch(root, region)?;
    if let Some(curve) = patch.curve.clone() {
        return curve_patch_scope_volume(root, &patch, &curve, thickness, 0.0);
    }
    surface_patch_scope_volume(&patch, thickness)
}

/// Scope volume of a 3D (non-curve) patch: the box-face slab, else the
/// generic preview volume.
fn surface_patch_scope_volume(patch: &BoundarySurfacePatch, thickness: f64) -> Option<Node> {
    let face = patch_face(patch).to_string();
    match &patch.owner.shape {
        Shape::Box3(shape) => {
            if let Some((index, sign)) = box_face_axis(&face) {
                let axis = frame_axis(&shape.frame, index);
                let mut half = shape.half_size;
                match index {
                    0 => half.x = thickness,
                    1 => half.y = thickness,
                    _ => half.z = thickness,
                }
                let center =
                    shape.center + axis * (sign * half_size_component(shape.half_size, index));
                return Some(Node {
                    name: format!("{}_{}_scope", patch.owner.name, patch.patch_id),
                    object_id: 0,
                    shape: Shape::Box3(Box3::new(center, half, shape.frame).ok()?),
                });
            }
            patch_preview_volume(patch, thickness)
        }
        _ => patch_preview_volume(patch, thickness),
    }
}

/// Scope volume for a curve patch: a thin prism limiting a region to its
/// edge or operand outline — the 2D analog of the box-face scope slab.
/// Every classification field of a 2D domain is a prism along the workplane
/// normal (placed fields ignore the plane offset), so the scope is one too.
///
/// `lateral` is the half-width across the curve; `tangential_pad` is the
/// slack past an edge's corners. They differ on purpose: the lateral band
/// must absorb criterion 1's on-boundary tolerance, while the tangential
/// end must stop AT the corner so an edge region never claims points on the
/// neighbor edge. Callers evaluate the result against `<= 0.0` — the slack
/// is baked into the volume, never into the eval limit (an eval limit pads
/// isotropically, which is exactly the corner bleed this shape avoids).
fn curve_patch_scope_volume(
    root: &Node,
    patch: &BoundarySurfacePatch,
    curve: &CurvePatchKind,
    lateral: f64,
    tangential_pad: f64,
) -> Option<Node> {
    let span = root_span(root);
    match curve {
        CurvePatchKind::Edge {
            start,
            end,
            outward,
        } => {
            let axis = *end - *start;
            let length = axis.length();
            if length <= 1.0e-12 {
                return None;
            }
            let axis_u = axis * (1.0 / length);
            let mut axis_w = axis_u.cross(*outward);
            axis_w = axis_w * (1.0 / axis_w.length().max(1.0e-12));
            let center = (*start + *end) * 0.5;
            Some(Node {
                name: format!("{}_{}_scope", patch.owner.name, patch.patch_id),
                object_id: 0,
                shape: Shape::Box3(
                    Box3::new(
                        center,
                        vec3(length * 0.5 + tangential_pad, lateral, span * 2.0),
                        Frame {
                            u: axis_u,
                            v: *outward,
                            w: axis_w,
                        },
                    )
                    .ok()?,
                ),
            })
        }
        CurvePatchKind::Outline => {
            let Shape::PlacedSdf2D(placed) = &patch.owner.shape else {
                return None;
            };
            let band = |offset: f64| -> Option<Node> {
                let profile =
                    Profile2D::distance_offset(placed.profile.clone(), offset).ok()?;
                let section =
                    PlacedSdf2D::new(profile, placed.origin, placed.axis_u, placed.axis_v, Vec::new())
                        .ok()?;
                Some(Node {
                    name: format!("{}_{}_band", patch.owner.name, patch.patch_id),
                    object_id: 0,
                    shape: Shape::PlacedSdf2D(section),
                })
            };
            // Difference(grown, shrunk) evaluates to |operand field| −
            // lateral: the band around the operand's outline (a closed
            // curve has no corners, so `tangential_pad` does not apply).
            let grown = band(lateral)?;
            let shrunk = band(-lateral)?;
            Shape::difference(grown, shrunk).ok().map(|shape| {
                Node::new(
                    format!("{}_{}_scope", patch.owner.name, patch.patch_id),
                    shape,
                )
            })
        }
    }
}

/// Preview/highlight volume for a patch (`_*_patch_preview_node` subset) —
/// also the scope volume for non-box owners.
fn patch_preview_volume(patch: &BoundarySurfacePatch, thickness: f64) -> Option<Node> {
    let normal = patch.normal.unwrap_or(vec3(0.0, 0.0, 0.0)) * patch.normal_sign;
    match &patch.owner.shape {
        Shape::Box3(shape) => {
            let (index, sign) = box_face_axis(patch_face(patch))?;
            let axis = frame_axis(&shape.frame, index);
            let mut half = shape.half_size;
            match index {
                0 => half.x = thickness,
                1 => half.y = thickness,
                _ => half.z = thickness,
            }
            let center = shape.center
                + axis * (sign * half_size_component(shape.half_size, index))
                + normal * (thickness * 2.0);
            Some(Node {
                name: format!("{}_{}_highlight", patch.owner.name, patch.patch_id),
                object_id: 0,
                shape: Shape::Box3(Box3::new(center, half, shape.frame).ok()?),
            })
        }
        Shape::Cylinder(_) | Shape::Cone(_) | Shape::CappedCone(_) => {
            let owner = &patch.owner;
            let radius = cylinder_like_radius(owner)?;
            let half_height = cylinder_like_half_height(owner)?;
            let center = cylinder_like_center(owner)?;
            let frame = *cylinder_like_frame(owner)?;
            match patch_face(patch) {
                "-Z_cap" | "+Z_cap" => {
                    let sign = if patch_face(patch) == "-Z_cap" { -1.0 } else { 1.0 };
                    let cap_center =
                        center + frame.w * (sign * half_height) + normal * (thickness * 2.0);
                    Some(Node {
                        name: format!("{}_{}_highlight", owner.name, patch.patch_id),
                        object_id: 0,
                        shape: Shape::Cylinder(
                            Cylinder::new(cap_center, radius, thickness, frame).ok()?,
                        ),
                    })
                }
                "side_wall" => {
                    let outer = Node {
                        name: format!("{}_{}_highlight_outer", owner.name, patch.patch_id),
                        object_id: 0,
                        shape: Shape::Cylinder(
                            Cylinder::new(center, radius + thickness, half_height, frame).ok()?,
                        ),
                    };
                    let inner = Node {
                        name: format!("{}_{}_highlight_inner", owner.name, patch.patch_id),
                        object_id: 0,
                        shape: Shape::Cylinder(
                            Cylinder::new(
                                center,
                                (radius - thickness).max(thickness),
                                half_height + thickness * 2.0,
                                frame,
                            )
                            .ok()?,
                        ),
                    };
                    Shape::difference(outer, inner).ok().map(|shape| {
                        Node::new(
                            format!("{}_{}_highlight", owner.name, patch.patch_id),
                            shape,
                        )
                    })
                }
                _ => None,
            }
        }
        Shape::Sphere(shape) => {
            let outer = Node {
                name: format!("{}_{}_highlight_outer", patch.owner.name, patch.patch_id),
                object_id: 0,
                shape: Shape::Sphere(Sphere::new(shape.center, shape.radius + thickness).ok()?),
            };
            let inner = Node {
                name: format!("{}_{}_highlight_inner", patch.owner.name, patch.patch_id),
                object_id: 0,
                shape: Shape::Sphere(
                    Sphere::new(shape.center, (shape.radius - thickness).max(thickness)).ok()?,
                ),
            };
            Shape::difference(outer, inner).ok().map(|shape| {
                Node::new(
                    format!("{}_{}_highlight", patch.owner.name, patch.patch_id),
                    shape,
                )
            })
        }
        _ => None,
    }
}

/// `boundary_region_scope_mask`: which points lie within the region's
/// analytic patch scope. Curve patches (2D domains) evaluate an exactly
/// padded volume against zero — the lateral band still absorbs the
/// on-boundary tolerance, but past an edge's corners only a hairline of
/// slack remains, so adjacent edge regions meet at the corner instead of
/// overlapping by a scale-relative band.
pub fn boundary_region_scope_mask(
    root: &Node,
    region: &BoundaryRegion,
    points: &[Vec3],
    tolerance: f64,
) -> GeometryResult<Vec<bool>> {
    let thickness = (PATCH_TOLERANCE * 4.0).max(0.006);
    let Some(patch) = region_patch(root, region) else {
        return Ok(vec![true; points.len()]);
    };
    if let Some(curve) = patch.curve.clone() {
        // Straight edges: membership points (outline rings, mesh boundary
        // nodes) lie ON the line, so the lateral band stays absolute —
        // widening it by the scale-relative tolerance is exactly what let
        // an edge region claim a stretch of the neighbor edge past the
        // corner. Whole outlines have no corners to bleed past; their band
        // keeps the tolerance slack for faceted/curved robustness.
        let lateral = match curve {
            CurvePatchKind::Edge { .. } => thickness,
            CurvePatchKind::Outline => thickness + tolerance.max(PATCH_TOLERANCE),
        };
        let Some(scope) = curve_patch_scope_volume(root, &patch, &curve, lateral, thickness)
        else {
            return Ok(vec![true; points.len()]);
        };
        return Ok(points
            .iter()
            .map(|point| scope.eval_point(*point) <= 0.0)
            .collect());
    }
    let Some(scope) = surface_patch_scope_volume(&patch, thickness) else {
        return Ok(vec![true; points.len()]);
    };
    let limit = tolerance.max(PATCH_TOLERANCE);
    Ok(points
        .iter()
        .map(|point| scope.eval_point(*point) <= limit)
        .collect())
}
