//! Patch-exact outline arcs for 2D boundary regions.
//!
//! The highlight/selection overlay for a curve patch (an edge or operand
//! outline of a 2D domain) must follow the patch's own analytic geometry —
//! polygon corners exact, bezier spans sampled on the true curve — clipped
//! exactly where the boolean swallows it. Resampling the merged field with
//! marching squares (the display-fill path) chamfers corners and cannot
//! place arc ends precisely; this module builds the polyline from the
//! operand instead and root-finds every membership transition.

use caso_kernel::boundary_ops::{BoundarySurfacePatch, CurvePatchKind};
use caso_kernel::sdf::node::{Node, Shape};
use caso_kernel::sdf::primitives_2d::Point2;
use caso_kernel::vec3::Vec3;

use crate::profiles2d::{profile_contour_rings, profile_outline};

/// Open polylines (arcs) lying exactly on the patch curve, clipped to the
/// parts that survive on the final domain boundary. Closed arcs (nothing
/// clipped) come back with the first point repeated at the end, so callers
/// always draw consecutive pairs with no wraparound.
pub fn curve_patch_arcs(
    root: &Node,
    patch: &BoundarySurfacePatch,
    resolution: usize,
) -> Vec<Vec<Vec3>> {
    let Some(curve) = &patch.curve else {
        return Vec::new();
    };
    let diagonal = root
        .bounding_box()
        .map(|bounds| bounds.diagonal())
        .unwrap_or(1.0)
        .max(1.0e-9);
    let chord = diagonal / resolution.max(16) as f64;
    let sources: Vec<(Vec<Vec3>, bool)> = match curve {
        CurvePatchKind::Edge { start, end, .. } => {
            vec![(sample_segment(*start, *end, chord), false)]
        }
        CurvePatchKind::Outline => outline_rings(patch, resolution, chord)
            .into_iter()
            .map(|ring| (ring, true))
            .collect(),
    };
    let band = diagonal * 1.0e-9;
    let mut arcs = Vec::new();
    for (points, closed) in sources {
        clip_to_boundary(root, &patch.owner, &points, closed, band, &mut arcs);
    }
    arcs
}

/// Uniform samples on the exact segment, endpoints included.
fn sample_segment(start: Vec3, end: Vec3, chord: f64) -> Vec<Vec3> {
    let length = (end - start).length();
    let steps = ((length / chord).ceil() as usize).max(1);
    (0..=steps)
        .map(|index| start + (end - start) * (index as f64 / steps as f64))
        .collect()
}

/// World-space rings of the operand's own outline: analytic when the
/// profile supports it (exact corners, on-curve bezier samples), contour
/// rings of the OPERAND field otherwise. Straight runs are subdivided to
/// `chord` so membership transitions inside them are caught; a run is only
/// subdivided when its midpoint stays on the operand outline (curved
/// outlines are already densely sampled and chord midpoints leave the
/// curve).
fn outline_rings(patch: &BoundarySurfacePatch, resolution: usize, chord: f64) -> Vec<Vec<Vec3>> {
    let Shape::PlacedSdf2D(placed) = &patch.owner.shape else {
        return Vec::new();
    };
    let in_plane =
        |point: Point2| placed.origin + placed.axis_u * point[0] + placed.axis_v * point[1];
    let analytic = profile_outline(&placed.profile, resolution.max(16) as u32);
    let rings: Vec<Vec<Point2>> = if analytic.is_empty() {
        profile_contour_rings(&placed.profile, resolution.clamp(16, 512))
    } else {
        vec![analytic]
    };
    rings
        .into_iter()
        .filter(|ring| ring.len() >= 3)
        .map(|ring| {
            let mut world = Vec::with_capacity(ring.len());
            for index in 0..ring.len() {
                let a = in_plane(ring[index]);
                let b = in_plane(ring[(index + 1) % ring.len()]);
                world.push(a);
                let length = (b - a).length();
                if length <= chord {
                    continue;
                }
                let mid = (a + b) * 0.5;
                if placed.eval_point(mid).abs() > chord * 1.0e-6 {
                    continue; // curved span, already densely sampled
                }
                let steps = (length / chord).ceil() as usize;
                for step in 1..steps {
                    world.push(a + (b - a) * (step as f64 / steps as f64));
                }
            }
            world
        })
        .collect()
}

/// Split one source polyline into the arcs that lie on the final domain
/// boundary. Membership compares the merged field against the operand's
/// own: a surviving point has `|root| = |operand|` (the operand controls),
/// a swallowed one has `|root|` equal to its depth inside the other
/// operand. Testing `|root| <= |operand| + band` instead of `|root| <=
/// band` matters on curved outlines, where a chord's interior points sit
/// off the zero set by the sagitta — both fields carry that offset, so it
/// cancels, and the transition still happens exactly at the swallowing
/// operand's boundary (on straight edges the operand term is zero and the
/// test is exact outright). Transitions are bisected on the segment
/// parameter so arc ends land on the junction regardless of sampling
/// density.
fn clip_to_boundary(
    root: &Node,
    operand: &Node,
    points: &[Vec3],
    closed: bool,
    band: f64,
    arcs: &mut Vec<Vec<Vec3>>,
) {
    if points.len() < 2 {
        return;
    }
    let member = |p: Vec3| root.eval_point(p).abs() <= operand.eval_point(p).abs() + band;
    let mask: Vec<bool> = points.iter().map(|p| member(*p)).collect();
    // Closed rings start the walk on a non-member vertex so no arc is split
    // across the seam; a fully-member ring closes onto itself.
    let (order, wrap): (Vec<usize>, bool) = if closed {
        match mask.iter().position(|m| !*m) {
            Some(start) => (
                (0..=points.len()).map(|i| (start + i) % points.len()).collect(),
                false,
            ),
            None => ((0..points.len()).collect(), true),
        }
    } else {
        ((0..points.len()).collect(), false)
    };
    let mut arc: Vec<Vec3> = Vec::new();
    for pair in 0..order.len() - if wrap { 0 } else { 1 } {
        let i = order[pair];
        let j = order[(pair + 1) % order.len()];
        let (p, q) = (points[i], points[j]);
        match (mask[i], mask[j]) {
            (true, true) => {
                if arc.is_empty() {
                    arc.push(p);
                }
                arc.push(q);
            }
            (true, false) => {
                if arc.is_empty() {
                    arc.push(p);
                }
                arc.push(bisect_membership(root, operand, p, q, band));
                arcs.push(std::mem::take(&mut arc));
            }
            (false, true) => {
                arc.push(bisect_membership(root, operand, p, q, band));
                arc.push(q);
            }
            (false, false) => {}
        }
    }
    if wrap && !arc.is_empty() {
        // fully-member closed ring: explicit closure, drop the duplicated
        // wrap vertex the loop already appended.
        arc.pop();
        arc.push(arc[0]);
    }
    if arc.len() >= 2 {
        arcs.push(arc);
    }
}

/// Parameter bisection of the membership transition on chord (p, q). Both
/// endpoints of a shared junction run this identical deterministic routine,
/// so adjacent arcs meet in a bitwise-identical vertex.
fn bisect_membership(root: &Node, operand: &Node, p: Vec3, q: Vec3, band: f64) -> Vec3 {
    let member =
        |point: Vec3| root.eval_point(point).abs() <= operand.eval_point(point).abs() + band;
    let start_member = member(p);
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        if member(p + (q - p) * mid) == start_member {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    p + (q - p) * (0.5 * (lo + hi))
}
