//! Exactness preconditions for generators and offsets (spec §5, §6),
//! ported from `core/preconditions.py`.
//!
//! Most exact operations are unconditional, but a few are exact only under a
//! geometric precondition. Sampled validators for the two well-defined ones:
//!
//! * **Revolve** is exact when the section profile stays on one side of the
//!   revolution axis, or is mirror-symmetric about it.
//! * **Erosion** (a negative distance offset) is exact only while the radius
//!   stays below the shape's reach; the necessary check enforced here is that
//!   it must not reach the shape's maximum inscribed depth.
//!
//! Sweep/tube self-overlap is deferred, as in the Python kernel. These checks
//! sample the field, so they belong to the expensive `compile_model` gate,
//! not the live grammar diagnostics.

use crate::sdf::node::{Node, Shape};
use crate::sdf::primitives_2d::Profile2D;
use crate::sdf::solid_from_2d::Revolve;

const REVOLVE_RESOLUTION: usize = 48;
const EROSION_RESOLUTION: usize = 64;
const TOL: f64 = 1.0e-9;
const SYMMETRY_TOL: f64 = 2.0e-4;

fn linspace(minimum: f64, maximum: f64, count: usize) -> Vec<f64> {
    if count == 1 {
        return vec![minimum];
    }
    let step = (maximum - minimum) / (count - 1) as f64;
    (0..count).map(|index| minimum + step * index as f64).collect()
}

/// Violation when the revolve's interior profile straddles its axis without
/// mirror symmetry.
pub fn revolve_violations(name: &str, revolve: &Revolve) -> Vec<String> {
    let resolution = REVOLVE_RESOLUTION;
    let section = revolve.section2d();
    let frame = revolve.axis_frame().expect("validated at construction");
    let (u_min, u_max, v_min, v_max) = section.profile.bounds();
    let us = linspace(u_min, u_max, resolution);
    let vs = linspace(v_min, v_max, resolution);
    let base = (section.origin - frame.origin).dot(frame.radial);
    let du = section.axis_u.dot(frame.radial);
    let dv = section.axis_v.dot(frame.radial);
    let mut r_min = f64::INFINITY;
    let mut r_max = f64::NEG_INFINITY;
    let mut any_interior = false;
    for u in &us {
        for v in &vs {
            if section.profile.eval(*u, *v) < 0.0 {
                any_interior = true;
                let radial_coord = base + u * du + v * dv;
                r_min = r_min.min(radial_coord);
                r_max = r_max.max(radial_coord);
            }
        }
    }
    if !any_interior || r_min >= -TOL || r_max <= TOL {
        return Vec::new();
    }
    if profile_is_mirror_symmetric_about_revolve_axis(revolve, resolution) {
        return Vec::new();
    }
    vec![format!(
        "Revolve '{name}': non-symmetric profile crosses the revolution axis \
         (radial coord spans {r_min:.3}..{r_max:.3}); a revolve is exact only \
         when the profile stays on one side of the axis or is mirror-symmetric \
         about it (§5)"
    )]
}

fn profile_is_mirror_symmetric_about_revolve_axis(revolve: &Revolve, resolution: usize) -> bool {
    let section = revolve.section2d();
    let frame = revolve.axis_frame().expect("validated at construction");
    let (u_min, u_max, v_min, v_max) = section.profile.bounds();
    let us = linspace(u_min, u_max, resolution);
    let vs = linspace(v_min, v_max, resolution);
    let span = (u_max - u_min).abs().max((v_max - v_min).abs()).max(1.0);
    let near_band = span / (resolution.max(2) - 1) as f64 * 2.0;
    let tolerance = SYMMETRY_TOL.max(span * 1.0e-4);
    let mut any_relevant = false;
    let mut worst: f64 = 0.0;
    for u in &us {
        for v in &vs {
            let world = section.origin + section.axis_u * *u + section.axis_v * *v;
            let radial_coord = (world - frame.origin).dot(frame.radial);
            let mirrored = world - frame.radial * (2.0 * radial_coord);
            let mirrored_delta = mirrored - section.origin;
            let mirrored_u = mirrored_delta.dot(section.axis_u);
            let mirrored_v = mirrored_delta.dot(section.axis_v);
            let profile = section.profile.eval(*u, *v);
            let mirrored_profile = section.profile.eval(mirrored_u, mirrored_v);
            let relevant = profile <= near_band
                || mirrored_profile <= near_band
                || profile.abs() <= near_band
                || mirrored_profile.abs() <= near_band;
            if relevant {
                any_relevant = true;
                worst = worst.max((profile - mirrored_profile).abs());
            }
        }
    }
    any_relevant && worst <= tolerance
}

/// Violation when a negative distance offset erodes past the child shape's
/// maximum inscribed depth (necessary `r < reach` condition, §6).
pub fn erosion_violations(child: &Profile2D, offset: f64) -> Vec<String> {
    if offset >= 0.0 {
        return Vec::new(); // dilation: unconditional (§6)
    }
    let resolution = EROSION_RESOLUTION;
    let (u_min, u_max, v_min, v_max) = child.bounds();
    let us = linspace(u_min, u_max, resolution);
    let vs = linspace(v_min, v_max, resolution);
    let mut min_child = f64::INFINITY;
    for u in &us {
        for v in &vs {
            min_child = min_child.min(child.eval(*u, *v));
        }
    }
    if offset <= min_child + TOL {
        return vec![format!(
            "DistanceOffsetProfile: erosion {offset:.3} reaches/exceeds the \
             shape's max inscribed depth ({:.3}); the eroded shape vanishes or \
             its field is no longer exact (§6, r < reach). Necessary check only \
             -- true reach can be stricter at concave features.",
            -min_child
        )]
    }
    Vec::new()
}

fn walk_profile_offsets(profile: &Profile2D, violations: &mut Vec<String>) {
    if let Profile2D::DistanceOffset { child, offset } = profile {
        violations.extend(erosion_violations(child, *offset));
    }
    match profile {
        Profile2D::Offset { child, .. } | Profile2D::DistanceOffset { child, .. } => {
            walk_profile_offsets(child, violations);
        }
        Profile2D::Binary { left, right, .. } => {
            walk_profile_offsets(left, violations);
            walk_profile_offsets(right, violations);
        }
        _ => {}
    }
}

/// Aggregate all generator/offset precondition violations in a region tree.
pub fn precondition_violations(region: &Node) -> Vec<String> {
    let mut violations = Vec::new();
    for node in region.walk() {
        match &node.shape {
            Shape::Revolve(revolve) => {
                violations.extend(revolve_violations(&node.name, revolve));
            }
            Shape::PlacedSdf2D(placed) => {
                walk_profile_offsets(&placed.profile, &mut violations);
            }
            Shape::PlacedPolyline1D(placed) => {
                walk_profile_offsets(&placed.profile, &mut violations);
            }
            _ => {}
        }
    }
    violations
}
