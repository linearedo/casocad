//! Regression: concave planar outlines must be filled by ear clipping, not
//! the center fan. A fan from the vertex mean is only valid for star-shaped
//! outlines — on a concave polygon its triangles spill outside the outline
//! and overlap ("self-intersecting" overdraw). Total mesh area against the
//! exact value catches any overdraw, which can only inflate it.

use std::f64::consts::PI;

use caso_kernel::sdf::node::{Node, Shape};
use caso_kernel::sdf::placed::PlacedSdf2D;
use caso_kernel::sdf::primitives_2d::Profile2D;
use caso_kernel::sdf::solid_from_2d::{Extrude, Revolve, RevolveAxis};
use caso_kernel::vec3::{vec3, Vec3};
use caso_surfaces::{build_viewport_surface, ViewportSurface, ViewportSurfaceKey};

/// Concave L-shape, CCW: unit square minus its top-right 0.6 x 0.6 corner.
/// Area 0.64, perimeter 4.0.
fn l_shape(u_offset: f64) -> Profile2D {
    Profile2D::polygon(vec![
        [u_offset, 0.0],
        [u_offset + 1.0, 0.0],
        [u_offset + 1.0, 0.4],
        [u_offset + 0.4, 0.4],
        [u_offset + 0.4, 1.0],
        [u_offset, 1.0],
    ])
    .expect("L-shape polygon")
}

fn section(profile: Profile2D) -> Node {
    Node::with_id(
        "section",
        3,
        Shape::PlacedSdf2D(
            PlacedSdf2D::new(
                profile,
                Vec3::ZERO,
                vec3(1.0, 0.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                Vec::new(),
            )
            .expect("section"),
        ),
    )
}

fn build(node: &Node) -> ViewportSurface {
    build_viewport_surface(
        node,
        ViewportSurfaceKey {
            object_id: node.object_id,
            scene_revision: 1,
            resolution: 24,
        },
    )
}

fn mesh_area(surface: &ViewportSurface) -> f64 {
    let vertex = |index: u32| {
        let v = surface.vertices[index as usize];
        vec3(v[0] as f64, v[1] as f64, v[2] as f64)
    };
    let mut area = 0.0;
    for triangle in surface.indices.chunks_exact(3) {
        let (a, b, c) = (vertex(triangle[0]), vertex(triangle[1]), vertex(triangle[2]));
        area += 0.5 * (b - a).cross(c - a).length();
    }
    area
}

#[test]
fn concave_section_fill_matches_polygon_area() {
    let surface = build(&section(l_shape(0.0)));
    assert!(!surface.indices.is_empty());
    let area = mesh_area(&surface);
    // f32 vertex storage leaves ~1e-7 relative noise; fan overdraw adds
    // whole triangles' worth of area.
    assert!(
        (area - 0.64).abs() < 1.0e-5,
        "L-shape fill area {area:.6} must equal 0.64 (fan overdraw inflates it)"
    );
}

#[test]
fn concave_extrude_caps_match_prism_area() {
    let node = Node::with_id(
        "prism",
        7,
        Shape::Extrude(Extrude::new(section(l_shape(0.0)), 1.0, 0.0).expect("extrude")),
    );
    let surface = build(&node);
    assert!(!surface.indices.is_empty());
    // Two exact caps plus the side walls: 2 * 0.64 + 4.0 * 1.0.
    let expected = 5.28;
    let area = mesh_area(&surface);
    assert!(
        (area - expected).abs() < 1.0e-4,
        "prism area {area:.6} must equal {expected} (fan caps overdraw a concave profile)"
    );
}

#[test]
fn concave_partial_revolve_caps_match_profile_area() {
    // L-shape at u in [0.5, 1.5] revolved 180 degrees about the V axis.
    let node = Node::with_id(
        "half_revolve",
        9,
        Shape::Revolve(
            Revolve::new(section(l_shape(0.5)), RevolveAxis::V, None, None, None, 180.0)
                .expect("revolve"),
        ),
    );
    let surface = build(&node);
    assert!(!surface.indices.is_empty());
    // Pappus per outline edge (angle * centroid radius * length) for the
    // lateral bands, plus the two flat caps. The swept bands are chordal, so
    // the mesh area sits slightly below the analytic value; fan-cap overdraw
    // on the concave caps would push it well above.
    let lateral = PI * (1.0 + 0.6 + 0.72 + 0.54 + 0.28 + 0.5);
    let expected = lateral + 2.0 * 0.64;
    let area = mesh_area(&surface);
    assert!(
        (area - expected).abs() / expected < 0.01,
        "half-revolve area {area:.6} should match {expected:.6} within 1%"
    );
}
