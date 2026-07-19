//! 2D boolean display regressions: a subtracted shape must show as a real
//! hole in the filled surface, even when it is tiny relative to the outer
//! profile (the coarse uniform grid used to miss it entirely, and any
//! detected hole used to bail to the sampled staircase fill).

use std::f64::consts::PI;

use caso_kernel::sdf::node::{Node, Shape};
use caso_kernel::sdf::placed::PlacedSdf2D;
use caso_kernel::sdf::primitives_1d::BooleanOp1D;
use caso_kernel::sdf::primitives_2d::Profile2D;
use caso_kernel::vec3::{vec3, Vec3};
use caso_surfaces::{build_viewport_surface, ViewportSurface, ViewportSurfaceKey};

fn placed(profile: Profile2D) -> Node {
    Node::with_id(
        "boolean_2d",
        11,
        Shape::PlacedSdf2D(
            PlacedSdf2D::new(
                profile,
                Vec3::ZERO,
                vec3(1.0, 0.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                Vec::new(),
            )
            .expect("placed"),
        ),
    )
}

fn build(node: &Node) -> ViewportSurface {
    build_viewport_surface(
        node,
        ViewportSurfaceKey {
            object_id: node.object_id,
            scene_revision: 1,
            resolution: 12,
        },
    )
}

fn binary(left: Profile2D, right: Profile2D, operation: BooleanOp1D) -> Profile2D {
    Profile2D::Binary {
        left: Box::new(left),
        right: Box::new(right),
        operation,
        smoothing: 0.1,
    }
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

/// True when any fill triangle covers (x, y) on the Z = 0 plane.
fn covers(surface: &ViewportSurface, x: f64, y: f64) -> bool {
    let cross = |ax: f64, ay: f64, bx: f64, by: f64| ax * by - ay * bx;
    surface.indices.chunks_exact(3).any(|triangle| {
        let p = |index: u32| {
            let v = surface.vertices[index as usize];
            (v[0] as f64, v[1] as f64)
        };
        let (a, b, c) = (p(triangle[0]), p(triangle[1]), p(triangle[2]));
        let d0 = cross(b.0 - a.0, b.1 - a.1, x - a.0, y - a.1);
        let d1 = cross(c.0 - b.0, c.1 - b.1, x - b.0, y - b.1);
        let d2 = cross(a.0 - c.0, a.1 - c.1, x - c.0, y - c.1);
        (d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0) || (d0 <= 0.0 && d1 <= 0.0 && d2 <= 0.0)
    })
}

fn contoured(surface: &ViewportSurface) {
    assert!(
        surface.message.contains("contoured"),
        "expected the contoured fill path, got: {}",
        surface.message
    );
}

#[test]
fn tiny_hole_in_huge_surface_is_visible() {
    // 200 x 200 rectangle minus an r = 0.5 circle at its center — the
    // shape `combine` produces (second operand wrapped in Offset).
    let rect = Profile2D::rectangle([0.0, 0.0], [100.0, 100.0]).expect("rect");
    let hole = Profile2D::circle([0.0, 0.0], 0.5).expect("circle");
    let offset = Profile2D::Offset {
        child: Box::new(hole),
        offset: [0.0, 0.0],
    };
    let surface = build(&placed(binary(rect, offset, BooleanOp1D::Difference)));
    contoured(&surface);
    assert!(!covers(&surface, 0.0, 0.0), "hole center must stay open");
    assert!(covers(&surface, 50.0, 50.0), "surface body must be filled");
    let expected = 200.0 * 200.0 - PI * 0.25;
    let area = mesh_area(&surface);
    assert!(
        (area - expected).abs() / expected < 1.0e-3,
        "area {area:.3} must be within 0.1% of {expected:.3}"
    );
}

#[test]
fn off_center_tiny_hole_is_visible() {
    let rect = Profile2D::rectangle([0.0, 0.0], [100.0, 100.0]).expect("rect");
    let hole = Profile2D::circle([0.0, 0.0], 0.5).expect("circle");
    let offset = Profile2D::Offset {
        child: Box::new(hole),
        offset: [70.0, -55.0],
    };
    let surface = build(&placed(binary(rect, offset, BooleanOp1D::Difference)));
    contoured(&surface);
    assert!(!covers(&surface, 70.0, -55.0), "hole center must stay open");
    assert!(covers(&surface, 0.0, 0.0), "surface body must be filled");
    let expected = 200.0 * 200.0 - PI * 0.25;
    let area = mesh_area(&surface);
    assert!(
        (area - expected).abs() / expected < 1.0e-3,
        "area {area:.3} must be within 0.1% of {expected:.3}"
    );
}

#[test]
fn union_of_distant_small_shapes_keeps_both() {
    // Two r = 0.5 circles 180 apart: the shared bounds are huge, so both
    // circles need refined grid cells to appear at all.
    let left = Profile2D::circle([-90.0, 0.0], 0.5).expect("left");
    let right = Profile2D::circle([90.0, 0.0], 0.5).expect("right");
    let surface = build(&placed(binary(left, right, BooleanOp1D::Union)));
    contoured(&surface);
    assert!(covers(&surface, -90.0, 0.0), "left circle must be filled");
    assert!(covers(&surface, 90.0, 0.0), "right circle must be filled");
    let expected = 2.0 * PI * 0.25;
    let area = mesh_area(&surface);
    assert!(
        (area - expected).abs() / expected < 0.02,
        "area {area:.4} must be within 2% of {expected:.4}"
    );
}

#[test]
fn island_inside_hole_survives() {
    // 20 x 20 square minus (8 x 8 hole minus 2 x 2 island): three nested
    // rings — outer boundary, hole, island kept as its own filled group.
    let outer = Profile2D::rectangle([0.0, 0.0], [10.0, 10.0]).expect("outer");
    let hole = Profile2D::rectangle([0.0, 0.0], [4.0, 4.0]).expect("hole");
    let island = Profile2D::rectangle([0.0, 0.0], [1.0, 1.0]).expect("island");
    let ring = binary(hole, island, BooleanOp1D::Difference);
    let surface = build(&placed(binary(outer, ring, BooleanOp1D::Difference)));
    contoured(&surface);
    assert!(covers(&surface, 8.0, 8.0), "body must be filled");
    assert!(!covers(&surface, 2.5, 2.5), "hole ring must stay open");
    assert!(covers(&surface, 0.0, 0.0), "island must be filled");
    let expected = 400.0 - 64.0 + 4.0;
    let area = mesh_area(&surface);
    assert!(
        (area - expected).abs() / expected < 0.01,
        "area {area:.3} must be within 1% of {expected:.3}"
    );
}

#[test]
fn comparable_size_hole_renders_contoured_not_sampled() {
    // Even a large hole used to bail to the sampled staircase fill.
    let rect = Profile2D::rectangle([0.0, 0.0], [2.0, 2.0]).expect("rect");
    let hole = Profile2D::circle([0.0, 0.0], 1.0).expect("circle");
    let surface = build(&placed(binary(rect, hole, BooleanOp1D::Difference)));
    contoured(&surface);
    assert!(!covers(&surface, 0.0, 0.0), "hole center must stay open");
    assert!(covers(&surface, 1.7, 1.7), "corners must be filled");
    let expected = 16.0 - PI;
    let area = mesh_area(&surface);
    assert!(
        (area - expected).abs() / expected < 0.01,
        "area {area:.4} must be within 1% of {expected:.4}"
    );
}
