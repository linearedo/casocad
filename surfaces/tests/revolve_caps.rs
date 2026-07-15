//! Regression: revolving a profile that straddles the revolution axis must
//! produce the end caps. The tessellator clips the section outline to the
//! non-negative signed-radial half-plane before sweeping; without the clip an
//! axis-crossing outline edge is swept by unsigned radius and collapses,
//! leaving a straddling rectangle's revolve as a capless side wall.

use std::f64::consts::PI;

use caso_kernel::sdf::node::{Node, Shape};
use caso_kernel::sdf::placed::PlacedSdf2D;
use caso_kernel::sdf::primitives_2d::Profile2D;
use caso_kernel::sdf::solid_from_2d::{Revolve, RevolveAxis};
use caso_kernel::vec3::{vec3, Vec3};
use caso_surfaces::{build_viewport_surface, ViewportSurfaceKey};

#[test]
fn straddling_rectangle_revolve_has_caps() {
    // Rectangle on the XY plane straddling the v-axis, revolved 360° about
    // it: a solid cylinder along Y, radius 0.3, height 1.0.
    let section = Node::with_id(
        "section",
        3,
        Shape::PlacedSdf2D(
            PlacedSdf2D::new(
                Profile2D::rectangle([0.0, 0.0], [0.3, 0.5]).expect("rectangle"),
                Vec3::ZERO,
                vec3(1.0, 0.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                Vec::new(),
            )
            .expect("section"),
        ),
    );
    let revolve = Node::with_id(
        "cyl",
        7,
        Shape::Revolve(
            Revolve::new(section, RevolveAxis::V, None, None, None, 360.0).expect("revolve"),
        ),
    );
    let surface = build_viewport_surface(
        &revolve,
        ViewportSurfaceKey {
            object_id: 7,
            scene_revision: 1,
            resolution: 24,
        },
    );
    assert!(!surface.vertices.is_empty() && !surface.indices.is_empty());

    // Side wall 2πrh plus two cap disks 2πr². A missing cap or a
    // double-covered mirror half both fail this by far more than the
    // ring-polygon discretization error.
    let vertex = |index: u32| {
        let v = surface.vertices[index as usize];
        vec3(v[0] as f64, v[1] as f64, v[2] as f64)
    };
    let mut area = 0.0;
    for triangle in surface.indices.chunks_exact(3) {
        let (a, b, c) = (vertex(triangle[0]), vertex(triangle[1]), vertex(triangle[2]));
        area += 0.5 * (b - a).cross(c - a).length();
    }
    let expected = 2.0 * PI * 0.3 * 1.0 + 2.0 * PI * 0.3 * 0.3;
    assert!(
        (area - expected).abs() / expected < 0.02,
        "surface area {area:.4} should match the capped cylinder {expected:.4}"
    );

    // The cap fans close on the axis: welded vertices at (0, ±0.5, 0).
    for cap_y in [-0.5, 0.5] {
        assert!(
            surface.vertices.iter().any(|v| {
                let radial = (v[0] as f64).hypot(v[2] as f64);
                radial < 1.0e-9 && (v[1] as f64 - cap_y).abs() < 1.0e-9
            }),
            "cap at y={cap_y} must close on the axis"
        );
    }
}
