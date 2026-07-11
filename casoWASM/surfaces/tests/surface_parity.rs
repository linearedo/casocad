//! Surface-builder parity with the Python viewport builders.
//!
//! `casoWASM/tools/export_surface_goldens.py` runs the Python
//! `build_viewport_surface` on a fixture subset at resolutions 12 (dense) and
//! 96 (narrow band / clip) and records status + vertex/triangle/wire counts +
//! max |sdf| at the produced vertices. This test rebuilds identical fixtures
//! and requires matching metrics.

use caso_kernel::frame::{Frame, IDENTITY_FRAME};
use caso_kernel::sdf::node::{Node, Shape};
use caso_kernel::sdf::placed::{PlacedPolyline1D, PlacedSdf1D, PlacedSdf2D};
use caso_kernel::sdf::primitives_1d::Profile1D;
use caso_kernel::sdf::primitives_2d::Profile2D;
use caso_kernel::sdf::primitives_3d::{
    Box3, BoxFrame, CappedCone, Cone, Cylinder, Pyramid, Sphere, Torus,
};
use caso_kernel::sdf::solid_from_2d::{Extrude, Revolve, RevolveAxis};
use caso_kernel::sdf::tubes::{CapStyle, PolylineTube, QuadraticBezierTube};
use caso_kernel::vec3::{vec3, Vec3};
use caso_surfaces::types::SurfaceStatus;
use caso_surfaces::{build_viewport_surface, ViewportSurfaceKey};

fn orient() -> Frame {
    Frame::orthonormal(
        vec3(0.6, 0.8, 0.0),
        vec3(-0.8, 0.6, 0.0),
        vec3(0.0, 0.0, 1.0),
    )
    .expect("orthonormal test frame")
}

fn node(name: &str, shape: Shape) -> Node {
    Node::with_id(name, 7, shape)
}

fn section(profile: Profile2D) -> Node {
    node(
        "section",
        Shape::PlacedSdf2D(
            PlacedSdf2D::new(
                profile,
                vec3(0.15, -0.1, 0.2),
                vec3(1.0, 0.0, 0.0),
                vec3(0.0, 0.6, 0.8),
                Vec::new(),
            )
            .expect("section"),
        ),
    )
}

fn op_sphere() -> Node {
    node(
        "op_sphere",
        Shape::Sphere(Sphere::new(vec3(0.2, 0.0, 0.0), 0.5).expect("sphere")),
    )
}

fn op_box() -> Node {
    node(
        "op_box",
        Shape::Box3(
            Box3::new(vec3(-0.1, 0.0, 0.0), vec3(0.45, 0.35, 0.3), IDENTITY_FRAME).expect("box"),
        ),
    )
}

fn build_fixture(name: &str) -> Node {
    match name {
        "sphere" => node(
            name,
            Shape::Sphere(Sphere::new(vec3(0.1, -0.2, 0.3), 0.7).expect("sphere")),
        ),
        "box_oriented" => node(
            name,
            Shape::Box3(Box3::new(vec3(0.0, 0.2, 0.1), vec3(0.4, 0.6, 0.25), orient()).expect("box")),
        ),
        "cylinder" => node(
            name,
            Shape::Cylinder(
                Cylinder::new(vec3(-0.1, 0.0, 0.2), 0.35, 0.5, IDENTITY_FRAME).expect("cylinder"),
            ),
        ),
        "torus" => node(
            name,
            Shape::Torus(Torus::new(vec3(0.1, 0.0, -0.05), 0.5, 0.14, orient()).expect("torus")),
        ),
        "pyramid" => node(
            name,
            Shape::Pyramid(
                Pyramid::new(vec3(0.0, 0.0, 0.05), 0.45, 0.4, IDENTITY_FRAME).expect("pyramid"),
            ),
        ),
        "cone" => node(
            name,
            Shape::Cone(Cone::new(vec3(0.0, 0.1, -0.1), 0.4, 0.5, IDENTITY_FRAME).expect("cone")),
        ),
        "cappedcone" => node(
            name,
            Shape::CappedCone(
                CappedCone::new(vec3(0.05, -0.05, 0.1), 0.45, 0.2, 0.5, IDENTITY_FRAME)
                    .expect("capped cone"),
            ),
        ),
        "boxframe" => node(
            name,
            Shape::BoxFrame(
                BoxFrame::new(vec3(0.0, 0.1, 0.0), vec3(0.5, 0.4, 0.3), 0.06, IDENTITY_FRAME)
                    .expect("box frame"),
            ),
        ),
        "polyline_tube_round" => node(
            name,
            Shape::PolylineTube(
                PolylineTube::new(
                    vec![
                        vec3(-0.7, -0.1, 0.0),
                        vec3(0.0, 0.5, 0.2),
                        vec3(0.7, 0.0, -0.2),
                    ],
                    0.12,
                    0.0,
                    CapStyle::Round,
                )
                .expect("tube"),
            ),
        ),
        "bezier_tube_round" => node(
            name,
            Shape::QuadraticBezierTube(
                QuadraticBezierTube::new(
                    vec![
                        vec3(-0.75, 0.0, 0.0),
                        vec3(0.0, 0.55, 0.3),
                        vec3(0.75, 0.0, 0.0),
                    ],
                    0.12,
                    0.0,
                    CapStyle::Round,
                )
                .expect("tube"),
            ),
        ),
        "extrude" => node(
            name,
            Shape::Extrude(
                Extrude::new(
                    section(
                        Profile2D::polygon(vec![
                            [-0.4, -0.3],
                            [0.5, -0.2],
                            [0.3, 0.4],
                            [-0.35, 0.3],
                        ])
                        .expect("polygon"),
                    ),
                    0.8,
                    0.15,
                )
                .expect("extrude"),
            ),
        ),
        "revolve_full" => node(
            name,
            Shape::Revolve(
                Revolve::new(
                    section(Profile2D::circle([0.45, 0.0], 0.15).expect("circle")),
                    RevolveAxis::V,
                    None,
                    None,
                    None,
                    360.0,
                )
                .expect("revolve"),
            ),
        ),
        "revolve_partial" => node(
            name,
            Shape::Revolve(
                Revolve::new(
                    section(Profile2D::circle([0.45, 0.0], 0.15).expect("circle")),
                    RevolveAxis::V,
                    None,
                    None,
                    None,
                    120.0,
                )
                .expect("revolve"),
            ),
        ),
        "von_karman" => {
            let flow = node(
                "flow_volume",
                Shape::Box3(
                    Box3::new(Vec3::ZERO, vec3(1.6, 0.7, 0.45), IDENTITY_FRAME).expect("box"),
                ),
            );
            let obstacle = node(
                "cylinder_obstacle",
                Shape::Cylinder(
                    Cylinder::new(Vec3::ZERO, 0.24, 0.55, IDENTITY_FRAME).expect("cylinder"),
                ),
            );
            node(name, Shape::difference(flow, obstacle).expect("difference"))
        }
        "op_union" => node(name, Shape::union(op_sphere(), op_box()).expect("union")),
        "op_xor" => node(name, Shape::xor(op_sphere(), op_box()).expect("xor")),
        "op_nested" => {
            let cyl = node(
                "op_cyl",
                Shape::Cylinder(
                    Cylinder::new(vec3(0.0, 0.2, 0.0), 0.25, 0.6, IDENTITY_FRAME).expect("cylinder"),
                ),
            );
            let torus = node(
                "op_torus",
                Shape::Torus(
                    Torus::new(vec3(0.0, 0.0, 0.1), 0.45, 0.12, IDENTITY_FRAME).expect("torus"),
                ),
            );
            let nested_i = node(
                "nested_i",
                Shape::intersection(op_box(), op_sphere()).expect("intersection"),
            );
            let nested_u = node("nested_u", Shape::union(cyl, torus).expect("union"));
            node(name, Shape::difference(nested_i, nested_u).expect("difference"))
        }
        "placed2d_circle" => section(Profile2D::circle([0.1, -0.05], 0.45).expect("circle")),
        "placed2d_polygon" => section(
            Profile2D::polygon(vec![
                [-0.6, -0.4],
                [0.6, -0.4],
                [0.2, 0.1],
                [0.5, 0.5],
                [-0.4, 0.4],
            ])
            .expect("polygon"),
        ),
        "placed2d_bezier_surface_open" => section(
            Profile2D::quadratic_bezier_surface(vec![
                [-0.65, -0.35],
                [-0.25, 0.55],
                [0.1, 0.25],
                [0.45, -0.05],
                [0.55, -0.45],
            ])
            .expect("bezier surface"),
        ),
        "placed1d_segment" => node(
            name,
            Shape::PlacedSdf1D(
                PlacedSdf1D::new(
                    Profile1D::segment(0.1, 0.6).expect("segment"),
                    vec3(0.0, 0.1, -0.05),
                    vec3(0.6, 0.8, 0.0),
                    Vec::new(),
                )
                .expect("placed 1d"),
            ),
        ),
        "placed_polyline_1d" => node(
            name,
            Shape::PlacedPolyline1D(
                PlacedPolyline1D::new(
                    Profile2D::polyline(vec![[-0.5, -0.2], [0.0, 0.3], [0.5, -0.1]])
                        .expect("polyline"),
                    vec3(0.0, 0.0, 0.1),
                    vec3(1.0, 0.0, 0.0),
                    vec3(0.0, 0.6, 0.8),
                )
                .expect("placed polyline"),
            ),
        ),
        other => panic!("unknown fixture {other}"),
    }
}

fn status_str(status: SurfaceStatus) -> &'static str {
    match status {
        SurfaceStatus::Ready => "ready",
        SurfaceStatus::Outline => "outline",
        SurfaceStatus::Empty => "empty",
        SurfaceStatus::Failed => "failed",
    }
}

#[test]
fn surface_metrics_match_python() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/goldens/surface_goldens.txt"
    );
    let text = std::fs::read_to_string(path).expect(
        "golden file missing — run `.venv/bin/python \
         casoWASM/tools/export_surface_goldens.py` from the casoCAD repo root",
    );
    let mut failures: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for line in text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 8 || parts[0] != "s" {
            continue;
        }
        let name = parts[1];
        let resolution: u32 = parts[2].parse().expect("resolution");
        let status = parts[3];
        let vertex_count: usize = parts[4].parse().expect("vertex count");
        let triangle_count: usize = parts[5].parse().expect("triangle count");
        let wire_len: usize = parts[6].parse().expect("wire length");
        let max_err: f64 = parts[7].parse().expect("max err");

        let fixture = build_fixture(name);
        let key = ViewportSurfaceKey {
            object_id: 7,
            scene_revision: 1,
            resolution,
        };
        let surface = build_viewport_surface(&fixture, key);
        let mut mismatches: Vec<String> = Vec::new();
        if status_str(surface.status) != status {
            mismatches.push(format!(
                "status {} != {status}",
                status_str(surface.status)
            ));
        }
        if surface.vertex_count() != vertex_count {
            mismatches.push(format!(
                "vertices {} != {vertex_count}",
                surface.vertex_count()
            ));
        }
        if surface.triangle_count() != triangle_count {
            mismatches.push(format!(
                "triangles {} != {triangle_count}",
                surface.triangle_count()
            ));
        }
        if surface.wire_indices.len() != wire_len {
            mismatches.push(format!(
                "wire {} != {wire_len}",
                surface.wire_indices.len()
            ));
        }
        // Max |sdf| at used vertices: same construction should agree closely.
        if !surface.vertices.is_empty() && !surface.indices.is_empty() {
            let mut used: Vec<u32> = surface.indices.clone();
            used.sort_unstable();
            used.dedup();
            let points: Vec<Vec3> = used
                .iter()
                .map(|index| {
                    let vertex = surface.vertices[*index as usize];
                    vec3(vertex[0] as f64, vertex[1] as f64, vertex[2] as f64)
                })
                .collect();
            let values = fixture.eval(&points);
            let rust_err = values.iter().fold(0.0f64, |acc, value| acc.max(value.abs()));
            let tolerance = 1.0e-6_f64.max(max_err * 0.2);
            if (rust_err - max_err).abs() > tolerance {
                mismatches.push(format!("max_err {rust_err:.6e} != {max_err:.6e}"));
            }
        }
        if !mismatches.is_empty() {
            failures.push(format!("{name}@{resolution}: {}", mismatches.join("; ")));
        }
        checked += 1;
    }
    assert!(checked >= 40, "expected the full golden set, found {checked}");
    assert!(
        failures.is_empty(),
        "surface parity failures:\n  {}",
        failures.join("\n  ")
    );
}
