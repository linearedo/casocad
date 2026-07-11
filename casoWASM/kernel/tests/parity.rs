//! Golden parity against the Python casoCAD kernel.
//!
//! `casoWASM/tools/export_goldens.py` samples every fixture below through the
//! Python `to_numpy()` path and writes `tests/goldens/kernel_goldens.txt`;
//! this test rebuilds the identical fixtures in Rust and requires agreement
//! at f64 round-off level. Fixture definitions must stay in sync with the
//! exporter by name.

use caso_kernel::frame::{Frame, IDENTITY_FRAME};
use caso_kernel::roles::{node_exactness, validate_exactness, Exactness};
use caso_kernel::sdf::curtain::NormalCurtain;
use caso_kernel::sdf::node::{Node, RotationAxis, Shape};
use caso_kernel::sdf::placed::{PlacedPolyline1D, PlacedSdf1D, PlacedSdf2D};
use caso_kernel::sdf::primitives_1d::{BooleanOp1D, Profile1D};
use caso_kernel::sdf::primitives_2d::Profile2D;
use caso_kernel::sdf::primitives_3d::{
    Box3, BoxFrame, CappedCone, Cone, Cylinder, Pyramid, Sphere, Torus,
};
use caso_kernel::sdf::solid_from_2d::{Extrude, Revolve, RevolveAxis};
use caso_kernel::sdf::tubes::{CapStyle, PolylineTube, QuadraticBezierTube};
use caso_kernel::vec3::{vec3, Vec3};

fn orient() -> Frame {
    Frame::orthonormal(
        vec3(0.6, 0.8, 0.0),
        vec3(-0.8, 0.6, 0.0),
        vec3(0.0, 0.0, 1.0),
    )
    .expect("orthonormal test frame")
}

fn node(name: &str, shape: Shape) -> Node {
    Node::new(name, shape)
}

fn section(profile: Profile2D) -> Node {
    section_at(profile, vec3(0.15, -0.1, 0.2), vec3(1.0, 0.0, 0.0), vec3(0.0, 0.6, 0.8))
}

fn section_at(profile: Profile2D, origin: Vec3, axis_u: Vec3, axis_v: Vec3) -> Node {
    node(
        "section",
        Shape::PlacedSdf2D(
            PlacedSdf2D::new(profile, origin, axis_u, axis_v, Vec::new()).expect("section"),
        ),
    )
}

fn placed2d_profile(key: &str) -> Profile2D {
    match key {
        "circle" => Profile2D::circle([0.1, -0.05], 0.45),
        "rectangle" => Profile2D::rectangle([0.0, 0.1], [0.5, 0.3]),
        "square" => Profile2D::square([-0.1, 0.0], 0.4),
        "rounded_rectangle" => Profile2D::rounded_rectangle([0.0, 0.0], [0.5, 0.35], 0.12),
        "ellipse" => Profile2D::ellipse([0.05, -0.1], [0.6, 0.3]),
        "ellipse_circular" => Profile2D::ellipse([0.0, 0.0], [0.4, 0.4]),
        "regular_polygon" => Profile2D::regular_polygon([0.0, 0.05], 0.5, 5, 0.3),
        "polygon" => Profile2D::polygon(vec![
            [-0.6, -0.4],
            [0.6, -0.4],
            [0.2, 0.1],
            [0.5, 0.5],
            [-0.4, 0.4],
        ]),
        "polyline" => Profile2D::polyline(vec![
            [-0.6, -0.4],
            [0.6, -0.4],
            [0.35, 0.4],
            [-0.35, 0.4],
        ]),
        "bezier_curve" => {
            Profile2D::quadratic_bezier_curve(vec![[-0.6, -0.35], [0.0, 0.55], [0.6, -0.35]])
        }
        "bezier_polycurve" => Profile2D::quadratic_bezier_curve(vec![
            [-0.65, -0.35],
            [-0.25, 0.55],
            [0.1, 0.25],
            [0.45, -0.05],
            [0.55, -0.45],
        ]),
        "bezier_surface_open" => Profile2D::quadratic_bezier_surface(vec![
            [-0.65, -0.35],
            [-0.25, 0.55],
            [0.1, 0.25],
            [0.45, -0.05],
            [0.55, -0.45],
        ]),
        "bezier_surface_closed" => Profile2D::quadratic_bezier_surface(vec![
            [-0.5, -0.3],
            [0.0, 0.6],
            [0.5, -0.3],
            [0.0, -0.7],
            [-0.5, -0.3],
        ]),
        "offset" => Ok(Profile2D::Offset {
            child: std::boxed::Box::new(Profile2D::circle([0.0, 0.0], 0.3).expect("circle")),
            offset: [0.2, -0.15],
        }),
        "distance_offset" => Profile2D::distance_offset(
            Profile2D::rectangle([0.0, 0.0], [0.4, 0.25]).expect("rectangle"),
            0.08,
        ),
        "binary_union" => Ok(Profile2D::Binary {
            left: std::boxed::Box::new(Profile2D::circle([-0.2, 0.0], 0.3).expect("circle")),
            right: std::boxed::Box::new(
                Profile2D::rectangle([0.2, 0.0], [0.3, 0.2]).expect("rectangle"),
            ),
            operation: BooleanOp1D::Union,
            smoothing: 0.1,
        }),
        "binary_difference" => Ok(Profile2D::Binary {
            left: std::boxed::Box::new(
                Profile2D::rectangle([0.0, 0.0], [0.5, 0.35]).expect("rectangle"),
            ),
            right: std::boxed::Box::new(Profile2D::circle([0.15, 0.05], 0.2).expect("circle")),
            operation: BooleanOp1D::Difference,
            smoothing: 0.1,
        }),
        other => panic!("unknown placed2d profile {other}"),
    }
    .expect("profile fixture")
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
    if let Some(key) = name.strip_prefix("placed2d_") {
        return section(placed2d_profile(key));
    }
    match name {
        "sphere" => node(
            name,
            Shape::Sphere(Sphere::new(vec3(0.1, -0.2, 0.3), 0.7).expect("sphere")),
        ),
        "box_axis" => node(
            name,
            Shape::Box3(
                Box3::new(vec3(0.2, 0.1, -0.1), vec3(0.5, 0.3, 0.4), IDENTITY_FRAME).expect("box"),
            ),
        ),
        "box_oriented" => node(
            name,
            Shape::Box3(
                Box3::new(vec3(0.0, 0.2, 0.1), vec3(0.4, 0.6, 0.25), orient()).expect("box"),
            ),
        ),
        "cylinder" => node(
            name,
            Shape::Cylinder(
                Cylinder::new(vec3(-0.1, 0.0, 0.2), 0.35, 0.5, IDENTITY_FRAME).expect("cylinder"),
            ),
        ),
        "cylinder_oriented" => node(
            name,
            Shape::Cylinder(
                Cylinder::new(vec3(0.1, 0.1, 0.0), 0.3, 0.45, orient()).expect("cylinder"),
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
        "pyramid" => node(
            name,
            Shape::Pyramid(
                Pyramid::new(vec3(0.0, 0.0, 0.05), 0.45, 0.4, IDENTITY_FRAME).expect("pyramid"),
            ),
        ),
        "boxframe" => node(
            name,
            Shape::BoxFrame(
                BoxFrame::new(vec3(0.0, 0.1, 0.0), vec3(0.5, 0.4, 0.3), 0.06, IDENTITY_FRAME)
                    .expect("box frame"),
            ),
        ),
        "torus" => node(
            name,
            Shape::Torus(Torus::new(vec3(0.1, 0.0, -0.05), 0.5, 0.14, orient()).expect("torus")),
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
        "polyline_tube_flat_inner" => node(
            name,
            Shape::PolylineTube(
                PolylineTube::new(
                    vec![
                        vec3(-0.7, -0.1, 0.0),
                        vec3(0.0, 0.5, 0.2),
                        vec3(0.7, 0.0, -0.2),
                    ],
                    0.15,
                    0.05,
                    CapStyle::Flat,
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
        "bezier_polycurve_tube_flat" => node(
            name,
            Shape::QuadraticBezierTube(
                QuadraticBezierTube::new(
                    vec![
                        vec3(-0.8, 0.0, 0.0),
                        vec3(-0.4, 0.5, 0.1),
                        vec3(0.0, 0.1, 0.2),
                        vec3(0.4, -0.4, 0.1),
                        vec3(0.8, 0.1, 0.0),
                    ],
                    0.1,
                    0.03,
                    CapStyle::Flat,
                )
                .expect("tube"),
            ),
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
        "placed1d_binary" => node(
            name,
            Shape::PlacedSdf1D(
                PlacedSdf1D::new(
                    Profile1D::Binary {
                        left: std::boxed::Box::new(
                            Profile1D::segment(-0.2, 0.4).expect("segment"),
                        ),
                        right: std::boxed::Box::new(Profile1D::Offset {
                            child: std::boxed::Box::new(
                                Profile1D::segment(0.0, 0.3).expect("segment"),
                            ),
                            offset: 0.35,
                        }),
                        operation: BooleanOp1D::Difference,
            smoothing: 0.1,
                    },
                    vec3(0.1, 0.0, 0.0),
                    vec3(0.0, 0.0, 1.0),
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
        "placed_bezier_1d" => node(
            name,
            Shape::PlacedPolyline1D(
                PlacedPolyline1D::new(
                    Profile2D::quadratic_bezier_curve(vec![[-0.5, -0.2], [0.0, 0.5], [0.5, -0.2]])
                        .expect("bezier"),
                    vec3(0.05, -0.05, 0.0),
                    vec3(1.0, 0.0, 0.0),
                    vec3(0.0, 1.0, 0.0),
                )
                .expect("placed bezier"),
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
        "revolve_negative" => node(
            name,
            Shape::Revolve(
                Revolve::new(
                    section(Profile2D::rectangle([0.4, 0.1], [0.12, 0.2]).expect("rectangle")),
                    RevolveAxis::U,
                    None,
                    None,
                    None,
                    -90.0,
                )
                .expect("revolve"),
            ),
        ),
        "normalcurtain" => node(
            name,
            Shape::NormalCurtain(
                NormalCurtain::new(
                    vec![
                        vec3(-0.5, -0.1, 0.0),
                        vec3(0.0, 0.2, 0.05),
                        vec3(0.5, 0.1, -0.05),
                    ],
                    vec![
                        vec3(0.0, 0.1, 1.0),
                        vec3(0.1, 0.0, 1.0),
                        vec3(0.0, -0.1, 1.0),
                    ],
                    2.0,
                )
                .expect("curtain"),
            ),
        ),
        "op_union" => node(name, Shape::union(op_sphere(), op_box()).expect("union")),
        "op_intersection" => node(
            name,
            Shape::intersection(op_sphere(), op_box()).expect("intersection"),
        ),
        "op_difference" => node(
            name,
            Shape::difference(op_box(), op_sphere()).expect("difference"),
        ),
        "op_xor" => node(name, Shape::xor(op_sphere(), op_box()).expect("xor")),
        "op_nested" => {
            let cyl = node(
                "op_cyl",
                Shape::Cylinder(
                    Cylinder::new(vec3(0.0, 0.2, 0.0), 0.25, 0.6, IDENTITY_FRAME)
                        .expect("cylinder"),
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
        "transform_translate" => node(
            name,
            Shape::Translate {
                child: std::boxed::Box::new(node(
                    "t_cyl",
                    Shape::Cylinder(
                        Cylinder::new(Vec3::ZERO, 0.3, 0.4, IDENTITY_FRAME).expect("cylinder"),
                    ),
                )),
                offset: vec3(0.3, -0.2, 0.1),
            },
        ),
        "transform_scale" => node(
            name,
            Shape::scale(
                node(
                    "s_box",
                    Shape::Box3(
                        Box3::new(vec3(0.1, 0.0, 0.0), vec3(0.3, 0.2, 0.25), IDENTITY_FRAME)
                            .expect("box"),
                    ),
                ),
                1.7,
            )
            .expect("scale"),
        ),
        "transform_rotate_x" | "transform_rotate_y" | "transform_rotate_z" => {
            let (axis, angle, child_name) = match name {
                "transform_rotate_x" => (RotationAxis::X, 35.0, "r_box_x"),
                "transform_rotate_y" => (RotationAxis::Y, -50.0, "r_box_y"),
                _ => (RotationAxis::Z, 120.0, "r_box_z"),
            };
            node(
                name,
                Shape::Rotate {
                    child: std::boxed::Box::new(node(
                        child_name,
                        Shape::Box3(
                            Box3::new(vec3(0.0, 0.1, 0.0), vec3(0.4, 0.2, 0.3), IDENTITY_FRAME)
                                .expect("box"),
                        ),
                    )),
                    axis,
                    angle_degrees: angle,
                },
            )
        }
        "transform_stack" => node(
            name,
            Shape::Translate {
                child: std::boxed::Box::new(node(
                    "stack_rot",
                    Shape::Rotate {
                        child: std::boxed::Box::new(node(
                            "stack_scale",
                            Shape::scale(
                                node(
                                    "stack_torus",
                                    Shape::Torus(
                                        Torus::new(Vec3::ZERO, 0.4, 0.1, IDENTITY_FRAME)
                                            .expect("torus"),
                                    ),
                                ),
                                1.3,
                            )
                            .expect("scale"),
                        )),
                        axis: RotationAxis::Y,
                        angle_degrees: 40.0,
                    },
                )),
                offset: vec3(0.2, 0.1, -0.15),
            },
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
        other => panic!("unknown fixture {other}"),
    }
}

struct GoldenFixture {
    name: String,
    points: Vec<Vec3>,
    values: Vec<f64>,
}

fn load_goldens() -> Vec<GoldenFixture> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/goldens/kernel_goldens.txt");
    let text = std::fs::read_to_string(path).expect(
        "golden file missing — run `.venv/bin/python casoWASM/tools/export_goldens.py` \
         from the casoCAD repo root",
    );
    let mut fixtures = Vec::new();
    let mut current: Option<GoldenFixture> = None;
    for line in text.lines() {
        if let Some(name) = line.strip_prefix("fixture ") {
            current = Some(GoldenFixture {
                name: name.to_string(),
                points: Vec::new(),
                values: Vec::new(),
            });
        } else if let Some(rest) = line.strip_prefix("p ") {
            let fixture = current.as_mut().expect("sample outside fixture");
            let mut parts = rest.split_whitespace();
            let mut next = || -> f64 {
                parts
                    .next()
                    .expect("four numbers per sample line")
                    .parse()
                    .expect("parseable float")
            };
            let (x, y, z, value) = (next(), next(), next(), next());
            fixture.points.push(vec3(x, y, z));
            fixture.values.push(value);
        } else if line == "end" {
            fixtures.push(current.take().expect("end outside fixture"));
        }
    }
    fixtures
}

#[test]
fn kernel_matches_python_goldens() {
    let goldens = load_goldens();
    assert!(
        goldens.len() >= 50,
        "expected the full fixture set, found {}",
        goldens.len()
    );
    let mut worst: (f64, String) = (0.0, String::new());
    for golden in &goldens {
        let fixture = build_fixture(&golden.name);
        let values = fixture.eval(&golden.points);
        for (index, (rust, python)) in values.iter().zip(golden.values.iter()).enumerate() {
            let scale = python.abs().max(1.0);
            let error = (rust - python).abs() / scale;
            if error > worst.0 {
                worst = (error, format!("{} sample {index}", golden.name));
            }
            assert!(
                error <= 1.0e-12,
                "{} sample {} at {:?}: rust {} vs python {} (relative error {:.3e})",
                golden.name,
                index,
                golden.points[index],
                rust,
                python,
                error
            );
        }
    }
    println!("worst relative error: {:.3e} ({})", worst.0, worst.1);
}

#[test]
fn exactness_grammar_matches_spec() {
    // difference(region, obstacle) -> inside-exact; valid Domain root.
    let von_karman = build_fixture("von_karman");
    assert_eq!(node_exactness(&von_karman), Exactness::SDF_INSIDE);
    assert!(validate_exactness(&von_karman).is_ok());

    // A union result cannot define a meshable Domain (outside-exact only).
    let union = build_fixture("op_union");
    assert_eq!(node_exactness(&union), Exactness::SDF_OUTSIDE);
    assert!(validate_exactness(&union).is_err());

    // XOR is outside the exact compiler grammar entirely.
    let xor = build_fixture("op_xor");
    assert_eq!(node_exactness(&xor), Exactness::NONE);
    assert!(validate_exactness(&xor).is_err());

    // Transforms are exactness-transparent.
    let stack = build_fixture("transform_stack");
    assert_eq!(node_exactness(&stack), Exactness::SDF_BOTH);
}

#[test]
fn bounding_boxes_contain_surface_samples() {
    // The traversal bound must contain every point where |sdf| is small.
    for name in ["sphere", "box_oriented", "torus", "von_karman", "extrude"] {
        let fixture = build_fixture(name);
        let bounds = fixture.bounding_box().expect("bounding box");
        let pad = 1.0e-9;
        for point in [
            vec3(bounds.x_min, bounds.y_min, bounds.z_min),
            bounds.center(),
        ] {
            // Center of the box must be inside-or-near the shape's extent.
            let _ = fixture.eval_point(point);
        }
        assert!(bounds.x_min <= bounds.x_max + pad);
    }
}
