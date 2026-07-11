//! Model compile invariants: exactness grammar, generator/offset
//! preconditions, and sampled Domain disjointness.

use caso_kernel::frame::IDENTITY_FRAME;
use caso_kernel::model::{compile_model, disjointness_violations, grammar_violations, Model};
use caso_kernel::preconditions::precondition_violations;
use caso_kernel::roles::{Domain, DomainKind};
use caso_kernel::sdf::node::{Node, Shape};
use caso_kernel::sdf::placed::PlacedSdf2D;
use caso_kernel::sdf::primitives_2d::Profile2D;
use caso_kernel::sdf::primitives_3d::{Box3, Cylinder, Sphere};
use caso_kernel::sdf::solid_from_2d::{Revolve, RevolveAxis};
use caso_kernel::vec3::{vec3, Vec3};

fn box_node(name: &str, center: Vec3, half_size: Vec3) -> Node {
    Node::new(
        name,
        Shape::Box3(Box3::new(center, half_size, IDENTITY_FRAME).expect("box")),
    )
}

fn fluid(name: &str, region: Node) -> Domain {
    Domain::new(name, DomainKind::Fluid, region).expect("domain")
}

fn section(profile: Profile2D) -> Node {
    Node::new(
        "section",
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

fn revolve_node(name: &str, profile: Profile2D) -> Node {
    Node::new(
        name,
        Shape::Revolve(
            Revolve::new(section(profile), RevolveAxis::V, None, None, None, 360.0)
                .expect("revolve"),
        ),
    )
}

#[test]
fn disjoint_model_compiles() {
    let von_karman = Node::new(
        "von_karman",
        Shape::difference(
            box_node("flow", Vec3::ZERO, vec3(1.6, 0.7, 0.45)),
            Node::new(
                "obstacle",
                Shape::Cylinder(Cylinder::new(Vec3::ZERO, 0.24, 0.55, IDENTITY_FRAME).expect("cyl")),
            ),
        )
        .expect("difference"),
    );
    let far_box = box_node("solid", vec3(10.0, 0.0, 0.0), vec3(0.5, 0.5, 0.5));
    let model = Model::new(vec![
        fluid("fluid", von_karman),
        Domain::new("solid", DomainKind::Solid, far_box).expect("domain"),
    ])
    .expect("model");
    assert!(grammar_violations(&model).is_empty());
    assert!(disjointness_violations(&model, 32).expect("check").is_empty());
    compile_model(&model, 32).expect("valid model compiles");
}

#[test]
fn overlapping_domains_refuse_to_compile() {
    let a = box_node("a", Vec3::ZERO, vec3(0.5, 0.5, 0.5));
    let b = box_node("b", vec3(0.4, 0.0, 0.0), vec3(0.5, 0.5, 0.5));
    let model = Model::new(vec![fluid("a", a), fluid("b", b)]).expect("model");
    let error = compile_model(&model, 32).expect_err("overlap must fail");
    assert!(error.0.contains("overlap"), "unexpected message: {}", error.0);
}

#[test]
fn union_rooted_domain_fails_grammar() {
    let union = Node::new(
        "u",
        Shape::union(
            box_node("a", Vec3::ZERO, vec3(0.5, 0.5, 0.5)),
            Node::new(
                "s",
                Shape::Sphere(Sphere::new(vec3(0.9, 0.0, 0.0), 0.4).expect("sphere")),
            ),
        )
        .expect("union"),
    );
    let model = Model::new(vec![fluid("u", union)]).expect("model");
    let violations = grammar_violations(&model);
    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("exact interior distance"));
    assert!(compile_model(&model, 16).is_err());
}

#[test]
fn erosion_past_reach_is_refused() {
    let too_deep = Profile2D::distance_offset(
        Profile2D::rectangle([0.0, 0.0], [0.4, 0.25]).expect("rectangle"),
        -0.5,
    )
    .expect("offset");
    let violations = precondition_violations(&section(too_deep));
    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("erosion"), "{}", violations[0]);

    let safe = Profile2D::distance_offset(
        Profile2D::rectangle([0.0, 0.0], [0.4, 0.25]).expect("rectangle"),
        -0.1,
    )
    .expect("offset");
    assert!(precondition_violations(&section(safe)).is_empty());
}

#[test]
fn revolve_axis_crossing_needs_symmetry() {
    // Asymmetric profile crossing the axis: violation.
    let asymmetric = revolve_node(
        "bad",
        Profile2D::circle([0.05, 0.0], 0.3).expect("circle"),
    );
    let violations = precondition_violations(&asymmetric);
    assert_eq!(violations.len(), 1, "{violations:?}");
    assert!(violations[0].contains("crosses the revolution axis"));

    // Mirror-symmetric crossing (circle centered on the axis): fine.
    let symmetric = revolve_node("ok", Profile2D::circle([0.0, 0.0], 0.3).expect("circle"));
    assert!(precondition_violations(&symmetric).is_empty());

    // One-sided profile: fine.
    let one_sided = revolve_node(
        "side",
        Profile2D::circle([0.45, 0.0], 0.15).expect("circle"),
    );
    assert!(precondition_violations(&one_sided).is_empty());
}
