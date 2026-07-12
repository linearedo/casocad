//! Boundary-cutter knife edge cases (design_docs/boundary_cutter_exactness.md):
//! point stencils (polygon, quadratic bezier surface) sit on the plane fitted
//! to the clicks and extrude one-sidedly, so closed surfaces never grow an
//! antipodal phantom cut. Fixtures: a unit sphere Domain and the default von
//! Kármán scene (box half-size 2.25×1.5×0.5 at (2.25, 0, 0.5) minus a
//! Y-axis cylinder r=0.15 at (1.8, 0, 0.5)).

use caso_kernel::boundary_paths::stencil_knife;
use caso_kernel::scene::{SceneDocument, ScenePayload};
use caso_kernel::sdf::node::{Node, Shape};
use caso_kernel::sdf::primitives_3d::Sphere;
use caso_kernel::serialization::{ghost_from_json, ghost_to_json};
use caso_kernel::vec3::{vec3, Vec3};

fn default_box_root() -> Node {
    let document = SceneDocument::default_scene().expect("default scene");
    let fluid_root = document.fluid_domain.as_ref().expect("fluid").root;
    let root = document.build_node(fluid_root).expect("root node");
    // Sanity: the scene still carries its flow box.
    assert!(document.walk().iter().any(|(id, _parent)| matches!(
        document.object(*id).expect("object").payload,
        ScenePayload::Box3(_)
    )));
    root
}

fn sphere_root() -> Node {
    Node::with_id(
        "sphere_domain",
        1,
        Shape::Sphere(Sphere::new(Vec3::ZERO, 1.0).expect("sphere")),
    )
}

/// Quasi-uniform unit-sphere sample directions (lat/long grid, poles included).
fn sphere_samples() -> Vec<Vec3> {
    let mut samples = vec![vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, -1.0)];
    for i in 1..12 {
        let polar = std::f64::consts::PI * (i as f64) / 12.0;
        for j in 0..24 {
            let azimuth = 2.0 * std::f64::consts::PI * (j as f64) / 24.0;
            samples.push(vec3(
                polar.sin() * azimuth.cos(),
                polar.sin() * azimuth.sin(),
                polar.cos(),
            ));
        }
    }
    samples
}

/// Three clicks at 45° latitude, 120° apart — an upper-cap triangle.
fn cap_triangle_clicks() -> [Vec3; 3] {
    let z = std::f64::consts::FRAC_1_SQRT_2;
    let r = std::f64::consts::FRAC_1_SQRT_2;
    [
        vec3(r, 0.0, z),
        vec3(-0.5 * r, 0.866_025_403_784_438_6 * r, z),
        vec3(-0.5 * r, -0.866_025_403_784_438_6 * r, z),
    ]
}

#[test]
fn polygon_stencil_on_a_sphere_has_no_antipodal_mirror() {
    let root = sphere_root();
    let clicks = cap_triangle_clicks();

    let (ghost, curved) = stencil_knife(&root, "polygon", &clicks).expect("stencil");

    assert!(curved, "45°-latitude clicks must raise the curved-surface flag");
    // The cap centroid direction is inside the stencil footprint and above
    // the one-sided extrusion's lower bound.
    assert!(ghost.eval_point(vec3(0.0, 0.0, 1.0)) < 0.0);
    // The mirrored cap (the phantom the user reported) is fully Outside.
    assert!(ghost.eval_point(vec3(0.0, 0.0, -1.0)) > 0.0);
    for click in clicks {
        let mirrored = vec3(click.x, click.y, -click.z);
        assert!(
            ghost.eval_point(mirrored) > 0.0,
            "mirrored click {mirrored:?} must be Outside"
        );
    }
    for sample in sphere_samples() {
        if ghost.eval_point(sample) < 0.0 {
            assert!(
                sample.z > 0.5,
                "negative stencil sample {sample:?} lies off the clicked cap"
            );
        }
    }
}

#[test]
fn bezier_stencil_on_a_sphere_has_no_antipodal_mirror() {
    let root = sphere_root();
    let corners = cap_triangle_clicks();
    // anchor, control, anchor (odd count).
    let clicks = [corners[0], corners[1], corners[2]];

    let (ghost, curved) =
        stencil_knife(&root, "quadratic_bezier_surface", &clicks).expect("stencil");

    assert!(curved);
    assert!(ghost.eval_point(vec3(0.0, 0.0, -1.0)) > 0.0);
    for sample in sphere_samples() {
        if ghost.eval_point(sample) < 0.0 {
            assert!(
                sample.z > 0.5,
                "negative bezier sample {sample:?} lies off the clicked cap"
            );
        }
    }
}

#[test]
fn flat_face_stencil_matches_the_drawn_polygon() {
    let root = default_box_root();
    let clicks = [
        vec3(0.0, -0.2, 0.3),
        vec3(0.0, 0.2, 0.3),
        vec3(0.0, 0.0, 0.7),
    ];

    let (ghost, curved) = stencil_knife(&root, "polygon", &clicks).expect("stencil");

    assert!(!curved, "a flat face must not raise the curvature flag");
    // In-plane probes on the -X face: triangle interior vs exterior.
    assert!(ghost.eval_point(vec3(0.0, 0.0, 0.4)) < 0.0);
    assert!(ghost.eval_point(vec3(0.0, 0.3, 0.8)) > 0.0);
    // The opposite (+X) face lies far below the one-sided extrusion.
    assert!(ghost.eval_point(vec3(4.5, 0.0, 0.4)) > 0.0);
}

#[test]
fn stencil_is_click_order_independent() {
    let root = sphere_root();
    let [a, b, c] = cap_triangle_clicks();
    let probes = [
        vec3(0.0, 0.0, 1.0),
        vec3(0.5, 0.5, 0.707),
        vec3(0.0, 0.0, -1.0),
    ];

    let (base, _) = stencil_knife(&root, "polygon", &[a, b, c]).expect("stencil");
    let (rotated, _) = stencil_knife(&root, "polygon", &[b, c, a]).expect("stencil");
    let (reversed, _) = stencil_knife(&root, "polygon", &[c, b, a]).expect("stencil");

    for probe in probes {
        let sign = base.eval_point(probe).signum();
        assert_eq!(sign, rotated.eval_point(probe).signum());
        assert_eq!(sign, reversed.eval_point(probe).signum());
    }
}

#[test]
fn opposing_click_normals_reject_the_stencil() {
    let root = sphere_root();
    let clicks = [
        vec3(1.0, 0.0, 0.0),
        vec3(-1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, -1.0, 0.0),
    ];

    let error = stencil_knife(&root, "polygon", &clicks).expect_err("must reject");
    assert!(
        error.to_string().contains("opposing surface normals"),
        "unexpected error: {error}"
    );
}

#[test]
fn stencil_ghosts_round_trip_through_json() {
    // Constructors re-normalize unit vectors on load (last-ULP differences),
    // so round-trip fidelity is asserted on the classification field itself.
    let root = sphere_root();
    let corners = cap_triangle_clicks();

    let (ghost, _) = stencil_knife(&root, "polygon", &corners).expect("stencil");
    assert!(matches!(ghost.shape, Shape::Extrude(_)));
    let round_trip =
        ghost_from_json(&ghost_to_json(&ghost).expect("to json")).expect("from json");
    assert_eq!(
        std::mem::discriminant(&ghost.shape),
        std::mem::discriminant(&round_trip.shape)
    );
    for probe in sphere_samples() {
        let original = ghost.eval_point(probe);
        let reloaded = round_trip.eval_point(probe);
        assert!(
            (original - reloaded).abs() <= 1.0e-9,
            "field drifted at {probe:?}: {original} vs {reloaded}"
        );
    }
}
