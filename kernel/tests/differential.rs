//! Differential-query gates (`design_docs/meshing_toolkit.md` §4): normals,
//! interior-seeded projection (positive starts refused), curvature stencils.
//! Fixture: the default von Kármán scene — fluid = Difference(flow box,
//! cylinder obstacle), box x in [0, 4.5], y in [-1.5, 1.5], z in [0, 1],
//! cylinder r 0.15 at (1.8, 0, 0.5) with its axis along +Y.

use caso_kernel::differential::{batch_normals, differential_steps, mean_curvature};
use caso_kernel::frame::IDENTITY_FRAME;
use caso_kernel::meshing::{load_meshable_domains_from_str, meshable_domains_from_document};
use caso_kernel::roles::DomainKind;
use caso_kernel::scene::SceneDocument;
use caso_kernel::sdf::node::{Node, RotationAxis, Shape};
use caso_kernel::sdf::primitives_3d::{Box3, Sphere};
use caso_kernel::serialization::save_scene_to_string;
use caso_kernel::vec3::{vec3, Vec3};

fn fluid_domain() -> caso_kernel::meshing::MeshableDomains {
    let document = SceneDocument::default_scene().expect("default scene");
    let saved = save_scene_to_string(&document).expect("save");
    load_meshable_domains_from_str(&saved).expect("domains")
}

fn sphere_node(radius: f64) -> Node {
    Node::new(
        "ball",
        Shape::Sphere(Sphere::new(Vec3::ZERO, radius).expect("sphere")),
    )
}

#[test]
fn normals_match_analytic_on_sphere_and_box_faces() {
    let ball = sphere_node(0.5);
    let (normal_step, _, _) = differential_steps(&ball);
    let surface: Vec<Vec3> = (0..8)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / 8.0;
            vec3(0.5 * angle.cos(), 0.5 * angle.sin(), 0.0)
        })
        .collect();
    for (point, normal) in surface.iter().zip(batch_normals(&ball, &surface, normal_step)) {
        let radial = *point * 2.0; // point / radius
        assert!((normal - radial).length() < 1e-9, "radial on the sphere");
    }

    let cube = Node::new(
        "cube",
        Shape::Box3(Box3::new(Vec3::ZERO, vec3(0.5, 0.5, 0.5), IDENTITY_FRAME).expect("box")),
    );
    let (normal_step, _, _) = differential_steps(&cube);
    let face_point = [vec3(0.5, 0.1, 0.2)];
    let normal = batch_normals(&cube, &face_point, normal_step)[0];
    assert!((normal - Vec3::X).length() < 1e-9, "+X face normal");
}

#[test]
fn projection_lands_on_surface_from_interior() {
    let domains = fluid_domain();
    let domain = domains.get("von_karman_fluid").expect("fluid");
    let (_, _, zero_band) = differential_steps(domain.region_node());
    // Unique nearest wall for each start: the -X face (0.3 away) and the
    // cylinder side wall (0.15 away).
    let starts = [vec3(0.3, 0.0, 0.5), vec3(1.5, 0.0, 0.5)];
    let start_values = domain.domain_sdf(&starts);
    let projections = domain.project_to_boundary(&starts).expect("projection");
    for (projection, start_value) in projections.iter().zip(start_values) {
        assert!(projection.converged);
        assert!(projection.residual.abs() <= zero_band, "on the wall");
        assert!(
            (projection.distance_moved - start_value.abs()).abs() < 1e-9,
            "the interior value is the exact travel distance"
        );
    }
    assert!(projections[0].point.x.abs() < 1e-8, "landed on the -X face");
    let radial = vec3(
        projections[1].point.x - 1.8,
        0.0,
        projections[1].point.z - 0.5,
    );
    assert!(
        (radial.length() - 0.15).abs() < 1e-8,
        "landed on the cylinder wall"
    );
}

#[test]
fn positive_start_is_refused_without_iterating() {
    let domains = fluid_domain();
    let domain = domains.get("von_karman_fluid").expect("fluid");
    let outside = vec3(-1.0, 0.0, 0.5);
    let projection = &domain.project_to_boundary(&[outside]).expect("projection")[0];
    assert!(!projection.converged);
    assert_eq!(projection.distance_moved, 0.0, "no iteration on a positive start");
    assert_eq!(projection.point, outside);
    assert!(projection.residual > 0.0);
}

#[test]
fn projection_reports_nonconvergence_on_equidistant_seam() {
    let domains = fluid_domain();
    let domain = domains.get("von_karman_fluid").expect("fluid");
    // Exactly midway between the parallel z = 0 and z = 1 walls, away from
    // every other wall: the gradient vanishes, no step can improve.
    let seam = vec3(3.5, 0.0, 0.5);
    let projection = &domain.project_to_boundary(&[seam]).expect("projection")[0];
    assert!(!projection.converged, "honest non-convergence at the seam");
    assert!((projection.residual + 0.5).abs() < 1e-12);
}

#[test]
fn projection_2d_stays_in_plane() {
    let mut document = SceneDocument::new();
    let section = document
        .add_primitive_from_drag("rectangle", vec3(0.0, 0.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
        .expect("rectangle");
    document
        .rotate_object(section, RotationAxis::Z, 90.0, Some(vec3(0.0, 0.0, 0.0)))
        .expect("rotate");
    document.rename(section, "fluid").expect("rename");
    document
        .set_domain_root(section, DomainKind::Fluid)
        .expect("domain");
    let domains = meshable_domains_from_document(&document).expect("domains");
    let domain = domains.get("fluid").expect("fluid");
    let space = domain.mesh_space().expect("space");

    // Interior start 0.2 from the b = +0.5 edge (every other edge farther).
    let start = space.point(0.0, 0.3);
    let projection = &domain.project_to_boundary(&[start]).expect("projection")[0];
    assert!(projection.converged);
    assert!(
        (projection.distance_moved - 0.2).abs() < 1e-9,
        "exact interior travel distance"
    );
    let local = space.coords(projection.point);
    assert!(local[2].abs() < 1e-12, "never drifts off the plane");
    assert!((local[1] - 0.5).abs() < 1e-8, "landed on the b = +0.5 edge");
}

#[test]
fn curvature_sphere_is_inverse_radius() {
    let ball = sphere_node(0.5);
    let (_, curvature_step, _) = differential_steps(&ball);
    let on_wall = vec3(0.5, 0.0, 0.0);
    let curvature = mean_curvature(&ball, on_wall, curvature_step);
    assert!(
        ((curvature - 2.0) / 2.0).abs() < 1e-5,
        "H = 1/r on the sphere, got {curvature}"
    );
}

#[test]
fn curvature_plane_face_is_zero() {
    let cube = Node::new(
        "cube",
        Shape::Box3(Box3::new(Vec3::ZERO, vec3(0.5, 0.5, 0.5), IDENTITY_FRAME).expect("box")),
    );
    let (_, curvature_step, _) = differential_steps(&cube);
    let curvature = mean_curvature(&cube, vec3(0.5, 0.1, 0.2), curvature_step);
    assert!(curvature.abs() < 1e-6, "flat face, got {curvature}");
}
