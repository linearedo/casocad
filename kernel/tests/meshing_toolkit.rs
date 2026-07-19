//! Toolkit gates (`design_docs/meshing_toolkit.md` §5-§6): total boundary
//! classification with owner attribution and precedence, and domain
//! interfaces between nested marked domains.

use caso_kernel::meshing::meshable_domains_from_document;
use caso_kernel::roles::DomainKind;
use caso_kernel::scene::{ObjectId, SceneDocument, ScenePayload};
use caso_kernel::sdf::node::{Node, Shape};
use caso_kernel::sdf::primitives_3d::Sphere;
use caso_kernel::sdf::solid_from_2d::RevolveAxis;
use caso_kernel::vec3::{vec3, Vec3};

fn default_scene_ids() -> (SceneDocument, ObjectId, ObjectId) {
    let document = SceneDocument::default_scene().expect("default scene");
    let mut box_id = 0;
    let mut cylinder_id = 0;
    for (id, _parent) in document.walk() {
        match document.object(id).expect("object").payload {
            ScenePayload::Box3(_) => box_id = id,
            ScenePayload::Cylinder(_) => cylinder_id = id,
            _ => {}
        }
    }
    assert!(box_id != 0 && cylinder_id != 0);
    (document, box_id, cylinder_id)
}

#[test]
fn classification_covers_tagged_and_untagged_boundary() {
    let (document, box_id, cylinder_id) = default_scene_ids();
    let domains = meshable_domains_from_document(&document).expect("domains");
    let domain = domains.get("fluid").expect("fluid");

    let points = [
        vec3(0.0, 0.0, 0.5),   // -X face centre: the inlet direction region
        vec3(1.95, 0.0, 0.5),  // cylinder wall: untagged boundary
        vec3(0.5, 0.0, 0.5),   // interior
    ];
    let classes = domain.classify_boundary(&points, None).expect("classify");

    assert!(classes[0].on_boundary);
    assert_eq!(classes[0].owner_object_id, box_id);
    let winner = classes[0].region_index.expect("inlet claims the -X face");
    assert_eq!(domain.boundary_regions[winner].name, "inlet");

    assert!(classes[1].on_boundary);
    assert_eq!(
        classes[1].owner_object_id, cylinder_id,
        "untagged boundary keeps its owner leaf for default patches"
    );
    assert_eq!(classes[1].region_index, None);

    assert!(!classes[2].on_boundary);
    assert_eq!(classes[2].region_index, None);
}

#[test]
fn explicit_tolerance_selects_the_band() {
    let (document, _box_id, cylinder_id) = default_scene_ids();
    let domains = meshable_domains_from_document(&document).expect("domains");
    let domain = domains.get("fluid").expect("fluid");

    // A centroid-like sample 1 mm off the cylinder wall (the sagitta case).
    let sample = [vec3(1.949, 0.0, 0.5)];
    let default_band = domain.classify_boundary(&sample, None).expect("classify");
    assert!(default_band[0].on_boundary, "default band accepts the sample");
    assert_eq!(default_band[0].owner_object_id, cylinder_id);

    let tight = 1e-9 * domain.bounds.diagonal();
    let tight_band = domain
        .classify_boundary(&sample, Some(tight))
        .expect("classify");
    assert!(
        !tight_band[0].on_boundary,
        "the projected-vertex band rejects an unprojected sample"
    );

    // A point exactly on the wall passes both bands.
    let on_wall = [vec3(1.95, 0.0, 0.5)];
    assert!(domain.classify_boundary(&on_wall, Some(tight)).expect("classify")[0].on_boundary);
}

fn region_name(
    domain: &caso_kernel::meshing::MeshableDomain,
    index: Option<usize>,
) -> &str {
    &domain.boundary_regions[index.expect("a region wins")].name
}

#[test]
fn classification_precedence_is_cuts_then_scope_then_creation_order() {
    let mut document = SceneDocument::new();
    let box_id = document.add_primitive("box", 1.0).expect("box");
    document
        .set_domain_root(box_id, DomainKind::Fluid)
        .expect("fluid domain");

    // Created first: a +X direction-scoped region (direction 1 = +X).
    let dir = document
        .add_boundary_region(box_id, Some(1), None, None)
        .expect("direction region");
    document
        .boundary_regions
        .iter_mut()
        .find(|region| region.object_id == dir)
        .expect("dir")
        .name = "outlet_face".to_string();
    // Then two whole-surface regions.
    let whole_a = document
        .add_boundary_region(box_id, None, None, None)
        .expect("whole a");
    document
        .boundary_regions
        .iter_mut()
        .find(|region| region.object_id == whole_a)
        .expect("a")
        .name = "whole_a".to_string();
    let whole_b = document
        .add_boundary_region(box_id, None, None, None)
        .expect("whole b");
    document
        .boundary_regions
        .iter_mut()
        .find(|region| region.object_id == whole_b)
        .expect("b")
        .name = "whole_b".to_string();

    let domains = meshable_domains_from_document(&document).expect("domains");
    let domain = domains.get("fluid").expect("fluid");
    let bounds = &domain.bounds;
    let center_y = (bounds.y_min + bounds.y_max) / 2.0;
    let center_z = (bounds.z_min + bounds.z_max) / 2.0;
    let minus_x = vec3(bounds.x_min, center_y, center_z);
    let plus_x = vec3(bounds.x_max, center_y, center_z);

    // -X face: only the whole regions match — later creation wins the tie.
    let classes = domain
        .classify_boundary(&[minus_x, plus_x], None)
        .expect("classify");
    assert_eq!(region_name(domain, classes[0].region_index), "whole_b");
    // +X face: the scoped region beats both whole-surface ones despite
    // being created first.
    assert_eq!(region_name(domain, classes[1].region_index), "outlet_face");
    // Multi-label view returns every match.
    let matches = domain
        .regions_containing(&[plus_x], None)
        .expect("containing");
    assert_eq!(matches[0].len(), 3);

    // Split whole_b with a knife around the -X face centre: the cut region
    // (1 cut) now beats every 0-cut region there.
    let ghost = Node::new(
        "knife",
        Shape::Sphere(Sphere::new(minus_x, 0.2).expect("sphere")),
    );
    document
        .split_boundary_region(whole_b, &ghost, None)
        .expect("split");
    let domains = meshable_domains_from_document(&document).expect("domains");
    let domain = domains.get("fluid").expect("fluid");
    let classes = domain.classify_boundary(&[minus_x], None).expect("classify");
    assert!(
        region_name(domain, classes[0].region_index).contains("inside"),
        "the knife-cut region is the most specific"
    );
}

/// Water box, solid spherical shell, gas core: `water = box − shell`,
/// `shell = outer − gas`, `gas = inner sphere` — all marked. Interfaces are
/// the directly nested pairs only.
fn nested_fixture() -> SceneDocument {
    let mut document = SceneDocument::new();
    let boxy = document.add_primitive("box", 4.0).expect("box");
    let outer = document.add_primitive("sphere", 1.0).expect("outer");
    if let ScenePayload::Sphere(sphere) = &mut document.object_mut(outer).expect("outer").payload {
        sphere.radius = 0.4;
    }
    let gas = document.add_primitive("sphere", 1.0).expect("gas");
    if let ScenePayload::Sphere(sphere) = &mut document.object_mut(gas).expect("gas").payload {
        sphere.radius = 0.2;
    }
    let shell = document.combine(outer, gas, "difference").expect("shell");
    let water = document.combine(boxy, shell, "difference").expect("water");
    document.rename(gas, "gas").expect("rename");
    document.rename(shell, "shell").expect("rename");
    document.rename(water, "water").expect("rename");
    document
        .set_domain_root(water, DomainKind::Fluid)
        .expect("water domain");
    document
        .set_domain_root(shell, DomainKind::Solid)
        .expect("shell domain");
    document
        .set_domain_root(gas, DomainKind::Fluid)
        .expect("gas domain");
    document
}

fn sphere_samples(radius: f64, count: usize) -> Vec<Vec3> {
    (0..count)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (count as f64);
            vec3(radius * angle.cos(), radius * angle.sin(), 0.0)
        })
        .collect()
}

#[test]
fn nested_domains_expose_directly_nested_interfaces_only() {
    let document = nested_fixture();
    let domains = meshable_domains_from_document(&document).expect("domains");

    assert_eq!(domains.interfaces().len(), 2);
    let water_shell = domains
        .interface_between("water", "shell")
        .expect("water<->shell");
    assert_eq!(water_shell.domain_b, "shell");
    // Order-independent lookup finds the same interface.
    assert!(domains.interface_between("shell", "water").is_ok());
    domains
        .interface_between("shell", "gas")
        .expect("shell<->gas");
    // Grand-parent pair is NOT an interface, and the error lists what is.
    let error = domains
        .interface_between("water", "gas")
        .expect_err("not directly nested");
    assert!(error.to_string().contains("water<->shell"));
    assert_eq!(domains.interfaces_of("shell").len(), 2);

    // The water<->shell wall is the outer sphere's zero set, clipped to the
    // contact area; the box wall is not on it.
    let on_wall = sphere_samples(0.4, 12);
    assert!(water_shell.contains(&on_wall).iter().all(|hit| *hit));
    let water = domains.get("water").expect("water");
    let box_face = vec3(water.bounds.x_max, 0.0, 0.0);
    assert!(!water_shell.contains(&[box_face])[0]);

    // Additive-base identity: interface samples lie on BOTH region
    // boundaries at the tight band (same node, same floats).
    let shell_gas = domains.interface_between("shell", "gas").expect("pair");
    let gas_wall = sphere_samples(0.2, 12);
    let tight = 1e-9 * water.bounds.diagonal();
    for value in domains.get("shell").expect("shell").domain_sdf(&gas_wall) {
        assert!(value.abs() <= tight, "shell boundary at the gas wall");
    }
    for value in domains.get("gas").expect("gas").domain_sdf(&gas_wall) {
        assert!(value.abs() <= tight, "gas boundary at the gas wall");
    }
    assert!(shell_gas.contains(&gas_wall).iter().all(|hit| *hit));
}

#[test]
fn interface_project_uses_interior_fields_only() {
    let document = nested_fixture();
    let domains = meshable_domains_from_document(&document).expect("domains");
    let shell_gas = domains.interface_between("shell", "gas").expect("pair");
    let tight = 1e-9 * domains.get("water").expect("water").bounds.diagonal();

    // From inside gas (inner side) and from inside the shell (outer side):
    // both land exactly on the shared wall.
    let starts = [vec3(0.15, 0.0, 0.0), vec3(0.25, 0.0, 0.0)];
    for projection in shell_gas.project(&starts) {
        assert!(projection.converged);
        assert!((projection.point.length() - 0.2).abs() <= tight);
        assert!(shell_gas.contains(&[projection.point])[0]);
    }

    // From neither interior (a water point): refused, no iteration.
    let refused = &shell_gas.project(&[vec3(0.7, 0.0, 0.0)])[0];
    assert!(!refused.converged);
    assert_eq!(refused.distance_moved, 0.0);
}

/// 2D: rectangle sheet (fluid) minus an ellipse (solid). The coplanar
/// boolean flattens into ONE Placed2D node — the subtracted operand
/// survives only in `sources`, yet stays a live nested domain.
fn nested_2d_fixture() -> (SceneDocument, ObjectId, ObjectId) {
    let mut document = SceneDocument::new();
    let rect = document
        .add_primitive_from_drag("rectangle", vec3(-2.0, -1.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
        .expect("rectangle");
    let ellipse = document
        .add_primitive_from_drag("ellipse", vec3(0.1, -0.2, 0.0), vec3(0.9, 0.2, 0.0), 1.0)
        .expect("ellipse");
    document.rename(ellipse, "ball").expect("rename");
    document
        .set_domain_root(ellipse, DomainKind::Solid)
        .expect("solid mark");
    let combined = document
        .combine(rect, ellipse, "difference")
        .expect("difference");
    document.rename(combined, "fluid").expect("rename");
    document
        .set_domain_root(combined, DomainKind::Fluid)
        .expect("fluid mark");
    (document, combined, ellipse)
}

#[test]
fn nested_2d_solid_meshes_and_pairs_an_interface() {
    // Ellipse from the drag: center (0.5, 0), semi-axes (0.4, 0.2).
    let (document, _combined, ellipse_id) = nested_2d_fixture();
    let domains = meshable_domains_from_document(&document).expect("domains");

    let fluid = domains.get("fluid").expect("fluid");
    let solid = domains.get("ball").expect("ball");
    assert_eq!(fluid.dimension, 2);
    assert_eq!(solid.dimension, 2);

    // The domains partition the sheet: the ellipse interior belongs to the
    // solid and is a hole in the fluid (sign checks only).
    let inside_ellipse = [vec3(0.5, 0.0, 0.0)];
    assert!(solid.domain_sdf(&inside_ellipse)[0] < 0.0);
    assert!(fluid.domain_sdf(&inside_ellipse)[0] > 0.0);
    let inside_fluid = [vec3(-1.0, 0.0, 0.0)];
    assert!(fluid.domain_sdf(&inside_fluid)[0] < 0.0);
    assert!(solid.domain_sdf(&inside_fluid)[0] > 0.0);

    // The interface is the ellipse outline; the rect edge is not on it.
    let interface = domains.interface_between("fluid", "ball").expect("pair");
    let on_outline = [
        vec3(0.9, 0.0, 0.0),
        vec3(0.1, 0.0, 0.0),
        vec3(0.5, 0.2, 0.0),
    ];
    assert!(interface.contains(&on_outline).iter().all(|hit| *hit));
    assert!(!interface.contains(&[vec3(-2.0, 0.0, 0.0)])[0]);

    // Interior-only projection from both sides lands on the outline.
    let starts = [vec3(0.7, 0.0, 0.0), vec3(1.2, 0.0, 0.0)];
    for projection in interface.project(&starts) {
        assert!(projection.converged);
        assert!(interface.contains(&[projection.point])[0]);
    }
    // Outside both interiors: refused without iterating.
    let refused = &interface.project(&[vec3(3.0, 0.0, 0.0)])[0];
    assert!(!refused.converged);
    assert_eq!(refused.distance_moved, 0.0);

    // The hole boundary keeps the ellipse leaf as its owner.
    let classes = fluid
        .classify_boundary(&[vec3(0.9, 0.0, 0.0)], None)
        .expect("classify");
    assert!(classes[0].on_boundary);
    assert_eq!(classes[0].owner_object_id, ellipse_id);
}

#[test]
fn subtracted_domain_roots_use_difference_parity() {
    // 2D flattened difference: the ellipse is a hole in the rendered sheet.
    let (document, _combined, ellipse_id) = nested_2d_fixture();
    assert_eq!(document.subtracted_domain_roots(), vec![ellipse_id]);

    // 3D water/shell/gas: shell (one difference-right crossing) is a hole;
    // gas (two crossings) re-enters the rendered volume.
    let document = nested_fixture();
    let mut shell_id = None;
    for (id, _parent) in document.walk() {
        if document.object(id).expect("object").name == "shell" {
            shell_id = Some(id);
        }
    }
    assert_eq!(
        document.subtracted_domain_roots(),
        vec![shell_id.expect("shell id")]
    );
}

#[test]
fn embedded_2d_source_follows_the_combined_placement() {
    let (mut document, combined, ellipse_id) = nested_2d_fixture();
    document
        .move_object(combined, vec3(0.5, 0.25, 0.0))
        .expect("move");
    let embedded = document.embedded_node(ellipse_id).expect("embedded");
    let center = embedded.bounding_box().expect("bounds").center();
    assert!((center - vec3(1.0, 0.25, 0.0)).length() < 1e-9);
}

#[test]
fn consumed_extrude_section_reports_a_meshing_error() {
    let mut document = SceneDocument::new();
    let section = document
        .add_primitive_from_drag("circle", vec3(-0.5, -0.5, 0.0), vec3(0.5, 0.5, 0.0), 1.0)
        .expect("circle");
    document.rename(section, "profile").expect("rename");
    document
        .solid_from_2d(section, "extrude", Some(1.0), RevolveAxis::U, None, None, None, 360.0)
        .expect("extrude");
    document
        .set_domain_root(section, DomainKind::Fluid)
        .expect("mark section");
    let error = meshable_domains_from_document(&document).expect_err("consumed section");
    let message = error.to_string();
    assert!(message.contains("profile"), "names the domain: {message}");
    assert!(message.contains("consumed"), "names the cause: {message}");
}

#[test]
fn sibling_domains_have_no_interface() {
    let mut document = SceneDocument::new();
    let water = document.add_primitive("sphere", 1.0).expect("water");
    let air = document.add_primitive("sphere", 1.0).expect("air");
    if let ScenePayload::Sphere(sphere) = &mut document.object_mut(air).expect("air").payload {
        sphere.center = vec3(3.0, 0.0, 0.0);
    }
    document
        .set_domain_root(water, DomainKind::Fluid)
        .expect("water domain");
    document
        .set_domain_root(air, DomainKind::Solid)
        .expect("air domain");
    let domains = meshable_domains_from_document(&document).expect("domains");
    assert!(domains.interfaces().is_empty());
    let error = domains.interface_between("water", "air").expect_err("none");
    assert!(error.to_string().contains("none"));
}
