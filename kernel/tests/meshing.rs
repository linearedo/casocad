//! Ports of `tests/test_mesh_api.py` — the public meshing API gate.

use caso_kernel::meshing::{
    load_meshable_domains_from_str, meshable_domains_from_document, model_from_document,
};
use caso_kernel::roles::DomainKind;
use caso_kernel::scene::{SceneDocument, ScenePayload};
use caso_kernel::sdf::node::RotationAxis;
use caso_kernel::serialization::save_scene_to_string;
use caso_kernel::vec3::vec3;

fn default_scene_json() -> String {
    let document = SceneDocument::default_scene().expect("default scene");
    save_scene_to_string(&document).expect("save default scene")
}

#[test]
fn load_meshable_domains_from_scene_json() {
    let domains = load_meshable_domains_from_str(&default_scene_json()).expect("domains");

    assert_eq!(domains.len(), 1);
    let domain = domains.get("von_karman_fluid").expect("by name");
    assert_eq!(domain.kind, DomainKind::Fluid);
    // Lookup is by NAME only — never by kind; the miss lists the names.
    let error = domains.get("fluid").expect_err("name-only lookup");
    assert!(error.to_string().contains("von_karman_fluid"));
    assert_eq!(domains.names(), vec!["von_karman_fluid".to_string()]);
    assert_eq!(domain.dimension, 3);
    // inlet + outlet are direction-only regions: addressable as regions,
    // with no cut-chain tag fields.
    assert_eq!(domain.boundary_regions.len(), 2);
    assert!(domain.boundary_tags.is_empty());

    let values = domain.domain_sdf(&[
        vec3(1.8, 0.0, 0.5),
        vec3(0.5, 0.0, 0.5),
        vec3(6.0, 4.0, 4.0),
    ]);
    assert!(
        values[0] > 0.0,
        "cylinder obstacle is carved out of the fluid"
    );
    assert!(values[1] < 0.0);
    assert!(values[2] > 0.0);
}

#[test]
fn kind_is_queried_explicitly_never_by_name() {
    let mut document = SceneDocument::new();
    let water = document.add_primitive("sphere", 1.0).expect("water");
    let air = document.add_primitive("sphere", 1.0).expect("air");
    if let ScenePayload::Sphere(sphere) = &mut document.object_mut(air).expect("air object").payload
    {
        sphere.center = vec3(3.0, 0.0, 0.0);
    }
    document.rename(water, "water").expect("rename");
    document.rename(air, "air").expect("rename");
    document
        .set_domain_root(water, DomainKind::Fluid)
        .expect("water domain");
    document
        .set_domain_root(air, DomainKind::Fluid)
        .expect("air domain");
    // Two fluid domains: name lookup works; the explicit kind query is the
    // only way to ask by kind ("fluid" is not a name here).
    let domains = meshable_domains_from_document(&document).expect("domains");
    assert_eq!(domains.get("water").expect("water").name, "water");
    assert_eq!(domains.by_kind(DomainKind::Fluid).len(), 2);
    let error = domains.get("fluid").expect_err("not a domain name");
    assert!(error.to_string().contains("water"));
    assert!(error.to_string().contains("air"));
}

#[test]
fn compile_gate_refuses_overlapping_domains() {
    let mut document = SceneDocument::new();
    let a = document.add_primitive("sphere", 1.0).expect("a");
    let b = document.add_primitive("sphere", 1.0).expect("b");
    if let ScenePayload::Sphere(sphere) = &mut document.object_mut(b).expect("b").payload {
        sphere.center = vec3(0.2, 0.0, 0.0);
    }
    document
        .set_domain_root(a, DomainKind::Fluid)
        .expect("a domain");
    document
        .set_domain_root(b, DomainKind::Solid)
        .expect("b domain");
    assert!(meshable_domains_from_document(&document).is_err());
}

#[test]
fn saved_solid_domain_is_loadable() {
    let mut document = SceneDocument::new();
    let handle = document.add_primitive("box", 1.0).expect("box");
    let name = document.object(handle).expect("box").name.clone();
    document
        .set_domain_root(handle, DomainKind::Solid)
        .expect("solid domain");
    let saved = save_scene_to_string(&document).expect("save");

    let domains = load_meshable_domains_from_str(&saved).expect("domains");
    assert_eq!(domains.len(), 1);
    let domain = domains.get(&name).expect("by name");
    assert_eq!(domain.name, name);
    assert_eq!(domain.kind, DomainKind::Solid);
}

#[test]
fn rotated_2d_domain_exposes_mesh_space() {
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
    assert_eq!(domain.dimension, 2);
    let space = domain.mesh_space().expect("space");
    let bounds = space.bounds();
    assert!((bounds[0] + 1.0).abs() < 1e-12);
    assert!((bounds[1] - 1.0).abs() < 1e-12);
    assert!((bounds[2] + 0.5).abs() < 1e-12);
    assert!((bounds[3] - 0.5).abs() < 1e-12);

    let center = space.point(0.0, 0.0);
    assert!(domain.domain_sdf(&[center])[0] < 0.0);
    assert!(space.sdf(0.0, 0.0) < 0.0);
    let local = space.coords(center);
    assert!(local.iter().all(|value| value.abs() < 1e-12));
}

#[test]
fn model_from_document_only_uses_declared_domains() {
    let mut document = SceneDocument::new();
    document.add_primitive("box", 1.0).expect("free object");
    let model = model_from_document(&document).expect("model");
    assert!(model.domains.is_empty(), "free objects are not Domains");
}

/// boundary_region_v2 §6: EVERY region is addressable and classifiable —
/// including direction-only ones the old contract silently dropped.
#[test]
fn boundary_regions_are_callable_from_mesher_scripts() {
    let mut document = SceneDocument::default_scene().expect("default scene");
    let mut box_id = 0;
    for (id, _parent) in document.walk() {
        if matches!(
            document.object(id).expect("object").payload,
            ScenePayload::Box3(_)
        ) {
            box_id = id;
        }
    }
    let whole = document
        .add_boundary_region(box_id, None, None, None)
        .expect("whole-surface region");
    let ghost = caso_kernel::Node::new(
        "knife",
        caso_kernel::Shape::Sphere(
            caso_kernel::sdf::primitives_3d::Sphere::new(vec3(0.0, 0.0, 0.5), 0.5).expect("sphere"),
        ),
    );
    let (inside_id, _outside_id) = document
        .split_boundary_region(whole, &ghost, None)
        .expect("split");
    document
        .boundary_regions
        .iter_mut()
        .find(|region| region.object_id == inside_id)
        .expect("inside region")
        .tag = Some("inlet".to_string());
    // Direction-only legacy regions must be callable too.
    let inlet_dir = document
        .add_boundary_region(box_id, Some(0), None, None)
        .expect("direction region");
    document
        .boundary_regions
        .iter_mut()
        .find(|region| region.object_id == inlet_dir)
        .expect("dir region")
        .name = "inlet".to_string();

    let saved = save_scene_to_string(&document).expect("save");
    let domains = load_meshable_domains_from_str(&saved).expect("domains");
    let domain = domains.get("von_karman_fluid").expect("fluid domain");
    let names: Vec<&str> = domain
        .boundary_regions
        .iter()
        .map(|region| region.name.as_str())
        .collect();
    assert!(names.iter().any(|name| name.contains("inside")));
    assert!(names.contains(&"inlet"));

    let jet = domain
        .boundary_regions
        .iter()
        .find(|region| region.tag.as_deref() == Some("inlet"))
        .expect("tagged region");
    let face_points = [
        vec3(0.0, 0.0, 0.5), // centre of the -X face: inside the knife
        vec3(0.0, 1.3, 0.9), // far corner of the -X face: outside the knife
        vec3(4.5, 0.0, 0.5), // +X face: wrong side of the box entirely
    ];
    let mask = jet.contains(&face_points).expect("contains");
    assert_eq!(mask, vec![true, false, false]);
    // owner_sdf is the exact field of the generating surface
    assert!(jet.owner_sdf(&[vec3(0.0, 0.0, 0.5)])[0].abs() < 1e-9);
    // selector_sdf is negative inside the kept knife-half
    let selector = jet.selector_sdf(&face_points).expect("selector field");
    assert!(selector[0] < 0.0);
    assert!(selector[1] > 0.0);

    let legacy = domain
        .boundary_regions
        .iter()
        .find(|region| region.name == "inlet")
        .expect("direction region");
    assert!(legacy.selector_sdf(&face_points).is_none());
    let legacy_mask = legacy.contains(&face_points).expect("contains");
    assert_eq!(legacy_mask, vec![true, true, false]);
}
