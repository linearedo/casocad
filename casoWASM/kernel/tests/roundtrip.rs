//! Scene.json round-trip parity with the Python serializer.
//!
//! `casoWASM/tools/export_scene_goldens.py` loads scenes through the Python
//! load/save path and writes the resaved JSON; the Rust loader/saver must
//! produce semantically identical JSON (same records — key order free) from
//! the same inputs, including the legacy boundary-selector migration.

use caso_kernel::scene::{ScenePayload, SceneDocument, TagRef};
use caso_kernel::serialization::{load_scene_from_str, save_scene_to_string, scene_to_value};
use caso_kernel::vec3::vec3;
use serde_json::Value;

fn manifest_path(relative: &str) -> String {
    format!("{}/{}", env!("CARGO_MANIFEST_DIR"), relative)
}

fn load_json(path: &str) -> Value {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("cannot read {path}: {error}"));
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("bad JSON in {path}: {error}"))
}

#[test]
fn legacy_scene_resave_matches_python() {
    let source = std::fs::read_to_string(manifest_path("../../scene.json"))
        .expect("repo-root scene.json fixture");
    let document = load_scene_from_str(&source).expect("load legacy scene");
    let resaved = scene_to_value(&document).expect("save scene");
    let golden = load_json(&manifest_path("tests/goldens/scene_python_resave.json"));
    assert_eq!(
        resaved, golden,
        "Rust resave differs from Python resave of scene.json"
    );
}

#[test]
fn default_scene_save_matches_python() {
    let document = SceneDocument::default_scene().expect("default scene");
    let saved = scene_to_value(&document).expect("save default scene");
    let golden = load_json(&manifest_path("tests/goldens/default_python_resave.json"));
    assert_eq!(saved, golden, "default scene differs from Python");
}

#[test]
fn save_load_save_is_idempotent() {
    let source = std::fs::read_to_string(manifest_path("../../scene.json"))
        .expect("repo-root scene.json fixture");
    let document = load_scene_from_str(&source).expect("load");
    let first = save_scene_to_string(&document).expect("first save");
    let reloaded = load_scene_from_str(&first).expect("reload own output");
    let second = save_scene_to_string(&reloaded).expect("second save");
    assert_eq!(first, second, "save/load/save must be a fixed point");
}

#[test]
fn loaded_scene_evaluates_like_the_kernel() {
    let source = std::fs::read_to_string(manifest_path("../../scene.json"))
        .expect("repo-root scene.json fixture");
    let document = load_scene_from_str(&source).expect("load");
    let root_id = document.roots[0];
    let root = document.build_node(root_id).expect("build root node");
    assert_eq!(root.kind(), "difference");
    // von Kármán: box(3.2, 1.4, 0.9) minus cylinder(r 0.24, h 1.1) at origin.
    // At the origin we are inside the cylinder -> outside the difference.
    assert!(root.eval_point(vec3(0.0, 0.0, 0.0)) > 0.0);
    // Near the box corner interior, far from the cylinder -> inside.
    assert!(root.eval_point(vec3(1.2, 0.5, 0.0)) < 0.0);
    // Outside the box -> positive.
    assert!(root.eval_point(vec3(5.0, 0.0, 0.0)) > 0.0);
}

#[test]
fn legacy_selector_migrates_into_cut_chain() {
    let source = std::fs::read_to_string(manifest_path("../../scene.json"))
        .expect("repo-root scene.json fixture");
    let document = load_scene_from_str(&source).expect("load");
    // The two split regions carry the migrated ghost as their first cut.
    let with_cuts: Vec<_> = document
        .boundary_regions
        .iter()
        .filter(|region| !region.cuts.is_empty())
        .collect();
    assert_eq!(with_cuts.len(), 2);
    for region in with_cuts {
        assert_eq!(region.cuts[0].ghost.kind(), "placed_sdf_2d");
        assert_eq!(region.cuts[0].ghost.object_id, 0);
        assert!(region.selector_id.is_none(), "volume selector must migrate");
    }
    // The hidden selector node is dropped from the scene graph.
    assert!(document
        .live_ids()
        .iter()
        .all(|id| !SceneDocument::is_internal_scene_node(
            &document.object(*id).expect("live object").name
        )));
}

#[test]
fn document_operations_smoke() {
    let mut document = SceneDocument::new();
    let sphere = document.add_primitive("sphere", 1.0).expect("sphere");
    let box_id = document.add_primitive("box", 1.0).expect("box");
    assert_eq!(document.roots.len(), 2);

    // Move the sphere in place.
    let moved = document.move_object(sphere, vec3(0.5, 0.0, 0.0)).expect("move");
    assert_eq!(moved, sphere);
    let ScenePayload::Sphere(sphere_shape) = &document.object(sphere).expect("sphere").payload
    else {
        panic!("expected sphere payload");
    };
    assert_eq!(sphere_shape.center, vec3(0.5, 0.0, 0.0));

    // Subtract the sphere from the box.
    let combined = document
        .combine(box_id, sphere, "difference")
        .expect("combine");
    assert_eq!(document.roots, vec![combined]);
    let node = document.build_node(combined).expect("build");
    assert_eq!(node.kind(), "difference");

    // Undo snapshot restores the pre-combine state.
    let snapshot_before_wrap = document.snapshot();
    let wrapped = document.wrap_transform(combined, "translate").expect("wrap");
    assert_eq!(document.roots, vec![wrapped]);
    document = snapshot_before_wrap;
    assert_eq!(document.roots, vec![combined]);

    // Deleting the sphere collapses the difference to the surviving box.
    let deleted = document.delete(sphere);
    assert_eq!(deleted, 1);
    assert_eq!(document.roots, vec![box_id]);

    // Round-trip the result.
    let saved = save_scene_to_string(&document).expect("save");
    let reloaded = load_scene_from_str(&saved).expect("reload");
    assert_eq!(reloaded.roots.len(), 1);
    let rebuilt = reloaded.build_node(reloaded.roots[0]).expect("build");
    assert_eq!(rebuilt.kind(), "box");
}

#[test]
fn default_scene_matches_python_default() {
    let document = SceneDocument::default_scene().expect("default");
    assert_eq!(document.roots.len(), 1);
    let fluid = document.fluid_domain.as_ref().expect("fluid domain");
    assert_eq!(fluid.tags.len(), 2);
    assert!(matches!(fluid.tags[0], TagRef::Region(_)));
    let root = document.build_node(document.roots[0]).expect("root");
    assert_eq!(root.name, "von_karman_fluid");
    assert_eq!(root.kind(), "difference");
}
