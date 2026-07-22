//! Scene.json round-trip goldens: `default_scene_resave.json` pins the saver's
//! output for the built-in default scene (same records — key order free), and
//! save/load/save must be a fixed point.

use caso_kernel::scene::{SceneDocument, ScenePayload, TagRef};
use caso_kernel::serialization::{load_scene_from_str, save_scene_to_string, scene_to_value};
use caso_kernel::vec3::vec3;
use serde_json::Value;

fn manifest_path(relative: &str) -> String {
    format!("{}/{}", env!("CARGO_MANIFEST_DIR"), relative)
}

fn load_json(path: &str) -> Value {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|error| panic!("cannot read {path}: {error}"));
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("bad JSON in {path}: {error}"))
}

#[test]
fn default_scene_save_matches_golden() {
    let document = SceneDocument::default_scene().expect("default scene");
    let saved = scene_to_value(&document).expect("save default scene");
    let golden = load_json(&manifest_path("tests/goldens/default_scene_resave.json"));
    assert_eq!(saved, golden, "default scene differs from its saved golden");
}

#[test]
fn save_load_save_is_idempotent() {
    let document = SceneDocument::default_scene().expect("default scene");
    let first = save_scene_to_string(&document).expect("first save");
    let reloaded = load_scene_from_str(&first).expect("reload own output");
    let second = save_scene_to_string(&reloaded).expect("second save");
    assert_eq!(first, second, "save/load/save must be a fixed point");
}

#[test]
fn meshing_controls_round_trip_and_old_scenes_default_empty() {
    let old = save_scene_to_string(&SceneDocument::default_scene().unwrap()).unwrap();
    let old_loaded = load_scene_from_str(&old).unwrap();
    assert!(old_loaded.meshing.control_script.is_empty());

    let mut document = SceneDocument::default_scene().unwrap();
    document.meshing.control_script = "controls.refinement_box(\"sea\", #{});".into();
    document.meshing.options.cells_2d = 72;
    let loaded = load_scene_from_str(&save_scene_to_string(&document).unwrap()).unwrap();
    assert_eq!(loaded.meshing, document.meshing);
}

#[test]
fn loaded_scene_evaluates_like_the_kernel() {
    let source = save_scene_to_string(&SceneDocument::default_scene().expect("default scene"))
        .expect("save default scene");
    let document = load_scene_from_str(&source).expect("load");
    let root_id = document.roots[0];
    let root = document.build_node(root_id).expect("build root node");
    assert_eq!(root.kind(), "difference");
    // von Kármán: channel x ∈ [0, 4.5] minus Y-axis cylinder at (1.8, 0, 0.5).
    // On the cylinder axis we are inside the obstacle -> outside the fluid.
    assert!(root.eval_point(vec3(1.8, 0.0, 0.5)) > 0.0);
    // Upstream of the obstacle, inside the channel -> inside.
    assert!(root.eval_point(vec3(0.5, 0.0, 0.5)) < 0.0);
    // Outside the box -> positive.
    assert!(root.eval_point(vec3(5.0, 0.0, 0.5)) > 0.0);
}

#[test]
fn document_operations_smoke() {
    let mut document = SceneDocument::new();
    let sphere = document.add_primitive("sphere", 1.0).expect("sphere");
    let box_id = document.add_primitive("box", 1.0).expect("box");
    assert_eq!(document.roots.len(), 2);

    // Move the sphere in place.
    let moved = document
        .move_object(sphere, vec3(0.5, 0.0, 0.0))
        .expect("move");
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
    let wrapped = document
        .wrap_transform(combined, "translate")
        .expect("wrap");
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
fn default_scene_has_expected_structure() {
    let document = SceneDocument::default_scene().expect("default");
    assert_eq!(document.roots.len(), 1);
    let fluid = document.fluid_domain.as_ref().expect("fluid domain");
    assert_eq!(fluid.tags.len(), 2);
    assert!(matches!(fluid.tags[0], TagRef::Region(_)));
    let root = document.build_node(document.roots[0]).expect("root");
    assert_eq!(root.name, "von_karman_fluid");
    assert_eq!(root.kind(), "difference");
}
