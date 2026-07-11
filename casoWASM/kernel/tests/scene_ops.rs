//! Tests for the interactive scene ops behind the viewport tools:
//! drag-sized creation, point-shape placement, duplicate/paste — plus the
//! Phase 5 acceptance workflow (draw, subtract, set domain, save/reload).

use caso_kernel::scene::{SceneDocument, ScenePayload};
use caso_kernel::sdf::primitives_2d::Profile2D;
use caso_kernel::serialization::{load_scene_from_str, save_scene_to_string};
use caso_kernel::vec3::vec3;

#[test]
fn drag_creates_box_sized_by_plane_extents() {
    let mut document = SceneDocument::new();
    let id = document
        .add_primitive_from_drag("box", vec3(0.0, 0.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
        .expect("box from drag");
    match &document.object(id).expect("object").payload {
        ScenePayload::Box3(shape) => {
            assert_eq!(shape.center, vec3(1.0, 0.5, 0.0));
            // Planar drag: the fallback fills the third axis with the max extent.
            assert_eq!(shape.half_size, vec3(1.0, 0.5, 1.0));
        }
        other => panic!("expected Box3, got {other:?}"),
    }
}

#[test]
fn drag_creates_circle_section_with_planar_radius() {
    let mut document = SceneDocument::new();
    let id = document
        .add_primitive_from_drag("circle", vec3(0.0, 0.0, 0.0), vec3(0.6, 0.8, 0.0), 1.0)
        .expect("circle from drag");
    match &document.object(id).expect("object").payload {
        ScenePayload::Placed2D { profile, origin, .. } => {
            assert_eq!(*origin, vec3(0.3, 0.4, 0.0));
            match profile {
                Profile2D::Circle { radius, .. } => assert!((radius - 0.5).abs() < 1e-12),
                other => panic!("expected Circle, got {other:?}"),
            }
        }
        other => panic!("expected Placed2D, got {other:?}"),
    }
}

#[test]
fn drag_creates_segment_along_direction() {
    let mut document = SceneDocument::new();
    let id = document
        .add_primitive_from_drag("segment", vec3(0.0, 0.0, 0.0), vec3(3.0, 4.0, 0.0), 1.0)
        .expect("segment from drag");
    match &document.object(id).expect("object").payload {
        ScenePayload::Placed1D { origin, axis_u, .. } => {
            assert_eq!(*origin, vec3(1.5, 2.0, 0.0));
            assert!((axis_u.x - 0.6).abs() < 1e-12 && (axis_u.y - 0.8).abs() < 1e-12);
        }
        other => panic!("expected Placed1D, got {other:?}"),
    }
}

#[test]
fn degenerate_drag_falls_back_to_scaled_minimums() {
    let mut document = SceneDocument::new();
    let scale = 0.001; // millimeter working unit
    let id = document
        .add_primitive_from_drag("sphere", vec3(1.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), scale)
        .expect("degenerate sphere");
    match &document.object(id).expect("object").payload {
        ScenePayload::Sphere(sphere) => {
            assert!((sphere.radius - 0.05 * scale).abs() < 1e-15);
        }
        other => panic!("expected Sphere, got {other:?}"),
    }
}

#[test]
fn point_shape_projects_world_points_to_plane_locals() {
    let mut document = SceneDocument::new();
    let points = [
        vec3(1.0, 1.0, 0.0),
        vec3(2.0, 1.0, 0.0),
        vec3(2.0, 2.0, 0.0),
    ];
    let id = document
        .add_point_shape_from_world_points("polygon", &points, "xy")
        .expect("polygon from points");
    match &document.object(id).expect("object").payload {
        ScenePayload::Placed2D { profile, origin, .. } => {
            assert_eq!(*origin, vec3(1.0, 1.0, 0.0));
            match profile {
                Profile2D::Polygon { points } => {
                    assert_eq!(points[0], [0.0, 0.0]);
                    assert_eq!(points[1], [1.0, 0.0]);
                    assert_eq!(points[2], [1.0, 1.0]);
                }
                other => panic!("expected Polygon, got {other:?}"),
            }
        }
        other => panic!("expected Placed2D, got {other:?}"),
    }
    assert!(document
        .add_point_shape_from_world_points("quadratic_bezier_curve", &points[..2], "xy")
        .is_err());
}

#[test]
fn duplicate_offsets_and_renames_a_deep_copy() {
    let mut document = SceneDocument::default_scene().expect("default scene");
    let root = document.roots[0];
    let before = document.live_ids().len();
    let pasted = document
        .duplicate_nodes(&[root], vec3(0.1, 0.1, 0.0))
        .expect("duplicate");
    assert_eq!(pasted.len(), 1);
    let copy = document.object(pasted[0]).expect("copy");
    assert!(copy.name.ends_with(" copy"));
    // The Difference root cannot translate in place, so the paste gains a
    // Translate wrapper on top of the deep copy (same as Python).
    assert_eq!(document.live_ids().len(), before * 2 + 1);
    assert!(matches!(copy.payload, ScenePayload::Translate { .. }));
    // Selecting both original + a nested child only copies the root once.
    let child = document.object(root).expect("root").payload.children()[0];
    let pasted = document
        .duplicate_nodes(&[root, child], vec3(0.1, 0.1, 0.0))
        .expect("duplicate with descendant");
    assert_eq!(pasted.len(), 1);
}

#[test]
fn clipboard_import_copies_across_documents() {
    let source = SceneDocument::default_scene().expect("default scene");
    let root = source.roots[0];
    let mut target = SceneDocument::new();
    let imported = target
        .import_subtree(&source, root, vec3(0.2, 0.0, 0.0))
        .expect("import");
    assert!(target.object(imported).expect("imported").name.ends_with(" copy"));
    // Translate wrapper on top of the imported Difference subtree.
    assert_eq!(target.live_ids().len(), source.live_ids().len() + 1);
    target.build_node(imported).expect("imported subtree builds");
}

/// Phase 5 acceptance gate: draw box, draw cylinder, subtract, set fluid
/// domain, save — and the file reloads to an identical document.
#[test]
fn acceptance_workflow_saves_and_reloads_identically() {
    let mut document = SceneDocument::new();
    let box_id = document
        .add_primitive_from_drag("box", vec3(-2.0, -1.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
        .expect("draw box");
    let cylinder_id = document
        .add_primitive_from_drag(
            "cylinder",
            vec3(-0.3, -0.3, 0.0),
            vec3(0.3, 0.3, 0.0),
            1.0,
        )
        .expect("draw cylinder");
    let result = document
        .combine(box_id, cylinder_id, "difference")
        .expect("subtract");
    document
        .set_domain_root(result, caso_kernel::roles::DomainKind::Fluid)
        .expect("set fluid domain");

    let saved = save_scene_to_string(&document).expect("save");
    let reloaded = load_scene_from_str(&saved).expect("reload");
    let resaved = save_scene_to_string(&reloaded).expect("resave");
    assert_eq!(saved, resaved, "save → load → save must be a fixed point");
    assert_eq!(reloaded.roots.len(), 1);
    // Loading renumbers ids deterministically; the fluid root must follow
    // the same (single) root object.
    let _ = result;
    let reloaded_root = reloaded.roots[0];
    assert!(reloaded
        .fluid_domain
        .as_ref()
        .is_some_and(|fluid| fluid.root == reloaded_root));
    assert!(matches!(
        reloaded.object(reloaded_root).expect("root").payload,
        ScenePayload::Operator { .. }
    ));
}
