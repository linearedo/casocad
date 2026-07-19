//! Combined 1D/2D booleans keep their operand ("source") objects in
//! lockstep: the Binary profile is what renders, the sources are what the
//! scene tree shows and edits — `resync_boolean_chains` must reconcile
//! both directions, or the properties panel drifts from the geometry.

use caso_kernel::scene::{ObjectId, SceneDocument, ScenePayload};
use caso_kernel::sdf::primitives_1d::Profile1D;
use caso_kernel::sdf::primitives_2d::Profile2D;
use caso_kernel::vec3::{vec3, Vec3};

fn rect_and_circle(document: &mut SceneDocument) -> (ObjectId, ObjectId, ObjectId) {
    let rect = document
        .add_primitive_from_drag("rectangle", vec3(-2.0, -1.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
        .expect("rectangle");
    let circle = document
        .add_primitive_from_drag("circle", vec3(0.2, -0.3, 0.0), vec3(0.8, 0.3, 0.0), 1.0)
        .expect("circle");
    let combined = document
        .combine(rect, circle, "difference")
        .expect("difference");
    (rect, circle, combined)
}

fn placed_2d_origin(document: &SceneDocument, id: ObjectId) -> Vec3 {
    match &document.object(id).expect("object").payload {
        ScenePayload::Placed2D { origin, .. } => *origin,
        other => panic!("expected Placed2D, got {other:?}"),
    }
}

/// The circle radius inside the combined node's Binary right subtree.
fn combined_right_circle_radius(document: &SceneDocument, id: ObjectId) -> f64 {
    let ScenePayload::Placed2D {
        profile: Profile2D::Binary { right, .. },
        ..
    } = &document.object(id).expect("object").payload
    else {
        panic!("expected a combined 2D boolean");
    };
    let mut child = right.as_ref();
    while let Profile2D::Offset { child: inner, .. } = child {
        child = inner;
    }
    match child {
        Profile2D::Circle { radius, .. } => *radius,
        other => panic!("expected the circle operand, got {other:?}"),
    }
}

fn signed_distance(document: &SceneDocument, id: ObjectId, point: Vec3) -> f64 {
    let node = document.build_node(id).expect("node");
    node.eval(&[point])[0]
}

#[test]
fn editing_a_source_updates_the_combined_profile() {
    let mut document = SceneDocument::new();
    let (_rect, circle, combined) = rect_and_circle(&mut document);
    // (1.4, 0, 0) is solid: outside the r=0.3 hole around (0.5, 0).
    assert!(signed_distance(&document, combined, vec3(1.4, 0.0, 0.0)) < 0.0);

    // Grow the circle through its own (source) object, as the properties
    // panel does, then resync.
    if let ScenePayload::Placed2D { profile, .. } =
        &mut document.object_mut(circle).expect("circle").payload
    {
        let mut target = profile;
        while let Profile2D::Offset { child, .. } = target {
            target = child;
        }
        match target {
            Profile2D::Circle { radius, .. } => *radius = 1.0,
            other => panic!("expected circle profile, got {other:?}"),
        }
    }
    document.resync_boolean_chains(circle);

    assert_eq!(combined_right_circle_radius(&document, combined), 1.0);
    // The grown hole now swallows (1.4, 0, 0).
    assert!(
        signed_distance(&document, combined, vec3(1.4, 0.0, 0.0)) > 0.0,
        "the combined geometry must reflect the edited source"
    );
}

#[test]
fn moving_the_combined_object_moves_its_sources() {
    let mut document = SceneDocument::new();
    let (rect, circle, combined) = rect_and_circle(&mut document);
    let rect_before = placed_2d_origin(&document, rect);
    let circle_before = placed_2d_origin(&document, circle);
    let delta = vec3(3.0, -2.0, 0.0);
    document.move_object(combined, delta).expect("move");
    assert_eq!(placed_2d_origin(&document, rect), rect_before + delta);
    assert_eq!(placed_2d_origin(&document, circle), circle_before + delta);
}

#[test]
fn moving_a_source_reoffsets_the_boolean() {
    let mut document = SceneDocument::new();
    let (_rect, circle, combined) = rect_and_circle(&mut document);
    // Hole starts around (0.5, 0): solid at (-0.5, 0), open at (0.5, 0).
    assert!(signed_distance(&document, combined, vec3(-0.5, 0.0, 0.0)) < 0.0);
    assert!(signed_distance(&document, combined, vec3(0.5, 0.0, 0.0)) > 0.0);

    document.move_object(circle, vec3(-1.0, 0.0, 0.0)).expect("move");

    // Hole followed the source to (-0.5, 0).
    assert!(signed_distance(&document, combined, vec3(-0.5, 0.0, 0.0)) > 0.0);
    assert!(signed_distance(&document, combined, vec3(0.5, 0.0, 0.0)) < 0.0);
}

#[test]
fn nested_combine_syncs_through_both_levels() {
    let mut document = SceneDocument::new();
    let (rect, _circle, inner) = rect_and_circle(&mut document);
    let square = document
        .add_primitive_from_drag("square", vec3(-1.8, 0.4, 0.0), vec3(-1.2, 1.0, 0.0), 1.0)
        .expect("square");
    let outer = document.combine(inner, square, "difference").expect("outer");

    // Move the deepest source; both boolean levels must follow it.
    let rect_before = placed_2d_origin(&document, rect);
    let delta = vec3(0.25, 0.5, 0.0);
    document.move_object(rect, delta).expect("move");
    assert_eq!(placed_2d_origin(&document, rect), rect_before + delta);
    assert_eq!(
        placed_2d_origin(&document, inner),
        rect_before + delta,
        "inner boolean tracks its first operand"
    );
    assert_eq!(
        placed_2d_origin(&document, outer),
        rect_before + delta,
        "outer boolean tracks the whole chain"
    );
}

#[test]
fn segment_boolean_sources_follow_moves() {
    let mut document = SceneDocument::new();
    let first = document
        .add_point_shape_from_world_points(
            "segment",
            &[vec3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0)],
            "xy",
        )
        .expect("first segment");
    let second = document
        .add_point_shape_from_world_points(
            "segment",
            &[vec3(2.0, 0.0, 0.0), vec3(3.0, 0.0, 0.0)],
            "xy",
        )
        .expect("second segment");
    let combined = document.combine(first, second, "union").expect("union");

    let origin_of = |document: &SceneDocument, id: ObjectId| match &document
        .object(id)
        .expect("object")
        .payload
    {
        ScenePayload::Placed1D { origin, .. } => *origin,
        other => panic!("expected Placed1D, got {other:?}"),
    };
    let first_before = origin_of(&document, first);
    let second_before = origin_of(&document, second);
    let delta = vec3(5.0, 0.0, 0.0);
    document.move_object(combined, delta).expect("move");
    assert_eq!(origin_of(&document, first), first_before + delta);
    assert_eq!(origin_of(&document, second), second_before + delta);

    // And the offset in the Binary stays the operand distance.
    let ScenePayload::Placed1D {
        profile: Profile1D::Binary { right, .. },
        ..
    } = &document.object(combined).expect("combined").payload
    else {
        panic!("expected a combined 1D boolean");
    };
    let Profile1D::Offset { offset, .. } = right.as_ref() else {
        panic!("expected the offset-wrapped second operand");
    };
    assert!((offset - 2.0).abs() < 1.0e-9, "operand spacing preserved, got {offset}");
}
