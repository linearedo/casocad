//! Tests for the interactive scene ops behind the viewport tools:
//! drag-sized creation, point-shape placement, duplicate/paste — plus the
//! Phase 5 acceptance workflow (draw, subtract, set domain, save/reload).

use caso_kernel::meshing::meshable_domains_from_document;
use caso_kernel::roles::DomainKind;
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

/// Two clicks — center, then a vertex — define a regular polygon: the
/// second click sets radius AND rotation, so the clicked point is a vertex
/// of the committed profile.
#[test]
fn regular_polygon_from_center_and_vertex_clicks() {
    let mut document = SceneDocument::new();
    let center = vec3(1.0, 1.0, 0.0);
    let vertex = vec3(1.0 + 3.0, 1.0 + 4.0, 0.0);
    let id = document
        .add_regular_polygon_from_world_points(&[center, vertex], 5, "xy")
        .expect("regular polygon from points");
    match &document.object(id).expect("object").payload {
        ScenePayload::Placed2D { profile, origin, .. } => {
            assert_eq!(*origin, center);
            match profile {
                Profile2D::RegularPolygon {
                    center,
                    radius,
                    side_count,
                    rotation,
                } => {
                    assert_eq!(*center, [0.0, 0.0]);
                    assert!((radius - 5.0).abs() < 1e-12);
                    assert_eq!(*side_count, 5);
                    assert!((rotation - (4.0f64).atan2(3.0)).abs() < 1e-12);
                }
                other => panic!("expected RegularPolygon, got {other:?}"),
            }
        }
        other => panic!("expected Placed2D, got {other:?}"),
    }
    // One point, three points, coincident clicks, and <3 sides all refuse.
    assert!(document
        .add_regular_polygon_from_world_points(&[center], 5, "xy")
        .is_err());
    assert!(document
        .add_regular_polygon_from_world_points(&[center, vertex, center], 5, "xy")
        .is_err());
    assert!(document
        .add_regular_polygon_from_world_points(&[center, center], 5, "xy")
        .is_err());
    assert!(document
        .add_regular_polygon_from_world_points(&[center, vertex], 2, "xy")
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
    // The Difference root translates in place (leaf placements shift), so
    // the paste stays a plain deep copy with no Translate wrapper.
    assert_eq!(document.live_ids().len(), before * 2);
    assert!(matches!(copy.payload, ScenePayload::Operator { .. }));
    // Selecting both original + a nested child only copies the root once.
    let child = document.object(root).expect("root").payload.children()[0];
    let pasted = document
        .duplicate_nodes(&[root, child], vec3(0.1, 0.1, 0.0))
        .expect("duplicate with descendant");
    assert_eq!(pasted.len(), 1);
}

/// Moving a boolean result mutates the leaf placements — it must never turn
/// the object into a Translate node in the scene graph (Python parity).
#[test]
fn move_operator_translates_leaves_in_place() {
    let mut document = SceneDocument::default_scene().expect("default scene");
    let root = document.roots[0];
    let children = document.object(root).expect("root").payload.children();
    let center_of = |document: &SceneDocument, id| {
        match &document.object(id).expect("child").payload {
            ScenePayload::Box3(shape) => shape.center,
            ScenePayload::Cylinder(shape) => shape.center,
            other => panic!("unexpected payload: {other:?}"),
        }
    };
    let before: Vec<_> = children
        .iter()
        .map(|id| center_of(&document, *id))
        .collect();
    let live = document.live_ids().len();
    let delta = vec3(0.3, -0.2, 0.1);
    let moved = document.move_object(root, delta).expect("move");
    assert_eq!(moved, root);
    assert_eq!(document.roots, vec![root]);
    assert_eq!(document.live_ids().len(), live);
    assert!(matches!(
        document.object(root).expect("root").payload,
        ScenePayload::Operator { .. }
    ));
    for (id, previous) in children.iter().zip(before) {
        let shifted = previous + delta;
        assert!((center_of(&document, *id) - shifted).length() < 1e-12);
    }
}

/// A Rotate/Scale wrapper in the subtree is the one case that still falls
/// back to a Translate wrapper (same as Python).
#[test]
fn move_rotate_wrapped_subtree_still_wraps() {
    let mut document = SceneDocument::default_scene().expect("default scene");
    let root = document.roots[0];
    let rotated = document.wrap_transform(root, "rotate").expect("wrap rotate");
    let moved = document.move_object(rotated, vec3(1.0, 0.0, 0.0)).expect("move");
    assert_ne!(moved, rotated);
    assert!(matches!(
        document.object(moved).expect("moved").payload,
        ScenePayload::Translate { .. }
    ));
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
    // The imported Difference subtree is offset in place — no wrapper.
    assert_eq!(target.live_ids().len(), source.live_ids().len());
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

/// Subtracting from the fluid root must move the domain mark to the new
/// root — not leave two nested fluid domains that meshing refuses.
#[test]
fn subtract_from_fluid_root_keeps_a_single_domain() {
    let mut document = SceneDocument::default_scene().expect("default scene");
    let fluid_root = document.fluid_domain.as_ref().expect("fluid").root;
    let sphere = document
        .add_primitive_from_drag("sphere", vec3(0.9, 0.0, 0.5), vec3(1.1, 0.0, 0.5), 1.0)
        .expect("sphere");
    let combined = document
        .combine(fluid_root, sphere, "difference")
        .expect("subtract");

    assert_eq!(document.fluid_domain.as_ref().expect("fluid").root, combined);
    let marks: Vec<_> = document
        .domain_kinds
        .iter()
        .map(|(id, kind)| (*id, *kind))
        .collect();
    assert_eq!(marks, vec![(combined, DomainKind::Fluid)]);
    let domains = meshable_domains_from_document(&document).expect("meshable");
    assert_eq!(domains.len(), 1);
}

/// Wrapping the fluid root in a transform moves the mark to the wrapper.
#[test]
fn transform_of_fluid_root_moves_the_domain_mark() {
    let mut document = SceneDocument::default_scene().expect("default scene");
    let fluid_root = document.fluid_domain.as_ref().expect("fluid").root;
    let wrapped = document
        .wrap_transform(fluid_root, "translate")
        .expect("wrap");
    assert_eq!(document.fluid_domain.as_ref().expect("fluid").root, wrapped);
    assert_eq!(document.domain_kinds.len(), 1);
    assert_eq!(document.domain_kinds.get(&wrapped), Some(&DomainKind::Fluid));
}

/// Solid domains follow the same rule as fluid: booleans carry the mark.
#[test]
fn difference_inherits_a_solid_domain_mark() {
    let mut document = SceneDocument::new();
    let block = document.add_primitive("box", 1.0).expect("box");
    let hole = document.add_primitive("sphere", 0.4).expect("sphere");
    document
        .set_domain_root(block, DomainKind::Solid)
        .expect("solid domain");
    let combined = document.combine(block, hole, "difference").expect("subtract");
    assert_eq!(document.domain_kinds.len(), 1);
    assert_eq!(document.domain_kinds.get(&combined), Some(&DomainKind::Solid));
}

/// Domain marks are legal on nested objects: a cutter consumed by a boolean
/// can still be (or become) a domain.
#[test]
fn set_domain_root_accepts_nested_nodes() {
    let mut document = SceneDocument::default_scene().expect("default scene");
    let root = document.roots[0];
    let nested = document.object(root).expect("root").payload.children()[1];
    document
        .set_domain_root(nested, DomainKind::Solid)
        .expect("nested nodes can carry domain marks");
    assert_eq!(document.domain_kinds.get(&nested), Some(&DomainKind::Solid));
}

/// Files saved with a stale nested domain mark (the old bug) self-heal on
/// load: the nested mark is dropped, the root domain and its tags survive.
#[test]
fn loading_drops_stale_nested_domain_marks() {
    let document = SceneDocument::default_scene().expect("default scene");
    let saved = save_scene_to_string(&document).expect("save");
    let mut payload: serde_json::Value = serde_json::from_str(&saved).expect("json");
    payload["domains"]["flow_volume"] =
        serde_json::json!({ "root": "flow_volume", "type": "fluid" });
    let corrupted = serde_json::to_string(&payload).expect("corrupted json");

    let reloaded = load_scene_from_str(&corrupted).expect("load");
    assert_eq!(reloaded.roots.len(), 1);
    let root = reloaded.roots[0];
    assert_eq!(reloaded.domain_kinds.len(), 1);
    assert_eq!(reloaded.domain_kinds.get(&root), Some(&DomainKind::Fluid));
    let fluid = reloaded.fluid_domain.as_ref().expect("fluid record");
    assert_eq!(fluid.root, root);
    assert_eq!(fluid.tags.len(), 2, "inlet/outlet tags survive the heal");
}

fn id_by_name(document: &SceneDocument, name: &str) -> u32 {
    document
        .live_ids()
        .into_iter()
        .find(|id| {
            document
                .object(*id)
                .map(|object| object.name == name)
                .unwrap_or(false)
        })
        .unwrap_or_else(|| panic!("no live object named {name}"))
}

/// The multi-region reference scene: a sea box (Fluid) minus a pipe, where
/// the pipe is a solid shell (Solid) minus the gas in its bore (Fluid).
/// Shapes are spheres/boxes for simplicity; the tree topology is the point.
fn build_sea_pipe_gas() -> SceneDocument {
    let mut document = SceneDocument::new();
    let shell = document
        .add_primitive_from_drag("sphere", vec3(-0.6, 0.0, 0.0), vec3(0.6, 0.0, 0.0), 1.0)
        .expect("shell");
    document
        .set_domain_root(shell, DomainKind::Solid)
        .expect("shell solid");
    let gas = document
        .add_primitive_from_drag("sphere", vec3(-0.3, 0.0, 0.0), vec3(0.3, 0.0, 0.0), 1.0)
        .expect("gas");
    document
        .set_domain_root(gas, DomainKind::Fluid)
        .expect("gas fluid");
    let pipe = document.combine(shell, gas, "difference").expect("pipe");
    let sea_box = document
        .add_primitive_from_drag("box", vec3(-2.0, -1.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
        .expect("sea box");
    document
        .set_domain_root(sea_box, DomainKind::Fluid)
        .expect("sea fluid");
    let sea = document.combine(sea_box, pipe, "difference").expect("sea");
    document.rename(pipe, "pipe").expect("rename pipe");
    document.rename(gas, "gas").expect("rename gas");
    document.rename(sea, "sea").expect("rename sea");
    document
}

/// A subtracted solid domain stays alive: the difference is the fluid, the
/// nested cutter is still the solid, and the two mesh as disjoint regions.
#[test]
fn subtracted_solid_domain_stays_alive() {
    let mut document = SceneDocument::new();
    let basin = document
        .add_primitive_from_drag("box", vec3(-2.0, -1.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
        .expect("basin");
    document
        .set_domain_root(basin, DomainKind::Fluid)
        .expect("fluid");
    let pipe = document
        .add_primitive_from_drag("sphere", vec3(-0.3, 0.0, 0.0), vec3(0.3, 0.0, 0.0), 1.0)
        .expect("pipe");
    document
        .set_domain_root(pipe, DomainKind::Solid)
        .expect("solid");
    let combined = document.combine(basin, pipe, "difference").expect("subtract");

    assert_eq!(document.domain_kinds.len(), 2);
    assert_eq!(document.domain_kinds.get(&combined), Some(&DomainKind::Fluid));
    assert_eq!(document.domain_kinds.get(&pipe), Some(&DomainKind::Solid));
    assert_eq!(document.fluid_domain.as_ref().expect("fluid").root, combined);
    let domains = meshable_domains_from_document(&document).expect("meshable");
    assert_eq!(domains.len(), 2);
}

/// The sea/pipe/gas case: each domain's meshing region excludes the domains
/// nested inside it, so the sea excludes the whole pipe envelope (shell AND
/// gas bore), not just the shell it subtracted.
#[test]
fn sea_pipe_gas_three_regions() {
    let document = build_sea_pipe_gas();
    assert_eq!(document.domain_kinds.len(), 3);
    let domains = meshable_domains_from_document(&document).expect("meshable");
    assert_eq!(domains.len(), 3);
    let sea = domains.get("sea").expect("sea");
    let pipe = domains.get("pipe").expect("pipe");
    let gas = domains.get("gas").expect("gas");

    // Bore point: gas only. Without the carve the sea would claim it too —
    // subtracting the shell from the box leaves the bore open.
    let bore = vec3(0.0, 0.0, 0.0);
    assert!(gas.domain_sdf(&[bore])[0] < 0.0);
    assert!(pipe.domain_sdf(&[bore])[0] > 0.0);
    assert!(sea.domain_sdf(&[bore])[0] > 0.0);

    // Shell-wall point: pipe only.
    let wall = vec3(0.45, 0.0, 0.0);
    assert!(pipe.domain_sdf(&[wall])[0] < 0.0);
    assert!(gas.domain_sdf(&[wall])[0] > 0.0);
    assert!(sea.domain_sdf(&[wall])[0] > 0.0);

    // Open-water point: sea only.
    let water = vec3(1.5, 0.5, 0.5);
    assert!(sea.domain_sdf(&[water])[0] < 0.0);
    assert!(pipe.domain_sdf(&[water])[0] > 0.0);
    assert!(gas.domain_sdf(&[water])[0] > 0.0);
}

/// A nested domain's meshing region follows ancestor transforms: wrapping
/// the difference in a Translate moves the nested solid's region too.
#[test]
fn nested_domain_follows_ancestor_transform() {
    let mut document = SceneDocument::new();
    let basin = document
        .add_primitive_from_drag("box", vec3(-2.0, -1.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
        .expect("basin");
    document
        .set_domain_root(basin, DomainKind::Fluid)
        .expect("fluid");
    let ball = document
        .add_primitive_from_drag("sphere", vec3(-0.3, 0.0, 0.0), vec3(0.3, 0.0, 0.0), 1.0)
        .expect("ball");
    document
        .set_domain_root(ball, DomainKind::Solid)
        .expect("solid");
    document.rename(ball, "ball").expect("rename");
    let combined = document.combine(basin, ball, "difference").expect("subtract");
    // Default translate offset is (0.1, 0, 0).
    document.wrap_transform(combined, "translate").expect("wrap");

    let domains = meshable_domains_from_document(&document).expect("meshable");
    assert_eq!(domains.len(), 2);
    let solid = domains.get("ball").expect("ball");
    // Inside the translated ball (0.25 from its center), outside the
    // original position (0.35 from the untranslated center).
    assert!(solid.domain_sdf(&[vec3(0.35, 0.0, 0.0)])[0] < 0.0);
    // Inside the original position, outside the translated ball.
    assert!(solid.domain_sdf(&[vec3(-0.25, 0.0, 0.0)])[0] > 0.0);
}

/// Deleting the cutter collapses the difference to the surviving operand —
/// the domain mark and the fluid record devolve to the survivor.
#[test]
fn deleting_the_cutter_devolves_the_mark() {
    let mut document = SceneDocument::new();
    let basin = document
        .add_primitive_from_drag("box", vec3(-2.0, -1.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
        .expect("basin");
    document
        .set_domain_root(basin, DomainKind::Fluid)
        .expect("fluid");
    let hole = document
        .add_primitive_from_drag("sphere", vec3(-0.3, 0.0, 0.0), vec3(0.3, 0.0, 0.0), 1.0)
        .expect("hole");
    let combined = document.combine(basin, hole, "difference").expect("subtract");
    assert_eq!(document.domain_kinds.get(&combined), Some(&DomainKind::Fluid));

    document.delete(hole);
    assert_eq!(document.roots, vec![basin]);
    let marks: Vec<_> = document
        .domain_kinds
        .iter()
        .map(|(id, kind)| (*id, *kind))
        .collect();
    assert_eq!(marks, vec![(basin, DomainKind::Fluid)]);
    assert_eq!(document.fluid_domain.as_ref().expect("fluid").root, basin);
}

/// Once an object has evolved into a domain, it cannot be marked again: two
/// marks may never sit on the same evolution chain.
#[test]
fn remarking_the_evolution_base_is_refused() {
    let mut document = SceneDocument::new();
    let basin = document
        .add_primitive_from_drag("box", vec3(-2.0, -1.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
        .expect("basin");
    document
        .set_domain_root(basin, DomainKind::Fluid)
        .expect("fluid");
    let hole = document
        .add_primitive_from_drag("sphere", vec3(-0.3, 0.0, 0.0), vec3(0.3, 0.0, 0.0), 1.0)
        .expect("hole");
    document.combine(basin, hole, "difference").expect("subtract");

    let error = document
        .set_domain_root(basin, DomainKind::Fluid)
        .expect_err("the evolved base must be refused");
    assert!(error.to_string().contains("evolution chain"));
    assert!(
        document.set_domain_root(basin, DomainKind::Solid).is_err(),
        "the rule is kind-independent"
    );
}

/// A marked cutter keeps its top-level exposure: the root entry and the
/// object inside the consumer's chain are the SAME object.
#[test]
fn marked_cutter_stays_a_root() {
    let document = build_sea_pipe_gas();
    let sea = id_by_name(&document, "sea");
    let pipe = id_by_name(&document, "pipe");
    let gas = id_by_name(&document, "gas");
    for id in [sea, pipe, gas] {
        assert!(document.roots.contains(&id), "domains stay top-level");
    }
    let ScenePayload::Operator { right, .. } = document.object(sea).expect("sea").payload else {
        panic!("sea must be an operator");
    };
    assert_eq!(right, pipe, "root entry and chain occurrence are one object");
}

/// Marking an already-nested object promotes it to a top-level root.
#[test]
fn marking_a_nested_object_promotes_it_to_root() {
    let mut document = SceneDocument::new();
    let basin = document
        .add_primitive_from_drag("box", vec3(-2.0, -1.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
        .expect("basin");
    let hole = document
        .add_primitive_from_drag("sphere", vec3(-0.3, 0.0, 0.0), vec3(0.3, 0.0, 0.0), 1.0)
        .expect("hole");
    document.combine(basin, hole, "difference").expect("subtract");
    assert!(!document.roots.contains(&hole), "unmarked cutter is consumed");

    document
        .set_domain_root(hole, DomainKind::Solid)
        .expect("mark nested");
    assert!(document.roots.contains(&hole), "marking exposes it as a root");
}

/// Unsetting a shared domain drops only its top-level exposure; a standalone
/// marked root stays a root.
#[test]
fn unsetting_a_shared_domain_removes_its_root_entry() {
    let mut document = build_sea_pipe_gas();
    let pipe = id_by_name(&document, "pipe");
    document.unset_domain_root(pipe);
    assert!(!document.roots.contains(&pipe), "exposure removed");
    assert!(
        document.live_ids().contains(&pipe),
        "still alive inside sea's chain"
    );

    let solo = document.add_primitive("box", 1.0).expect("solo box");
    document
        .set_domain_root(solo, DomainKind::Solid)
        .expect("solid");
    document.unset_domain_root(solo);
    assert!(document.roots.contains(&solo), "standalone root survives");
}

/// The shared exposure survives save/load as ONE object, not a copy.
#[test]
fn shared_domain_roundtrips_as_one_object() {
    let document = build_sea_pipe_gas();
    let saved = save_scene_to_string(&document).expect("save");
    let reloaded = load_scene_from_str(&saved).expect("load");

    assert_eq!(reloaded.roots.len(), 3);
    let sea = id_by_name(&reloaded, "sea");
    let pipe = id_by_name(&reloaded, "pipe");
    assert!(reloaded.roots.contains(&pipe));
    let ScenePayload::Operator { right, .. } = reloaded.object(sea).expect("sea").payload else {
        panic!("sea must be an operator");
    };
    assert_eq!(right, pipe, "one shared object after reload");
}

/// When a shared domain evolves (here: wrapped in a transform), every
/// reference to it follows the evolution — the consumer's chain and the
/// top-level exposure stay synchronized.
#[test]
fn evolving_a_shared_domain_rewrites_references() {
    let mut document = build_sea_pipe_gas();
    let sea = id_by_name(&document, "sea");
    let pipe = id_by_name(&document, "pipe");
    let wrapped = document.wrap_transform(pipe, "translate").expect("wrap");

    let ScenePayload::Operator { right, .. } = document.object(sea).expect("sea").payload else {
        panic!("sea must be an operator");
    };
    assert_eq!(right, wrapped, "sea's chain follows the evolution");
    assert!(document.roots.contains(&wrapped));
    assert!(!document.roots.contains(&pipe));
    assert_eq!(document.domain_kinds.get(&wrapped), Some(&DomainKind::Solid));
    assert!(!document.domain_kinds.contains_key(&pipe));
    let domains = meshable_domains_from_document(&document).expect("meshable");
    assert_eq!(domains.len(), 3);
}

/// The viewport draws primary roots only: a domain exposed as a root while
/// nested in another chain is not meshed twice.
#[test]
fn primary_roots_hide_shared_domains() {
    let document = build_sea_pipe_gas();
    let sea = id_by_name(&document, "sea");
    assert_eq!(document.primary_roots(), vec![sea]);
}

/// Nested domain marks round-trip through save/load unchanged.
#[test]
fn nested_marks_survive_save_and_load() {
    let document = build_sea_pipe_gas();
    let saved = save_scene_to_string(&document).expect("save");
    let reloaded = load_scene_from_str(&saved).expect("load");

    assert_eq!(reloaded.domain_kinds.len(), 3);
    let sea = id_by_name(&reloaded, "sea");
    let pipe = id_by_name(&reloaded, "pipe");
    let gas = id_by_name(&reloaded, "gas");
    assert_eq!(reloaded.domain_kinds.get(&sea), Some(&DomainKind::Fluid));
    assert_eq!(reloaded.domain_kinds.get(&pipe), Some(&DomainKind::Solid));
    assert_eq!(reloaded.domain_kinds.get(&gas), Some(&DomainKind::Fluid));
    assert_eq!(reloaded.fluid_domain.as_ref().expect("fluid").root, sea);
    let domains = meshable_domains_from_document(&reloaded).expect("meshable");
    assert_eq!(domains.len(), 3);
}
