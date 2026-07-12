//! Ports of `tests/test_boundary_region_classifier.py` and
//! `tests/test_boundary_split.py`. Fixture: the default von Kármán scene —
//! root = Difference(flow box, cylinder obstacle), box half_size
//! (1.6, 0.7, 0.45), cylinder radius 0.24 through Z.

use caso_kernel::boundary::{BoundaryCut, BoundaryRegion, CutSide};
use caso_kernel::boundary_ops::{
    boundary_region_base_mask, boundary_region_mask, cut_volume, owner_active_mask,
    pick_boundary_patch, region_tolerance, surface_patches_for_root,
};
use caso_kernel::scene::{ObjectId, SceneDocument, ScenePayload};
use caso_kernel::sdf::node::{Node, Shape};
use caso_kernel::sdf::primitives_3d::{Box3, Pyramid, Sphere};
use caso_kernel::frame::IDENTITY_FRAME;
use caso_kernel::vec3::{vec3, Vec3};

struct Fixture {
    document: SceneDocument,
    root: Node,
    box_id: ObjectId,
    cylinder_id: ObjectId,
}

fn default_scene() -> Fixture {
    let document = SceneDocument::default_scene().expect("default scene");
    let fluid_root = document.fluid_domain.as_ref().expect("fluid").root;
    let root = document.build_node(fluid_root).expect("root node");
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
    Fixture {
        document,
        root,
        box_id,
        cylinder_id,
    }
}

fn face_points(x: f64, count: usize) -> Vec<Vec3> {
    let mut points = Vec::new();
    for i in 0..count {
        let y = -0.6 + 1.2 * (i as f64) / ((count - 1) as f64);
        for j in 0..count {
            let z = -0.35 + 0.7 * (j as f64) / ((count - 1) as f64);
            points.push(vec3(x, y, z));
        }
    }
    points
}

fn cylinder_wall_points() -> Vec<Vec3> {
    let radius = 0.24;
    let mut points = Vec::new();
    for a in 0..24 {
        let angle = 2.0 * std::f64::consts::PI * (a as f64) / 24.0;
        for k in 0..5 {
            let z = -0.3 + 0.6 * (k as f64) / 4.0;
            points.push(vec3(radius * angle.cos(), radius * angle.sin(), z));
        }
    }
    points
}

fn region(owner_id: ObjectId) -> BoundaryRegion {
    BoundaryRegion::new("r", 999, owner_id)
}

fn ghost_sphere(center: Vec3, radius: f64) -> Node {
    Node::new(
        "ghost",
        Shape::Sphere(Sphere::new(center, radius).expect("sphere")),
    )
}

fn ghost_box(center: Vec3, half: Vec3) -> Node {
    Node::new(
        "ghost_box",
        Shape::Box3(Box3::new(center, half, IDENTITY_FRAME).expect("box")),
    )
}

fn all(mask: &[bool]) -> bool {
    mask.iter().all(|hit| *hit)
}

fn any(mask: &[bool]) -> bool {
    mask.iter().any(|hit| *hit)
}

#[test]
fn obstacle_owns_the_cut_surface() {
    let fixture = default_scene();
    let wall = cylinder_wall_points();
    let tol = region_tolerance(&fixture.root, &region(fixture.cylinder_id));

    assert!(all(&owner_active_mask(
        &fixture.root,
        fixture.cylinder_id,
        &wall,
        tol
    )));
    let cylinder_mask =
        boundary_region_mask(&fixture.root, &region(fixture.cylinder_id), &wall, None)
            .expect("mask");
    let box_mask = boundary_region_mask(&fixture.root, &region(fixture.box_id), &wall, None)
        .expect("mask");
    assert!(all(&cylinder_mask));
    assert!(!any(&box_mask));
}

#[test]
fn direction_region_selects_one_face() {
    let fixture = default_scene();
    let mut inlet = region(fixture.box_id);
    inlet.outside_direction = Some(0); // -X face
    let minus_x = face_points(-1.6, 7);
    let plus_x = face_points(1.6, 7);
    let wall = cylinder_wall_points();

    assert!(all(
        &boundary_region_mask(&fixture.root, &inlet, &minus_x, None).expect("mask")
    ));
    assert!(!any(
        &boundary_region_mask(&fixture.root, &inlet, &plus_x, None).expect("mask")
    ));
    assert!(!any(
        &boundary_region_mask(&fixture.root, &inlet, &wall, None).expect("mask")
    ));
}

#[test]
fn one_cut_partitions_the_parent_exactly() {
    let fixture = default_scene();
    let mut samples = face_points(-1.6, 7);
    samples.extend(face_points(1.6, 7));
    samples.extend(cylinder_wall_points());
    let parent = region(fixture.box_id);
    let ghost = ghost_sphere(vec3(-1.6, 0.0, 0.0), 0.5);
    let mut inside = region(fixture.box_id);
    inside.cuts = vec![BoundaryCut {
        side: CutSide::Inside,
        ghost: ghost.clone(),
    }];
    let mut outside = region(fixture.box_id);
    outside.cuts = vec![BoundaryCut {
        side: CutSide::Outside,
        ghost,
    }];

    let parent_mask = boundary_region_mask(&fixture.root, &parent, &samples, None).expect("mask");
    let inside_mask = boundary_region_mask(&fixture.root, &inside, &samples, None).expect("mask");
    let outside_mask =
        boundary_region_mask(&fixture.root, &outside, &samples, None).expect("mask");

    assert!(any(&parent_mask) && any(&inside_mask) && any(&outside_mask));
    for i in 0..samples.len() {
        assert!(!(inside_mask[i] && outside_mask[i]), "no double-tagging");
        assert_eq!(inside_mask[i] || outside_mask[i], parent_mask[i], "no gaps");
    }
    for (i, point) in samples.iter().enumerate() {
        if inside_mask[i] {
            assert!((*point - vec3(-1.6, 0.0, 0.0)).length() < 0.55);
        }
    }
}

#[test]
fn cut_chain_is_a_conjunction() {
    let fixture = default_scene();
    let samples = face_points(-1.6, 13);
    let sphere = ghost_sphere(vec3(-1.6, 0.0, 0.0), 0.4);
    let upper = ghost_box(vec3(-1.6, 1.0, 0.0), vec3(1.0, 1.0, 1.0));
    let mut chained = region(fixture.box_id);
    chained.cuts = vec![
        BoundaryCut {
            side: CutSide::Outside,
            ghost: sphere.clone(),
        },
        BoundaryCut {
            side: CutSide::Inside,
            ghost: upper,
        },
    ];

    let mask = boundary_region_mask(&fixture.root, &chained, &samples, None).expect("mask");
    let mut inside_sphere = region(fixture.box_id);
    inside_sphere.cuts = vec![BoundaryCut {
        side: CutSide::Inside,
        ghost: sphere,
    }];
    let ring: Vec<bool> =
        boundary_region_mask(&fixture.root, &inside_sphere, &samples, None)
            .expect("mask")
            .into_iter()
            .map(|hit| !hit)
            .collect();
    let parent =
        boundary_region_mask(&fixture.root, &region(fixture.box_id), &samples, None)
            .expect("mask");

    assert!(any(&mask));
    for i in 0..samples.len() {
        let expected = parent[i] && ring[i] && samples[i].y >= 0.0;
        assert_eq!(mask[i], expected);
    }
}

#[test]
fn lower_dimensional_ghost_extrudes_through_the_scene() {
    let mut fixture = default_scene();
    let handle = fixture
        .document
        .add_primitive_from_drag(
            "segment",
            vec3(-1.6, -0.2, 0.0),
            vec3(-1.6, 0.2, 0.0),
            1.0,
        )
        .expect("segment");
    let segment = fixture.document.build_node(handle).expect("segment node");
    let mut region = region(fixture.box_id);
    region.cuts = vec![BoundaryCut {
        side: CutSide::Inside,
        ghost: segment,
    }];
    let samples = face_points(-1.6, 13);

    let mask = boundary_region_mask(&fixture.root, &region, &samples, None).expect("mask");

    assert!(any(&mask));
    for (i, point) in samples.iter().enumerate() {
        if mask[i] {
            assert!(point.y.abs() <= 0.25);
        }
        if point.y.abs() > 0.3 {
            assert!(!mask[i]);
        }
    }
}

#[test]
fn mask_composes_base_mask_and_cut_chain() {
    // Pins the base/full split: full mask == base mask (criteria 1-3)
    // ∧ every cut's tol-banded sign test (criterion 4).
    let fixture = default_scene();
    let mut samples = face_points(-1.6, 9);
    samples.extend(cylinder_wall_points());

    let plain = region(fixture.box_id);
    let full = boundary_region_mask(&fixture.root, &plain, &samples, None).expect("mask");
    let base =
        boundary_region_base_mask(&fixture.root, &plain, &samples, None).expect("mask");
    assert_eq!(full, base, "cut-free regions: full == base");

    let mut with_cut = region(fixture.box_id);
    with_cut.cuts = vec![BoundaryCut {
        side: CutSide::Inside,
        ghost: ghost_sphere(vec3(-1.6, 0.0, 0.0), 0.5),
    }];
    let tol = region_tolerance(&fixture.root, &with_cut);
    let volume = cut_volume(&fixture.root, &with_cut.cuts[0]).expect("volume");
    let full =
        boundary_region_mask(&fixture.root, &with_cut, &samples, None).expect("mask");
    let base =
        boundary_region_base_mask(&fixture.root, &with_cut, &samples, None).expect("mask");
    for (i, point) in samples.iter().enumerate() {
        let expected = base[i] && volume.eval_point(*point) <= tol;
        assert_eq!(full[i], expected, "composition mismatch at {point:?}");
    }
}

#[test]
fn tolerance_scales_with_owner_size() {
    let scale = 0.001; // a mm-scale scene
    let owner = Node::with_id(
        "b",
        1,
        Shape::Box3(
            Box3::new(
                Vec3::ZERO,
                vec3(1.6 * scale, 0.7 * scale, 0.45 * scale),
                IDENTITY_FRAME,
            )
            .expect("box"),
        ),
    );
    let region = region(1);
    let face = [
        vec3(-1.6 * scale, 0.0, 0.0),
        vec3(-1.6 * scale, 0.0002, 0.0001),
    ];
    let off_surface = [vec3(-1.55 * scale, 0.0, 0.0)];

    assert!(all(
        &boundary_region_mask(&owner, &region, &face, None).expect("mask")
    ));
    assert!(!any(
        &boundary_region_mask(&owner, &region, &off_surface, None).expect("mask")
    ));
}

#[test]
fn generic_leaf_patches_make_pyramid_pickable() {
    let pyramid = Node::with_id(
        "p",
        7,
        Shape::Pyramid(Pyramid::new(Vec3::ZERO, 0.45, 0.6, IDENTITY_FRAME).expect("pyramid")),
    );
    let patches = surface_patches_for_root(&pyramid);
    assert_eq!(patches.len(), 1);
    assert_eq!(patches[0].owner_object_id, 7);
    assert_eq!(patches[0].patch_id, "surface");

    let hit = pick_boundary_patch(
        &pyramid,
        vec3(0.1, 0.05, 3.0),
        vec3(0.0, 0.0, -1.0),
        0.0008,
        100.0,
    )
    .expect("hit");
    assert_eq!(hit.owner_object_id, 7);
    assert_eq!(hit.patch_id, "surface");
}

// --- split_boundary_region (test_boundary_split.py) ---

fn scene_with_whole_surface_region() -> (SceneDocument, ObjectId, u32) {
    let document = SceneDocument::default_scene().expect("default scene");
    let mut document = document;
    let mut box_id = 0;
    for (id, _parent) in document.walk() {
        if matches!(
            document.object(id).expect("object").payload,
            ScenePayload::Box3(_)
        ) {
            box_id = id;
        }
    }
    let region_id = document
        .add_boundary_region(box_id, None, None, None)
        .expect("region");
    (document, box_id, region_id)
}

#[test]
fn split_replaces_parent_and_keeps_ghost_out_of_scene() {
    let (mut document, _box_id, region_id) = scene_with_whole_surface_region();
    let objects_before = document.live_ids();
    let ghost = ghost_sphere(vec3(-1.6, 0.0, 0.0), 0.5);

    let (inside_id, outside_id) = document
        .split_boundary_region(region_id, &ghost, None)
        .expect("split");

    assert!(!document
        .boundary_regions
        .iter()
        .any(|region| region.object_id == region_id));
    let inside = document
        .boundary_regions
        .iter()
        .find(|region| region.object_id == inside_id)
        .expect("inside");
    let outside = document
        .boundary_regions
        .iter()
        .find(|region| region.object_id == outside_id)
        .expect("outside");
    let sides = [
        inside.cuts.last().expect("cut").side,
        outside.cuts.last().expect("cut").side,
    ];
    assert!(sides.contains(&CutSide::Inside) && sides.contains(&CutSide::Outside));
    assert_eq!(inside.cuts.len(), 1);
    assert_eq!(outside.cuts.len(), 1);
    assert_eq!(document.live_ids(), objects_before, "ghost never became a node");
    let fluid = document.fluid_domain.as_ref().expect("fluid");
    use caso_kernel::scene::TagRef;
    assert!(!fluid.tags.contains(&TagRef::Region(region_id)));
    assert!(fluid.tags.contains(&TagRef::Region(inside_id)));
    assert!(fluid.tags.contains(&TagRef::Region(outside_id)));
}

#[test]
fn nested_split_composes_chains_and_partitions() {
    let (mut document, _box_id, region_id) = scene_with_whole_surface_region();
    let root_id = document.fluid_domain.as_ref().expect("fluid").root;
    let root = document.build_node(root_id).expect("root");
    let sphere = ghost_sphere(vec3(-1.6, 0.0, 0.0), 0.5);
    let (inside_id, outside_id) = document
        .split_boundary_region(region_id, &sphere, None)
        .expect("split 1");

    let upper = ghost_box(vec3(0.0, 1.0, 0.0), vec3(4.0, 1.0, 1.0));
    let (top_id, bottom_id) = document
        .split_boundary_region(outside_id, &upper, None)
        .expect("split 2");

    let find = |id: u32| {
        document
            .boundary_regions
            .iter()
            .find(|region| region.object_id == id)
            .expect("region")
    };
    assert_eq!(find(top_id).cuts.len(), 2);
    assert_eq!(find(bottom_id).cuts.len(), 2);
    let samples = face_points(-1.6, 9);
    let inside_mask =
        boundary_region_mask(&root, find(inside_id), &samples, None).expect("mask");
    let top_mask = boundary_region_mask(&root, find(top_id), &samples, None).expect("mask");
    let bottom_mask =
        boundary_region_mask(&root, find(bottom_id), &samples, None).expect("mask");
    for i in 0..samples.len() {
        let total =
            inside_mask[i] as usize + top_mask[i] as usize + bottom_mask[i] as usize;
        assert_eq!(total, 1, "the three leaves partition the face samples");
    }
}

#[test]
fn missed_knife_refuses_the_split_and_keeps_the_parent() {
    let (mut document, _box_id, region_id) = scene_with_whole_surface_region();
    let regions_before = document.boundary_regions.clone();
    let faraway = ghost_sphere(vec3(50.0, 50.0, 50.0), 0.1);

    let error = document
        .split_boundary_region(region_id, &faraway, None)
        .expect_err("must refuse");
    assert!(error.to_string().contains("does not cross"));
    assert_eq!(document.boundary_regions, regions_before);
}

#[test]
fn legacy_selector_region_converts_into_chain() {
    let (mut document, box_id, _region_id) = scene_with_whole_surface_region();
    // A legacy region referencing a scene sphere as its selector.
    let selector_id = document.add_primitive("sphere", 1.0).expect("selector");
    if let ScenePayload::Sphere(sphere) = &mut document
        .object_mut(selector_id)
        .expect("selector object")
        .payload
    {
        sphere.center = vec3(-1.6, 0.0, 0.0);
    }
    let legacy_object_id = document.allocate_object_id().expect("id");
    let mut legacy = BoundaryRegion::new("legacy", legacy_object_id, box_id);
    legacy.patch_id = Some("-X".to_string());
    legacy.selector_id = Some(format!("selector:{selector_id}"));
    legacy.selector_type = Some("surface_sdf_subregion".to_string());
    legacy.selector_side = CutSide::Inside;
    document.boundary_regions.push(legacy);
    if let Some(fluid) = document.fluid_domain.as_mut() {
        fluid.tags.push(caso_kernel::scene::TagRef::Region(legacy_object_id));
        fluid.selectors.push(selector_id);
    }

    let knife = ghost_box(vec3(0.0, 1.0, 0.0), vec3(4.0, 1.0, 1.0));
    let (top_id, _bottom_id) = document
        .split_boundary_region(legacy_object_id, &knife, None)
        .expect("split legacy");
    let top = document
        .boundary_regions
        .iter()
        .find(|region| region.object_id == top_id)
        .expect("top");

    assert_eq!(top.cuts.len(), 2, "converted legacy cut + the new one");
    assert_eq!(top.cuts[0].side, CutSide::Inside);
}
