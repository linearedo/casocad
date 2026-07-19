//! 2D boundary regions (design_docs/boundary_region_2d.md): curve patches,
//! the ray∩workplane pick, edge/outline scope, and the point knife. Fixture:
//! a planar flow case on the z = 0 workplane — flowbox rectangle (center
//! (0, 0), half 2×1) minus a circle obstacle (center (0.5, 0), r 0.3),
//! merged the way coplanar 2D booleans merge in the scene: one placed node
//! whose profile is Binary{rect, Difference, circle} with the operand nodes
//! kept in `sources`.

use caso_kernel::boundary::{BoundaryCut, BoundaryRegion, CutSide};
use caso_kernel::boundary_ops::{
    boundary_region_mask, pick_boundary_patch, pick_boundary_patch_with_radius,
    pick_outline_point, surface_patches_for_root,
};
use caso_kernel::boundary_paths::{point_knife, straight_knife, workplane_normal};
use caso_kernel::sdf::node::{Node, Shape};
use caso_kernel::sdf::placed::PlacedSdf2D;
use caso_kernel::sdf::primitives_1d::BooleanOp1D;
use caso_kernel::sdf::primitives_2d::Profile2D;
use caso_kernel::sdf::primitives_3d::Sphere;
use caso_kernel::vec3::{vec3, Vec3};

const PICK_TOLERANCE: f64 = 0.0008;
const PICK_TRAVEL: f64 = 100.0;

fn fixture_root() -> Node {
    let rect_profile = Profile2D::Rectangle {
        center: [0.0, 0.0],
        half_size: [2.0, 1.0],
    };
    let circle_profile = Profile2D::Circle {
        center: [0.5, 0.0],
        radius: 0.3,
    };
    let axis_u = vec3(1.0, 0.0, 0.0);
    let axis_v = vec3(0.0, 1.0, 0.0);
    let rect = Node::with_id(
        "flowbox",
        10,
        Shape::PlacedSdf2D(
            PlacedSdf2D::new(rect_profile.clone(), Vec3::ZERO, axis_u, axis_v, Vec::new())
                .expect("rect"),
        ),
    );
    let circle = Node::with_id(
        "obstacle",
        11,
        Shape::PlacedSdf2D(
            PlacedSdf2D::new(circle_profile.clone(), Vec3::ZERO, axis_u, axis_v, Vec::new())
                .expect("circle"),
        ),
    );
    let merged = Profile2D::Binary {
        left: Box::new(rect_profile),
        right: Box::new(Profile2D::Offset {
            child: Box::new(circle_profile),
            offset: [0.0, 0.0],
        }),
        operation: BooleanOp1D::Difference,
        smoothing: 0.1,
    };
    Node::with_id(
        "domain",
        12,
        Shape::PlacedSdf2D(
            PlacedSdf2D::new(merged, Vec3::ZERO, axis_u, axis_v, vec![rect, circle])
                .expect("merged"),
        ),
    )
}

fn region_for(root: &Node, patch_id: &str) -> BoundaryRegion {
    let patch = surface_patches_for_root(root)
        .into_iter()
        .find(|patch| patch.patch_id == patch_id)
        .unwrap_or_else(|| panic!("patch {patch_id} exists"));
    let mut region = BoundaryRegion::new(patch_id, 100, patch.owner_object_id);
    region.patch_id = Some(patch.patch_id.clone());
    region.patch_type = Some(patch.patch_type.clone());
    region.outside_direction = patch.outside_direction;
    region
}

fn pick_down(root: &Node, x: f64, y: f64) -> Option<caso_kernel::boundary_ops::BoundaryPatchHit> {
    pick_boundary_patch(
        root,
        vec3(x, y, 5.0),
        vec3(0.0, 0.0, -1.0),
        PICK_TOLERANCE,
        PICK_TRAVEL,
    )
}

fn left_edge_samples(count: usize) -> Vec<Vec3> {
    (0..count)
        .map(|i| vec3(-2.0, -0.99 + 1.98 * (i as f64) / ((count - 1) as f64), 0.0))
        .collect()
}

#[test]
fn curve_patches_name_edges_and_the_cut_outline() {
    let root = fixture_root();
    let patches = surface_patches_for_root(&root);
    let ids: Vec<&str> = patches.iter().map(|patch| patch.patch_id.as_str()).collect();
    for wanted in [
        "flowbox.-U",
        "flowbox.+U",
        "flowbox.-V",
        "flowbox.+V",
        "cut_surface.obstacle.outline",
    ] {
        assert!(ids.contains(&wanted), "missing patch {wanted} in {ids:?}");
    }
    // Every patch reports the merged root as its provenance owner — the one
    // leaf `boundary_owner_ids` yields for a 2D domain.
    assert!(patches.iter().all(|patch| patch.owner_object_id == 12));
}

#[test]
fn pick_names_each_rectangle_edge() {
    let root = fixture_root();
    for (x, y, wanted) in [
        (-2.0, 0.0, "flowbox.-U"),
        (2.0, 0.3, "flowbox.+U"),
        (-1.0, -1.0, "flowbox.-V"),
        (-1.0, 1.0, "flowbox.+V"),
    ] {
        let hit = pick_down(&root, x, y).unwrap_or_else(|| panic!("hit at ({x}, {y})"));
        assert_eq!(hit.patch_id, wanted);
        assert_eq!(hit.owner_object_id, 12);
        // The hit point is snapped onto the outline, on the workplane.
        assert!(root.eval_point(hit.point).abs() < 1.0e-6);
        assert!(hit.point.z.abs() < 1.0e-9);
    }
}

#[test]
fn pick_prefers_the_cut_surface_on_the_obstacle() {
    let root = fixture_root();
    let hit = pick_down(&root, 0.2, 0.0).expect("obstacle hit");
    assert_eq!(hit.patch_id, "cut_surface.obstacle.outline");
    assert_eq!(hit.patch_type, "cut_surface");
    assert!((hit.point - vec3(0.2, 0.0, 0.0)).length() < 1.0e-6);
}

#[test]
fn pick_misses_far_from_the_outline() {
    let root = fixture_root();
    assert!(pick_down(&root, 1.5, 0.35).is_none());
}

#[test]
fn edge_scope_limits_membership_to_its_edge() {
    let root = fixture_root();
    let left = region_for(&root, "flowbox.-U");
    let cut = region_for(&root, "cut_surface.obstacle.outline");
    let points = vec![
        vec3(-2.0, 0.5, 0.0),  // left edge
        vec3(-2.0, -0.5, 0.0), // left edge
        vec3(0.0, -1.0, 0.0),  // bottom edge
        vec3(0.2, 0.0, 0.0),   // obstacle outline
    ];
    let left_mask = boundary_region_mask(&root, &left, &points, None).expect("mask");
    assert_eq!(left_mask, vec![true, true, false, false]);
    let cut_mask = boundary_region_mask(&root, &cut, &points, None).expect("mask");
    assert_eq!(cut_mask, vec![false, false, false, true]);
}

#[test]
fn point_knife_partitions_an_edge_region_exactly() {
    let root = fixture_root();
    let parent = region_for(&root, "flowbox.-U");
    let click = vec3(-2.0, 0.25, 0.0);
    let ghost = point_knife(&root, click).expect("point knife");
    let child = |side: CutSide| {
        let mut child = parent.clone();
        child.cuts.push(BoundaryCut {
            side,
            ghost: ghost.clone(),
        });
        child
    };
    let points = left_edge_samples(200);
    let parent_mask = boundary_region_mask(&root, &parent, &points, None).expect("mask");
    assert!(parent_mask.iter().all(|hit| *hit));
    let inside = boundary_region_mask(&root, &child(CutSide::Inside), &points, None)
        .expect("inside");
    let outside = boundary_region_mask(&root, &child(CutSide::Outside), &points, None)
        .expect("outside");
    let mut inside_count = 0;
    let mut outside_count = 0;
    for index in 0..points.len() {
        // Inside/Outside complement each other point-by-point: an exact,
        // crack- and overlap-free partition of the parent arc.
        assert_ne!(inside[index], outside[index], "point {index} must be in exactly one child");
        if inside[index] {
            inside_count += 1;
        } else {
            outside_count += 1;
        }
    }
    assert!(inside_count > 0 && outside_count > 0, "the click splits the arc");
    // The split lands at the click: each child is one contiguous y-interval
    // ending within a classifier tolerance of y = 0.25.
    let inside_ys: Vec<f64> = points
        .iter()
        .zip(&inside)
        .filter(|(_, hit)| **hit)
        .map(|(p, _)| p.y)
        .collect();
    let outside_ys: Vec<f64> = points
        .iter()
        .zip(&outside)
        .filter(|(_, hit)| **hit)
        .map(|(p, _)| p.y)
        .collect();
    let inside_span = (inside_ys[0], *inside_ys.last().expect("nonempty"));
    let outside_span = (outside_ys[0], *outside_ys.last().expect("nonempty"));
    let (below, above) = if inside_span.1 < outside_span.0 {
        (inside_span, outside_span)
    } else {
        (outside_span, inside_span)
    };
    // The boundary between the children lies within the classifier's
    // scale-relative tolerance band around the click (mask criterion 4 is
    // `eval <= tol`), so compare against the click with that slack.
    assert!(below.1 < above.0, "children overlap in y");
    assert!((below.1 - 0.25).abs() < 0.05 && (above.0 - 0.25).abs() < 0.05);
}

#[test]
fn segment_knife_2d_is_click_order_independent() {
    let root = fixture_root();
    let normal = workplane_normal(&root).expect("plane normal");
    let first = vec3(-2.0, -0.5, 0.0);
    let second = vec3(2.0, 0.5, 0.0);
    let forward = straight_knife(&root, first, second, normal).expect("forward");
    let backward = straight_knife(&root, second, first, normal).expect("backward");
    let probes = [
        vec3(-2.0, 0.9, 0.0),
        vec3(-2.0, -0.9, 0.0),
        vec3(2.0, 0.9, 0.0),
        vec3(2.0, -0.9, 0.0),
    ];
    let signs: Vec<(bool, bool)> = probes
        .iter()
        .map(|p| (forward.eval_point(*p) <= 0.0, backward.eval_point(*p) <= 0.0))
        .collect();
    let all_equal = signs.iter().all(|(f, b)| f == b);
    let all_flipped = signs.iter().all(|(f, b)| f != b);
    assert!(all_equal || all_flipped, "click order changed the partition: {signs:?}");
}

#[test]
fn point_knife_refuses_3d_domains() {
    let sphere = Node::with_id(
        "sphere",
        1,
        Shape::Sphere(Sphere::new(Vec3::ZERO, 1.0).expect("sphere")),
    );
    let error = point_knife(&sphere, vec3(1.0, 0.0, 0.0)).expect_err("3D root refused");
    assert!(error.to_string().contains("2D"), "unexpected error: {error}");
}

/// End to end through the document, on the same call path the UI takes:
/// drag-create rectangle and circle, boolean difference (which merges the
/// coplanar profiles into one placed node), mark the fluid domain, pick an
/// edge, create the region, split it with a point knife.
#[test]
fn document_flow_creates_and_splits_a_2d_region() {
    use caso_kernel::roles::DomainKind;
    use caso_kernel::scene::SceneDocument;

    let mut document = SceneDocument::new();
    let rect = document
        .add_primitive_from_drag("rectangle", vec3(-2.0, -1.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
        .expect("rectangle");
    let circle = document
        .add_primitive_from_drag("circle", vec3(0.2, -0.3, 0.0), vec3(0.8, 0.3, 0.0), 1.0)
        .expect("circle");
    let domain = document.combine(rect, circle, "difference").expect("difference");
    document
        .set_domain_root(domain, DomainKind::Fluid)
        .expect("fluid domain");
    let fluid_root = document.fluid_domain.as_ref().expect("fluid").root;
    let root = document.build_node(fluid_root).expect("root node");
    assert_eq!(root.dimension(), 2, "the merged domain stays 2D");

    let hit = pick_down(&root, -2.0, 0.5).expect("left edge hit");
    assert!(hit.patch_id.ends_with("-U"), "unexpected patch {}", hit.patch_id);
    let region_id = document
        .add_boundary_region(
            hit.owner_object_id,
            hit.outside_direction,
            Some(&hit.patch_id),
            Some(&hit.patch_type),
        )
        .expect("region created");

    let ghost = point_knife(&root, vec3(-2.0, 0.25, 0.0)).expect("point knife");
    let (first, second) = document
        .split_boundary_region(region_id, &ghost, None)
        .expect("split");
    let ids: Vec<u32> = document
        .boundary_regions
        .iter()
        .map(|region| region.object_id)
        .collect();
    assert!(ids.contains(&first) && ids.contains(&second));
    assert!(!ids.contains(&region_id), "children replace the parent");
}

/// Regions are not fluid-only: the same flow works on a SOLID domain with
/// no fluid domain in the document at all, and survives save/load with the
/// region's domain recorded.
#[test]
fn solid_domain_regions_work_without_any_fluid_domain() {
    use caso_kernel::roles::DomainKind;
    use caso_kernel::scene::SceneDocument;
    use caso_kernel::serialization::{load_scene_from_str, save_scene_to_string};

    let mut document = SceneDocument::new();
    let rect = document
        .add_primitive_from_drag("rectangle", vec3(-2.0, -1.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
        .expect("rectangle");
    let circle = document
        .add_primitive_from_drag("circle", vec3(0.2, -0.3, 0.0), vec3(0.8, 0.3, 0.0), 1.0)
        .expect("circle");
    let domain = document.combine(rect, circle, "difference").expect("difference");
    document
        .set_domain_root(domain, DomainKind::Solid)
        .expect("solid domain");
    assert!(document.fluid_domain.is_none(), "no fluid domain in this document");

    let root = document.build_node(domain).expect("root node");
    let hit = pick_down(&root, -2.0, 0.5).expect("left edge hit");
    let region_id = document
        .add_boundary_region(
            hit.owner_object_id,
            hit.outside_direction,
            Some(&hit.patch_id),
            Some(&hit.patch_type),
        )
        .expect("region on the solid domain");
    let region = document
        .boundary_regions
        .iter()
        .find(|region| region.object_id == region_id)
        .expect("stored");
    assert_eq!(region.domain_root, Some(domain));

    let ghost = point_knife(&root, vec3(-2.0, 0.25, 0.0)).expect("point knife");
    let (first, second) = document
        .split_boundary_region(region_id, &ghost, None)
        .expect("split without a fluid domain");

    let saved = save_scene_to_string(&document).expect("save");
    let loaded = load_scene_from_str(&saved).expect("load");
    assert_eq!(loaded.boundary_regions.len(), 2);
    for region in &loaded.boundary_regions {
        assert!(
            [first, second].contains(&region.object_id)
                || loaded.region_domain_root(region).is_some(),
            "children survive the roundtrip"
        );
        assert!(
            loaded.region_domain_root(region).is_some(),
            "the region's domain is recorded in the file"
        );
    }
}

/// A solid marked inside the fluid difference (conjugate heat transfer
/// setup): each domain gets its own region on the SAME physical wall — the
/// fluid's cut surface and the solid's side wall — and meshing exposes
/// both under their own domains.
#[test]
fn nested_solid_and_fluid_each_carry_their_own_wall_region() {
    use caso_kernel::meshing::meshable_domains_from_document;
    use caso_kernel::roles::DomainKind;
    use caso_kernel::scene::{ScenePayload, SceneDocument};

    let mut document = SceneDocument::default_scene().expect("default scene");
    let fluid_root = document.fluid_domain.as_ref().expect("fluid").root;
    let cylinder_id = document
        .walk()
        .into_iter()
        .find_map(|(id, _parent)| {
            matches!(
                document.object(id).expect("object").payload,
                ScenePayload::Cylinder(_)
            )
            .then_some(id)
        })
        .expect("cylinder");
    document
        .set_domain_root(cylinder_id, DomainKind::Solid)
        .expect("solid mark");

    let fluid_region = document
        .add_boundary_region_in(
            fluid_root,
            cylinder_id,
            None,
            Some("cut_surface.side_wall"),
            Some("cut_surface"),
        )
        .expect("fluid wall region");
    let solid_region = document
        .add_boundary_region_in(
            cylinder_id,
            cylinder_id,
            None,
            Some("side_wall"),
            Some("side_wall"),
        )
        .expect("solid wall region");
    let domain_of = |id: u32| {
        document
            .boundary_regions
            .iter()
            .find(|region| region.object_id == id)
            .and_then(|region| document.region_domain_root(region))
    };
    assert_eq!(domain_of(fluid_region), Some(fluid_root));
    assert_eq!(domain_of(solid_region), Some(cylinder_id));

    let domains = meshable_domains_from_document(&document).expect("meshable");
    let solid = domains.get("cylinder_obstacle").expect("one solid domain");
    assert_eq!(
        solid
            .boundary_regions
            .iter()
            .map(|region| region.name.as_str())
            .collect::<Vec<_>>(),
        vec!["cylinder_obstacle side_wall"],
    );
    let fluid = domains.get("von_karman_fluid").expect("one fluid domain");
    assert!(fluid
        .boundary_regions
        .iter()
        .any(|region| region.name.contains("cut_surface.side_wall")));
}

/// Like `fixture_root`, but with the obstacle circle close to the +U edge
/// so a hover between them has both curves inside the pick radius.
fn fixture_with_near_edge_obstacle() -> Node {
    let rect_profile = Profile2D::Rectangle {
        center: [0.0, 0.0],
        half_size: [2.0, 1.0],
    };
    let circle_profile = Profile2D::Circle {
        center: [1.64, 0.0],
        radius: 0.3,
    };
    let axis_u = vec3(1.0, 0.0, 0.0);
    let axis_v = vec3(0.0, 1.0, 0.0);
    let rect = Node::with_id(
        "flowbox",
        10,
        Shape::PlacedSdf2D(
            PlacedSdf2D::new(rect_profile.clone(), Vec3::ZERO, axis_u, axis_v, Vec::new())
                .expect("rect"),
        ),
    );
    let circle = Node::with_id(
        "obstacle",
        11,
        Shape::PlacedSdf2D(
            PlacedSdf2D::new(circle_profile.clone(), Vec3::ZERO, axis_u, axis_v, Vec::new())
                .expect("circle"),
        ),
    );
    let merged = Profile2D::Binary {
        left: Box::new(rect_profile),
        right: Box::new(Profile2D::Offset {
            child: Box::new(circle_profile),
            offset: [0.0, 0.0],
        }),
        operation: BooleanOp1D::Difference,
        smoothing: 0.1,
    };
    Node::with_id(
        "domain",
        12,
        Shape::PlacedSdf2D(
            PlacedSdf2D::new(merged, Vec3::ZERO, axis_u, axis_v, vec![rect, circle])
                .expect("merged"),
        ),
    )
}

/// The nearest curve wins the hover; a cut surface no longer steals a pick
/// from a clearly closer regular edge (it is preferred only on near-ties,
/// the coincident-boundary case).
#[test]
fn pick_is_nearest_wins_between_cut_and_edge() {
    let root = fixture_with_near_edge_obstacle();
    // Between circle (rightmost point x = 1.94) and the +U edge (x = 2):
    // 0.025 from the edge, 0.035 from the circle — both within the radius.
    let hit = pick_down(&root, 1.975, 0.0).expect("hit between the curves");
    assert_eq!(hit.patch_id, "flowbox.+U", "the closer edge wins");
    // Close to the circle only: the cut surface still picks normally.
    let hit = pick_down(&root, 1.93, 0.0).expect("hit at the circle");
    assert_eq!(hit.patch_id, "cut_surface.obstacle.outline");
}

/// The caller-supplied pick radius bounds the hover snap distance.
#[test]
fn pick_radius_parameter_is_honored() {
    let root = fixture_root();
    let pick = |radius: f64| {
        pick_boundary_patch_with_radius(
            &root,
            vec3(-1.99, 0.5, 5.0),
            vec3(0.0, 0.0, -1.0),
            PICK_TOLERANCE,
            PICK_TRAVEL,
            Some(radius),
        )
    };
    // The ray's plane hit is 0.01 from the -U edge.
    assert!(pick(0.005).is_none(), "radius below the distance: no hit");
    let hit = pick(0.02).expect("radius above the distance: hit");
    assert_eq!(hit.patch_id, "flowbox.-U");
}

/// An edge region ends at its corners: a point on the neighbor edge just
/// past the corner belongs to the neighbor only (the scope box used to
/// bleed a scale-relative band around the corner).
#[test]
fn edge_scope_stops_at_the_corner() {
    let root = fixture_root();
    let left = region_for(&root, "flowbox.-U");
    let bottom = region_for(&root, "flowbox.-V");
    let corner = vec3(-2.0, -1.0, 0.0);
    let past_corner = vec3(-1.992, -1.0, 0.0); // on the bottom edge, 8 mm in
    let points = vec![corner, past_corner];
    let left_mask = boundary_region_mask(&root, &left, &points, None).expect("mask");
    let bottom_mask = boundary_region_mask(&root, &bottom, &points, None).expect("mask");
    assert_eq!(bottom_mask, vec![true, true], "the bottom edge owns both");
    assert!(left_mask[0], "the shared corner belongs to both edges");
    assert!(
        !left_mask[1],
        "past the corner the left edge region must not claim the bottom edge"
    );
}

#[test]
fn outline_point_pick_snaps_onto_the_outline() {
    let root = fixture_root();
    // A ray slightly inside the left edge snaps onto it.
    let snapped = pick_outline_point(&root, vec3(-1.95, 0.4, 5.0), vec3(0.0, 0.0, -1.0))
        .expect("plane hit");
    assert!(root.eval_point(snapped).abs() < 1.0e-9);
    assert!((snapped.x + 2.0).abs() < 1.0e-6);
    // Far from the outline the raw in-plane point comes back (forgiving
    // knife endpoints).
    let raw = pick_outline_point(&root, vec3(1.5, 0.35, 5.0), vec3(0.0, 0.0, -1.0))
        .expect("plane hit");
    assert!((raw - vec3(1.5, 0.35, 0.0)).length() < 1.0e-9);
}
