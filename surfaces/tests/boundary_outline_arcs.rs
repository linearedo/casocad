//! Patch-exact boundary arcs (design_docs/boundary_region_2d_precision.md):
//! the highlight polyline for a curve patch must run corner-to-corner on
//! straight edges and end exactly on the boolean junctions — the resampled
//! contour-ring path lost 4-35% of an edge and chamfered every corner.
//! Fixture mirrors the reported case: a quadratic bezier surface minus a
//! polygon.

use caso_kernel::boundary_ops::surface_patches_for_root;
use caso_kernel::sdf::node::{Node, Shape};
use caso_kernel::sdf::placed::PlacedSdf2D;
use caso_kernel::sdf::primitives_1d::BooleanOp1D;
use caso_kernel::sdf::primitives_2d::Profile2D;
use caso_kernel::vec3::{vec3, Vec3};
use caso_surfaces::boundary_outline::curve_patch_arcs;

const RESOLUTION: usize = 192;

/// Closed bezier blob roughly 8 x 6 around the origin.
fn blob_profile() -> Profile2D {
    Profile2D::quadratic_bezier_surface(vec![
        [-4.0, 0.0],
        [-4.0, 3.5],
        [0.0, 3.0],
        [4.0, 3.5],
        [4.0, 0.0],
        [4.0, -3.5],
        [0.0, -3.0],
        [-4.0, -3.5],
        [-4.0, 0.0],
    ])
    .expect("bezier")
}

/// blob minus `polygon`, merged the way coplanar 2D booleans merge in the
/// scene: one placed node, operands kept in `sources`.
fn fixture(polygon: Vec<[f64; 2]>) -> Node {
    let blob = blob_profile();
    let poly = Profile2D::polygon(polygon).expect("polygon");
    let axis_u = vec3(1.0, 0.0, 0.0);
    let axis_v = vec3(0.0, 1.0, 0.0);
    let blob_node = Node::with_id(
        "blob",
        10,
        Shape::PlacedSdf2D(
            PlacedSdf2D::new(blob.clone(), Vec3::ZERO, axis_u, axis_v, Vec::new()).expect("blob"),
        ),
    );
    let poly_node = Node::with_id(
        "hole",
        11,
        Shape::PlacedSdf2D(
            PlacedSdf2D::new(poly.clone(), Vec3::ZERO, axis_u, axis_v, Vec::new()).expect("hole"),
        ),
    );
    let merged = Profile2D::Binary {
        left: Box::new(blob),
        right: Box::new(Profile2D::Offset {
            child: Box::new(poly),
            offset: [0.0, 0.0],
        }),
        operation: BooleanOp1D::Difference,
        smoothing: 0.1,
    };
    Node::with_id(
        "domain",
        12,
        Shape::PlacedSdf2D(
            PlacedSdf2D::new(merged, Vec3::ZERO, axis_u, axis_v, vec![blob_node, poly_node])
                .expect("merged"),
        ),
    )
}

fn arcs_for(root: &Node, patch_id: &str) -> Vec<Vec<Vec3>> {
    let patch = surface_patches_for_root(root)
        .into_iter()
        .find(|patch| patch.patch_id == patch_id)
        .unwrap_or_else(|| panic!("patch {patch_id} exists"));
    curve_patch_arcs(root, &patch, RESOLUTION)
}

fn total_length(arcs: &[Vec<Vec3>]) -> f64 {
    arcs.iter()
        .map(|arc| arc.windows(2).map(|pair| (pair[1] - pair[0]).length()).sum::<f64>())
        .sum()
}

#[test]
fn interior_hole_edges_run_corner_to_corner() {
    // side-0.8 square hole fully inside the blob
    let corners = [[1.1, 0.1], [1.9, 0.1], [1.9, 0.9], [1.1, 0.9]];
    let root = fixture(corners.to_vec());
    for (index, corner) in corners.iter().enumerate() {
        let next = corners[(index + 1) % corners.len()];
        let arcs = arcs_for(&root, &format!("cut_surface.hole.edge_{index}"));
        assert_eq!(arcs.len(), 1, "edge_{index} is one unbroken arc");
        let arc = &arcs[0];
        let start = vec3(corner[0], corner[1], 0.0);
        let end = vec3(next[0], next[1], 0.0);
        assert!(
            (arc[0] - start).length() < 1.0e-12 && (arc[arc.len() - 1] - end).length() < 1.0e-12,
            "edge_{index} arc runs exactly corner to corner"
        );
        let length = total_length(&arcs);
        assert!(
            (length - 0.8).abs() < 1.0e-9,
            "edge_{index} highlights its full 0.8 length, got {length}"
        );
    }
}

#[test]
fn small_hole_edges_keep_their_full_length() {
    // side-0.2 hole: the contour-ring path used to highlight only 0.131
    let root = fixture(vec![[1.4, 0.4], [1.6, 0.4], [1.6, 0.6], [1.4, 0.6]]);
    for index in 0..4 {
        let arcs = arcs_for(&root, &format!("cut_surface.hole.edge_{index}"));
        let length = total_length(&arcs);
        assert!(
            (length - 0.2).abs() < 1.0e-9,
            "edge_{index} highlights its full 0.2 length, got {length}"
        );
    }
}

#[test]
fn untouched_outline_is_one_closed_ring() {
    let root = fixture(vec![[1.1, 0.1], [1.9, 0.1], [1.9, 0.9], [1.1, 0.9]]);
    let arcs = arcs_for(&root, "blob.outline");
    assert_eq!(arcs.len(), 1, "an interior hole never splits the outline");
    let arc = &arcs[0];
    assert!(
        (arc[0] - arc[arc.len() - 1]).length() < 1.0e-12,
        "the outline arc closes onto itself"
    );
    assert!(total_length(&arcs) > 20.0, "full blob perimeter (~23.6)");
}

#[test]
fn bite_arcs_end_exactly_on_the_junctions() {
    // polygon overlapping the blob's right lobe: two edges are partially
    // swallowed, one lies fully inside, one fully outside
    let root = fixture(vec![[3.0, -1.0], [5.0, -1.0], [5.0, 1.0], [3.0, 1.0]]);
    let blob = Node::with_id(
        "blob_ghost",
        0,
        Shape::PlacedSdf2D(
            PlacedSdf2D::new(
                blob_profile(),
                Vec3::ZERO,
                vec3(1.0, 0.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                Vec::new(),
            )
            .expect("blob"),
        ),
    );

    // left edge (3,1)->(3,-1) lies fully inside: full 2.0 length (the ring
    // path showed 1.927)
    let left = arcs_for(&root, "cut_surface.hole.edge_3");
    assert!((total_length(&left) - 2.0).abs() < 1.0e-9, "full left edge");

    // right edge (5,-1)->(5,1) lies fully outside the blob: nothing
    assert!(arcs_for(&root, "cut_surface.hole.edge_1").is_empty());

    // bottom edge (3,-1)->(5,-1): starts at the corner, ends where the blob
    // outline crosses y = -1 — the free end must sit on BOTH curves
    let bottom = arcs_for(&root, "cut_surface.hole.edge_0");
    assert_eq!(bottom.len(), 1);
    let arc = &bottom[0];
    assert!((arc[0] - vec3(3.0, -1.0, 0.0)).length() < 1.0e-12, "starts at the corner");
    let junction = arc[arc.len() - 1];
    assert!((junction.y + 1.0).abs() < 1.0e-12, "junction stays on the edge line");
    assert!(
        blob.eval_point(junction).abs() < 1.0e-6,
        "junction lies on the blob outline (root-found, not sampled): |blob| = {}",
        blob.eval_point(junction).abs()
    );

    // the blob outline itself is clipped at the same polygon boundary: every
    // open arc end lies on the subtracted polygon's outline. Tolerance is
    // the chord sagitta (~6e-6 at this sampling) — the junction is exactly
    // as accurate as the drawn polyline can be, since it lies ON a chord.
    let outline = arcs_for(&root, "blob.outline");
    assert!(!outline.is_empty());
    let poly = Node::with_id(
        "poly_ghost",
        0,
        Shape::PlacedSdf2D(
            PlacedSdf2D::new(
                Profile2D::polygon(vec![[3.0, -1.0], [5.0, -1.0], [5.0, 1.0], [3.0, 1.0]])
                    .expect("polygon"),
                Vec3::ZERO,
                vec3(1.0, 0.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                Vec::new(),
            )
            .expect("poly"),
        ),
    );
    for arc in &outline {
        let closed = (arc[0] - arc[arc.len() - 1]).length() < 1.0e-12;
        assert!(!closed, "the bite splits the outline into open arcs");
        for end in [arc[0], arc[arc.len() - 1]] {
            assert!(
                poly.eval_point(end).abs() < 1.0e-4,
                "outline arc end lies on the polygon boundary, off by {}",
                poly.eval_point(end).abs()
            );
        }
    }
}
