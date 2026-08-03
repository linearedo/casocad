use caso_delaunay::predicates::{incircle, insphere, orient2d, orient3d, Sign};
use caso_delaunay::{
    refine_2d, refine_3d, tetrahedralize_3d, triangulate_2d, ConstraintPolicy, Error,
    FacetConstraint, Input2d, Input3d, Limits, Refine2d, Refine3d, SegmentConstraint,
};

fn limits() -> Limits {
    Limits {
        max_vertices: 2_000,
        max_cells: 8_000,
        max_iterations: 200,
    }
}

#[test]
fn exact_predicates_and_degeneracies_are_deterministic() {
    let epsilon = f64::EPSILON;
    assert_eq!(
        orient2d([0.0, 0.0], [1.0, epsilon], [2.0, 2.0 * epsilon]),
        Sign::Zero
    );
    assert_eq!(
        orient2d(
            [0.0, 0.0],
            [1.0, epsilon],
            [2.0, f64::from_bits((2.0 * epsilon).to_bits() + 1)]
        ),
        Sign::Positive
    );
    assert_eq!(
        incircle([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]),
        Sign::Zero
    );
    assert_ne!(
        orient3d(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ),
        Sign::Zero
    );
    assert_eq!(
        insphere(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ),
        Sign::Zero
    );

    let input = Input2d {
        points: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        constraints: Vec::new(),
    };
    assert_eq!(
        triangulate_2d(&input, limits(), || true).unwrap(),
        triangulate_2d(&input, limits(), || true).unwrap()
    );
}

#[test]
fn constrained_square_with_hole_refines_within_bounds() {
    let points = vec![
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [-0.25, -0.25],
        [0.25, -0.25],
        [0.25, 0.25],
        [-0.25, 0.25],
    ];
    let constraints = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
    ]
    .map(|vertices| SegmentConstraint {
        vertices,
        policy: ConstraintPolicy::Fixed,
    })
    .to_vec();
    let input = Input2d {
        points,
        constraints,
    };
    let mesh = refine_2d(
        &input,
        Refine2d {
            max_area: 0.2,
            min_angle_degrees: 0.0,
            limits: limits(),
        },
        || true,
    )
    .unwrap();
    assert!(mesh.vertices.len() > input.points.len());
    assert!(mesh.triangles.len() <= limits().max_cells);
    assert!(mesh.constraints.iter().all(|chain| !chain.is_empty()));
    for (index, adjacent) in mesh.adjacency.iter().enumerate() {
        assert!(adjacent.iter().flatten().all(|other| *other != index));
    }
}

#[test]
fn closed_polyhedral_cavity_recovers_facets_and_refines() {
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let facets = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]
        .map(|vertices| FacetConstraint {
            vertices,
            policy: ConstraintPolicy::Fixed,
        })
        .to_vec();
    let input = Input3d {
        points,
        segments: Vec::new(),
        facets,
    };
    let initial = tetrahedralize_3d(&input, limits(), || true).unwrap();
    assert_eq!(initial.tetrahedra.len(), 1);
    assert!(initial.facets.iter().all(|patch| patch.len() == 1));
    let refined = refine_3d(
        &input,
        Refine3d {
            max_volume: 0.03,
            limits: limits(),
        },
        || true,
    )
    .unwrap();
    assert!(refined.tetrahedra.len() > initial.tetrahedra.len());
    assert!(refined.facets.iter().all(|patch| !patch.is_empty()));
}

#[test]
fn invalid_inputs_cancellation_and_limits_are_typed() {
    let duplicate = Input2d {
        points: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        constraints: Vec::new(),
    };
    assert!(matches!(
        triangulate_2d(&duplicate, limits(), || true),
        Err(Error::DuplicateVertex { .. })
    ));
    let crossed = Input2d {
        points: vec![[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
        constraints: vec![
            SegmentConstraint {
                vertices: [0, 1],
                policy: ConstraintPolicy::Fixed,
            },
            SegmentConstraint {
                vertices: [2, 3],
                policy: ConstraintPolicy::Fixed,
            },
        ],
    };
    assert!(matches!(
        triangulate_2d(&crossed, limits(), || true),
        Err(Error::CrossedConstraints { .. })
    ));
    let triangle = Input2d {
        points: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        constraints: Vec::new(),
    };
    assert_eq!(
        triangulate_2d(&triangle, limits(), || false),
        Err(Error::Cancelled)
    );
    assert!(matches!(
        triangulate_2d(
            &triangle,
            Limits {
                max_vertices: 2,
                ..limits()
            },
            || true
        ),
        Err(Error::LimitExceeded("vertex"))
    ));
}
