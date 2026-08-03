use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::predicates::{insphere, orient3d, tie, Sign};
use crate::recover_3d::{recover_constraints, validate_constraints_3d};
use crate::{check_points, continued, Error, Input3d, Limits, Tetrahedralization3d};

pub(crate) type TetTopology = (Vec<[usize; 4]>, Vec<[Option<usize>; 4]>);

pub fn tetrahedralize_3d(
    input: &Input3d,
    limits: Limits,
    mut check: impl FnMut() -> bool,
) -> Result<Tetrahedralization3d, Error> {
    check_points(&input.points)?;
    if input.points.len() < 4 {
        return Err(Error::DegenerateInput(
            "3D tetrahedralization requires at least four vertices",
        ));
    }
    if input.points.len() > limits.max_vertices {
        return Err(Error::LimitExceeded("vertex"));
    }
    validate_constraints_3d(input)?;
    continued(&mut check)?;
    recover_constraints(input, limits, &mut check, |points, check| {
        bowyer_watson_3d(points, limits, check)
    })
}

pub(crate) fn bowyer_watson_3d(
    points: &[[f64; 3]],
    limits: Limits,
    check: &mut (impl FnMut() -> bool + ?Sized),
) -> Result<TetTopology, Error> {
    let (min, max) = bounds(points);
    let span = (max[0] - min[0]).max(max[1] - min[1]).max(max[2] - min[2]);
    if span == 0.0 || !span.is_finite() {
        return Err(Error::DegenerateInput(
            "3D vertices are coplanar or unbounded",
        ));
    }
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let scale = 64.0 * span;
    let mut vertices = points.to_vec();
    let first_super = vertices.len();
    vertices.extend([
        [center[0] - scale, center[1] - scale, center[2] - scale],
        [center[0] + scale, center[1] + scale, center[2] - scale],
        [center[0] + scale, center[1] - scale, center[2] + scale],
        [center[0] - scale, center[1] + scale, center[2] + scale],
    ]);
    if vertices[first_super..]
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(Error::DegenerateInput(
            "3D coordinate range cannot form a finite enclosing simplex",
        ));
    }
    let super_tet = oriented_tetrahedron(
        [
            first_super,
            first_super + 1,
            first_super + 2,
            first_super + 3,
        ],
        &vertices,
    )
    .ok_or(Error::DegenerateInput("invalid enclosing simplex"))?;
    let mut tetrahedra = vec![super_tet];
    let mut order = (0..points.len()).collect::<Vec<_>>();
    order.sort_by(|a, b| {
        points[*a][0]
            .total_cmp(&points[*b][0])
            .then(points[*a][1].total_cmp(&points[*b][1]))
            .then(points[*a][2].total_cmp(&points[*b][2]))
            .then(a.cmp(b))
    });
    for (step, point) in order.into_iter().enumerate() {
        if step.is_multiple_of(64) {
            continued(check)?;
        }
        let mut bad = Vec::new();
        for (index, tetrahedron) in tetrahedra.iter().enumerate() {
            let [a, b, c, d] = tetrahedron.map(|vertex| vertices[vertex]);
            let sphere = insphere(a, b, c, d, vertices[point]).ordering();
            let orientation = orient3d(a, b, c, d).ordering();
            let inside = sphere == orientation
                || (sphere == Ordering::Equal
                    && tie(&[
                        tetrahedron[0],
                        tetrahedron[1],
                        tetrahedron[2],
                        tetrahedron[3],
                        point,
                    ]) == Ordering::Greater);
            if inside {
                bad.push(index);
            }
        }
        let mut boundary = BTreeMap::<[usize; 3], usize>::new();
        for &index in &bad {
            for face in tetrahedron_faces(tetrahedra[index]) {
                *boundary.entry(ordered_face(face)).or_default() += 1;
            }
        }
        let bad = bad.into_iter().collect::<BTreeSet<_>>();
        tetrahedra = tetrahedra
            .into_iter()
            .enumerate()
            .filter_map(|(index, tetrahedron)| (!bad.contains(&index)).then_some(tetrahedron))
            .collect();
        for (face, incidence) in boundary {
            if incidence == 1 {
                if let Some(tetrahedron) =
                    oriented_tetrahedron([face[0], face[1], face[2], point], &vertices)
                {
                    tetrahedra.push(tetrahedron);
                }
            }
        }
        if tetrahedra.len() > limits.max_cells.saturating_mul(16).max(128) {
            return Err(Error::LimitExceeded("cell"));
        }
    }
    tetrahedra.retain(|tetrahedron| tetrahedron.iter().all(|vertex| *vertex < first_super));
    tetrahedra.sort_unstable();
    tetrahedra.dedup();
    if tetrahedra.is_empty() {
        return Err(Error::DegenerateInput("3D vertices are coplanar"));
    }
    if tetrahedra.len() > limits.max_cells {
        return Err(Error::LimitExceeded("cell"));
    }
    let adjacency = tetrahedron_adjacency(&tetrahedra)?;
    Ok((tetrahedra, adjacency))
}

pub(crate) fn tetrahedron_adjacency(
    tetrahedra: &[[usize; 4]],
) -> Result<Vec<[Option<usize>; 4]>, Error> {
    let mut adjacency = vec![[None; 4]; tetrahedra.len()];
    let mut faces = BTreeMap::<[usize; 3], (usize, usize)>::new();
    for (tetrahedron_index, tetrahedron) in tetrahedra.iter().enumerate() {
        for (face_index, face) in tetrahedron_faces(*tetrahedron).into_iter().enumerate() {
            let face = ordered_face(face);
            if let Some((other_tetrahedron, other_face)) = faces.remove(&face) {
                adjacency[tetrahedron_index][face_index] = Some(other_tetrahedron);
                adjacency[other_tetrahedron][other_face] = Some(tetrahedron_index);
            } else {
                faces.insert(face, (tetrahedron_index, face_index));
            }
        }
    }
    Ok(adjacency)
}

pub(crate) fn tetrahedron_faces(tetrahedron: [usize; 4]) -> [[usize; 3]; 4] {
    [
        [tetrahedron[1], tetrahedron[2], tetrahedron[3]],
        [tetrahedron[0], tetrahedron[3], tetrahedron[2]],
        [tetrahedron[0], tetrahedron[1], tetrahedron[3]],
        [tetrahedron[0], tetrahedron[2], tetrahedron[1]],
    ]
}

fn oriented_tetrahedron(mut tetrahedron: [usize; 4], points: &[[f64; 3]]) -> Option<[usize; 4]> {
    match orient3d(
        points[tetrahedron[0]],
        points[tetrahedron[1]],
        points[tetrahedron[2]],
        points[tetrahedron[3]],
    ) {
        Sign::Positive => Some(tetrahedron),
        Sign::Negative => {
            tetrahedron.swap(0, 1);
            Some(tetrahedron)
        }
        Sign::Zero => None,
    }
}

pub(crate) fn ordered_face(mut face: [usize; 3]) -> [usize; 3] {
    face.sort_unstable();
    face
}

fn bounds(points: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    points.iter().fold(
        ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]),
        |(mut min, mut max), point| {
            for axis in 0..3 {
                min[axis] = min[axis].min(point[axis]);
                max[axis] = max[axis].max(point[axis]);
            }
            (min, max)
        },
    )
}
