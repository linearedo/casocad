use std::collections::BTreeMap;

use caso_delaunay::predicates::{orient3d, Sign};
use caso_kernel::meshing::{BoundaryBand, MeshableDomain};
use caso_kernel::vec3::Vec3;

use super::distmesh_3d::VolumeMesh;
use crate::algorithm::MeshingContext;
use crate::controls::BoundaryLayerControl;
use crate::error::{MeshError, MeshResult};
use crate::quality::{quality_score, QualityMetric};

pub(super) fn apply_boundary_layers(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    mesh: &mut VolumeMesh,
) -> MeshResult<()> {
    let controls = context
        .controls
        .boundary_layers
        .iter()
        .filter(|control| control.domain == domain.name)
        .collect::<Vec<_>>();
    if controls.is_empty() {
        return Ok(());
    }
    let mut selected = Vec::<([usize; 3], &BoundaryLayerControl)>::new();
    for face in &mesh.boundary_faces {
        let center = centroid(face.iter().map(|vertex| mesh.points[*vertex]));
        let class = domain
            .classify_boundary(
                &[Vec3::from_array(center)],
                BoundaryBand::UnprojectedSamples,
            )
            .map_err(|error| MeshError::InvalidInput(error.to_string()))?
            .into_iter()
            .next()
            .expect("one boundary classification");
        if let Some(control) = controls
            .iter()
            .find(|control| class.region_name.as_deref() == Some(&control.boundary_region))
        {
            selected.push((*face, *control));
        }
    }
    if selected.is_empty() {
        return Err(MeshError::InvalidInput(format!(
            "domain {:?} boundary-layer controls match no recovered surface facets",
            domain.name
        )));
    }
    let mut selected_edge_incidence = BTreeMap::<(usize, usize), usize>::new();
    for (face, _) in &selected {
        for edge in face_edges(*face) {
            *selected_edge_incidence.entry(edge).or_default() += 1;
        }
    }
    let mut replacements =
        BTreeMap::<usize, (Vec<[usize; 6]>, Vec<[usize; 5]>, Vec<[usize; 4]>)>::new();
    let mut layer_points = BTreeMap::<(usize, usize, usize), usize>::new();
    for (face, control) in selected {
        context.check()?;
        let Some((cell_index, &cell)) = mesh
            .cells
            .iter()
            .enumerate()
            .find(|(_, cell)| face.iter().all(|vertex| cell.contains(vertex)))
        else {
            return Err(MeshError::InvalidInput(format!(
                "domain {:?} cannot attach a layer column to facet {face:?}",
                domain.name
            )));
        };
        if replacements.contains_key(&cell_index) {
            return Err(MeshError::InvalidInput(format!(
                "domain {:?} has overlapping layer columns",
                domain.name
            )));
        }
        let apex = *cell
            .iter()
            .find(|vertex| !face.contains(vertex))
            .expect("boundary tetrahedron apex");
        let altitude = point_plane_distance(
            mesh.points[apex],
            mesh.points[face[0]],
            mesh.points[face[1]],
            mesh.points[face[2]],
        );
        if !altitude.is_finite() || altitude <= 0.0 || control.total_height() >= altitude * 0.9 {
            return Err(MeshError::InvalidInput(format!(
                "domain {:?} rejected boundary layer thickness {:.6e} beyond local reach {:.6e}",
                domain.name,
                control.total_height(),
                altitude
            )));
        }
        let mut levels = vec![face];
        let mut distance = 0.0;
        let mut height = control.hwall_n;
        for level in 0..control.layers {
            distance += height;
            let fraction = distance / altitude;
            let triangle = face.map(|outer| {
                *layer_points.entry((outer, apex, level)).or_insert_with(|| {
                    let vertex = mesh.points.len();
                    mesh.points
                        .push(lerp(mesh.points[outer], mesh.points[apex], fraction));
                    vertex
                })
            });
            levels.push(triangle);
            height *= control.ratio;
        }
        let mut prisms = levels
            .windows(2)
            .map(|levels| {
                [
                    levels[0][0],
                    levels[0][1],
                    levels[0][2],
                    levels[1][0],
                    levels[1][1],
                    levels[1][2],
                ]
            })
            .collect::<Vec<_>>();
        let rim = face_edges(face)
            .into_iter()
            .any(|edge| selected_edge_incidence.get(&edge) == Some(&1));
        let mut pyramids = Vec::new();
        let mut transition_tets = Vec::new();
        if rim {
            let prism = prisms.remove(0);
            let candidates = [
                [prism[0], prism[1], prism[4], prism[3], prism[2]],
                [prism[1], prism[0], prism[3], prism[4], prism[2]],
            ];
            let pyramid = candidates
                .into_iter()
                .find(|candidate| {
                    quality_score(
                        "pyramid5",
                        &candidate.map(|vertex| mesh.points[vertex]),
                        QualityMetric::ScaledJacobian,
                    )
                    .is_some_and(|quality| quality > 0.0)
                })
                .ok_or_else(|| {
                    MeshError::InvalidInput(format!(
                        "domain {:?} produced an inverted pyramid layer transition",
                        domain.name
                    ))
                })?;
            pyramids.push(pyramid);
            transition_tets.push(oriented_tet(
                [prism[2], prism[3], prism[4], prism[5]],
                &mesh.points,
            ));
        }
        let inner = *levels.last().expect("at least one layer");
        transition_tets.push(oriented_tet(
            [apex, inner[0], inner[1], inner[2]],
            &mesh.points,
        ));
        replacements.insert(cell_index, (prisms, pyramids, transition_tets));
    }
    let old = std::mem::take(&mut mesh.cells);
    for (index, cell) in old.into_iter().enumerate() {
        if let Some((prisms, pyramids, tetrahedra)) = replacements.remove(&index) {
            mesh.prisms.extend(prisms);
            mesh.pyramids.extend(pyramids);
            mesh.cells.extend(tetrahedra);
        } else {
            mesh.cells.push(cell);
        }
    }
    if mesh.cells.len() + mesh.prisms.len() + mesh.pyramids.len()
        > usize::try_from(context.limits.max_cells).unwrap_or(usize::MAX)
    {
        return Err(MeshError::LimitExceeded(format!(
            "3D boundary layers exceed the configured {} cell limit",
            context.limits.max_cells
        )));
    }
    Ok(())
}

fn oriented_tet(mut tetrahedron: [usize; 4], points: &[[f64; 3]]) -> [usize; 4] {
    if orient3d(
        points[tetrahedron[0]],
        points[tetrahedron[1]],
        points[tetrahedron[2]],
        points[tetrahedron[3]],
    ) == Sign::Negative
    {
        tetrahedron.swap(0, 1);
    }
    tetrahedron
}

fn face_edges(face: [usize; 3]) -> [(usize, usize); 3] {
    [
        ordered_pair(face[0], face[1]),
        ordered_pair(face[1], face[2]),
        ordered_pair(face[2], face[0]),
    ]
}

fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn centroid(points: impl IntoIterator<Item = [f64; 3]>) -> [f64; 3] {
    let (sum, count) = points
        .into_iter()
        .fold(([0.0; 3], 0usize), |(mut sum, count), point| {
            for axis in 0..3 {
                sum[axis] += point[axis];
            }
            (sum, count + 1)
        });
    sum.map(|value| value / count as f64)
}

fn lerp(a: [f64; 3], b: [f64; 3], amount: f64) -> [f64; 3] {
    [
        a[0] + (b[0] - a[0]) * amount,
        a[1] + (b[1] - a[1]) * amount,
        a[2] + (b[2] - a[2]) * amount,
    ]
}

fn point_plane_distance(point: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let normal = cross(subtract(b, a), subtract(c, a));
    let length = normal
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if length == 0.0 {
        0.0
    } else {
        subtract(point, a)
            .into_iter()
            .zip(normal)
            .map(|(left, right)| left * right)
            .sum::<f64>()
            .abs()
            / length
    }
}

fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
