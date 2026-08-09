use std::collections::BTreeMap;

use caso_delaunay::predicates::{orient3d, Sign};
use caso_kernel::meshing::MeshableDomain;
use caso_kernel::vec3::Vec3;

use super::distmesh_3d::VolumeMesh;
use crate::algorithm::MeshingContext;
use crate::controls::BoundaryLayerControl;
use crate::error::{MeshError, MeshResult};
use crate::quality::{quality_score, QualityMetric};

struct SelectedFace<'a> {
    vertices: [usize; 3],
    control: &'a BoundaryLayerControl,
    control_index: usize,
    inward: [f64; 3],
}

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

    let mut selected = Vec::new();
    for &face in &mesh.boundary_faces {
        context.check()?;
        let samples = projected_face_samples(domain, mesh, face)?;
        for (control_index, control) in controls.iter().copied().enumerate() {
            let region = domain
                .region_by_name(&control.boundary_region)
                .map_err(|error| MeshError::InvalidInput(error.to_string()))?;
            let matches = region
                .contains(
                    &samples
                        .iter()
                        .copied()
                        .map(Vec3::from_array)
                        .collect::<Vec<_>>(),
                )
                .map_err(|error| MeshError::InvalidInput(error.to_string()))?;
            if matches.into_iter().all(|matches| matches) {
                let longest = face_edges(face)
                    .into_iter()
                    .map(|(a, b)| distance(mesh.points[a], mesh.points[b]))
                    .fold(0.0, f64::max);
                if longest > control.hwall_t * 1.5 {
                    return Err(MeshError::InvalidInput(format!(
                        "domain {:?} needs a finer 3D surface background: controlled facet length {longest:.6e} exceeds soft tangential size {:.6e}",
                        domain.name, control.hwall_t
                    )));
                }
                let center = samples[0];
                let inward = normalized(negate(
                    domain.normals(&[Vec3::from_array(center)])[0].to_array(),
                ))
                .ok_or_else(|| layer_error(domain, "has no exact inward normal"))?;
                selected.push(SelectedFace {
                    vertices: face,
                    control,
                    control_index,
                    inward,
                });
                break;
            }
        }
    }
    if selected.is_empty() {
        return Err(layer_error(
            domain,
            "boundary-layer controls match no exactly classified surface facets",
        ));
    }

    let mut vertex_controls = BTreeMap::<usize, usize>::new();
    let mut incident_directions = BTreeMap::<usize, Vec<[f64; 3]>>::new();
    let mut edge_incidence = BTreeMap::<(usize, usize), usize>::new();
    for face in &selected {
        for vertex in face.vertices {
            if vertex_controls
                .insert(vertex, face.control_index)
                .is_some_and(|other| other != face.control_index)
            {
                return Err(layer_error(
                    domain,
                    "controlled regions request incompatible columns at a shared seam",
                ));
            }
            incident_directions
                .entry(vertex)
                .or_default()
                .push(face.inward);
        }
        for edge in face_edges(face.vertices) {
            *edge_incidence.entry(edge).or_default() += 1;
        }
    }

    let mut directions = BTreeMap::new();
    for (&vertex, incident) in &incident_directions {
        let exact = negate(domain.normals(&[Vec3::from_array(mesh.points[vertex])])[0].to_array());
        let direction = normalized(incident.iter().copied().chain([exact]).fold([0.0; 3], add))
            .ok_or_else(|| layer_error(domain, "cannot solve a common inward seam direction"))?;
        directions.insert(vertex, direction);
    }

    let tolerance = domain.bounds.diagonal() * 1.0e-10 + f64::EPSILON;
    let mut columns = BTreeMap::<(usize, usize), usize>::new();
    for (&vertex, &control_index) in &vertex_controls {
        let control = controls[control_index];
        let mut height = control.hwall_n;
        let mut cumulative = 0.0;
        for level in 0..control.layers {
            cumulative += height;
            let point = translate(mesh.points[vertex], directions[&vertex], cumulative);
            if domain.domain_sdf(&[Vec3::from_array(point)])[0] >= -tolerance {
                return Err(MeshError::InvalidInput(format!(
                    "domain {:?} rejected boundary layer thickness {:.6e}: the requested physical offset does not fit the exact geometry near {:?}",
                    domain.name,
                    control.total_height(),
                    mesh.points[vertex]
                )));
            }
            let index = mesh.points.len();
            mesh.points.push(point);
            columns.insert((vertex, level), index);
            height *= control.ratio;
        }
    }

    let mut replacements = BTreeMap::<usize, Vec<[usize; 4]>>::new();
    let mut prisms = Vec::new();
    let mut pyramids = Vec::new();
    for face in &selected {
        context.check()?;
        let Some((cell_index, cell)) = mesh
            .cells
            .iter()
            .copied()
            .enumerate()
            .find(|(_, cell)| face.vertices.iter().all(|vertex| cell.contains(vertex)))
        else {
            return Err(layer_error(
                domain,
                "cannot attach the shared surface front to its core",
            ));
        };
        let apex = *cell
            .iter()
            .find(|vertex| !face.vertices.contains(vertex))
            .expect("boundary tetrahedron apex");
        let mut previous = face.vertices;
        for level in 0..face.control.layers {
            let next = face.vertices.map(|vertex| columns[&(vertex, level)]);
            prisms.push(oriented_prism(previous, next, &mesh.points).ok_or_else(|| {
                layer_error(
                    domain,
                    "produced an inverted shared column; a finer surface is required",
                )
            })?);
            previous = next;
        }
        replacements
            .entry(cell_index)
            .or_default()
            .push(positive_tet(
                [apex, previous[0], previous[1], previous[2]],
                &mesh.points,
            )?);

        for (a, b) in face_edges(face.vertices) {
            if edge_incidence.get(&ordered_pair(a, b)) != Some(&1) {
                continue;
            }
            let inner_a = columns[&(a, face.control.layers - 1)];
            let inner_b = columns[&(b, face.control.layers - 1)];
            pyramids.push(
                oriented_pyramid([a, b, inner_b, inner_a], apex, &mesh.points).ok_or_else(
                    || layer_error(domain, "produced an inverted open-rim transition"),
                )?,
            );
        }
    }

    let old = std::mem::take(&mut mesh.cells);
    for (index, cell) in old.into_iter().enumerate() {
        if let Some(tetrahedra) = replacements.remove(&index) {
            mesh.cells.extend(tetrahedra);
        } else {
            mesh.cells.push(cell);
        }
    }
    mesh.prisms.extend(prisms);
    mesh.pyramids.extend(pyramids);
    let cells = mesh.cells.len() + mesh.prisms.len() + mesh.pyramids.len();
    if cells > usize::try_from(context.limits.max_cells).unwrap_or(usize::MAX) {
        return Err(MeshError::LimitExceeded(format!(
            "3D boundary layers exceed the configured {} cell limit",
            context.limits.max_cells
        )));
    }
    Ok(())
}

fn projected_face_samples(
    domain: &MeshableDomain,
    mesh: &VolumeMesh,
    face: [usize; 3],
) -> MeshResult<[[f64; 3]; 4]> {
    let points = face.map(|vertex| mesh.points[vertex]);
    let seeds = [
        centroid(points),
        midpoint(points[0], points[1]),
        midpoint(points[1], points[2]),
        midpoint(points[2], points[0]),
    ];
    let projected = domain
        .project_to_boundary(&seeds.map(Vec3::from_array))
        .map_err(|error| MeshError::InvalidInput(error.to_string()))?;
    if projected.iter().any(|projection| !projection.converged) {
        return Err(layer_error(
            domain,
            "needs a finer surface background for exact region classification",
        ));
    }
    Ok(std::array::from_fn(|index| {
        projected[index].point.to_array()
    }))
}

fn oriented_prism(outer: [usize; 3], inner: [usize; 3], points: &[[f64; 3]]) -> Option<[usize; 6]> {
    [
        [outer[0], outer[1], outer[2], inner[0], inner[1], inner[2]],
        [outer[0], outer[2], outer[1], inner[0], inner[2], inner[1]],
    ]
    .into_iter()
    .find(|cell| {
        quality_score(
            "prism6",
            &cell.map(|vertex| points[vertex]),
            QualityMetric::ScaledJacobian,
        )
        .is_some_and(|quality| quality > 0.0)
    })
}

fn oriented_pyramid(base: [usize; 4], apex: usize, points: &[[f64; 3]]) -> Option<[usize; 5]> {
    [
        [base[0], base[1], base[2], base[3], apex],
        [base[3], base[2], base[1], base[0], apex],
    ]
    .into_iter()
    .find(|cell| {
        quality_score(
            "pyramid5",
            &cell.map(|vertex| points[vertex]),
            QualityMetric::ScaledJacobian,
        )
        .is_some_and(|quality| quality > 0.0)
    })
}

fn positive_tet(mut cell: [usize; 4], points: &[[f64; 3]]) -> MeshResult<[usize; 4]> {
    match orient3d(
        points[cell[0]],
        points[cell[1]],
        points[cell[2]],
        points[cell[3]],
    ) {
        Sign::Positive => Ok(cell),
        Sign::Negative => {
            cell.swap(0, 1);
            Ok(cell)
        }
        Sign::Zero => Err(MeshError::InvalidInput(
            "3D boundary layer produced a degenerate core transition; a finer surface is required"
                .into(),
        )),
    }
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

fn layer_error(domain: &MeshableDomain, reason: &str) -> MeshError {
    MeshError::InvalidInput(format!("domain {:?} boundary layer {reason}", domain.name))
}

fn centroid(points: [[f64; 3]; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| (points[0][axis] + points[1][axis] + points[2][axis]) / 3.0)
}

fn midpoint(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| (a[axis] + b[axis]) * 0.5)
}

fn add(mut a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    for axis in 0..3 {
        a[axis] += b[axis];
    }
    a
}

fn negate(point: [f64; 3]) -> [f64; 3] {
    point.map(|value| -value)
}

fn normalized(point: [f64; 3]) -> Option<[f64; 3]> {
    let length = point.iter().map(|value| value * value).sum::<f64>().sqrt();
    (length.is_finite() && length > 1.0e-12).then(|| point.map(|value| value / length))
}

fn translate(point: [f64; 3], direction: [f64; 3], distance: f64) -> [f64; 3] {
    std::array::from_fn(|axis| point[axis] + direction[axis] * distance)
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter()
        .zip(b)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}
