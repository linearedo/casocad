use std::collections::{BTreeMap, BTreeSet};

use caso_delaunay::predicates::{orient3d, Sign};
use caso_kernel::bbox::BoundingBox3D;
use caso_kernel::meshing::{BoundaryBand, MeshableDomain};
use caso_kernel::vec3::Vec3;

use crate::algorithm::{
    MeshSink, MeshingContext, MeshingPhase, MeshingProgress, MeshingStatistics,
};
use crate::chunk::{MeshChunkBuilder, MeshId};
use crate::error::{MeshError, MeshResult};
use crate::schema::Bounds3;

const ESTIMATED_BYTES_PER_TET: usize = 256;
const ROOT_STEPS: usize = 64;

#[derive(Clone, Copy)]
struct Sample {
    position: [f64; 3],
    sdf: f64,
}

#[derive(Default)]
pub(super) struct VolumeMesh {
    pub(super) points: Vec<[f64; 3]>,
    pub(super) boundary_points: BTreeSet<usize>,
    pub(super) cells: Vec<[usize; 4]>,
    pub(super) prisms: Vec<[usize; 6]>,
    pub(super) pyramids: Vec<[usize; 5]>,
    pub(super) boundary_faces: Vec<[usize; 3]>,
    lattice_vertices: BTreeMap<usize, usize>,
    crossings: BTreeMap<(usize, usize), usize>,
}

pub(super) fn generate(
    context: &MeshingContext<'_>,
    sink: &mut dyn MeshSink,
) -> MeshResult<MeshingStatistics> {
    let mut statistics = MeshingStatistics {
        domains: context.domains.len() as u64,
        ..MeshingStatistics::default()
    };
    let domain_bounds = context
        .domains
        .iter()
        .map(|domain| domain.bounds)
        .reduce(|bounds, other| bounds.union(&other))
        .expect("meshing context has domains");
    let padding = context.target_size;
    let sampling_bounds = BoundingBox3D::new(
        domain_bounds.x_min - padding,
        domain_bounds.x_max + padding,
        domain_bounds.y_min - padding,
        domain_bounds.y_max + padding,
        domain_bounds.z_min - padding,
        domain_bounds.z_max + padding,
    )
    .map_err(|error| MeshError::InvalidInput(error.to_string()))?;
    let mut shared_boundary_points = Vec::<([f64; 3], MeshId)>::new();
    for domain in context.domains.iter() {
        context.check()?;
        let mesh = build_domain(domain, sampling_bounds, context)?;
        emit_domain(
            domain,
            context,
            sink,
            &mesh,
            &mut statistics,
            &mut shared_boundary_points,
        )?;
    }
    Ok(statistics)
}

fn build_domain(
    domain: &MeshableDomain,
    bounds: BoundingBox3D,
    context: &MeshingContext<'_>,
) -> MeshResult<VolumeMesh> {
    let lengths = [
        bounds.x_max - bounds.x_min,
        bounds.y_max - bounds.y_min,
        bounds.z_max - bounds.z_min,
    ];
    if lengths
        .iter()
        .any(|length| !length.is_finite() || *length <= 0.0)
    {
        return Err(MeshError::InvalidInput(format!(
            "domain {:?} has invalid 3D bounds",
            domain.name
        )));
    }
    let counts =
        lengths.map(|length| ((length / context.target_size).ceil() as usize).clamp(1, 1_024));
    let background_tets = counts
        .into_iter()
        .try_fold(6usize, usize::checked_mul)
        .ok_or_else(|| MeshError::LimitExceeded("3D background grid size overflowed".into()))?;
    if background_tets > usize::try_from(context.limits.max_cells).unwrap_or(usize::MAX) * 4 {
        return Err(MeshError::LimitExceeded(format!(
            "adaptive 3D background requires about {background_tets} tetrahedra, exceeding the configured {} cell limit",
            context.limits.max_cells
        )));
    }
    let dimensions = [counts[0] + 1, counts[1] + 1, counts[2] + 1];
    let mut samples = Vec::with_capacity(dimensions.into_iter().product());
    for z in 0..dimensions[2] {
        for y in 0..dimensions[1] {
            for x in 0..dimensions[0] {
                let position = [
                    bounds.x_min + lengths[0] * x as f64 / counts[0] as f64,
                    bounds.y_min + lengths[1] * y as f64 / counts[1] as f64,
                    bounds.z_min + lengths[2] * z as f64 / counts[2] as f64,
                ];
                let sdf = domain.domain_sdf(&[Vec3::from_array(position)])[0];
                if !sdf.is_finite() {
                    return Err(MeshError::InvalidInput(format!(
                        "domain {:?} returned a non-finite SDF value near {position:?}",
                        domain.name
                    )));
                }
                samples.push(Sample { position, sdf });
            }
        }
    }

    let mut mesh = VolumeMesh::default();
    let mut visited = 0usize;
    for z in 0..counts[2] {
        for y in 0..counts[1] {
            for x in 0..counts[0] {
                if visited.is_multiple_of(128) {
                    context.check()?;
                }
                visited += 1;
                let cube = cube_vertices(x, y, z, dimensions);
                for permutation in AXIS_PERMUTATIONS {
                    let tetrahedron = freudenthal_tetrahedron(cube, permutation);
                    clip_tetrahedron(domain, context, &samples, tetrahedron, &mut mesh)?;
                    if mesh.cells.len()
                        > usize::try_from(context.limits.max_cells).unwrap_or(usize::MAX)
                    {
                        return Err(MeshError::LimitExceeded(format!(
                            "3D mesh exceeds the configured {} cell limit",
                            context.limits.max_cells
                        )));
                    }
                }
            }
        }
    }
    if mesh.cells.is_empty() {
        return Err(MeshError::InvalidInput(format!(
            "domain {:?} produced no valid 3D elements at target size {:.6e}",
            domain.name, context.target_size
        )));
    }
    audit_mesh(domain, &mut mesh)?;
    super::layers_3d::apply_boundary_layers(domain, context, &mut mesh)?;
    Ok(mesh)
}

const AXIS_PERMUTATIONS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

fn cube_vertices(x: usize, y: usize, z: usize, dimensions: [usize; 3]) -> [usize; 8] {
    let index =
        |dx, dy, dz| (z + dz) * dimensions[0] * dimensions[1] + (y + dy) * dimensions[0] + x + dx;
    [
        index(0, 0, 0),
        index(1, 0, 0),
        index(0, 1, 0),
        index(1, 1, 0),
        index(0, 0, 1),
        index(1, 0, 1),
        index(0, 1, 1),
        index(1, 1, 1),
    ]
}

fn freudenthal_tetrahedron(cube: [usize; 8], permutation: [usize; 3]) -> [usize; 4] {
    let mut bits = 0usize;
    let mut result = [cube[0]; 4];
    for (step, axis) in permutation.into_iter().enumerate() {
        bits |= 1 << axis;
        result[step + 1] = cube[bits];
    }
    result
}

fn clip_tetrahedron(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    samples: &[Sample],
    tetrahedron: [usize; 4],
    mesh: &mut VolumeMesh,
) -> MeshResult<()> {
    let inside = tetrahedron.map(|vertex| samples[vertex].sdf <= 0.0);
    let inside_count = inside.into_iter().filter(|inside| *inside).count();
    if inside_count == 0 {
        return Ok(());
    }
    if inside_count == 4 {
        let cell = tetrahedron.map(|vertex| lattice_vertex(mesh, samples, vertex));
        push_oriented_cell(mesh, cell)?;
        return Ok(());
    }

    let faces = tetrahedron_faces(tetrahedron);
    let mut polygons = Vec::new();
    for face in faces {
        let polygon = clipped_face(domain, context, samples, mesh, face)?;
        if polygon.len() >= 3 {
            polygons.push(polygon);
        }
    }
    let mut cut = tetrahedron_edges(tetrahedron)
        .into_iter()
        .filter(|[a, b]| (samples[*a].sdf <= 0.0) != (samples[*b].sdf <= 0.0))
        .map(|edge| crossing_vertex(domain, context, samples, mesh, edge))
        .collect::<MeshResult<Vec<_>>>()?;
    cut.sort_unstable();
    cut.dedup();
    if cut.len() >= 3 {
        sort_polygon(&mut cut, &mesh.points);
        polygons.push(cut.clone());
    }
    let mut unique = BTreeMap::<Vec<usize>, Vec<usize>>::new();
    for polygon in polygons {
        let mut key = polygon.clone();
        key.sort_unstable();
        unique.entry(key).or_insert(polygon);
    }
    let polygons = unique.into_values().collect::<Vec<_>>();
    let mut polyhedron_vertices = polygons.iter().flatten().copied().collect::<Vec<_>>();
    polyhedron_vertices.sort_unstable();
    polyhedron_vertices.dedup();
    if polyhedron_vertices.len() < 4 {
        return Ok(());
    }
    let center = centroid(
        polyhedron_vertices
            .iter()
            .map(|vertex| mesh.points[*vertex]),
    );
    let center_vertex = mesh.points.len();
    mesh.points.push(center);
    for polygon in polygons {
        for triangle in triangulate_polygon(polygon) {
            push_oriented_cell(mesh, [center_vertex, triangle[0], triangle[1], triangle[2]])?;
            if triangle
                .iter()
                .all(|vertex| mesh.boundary_points.contains(vertex))
            {
                mesh.boundary_faces.push(triangle);
            }
        }
    }
    Ok(())
}

fn clipped_face(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    samples: &[Sample],
    mesh: &mut VolumeMesh,
    face: [usize; 3],
) -> MeshResult<Vec<usize>> {
    let mut polygon = Vec::new();
    for edge in 0..3 {
        let a = face[edge];
        let b = face[(edge + 1) % 3];
        if samples[a].sdf <= 0.0 {
            polygon.push(lattice_vertex(mesh, samples, a));
        }
        if (samples[a].sdf <= 0.0) != (samples[b].sdf <= 0.0) {
            polygon.push(crossing_vertex(domain, context, samples, mesh, [a, b])?);
        }
    }
    polygon.dedup();
    Ok(polygon)
}

fn crossing_vertex(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    samples: &[Sample],
    mesh: &mut VolumeMesh,
    edge: [usize; 2],
) -> MeshResult<usize> {
    let key = ordered_pair(edge[0], edge[1]);
    if let Some(vertex) = mesh.crossings.get(&key) {
        return Ok(*vertex);
    }
    let (mut inside, mut outside) = if samples[edge[0]].sdf <= 0.0 {
        (samples[edge[0]].position, samples[edge[1]].position)
    } else {
        (samples[edge[1]].position, samples[edge[0]].position)
    };
    for step in 0..ROOT_STEPS {
        if step.is_multiple_of(16) {
            context.check()?;
        }
        let middle = midpoint(inside, outside);
        if domain.domain_sdf(&[Vec3::from_array(middle)])[0] <= 0.0 {
            inside = middle;
        } else {
            outside = middle;
        }
    }
    let projected = domain
        .project_to_boundary(&[Vec3::from_array(inside)])
        .map_err(|error| MeshError::InvalidInput(error.to_string()))?
        .into_iter()
        .next()
        .expect("one boundary projection");
    let position = if projected.converged {
        projected.point.to_array()
    } else {
        midpoint(inside, outside)
    };
    let vertex = mesh.points.len();
    mesh.points.push(position);
    mesh.boundary_points.insert(vertex);
    mesh.crossings.insert(key, vertex);
    Ok(vertex)
}

fn lattice_vertex(mesh: &mut VolumeMesh, samples: &[Sample], lattice: usize) -> usize {
    if let Some(vertex) = mesh.lattice_vertices.get(&lattice) {
        return *vertex;
    }
    let vertex = mesh.points.len();
    mesh.points.push(samples[lattice].position);
    mesh.lattice_vertices.insert(lattice, vertex);
    vertex
}

fn push_oriented_cell(mesh: &mut VolumeMesh, mut cell: [usize; 4]) -> MeshResult<()> {
    match orient3d(
        mesh.points[cell[0]],
        mesh.points[cell[1]],
        mesh.points[cell[2]],
        mesh.points[cell[3]],
    ) {
        Sign::Positive => {}
        Sign::Negative => cell.swap(0, 1),
        Sign::Zero => return Ok(()),
    }
    mesh.cells.push(cell);
    Ok(())
}

fn audit_mesh(domain: &MeshableDomain, mesh: &mut VolumeMesh) -> MeshResult<()> {
    let mut faces = BTreeMap::<[usize; 3], usize>::new();
    for cell in &mesh.cells {
        for face in tetrahedron_faces(*cell) {
            let mut face = face;
            face.sort_unstable();
            *faces.entry(face).or_default() += 1;
        }
    }
    if let Some((face, incidence)) = faces.iter().find(|(_, incidence)| **incidence > 2) {
        return Err(MeshError::InvalidInput(format!(
            "domain {:?} produced non-manifold face {face:?} with incidence {incidence}",
            domain.name
        )));
    }
    let mut boundary = mesh
        .boundary_faces
        .iter()
        .map(|face| {
            let mut face = *face;
            face.sort_unstable();
            face
        })
        .collect::<BTreeSet<_>>();
    boundary.retain(|face| faces.get(face) == Some(&1));
    mesh.boundary_faces = boundary.into_iter().collect();
    mesh.boundary_points = mesh.boundary_faces.iter().flatten().copied().collect();
    if mesh.boundary_faces.is_empty() {
        return Err(MeshError::InvalidInput(format!(
            "domain {:?} recovered no exterior 3D boundary facets",
            domain.name
        )));
    }
    Ok(())
}

fn emit_domain(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    sink: &mut dyn MeshSink,
    mesh: &VolumeMesh,
    statistics: &mut MeshingStatistics,
    shared_boundary_points: &mut Vec<([f64; 3], MeshId)>,
) -> MeshResult<()> {
    let estimated = (mesh.cells.len() + mesh.prisms.len() + mesh.pyramids.len())
        .saturating_mul(ESTIMATED_BYTES_PER_TET);
    if estimated > context.limits.target_chunk_bytes {
        return Err(MeshError::LimitExceeded(format!(
            "3D Delaunay topology and writer batch require about {estimated} bytes, exceeding the configured {} byte chunk target",
            context.limits.target_chunk_bytes
        )));
    }
    let chunk_id = sink.allocate_chunk_id()?;
    let bounds = Bounds3::from_points(mesh.points.iter().copied())
        .expanded(domain.bounds.diagonal() * 1.0e-12 + f64::EPSILON);
    let mut builder = MeshChunkBuilder::new(chunk_id, bounds)?;
    let shared_tolerance = domain.bounds.diagonal() * 1.0e-10 + f64::EPSILON;
    let previous_shared = shared_boundary_points.len();
    let mut ids = BTreeMap::new();
    for (index, point) in mesh.points.iter().enumerate() {
        let boundary = mesh.boundary_points.contains(&index);
        if boundary {
            if let Some((position, id)) = shared_boundary_points[..previous_shared]
                .iter()
                .find(|(position, _)| distance(*position, *point) <= shared_tolerance)
            {
                builder.point_copy(*id, *position, "boundary", Vec::new())?;
                ids.insert(index, *id);
                continue;
            }
        }
        let id = builder.classified_point(
            *point,
            if boundary { "boundary" } else { "interior" },
            Vec::new(),
        )?;
        if boundary {
            shared_boundary_points.push((*point, id));
        }
        ids.insert(index, id);
    }
    let catalog = context.catalog.domain(&domain.name)?;
    for cell in &mesh.cells {
        builder.tet4(
            cell.map(|vertex| ids[&vertex]),
            catalog.zone,
            catalog.source,
        )?;
    }
    for cell in &mesh.prisms {
        builder.prism6(
            cell.map(|vertex| ids[&vertex]),
            catalog.zone,
            catalog.source,
        )?;
    }
    for cell in &mesh.pyramids {
        builder.pyramid5(
            cell.map(|vertex| ids[&vertex]),
            catalog.zone,
            catalog.source,
        )?;
    }
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
        let tag = class
            .region_name
            .as_deref()
            .and_then(|region| context.catalog.boundary_tag(&domain.name, region))
            .unwrap_or(catalog.wall_tag);
        builder.boundary_face("tri3", &face.map(|vertex| ids[&vertex]), vec![tag])?;
    }
    let chunk = builder.build(3)?;
    let active = chunk.decoded_bytes() as u64;
    if active > context.limits.target_chunk_bytes as u64 {
        return Err(MeshError::LimitExceeded(format!(
            "3D chunk {chunk_id} requires {active} bytes, exceeding the configured {} byte chunk target",
            context.limits.target_chunk_bytes
        )));
    }
    let points = chunk.points.len() as u64;
    let cells = chunk.cells.len() as u64;
    sink.emit(chunk)?;
    statistics.chunks += 1;
    statistics.points += points;
    statistics.cells += cells;
    statistics.peak_active_bytes = statistics.peak_active_bytes.max(active);
    context.job_control.report(MeshingProgress {
        phase: MeshingPhase::Generating,
        phase_completed: statistics.chunks,
        phase_total: context.domains.len() as u64,
        completed_chunks: statistics.chunks,
        cells_committed: statistics.cells,
        active_bytes: active,
    });
    Ok(())
}

fn tetrahedron_edges(tetrahedron: [usize; 4]) -> [[usize; 2]; 6] {
    [
        [tetrahedron[0], tetrahedron[1]],
        [tetrahedron[0], tetrahedron[2]],
        [tetrahedron[0], tetrahedron[3]],
        [tetrahedron[1], tetrahedron[2]],
        [tetrahedron[1], tetrahedron[3]],
        [tetrahedron[2], tetrahedron[3]],
    ]
}

fn tetrahedron_faces(tetrahedron: [usize; 4]) -> [[usize; 3]; 4] {
    [
        [tetrahedron[1], tetrahedron[2], tetrahedron[3]],
        [tetrahedron[0], tetrahedron[3], tetrahedron[2]],
        [tetrahedron[0], tetrahedron[1], tetrahedron[3]],
        [tetrahedron[0], tetrahedron[2], tetrahedron[1]],
    ]
}

fn triangulate_polygon(mut polygon: Vec<usize>) -> Vec<[usize; 3]> {
    let first = polygon
        .iter()
        .enumerate()
        .min_by_key(|(_, vertex)| **vertex)
        .map(|(index, _)| index)
        .unwrap_or(0);
    polygon.rotate_left(first);
    (1..polygon.len() - 1)
        .map(|index| [polygon[0], polygon[index], polygon[index + 1]])
        .collect()
}

fn sort_polygon(polygon: &mut [usize], points: &[[f64; 3]]) {
    let center = centroid(polygon.iter().map(|vertex| points[*vertex]));
    let normal = if polygon.len() >= 3 {
        cross(
            subtract(points[polygon[1]], points[polygon[0]]),
            subtract(points[polygon[2]], points[polygon[0]]),
        )
    } else {
        [0.0, 0.0, 1.0]
    };
    let axis = (0..3)
        .max_by(|a, b| normal[*a].abs().total_cmp(&normal[*b].abs()))
        .unwrap_or(2);
    polygon.sort_by(|a, b| {
        let angle = |vertex: usize| {
            let point = subtract(points[vertex], center);
            match axis {
                0 => point[2].atan2(point[1]),
                1 => point[2].atan2(point[0]),
                _ => point[1].atan2(point[0]),
            }
        };
        angle(*a).total_cmp(&angle(*b)).then(a.cmp(b))
    });
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

fn midpoint(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ]
}

fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
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

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    subtract(a, b)
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}
