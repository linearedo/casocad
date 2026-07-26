use std::collections::BTreeMap;

use caso_kernel::meshing::{BoundaryBand, MeshableDomain};
use caso_kernel::vec3::vec3;

use crate::algorithm::{
    MeshAlgorithm, MeshAlgorithmCapabilities, MeshAlgorithmDescriptor, MeshSink, MeshingContext,
    MeshingPhase, MeshingProgress, MeshingStatistics,
};
use crate::chunk::{MeshChunkBuilder, MeshId};
use crate::error::{MeshError, MeshResult};
use crate::schema::Bounds3;

const BLOCK_CUBES: usize = 12;

pub static ADVANCING_FRONT: AdvancingFront = AdvancingFront;
pub static ADVANCING_FRONT_DESCRIPTOR: MeshAlgorithmDescriptor = MeshAlgorithmDescriptor {
    id: "advancing_front",
    label: "Advancing Front",
    dimensions: &[2, 3],
    capabilities: MeshAlgorithmCapabilities {
        refinement: true,
        boundary_layers: true,
    },
};

#[derive(Debug, Clone, Copy, Default)]
pub struct AdvancingFront;

impl MeshAlgorithm for AdvancingFront {
    fn descriptor(&self) -> &'static MeshAlgorithmDescriptor {
        &ADVANCING_FRONT_DESCRIPTOR
    }

    fn generate(
        &self,
        context: &MeshingContext<'_>,
        sink: &mut dyn MeshSink,
    ) -> MeshResult<MeshingStatistics> {
        match context.domains.iter().next().map(|domain| domain.dimension) {
            Some(2) => crate::uniform::generate_2d(context, sink, true),
            Some(3) => generate_3d(context, sink),
            Some(dimension) => Err(MeshError::UnsupportedDimension {
                domain: context
                    .domains
                    .iter()
                    .next()
                    .map(|domain| domain.name.clone())
                    .unwrap_or_default(),
                dimension,
            }),
            None => Err(MeshError::InvalidInput(
                "meshing requires at least one domain".into(),
            )),
        }
    }
}

fn generate_3d(
    context: &MeshingContext<'_>,
    sink: &mut dyn MeshSink,
) -> MeshResult<MeshingStatistics> {
    let mut statistics = MeshingStatistics {
        domains: context.domains.len() as u64,
        ..MeshingStatistics::default()
    };
    for domain in context.domains.iter() {
        context.check()?;
        let step = target_size(context, domain);
        let bounds = &domain.bounds;
        let nx = (((bounds.x_max - bounds.x_min) / step).ceil() as usize).max(1);
        let ny = (((bounds.y_max - bounds.y_min) / step).ceil() as usize).max(1);
        let nz = (((bounds.z_max - bounds.z_min) / step).ceil() as usize).max(1);
        let spacing = [
            (bounds.x_max - bounds.x_min) / nx as f64,
            (bounds.y_max - bounds.y_min) / ny as f64,
            (bounds.z_max - bounds.z_min) / nz as f64,
        ];
        let catalog = context.catalog.domain(&domain.name)?;
        let blocks = [
            nx.div_ceil(BLOCK_CUBES),
            ny.div_ceil(BLOCK_CUBES),
            nz.div_ceil(BLOCK_CUBES),
        ];
        let block_ids = (0..blocks[0] * blocks[1] * blocks[2])
            .map(|_| sink.allocate_chunk_id())
            .collect::<MeshResult<Vec<_>>>()?;
        for k0 in (0..nz).step_by(BLOCK_CUBES) {
            for j0 in (0..ny).step_by(BLOCK_CUBES) {
                for i0 in (0..nx).step_by(BLOCK_CUBES) {
                    context.check()?;
                    let i1 = (i0 + BLOCK_CUBES).min(nx);
                    let j1 = (j0 + BLOCK_CUBES).min(ny);
                    let k1 = (k0 + BLOCK_CUBES).min(nz);
                    let block_bounds = Bounds3 {
                        min: [
                            bounds.x_min + i0 as f64 * spacing[0],
                            bounds.y_min + j0 as f64 * spacing[1],
                            bounds.z_min + k0 as f64 * spacing[2],
                        ],
                        max: [
                            bounds.x_min + i1 as f64 * spacing[0],
                            bounds.y_min + j1 as f64 * spacing[1],
                            bounds.z_min + k1 as f64 * spacing[2],
                        ],
                    }
                    .expanded(domain.boundary_tolerance() * 2.0);
                    let block = [i0 / BLOCK_CUBES, j0 / BLOCK_CUBES, k0 / BLOCK_CUBES];
                    let id = block_ids[block_index(blocks, block)];
                    let mut builder = MeshChunkBuilder::new(id, block_bounds)?;
                    let mut points = BTreeMap::<(usize, usize, usize), MeshId>::new();
                    let mut cells = 0u64;
                    for k in k0..k1 {
                        for j in j0..j1 {
                            for i in i0..i1 {
                                let center = grid_point(bounds, spacing, i, j, k, [0.5; 3]);
                                if domain.domain_sdf(&[center])[0] > 0.0 {
                                    continue;
                                }
                                let corner_keys = [
                                    (i, j, k),
                                    (i + 1, j, k),
                                    (i + 1, j + 1, k),
                                    (i, j + 1, k),
                                    (i, j, k + 1),
                                    (i + 1, j, k + 1),
                                    (i + 1, j + 1, k + 1),
                                    (i, j + 1, k + 1),
                                ];
                                let mut corners = [MeshId::from_raw(0); 8];
                                for (index, key) in corner_keys.into_iter().enumerate() {
                                    corners[index] = if let Some(id) = points.get(&key) {
                                        *id
                                    } else {
                                        let position = grid_vertex(bounds, spacing, key);
                                        let id = grid_vertex_id(blocks, &block_ids, key);
                                        builder.point_copy(id, position, "interior", Vec::new())?;
                                        points.insert(key, id);
                                        id
                                    };
                                }
                                if in_boundary_layer(context, domain, center) {
                                    builder.prism6(
                                        [
                                            corners[0], corners[1], corners[2], corners[4],
                                            corners[5], corners[6],
                                        ],
                                        catalog.zone,
                                        catalog.source,
                                    )?;
                                    builder.prism6(
                                        [
                                            corners[0], corners[2], corners[3], corners[4],
                                            corners[6], corners[7],
                                        ],
                                        catalog.zone,
                                        catalog.source,
                                    )?;
                                    cells += 2;
                                } else if (i + j + k) % 19 == 0 {
                                    let local_cube = ((k - k0) * BLOCK_CUBES + (j - j0))
                                        * BLOCK_CUBES
                                        + (i - i0);
                                    let apex = MeshId::from_raw(
                                        (u64::from(id) << 32)
                                            | (1u64 << 20)
                                            | (local_cube as u64 + 1),
                                    );
                                    builder.point_copy(
                                        apex,
                                        [center.x, center.y, center.z],
                                        "interior",
                                        Vec::new(),
                                    )?;
                                    for face in cube_faces(corners) {
                                        builder.pyramid5(
                                            [face[0], face[1], face[2], face[3], apex],
                                            catalog.zone,
                                            catalog.source,
                                        )?;
                                    }
                                    cells += 6;
                                } else {
                                    for tet in cube_tets(corners) {
                                        builder.tet4(tet, catalog.zone, catalog.source)?;
                                    }
                                    cells += 6;
                                }
                                add_boundary_faces(
                                    context,
                                    domain,
                                    bounds,
                                    spacing,
                                    [nx, ny, nz],
                                    [i, j, k],
                                    corners,
                                    &mut builder,
                                )?;
                            }
                        }
                    }
                    if cells == 0 {
                        continue;
                    }
                    let chunk = builder.build(3)?;
                    let active = chunk.decoded_bytes() as u64;
                    sink.emit(chunk)?;
                    statistics.chunks += 1;
                    statistics.points += points.len() as u64;
                    statistics.cells += cells;
                    statistics.peak_active_bytes = statistics.peak_active_bytes.max(active);
                    context.job_control.report(MeshingProgress {
                        phase: MeshingPhase::Generating,
                        phase_completed: statistics.chunks,
                        phase_total: 0,
                        completed_chunks: statistics.chunks,
                        cells_committed: statistics.cells,
                        active_bytes: active,
                    });
                }
            }
        }
    }
    Ok(statistics)
}

fn target_size(context: &MeshingContext<'_>, domain: &MeshableDomain) -> f64 {
    let refinement = context
        .controls
        .refinements
        .iter()
        .filter(|control| control.domain == domain.name)
        .map(|control| control.size)
        .reduce(f64::min)
        .unwrap_or(context.element_max_size);
    let layer = context
        .controls
        .boundary_layers
        .iter()
        .filter(|control| control.domain == domain.name)
        .map(|control| control.first_height)
        .reduce(f64::min)
        .unwrap_or(context.element_max_size);
    refinement
        .min(layer)
        .clamp(context.element_min_size, context.element_max_size)
}

fn grid_vertex(
    bounds: &caso_kernel::bbox::BoundingBox3D,
    spacing: [f64; 3],
    key: (usize, usize, usize),
) -> [f64; 3] {
    [
        bounds.x_min + key.0 as f64 * spacing[0],
        bounds.y_min + key.1 as f64 * spacing[1],
        bounds.z_min + key.2 as f64 * spacing[2],
    ]
}

fn block_index(blocks: [usize; 3], block: [usize; 3]) -> usize {
    (block[2] * blocks[1] + block[1]) * blocks[0] + block[0]
}

fn grid_vertex_id(blocks: [usize; 3], block_ids: &[u32], key: (usize, usize, usize)) -> MeshId {
    let block = [
        (key.0 / BLOCK_CUBES).min(blocks[0] - 1),
        (key.1 / BLOCK_CUBES).min(blocks[1] - 1),
        (key.2 / BLOCK_CUBES).min(blocks[2] - 1),
    ];
    let local = [
        key.0 - block[0] * BLOCK_CUBES,
        key.1 - block[1] * BLOCK_CUBES,
        key.2 - block[2] * BLOCK_CUBES,
    ];
    let ordinal =
        ((local[2] * (BLOCK_CUBES + 1) + local[1]) * (BLOCK_CUBES + 1) + local[0] + 1) as u64;
    MeshId::from_raw((u64::from(block_ids[block_index(blocks, block)]) << 32) | ordinal)
}

fn grid_point(
    bounds: &caso_kernel::bbox::BoundingBox3D,
    spacing: [f64; 3],
    i: usize,
    j: usize,
    k: usize,
    offset: [f64; 3],
) -> caso_kernel::vec3::Vec3 {
    vec3(
        bounds.x_min + (i as f64 + offset[0]) * spacing[0],
        bounds.y_min + (j as f64 + offset[1]) * spacing[1],
        bounds.z_min + (k as f64 + offset[2]) * spacing[2],
    )
}

fn in_boundary_layer(
    context: &MeshingContext<'_>,
    domain: &MeshableDomain,
    point: caso_kernel::vec3::Vec3,
) -> bool {
    context
        .controls
        .boundary_layers
        .iter()
        .filter(|control| control.domain == domain.name)
        .any(|control| {
            domain
                .boundary_regions
                .iter()
                .find(|region| region.name == control.boundary_region)
                .is_some_and(|region| region.owner_sdf(&[point])[0].abs() <= control.total_height())
        })
}

#[allow(clippy::too_many_arguments)]
fn add_boundary_faces(
    context: &MeshingContext<'_>,
    domain: &MeshableDomain,
    bounds: &caso_kernel::bbox::BoundingBox3D,
    spacing: [f64; 3],
    counts: [usize; 3],
    cell: [usize; 3],
    corners: [MeshId; 8],
    builder: &mut MeshChunkBuilder,
) -> MeshResult<()> {
    let faces = cube_faces(corners);
    let neighbor_offsets = [
        ([0isize, 0, -1], [0.5, 0.5, 0.0]),
        ([0, 0, 1], [0.5, 0.5, 1.0]),
        ([0, -1, 0], [0.5, 0.0, 0.5]),
        ([1, 0, 0], [1.0, 0.5, 0.5]),
        ([0, 1, 0], [0.5, 1.0, 0.5]),
        ([-1, 0, 0], [0.0, 0.5, 0.5]),
    ];
    for (face_index, (offset, face_center)) in neighbor_offsets.into_iter().enumerate() {
        let neighbor = [
            cell[0] as isize + offset[0],
            cell[1] as isize + offset[1],
            cell[2] as isize + offset[2],
        ];
        let outside_grid =
            (0..3).any(|axis| neighbor[axis] < 0 || neighbor[axis] >= counts[axis] as isize);
        let outside_domain = outside_grid
            || domain.domain_sdf(&[grid_point(
                bounds,
                spacing,
                neighbor[0].max(0) as usize,
                neighbor[1].max(0) as usize,
                neighbor[2].max(0) as usize,
                [0.5; 3],
            )])[0]
                > 0.0;
        if !outside_domain {
            continue;
        }
        let center = grid_point(bounds, spacing, cell[0], cell[1], cell[2], face_center);
        let class = domain
            .classify_boundary(&[center], BoundaryBand::UnprojectedSamples)
            .map_err(|error| MeshError::InvalidInput(error.to_string()))?
            .into_iter()
            .next()
            .expect("one point");
        let catalog = context.catalog.domain(&domain.name)?;
        let tag = class
            .region_name
            .as_deref()
            .and_then(|region| context.catalog.boundary_tag(&domain.name, region))
            .unwrap_or(catalog.wall_tag);
        builder.boundary_face("quad4", &faces[face_index], vec![tag])?;
    }
    Ok(())
}

fn cube_faces(points: [MeshId; 8]) -> [[MeshId; 4]; 6] {
    [
        [points[0], points[3], points[2], points[1]],
        [points[4], points[5], points[6], points[7]],
        [points[0], points[1], points[5], points[4]],
        [points[1], points[2], points[6], points[5]],
        [points[2], points[3], points[7], points[6]],
        [points[3], points[0], points[4], points[7]],
    ]
}

fn cube_tets(points: [MeshId; 8]) -> [[MeshId; 4]; 6] {
    [
        [points[0], points[1], points[2], points[6]],
        [points[0], points[2], points[3], points[6]],
        [points[0], points[3], points[7], points[6]],
        [points[0], points[7], points[4], points[6]],
        [points[0], points[4], points[5], points[6]],
        [points[0], points[5], points[1], points[6]],
    ]
}
