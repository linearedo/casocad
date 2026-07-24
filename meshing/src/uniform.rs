use std::collections::{BTreeMap, BTreeSet};

use caso_kernel::meshing::{BoundaryBand, MeshableDomain, MeshableDomainSpace};

use crate::algorithm::{
    MeshAlgorithm, MeshAlgorithmCapabilities, MeshAlgorithmDescriptor, MeshSink, MeshingContext,
    MeshingProgress, MeshingStatistics,
};
use crate::chunk::{MeshChunkBuilder, MeshId};
use crate::error::{MeshError, MeshResult};
use crate::schema::Bounds3;

const TILE_SQUARES: u64 = 128;
const BISECTION_LIMIT: usize = 64;

type BoundarySegments = BTreeMap<(MeshId, MeshId), (Vec<u64>, [Vertex; 2])>;

pub static UNIFORM_2D: Uniform2d = Uniform2d;
pub static UNIFORM_2D_DESCRIPTOR: MeshAlgorithmDescriptor = MeshAlgorithmDescriptor {
    id: "uniform_2d",
    label: "Uniform 2D",
    dimensions: &[2],
    capabilities: MeshAlgorithmCapabilities {
        refinement: false,
        boundary_layers: false,
    },
};

#[derive(Debug, Clone, Copy, Default)]
pub struct Uniform2d;

impl MeshAlgorithm for Uniform2d {
    fn descriptor(&self) -> &'static MeshAlgorithmDescriptor {
        &UNIFORM_2D_DESCRIPTOR
    }

    fn generate(
        &self,
        context: &MeshingContext<'_>,
        sink: &mut dyn MeshSink,
    ) -> MeshResult<MeshingStatistics> {
        generate_2d(context, sink, false)
    }
}

pub(crate) fn generate_2d(
    context: &MeshingContext<'_>,
    sink: &mut dyn MeshSink,
    advancing_front_mode: bool,
) -> MeshResult<MeshingStatistics> {
    let mut statistics = MeshingStatistics {
        domains: context.domains.len() as u64,
        ..MeshingStatistics::default()
    };
    for domain in context.domains.iter() {
        context.check()?;
        if domain.dimension != 2 {
            return Err(MeshError::UnsupportedDimension {
                domain: domain.name.clone(),
                dimension: domain.dimension,
            });
        }
        let space = domain
            .mesh_space()
            .map_err(|error| MeshError::InvalidInput(error.to_string()))?;
        let step = if advancing_front_mode {
            adaptive_background_size(context, domain)
        } else {
            context.element_max_size
        };
        let grid = LogicalGrid::new(&space, step, &domain.name)?;
        let ids = (0..grid.tile_count())
            .map(|_| sink.allocate_chunk_id())
            .collect::<MeshResult<Vec<_>>>()?;
        let catalog = context.catalog.domain(&domain.name)?;
        for tile_y in 0..grid.tile_rows {
            for tile_x in 0..grid.tile_columns {
                context.check()?;
                let tile_index = (tile_y * grid.tile_columns + tile_x) as usize;
                let tile = GridTile {
                    id: ids[tile_index],
                    i0: tile_x * TILE_SQUARES,
                    i1: ((tile_x + 1) * TILE_SQUARES).min(grid.nx),
                    j0: tile_y * TILE_SQUARES,
                    j1: ((tile_y + 1) * TILE_SQUARES).min(grid.ny),
                };
                let chunk = mesh_tile(
                    domain,
                    &space,
                    &grid,
                    tile,
                    &ids,
                    context,
                    advancing_front_mode,
                )?;
                let points = chunk.points.len() as u64;
                let cells = chunk.cells.len() as u64;
                let active = chunk.decoded_bytes() as u64;
                if cells > 0 {
                    sink.emit(chunk)?;
                    statistics.chunks += 1;
                    statistics.points += points;
                    statistics.cells += cells;
                    statistics.peak_active_bytes = statistics.peak_active_bytes.max(active);
                    context.job_control.report(MeshingProgress {
                        completed_chunks: statistics.chunks,
                        cells_committed: statistics.cells,
                        active_bytes: active,
                    });
                }
            }
        }
        let _ = catalog;
    }
    Ok(statistics)
}

fn adaptive_background_size(context: &MeshingContext<'_>, domain: &MeshableDomain) -> f64 {
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

#[derive(Debug, Clone, Copy)]
struct LogicalGrid {
    u_min: f64,
    v_min: f64,
    du: f64,
    dv: f64,
    nx: u64,
    ny: u64,
    tile_columns: u64,
    tile_rows: u64,
}

impl LogicalGrid {
    fn new(space: &MeshableDomainSpace, size: f64, name: &str) -> MeshResult<Self> {
        let [u_min, u_max, v_min, v_max] = space.bounds();
        let width = u_max - u_min;
        let height = v_max - v_min;
        if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
            return Err(MeshError::InvalidInput(format!(
                "domain {name:?} has invalid local 2D bounds"
            )));
        }
        let nx = ((width / size).ceil() as u64).max(1);
        let ny = ((height / size).ceil() as u64).max(1);
        Ok(Self {
            u_min,
            v_min,
            du: width / nx as f64,
            dv: height / ny as f64,
            nx,
            ny,
            tile_columns: nx.div_ceil(TILE_SQUARES),
            tile_rows: ny.div_ceil(TILE_SQUARES),
        })
    }

    fn tile_count(self) -> u64 {
        self.tile_columns * self.tile_rows
    }

    fn owner_chunk(self, ids: &[u32], i: u64, j: u64) -> u32 {
        let x = (i / TILE_SQUARES).min(self.tile_columns - 1);
        let y = (j / TILE_SQUARES).min(self.tile_rows - 1);
        ids[(y * self.tile_columns + x) as usize]
    }

    fn grid_id(self, ids: &[u32], i: u64, j: u64) -> MeshId {
        let owner = self.owner_chunk(ids, i, j);
        let tile_x = (i / TILE_SQUARES).min(self.tile_columns - 1);
        let tile_y = (j / TILE_SQUARES).min(self.tile_rows - 1);
        let local_i = i - tile_x * TILE_SQUARES;
        let local_j = j - tile_y * TILE_SQUARES;
        let ordinal = 1 + local_j * (TILE_SQUARES + 1) + local_i;
        MeshId::from_raw((u64::from(owner) << 32) | ordinal)
    }
}

#[derive(Debug, Clone, Copy)]
struct GridTile {
    id: u32,
    i0: u64,
    i1: u64,
    j0: u64,
    j1: u64,
}

#[derive(Debug, Clone, Copy)]
struct Vertex {
    id: MeshId,
    grid: Option<(u64, u64)>,
    uv: [f64; 2],
    world: [f64; 3],
    sdf: f64,
    boundary: bool,
}

fn mesh_tile(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    grid: &LogicalGrid,
    tile: GridTile,
    chunk_ids: &[u32],
    context: &MeshingContext<'_>,
    advancing_front_mode: bool,
) -> MeshResult<crate::chunk::MeshChunk> {
    let u = [
        grid.u_min + tile.i0 as f64 * grid.du,
        grid.u_min + tile.i1 as f64 * grid.du,
    ];
    let v = [
        grid.v_min + tile.j0 as f64 * grid.dv,
        grid.v_min + tile.j1 as f64 * grid.dv,
    ];
    let bounds = Bounds3::from_points(
        [(u[0], v[0]), (u[1], v[0]), (u[1], v[1]), (u[0], v[1])].map(|(u, v)| {
            let point = space.point(u, v);
            [point.x, point.y, point.z]
        }),
    )
    .expanded(domain.boundary_tolerance() * 2.0);
    let mut builder = MeshChunkBuilder::new(tile.id, bounds)?;
    let width = tile.i1 - tile.i0 + 1;
    let height = tile.j1 - tile.j0 + 1;
    let mut samples = Vec::with_capacity((width * height) as usize);
    for j in tile.j0..=tile.j1 {
        for i in tile.i0..=tile.i1 {
            let u = grid.u_min + i as f64 * grid.du;
            let v = grid.v_min + j as f64 * grid.dv;
            let point = space.point(u, v);
            let sdf = space.sdf(u, v);
            if !sdf.is_finite() {
                return Err(MeshError::InvalidInput(format!(
                    "domain {:?} returned a non-finite SDF value",
                    domain.name
                )));
            }
            samples.push(Vertex {
                id: grid.grid_id(chunk_ids, i, j),
                grid: Some((i, j)),
                uv: [u, v],
                world: [point.x, point.y, point.z],
                sdf,
                boundary: sdf.abs() <= domain.boundary_tolerance(),
            });
        }
    }
    let sample =
        |i: u64, j: u64| -> Vertex { samples[((j - tile.j0) * width + (i - tile.i0)) as usize] };
    let mut added_points = BTreeSet::new();
    let mut boundary = BTreeMap::<(MeshId, MeshId), (Vec<u64>, [Vertex; 2])>::new();
    let catalog = context.catalog.domain(&domain.name)?;
    let use_quads = advancing_front_mode
        && context
            .controls
            .boundary_layers
            .iter()
            .any(|control| control.domain == domain.name);

    for j in tile.j0..tile.j1 {
        for i in tile.i0..tile.i1 {
            let corners = [
                sample(i, j),
                sample(i + 1, j),
                sample(i + 1, j + 1),
                sample(i, j + 1),
            ];
            if use_quads && corners.iter().all(|point| point.sdf <= 0.0) {
                for point in corners {
                    add_vertex(&mut builder, &mut added_points, point)?;
                }
                builder.quad4(corners.map(|point| point.id), catalog.zone, catalog.source)?;
                continue;
            }
            let triangles = if (i + j) % 2 == 0 {
                [
                    [corners[0], corners[1], corners[2]],
                    [corners[0], corners[2], corners[3]],
                ]
            } else {
                [
                    [corners[0], corners[1], corners[3]],
                    [corners[1], corners[2], corners[3]],
                ]
            };
            for triangle in triangles {
                let polygon = clip_triangle(domain, space, grid, chunk_ids, triangle)?;
                if polygon.len() < 3 {
                    continue;
                }
                collect_boundary(domain, space, &polygon, context, &mut boundary)?;
                for mut fragment in triangulate(&polygon) {
                    if signed_area(fragment) < 0.0 {
                        fragment.swap(1, 2);
                    }
                    if signed_area(fragment).abs() <= f64::EPSILON {
                        continue;
                    }
                    for point in fragment {
                        add_vertex(&mut builder, &mut added_points, point)?;
                    }
                    builder.tri3(fragment.map(|point| point.id), catalog.zone, catalog.source)?;
                }
            }
        }
    }
    for (_, (tags, edge)) in boundary {
        for point in edge {
            add_vertex(&mut builder, &mut added_points, point)?;
        }
        builder.boundary_edge(edge.map(|point| point.id), tags)?;
    }
    builder.build(2)
}

fn add_vertex(
    builder: &mut MeshChunkBuilder,
    added: &mut BTreeSet<MeshId>,
    vertex: Vertex,
) -> MeshResult<()> {
    if added.insert(vertex.id) {
        builder.point_copy(
            vertex.id,
            vertex.world,
            if vertex.boundary {
                "boundary"
            } else {
                "interior"
            },
            Vec::new(),
        )?;
    }
    Ok(())
}

fn clip_triangle(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    grid: &LogicalGrid,
    chunk_ids: &[u32],
    triangle: [Vertex; 3],
) -> MeshResult<Vec<Vertex>> {
    let mut polygon = Vec::with_capacity(4);
    for edge in 0..3 {
        let a = triangle[edge];
        let b = triangle[(edge + 1) % 3];
        match (a.sdf <= 0.0, b.sdf <= 0.0) {
            (true, true) => polygon.push(b),
            (true, false) => polygon.push(boundary_crossing(domain, space, grid, chunk_ids, a, b)?),
            (false, true) => {
                polygon.push(boundary_crossing(domain, space, grid, chunk_ids, a, b)?);
                polygon.push(b);
            }
            (false, false) => {}
        }
    }
    polygon.dedup_by_key(|point| point.id);
    if polygon.len() > 1
        && polygon.first().map(|point| point.id) == polygon.last().map(|point| point.id)
    {
        polygon.pop();
    }
    Ok(polygon)
}

fn boundary_crossing(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    grid: &LogicalGrid,
    chunk_ids: &[u32],
    mut a: Vertex,
    mut b: Vertex,
) -> MeshResult<Vertex> {
    let ga = a
        .grid
        .ok_or_else(|| MeshError::InvalidInput("crossing endpoint is not a grid point".into()))?;
    let gb = b
        .grid
        .ok_or_else(|| MeshError::InvalidInput("crossing endpoint is not a grid point".into()))?;
    if ga > gb {
        std::mem::swap(&mut a, &mut b);
    }
    if a.sdf.abs() <= domain.boundary_tolerance() {
        a.boundary = true;
        return Ok(a);
    }
    if b.sdf.abs() <= domain.boundary_tolerance() {
        b.boundary = true;
        return Ok(b);
    }
    let mut inside = if a.sdf <= 0.0 { a } else { b };
    let mut outside = if a.sdf <= 0.0 { b } else { a };
    let mut result = inside;
    for _ in 0..BISECTION_LIMIT {
        let uv = [
            (inside.uv[0] + outside.uv[0]) * 0.5,
            (inside.uv[1] + outside.uv[1]) * 0.5,
        ];
        let point = space.point(uv[0], uv[1]);
        let sdf = space.sdf(uv[0], uv[1]);
        result = Vertex {
            id: crossing_id(*grid, chunk_ids, ga, gb),
            grid: None,
            uv,
            world: [point.x, point.y, point.z],
            sdf,
            boundary: true,
        };
        if sdf.abs() <= domain.boundary_tolerance() {
            break;
        }
        if sdf <= 0.0 {
            inside = result;
        } else {
            outside = result;
        }
    }
    Ok(result)
}

fn crossing_id(grid: LogicalGrid, chunk_ids: &[u32], a: (u64, u64), b: (u64, u64)) -> MeshId {
    let ((ai, aj), (bi, bj)) = if a <= b { (a, b) } else { (b, a) };
    let (owner_i, owner_j, ordinal) = if aj == bj {
        let owner_j = aj.saturating_sub(u64::from(aj % TILE_SQUARES == 0 && aj > 0));
        let local_i = ai % TILE_SQUARES;
        let local_j = aj - (owner_j / TILE_SQUARES) * TILE_SQUARES;
        (ai, owner_j, 100_000 + local_j * TILE_SQUARES + local_i)
    } else if ai == bi {
        let owner_i = ai.saturating_sub(u64::from(ai % TILE_SQUARES == 0 && ai > 0));
        let local_i = ai - (owner_i / TILE_SQUARES) * TILE_SQUARES;
        let local_j = aj % TILE_SQUARES;
        (owner_i, aj, 200_000 + local_i * TILE_SQUARES + local_j)
    } else {
        (
            ai,
            aj,
            300_000 + (aj % TILE_SQUARES) * TILE_SQUARES + ai % TILE_SQUARES,
        )
    };
    let owner = grid.owner_chunk(chunk_ids, owner_i, owner_j);
    let _ = (bi, bj);
    MeshId::from_raw((u64::from(owner) << 32) | ordinal)
}

fn triangulate(polygon: &[Vertex]) -> Vec<[Vertex; 3]> {
    (1..polygon.len().saturating_sub(1))
        .map(|index| [polygon[0], polygon[index], polygon[index + 1]])
        .collect()
}

fn signed_area(triangle: [Vertex; 3]) -> f64 {
    let [a, b, c] = triangle.map(|vertex| vertex.uv);
    0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
}

fn collect_boundary(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    polygon: &[Vertex],
    context: &MeshingContext<'_>,
    result: &mut BoundarySegments,
) -> MeshResult<()> {
    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        if !a.boundary || !b.boundary || a.id == b.id {
            continue;
        }
        let uv = [(a.uv[0] + b.uv[0]) * 0.5, (a.uv[1] + b.uv[1]) * 0.5];
        if space.sdf(uv[0], uv[1]).abs() > domain.boundary_tolerance() * 2.0 {
            continue;
        }
        let point = space.point(uv[0], uv[1]);
        let class = domain
            .classify_boundary(&[point], BoundaryBand::UnprojectedSamples)
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
        let key = if a.id < b.id {
            (a.id, b.id)
        } else {
            (b.id, a.id)
        };
        result.entry(key).or_insert((vec![tag], [a, b]));
    }
    Ok(())
}
