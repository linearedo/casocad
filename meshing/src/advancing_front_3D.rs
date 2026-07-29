use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use caso_kernel::meshing::{BoundaryBand, MeshableDomain};
use caso_kernel::vec3::Vec3;

use crate::algorithm::{
    MeshSink, MeshingContext, MeshingPhase, MeshingProgress, MeshingStatistics,
};
use crate::chunk::{MeshChunkBuilder, MeshId};
use crate::error::{MeshError, MeshResult};
use crate::quality::{quality_score, QualityMetric};
use crate::schema::Bounds3;

const QUALITY_FLOOR: f64 = 0.05;
const VALID_QUALITY: f64 = 1.0e-8;
const SNAP_RATIO: f64 = 0.04;
const ESTIMATED_CHUNK_BYTES_PER_CELL: usize = 4_096;
const TET_EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
const TET_FACES: [[usize; 3]; 4] = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Leaf {
    level: u8,
    x: u32,
    y: u32,
    z: u32,
}

impl Leaf {
    fn children(self) -> [Self; 8] {
        std::array::from_fn(|index| Self {
            level: self.level + 1,
            x: self.x * 2 + (index & 1) as u32,
            y: self.y * 2 + ((index >> 1) & 1) as u32,
            z: self.z * 2 + ((index >> 2) & 1) as u32,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Lattice {
    x: u64,
    y: u64,
    z: u64,
}

#[derive(Debug, Clone, Copy)]
struct Grid {
    min: [f64; 3],
    max: [f64; 3],
    base: [u32; 3],
    max_depth: u8,
}

impl Grid {
    fn new(domain: &MeshableDomain, context: &MeshingContext<'_>) -> MeshResult<Self> {
        let bounds = &domain.bounds;
        let min = [bounds.x_min, bounds.y_min, bounds.z_min];
        let max = [bounds.x_max, bounds.y_max, bounds.z_max];
        let lengths = std::array::from_fn::<_, 3, _>(|axis| max[axis] - min[axis]);
        if lengths
            .into_iter()
            .any(|length| !length.is_finite() || length <= 0.0)
        {
            return Err(MeshError::InvalidInput(format!(
                "domain {:?} has invalid 3D bounds",
                domain.name
            )));
        }
        let base = lengths
            .map(|length| ((length / context.element_max_size).ceil() as u32).clamp(1, 1 << 18));
        let base_size = (0..3)
            .map(|axis| lengths[axis] / f64::from(base[axis]))
            .fold(0.0, f64::max);
        let mut max_depth = 0;
        let mut size = base_size;
        while max_depth < 24 && size * 0.5 >= context.element_min_size * (1.0 - 1.0e-12) {
            max_depth += 1;
            size *= 0.5;
        }
        Ok(Self {
            min,
            max,
            base,
            max_depth,
        })
    }

    fn fine_scale(self, leaf: Leaf) -> u64 {
        1u64 << (self.max_depth - leaf.level)
    }

    fn fine_bounds(self, leaf: Leaf) -> [u64; 6] {
        let scale = self.fine_scale(leaf);
        [
            u64::from(leaf.x) * scale,
            u64::from(leaf.x + 1) * scale,
            u64::from(leaf.y) * scale,
            u64::from(leaf.y + 1) * scale,
            u64::from(leaf.z) * scale,
            u64::from(leaf.z + 1) * scale,
        ]
    }

    fn corners(self, leaf: Leaf) -> [Lattice; 8] {
        let [x0, x1, y0, y1, z0, z1] = self.fine_bounds(leaf);
        [
            lattice(2 * x0, 2 * y0, 2 * z0),
            lattice(2 * x1, 2 * y0, 2 * z0),
            lattice(2 * x1, 2 * y1, 2 * z0),
            lattice(2 * x0, 2 * y1, 2 * z0),
            lattice(2 * x0, 2 * y0, 2 * z1),
            lattice(2 * x1, 2 * y0, 2 * z1),
            lattice(2 * x1, 2 * y1, 2 * z1),
            lattice(2 * x0, 2 * y1, 2 * z1),
        ]
    }

    fn center(self, leaf: Leaf) -> Lattice {
        let [x0, x1, y0, y1, z0, z1] = self.fine_bounds(leaf);
        lattice(x0 + x1, y0 + y1, z0 + z1)
    }

    fn world(self, key: Lattice) -> [f64; 3] {
        let scale = (1u64 << self.max_depth) as f64;
        let denominator = self.base.map(|count| 2.0 * f64::from(count) * scale);
        [
            (self.max[0] - self.min[0]).mul_add(key.x as f64 / denominator[0], self.min[0]),
            (self.max[1] - self.min[1]).mul_add(key.y as f64 / denominator[1], self.min[1]),
            (self.max[2] - self.min[2]).mul_add(key.z as f64 / denominator[2], self.min[2]),
        ]
    }

    fn roots(self) -> Vec<Leaf> {
        let capacity = u64::from(self.base[0])
            .saturating_mul(u64::from(self.base[1]))
            .saturating_mul(u64::from(self.base[2]))
            .min(usize::MAX as u64) as usize;
        let mut result = Vec::with_capacity(capacity);
        for z in 0..self.base[2] {
            for y in 0..self.base[1] {
                for x in 0..self.base[0] {
                    result.push(Leaf { level: 0, x, y, z });
                }
            }
        }
        result
    }
}

#[derive(Debug, Clone, Copy)]
struct Sample {
    key: Lattice,
    world: [f64; 3],
    sdf: f64,
}

struct Sampler<'a> {
    domain: &'a MeshableDomain,
    grid: Grid,
    cache: BTreeMap<Lattice, Sample>,
}

impl<'a> Sampler<'a> {
    fn new(domain: &'a MeshableDomain, grid: Grid) -> Self {
        Self {
            domain,
            grid,
            cache: BTreeMap::new(),
        }
    }

    fn sample(&mut self, key: Lattice) -> MeshResult<Sample> {
        if let Some(sample) = self.cache.get(&key) {
            return Ok(*sample);
        }
        let world = self.grid.world(key);
        let sdf = self.domain.domain_sdf(&[Vec3::from_array(world)])[0];
        if !sdf.is_finite() {
            return Err(MeshError::InvalidInput(format!(
                "domain {:?} returned a non-finite SDF value",
                self.domain.name
            )));
        }
        let sample = Sample { key, world, sdf };
        self.cache.insert(key, sample);
        Ok(sample)
    }

    fn leaf_samples(&mut self, leaf: Leaf) -> MeshResult<Vec<Sample>> {
        let [x0, x1, y0, y1, z0, z1] = self.grid.fine_bounds(leaf);
        let coordinates = [
            [2 * x0, x0 + x1, 2 * x1],
            [2 * y0, y0 + y1, 2 * y1],
            [2 * z0, z0 + z1, 2 * z1],
        ];
        let mut result = Vec::with_capacity(27);
        for z in coordinates[2] {
            for y in coordinates[1] {
                for x in coordinates[0] {
                    result.push(self.sample(lattice(x, y, z))?);
                }
            }
        }
        Ok(result)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PointKey {
    Lattice(Lattice),
    Crossing(Lattice, Lattice),
    Steiner(u64),
}

type QualityObjective = (usize, f64);
type TetTemplate = (QualityObjective, Vec<[PointKey; 4]>);
type FanTemplate = (QualityObjective, [f64; 3], Vec<[PointKey; 3]>);

#[derive(Debug, Clone, Copy)]
struct Point {
    world: [f64; 3],
    boundary: bool,
}

#[derive(Debug, Clone)]
struct Cell {
    points: [PointKey; 4],
    leaf: Leaf,
    certified_interior: bool,
}

#[derive(Debug)]
struct Candidate {
    points: BTreeMap<PointKey, Point>,
    cells: Vec<Cell>,
    construction_failures: BTreeSet<Leaf>,
    next_steiner: u64,
}

#[derive(Debug, Clone)]
struct BoundaryFace {
    points: [PointKey; 3],
    cell: usize,
}

#[derive(Debug)]
struct Assessment {
    boundary: Vec<BoundaryFace>,
    boundary_vertices: BTreeSet<PointKey>,
    poor: BinaryHeap<PoorCell>,
    refine: BTreeSet<Leaf>,
    reason: Option<String>,
    location: Option<[f64; 3]>,
    worst_quality: f64,
}

#[derive(Debug, Clone, Copy)]
struct PoorCell {
    quality: f64,
    index: usize,
}

impl PartialEq for PoorCell {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.quality.to_bits() == other.quality.to_bits()
    }
}

impl Eq for PoorCell {}

impl PartialOrd for PoorCell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PoorCell {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .quality
            .total_cmp(&self.quality)
            .then_with(|| other.index.cmp(&self.index))
    }
}

pub(crate) fn generate(
    context: &MeshingContext<'_>,
    sink: &mut dyn MeshSink,
) -> MeshResult<MeshingStatistics> {
    let mut statistics = MeshingStatistics {
        domains: context.domains.len() as u64,
        ..MeshingStatistics::default()
    };
    for domain in context.domains.iter() {
        context.check()?;
        if domain.dimension != 3 {
            return Err(MeshError::UnsupportedDimension {
                domain: domain.name.clone(),
                dimension: domain.dimension,
            });
        }
        let grid = Grid::new(domain, context)?;
        let mut sampler = Sampler::new(domain, grid);
        let mut leaves = discover(context, domain, &mut sampler)?;
        balance(context, grid, &mut leaves)?;

        let (candidate, assessment) = loop {
            context.check()?;
            let mut candidate = build_candidate(context, domain, &mut sampler, &leaves)?;
            let mut assessment = assess(domain, context, &candidate)?;
            if assessment.refine.is_empty() && !assessment.poor.is_empty() {
                repair(domain, context, &mut candidate, &mut assessment)?;
            }
            if assessment.refine.is_empty() && assessment.poor.is_empty() {
                break (candidate, assessment);
            }
            let requested = if assessment.refine.is_empty() {
                assessment
                    .poor
                    .iter()
                    .map(|poor| candidate.cells[poor.index].leaf)
                    .collect()
            } else {
                assessment.refine.clone()
            };
            let splittable = requested
                .iter()
                .copied()
                .filter(|leaf| leaf.level < grid.max_depth)
                .collect::<BTreeSet<_>>();
            if splittable.len() != requested.len() {
                return Err(minimum_size_error(
                    domain,
                    context,
                    assessment.reason.as_deref().unwrap_or(
                        "local repair could not satisfy the Scaled Jacobian validity floor",
                    ),
                    assessment.location,
                    assessment.worst_quality,
                ));
            }
            refine_leaves(context, &mut leaves, &splittable)?;
            balance(context, grid, &mut leaves)?;
        };
        emit(
            context,
            domain,
            &candidate,
            &assessment,
            sink,
            &mut statistics,
        )?;
    }
    Ok(statistics)
}

fn discover(
    context: &MeshingContext<'_>,
    domain: &MeshableDomain,
    sampler: &mut Sampler<'_>,
) -> MeshResult<Vec<Leaf>> {
    let mut pending = sampler.grid.roots();
    pending.reverse();
    let mut leaves = Vec::new();
    let mut visited = 0usize;
    while let Some(leaf) = pending.pop() {
        if visited.is_multiple_of(128) {
            context.check()?;
        }
        visited += 1;
        let samples = sampler.leaf_samples(leaf)?;
        let center = samples[13];
        let corners = sampler.grid.corners(leaf);
        let diagonal = distance3(
            sampler.grid.world(corners[0]),
            sampler.grid.world(corners[6]),
        );
        let radius = diagonal * 0.5;
        let size = leaf_size_3d(sampler.grid, leaf);
        let tolerance = root_tolerance(domain, size);
        let target = samples
            .iter()
            .map(|sample| {
                context
                    .controls
                    .size_at(
                        &domain.name,
                        Vec3::from_array(sample.world),
                        context.element_max_size,
                    )
                    .clamp(context.element_min_size, context.element_max_size)
            })
            .reduce(f64::min)
            .unwrap_or(context.element_max_size);
        let negative = samples
            .iter()
            .filter(|sample| sample.sdf < -tolerance)
            .count();
        let positive = samples
            .iter()
            .filter(|sample| sample.sdf > tolerance)
            .count();
        let on_boundary = samples.len() - negative - positive;
        let mixed = (negative > 0 && positive > 0) || on_boundary > 0;
        let certified_inside = center.sdf < 0.0 && -center.sdf > radius + tolerance;
        let certified_outside = center.sdf > 0.0 && center.sdf > radius + tolerance;
        let unresolved_uniform = !mixed && !certified_inside && !certified_outside;
        let unresolved_curvature =
            mixed && curvature_requires_refinement(domain, &samples, radius, size)?;
        let split = leaf.level < sampler.grid.max_depth
            && (size > target * 1.35
                || (mixed && size > target * 1.05)
                || unresolved_uniform
                || unresolved_curvature);
        if split {
            for child in leaf.children().into_iter().rev() {
                pending.push(child);
            }
        } else if !certified_outside || negative > 0 || on_boundary > 0 {
            leaves.push(leaf);
        }
        if leaves.len().saturating_add(pending.len()) as u64
            > context.limits.max_cells.saturating_mul(2)
        {
            return Err(MeshError::LimitExceeded(
                "adaptive 3D discovery exceeded the configured cell limit".into(),
            ));
        }
    }
    leaves.sort_unstable();
    Ok(leaves)
}

fn curvature_requires_refinement(
    domain: &MeshableDomain,
    samples: &[Sample],
    radius: f64,
    local_size: f64,
) -> MeshResult<bool> {
    let Some(seed) = samples
        .iter()
        .filter(|sample| sample.sdf <= 0.0)
        .max_by(|a, b| a.sdf.total_cmp(&b.sdf))
    else {
        return Ok(false);
    };
    let projection = domain
        .project_to_boundary(&[Vec3::from_array(seed.world)])
        .map_err(|error| MeshError::InvalidInput(error.to_string()))?[0];
    if !projection.converged {
        return Ok(true);
    }
    let curvature = domain
        .curvature(&[projection.point])
        .map_err(|error| MeshError::InvalidInput(error.to_string()))?[0];
    Ok(!curvature.is_finite()
        || curvature.abs() * radius.powi(2) > chord_tolerance(domain, local_size))
}

fn balance(context: &MeshingContext<'_>, grid: Grid, leaves: &mut Vec<Leaf>) -> MeshResult<()> {
    loop {
        context.check()?;
        let index = LeafIndex::new(grid, leaves);
        let mut split = BTreeSet::new();
        for (leaf_index, &leaf) in leaves.iter().enumerate() {
            if leaf_index % 128 == 0 {
                context.check()?;
            }
            for side in 0..6 {
                if index
                    .neighbors(leaf, side)
                    .into_iter()
                    .flatten()
                    .any(|neighbor| neighbor.level > leaf.level + 1)
                {
                    split.insert(leaf);
                }
            }
        }
        if split.is_empty() {
            return Ok(());
        }
        refine_leaves(context, leaves, &split)?;
    }
}

fn refine_leaves(
    context: &MeshingContext<'_>,
    leaves: &mut Vec<Leaf>,
    split: &BTreeSet<Leaf>,
) -> MeshResult<()> {
    let mut refined = Vec::with_capacity(leaves.len() + split.len() * 7);
    for leaf in leaves.drain(..) {
        if split.contains(&leaf) {
            refined.extend(leaf.children());
        } else {
            refined.push(leaf);
        }
    }
    if refined.len() as u64 > context.limits.max_cells.saturating_mul(2) {
        return Err(MeshError::LimitExceeded(
            "adaptive 3D refinement exceeded the configured cell limit".into(),
        ));
    }
    refined.sort_unstable();
    *leaves = refined;
    Ok(())
}

struct LeafIndex {
    grid: Grid,
    leaves: BTreeMap<(u8, u32, u32, u32), Leaf>,
}

impl LeafIndex {
    fn new(grid: Grid, leaves: &[Leaf]) -> Self {
        Self {
            grid,
            leaves: leaves
                .iter()
                .copied()
                .map(|leaf| ((leaf.level, leaf.x, leaf.y, leaf.z), leaf))
                .collect(),
        }
    }

    fn owner(&self, x: u64, y: u64, z: u64) -> Option<Leaf> {
        for level in (0..=self.grid.max_depth).rev() {
            let scale = 1u64 << (self.grid.max_depth - level);
            let key = (
                level,
                (x / scale) as u32,
                (y / scale) as u32,
                (z / scale) as u32,
            );
            if let Some(leaf) = self.leaves.get(&key) {
                return Some(*leaf);
            }
        }
        None
    }

    fn neighbors(&self, leaf: Leaf, side: usize) -> [Option<Leaf>; 4] {
        let [x0, x1, y0, y1, z0, z1] = self.grid.fine_bounds(leaf);
        let quarters = |a: u64, b: u64| {
            if b - a <= 1 {
                [a, a]
            } else {
                [a + (b - a) / 4, a + 3 * (b - a) / 4]
            }
        };
        let xs = quarters(x0, x1);
        let ys = quarters(y0, y1);
        let zs = quarters(z0, z1);
        let max = [
            u64::from(self.grid.base[0]) * (1u64 << self.grid.max_depth),
            u64::from(self.grid.base[1]) * (1u64 << self.grid.max_depth),
            u64::from(self.grid.base[2]) * (1u64 << self.grid.max_depth),
        ];
        let coordinates = match side {
            0 if z0 > 0 => [
                (xs[0], ys[0], z0 - 1),
                (xs[1], ys[0], z0 - 1),
                (xs[1], ys[1], z0 - 1),
                (xs[0], ys[1], z0 - 1),
            ],
            1 if z1 < max[2] => [
                (xs[0], ys[0], z1),
                (xs[1], ys[0], z1),
                (xs[1], ys[1], z1),
                (xs[0], ys[1], z1),
            ],
            2 if y0 > 0 => [
                (xs[0], y0 - 1, zs[0]),
                (xs[1], y0 - 1, zs[0]),
                (xs[1], y0 - 1, zs[1]),
                (xs[0], y0 - 1, zs[1]),
            ],
            3 if x1 < max[0] => [
                (x1, ys[0], zs[0]),
                (x1, ys[1], zs[0]),
                (x1, ys[1], zs[1]),
                (x1, ys[0], zs[1]),
            ],
            4 if y1 < max[1] => [
                (xs[0], y1, zs[0]),
                (xs[1], y1, zs[0]),
                (xs[1], y1, zs[1]),
                (xs[0], y1, zs[1]),
            ],
            5 if x0 > 0 => [
                (x0 - 1, ys[0], zs[0]),
                (x0 - 1, ys[1], zs[0]),
                (x0 - 1, ys[1], zs[1]),
                (x0 - 1, ys[0], zs[1]),
            ],
            _ => return [None; 4],
        };
        coordinates.map(|(x, y, z)| self.owner(x, y, z))
    }

    fn has_finer_neighbor(&self, leaf: Leaf, side: usize) -> bool {
        self.neighbors(leaf, side)
            .into_iter()
            .flatten()
            .any(|neighbor| neighbor.level > leaf.level)
    }

    fn edge_has_finer(&self, leaf: Leaf, a: Lattice, b: Lattice) -> bool {
        let a = [a.x / 2, a.y / 2, a.z / 2];
        let b = [b.x / 2, b.y / 2, b.z / 2];
        let max = [
            u64::from(self.grid.base[0]) * (1u64 << self.grid.max_depth),
            u64::from(self.grid.base[1]) * (1u64 << self.grid.max_depth),
            u64::from(self.grid.base[2]) * (1u64 << self.grid.max_depth),
        ];
        let varying = (0..3).find(|axis| a[*axis] != b[*axis]);
        let Some(axis) = varying else {
            return false;
        };
        let along = (a[axis].min(b[axis]) + a[axis].max(b[axis]) - 1) / 2;
        let perpendicular = (0..3).filter(|other| *other != axis).collect::<Vec<_>>();
        let adjacent = |coordinate: u64, limit: u64| {
            [
                (coordinate > 0).then_some(coordinate.saturating_sub(1)),
                (coordinate < limit).then_some(coordinate),
            ]
        };
        for first in adjacent(a[perpendicular[0]], max[perpendicular[0]])
            .into_iter()
            .flatten()
        {
            for second in adjacent(a[perpendicular[1]], max[perpendicular[1]])
                .into_iter()
                .flatten()
            {
                let mut cell = [0u64; 3];
                cell[axis] = along;
                cell[perpendicular[0]] = first;
                cell[perpendicular[1]] = second;
                if self
                    .owner(cell[0], cell[1], cell[2])
                    .is_some_and(|owner| owner.level > leaf.level)
                {
                    return true;
                }
            }
        }
        false
    }
}

fn build_candidate(
    context: &MeshingContext<'_>,
    domain: &MeshableDomain,
    sampler: &mut Sampler<'_>,
    leaves: &[Leaf],
) -> MeshResult<Candidate> {
    let index = LeafIndex::new(sampler.grid, leaves);
    let mut candidate = Candidate {
        points: BTreeMap::new(),
        cells: Vec::new(),
        construction_failures: BTreeSet::new(),
        next_steiner: 1,
    };
    let mut crossings = BTreeMap::new();
    for (leaf_index, &leaf) in leaves.iter().enumerate() {
        if leaf_index % 128 == 0 {
            context.check()?;
        }
        let corners = sampler.grid.corners(leaf);
        let center = sampler.grid.center(leaf);
        let center_sample = sampler.sample(center)?;
        let radius = distance3(
            sampler.grid.world(corners[0]),
            sampler.grid.world(corners[6]),
        ) * 0.5;
        let local_size = leaf_size_3d(sampler.grid, leaf);
        let certified_interior = center_sample.sdf < 0.0
            && -center_sample.sdf > radius + root_tolerance(domain, local_size);
        let faces = [
            [corners[0], corners[3], corners[2], corners[1]],
            [corners[4], corners[5], corners[6], corners[7]],
            [corners[0], corners[1], corners[5], corners[4]],
            [corners[1], corners[2], corners[6], corners[5]],
            [corners[2], corners[3], corners[7], corners[6]],
            [corners[3], corners[0], corners[4], corners[7]],
        ];
        let transition = (0..6).any(|side| {
            index
                .neighbors(leaf, side)
                .into_iter()
                .flatten()
                .any(|neighbor| neighbor.level != leaf.level)
                || (0..4).any(|edge| {
                    index.edge_has_finer(leaf, faces[side][edge], faces[side][(edge + 1) % 4])
                })
        });
        if !transition {
            let regular = [
                [corners[0], corners[1], corners[2], corners[6]],
                [corners[0], corners[2], corners[3], corners[6]],
                [corners[0], corners[3], corners[7], corners[6]],
                [corners[0], corners[7], corners[4], corners[6]],
                [corners[0], corners[4], corners[5], corners[6]],
                [corners[0], corners[5], corners[1], corners[6]],
            ];
            for mut tet in regular {
                let mut samples = [
                    sampler.sample(tet[0])?,
                    sampler.sample(tet[1])?,
                    sampler.sample(tet[2])?,
                    sampler.sample(tet[3])?,
                ];
                if signed_volume_samples(samples) < 0.0 {
                    tet.swap(2, 3);
                    samples.swap(2, 3);
                }
                clip_tetrahedron(
                    domain,
                    local_size,
                    samples,
                    &mut candidate,
                    &mut crossings,
                    leaf,
                    certified_interior,
                )?;
            }
            continue;
        }
        for (side, face) in faces.into_iter().enumerate() {
            for triangle in conforming_face_triangles(face, leaf, side, &index, sampler)? {
                let mut tet = [center, triangle[0], triangle[1], triangle[2]];
                let samples = [
                    sampler.sample(tet[0])?,
                    sampler.sample(tet[1])?,
                    sampler.sample(tet[2])?,
                    sampler.sample(tet[3])?,
                ];
                if signed_volume_samples(samples) < 0.0 {
                    tet.swap(2, 3);
                }
                let samples = [
                    sampler.sample(tet[0])?,
                    sampler.sample(tet[1])?,
                    sampler.sample(tet[2])?,
                    sampler.sample(tet[3])?,
                ];
                clip_tetrahedron(
                    domain,
                    leaf_size_3d(sampler.grid, leaf),
                    samples,
                    &mut candidate,
                    &mut crossings,
                    leaf,
                    certified_interior,
                )?;
            }
        }
    }
    Ok(candidate)
}

fn conforming_face_triangles(
    face: [Lattice; 4],
    leaf: Leaf,
    side: usize,
    index: &LeafIndex,
    sampler: &mut Sampler<'_>,
) -> MeshResult<Vec<[Lattice; 3]>> {
    if index.has_finer_neighbor(leaf, side) {
        let mut result = Vec::with_capacity(8);
        for quad in subdivide_quad(face) {
            result.extend(triangulate_face(quad, sampler)?);
        }
        return Ok(result);
    }
    let split = std::array::from_fn::<_, 4, _>(|edge| {
        index.edge_has_finer(leaf, face[edge], face[(edge + 1) % 4])
    });
    if !split.into_iter().any(|value| value) {
        return Ok(triangulate_face(face, sampler)?.to_vec());
    }
    let mut ring = Vec::with_capacity(8);
    for edge in 0..4 {
        ring.push(face[edge]);
        if split[edge] {
            ring.push(midpoint(face[edge], face[(edge + 1) % 4]));
        }
    }
    let center = midpoint(face[0], face[2]);
    Ok((0..ring.len())
        .map(|edge| [center, ring[edge], ring[(edge + 1) % ring.len()]])
        .collect())
}

fn subdivide_quad(quad: [Lattice; 4]) -> [[Lattice; 4]; 4] {
    let ab = midpoint(quad[0], quad[1]);
    let bc = midpoint(quad[1], quad[2]);
    let cd = midpoint(quad[2], quad[3]);
    let da = midpoint(quad[3], quad[0]);
    let center = midpoint(ab, cd);
    [
        [quad[0], ab, center, da],
        [ab, quad[1], bc, center],
        [center, bc, quad[2], cd],
        [da, center, cd, quad[3]],
    ]
}

fn triangulate_face(
    quad: [Lattice; 4],
    sampler: &mut Sampler<'_>,
) -> MeshResult<[[Lattice; 3]; 2]> {
    let points = [
        sampler.sample(quad[0])?.world,
        sampler.sample(quad[1])?.world,
        sampler.sample(quad[2])?.world,
        sampler.sample(quad[3])?.world,
    ];
    let first = [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]];
    let second = [[quad[0], quad[1], quad[3]], [quad[1], quad[2], quad[3]]];
    let first_score = face_pair_score([
        [points[0], points[1], points[2]],
        [points[0], points[2], points[3]],
    ]);
    let second_score = face_pair_score([
        [points[0], points[1], points[3]],
        [points[1], points[2], points[3]],
    ]);
    Ok(
        if first_score > second_score + 1.0e-12
            || ((first_score - second_score).abs() <= 1.0e-12
                && ordered_pair(quad[0], quad[2]) <= ordered_pair(quad[1], quad[3]))
        {
            first
        } else {
            second
        },
    )
}

fn clip_tetrahedron(
    domain: &MeshableDomain,
    local_size: f64,
    samples: [Sample; 4],
    candidate: &mut Candidate,
    crossings: &mut BTreeMap<(Lattice, Lattice), PointKey>,
    leaf: Leaf,
    certified_interior: bool,
) -> MeshResult<()> {
    let mask = samples
        .iter()
        .enumerate()
        .fold(0u8, |mask, (index, sample)| {
            mask | (u8::from(sample.sdf <= 0.0) << index)
        });
    if mask == 0 {
        return Ok(());
    }
    for sample in samples {
        if sample.sdf <= 0.0 {
            candidate
                .points
                .entry(PointKey::Lattice(sample.key))
                .or_insert(Point {
                    world: sample.world,
                    boundary: sample.sdf == 0.0,
                });
        }
    }
    if mask == 15 {
        let mut points = samples.map(|sample| PointKey::Lattice(sample.key));
        if signed_volume(points, &candidate.points) < 0.0 {
            points.swap(2, 3);
        }
        candidate.cells.push(Cell {
            points,
            leaf,
            certified_interior,
        });
        return Ok(());
    }

    let mut edge_crossings = BTreeMap::<(usize, usize), PointKey>::new();
    for (a, b) in marching_edges(mask) {
        let key = crossing(
            domain,
            local_size,
            samples[a],
            samples[b],
            &mut candidate.points,
            crossings,
        )?;
        edge_crossings.insert(ordered_pair(a, b), key);
    }

    let mut faces = Vec::<Vec<PointKey>>::new();
    let mut cut_adjacency = BTreeMap::<PointKey, BTreeSet<PointKey>>::new();
    for face in TET_FACES {
        let face_samples = [samples[face[0]], samples[face[1]], samples[face[2]]];
        let polygon = clipped_triangle(
            face_samples,
            |a, b| {
                edge_crossings
                    .get(&ordered_pair(a, b))
                    .copied()
                    .ok_or_else(|| {
                        MeshError::InvalidInput(
                            "marching-tetrahedron face is missing a cached crossing".into(),
                        )
                    })
            },
            face,
        )?;
        let polygon = dedup_polygon(polygon);
        if polygon.len() >= 3 {
            faces.push(polygon);
        }
        let crossing_keys = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
            .into_iter()
            .filter_map(|edge| edge_crossings.get(&ordered_pair(edge.0, edge.1)).copied())
            .collect::<BTreeSet<_>>();
        if crossing_keys.len() == 2 {
            let mut iter = crossing_keys.into_iter();
            let a = iter.next().expect("two crossings");
            let b = iter.next().expect("two crossings");
            cut_adjacency.entry(a).or_default().insert(b);
            cut_adjacency.entry(b).or_default().insert(a);
        }
    }
    if !cut_adjacency.is_empty() {
        let cut = trace_cycle(&cut_adjacency);
        if cut.len() >= 3 {
            faces.push(cut);
        }
    }
    let vertices = faces
        .iter()
        .flat_map(|face| face.iter().copied())
        .collect::<BTreeSet<_>>();
    if vertices.len() < 4 || faces.len() < 4 {
        return Ok(());
    }
    if vertices.len() == 4 && faces.iter().all(|face| face.len() == 3) {
        let mut points = vertices.into_iter().collect::<Vec<_>>();
        let mut tet = [
            points.remove(0),
            points.remove(0),
            points.remove(0),
            points.remove(0),
        ];
        if signed_volume(tet, &candidate.points) < 0.0 {
            tet.swap(2, 3);
        }
        if signed_volume(tet, &candidate.points).abs() <= volume_tolerance(local_size) {
        } else {
            candidate.cells.push(Cell {
                points: tet,
                leaf,
                certified_interior: false,
            });
        }
        return Ok(());
    }
    let mut direct = None;
    if vertices.len() == 6 {
        let inside = (0..4)
            .filter(|index| mask & (1 << index) != 0)
            .collect::<Vec<_>>();
        let outside = (0..4)
            .filter(|index| mask & (1 << index) == 0)
            .collect::<Vec<_>>();
        let prism = if inside.len() == 3 {
            let outside = outside[0];
            Some([
                [
                    PointKey::Lattice(samples[inside[0]].key),
                    PointKey::Lattice(samples[inside[1]].key),
                    PointKey::Lattice(samples[inside[2]].key),
                ],
                [
                    edge_crossings[&ordered_pair(inside[0], outside)],
                    edge_crossings[&ordered_pair(inside[1], outside)],
                    edge_crossings[&ordered_pair(inside[2], outside)],
                ],
            ])
        } else if inside.len() == 2 {
            Some([
                [
                    PointKey::Lattice(samples[inside[0]].key),
                    edge_crossings[&ordered_pair(inside[0], outside[0])],
                    edge_crossings[&ordered_pair(inside[0], outside[1])],
                ],
                [
                    PointKey::Lattice(samples[inside[1]].key),
                    edge_crossings[&ordered_pair(inside[1], outside[0])],
                    edge_crossings[&ordered_pair(inside[1], outside[1])],
                ],
            ])
        } else {
            None
        };
        direct = prism
            .and_then(|prism| best_prism_template(prism, &faces, &candidate.points, local_size));
    }

    let fan = best_fan_template(&vertices, &faces, &candidate.points, local_size);
    if direct.as_ref().is_some_and(|(direct_objective, _)| {
        fan.as_ref()
            .is_none_or(|(fan_objective, _, _)| !improves(*direct_objective, *fan_objective))
    }) {
        let (_, tets) = direct.expect("checked direct template");
        candidate.cells.extend(tets.into_iter().map(|points| Cell {
            points,
            leaf,
            certified_interior: false,
        }));
        return Ok(());
    }
    if let Some((_, center, triangles)) = fan {
        let center_key = PointKey::Steiner(candidate.next_steiner);
        candidate.next_steiner += 1;
        candidate.points.insert(
            center_key,
            Point {
                world: center,
                boundary: false,
            },
        );
        for triangle in triangles {
            let mut tet = [center_key, triangle[0], triangle[1], triangle[2]];
            orient_tet(&mut tet, &candidate.points);
            candidate.cells.push(Cell {
                points: tet,
                leaf,
                certified_interior: false,
            });
        }
    } else {
        candidate.construction_failures.insert(leaf);
    }
    Ok(())
}

fn clipped_triangle(
    samples: [Sample; 3],
    mut crossing_for_indices: impl FnMut(usize, usize) -> MeshResult<PointKey>,
    indices: [usize; 3],
) -> MeshResult<Vec<PointKey>> {
    let mut polygon = Vec::with_capacity(4);
    for edge in 0..3 {
        let a = edge;
        let b = (edge + 1) % 3;
        match (samples[a].sdf <= 0.0, samples[b].sdf <= 0.0) {
            (true, true) => polygon.push(PointKey::Lattice(samples[b].key)),
            (true, false) => polygon.push(crossing_for_indices(indices[a], indices[b])?),
            (false, true) => {
                polygon.push(crossing_for_indices(indices[a], indices[b])?);
                polygon.push(PointKey::Lattice(samples[b].key));
            }
            (false, false) => {}
        }
    }
    Ok(polygon)
}

fn crossing(
    domain: &MeshableDomain,
    local_size: f64,
    a: Sample,
    b: Sample,
    points: &mut BTreeMap<PointKey, Point>,
    crossings: &mut BTreeMap<(Lattice, Lattice), PointKey>,
) -> MeshResult<PointKey> {
    let edge = ordered_pair(a.key, b.key);
    if let Some(key) = crossings.get(&edge) {
        return Ok(*key);
    }
    if a.sdf == 0.0 || b.sdf == 0.0 {
        let sample = if a.sdf == 0.0 { a } else { b };
        let key = PointKey::Lattice(sample.key);
        points.insert(
            key,
            Point {
                world: sample.world,
                boundary: true,
            },
        );
        crossings.insert(edge, key);
        return Ok(key);
    }
    let (mut inside, mut outside, mut ti, mut to) = if a.sdf < 0.0 {
        (a, b, 0.0, 1.0)
    } else {
        (b, a, 1.0, 0.0)
    };
    let tolerance = root_tolerance(domain, local_size);
    for _ in 0..64 {
        if distance3(inside.world, outside.world) <= tolerance {
            break;
        }
        let t = (ti + to) * 0.5;
        let world = [
            a.world[0] + t * (b.world[0] - a.world[0]),
            a.world[1] + t * (b.world[1] - a.world[1]),
            a.world[2] + t * (b.world[2] - a.world[2]),
        ];
        let sdf = domain.domain_sdf(&[Vec3::from_array(world)])[0];
        if !sdf.is_finite() {
            return Err(MeshError::InvalidInput(format!(
                "domain {:?} returned a non-finite SDF value",
                domain.name
            )));
        }
        let sample = Sample {
            key: a.key,
            world,
            sdf,
        };
        if sdf <= 0.0 {
            inside = sample;
            ti = t;
        } else {
            outside = sample;
            to = t;
        }
    }
    let t = (ti + to) * 0.5;
    let snapped = if t <= SNAP_RATIO {
        Some(a.key)
    } else if 1.0 - t <= SNAP_RATIO {
        Some(b.key)
    } else {
        None
    };
    let projection = domain
        .project_to_boundary(&[Vec3::from_array(inside.world)])
        .map_err(|error| MeshError::InvalidInput(error.to_string()))?
        .into_iter()
        .next()
        .expect("one projection");
    let world = if projection.converged {
        projection.point.to_array()
    } else {
        inside.world
    };
    let point = Point {
        world,
        boundary: true,
    };
    let key = snapped
        .map(PointKey::Lattice)
        .unwrap_or(PointKey::Crossing(edge.0, edge.1));
    points
        .entry(key)
        .and_modify(|existing| {
            if !existing.boundary {
                *existing = point;
            }
        })
        .or_insert(point);
    crossings.insert(edge, key);
    Ok(key)
}

fn marching_edges(mask: u8) -> Vec<(usize, usize)> {
    TET_EDGES
        .into_iter()
        .filter(|(a, b)| ((mask >> a) & 1) != ((mask >> b) & 1))
        .collect()
}

fn trace_cycle(adjacency: &BTreeMap<PointKey, BTreeSet<PointKey>>) -> Vec<PointKey> {
    let Some(&start) = adjacency.keys().next() else {
        return Vec::new();
    };
    let mut result = vec![start];
    let mut previous = None;
    let mut current = start;
    for _ in 0..adjacency.len() {
        let Some(next) = adjacency.get(&current).and_then(|neighbors| {
            neighbors
                .iter()
                .copied()
                .find(|neighbor| Some(*neighbor) != previous)
        }) else {
            return Vec::new();
        };
        if next == start {
            return result;
        }
        result.push(next);
        previous = Some(current);
        current = next;
    }
    Vec::new()
}

fn triangulate_polygon(
    polygon: &[PointKey],
    points: &BTreeMap<PointKey, Point>,
) -> Vec<[PointKey; 3]> {
    match polygon {
        [a, b, c] => vec![[*a, *b, *c]],
        [a, b, c, d] => {
            let first = [[*a, *b, *c], [*a, *c, *d]];
            let second = [[*a, *b, *d], [*b, *c, *d]];
            let first_score = triangle_pair_quality(first, points);
            let second_score = triangle_pair_quality(second, points);
            if first_score > second_score + 1.0e-12
                || ((first_score - second_score).abs() <= 1.0e-12
                    && ordered_pair(*a, *c) <= ordered_pair(*b, *d))
            {
                first.to_vec()
            } else {
                second.to_vec()
            }
        }
        _ => (1..polygon.len().saturating_sub(1))
            .map(|index| [polygon[0], polygon[index], polygon[index + 1]])
            .collect(),
    }
}

fn best_prism_template(
    prism: [[PointKey; 3]; 2],
    faces: &[Vec<PointKey>],
    points: &BTreeMap<PointKey, Point>,
    local_size: f64,
) -> Option<TetTemplate> {
    const PERMUTATIONS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let desired = faces
        .iter()
        .flat_map(|face| triangulate_polygon(face, points))
        .map(sorted3)
        .collect::<BTreeSet<_>>();
    let mut best: Option<TetTemplate> = None;
    for permutation in PERMUTATIONS {
        let p = permutation.map(|index| prism[0][index]);
        let q = permutation.map(|index| prism[1][index]);
        let mut tets = vec![
            [p[0], p[1], p[2], q[0]],
            [p[1], p[2], q[0], q[1]],
            [p[2], q[0], q[1], q[2]],
        ];
        for tet in &mut tets {
            orient_tet(tet, points);
        }
        if tets
            .iter()
            .any(|tet| signed_volume(*tet, points) <= volume_tolerance(local_size))
        {
            continue;
        }
        let mut incidence = BTreeMap::<[PointKey; 3], usize>::new();
        for tet in &tets {
            for face in TET_FACES {
                *incidence
                    .entry(sorted3([tet[face[0]], tet[face[1]], tet[face[2]]]))
                    .or_default() += 1;
            }
        }
        let exposed = incidence
            .into_iter()
            .filter_map(|(face, count)| (count == 1).then_some(face))
            .collect::<BTreeSet<_>>();
        if exposed != desired {
            continue;
        }
        let objective = tets
            .iter()
            .map(|tet| {
                quality_score(
                    "tet4",
                    &tet.map(|key| points[&key].world),
                    QualityMetric::ScaledJacobian,
                )
                .unwrap_or(0.0)
            })
            .fold((0, 1.0_f64), |(below, worst), quality| {
                (
                    below + usize::from(quality < QUALITY_FLOOR),
                    worst.min(quality),
                )
            });
        let replace = best.as_ref().is_none_or(|(best_objective, best_tets)| {
            improves(*best_objective, objective)
                || (objective == *best_objective && tets < *best_tets)
        });
        if replace {
            best = Some((objective, tets));
        }
    }
    best
}

fn best_fan_template(
    vertices: &BTreeSet<PointKey>,
    faces: &[Vec<PointKey>],
    points: &BTreeMap<PointKey, Point>,
    local_size: f64,
) -> Option<FanTemplate> {
    let triangles = faces
        .iter()
        .flat_map(|face| triangulate_polygon(face, points))
        .collect::<Vec<_>>();
    if triangles.len() < 4 {
        return None;
    }
    let worlds = vertices
        .iter()
        .map(|key| points[key].world)
        .collect::<Vec<_>>();
    let mean = mean_points(&worlds);
    let mut targets = worlds;
    targets.extend(
        faces
            .iter()
            .map(|face| mean_points(&face.iter().map(|key| points[key].world).collect::<Vec<_>>())),
    );
    let volume_center = polyhedron_centroid(mean, &triangles, points);
    let mut best = [mean, volume_center]
        .into_iter()
        .filter_map(|center| {
            fan_objective(center, &triangles, points, local_size)
                .map(|objective| (objective, center))
        })
        .max_by(compare_scored_point)?;

    for step in [0.5, 0.25, 0.125, 0.0625, 0.03125] {
        let origin = best.1;
        let trial = targets
            .iter()
            .map(|target| {
                std::array::from_fn(|axis| origin[axis] + step * (target[axis] - origin[axis]))
            })
            .filter_map(|center| {
                fan_objective(center, &triangles, points, local_size)
                    .map(|objective| (objective, center))
            })
            .max_by(compare_scored_point);
        if trial
            .as_ref()
            .is_some_and(|(objective, _)| improves(best.0, *objective))
        {
            best = trial.expect("checked improving fan center");
        }
    }
    Some((best.0, best.1, triangles))
}

fn fan_objective(
    center: [f64; 3],
    triangles: &[[PointKey; 3]],
    points: &BTreeMap<PointKey, Point>,
    local_size: f64,
) -> Option<(usize, f64)> {
    triangles
        .iter()
        .try_fold((0, 1.0_f64), |(below, worst), triangle| {
            let mut positions = [
                center,
                points[&triangle[0]].world,
                points[&triangle[1]].world,
                points[&triangle[2]].world,
            ];
            let volume = signed_volume_points(positions);
            if volume.abs() <= volume_tolerance(local_size) {
                return None;
            }
            if volume < 0.0 {
                positions.swap(2, 3);
            }
            quality_score("tet4", &positions, QualityMetric::ScaledJacobian)
                .filter(|score| *score > VALID_QUALITY)
                .map(|score| (below + usize::from(score < QUALITY_FLOOR), worst.min(score)))
        })
}

fn polyhedron_centroid(
    center: [f64; 3],
    triangles: &[[PointKey; 3]],
    points: &BTreeMap<PointKey, Point>,
) -> [f64; 3] {
    let mut weighted = [0.0; 3];
    let mut total = 0.0;
    for triangle in triangles {
        let tet = [
            center,
            points[&triangle[0]].world,
            points[&triangle[1]].world,
            points[&triangle[2]].world,
        ];
        let weight = signed_volume_points(tet).abs();
        let tet_center = centroid(&tet);
        for axis in 0..3 {
            weighted[axis] += weight * tet_center[axis];
        }
        total += weight;
    }
    if total <= f64::EPSILON {
        center
    } else {
        weighted.map(|value| value / total)
    }
}

fn compare_scored_point(
    left: &((usize, f64), [f64; 3]),
    right: &((usize, f64), [f64; 3]),
) -> Ordering {
    right
        .0
         .0
        .cmp(&left.0 .0)
        .then_with(|| left.0 .1.total_cmp(&right.0 .1))
        .then_with(|| compare_point(left.1, right.1))
}

fn compare_point(left: [f64; 3], right: [f64; 3]) -> Ordering {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| left.total_cmp(&right))
        .find(|ordering| *ordering != Ordering::Equal)
        .unwrap_or(Ordering::Equal)
}

fn mean_points(points: &[[f64; 3]]) -> [f64; 3] {
    let mut result = [0.0; 3];
    for point in points {
        for axis in 0..3 {
            result[axis] += point[axis] / points.len() as f64;
        }
    }
    result
}

fn dedup_polygon(points: Vec<PointKey>) -> Vec<PointKey> {
    let mut result = Vec::with_capacity(points.len());
    for point in points {
        if result.last() != Some(&point) {
            result.push(point);
        }
    }
    if result.len() > 1 && result.first() == result.last() {
        result.pop();
    }
    result
}

fn assess(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    candidate: &Candidate,
) -> MeshResult<Assessment> {
    let mut assessment = Assessment {
        boundary: Vec::new(),
        boundary_vertices: BTreeSet::new(),
        poor: BinaryHeap::new(),
        refine: candidate.construction_failures.clone(),
        reason: (!candidate.construction_failures.is_empty())
            .then(|| "cut polyhedron produced a degenerate tetrahedron".into()),
        location: None,
        worst_quality: 1.0,
    };
    let mut incidence = BTreeMap::<[PointKey; 3], Vec<(usize, [PointKey; 3])>>::new();
    for (index, cell) in candidate.cells.iter().enumerate() {
        if index % 512 == 0 {
            context.check()?;
        }
        let positions = cell.points.map(|key| candidate.points[&key].world);
        let size = maximum_edge_3d(positions);
        let volume = signed_volume(cell.points, &candidate.points);
        let quality =
            quality_score("tet4", &positions, QualityMetric::ScaledJacobian).unwrap_or(0.0);
        assessment.worst_quality = assessment.worst_quality.min(quality);
        if volume <= volume_tolerance(size) || quality <= VALID_QUALITY {
            record_refinement(
                &mut assessment,
                cell.leaf,
                "tetrahedron is inverted, degenerate, or below the Scaled Jacobian validity floor",
                centroid(&positions),
            );
        } else if !cell.certified_interior
            && !cell_is_contained_3d(domain, &positions, chord_tolerance(domain, size))
        {
            record_refinement(
                &mut assessment,
                cell.leaf,
                "tetrahedron containment samples leave the negative SDF domain",
                centroid(&positions),
            );
        } else if quality < QUALITY_FLOOR {
            assessment.poor.push(PoorCell { quality, index });
            if assessment.location.is_none() {
                assessment.location = Some(centroid(&positions));
            }
        }
        for face in TET_FACES {
            let oriented = [
                cell.points[face[0]],
                cell.points[face[1]],
                cell.points[face[2]],
            ];
            incidence
                .entry(sorted3(oriented))
                .or_default()
                .push((index, oriented));
        }
    }
    for entries in incidence.values() {
        if entries.len() > 2 {
            for (cell, _) in entries {
                record_refinement(
                    &mut assessment,
                    candidate.cells[*cell].leaf,
                    "non-manifold tetrahedron face incidence",
                    cell_centroid(candidate, *cell),
                );
            }
        } else if let [(cell, oriented)] = entries.as_slice() {
            assessment.boundary.push(BoundaryFace {
                points: *oriented,
                cell: *cell,
            });
            assessment.boundary_vertices.extend(oriented);
        }
    }

    let boundary = assessment.boundary.clone();
    let mut boundary_edges = BTreeMap::<(PointKey, PointKey), Vec<usize>>::new();
    for (face_index, face) in boundary.iter().enumerate() {
        if face_index % 128 == 0 {
            context.check()?;
        }
        for edge in 0..3 {
            boundary_edges
                .entry(ordered_pair(face.points[edge], face.points[(edge + 1) % 3]))
                .or_default()
                .push(face_index);
        }
        let positions = face.points.map(|key| candidate.points[&key].world);
        let center = centroid(&positions);
        let size = maximum_edge_face(positions);
        let tolerance = chord_tolerance(domain, size);
        let sdf = domain.domain_sdf(&[Vec3::from_array(center)])[0];
        if !sdf.is_finite() || sdf.abs() > tolerance {
            record_refinement(
                &mut assessment,
                candidate.cells[face.cell].leaf,
                "exposed tetrahedron face is not owned by the SDF boundary",
                center,
            );
        } else {
            let class = domain
                .classify_boundary(&[Vec3::from_array(center)], BoundaryBand::Custom(tolerance))
                .map_err(|error| MeshError::InvalidInput(error.to_string()))?;
            if !class[0].on_boundary {
                record_refinement(
                    &mut assessment,
                    candidate.cells[face.cell].leaf,
                    "SDF boundary ownership classification failed",
                    center,
                );
            }
        }
    }
    for faces in boundary_edges.values() {
        if faces.len() != 2 {
            for &face_index in faces {
                let face = &boundary[face_index];
                record_refinement(
                    &mut assessment,
                    candidate.cells[face.cell].leaf,
                    "boundary triangle edges do not have manifold incidence two",
                    centroid(&face.points.map(|key| candidate.points[&key].world)),
                );
            }
        }
    }
    assessment
        .boundary
        .sort_by_key(|face| (sorted3(face.points), face.cell, face.points));
    Ok(assessment)
}

fn record_refinement(assessment: &mut Assessment, leaf: Leaf, reason: &str, location: [f64; 3]) {
    assessment.refine.insert(leaf);
    if assessment.reason.is_none() {
        assessment.reason = Some(reason.into());
        assessment.location = Some(location);
    }
}

fn repair(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &mut Assessment,
) -> MeshResult<()> {
    loop {
        context.check()?;
        let topology = RepairTopology::new(context, candidate)?;
        let before = (assessment.poor.len(), assessment.worst_quality);
        let mut smoothed = 0usize;
        let limit = assessment.poor.len();
        for attempt in 0..limit {
            if attempt % 8 == 0 {
                context.check()?;
            }
            let Some(entry) = assessment.poor.pop() else {
                break;
            };
            if entry.index < candidate.cells.len()
                && try_smooth(domain, candidate, &topology, entry.index)
            {
                smoothed += 1;
            }
        }
        *assessment = assess(domain, context, candidate)?;
        let smoothing_progress =
            assessment.poor.len() < before.0 || assessment.worst_quality > before.1 + 1.0e-12;
        if assessment.poor.is_empty() || !assessment.refine.is_empty() {
            break;
        }
        if smoothed > 0 && smoothing_progress {
            continue;
        }

        let topology = RepairTopology::new(context, candidate)?;
        let mut replaced = 0usize;
        let mut touched = BTreeSet::new();
        let limit = assessment.poor.len();
        for attempt in 0..limit {
            if attempt % 8 == 0 {
                context.check()?;
            }
            let Some(entry) = assessment.poor.pop() else {
                break;
            };
            if entry.index >= candidate.cells.len() {
                continue;
            }
            let neighborhood = candidate.cells[entry.index]
                .points
                .iter()
                .flat_map(|point| {
                    topology
                        .vertex_cells
                        .get(point)
                        .into_iter()
                        .flatten()
                        .copied()
                })
                .collect::<BTreeSet<_>>();
            if neighborhood.iter().any(|cell| touched.contains(cell)) {
                continue;
            }
            if try_template_flip(domain, candidate, &topology, entry.index)
                || try_steiner_insertion(domain, candidate, entry.index)
            {
                replaced += 1;
                touched.extend(neighborhood);
            }
        }
        *assessment = assess(domain, context, candidate)?;
        if replaced == 0 || assessment.poor.is_empty() || !assessment.refine.is_empty() {
            break;
        }
    }
    for poor in &assessment.poor {
        assessment.refine.insert(candidate.cells[poor.index].leaf);
    }
    Ok(())
}

struct RepairTopology {
    faces: BTreeMap<[PointKey; 3], Vec<usize>>,
    edges: BTreeSet<(PointKey, PointKey)>,
    vertex_cells: BTreeMap<PointKey, Vec<usize>>,
}

impl RepairTopology {
    fn new(context: &MeshingContext<'_>, candidate: &Candidate) -> MeshResult<Self> {
        let mut result = Self {
            faces: BTreeMap::new(),
            edges: BTreeSet::new(),
            vertex_cells: BTreeMap::new(),
        };
        for (index, cell) in candidate.cells.iter().enumerate() {
            if index % 512 == 0 {
                context.check()?;
            }
            for point in cell.points {
                result.vertex_cells.entry(point).or_default().push(index);
            }
            for (a, b) in TET_EDGES {
                result
                    .edges
                    .insert(ordered_pair(cell.points[a], cell.points[b]));
            }
            for face in TET_FACES {
                result
                    .faces
                    .entry(sorted3([
                        cell.points[face[0]],
                        cell.points[face[1]],
                        cell.points[face[2]],
                    ]))
                    .or_default()
                    .push(index);
            }
        }
        Ok(result)
    }
}

fn try_template_flip(
    domain: &MeshableDomain,
    candidate: &mut Candidate,
    topology: &RepairTopology,
    cell_index: usize,
) -> bool {
    let cell = candidate.cells[cell_index].clone();
    for face in TET_FACES {
        let shared = sorted3([
            cell.points[face[0]],
            cell.points[face[1]],
            cell.points[face[2]],
        ]);
        let Some(pair) = topology.faces.get(&shared).filter(|pair| pair.len() == 2) else {
            continue;
        };
        let other_index = if pair[0] == cell_index {
            pair[1]
        } else if pair[1] == cell_index {
            pair[0]
        } else {
            continue;
        };
        if candidate.cells[other_index].leaf != cell.leaf {
            continue;
        }
        let Some(d) = cell
            .points
            .into_iter()
            .find(|point| !shared.contains(point))
        else {
            continue;
        };
        let Some(e) = candidate.cells[other_index]
            .points
            .into_iter()
            .find(|point| !shared.contains(point))
        else {
            continue;
        };
        if d == e || topology.edges.contains(&ordered_pair(d, e)) {
            continue;
        }
        let before = objective(candidate, &[cell_index, other_index]);
        let old = [
            candidate.cells[cell_index].clone(),
            candidate.cells[other_index].clone(),
        ];
        let mut replacements = [
            [d, e, shared[0], shared[1]],
            [d, e, shared[1], shared[2]],
            [d, e, shared[2], shared[0]],
        ];
        for tet in &mut replacements {
            orient_tet(tet, &candidate.points);
        }
        candidate.cells[cell_index].points = replacements[0];
        candidate.cells[cell_index].certified_interior = false;
        candidate.cells[other_index].points = replacements[1];
        candidate.cells[other_index].certified_interior = false;
        candidate.cells.push(Cell {
            points: replacements[2],
            leaf: cell.leaf,
            certified_interior: false,
        });
        let affected = [cell_index, other_index, candidate.cells.len() - 1];
        if affected_valid(domain, candidate, &affected)
            && improves(before, objective(candidate, &affected))
        {
            return true;
        }
        candidate.cells.pop();
        candidate.cells[cell_index] = old[0].clone();
        candidate.cells[other_index] = old[1].clone();
    }
    false
}

fn try_smooth(
    domain: &MeshableDomain,
    candidate: &mut Candidate,
    topology: &RepairTopology,
    cell_index: usize,
) -> bool {
    for key in candidate.cells[cell_index].points {
        let old = candidate.points[&key];
        let incident = topology.vertex_cells.get(&key).cloned().unwrap_or_default();
        let mut neighbors = BTreeSet::new();
        for &index in &incident {
            neighbors.extend(candidate.cells[index].points.into_iter().filter(|point| {
                *point != key && (!old.boundary || candidate.points[point].boundary)
            }));
        }
        if neighbors.len() < if old.boundary { 2 } else { 4 } {
            continue;
        }
        let target = neighbors
            .iter()
            .map(|neighbor| candidate.points[neighbor].world)
            .fold([0.0; 3], |sum, value| {
                [sum[0] + value[0], sum[1] + value[1], sum[2] + value[2]]
            })
            .map(|value| value / neighbors.len() as f64);
        let before = objective(candidate, &incident);
        let anchor = incident
            .iter()
            .map(|index| cell_centroid(candidate, *index))
            .fold([0.0; 3], |sum, value| {
                [sum[0] + value[0], sum[1] + value[1], sum[2] + value[2]]
            })
            .map(|value| value / incident.len() as f64);
        let mut best = None::<((usize, f64), Point)>;
        for fraction in [1.0, 0.75, 0.5, 0.25] {
            let trial = std::array::from_fn(|axis| {
                old.world[axis] + fraction * (target[axis] - old.world[axis])
            });
            let world = if old.boundary {
                let interior = (1..=8)
                    .map(|step| step as f64 / 8.0)
                    .map(|weight| {
                        std::array::from_fn(|axis| {
                            trial[axis] + weight * (anchor[axis] - trial[axis])
                        })
                    })
                    .find(|point| domain.domain_sdf(&[Vec3::from_array(*point)])[0] < 0.0);
                let Some(interior) = interior else {
                    continue;
                };
                let Ok(projected) = domain.project_to_boundary(&[Vec3::from_array(interior)])
                else {
                    continue;
                };
                let projection = projected[0];
                if !projection.converged {
                    continue;
                }
                projection.point.to_array()
            } else {
                if domain.domain_sdf(&[Vec3::from_array(trial)])[0] >= 0.0 {
                    continue;
                }
                trial
            };
            candidate.points.insert(
                key,
                Point {
                    world,
                    boundary: old.boundary,
                },
            );
            if affected_valid(domain, candidate, &incident) {
                let after = objective(candidate, &incident);
                if improves(before, after)
                    && best
                        .as_ref()
                        .is_none_or(|(best_objective, _)| improves(*best_objective, after))
                {
                    best = Some((
                        after,
                        Point {
                            world,
                            boundary: old.boundary,
                        },
                    ));
                }
            }
            candidate.points.insert(key, old);
        }
        if let Some((_, point)) = best {
            candidate.points.insert(key, point);
            return true;
        }
    }
    false
}

fn try_steiner_insertion(
    domain: &MeshableDomain,
    candidate: &mut Candidate,
    cell_index: usize,
) -> bool {
    let cell = candidate.cells[cell_index].clone();
    let positions = cell.points.map(|key| candidate.points[&key].world);
    let center = incenter_tet(positions);
    if domain.domain_sdf(&[Vec3::from_array(center)])[0] >= 0.0 {
        return false;
    }
    let before = objective(candidate, &[cell_index]);
    let key = PointKey::Steiner(candidate.next_steiner);
    candidate.next_steiner += 1;
    candidate.points.insert(
        key,
        Point {
            world: center,
            boundary: false,
        },
    );
    let replacements = TET_FACES.map(|face| {
        let mut points = [
            key,
            cell.points[face[0]],
            cell.points[face[1]],
            cell.points[face[2]],
        ];
        orient_tet(&mut points, &candidate.points);
        Cell {
            points,
            leaf: cell.leaf,
            certified_interior: false,
        }
    });
    candidate.cells[cell_index] = replacements[0].clone();
    candidate.cells.extend(replacements[1..].iter().cloned());
    let affected = [
        cell_index,
        candidate.cells.len() - 3,
        candidate.cells.len() - 2,
        candidate.cells.len() - 1,
    ];
    if affected_valid(domain, candidate, &affected)
        && improves(before, objective(candidate, &affected))
    {
        return true;
    }
    candidate.cells.truncate(candidate.cells.len() - 3);
    candidate.cells[cell_index] = cell;
    candidate.points.remove(&key);
    false
}

fn affected_valid(domain: &MeshableDomain, candidate: &Candidate, cells: &[usize]) -> bool {
    cells.iter().all(|&index| {
        let positions = candidate.cells[index]
            .points
            .map(|key| candidate.points[&key].world);
        let size = maximum_edge_3d(positions);
        signed_volume(candidate.cells[index].points, &candidate.points) > volume_tolerance(size)
            && quality_score("tet4", &positions, QualityMetric::ScaledJacobian)
                .is_some_and(|quality| quality > VALID_QUALITY)
            && cell_is_contained_3d(domain, &positions, chord_tolerance(domain, size))
    })
}

fn objective(candidate: &Candidate, cells: &[usize]) -> (usize, f64) {
    cells
        .iter()
        .map(|&index| {
            let positions = candidate.cells[index]
                .points
                .map(|key| candidate.points[&key].world);
            quality_score("tet4", &positions, QualityMetric::ScaledJacobian).unwrap_or(0.0)
        })
        .fold((0, 1.0_f64), |(below, worst), quality| {
            (
                below + usize::from(quality < QUALITY_FLOOR),
                worst.min(quality),
            )
        })
}

fn improves(before: (usize, f64), after: (usize, f64)) -> bool {
    after.0 < before.0 || (after.0 == before.0 && after.1 > before.1 + 1.0e-12)
}

fn cell_is_contained_3d(domain: &MeshableDomain, points: &[[f64; 3]; 4], tolerance: f64) -> bool {
    let mut samples = Vec::with_capacity(11);
    samples.push(Vec3::from_array(centroid(points)));
    for (a, b) in TET_EDGES {
        samples.push(Vec3::from_array(midpoint3(points[a], points[b])));
    }
    for face in TET_FACES {
        samples.push(Vec3::from_array(centroid(&[
            points[face[0]],
            points[face[1]],
            points[face[2]],
        ])));
    }
    domain
        .domain_sdf(&samples)
        .into_iter()
        .all(|value| value.is_finite() && value <= tolerance)
}

fn emit(
    context: &MeshingContext<'_>,
    domain: &MeshableDomain,
    candidate: &Candidate,
    assessment: &Assessment,
    sink: &mut dyn MeshSink,
    statistics: &mut MeshingStatistics,
) -> MeshResult<()> {
    if candidate.cells.is_empty() {
        return Err(MeshError::InvalidInput(format!(
            "domain {:?} produced no valid 3D elements",
            domain.name
        )));
    }
    if candidate.cells.len() as u64 > context.limits.max_cells {
        return Err(MeshError::LimitExceeded(format!(
            "mesh exceeds the configured {} cell limit",
            context.limits.max_cells
        )));
    }
    let cells_per_chunk =
        (context.limits.target_chunk_bytes / ESTIMATED_CHUNK_BYTES_PER_CELL).max(1);
    let chunk_count = candidate.cells.len().div_ceil(cells_per_chunk);
    let chunk_ids = (0..chunk_count)
        .map(|_| sink.allocate_chunk_id())
        .collect::<MeshResult<Vec<_>>>()?;
    let cell_chunk = (0..candidate.cells.len())
        .map(|index| index / cells_per_chunk)
        .collect::<Vec<_>>();
    let mut uses = BTreeMap::<PointKey, BTreeSet<usize>>::new();
    for (cell_index, cell) in candidate.cells.iter().enumerate() {
        for point in cell.points {
            uses.entry(point)
                .or_default()
                .insert(cell_chunk[cell_index]);
        }
    }
    let mut ordinals = vec![1u32; chunk_count];
    let mut ids = BTreeMap::<PointKey, MeshId>::new();
    for (&key, chunks) in &uses {
        let owner = *chunks.first().expect("used point has a chunk");
        let ordinal = ordinals[owner];
        ordinals[owner] = ordinal
            .checked_add(1)
            .ok_or_else(|| MeshError::LimitExceeded("3D point ID space exhausted".into()))?;
        ids.insert(
            key,
            MeshId::from_raw((u64::from(chunk_ids[owner]) << 32) | u64::from(ordinal)),
        );
    }
    let catalog = context.catalog.domain(&domain.name)?;
    for (chunk_index, &chunk_id) in chunk_ids.iter().enumerate() {
        context.check()?;
        let start = chunk_index * cells_per_chunk;
        let end = (start + cells_per_chunk).min(candidate.cells.len());
        let used = candidate.cells[start..end]
            .iter()
            .flat_map(|cell| cell.points)
            .collect::<BTreeSet<_>>();
        let bounds = Bounds3::from_points(used.iter().map(|key| candidate.points[key].world))
            .expanded(root_tolerance(domain, context.element_min_size));
        let mut builder = MeshChunkBuilder::new(chunk_id, bounds)?;
        for key in &used {
            let point = candidate.points[key];
            builder.point_copy(
                ids[key],
                point.world,
                if assessment.boundary_vertices.contains(key) {
                    "boundary"
                } else {
                    "interior"
                },
                Vec::new(),
            )?;
        }
        for cell in &candidate.cells[start..end] {
            builder.tet4(
                cell.points.map(|key| ids[&key]),
                catalog.zone,
                catalog.source,
            )?;
        }
        for face in assessment
            .boundary
            .iter()
            .filter(|face| (start..end).contains(&face.cell))
        {
            let positions = face.points.map(|key| candidate.points[&key].world);
            let center = centroid(&positions);
            let class = domain
                .classify_boundary(
                    &[Vec3::from_array(center)],
                    BoundaryBand::UnprojectedSamples,
                )
                .map_err(|error| MeshError::InvalidInput(error.to_string()))?
                .into_iter()
                .next()
                .expect("one boundary point");
            let tag = class
                .region_name
                .as_deref()
                .and_then(|region| context.catalog.boundary_tag(&domain.name, region))
                .unwrap_or(catalog.wall_tag);
            builder.boundary_face("tri3", &face.points.map(|key| ids[&key]), vec![tag])?;
        }
        let chunk = builder.build(3)?;
        let active = chunk.decoded_bytes() as u64;
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
            phase_total: chunk_count as u64,
            completed_chunks: statistics.chunks,
            cells_committed: statistics.cells,
            active_bytes: active,
        });
    }
    Ok(())
}

fn signed_volume(points: [PointKey; 4], map: &BTreeMap<PointKey, Point>) -> f64 {
    signed_volume_points(points.map(|key| map[&key].world))
}

fn signed_volume_points(points: [[f64; 3]; 4]) -> f64 {
    let [a, b, c, d] = points.map(Vec3::from_array);
    (b - a).dot((c - a).cross(d - a)) / 6.0
}

fn signed_volume_samples(samples: [Sample; 4]) -> f64 {
    let [a, b, c, d] = samples.map(|sample| Vec3::from_array(sample.world));
    (b - a).dot((c - a).cross(d - a)) / 6.0
}

fn orient_tet(points: &mut [PointKey; 4], map: &BTreeMap<PointKey, Point>) {
    if signed_volume(*points, map) < 0.0 {
        points.swap(2, 3);
    }
}

fn face_pair_score(triangles: [[[f64; 3]; 3]; 2]) -> f64 {
    triangles
        .into_iter()
        .map(|triangle| {
            quality_score("tri3", &triangle, QualityMetric::ScaledJacobian).unwrap_or(0.0)
        })
        .fold(1.0, f64::min)
}

fn triangle_pair_quality(triangles: [[PointKey; 3]; 2], points: &BTreeMap<PointKey, Point>) -> f64 {
    triangles
        .into_iter()
        .map(|triangle| {
            quality_score(
                "tri3",
                &triangle.map(|key| points[&key].world),
                QualityMetric::ScaledJacobian,
            )
            .unwrap_or(0.0)
        })
        .fold(1.0, f64::min)
}

fn leaf_size_3d(grid: Grid, leaf: Leaf) -> f64 {
    let corners = grid.corners(leaf);
    let origin = grid.world(corners[0]);
    [corners[1], corners[3], corners[4]]
        .into_iter()
        .map(|corner| distance3(origin, grid.world(corner)))
        .fold(0.0, f64::max)
}

fn maximum_edge_3d(points: [[f64; 3]; 4]) -> f64 {
    TET_EDGES
        .into_iter()
        .map(|(a, b)| distance3(points[a], points[b]))
        .fold(0.0, f64::max)
}

fn maximum_edge_face(points: [[f64; 3]; 3]) -> f64 {
    distance3(points[0], points[1])
        .max(distance3(points[1], points[2]))
        .max(distance3(points[2], points[0]))
}

fn root_tolerance(domain: &MeshableDomain, local_size: f64) -> f64 {
    (domain.bounds.diagonal() * 1.0e-12)
        .max(local_size * 1.0e-6)
        .max(f64::EPSILON * domain.bounds.diagonal() * 64.0)
}

fn chord_tolerance(domain: &MeshableDomain, local_size: f64) -> f64 {
    root_tolerance(domain, local_size).max(local_size * 0.35)
}

fn volume_tolerance(local_size: f64) -> f64 {
    local_size.powi(3) * 1.0e-12
}

fn minimum_size_error(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    reason: &str,
    location: Option<[f64; 3]>,
    quality: f64,
) -> MeshError {
    let location = location.unwrap_or_else(|| domain.bounds.center().to_array());
    MeshError::InvalidInput(format!(
        "domain {:?} could not produce a valid adaptive 3D mesh at the {:.6e} element-size floor near ({:.6}, {:.6}, {:.6}): {reason}; worst Scaled Jacobian={quality:.6e}, required > {VALID_QUALITY:.1e} and generation target {QUALITY_FLOOR:.2}",
        domain.name,
        context.element_min_size,
        location[0],
        location[1],
        location[2],
    ))
}

fn incenter_tet(points: [[f64; 3]; 4]) -> [f64; 3] {
    let weights = TET_FACES.map(|face| {
        let a = Vec3::from_array(points[face[0]]);
        let b = Vec3::from_array(points[face[1]]);
        let c = Vec3::from_array(points[face[2]]);
        (b - a).cross(c - a).length() * 0.5
    });
    let sum = weights.iter().sum::<f64>();
    if sum <= f64::EPSILON {
        return centroid(&points);
    }
    std::array::from_fn(|axis| {
        (0..4)
            .map(|index| points[index][axis] * weights[index])
            .sum::<f64>()
            / sum
    })
}

fn cell_centroid(candidate: &Candidate, cell: usize) -> [f64; 3] {
    centroid(
        &candidate.cells[cell]
            .points
            .map(|key| candidate.points[&key].world),
    )
}

fn midpoint(a: Lattice, b: Lattice) -> Lattice {
    lattice((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2)
}

const fn lattice(x: u64, y: u64, z: u64) -> Lattice {
    Lattice { x, y, z }
}

fn ordered_pair<T: Ord + Copy>(a: T, b: T) -> (T, T) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn sorted3<T: Ord + Copy>(mut points: [T; 3]) -> [T; 3] {
    points.sort_unstable();
    points
}

fn distance3(a: [f64; 3], b: [f64; 3]) -> f64 {
    (Vec3::from_array(a) - Vec3::from_array(b)).length()
}

fn midpoint3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ]
}

fn centroid<const N: usize>(points: &[[f64; 3]; N]) -> [f64; 3] {
    let mut result = [0.0; 3];
    for point in points {
        for axis in 0..3 {
            result[axis] += point[axis] / N as f64;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marching_tetrahedra_cover_all_sixteen_sign_masks() {
        let expected_crossings = [0, 3, 3, 4, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 3, 0];
        for mask in 0u8..16 {
            let edges = marching_edges(mask);
            assert_eq!(
                edges.len(),
                expected_crossings[mask as usize],
                "mask {mask:04b}"
            );
            assert!(edges
                .iter()
                .all(|&(a, b)| { ((mask >> a) & 1) != ((mask >> b) & 1) }));
        }
    }

    #[test]
    fn zero_values_are_retained_by_the_marching_sign_rule() {
        let samples = [
            Sample {
                key: lattice(0, 0, 0),
                world: [0.0; 3],
                sdf: 0.0,
            },
            Sample {
                key: lattice(1, 0, 0),
                world: [1.0, 0.0, 0.0],
                sdf: 1.0,
            },
            Sample {
                key: lattice(0, 1, 0),
                world: [0.0, 1.0, 0.0],
                sdf: 1.0,
            },
            Sample {
                key: lattice(0, 0, 1),
                world: [0.0, 0.0, 1.0],
                sdf: 1.0,
            },
        ];
        let mask = samples
            .iter()
            .enumerate()
            .fold(0u8, |mask, (index, sample)| {
                mask | (u8::from(sample.sdf <= 0.0) << index)
            });
        assert_eq!(mask, 1);
        assert_eq!(marching_edges(mask).len(), 3);
    }

    #[test]
    fn poor_cell_order_is_deterministic() {
        let mut queue = BinaryHeap::new();
        queue.push(PoorCell {
            quality: 0.04,
            index: 3,
        });
        queue.push(PoorCell {
            quality: 0.01,
            index: 8,
        });
        queue.push(PoorCell {
            quality: 0.01,
            index: 2,
        });
        assert_eq!(queue.pop().unwrap().index, 2);
        assert_eq!(queue.pop().unwrap().index, 8);
        assert_eq!(queue.pop().unwrap().index, 3);
    }

    #[test]
    fn face_subdivision_is_conforming_and_complete() {
        let quad = [
            lattice(0, 0, 0),
            lattice(2, 0, 0),
            lattice(2, 2, 0),
            lattice(0, 2, 0),
        ];
        let children = subdivide_quad(quad);
        assert_eq!(children.len(), 4);
        let center = lattice(1, 1, 0);
        assert!(children.iter().all(|child| child.contains(&center)));
    }
}
