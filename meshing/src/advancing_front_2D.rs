use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use caso_kernel::meshing::{BoundaryBand, MeshableDomain, MeshableDomainSpace};
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
const SNAP_RATIO: f64 = 0.06;
const ESTIMATED_CHUNK_BYTES_PER_CELL: usize = 2_048;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Leaf {
    level: u8,
    x: u32,
    y: u32,
}

impl Leaf {
    fn children(self) -> [Self; 4] {
        [
            Self {
                level: self.level + 1,
                x: self.x * 2,
                y: self.y * 2,
            },
            Self {
                level: self.level + 1,
                x: self.x * 2 + 1,
                y: self.y * 2,
            },
            Self {
                level: self.level + 1,
                x: self.x * 2 + 1,
                y: self.y * 2 + 1,
            },
            Self {
                level: self.level + 1,
                x: self.x * 2,
                y: self.y * 2 + 1,
            },
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Lattice {
    x: u64,
    y: u64,
}

#[derive(Debug, Clone, Copy)]
struct Grid {
    bounds: [f64; 4],
    base: [u32; 2],
    max_depth: u8,
}

impl Grid {
    fn new(
        space: &MeshableDomainSpace,
        context: &MeshingContext<'_>,
        domain_name: &str,
    ) -> MeshResult<Self> {
        let bounds = space.bounds();
        let du = space.point(bounds[1], bounds[2]) - space.point(bounds[0], bounds[2]);
        let dv = space.point(bounds[0], bounds[3]) - space.point(bounds[0], bounds[2]);
        let lengths = [du.length(), dv.length()];
        if lengths
            .into_iter()
            .any(|length| !length.is_finite() || length <= 0.0)
        {
            return Err(MeshError::InvalidInput(format!(
                "domain {domain_name:?} has invalid local 2D bounds"
            )));
        }
        let base = lengths
            .map(|length| ((length / context.element_max_size).ceil() as u32).clamp(1, 1 << 20));
        if u64::from(base[0]).saturating_mul(u64::from(base[1]))
            > context.limits.max_cells.saturating_mul(4)
        {
            return Err(MeshError::LimitExceeded(
                "adaptive 2D base grid exceeds the configured cell limit".into(),
            ));
        }
        let base_size = (lengths[0] / f64::from(base[0])).max(lengths[1] / f64::from(base[1]));
        let mut max_depth = 0;
        let mut size = base_size;
        while max_depth < 30 && size * 0.5 >= context.element_min_size * (1.0 - 1.0e-12) {
            max_depth += 1;
            size *= 0.5;
        }
        Ok(Self {
            bounds,
            base,
            max_depth,
        })
    }

    fn fine_scale(self, leaf: Leaf) -> u64 {
        1u64 << (self.max_depth - leaf.level)
    }

    fn fine_bounds(self, leaf: Leaf) -> [u64; 4] {
        let scale = self.fine_scale(leaf);
        [
            u64::from(leaf.x) * scale,
            u64::from(leaf.x + 1) * scale,
            u64::from(leaf.y) * scale,
            u64::from(leaf.y + 1) * scale,
        ]
    }

    fn corners(self, leaf: Leaf) -> [Lattice; 4] {
        let [x0, x1, y0, y1] = self.fine_bounds(leaf);
        [
            Lattice {
                x: 2 * x0,
                y: 2 * y0,
            },
            Lattice {
                x: 2 * x1,
                y: 2 * y0,
            },
            Lattice {
                x: 2 * x1,
                y: 2 * y1,
            },
            Lattice {
                x: 2 * x0,
                y: 2 * y1,
            },
        ]
    }

    fn center(self, leaf: Leaf) -> Lattice {
        let [x0, x1, y0, y1] = self.fine_bounds(leaf);
        Lattice {
            x: x0 + x1,
            y: y0 + y1,
        }
    }

    fn uv(self, key: Lattice) -> [f64; 2] {
        let scale = (1u64 << self.max_depth) as f64;
        let nx = 2.0 * f64::from(self.base[0]) * scale;
        let ny = 2.0 * f64::from(self.base[1]) * scale;
        [
            (self.bounds[1] - self.bounds[0]).mul_add(key.x as f64 / nx, self.bounds[0]),
            (self.bounds[3] - self.bounds[2]).mul_add(key.y as f64 / ny, self.bounds[2]),
        ]
    }

    fn root_leaves(self) -> Vec<Leaf> {
        let mut leaves = Vec::with_capacity(self.base[0] as usize * self.base[1] as usize);
        for y in 0..self.base[1] {
            for x in 0..self.base[0] {
                leaves.push(Leaf { level: 0, x, y });
            }
        }
        leaves
    }
}

#[derive(Debug, Clone, Copy)]
struct Sample {
    key: Lattice,
    uv: [f64; 2],
    world: [f64; 3],
    sdf: f64,
}

struct Sampler<'a> {
    domain: &'a MeshableDomain,
    space: &'a MeshableDomainSpace,
    grid: Grid,
    cache: BTreeMap<Lattice, Sample>,
}

impl<'a> Sampler<'a> {
    fn new(domain: &'a MeshableDomain, space: &'a MeshableDomainSpace, grid: Grid) -> Self {
        Self {
            domain,
            space,
            grid,
            cache: BTreeMap::new(),
        }
    }

    fn sample(&mut self, key: Lattice) -> MeshResult<Sample> {
        if let Some(sample) = self.cache.get(&key) {
            return Ok(*sample);
        }
        let uv = self.grid.uv(key);
        let point = self.space.point(uv[0], uv[1]);
        let sdf = self.domain.domain_sdf(&[point])[0];
        if !sdf.is_finite() {
            return Err(MeshError::InvalidInput(format!(
                "domain {:?} returned a non-finite SDF value",
                self.domain.name
            )));
        }
        let sample = Sample {
            key,
            uv,
            world: point.to_array(),
            sdf,
        };
        self.cache.insert(key, sample);
        Ok(sample)
    }

    fn leaf_samples(&mut self, leaf: Leaf) -> MeshResult<[Sample; 9]> {
        let corners = self.grid.corners(leaf);
        let center = self.grid.center(leaf);
        let mids = [
            midpoint(corners[0], corners[1]),
            midpoint(corners[1], corners[2]),
            midpoint(corners[2], corners[3]),
            midpoint(corners[3], corners[0]),
        ];
        let keys = [
            corners[0], corners[1], corners[2], corners[3], mids[0], mids[1], mids[2], mids[3],
            center,
        ];
        let mut samples = [self.sample(keys[0])?; 9];
        for (index, key) in keys.into_iter().enumerate().skip(1) {
            samples[index] = self.sample(key)?;
        }
        Ok(samples)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PointKey {
    Lattice(Lattice),
    Crossing(Lattice, Lattice),
    Inserted(u64),
}

#[derive(Debug, Clone, Copy)]
struct Point {
    uv: [f64; 2],
    world: [f64; 3],
    boundary: bool,
}

#[derive(Debug, Clone)]
struct Cell {
    points: [PointKey; 3],
    leaf: Leaf,
}

#[derive(Debug)]
struct Candidate {
    points: BTreeMap<PointKey, Point>,
    cells: Vec<Cell>,
    construction_failures: BTreeSet<Leaf>,
    next_inserted: u64,
}

#[derive(Debug, Clone)]
struct BoundaryEdge {
    points: [PointKey; 2],
    cell: usize,
}

#[derive(Debug)]
struct Assessment {
    boundary: Vec<BoundaryEdge>,
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
        if domain.dimension != 2 {
            return Err(MeshError::UnsupportedDimension {
                domain: domain.name.clone(),
                dimension: domain.dimension,
            });
        }
        let space = domain
            .mesh_space()
            .map_err(|error| MeshError::InvalidInput(error.to_string()))?;
        let grid = Grid::new(&space, context, &domain.name)?;
        let mut sampler = Sampler::new(domain, &space, grid);
        let mut leaves = discover(context, domain, &space, &mut sampler)?;
        balance(context, grid, &mut leaves)?;

        let (candidate, assessment) = loop {
            context.check()?;
            let mut candidate = build_candidate(context, domain, &space, &mut sampler, &leaves)?;
            let mut assessment = assess(domain, &space, context, &candidate)?;
            if assessment.refine.is_empty() && !assessment.poor.is_empty() {
                repair(domain, &space, context, &mut candidate, &mut assessment)?;
            }
            if assessment.refine.is_empty() && assessment.poor.is_empty() {
                break (candidate, assessment);
            }
            let requested = if assessment.refine.is_empty() {
                assessment
                    .poor
                    .iter()
                    .map(|entry| candidate.cells[entry.index].leaf)
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
    space: &MeshableDomainSpace,
    sampler: &mut Sampler<'_>,
) -> MeshResult<Vec<Leaf>> {
    let mut pending = sampler.grid.root_leaves();
    pending.reverse();
    let mut leaves = Vec::new();
    let mut visited = 0usize;
    while let Some(leaf) = pending.pop() {
        if visited.is_multiple_of(256) {
            context.check()?;
        }
        visited += 1;
        let samples = sampler.leaf_samples(leaf)?;
        let center = samples[8];
        let p0 = Vec3::from_array(samples[0].world);
        let p2 = Vec3::from_array(samples[2].world);
        let radius = (p2 - p0).length() * 0.5;
        let size = leaf_size(&samples);
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
        let certified_inside = center.sdf < 0.0 && -center.sdf > radius + tolerance;
        let certified_outside = center.sdf > 0.0 && center.sdf > radius + tolerance;
        let mixed = (negative > 0 && positive > 0) || on_boundary > 0;
        let unresolved_uniform = !mixed && !certified_inside && !certified_outside;
        let wants_size = size > target * 1.35;
        let wants_geometry = mixed && size > target * 1.05;
        let unresolved_curvature =
            mixed && curvature_requires_refinement(domain, &samples, radius, size)?;
        let split = leaf.level < sampler.grid.max_depth
            && (wants_size || wants_geometry || unresolved_uniform || unresolved_curvature);
        if split {
            for child in leaf.children().into_iter().rev() {
                pending.push(child);
            }
        } else if !certified_outside || negative > 0 || on_boundary > 0 {
            leaves.push(leaf);
        }
        if leaves.len().saturating_add(pending.len()) as u64
            > context.limits.max_cells.saturating_mul(4)
        {
            return Err(MeshError::LimitExceeded(
                "adaptive 2D discovery exceeded the configured cell limit".into(),
            ));
        }
        let _ = space;
    }
    leaves.sort_unstable();
    Ok(leaves)
}

fn leaf_size(samples: &[Sample; 9]) -> f64 {
    let horizontal =
        (Vec3::from_array(samples[1].world) - Vec3::from_array(samples[0].world)).length();
    let vertical =
        (Vec3::from_array(samples[3].world) - Vec3::from_array(samples[0].world)).length();
    horizontal.max(vertical)
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
            if leaf_index % 256 == 0 {
                context.check()?;
            }
            for side in 0..4 {
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
    let mut refined = Vec::with_capacity(leaves.len() + split.len() * 3);
    for leaf in leaves.drain(..) {
        if split.contains(&leaf) {
            refined.extend(leaf.children());
        } else {
            refined.push(leaf);
        }
    }
    if refined.len() as u64 > context.limits.max_cells.saturating_mul(4) {
        return Err(MeshError::LimitExceeded(
            "adaptive 2D refinement exceeded the configured cell limit".into(),
        ));
    }
    refined.sort_unstable();
    *leaves = refined;
    Ok(())
}

struct LeafIndex {
    grid: Grid,
    leaves: BTreeMap<(u8, u32, u32), Leaf>,
}

impl LeafIndex {
    fn new(grid: Grid, leaves: &[Leaf]) -> Self {
        Self {
            grid,
            leaves: leaves
                .iter()
                .copied()
                .map(|leaf| ((leaf.level, leaf.x, leaf.y), leaf))
                .collect(),
        }
    }

    fn owner(&self, x: u64, y: u64) -> Option<Leaf> {
        for level in (0..=self.grid.max_depth).rev() {
            let scale = 1u64 << (self.grid.max_depth - level);
            let key = (level, (x / scale) as u32, (y / scale) as u32);
            if let Some(leaf) = self.leaves.get(&key) {
                return Some(*leaf);
            }
        }
        None
    }

    fn neighbors(&self, leaf: Leaf, side: usize) -> [Option<Leaf>; 2] {
        let [x0, x1, y0, y1] = self.grid.fine_bounds(leaf);
        let along = |a: u64, b: u64, upper: bool| {
            if b - a <= 1 {
                a
            } else if upper {
                a + 3 * (b - a) / 4
            } else {
                a + (b - a) / 4
            }
        };
        match side {
            0 if y0 > 0 => [
                self.owner(along(x0, x1, false), y0 - 1),
                self.owner(along(x0, x1, true), y0 - 1),
            ],
            1 if x1 < u64::from(self.grid.base[0]) * (1u64 << self.grid.max_depth) => [
                self.owner(x1, along(y0, y1, false)),
                self.owner(x1, along(y0, y1, true)),
            ],
            2 if y1 < u64::from(self.grid.base[1]) * (1u64 << self.grid.max_depth) => [
                self.owner(along(x0, x1, false), y1),
                self.owner(along(x0, x1, true), y1),
            ],
            3 if x0 > 0 => [
                self.owner(x0 - 1, along(y0, y1, false)),
                self.owner(x0 - 1, along(y0, y1, true)),
            ],
            _ => [None, None],
        }
    }

    fn has_finer_neighbor(&self, leaf: Leaf, side: usize) -> bool {
        self.neighbors(leaf, side)
            .into_iter()
            .flatten()
            .any(|neighbor| neighbor.level > leaf.level)
    }
}

fn build_candidate(
    context: &MeshingContext<'_>,
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    sampler: &mut Sampler<'_>,
    leaves: &[Leaf],
) -> MeshResult<Candidate> {
    let index = LeafIndex::new(sampler.grid, leaves);
    let mut candidate = Candidate {
        points: BTreeMap::new(),
        cells: Vec::new(),
        construction_failures: BTreeSet::new(),
        next_inserted: 1,
    };
    let mut crossings = BTreeMap::new();
    for (leaf_index, &leaf) in leaves.iter().enumerate() {
        if leaf_index % 256 == 0 {
            context.check()?;
        }
        let corners = sampler.grid.corners(leaf);
        let mut ring = vec![corners[0]];
        if index.has_finer_neighbor(leaf, 0) {
            ring.push(midpoint(corners[0], corners[1]));
        }
        ring.push(corners[1]);
        if index.has_finer_neighbor(leaf, 1) {
            ring.push(midpoint(corners[1], corners[2]));
        }
        ring.push(corners[2]);
        if index.has_finer_neighbor(leaf, 2) {
            ring.push(midpoint(corners[2], corners[3]));
        }
        ring.push(corners[3]);
        if index.has_finer_neighbor(leaf, 3) {
            ring.push(midpoint(corners[3], corners[0]));
        }
        let center = sampler.grid.center(leaf);
        for edge in 0..ring.len() {
            let keys = [center, ring[edge], ring[(edge + 1) % ring.len()]];
            let samples = [
                sampler.sample(keys[0])?,
                sampler.sample(keys[1])?,
                sampler.sample(keys[2])?,
            ];
            clip_triangle(
                domain,
                space,
                leaf_size_from_triangle(samples),
                samples,
                &mut candidate,
                &mut crossings,
                leaf,
            )?;
        }
    }
    Ok(candidate)
}

fn leaf_size_from_triangle(samples: [Sample; 3]) -> f64 {
    (Vec3::from_array(samples[1].world) - Vec3::from_array(samples[2].world))
        .length()
        .max((Vec3::from_array(samples[0].world) - Vec3::from_array(samples[1].world)).length())
}

fn clip_triangle(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    local_size: f64,
    samples: [Sample; 3],
    candidate: &mut Candidate,
    crossings: &mut BTreeMap<(Lattice, Lattice), PointKey>,
    leaf: Leaf,
) -> MeshResult<()> {
    for sample in samples {
        if sample.sdf <= 0.0 {
            candidate
                .points
                .entry(PointKey::Lattice(sample.key))
                .or_insert(Point {
                    uv: sample.uv,
                    world: sample.world,
                    boundary: sample.sdf == 0.0,
                });
        }
    }
    let polygon = clipped_polygon(samples, |a, b| {
        crossing(
            domain,
            space,
            local_size,
            a,
            b,
            &mut candidate.points,
            crossings,
        )
    })?;
    let mut polygon = dedup_polygon(polygon);
    if polygon.len() < 3 {
        return Ok(());
    }
    if polygon.len() > 4 {
        candidate.construction_failures.insert(leaf);
        return Ok(());
    }
    let triangles = if polygon.len() == 3 {
        vec![[polygon[0], polygon[1], polygon[2]]]
    } else {
        let first = [
            [polygon[0], polygon[1], polygon[2]],
            [polygon[0], polygon[2], polygon[3]],
        ];
        let second = [
            [polygon[0], polygon[1], polygon[3]],
            [polygon[1], polygon[2], polygon[3]],
        ];
        if pair_quality(first, &candidate.points) >= pair_quality(second, &candidate.points) {
            first.to_vec()
        } else {
            second.to_vec()
        }
    };
    for mut triangle in triangles {
        let area = signed_area(triangle, &candidate.points);
        if area < 0.0 {
            triangle.swap(1, 2);
        }
        if signed_area(triangle, &candidate.points) <= orientation_tolerance(local_size) {
            candidate.construction_failures.insert(leaf);
            continue;
        }
        candidate.cells.push(Cell {
            points: triangle,
            leaf,
        });
    }
    polygon.clear();
    Ok(())
}

fn clipped_polygon(
    samples: [Sample; 3],
    mut edge_crossing: impl FnMut(Sample, Sample) -> MeshResult<PointKey>,
) -> MeshResult<Vec<PointKey>> {
    let mut polygon = Vec::with_capacity(4);
    for edge in 0..3 {
        let a = samples[edge];
        let b = samples[(edge + 1) % 3];
        match (a.sdf <= 0.0, b.sdf <= 0.0) {
            (true, true) => polygon.push(PointKey::Lattice(b.key)),
            (true, false) => polygon.push(edge_crossing(a, b)?),
            (false, true) => {
                polygon.push(edge_crossing(a, b)?);
                polygon.push(PointKey::Lattice(b.key));
            }
            (false, false) => {}
        }
    }
    Ok(polygon)
}

fn crossing(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    local_size: f64,
    a: Sample,
    b: Sample,
    points: &mut BTreeMap<PointKey, Point>,
    crossings: &mut BTreeMap<(Lattice, Lattice), PointKey>,
) -> MeshResult<PointKey> {
    let edge = ordered_lattice(a.key, b.key);
    if let Some(key) = crossings.get(&edge) {
        return Ok(*key);
    }
    if a.sdf == 0.0 {
        let key = PointKey::Lattice(a.key);
        points.insert(
            key,
            Point {
                uv: a.uv,
                world: a.world,
                boundary: true,
            },
        );
        crossings.insert(edge, key);
        return Ok(key);
    }
    if b.sdf == 0.0 {
        let key = PointKey::Lattice(b.key);
        points.insert(
            key,
            Point {
                uv: b.uv,
                world: b.world,
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
        if (Vec3::from_array(inside.world) - Vec3::from_array(outside.world)).length() <= tolerance
        {
            break;
        }
        let t = (ti + to) * 0.5;
        let uv = [
            a.uv[0] + t * (b.uv[0] - a.uv[0]),
            a.uv[1] + t * (b.uv[1] - a.uv[1]),
        ];
        let world = space.point(uv[0], uv[1]);
        let sdf = domain.domain_sdf(&[world])[0];
        if !sdf.is_finite() {
            return Err(MeshError::InvalidInput(format!(
                "domain {:?} returned a non-finite SDF value",
                domain.name
            )));
        }
        let sample = Sample {
            key: a.key,
            uv,
            world: world.to_array(),
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
    let projection = domain
        .project_to_boundary(&[Vec3::from_array(inside.world)])
        .map_err(|error| MeshError::InvalidInput(error.to_string()))?
        .into_iter()
        .next()
        .expect("one projection");
    let world = if projection.converged {
        projection.point
    } else {
        Vec3::from_array(inside.world)
    };
    let t = (ti + to) * 0.5;
    let snapped = if t <= SNAP_RATIO {
        Some(a.key)
    } else if t >= 1.0 - SNAP_RATIO {
        Some(b.key)
    } else {
        None
    };
    let coords = space.coords(world);
    let point = Point {
        uv: [coords[0], coords[1]],
        world: world.to_array(),
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
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &Candidate,
) -> MeshResult<Assessment> {
    let mut assessment = Assessment {
        boundary: Vec::new(),
        boundary_vertices: BTreeSet::new(),
        poor: BinaryHeap::new(),
        refine: candidate.construction_failures.clone(),
        reason: (!candidate.construction_failures.is_empty())
            .then(|| "boundary clipping produced a degenerate triangle".into()),
        location: None,
        worst_quality: 1.0,
    };
    let mut incidence = BTreeMap::<(PointKey, PointKey), Vec<(usize, [PointKey; 2])>>::new();
    for (index, cell) in candidate.cells.iter().enumerate() {
        if index % 512 == 0 {
            context.check()?;
        }
        let positions = cell.points.map(|key| candidate.points[&key].world);
        let size = maximum_edge_2d(positions);
        let area = signed_area(cell.points, &candidate.points);
        let quality =
            quality_score("tri3", &positions, QualityMetric::ScaledJacobian).unwrap_or(0.0);
        assessment.worst_quality = assessment.worst_quality.min(quality);
        if area <= orientation_tolerance(size) || quality <= VALID_QUALITY {
            record_refinement(
                &mut assessment,
                cell.leaf,
                "triangle is inverted, degenerate, or below the Scaled Jacobian validity floor",
                centroid(&positions),
            );
        } else if !cell_is_contained_2d(domain, &positions, chord_tolerance(domain, size)) {
            record_refinement(
                &mut assessment,
                cell.leaf,
                "triangle containment samples leave the negative SDF domain",
                centroid(&positions),
            );
        } else if quality < QUALITY_FLOOR {
            assessment.poor.push(PoorCell { quality, index });
            if assessment.location.is_none() {
                assessment.location = Some(centroid(&positions));
            }
        }
        for edge in 0..3 {
            let oriented = [cell.points[edge], cell.points[(edge + 1) % 3]];
            incidence
                .entry(ordered_pair(oriented[0], oriented[1]))
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
                    "non-manifold triangle edge incidence",
                    cell_centroid(candidate, *cell),
                );
            }
        } else if let [(cell, oriented)] = entries.as_slice() {
            assessment.boundary.push(BoundaryEdge {
                points: *oriented,
                cell: *cell,
            });
            assessment.boundary_vertices.extend(oriented);
        }
    }
    let mut degree = BTreeMap::<PointKey, usize>::new();
    let boundary = assessment.boundary.clone();
    for (edge_index, edge) in boundary.iter().enumerate() {
        if edge_index % 256 == 0 {
            context.check()?;
        }
        *degree.entry(edge.points[0]).or_default() += 1;
        *degree.entry(edge.points[1]).or_default() += 1;
        let a = candidate.points[&edge.points[0]].world;
        let b = candidate.points[&edge.points[1]].world;
        let midpoint = midpoint3(a, b);
        let size = distance3(a, b);
        let sdf = domain.domain_sdf(&[Vec3::from_array(midpoint)])[0];
        let tolerance = chord_tolerance(domain, size);
        if !sdf.is_finite() || sdf.abs() > tolerance {
            record_refinement(
                &mut assessment,
                candidate.cells[edge.cell].leaf,
                "exposed triangle edge is not owned by the SDF boundary",
                midpoint,
            );
        } else {
            let class = domain
                .classify_boundary(
                    &[Vec3::from_array(midpoint)],
                    BoundaryBand::Custom(tolerance),
                )
                .map_err(|error| MeshError::InvalidInput(error.to_string()))?;
            if !class[0].on_boundary {
                record_refinement(
                    &mut assessment,
                    candidate.cells[edge.cell].leaf,
                    "SDF boundary ownership classification failed",
                    midpoint,
                );
            }
        }
    }
    for (point, count) in degree {
        if count != 2 {
            if let Some(edge) = assessment
                .boundary
                .iter()
                .find(|edge| edge.points.contains(&point))
            {
                let leaf = candidate.cells[edge.cell].leaf;
                let location = candidate.points[&point].world;
                record_refinement(
                    &mut assessment,
                    leaf,
                    "boundary vertex does not have manifold degree two",
                    location,
                );
            }
        }
    }
    assessment.boundary.sort_by_key(|edge| {
        (
            ordered_pair(edge.points[0], edge.points[1]),
            edge.cell,
            edge.points,
        )
    });
    let _ = space;
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
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &mut Assessment,
) -> MeshResult<()> {
    let mut exhausted = BTreeSet::new();
    let limit = candidate.cells.len().saturating_mul(8).max(32);
    for attempt in 0..limit {
        if attempt % 128 == 0 {
            context.check()?;
        }
        let Some(entry) = assessment.poor.pop() else {
            return Ok(());
        };
        if entry.index >= candidate.cells.len() || exhausted.contains(&entry.index) {
            continue;
        }
        if try_flip(domain, context, candidate, entry.index)?
            || try_smooth(domain, space, context, candidate, entry.index)?
            || try_boundary_smooth(domain, space, context, candidate, assessment, entry.index)?
            || try_offcenter(domain, space, candidate, entry.index)
            || try_remove_boundary_ear(domain, candidate, assessment, entry.index)
        {
            *assessment = assess(domain, space, context, candidate)?;
            exhausted.clear();
            if !assessment.refine.is_empty() {
                return Ok(());
            }
        } else {
            exhausted.insert(entry.index);
            assessment.refine.insert(candidate.cells[entry.index].leaf);
        }
        if assessment.poor.is_empty() {
            return Ok(());
        }
    }
    for poor in assessment.poor.iter() {
        assessment.refine.insert(candidate.cells[poor.index].leaf);
    }
    Ok(())
}

fn try_flip(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    cell_index: usize,
) -> MeshResult<bool> {
    let mut incidence = BTreeMap::<(PointKey, PointKey), Vec<usize>>::new();
    for (index, cell) in candidate.cells.iter().enumerate() {
        if index % 512 == 0 {
            context.check()?;
        }
        for edge in 0..3 {
            incidence
                .entry(ordered_pair(cell.points[edge], cell.points[(edge + 1) % 3]))
                .or_default()
                .push(index);
        }
    }
    let old = candidate.cells[cell_index].clone();
    for edge in 0..3 {
        let a = old.points[edge];
        let b = old.points[(edge + 1) % 3];
        let signature = ordered_pair(a, b);
        let Some(pair) = incidence.get(&signature).filter(|pair| pair.len() == 2) else {
            continue;
        };
        let other_index = if pair[0] == cell_index {
            pair[1]
        } else if pair[1] == cell_index {
            pair[0]
        } else {
            continue;
        };
        let Some(c) = old
            .points
            .into_iter()
            .find(|point| *point != a && *point != b)
        else {
            continue;
        };
        let Some(d) = candidate.cells[other_index]
            .points
            .into_iter()
            .find(|point| *point != a && *point != b)
        else {
            continue;
        };
        if c == d || incidence.contains_key(&ordered_pair(c, d)) {
            continue;
        }
        let before = objective(candidate, &[cell_index, other_index]);
        let mut replacements = [[c, d, a], [d, c, b]];
        for triangle in &mut replacements {
            if signed_area(*triangle, &candidate.points) < 0.0 {
                triangle.swap(1, 2);
            }
        }
        let old_cells = [
            candidate.cells[cell_index].clone(),
            candidate.cells[other_index].clone(),
        ];
        candidate.cells[cell_index].points = replacements[0];
        candidate.cells[other_index].points = replacements[1];
        if affected_valid(domain, candidate, &[cell_index, other_index])
            && improves(before, objective(candidate, &[cell_index, other_index]))
        {
            return Ok(true);
        }
        candidate.cells[cell_index] = old_cells[0].clone();
        candidate.cells[other_index] = old_cells[1].clone();
    }
    Ok(false)
}

fn try_smooth(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    cell_index: usize,
) -> MeshResult<bool> {
    for key in candidate.cells[cell_index].points {
        if candidate.points[&key].boundary {
            continue;
        }
        let incident = incident_cells(context, candidate, key)?;
        let mut neighbors = BTreeSet::new();
        for &index in &incident {
            neighbors.extend(
                candidate.cells[index]
                    .points
                    .into_iter()
                    .filter(|point| *point != key),
            );
        }
        if neighbors.len() < 3 {
            continue;
        }
        let uv = neighbors
            .iter()
            .map(|neighbor| candidate.points[neighbor].uv)
            .fold([0.0; 2], |sum, uv| [sum[0] + uv[0], sum[1] + uv[1]])
            .map(|value| value / neighbors.len() as f64);
        let world = space.point(uv[0], uv[1]);
        if domain.domain_sdf(&[world])[0] >= 0.0 {
            continue;
        }
        let before = objective(candidate, &incident);
        let old = candidate.points[&key];
        candidate.points.insert(
            key,
            Point {
                uv,
                world: world.to_array(),
                boundary: false,
            },
        );
        if affected_valid(domain, candidate, &incident)
            && improves(before, objective(candidate, &incident))
        {
            return Ok(true);
        }
        candidate.points.insert(key, old);
    }
    Ok(false)
}

fn try_boundary_smooth(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &Assessment,
    cell_index: usize,
) -> MeshResult<bool> {
    for key in candidate.cells[cell_index].points {
        if !candidate.points[&key].boundary {
            continue;
        }
        let neighbors = assessment
            .boundary
            .iter()
            .filter_map(|edge| {
                if edge.points[0] == key {
                    Some(edge.points[1])
                } else if edge.points[1] == key {
                    Some(edge.points[0])
                } else {
                    None
                }
            })
            .collect::<BTreeSet<_>>();
        if neighbors.len() != 2 {
            continue;
        }
        let mut iter = neighbors.into_iter();
        let a = candidate.points[&iter.next().expect("two neighbors")];
        let b = candidate.points[&iter.next().expect("two neighbors")];
        let target = [(a.uv[0] + b.uv[0]) * 0.5, (a.uv[1] + b.uv[1]) * 0.5];
        let incident = incident_cells(context, candidate, key)?;
        let before = objective(candidate, &incident);
        let old = candidate.points[&key];
        let center = incident
            .iter()
            .map(|index| cell_centroid(candidate, *index))
            .fold([0.0; 3], |sum, value| {
                [sum[0] + value[0], sum[1] + value[1], sum[2] + value[2]]
            })
            .map(|value| value / incident.len() as f64);
        let center = Vec3::from_array(center);
        let mut best = None::<((usize, f64), Point)>;
        for fraction in [1.0, 0.75, 0.5, 0.25, 0.125] {
            let uv = [
                old.uv[0] + fraction * (target[0] - old.uv[0]),
                old.uv[1] + fraction * (target[1] - old.uv[1]),
            ];
            let boundary_world = space.point(uv[0], uv[1]);
            let seed = (1..=16)
                .map(|step| step as f64 / 16.0)
                .map(|weight| boundary_world + (center - boundary_world) * weight)
                .find(|point| domain.domain_sdf(&[*point])[0] < 0.0);
            let Some(seed) = seed else {
                continue;
            };
            let projection = domain
                .project_to_boundary(&[seed])
                .map_err(|error| MeshError::InvalidInput(error.to_string()))?[0];
            if !projection.converged {
                continue;
            }
            let coords = space.coords(projection.point);
            let point = Point {
                uv: [coords[0], coords[1]],
                world: projection.point.to_array(),
                boundary: true,
            };
            candidate.points.insert(key, point);
            if affected_valid(domain, candidate, &incident) {
                let after = objective(candidate, &incident);
                if improves(before, after)
                    && best
                        .as_ref()
                        .is_none_or(|(best_objective, _)| improves(*best_objective, after))
                {
                    best = Some((after, point));
                }
            }
            candidate.points.insert(key, old);
        }
        if let Some((_, point)) = best {
            candidate.points.insert(key, point);
            return Ok(true);
        }
    }
    Ok(false)
}

fn try_offcenter(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    candidate: &mut Candidate,
    cell_index: usize,
) -> bool {
    let cell = candidate.cells[cell_index].clone();
    let points = cell.points.map(|key| candidate.points[&key]);
    let lengths = [
        distance3(points[1].world, points[2].world),
        distance3(points[2].world, points[0].world),
        distance3(points[0].world, points[1].world),
    ];
    let sum = lengths.iter().sum::<f64>();
    if sum <= f64::EPSILON {
        return false;
    }
    let uv = [
        (0..3).map(|i| points[i].uv[0] * lengths[i]).sum::<f64>() / sum,
        (0..3).map(|i| points[i].uv[1] * lengths[i]).sum::<f64>() / sum,
    ];
    let world = space.point(uv[0], uv[1]);
    if domain.domain_sdf(&[world])[0] >= 0.0 {
        return false;
    }
    let before = objective(candidate, &[cell_index]);
    let key = PointKey::Inserted(candidate.next_inserted);
    candidate.next_inserted += 1;
    candidate.points.insert(
        key,
        Point {
            uv,
            world: world.to_array(),
            boundary: false,
        },
    );
    let replacements = [
        Cell {
            points: [cell.points[0], cell.points[1], key],
            leaf: cell.leaf,
        },
        Cell {
            points: [cell.points[1], cell.points[2], key],
            leaf: cell.leaf,
        },
        Cell {
            points: [cell.points[2], cell.points[0], key],
            leaf: cell.leaf,
        },
    ];
    candidate.cells[cell_index] = replacements[0].clone();
    candidate.cells.push(replacements[1].clone());
    candidate.cells.push(replacements[2].clone());
    let affected = [
        cell_index,
        candidate.cells.len() - 2,
        candidate.cells.len() - 1,
    ];
    if affected_valid(domain, candidate, &affected)
        && improves(before, objective(candidate, &affected))
    {
        return true;
    }
    candidate.cells.truncate(candidate.cells.len() - 2);
    candidate.cells[cell_index] = cell;
    candidate.points.remove(&key);
    false
}

fn try_remove_boundary_ear(
    domain: &MeshableDomain,
    candidate: &mut Candidate,
    assessment: &Assessment,
    cell_index: usize,
) -> bool {
    let cell = &candidate.cells[cell_index];
    let exposed = assessment
        .boundary
        .iter()
        .filter(|edge| edge.cell == cell_index)
        .map(|edge| ordered_pair(edge.points[0], edge.points[1]))
        .collect::<BTreeSet<_>>();
    if exposed.len() != 2 {
        return false;
    }
    let Some(replacement) = (0..3)
        .map(|edge| {
            ordered_pair(
                cell.points[edge],
                cell.points[(edge + 1) % cell.points.len()],
            )
        })
        .find(|edge| !exposed.contains(edge))
    else {
        return false;
    };
    if !candidate.points[&replacement.0].boundary || !candidate.points[&replacement.1].boundary {
        return false;
    }
    let incidence = candidate
        .cells
        .iter()
        .filter(|cell| cell.points.contains(&replacement.0) && cell.points.contains(&replacement.1))
        .count();
    if incidence != 2 {
        return false;
    }
    let a = candidate.points[&replacement.0].world;
    let b = candidate.points[&replacement.1].world;
    let midpoint = midpoint3(a, b);
    let size = distance3(a, b);
    let sdf = domain.domain_sdf(&[Vec3::from_array(midpoint)])[0];
    if !sdf.is_finite() || sdf.abs() > chord_tolerance(domain, size) {
        return false;
    }
    candidate.cells.remove(cell_index);
    true
}

fn affected_valid(domain: &MeshableDomain, candidate: &Candidate, cells: &[usize]) -> bool {
    cells.iter().all(|&index| {
        let positions = candidate.cells[index]
            .points
            .map(|key| candidate.points[&key].world);
        let size = maximum_edge_2d(positions);
        signed_area(candidate.cells[index].points, &candidate.points) > orientation_tolerance(size)
            && quality_score("tri3", &positions, QualityMetric::ScaledJacobian)
                .is_some_and(|quality| quality > VALID_QUALITY)
            && cell_is_contained_2d(domain, &positions, chord_tolerance(domain, size))
    })
}

fn incident_cells(
    context: &MeshingContext<'_>,
    candidate: &Candidate,
    key: PointKey,
) -> MeshResult<Vec<usize>> {
    let mut result = Vec::new();
    for (index, cell) in candidate.cells.iter().enumerate() {
        if index % 512 == 0 {
            context.check()?;
        }
        if cell.points.contains(&key) {
            result.push(index);
        }
    }
    Ok(result)
}

fn objective(candidate: &Candidate, cells: &[usize]) -> (usize, f64) {
    let qualities = cells.iter().map(|&index| {
        let positions = candidate.cells[index]
            .points
            .map(|key| candidate.points[&key].world);
        quality_score("tri3", &positions, QualityMetric::ScaledJacobian).unwrap_or(0.0)
    });
    qualities.fold((0, 1.0_f64), |(below, worst), quality| {
        (
            below + usize::from(quality < QUALITY_FLOOR),
            worst.min(quality),
        )
    })
}

fn improves(before: (usize, f64), after: (usize, f64)) -> bool {
    after.0 < before.0 || (after.0 == before.0 && after.1 > before.1 + 1.0e-12)
}

fn cell_is_contained_2d(domain: &MeshableDomain, points: &[[f64; 3]; 3], tolerance: f64) -> bool {
    let samples = [
        centroid(points),
        midpoint3(points[0], points[1]),
        midpoint3(points[1], points[2]),
        midpoint3(points[2], points[0]),
    ]
    .map(Vec3::from_array);
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
            "domain {:?} produced no valid 2D elements",
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
            .ok_or_else(|| MeshError::LimitExceeded("2D point ID space exhausted".into()))?;
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
            builder.tri3(
                cell.points.map(|key| ids[&key]),
                catalog.zone,
                catalog.source,
            )?;
        }
        for edge in assessment
            .boundary
            .iter()
            .filter(|edge| (start..end).contains(&edge.cell))
        {
            let a = candidate.points[&edge.points[0]].world;
            let b = candidate.points[&edge.points[1]].world;
            let midpoint = midpoint3(a, b);
            let class = domain
                .classify_boundary(
                    &[Vec3::from_array(midpoint)],
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
            builder.boundary_edge(edge.points.map(|key| ids[&key]), vec![tag])?;
        }
        let chunk = builder.build(2)?;
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

fn signed_area(triangle: [PointKey; 3], points: &BTreeMap<PointKey, Point>) -> f64 {
    let [a, b, c] = triangle.map(|key| points[&key].uv);
    0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
}

fn pair_quality(triangles: [[PointKey; 3]; 2], points: &BTreeMap<PointKey, Point>) -> f64 {
    triangles
        .into_iter()
        .map(|triangle| {
            let positions = triangle.map(|key| points[&key].world);
            quality_score("tri3", &positions, QualityMetric::ScaledJacobian).unwrap_or(0.0)
        })
        .fold(1.0, f64::min)
}

fn cell_centroid(candidate: &Candidate, cell: usize) -> [f64; 3] {
    let positions = candidate.cells[cell]
        .points
        .map(|key| candidate.points[&key].world);
    centroid(&positions)
}

fn maximum_edge_2d(points: [[f64; 3]; 3]) -> f64 {
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
    root_tolerance(domain, local_size).max(local_size * 0.5)
}

fn orientation_tolerance(local_size: f64) -> f64 {
    local_size.powi(2) * 1.0e-12
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
        "domain {:?} could not produce a valid adaptive 2D mesh at the {:.6e} element-size floor near ({:.6}, {:.6}, {:.6}): {reason}; worst Scaled Jacobian={quality:.6e}, required > {VALID_QUALITY:.1e} and generation target {QUALITY_FLOOR:.2}",
        domain.name,
        context.element_min_size,
        location[0],
        location[1],
        location[2],
    ))
}

fn midpoint(a: Lattice, b: Lattice) -> Lattice {
    Lattice {
        x: (a.x + b.x) / 2,
        y: (a.y + b.y) / 2,
    }
}

fn ordered_lattice(a: Lattice, b: Lattice) -> (Lattice, Lattice) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn ordered_pair<T: Ord + Copy>(a: T, b: T) -> (T, T) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
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
    use caso_kernel::meshing::meshable_domains_from_document;
    use caso_kernel::roles::DomainKind;
    use caso_kernel::scene::SceneDocument;
    use caso_kernel::vec3::vec3;

    use crate::{
        run_meshing, ControlSet, GenerationLimits, JobControl, MemoryArtifact, MemoryStorage,
        MeshArtifact, MeshCatalog, MeshChunk, MeshFile, MeshSink, MeshingContext, MeshingRequest,
    };

    #[derive(Default)]
    struct TestSink {
        next: u32,
        cells: usize,
    }

    impl MeshSink for TestSink {
        fn allocate_chunk_id(&mut self) -> MeshResult<u32> {
            self.next += 1;
            Ok(self.next)
        }

        fn emit(&mut self, chunk: MeshChunk) -> MeshResult<()> {
            self.cells += chunk.cells.len();
            Ok(())
        }
    }

    fn sample(index: u64, sdf: f64) -> Sample {
        Sample {
            key: Lattice { x: index, y: 0 },
            uv: [index as f64, 0.0],
            world: [index as f64, 0.0, 0.0],
            sdf,
        }
    }

    fn mesh_document(
        document: &SceneDocument,
        element_min_size: f64,
        element_max_size: f64,
    ) -> MemoryArtifact {
        let output = run_meshing(
            MeshingRequest {
                domains: meshable_domains_from_document(document).expect("meshable domains"),
                algorithm_id: "advancing_front".into(),
                element_min_size,
                element_max_size,
                controls: ControlSet::default(),
                limits: GenerationLimits::default(),
                job_control: JobControl::default(),
            },
            MemoryStorage::new(128 * 1024 * 1024).expect("memory storage"),
        )
        .expect("valid 2D domain must mesh");
        assert!(output.statistics.cells > 0);
        let MeshArtifact::Memory(artifact) = output.artifact else {
            panic!("expected an in-memory mesh");
        };
        MeshFile::from_memory(artifact.clone())
            .expect("read generated mesh")
            .full_audit(&JobControl::default())
            .expect("audit generated mesh");
        artifact
    }

    fn primitive_document(kind: &str, start: Vec3, end: Vec3) -> SceneDocument {
        let mut document = SceneDocument::new();
        let root = document
            .add_primitive_from_drag(kind, start, end, 1.0)
            .expect("2D primitive");
        document.rename(root, kind).expect("rename primitive");
        document
            .set_domain_root(root, DomainKind::Fluid)
            .expect("mark fluid domain");
        document
    }

    fn mesh_document_fast(
        document: &SceneDocument,
        element_min_size: f64,
        element_max_size: f64,
    ) -> MeshResult<()> {
        let domains = meshable_domains_from_document(document)
            .map_err(|error| MeshError::InvalidInput(error.to_string()))?;
        let controls = ControlSet::default();
        let job_control = JobControl::default();
        let catalog = MeshCatalog::from_domains(&domains, "advancing_front");
        let context = MeshingContext {
            domains: &domains,
            element_min_size,
            element_max_size,
            controls: &controls,
            job_control: &job_control,
            limits: GenerationLimits::default(),
            catalog: &catalog,
        };
        let mut sink = TestSink::default();
        let statistics = generate(&context, &mut sink)?;
        assert!(statistics.cells > 0);
        assert_eq!(statistics.cells as usize, sink.cells);
        Ok(())
    }

    #[test]
    fn marching_triangles_cover_all_sign_masks_and_zero_vertices() {
        let expected = [0, 3, 3, 4, 3, 4, 4, 3];
        for mask in 0u8..8 {
            let samples = std::array::from_fn(|index| {
                sample(
                    index as u64,
                    if mask & (1 << index) != 0 { -1.0 } else { 1.0 },
                )
            });
            let polygon = clipped_polygon(samples, |a, b| {
                Ok(PointKey::Crossing(
                    ordered_lattice(a.key, b.key).0,
                    ordered_lattice(a.key, b.key).1,
                ))
            })
            .unwrap();
            assert_eq!(
                dedup_polygon(polygon).len(),
                expected[mask as usize],
                "mask {mask:03b}"
            );
        }

        let polygon = clipped_polygon([sample(0, 0.0), sample(1, 1.0), sample(2, -1.0)], |a, b| {
            if a.sdf == 0.0 {
                Ok(PointKey::Lattice(a.key))
            } else if b.sdf == 0.0 {
                Ok(PointKey::Lattice(b.key))
            } else {
                Ok(PointKey::Crossing(
                    ordered_lattice(a.key, b.key).0,
                    ordered_lattice(a.key, b.key).1,
                ))
            }
        })
        .unwrap();
        assert_eq!(dedup_polygon(polygon).len(), 3);
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
    fn curved_sharp_thin_and_transformed_primitives_mesh() {
        let fixtures = [
            primitive_document("circle", vec3(-0.8, -0.8, 0.0), vec3(0.8, 0.8, 0.0)),
            primitive_document("ellipse", vec3(-1.2, -0.35, 0.0), vec3(1.2, 0.35, 0.0)),
            primitive_document(
                "rounded_rectangle",
                vec3(-1.0, -0.55, 0.0),
                vec3(1.0, 0.55, 0.0),
            ),
            primitive_document("square", vec3(-0.75, -0.75, 0.0), vec3(0.75, 0.75, 0.0)),
            primitive_document("rectangle", vec3(-1.5, -0.08, 0.0), vec3(1.5, 0.08, 0.0)),
        ];
        for document in fixtures {
            mesh_document(&document, 0.025, 0.2);
        }

        let mut transformed =
            primitive_document("ellipse", vec3(-1.0, -0.4, 0.0), vec3(1.0, 0.4, 0.0));
        let root = transformed.fluid_domain.as_ref().expect("fluid root").root;
        transformed
            .move_object(root, vec3(2.25, -1.75, 0.5))
            .expect("translate profile");
        mesh_document(&transformed, 0.025, 0.2);
    }

    #[test]
    fn holes_and_disconnected_csg_mesh_deterministically() {
        let mut hole = SceneDocument::new();
        let outer = hole
            .add_primitive_from_drag(
                "rectangle",
                vec3(-2.0, -1.25, 0.0),
                vec3(2.0, 1.25, 0.0),
                1.0,
            )
            .expect("outer rectangle");
        let inner = hole
            .add_primitive_from_drag(
                "circle",
                vec3(-0.55, -0.55, 0.0),
                vec3(0.55, 0.55, 0.0),
                1.0,
            )
            .expect("inner circle");
        let difference = hole
            .combine(outer, inner, "difference")
            .expect("difference");
        hole.rename(difference, "annulus")
            .expect("rename difference");
        hole.set_domain_root(difference, DomainKind::Fluid)
            .expect("mark difference");
        let first = mesh_document(&hole, 0.025, 0.2);
        let second = mesh_document(&hole, 0.025, 0.2);
        assert_eq!(first, second);

        let mut disconnected = SceneDocument::new();
        let left = disconnected
            .add_primitive_from_drag("circle", vec3(-2.0, -0.6, 0.0), vec3(-0.8, 0.6, 0.0), 1.0)
            .expect("left circle");
        let right = disconnected
            .add_primitive_from_drag("circle", vec3(0.8, -0.6, 0.0), vec3(2.0, 0.6, 0.0), 1.0)
            .expect("right circle");
        let union = disconnected.combine(left, right, "union").expect("union");
        disconnected.rename(union, "islands").expect("rename union");
        disconnected
            .set_domain_root(union, DomainKind::Fluid)
            .expect("mark union");
        mesh_document(&disconnected, 0.025, 0.2);
    }

    #[test]
    fn shifted_and_near_grid_degenerate_domains_do_not_fail() {
        let kinds = ["circle", "ellipse", "rounded_rectangle", "square"];
        for index in 0..48 {
            let x = (index % 7) as f64 * 0.017 - 0.051;
            let y = (index % 5) as f64 * 0.013 - 0.026;
            let width = 0.35 + (index % 4) as f64 * 0.11;
            let height = 0.19 + (index % 3) as f64 * 0.09;
            let kind = kinds[index % kinds.len()];
            let document = primitive_document(
                kind,
                vec3(x - width, y - height, 0.0),
                vec3(x + width, y + height, 0.0),
            );
            let maximum = [0.08, 0.11, 0.16][index % 3];
            mesh_document_fast(&document, maximum * 0.125, maximum)
                .unwrap_or_else(|error| panic!("fixture {index} ({kind}) failed: {error}"));
            if index < 12 {
                mesh_document_fast(&document, maximum, maximum).unwrap_or_else(|error| {
                    panic!("fixed-size fixture {index} ({kind}) failed: {error}")
                });
            }
        }

        for index in 0..12 {
            let offset = index as f64 * 0.0075;
            let mut document = SceneDocument::new();
            let outer = document
                .add_primitive_from_drag(
                    "rectangle",
                    vec3(-1.0, -0.7, 0.0),
                    vec3(1.0, 0.7, 0.0),
                    1.0,
                )
                .expect("outer rectangle");
            let inner = document
                .add_primitive_from_drag(
                    "ellipse",
                    vec3(-0.32 + offset, -0.18, 0.0),
                    vec3(0.32 + offset, 0.18, 0.0),
                    1.0,
                )
                .expect("inner ellipse");
            let root = document
                .combine(outer, inner, "difference")
                .expect("difference");
            document.rename(root, "offset_hole").expect("rename domain");
            document
                .set_domain_root(root, DomainKind::Fluid)
                .expect("mark domain");
            mesh_document_fast(&document, 0.0125, 0.1)
                .unwrap_or_else(|error| panic!("offset-hole fixture {index} failed: {error}"));
            mesh_document_fast(&document, 0.1, 0.1).unwrap_or_else(|error| {
                panic!("fixed-size offset-hole fixture {index} failed: {error}")
            });
        }

        for index in 0..9 {
            let offset = 0.50 + index as f64 * 0.02;
            let mut document = SceneDocument::new();
            let outer = document
                .add_primitive_from_drag(
                    "rectangle",
                    vec3(-1.0, -0.7, 0.0),
                    vec3(1.0, 0.7, 0.0),
                    1.0,
                )
                .expect("outer rectangle");
            let inner = document
                .add_primitive_from_drag(
                    "circle",
                    vec3(offset - 0.28, -0.28, 0.0),
                    vec3(offset + 0.28, 0.28, 0.0),
                    1.0,
                )
                .expect("near-wall hole");
            let root = document
                .combine(outer, inner, "difference")
                .expect("near-wall difference");
            document
                .rename(root, "narrow_ligament")
                .expect("rename domain");
            document
                .set_domain_root(root, DomainKind::Fluid)
                .expect("mark domain");
            mesh_document_fast(&document, 0.005, 0.1)
                .unwrap_or_else(|error| panic!("narrow-ligament fixture {index} failed: {error}"));
        }
    }

    #[test]
    fn concave_and_acute_polygons_mesh_without_boundary_gaps() {
        let fixtures = [
            vec![
                vec3(-1.0, -1.0, 0.0),
                vec3(1.0, -1.0, 0.0),
                vec3(1.0, -0.2, 0.0),
                vec3(0.2, -0.2, 0.0),
                vec3(0.2, 1.0, 0.0),
                vec3(-1.0, 1.0, 0.0),
            ],
            vec![
                vec3(-1.1, -0.6, 0.0),
                vec3(0.0, -0.08, 0.0),
                vec3(1.1, -0.6, 0.0),
                vec3(0.32, 0.08, 0.0),
                vec3(0.7, 1.0, 0.0),
                vec3(0.0, 0.28, 0.0),
                vec3(-0.7, 1.0, 0.0),
                vec3(-0.32, 0.08, 0.0),
            ],
            vec![
                vec3(-1.0, -0.25, 0.0),
                vec3(0.82, -0.25, 0.0),
                vec3(1.0, 0.0, 0.0),
                vec3(0.82, 0.25, 0.0),
                vec3(-1.0, 0.25, 0.0),
            ],
        ];
        for (index, points) in fixtures.into_iter().enumerate() {
            let mut document = SceneDocument::new();
            let root = document
                .add_point_shape_from_world_points("polygon", &points, "xy")
                .expect("polygon");
            document
                .rename(root, format!("polygon_{index}"))
                .expect("rename");
            document
                .set_domain_root(root, DomainKind::Fluid)
                .expect("mark polygon");
            mesh_document_fast(&document, 0.01, 0.1)
                .unwrap_or_else(|error| panic!("polygon fixture {index} failed: {error}"));
        }

        for sides in 3..=9 {
            let mut document = SceneDocument::new();
            let root = document
                .add_regular_polygon_from_world_points(
                    &[vec3(0.0, 0.0, 0.0), vec3(0.83, 0.17, 0.0)],
                    sides,
                    "xy",
                )
                .expect("regular polygon");
            document
                .set_domain_root(root, DomainKind::Fluid)
                .expect("mark regular polygon");
            mesh_document_fast(&document, 0.01, 0.12)
                .unwrap_or_else(|error| panic!("{sides}-sided polygon failed: {error}"));
        }

        for fixture in 0..16 {
            let tips = 4 + fixture % 5;
            let rotation = fixture as f64 * 0.037;
            let points = (0..tips * 2)
                .map(|vertex| {
                    let angle =
                        rotation + std::f64::consts::TAU * vertex as f64 / (tips * 2) as f64;
                    let radius = if vertex % 2 == 0 {
                        0.9
                    } else {
                        0.12 + 0.025 * (fixture % 4) as f64
                    };
                    vec3(radius * angle.cos(), radius * angle.sin(), 0.0)
                })
                .collect::<Vec<_>>();
            let mut document = SceneDocument::new();
            let root = document
                .add_point_shape_from_world_points("polygon", &points, "xy")
                .expect("star polygon");
            document
                .set_domain_root(root, DomainKind::Fluid)
                .expect("mark star polygon");
            mesh_document_fast(&document, 0.006, 0.08)
                .unwrap_or_else(|error| panic!("star fixture {fixture} failed: {error}"));
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn active_generation_cancels_promptly() {
        let document = primitive_document("rectangle", vec3(-5.0, -5.0, 0.0), vec3(5.0, 5.0, 0.0));
        let domains = meshable_domains_from_document(&document).expect("meshable domains");
        let job_control = JobControl::default();
        let cancel = job_control.clone();
        let worker = std::thread::spawn(move || {
            run_meshing(
                MeshingRequest {
                    domains,
                    algorithm_id: "advancing_front".into(),
                    element_min_size: 0.005,
                    element_max_size: 0.01,
                    controls: ControlSet::default(),
                    limits: GenerationLimits::default(),
                    job_control,
                },
                MemoryStorage::new(32 * 1024 * 1024).expect("memory storage"),
            )
        });
        std::thread::sleep(std::time::Duration::from_millis(10));
        let started = std::time::Instant::now();
        cancel.cancel();
        assert!(matches!(
            worker.join().expect("meshing worker"),
            Err(MeshError::Cancelled)
        ));
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "cancellation took {:?}",
            started.elapsed()
        );
    }
}
