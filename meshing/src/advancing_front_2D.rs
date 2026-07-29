use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use caso_kernel::meshing::{BoundaryBand, MeshableDomain, MeshableDomainSpace};
use caso_kernel::vec3::Vec3;

use crate::algorithm::{
    MeshSink, MeshingContext, MeshingPhase, MeshingProgress, MeshingStatistics,
};
use crate::chunk::{MeshChunkBuilder, MeshId};
use crate::controls::BoundaryLayerControl;
use crate::error::{MeshError, MeshResult};
use crate::quality::{quality_score, QualityMetric};
use crate::schema::Bounds3;

const QUALITY_TARGET: f64 = 0.40;
const VALID_QUALITY: f64 = 1.0e-8;
const EDGE_RATIO_MIN: f64 = std::f64::consts::FRAC_1_SQRT_2;
const EDGE_RATIO_MAX: f64 = std::f64::consts::SQRT_2;
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
    protected: bool,
}

#[derive(Debug, Clone)]
struct Cell {
    points: Vec<PointKey>,
    leaf: Leaf,
    protected: bool,
}

impl Cell {
    fn triangle(points: [PointKey; 3], leaf: Leaf) -> Self {
        Self {
            points: points.into(),
            leaf,
            protected: false,
        }
    }

    fn quad(points: [PointKey; 4], leaf: Leaf, protected: bool) -> Self {
        Self {
            points: points.into(),
            leaf,
            protected,
        }
    }

    fn element_type(&self) -> &'static str {
        match self.points.len() {
            3 => "tri3",
            4 => "quad4",
            _ => unreachable!("2D candidate cells are triangles or quads"),
        }
    }
}

#[derive(Debug, Clone)]
struct Candidate {
    points: BTreeMap<PointKey, Point>,
    cells: Vec<Cell>,
    construction_failures: BTreeSet<Leaf>,
    next_inserted: u64,
    layer_edge_targets: BTreeMap<(PointKey, PointKey), f64>,
}

#[derive(Debug, Clone)]
struct BoundaryEdge {
    points: [PointKey; 2],
    cell: usize,
    owner: Option<String>,
}

#[derive(Debug)]
struct Assessment {
    boundary: Vec<BoundaryEdge>,
    boundary_vertices: BTreeSet<PointKey>,
    refine: BTreeSet<Leaf>,
    reason: Option<String>,
    location: Option<[f64; 3]>,
    worst_quality: f64,
    violations: Vec<Violation>,
    score: PatchScore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Entity {
    Cell(usize),
    Edge(PointKey, PointKey),
}

#[derive(Debug, Clone, Copy)]
struct Violation {
    severity: f64,
    entity: Entity,
}

#[derive(Debug, Clone, Copy)]
struct PatchScore {
    hard_invalid: usize,
    unmet_targets: usize,
    worst_violation: f64,
    worst_quality: f64,
    mean_squared_log_size_error: f64,
    quad_count: usize,
    triangle_count: usize,
    cell_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Action {
    Merge(PointKey, PointKey),
    SplitQuad(usize),
    Flip(PointKey, PointKey),
    RelocateInterior(PointKey, u8),
    RelocateBoundary(PointKey, u8),
    Split(PointKey, PointKey),
    Insert(usize),
    Collapse(PointKey, PointKey),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct LayerKey {
    first_height: u64,
    layers: usize,
    growth: u64,
}

impl LayerKey {
    fn from_control(control: &BoundaryLayerControl) -> Self {
        Self {
            first_height: control.first_height.to_bits(),
            layers: control.layers,
            growth: control.growth.to_bits(),
        }
    }

    fn first_height(self) -> f64 {
        f64::from_bits(self.first_height)
    }

    fn growth(self) -> f64 {
        f64::from_bits(self.growth)
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

        let (mut candidate, mut assessment) = loop {
            context.check()?;
            let candidate = build_candidate(context, domain, &space, &mut sampler, &leaves)?;
            let assessment = assess(domain, &space, context, &candidate)?;
            if assessment.refine.is_empty() {
                break (candidate, assessment);
            }
            let requested = assessment.refine.clone();
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
                        "topology discovery could not establish a valid boundary-conforming mesh",
                    ),
                    assessment.location,
                    assessment.worst_quality,
                ));
            }
            refine_leaves(context, &mut leaves, &splittable)?;
            balance(context, grid, &mut leaves)?;
        };
        let has_layers = context
            .controls
            .boundary_layers
            .iter()
            .any(|control| control.domain == domain.name);
        if has_layers {
            prepare_layer_boundaries(domain, &space, context, &mut candidate, &mut assessment)?;
        }
        optimize(domain, &space, context, &mut candidate, &mut assessment)?;
        if has_layers {
            apply_boundary_layers(domain, &space, context, &mut candidate, &mut assessment)?;
        }

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
        let probes = samples.map(|sample| Vec3::from_array(sample.world));
        let target = regional_target(
            context,
            &domain.name,
            Vec3::from_array(center.world),
            radius,
            &probes,
        );
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

fn regional_target(
    context: &MeshingContext<'_>,
    domain: &str,
    center: Vec3,
    radius: f64,
    probes: &[Vec3],
) -> f64 {
    regional_target_from_controls(
        context.controls,
        domain,
        center,
        radius,
        probes,
        context.element_min_size,
        context.element_max_size,
    )
}

fn regional_target_from_controls(
    controls: &crate::ControlSet,
    domain: &str,
    center: Vec3,
    radius: f64,
    probes: &[Vec3],
    minimum: f64,
    maximum: f64,
) -> f64 {
    let sampled = probes
        .iter()
        .map(|point| controls.size_at(domain, *point, maximum))
        .fold(maximum, f64::min);
    controls
        .refinements
        .iter()
        .filter(|control| control.domain == domain)
        .fold(sampled, |target, control| {
            let lower_bound = control.region.sdf(center) - radius.max(0.0);
            target.min(control.size + control.gradation * lower_bound.max(0.0))
        })
        .clamp(minimum, maximum)
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
        layer_edge_targets: BTreeMap::new(),
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
                    protected: false,
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
        candidate.cells.push(Cell::triangle(triangle, leaf));
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
                protected: false,
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
                protected: false,
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
        protected: false,
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
    let construction_failures = candidate.construction_failures.len();
    let mut assessment = Assessment {
        boundary: Vec::new(),
        boundary_vertices: BTreeSet::new(),
        refine: candidate.construction_failures.clone(),
        reason: (!candidate.construction_failures.is_empty())
            .then(|| "boundary clipping produced a degenerate triangle".into()),
        location: None,
        worst_quality: 1.0,
        violations: Vec::new(),
        score: PatchScore {
            hard_invalid: construction_failures,
            unmet_targets: 0,
            worst_violation: 0.0,
            worst_quality: 1.0,
            mean_squared_log_size_error: 0.0,
            quad_count: candidate
                .cells
                .iter()
                .filter(|cell| cell.points.len() == 4)
                .count(),
            triangle_count: candidate
                .cells
                .iter()
                .filter(|cell| cell.points.len() == 3)
                .count(),
            cell_count: candidate.cells.len(),
        },
    };
    let mut incidence = BTreeMap::<(PointKey, PointKey), Vec<(usize, [PointKey; 2])>>::new();
    let mut unique_cells = BTreeMap::<(usize, Vec<PointKey>), usize>::new();
    for (index, cell) in candidate.cells.iter().enumerate() {
        if index % 512 == 0 {
            context.check()?;
        }
        let mut signature = cell.points.clone();
        signature.sort_unstable();
        if unique_cells
            .insert((signature.len(), signature), index)
            .is_some()
        {
            assessment.score.hard_invalid += 1;
            record_refinement(
                &mut assessment,
                cell.leaf,
                "duplicate cell connectivity",
                cell_centroid(candidate, index),
            );
        }
        let positions = cell
            .points
            .iter()
            .map(|key| candidate.points[key].world)
            .collect::<Vec<_>>();
        let size = maximum_edge_2d(&positions);
        let area = signed_area_polygon(&cell.points, &candidate.points);
        let containment_tolerance = chord_tolerance(domain, size);
        let containment_residual = cell_containment_residual(domain, &positions);
        let quality = quality_score(
            cell.element_type(),
            &positions,
            QualityMetric::ScaledJacobian,
        )
        .unwrap_or(0.0);
        assessment.worst_quality = assessment.worst_quality.min(quality);
        if area <= orientation_tolerance(size)
            || quality <= VALID_QUALITY
            || (cell.points.len() == 4 && polygon_self_intersects(&cell.points, &candidate.points))
        {
            assessment.score.hard_invalid += 1;
            record_refinement(
                &mut assessment,
                cell.leaf,
                "cell is inverted, self-intersecting, degenerate, or below the Scaled Jacobian validity floor",
                centroid_slice(&positions),
            );
        } else if containment_residual > containment_tolerance {
            assessment.score.hard_invalid += 1;
            let reason = format!(
                "{} containment samples leave the negative SDF domain (residual={containment_residual:.6e}, tolerance={containment_tolerance:.6e})",
                cell.element_type(),
            );
            record_refinement(
                &mut assessment,
                cell.leaf,
                &reason,
                centroid_slice(&positions),
            );
        } else if quality < QUALITY_TARGET {
            assessment.score.unmet_targets += 1;
            assessment.violations.push(Violation {
                severity: (QUALITY_TARGET - quality) / QUALITY_TARGET,
                entity: Entity::Cell(index),
            });
            if assessment.location.is_none() {
                assessment.location = Some(centroid_slice(&positions));
            }
        }
        for edge in 0..cell.points.len() {
            let oriented = [
                cell.points[edge],
                cell.points[(edge + 1) % cell.points.len()],
            ];
            incidence
                .entry(ordered_pair(oriented[0], oriented[1]))
                .or_default()
                .push((index, oriented));
        }
    }
    let mut size_error = 0.0;
    for (&edge, entries) in &incidence {
        let a = candidate.points[&edge.0].world;
        let b = candidate.points[&edge.1].world;
        let midpoint = midpoint3(a, b);
        let length = distance3(a, b);
        let probes = [
            Vec3::from_array(a),
            Vec3::from_array(b),
            Vec3::from_array(midpoint),
        ];
        let target = candidate
            .layer_edge_targets
            .get(&edge)
            .copied()
            .unwrap_or_else(|| {
                regional_target(
                    context,
                    &domain.name,
                    Vec3::from_array(midpoint),
                    length * 0.5,
                    &probes,
                )
            });
        let ratio = length / target;
        let log_error = ratio.max(f64::MIN_POSITIVE).ln();
        size_error += log_error * log_error;
        let severity = if ratio < EDGE_RATIO_MIN {
            EDGE_RATIO_MIN / ratio.max(f64::MIN_POSITIVE) - 1.0
        } else if ratio > EDGE_RATIO_MAX {
            ratio / EDGE_RATIO_MAX - 1.0
        } else {
            0.0
        };
        if severity > 0.0 {
            assessment.score.unmet_targets += 1;
            assessment.violations.push(Violation {
                severity,
                entity: Entity::Edge(edge.0, edge.1),
            });
        }
        if entries.len() > 2 {
            for (cell, _) in entries {
                assessment.score.hard_invalid += 1;
                record_refinement(
                    &mut assessment,
                    candidate.cells[*cell].leaf,
                    "non-manifold cell edge incidence",
                    cell_centroid(candidate, *cell),
                );
            }
        } else if let [(cell, oriented)] = entries.as_slice() {
            assessment.boundary.push(BoundaryEdge {
                points: *oriented,
                cell: *cell,
                owner: None,
            });
            assessment.boundary_vertices.extend(oriented);
        }
    }
    assessment.score.mean_squared_log_size_error = if incidence.is_empty() {
        0.0
    } else {
        size_error / incidence.len() as f64
    };
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
        let probes = [
            Vec3::from_array(a),
            Vec3::from_array(b),
            Vec3::from_array(midpoint),
        ];
        let target = regional_target(
            context,
            &domain.name,
            Vec3::from_array(midpoint),
            size * 0.5,
            &probes,
        );
        let topology_tolerance = chord_tolerance(domain, size);
        if !sdf.is_finite() || sdf.abs() > topology_tolerance {
            assessment.score.hard_invalid += 1;
            record_refinement(
                &mut assessment,
                candidate.cells[edge.cell].leaf,
                "exposed cell edge is not owned by the SDF boundary",
                midpoint,
            );
        } else {
            let class = domain
                .classify_boundary(
                    &[Vec3::from_array(midpoint)],
                    BoundaryBand::Custom(topology_tolerance),
                )
                .map_err(|error| MeshError::InvalidInput(error.to_string()))?;
            if !class[0].on_boundary {
                assessment.score.hard_invalid += 1;
                record_refinement(
                    &mut assessment,
                    candidate.cells[edge.cell].leaf,
                    "SDF boundary ownership classification failed",
                    midpoint,
                );
            }
            assessment.boundary[edge_index].owner = class[0].region_name.clone();
        }
        let projection = domain
            .project_to_boundary(&[Vec3::from_array(midpoint)])
            .map_err(|error| MeshError::InvalidInput(error.to_string()))?[0];
        let projection_error = if projection.converged {
            (projection.point - Vec3::from_array(midpoint)).length()
        } else {
            f64::INFINITY
        };
        let residual = sdf.abs().max(projection_error);
        let tolerance = boundary_tolerance(domain, target);
        if residual > tolerance {
            assessment.score.unmet_targets += 1;
            assessment.violations.push(Violation {
                severity: residual / tolerance - 1.0,
                entity: Entity::Edge(
                    ordered_pair(edge.points[0], edge.points[1]).0,
                    ordered_pair(edge.points[0], edge.points[1]).1,
                ),
            });
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
                assessment.score.hard_invalid += 1;
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
    assessment.violations.sort_by(|a, b| {
        b.severity
            .total_cmp(&a.severity)
            .then_with(|| a.entity.cmp(&b.entity))
    });
    assessment.score.worst_violation = assessment
        .violations
        .first()
        .map_or(0.0, |violation| violation.severity);
    assessment.score.worst_quality = assessment.worst_quality;
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

fn prepare_layer_boundaries(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &mut Assessment,
) -> MeshResult<()> {
    for pass in 0..128 {
        context.check()?;
        let mut split = None;
        for (edge_index, edge) in assessment.boundary.iter().enumerate() {
            if edge_index.is_multiple_of(128) {
                context.check()?;
            }
            let a = candidate.points[&edge.points[0]].world;
            let b = candidate.points[&edge.points[1]].world;
            let tolerance = chord_tolerance(domain, distance3(a, b));
            let signatures = (1..16)
                .map(|sample| lerp3(a, b, sample as f64 / 16.0))
                .map(|point| layer_memberships(domain, context, point, tolerance))
                .collect::<MeshResult<Vec<_>>>()?;
            if signatures.windows(2).any(|pair| pair[0] != pair[1]) {
                split = Some(edge.points);
                break;
            }
        }
        let Some([a, b]) = split else {
            break;
        };
        if pass == 127 || !apply_split(domain, space, context, candidate, a, b)? {
            return Err(MeshError::InvalidInput(format!(
                "domain {:?} could not split a boundary edge at a boundary-layer region transition",
                domain.name
            )));
        }
        *assessment = assess(domain, space, context, candidate)?;
        if !assessment.refine.is_empty() {
            return Err(MeshError::InvalidInput(format!(
                "domain {:?} produced invalid connectivity while splitting a boundary-layer region transition",
                domain.name
            )));
        }
    }
    for (edge_index, edge) in assessment.boundary.iter().enumerate() {
        if edge_index.is_multiple_of(128) {
            context.check()?;
        }
        let a = candidate.points[&edge.points[0]].world;
        let b = candidate.points[&edge.points[1]].world;
        if !layer_memberships(
            domain,
            context,
            midpoint3(a, b),
            chord_tolerance(domain, distance3(a, b)),
        )?
        .is_empty()
        {
            for key in edge.points {
                candidate
                    .points
                    .get_mut(&key)
                    .expect("boundary point")
                    .protected = true;
            }
        }
    }
    Ok(())
}

fn layer_memberships(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    point: [f64; 3],
    tolerance: f64,
) -> MeshResult<BTreeSet<usize>> {
    let point = Vec3::from_array(point);
    let projection = domain
        .project_to_boundary(&[point])
        .map_err(|error| MeshError::InvalidInput(error.to_string()))?[0];
    let (point, band) = if projection.converged {
        (projection.point, BoundaryBand::ProjectedVertices)
    } else {
        (point, BoundaryBand::Custom(tolerance))
    };
    let names = domain
        .regions_containing(&[point], band)
        .map_err(|error| MeshError::InvalidInput(error.to_string()))?
        .pop()
        .unwrap_or_default()
        .into_iter()
        .collect::<BTreeSet<_>>();
    Ok(context
        .controls
        .boundary_layers
        .iter()
        .enumerate()
        .filter(|(_, control)| {
            control.domain == domain.name && names.contains(&control.boundary_region)
        })
        .map(|(index, _)| index)
        .collect())
}

fn apply_boundary_layers(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &mut Assessment,
) -> MeshResult<()> {
    let controls = &context.controls.boundary_layers;
    let mut matched = vec![false; controls.len()];
    let mut groups = BTreeMap::<LayerKey, Vec<BoundaryEdge>>::new();
    let mut vertex_specs = BTreeMap::<PointKey, LayerKey>::new();
    for (edge_index, edge) in assessment.boundary.iter().enumerate() {
        if edge_index.is_multiple_of(128) {
            context.check()?;
        }
        let a = candidate.points[&edge.points[0]].world;
        let b = candidate.points[&edge.points[1]].world;
        let memberships = layer_memberships(
            domain,
            context,
            midpoint3(a, b),
            chord_tolerance(domain, distance3(a, b)),
        )?;
        let mut keys = memberships
            .iter()
            .map(|index| LayerKey::from_control(&controls[*index]))
            .collect::<BTreeSet<_>>();
        if keys.len() > 1 {
            return Err(MeshError::InvalidInput(format!(
                "domain {:?} has overlapping boundary-layer controls with incompatible layer count, first height, or growth",
                domain.name
            )));
        }
        let Some(key) = keys.pop_first() else {
            continue;
        };
        for index in memberships {
            matched[index] = true;
        }
        for point in edge.points {
            if vertex_specs
                .insert(point, key)
                .is_some_and(|other| other != key)
            {
                return Err(MeshError::InvalidInput(format!(
                    "domain {:?} has adjacent boundary-layer controls with incompatible layer count, first height, or growth",
                    domain.name
                )));
            }
        }
        groups.entry(key).or_default().push(edge.clone());
    }
    for (index, control) in controls.iter().enumerate() {
        if control.domain == domain.name && !matched[index] {
            return Err(MeshError::InvalidInput(format!(
                "boundary-layer control for domain {:?} matched no boundary edges in region {:?}",
                domain.name, control.boundary_region
            )));
        }
    }
    let added_cells = groups
        .iter()
        .map(|(key, edges)| key.layers.saturating_mul(edges.len()))
        .sum::<usize>();
    if candidate.cells.len().saturating_add(added_cells)
        > usize::try_from(context.limits.max_cells).unwrap_or(usize::MAX)
    {
        return Err(MeshError::LimitExceeded(format!(
            "requested boundary layers exceed the configured {} cell limit",
            context.limits.max_cells
        )));
    }

    for (key, edges) in groups {
        let mut degree = BTreeMap::<PointKey, usize>::new();
        let mut directions = BTreeMap::<PointKey, [f64; 2]>::new();
        for edge in &edges {
            let a = candidate.points[&edge.points[0]].uv;
            let b = candidate.points[&edge.points[1]].uv;
            let delta = [b[0] - a[0], b[1] - a[1]];
            let length = delta[0].hypot(delta[1]);
            if length <= f64::EPSILON {
                return Err(layer_error(
                    domain,
                    "contains a zero-length controlled boundary edge",
                ));
            }
            let inward = [-delta[1] / length, delta[0] / length];
            for point in edge.points {
                *degree.entry(point).or_default() += 1;
                let sum = directions.entry(point).or_default();
                sum[0] += inward[0];
                sum[1] += inward[1];
            }
        }
        if degree.values().any(|count| *count > 2) {
            return Err(layer_error(
                domain,
                "controlled boundary edges do not form manifold open or closed patches",
            ));
        }
        let endpoints = degree
            .iter()
            .filter_map(|(point, count)| (*count == 1).then_some(*point))
            .collect::<BTreeSet<_>>();
        let original = degree
            .keys()
            .map(|point| (*point, candidate.points[point]))
            .collect::<BTreeMap<_, _>>();
        let mut distances = Vec::with_capacity(key.layers + 1);
        distances.push(0.0);
        let mut height = key.first_height();
        for _ in 0..key.layers {
            distances.push(distances.last().copied().unwrap_or(0.0) + height);
            height *= key.growth();
        }
        deform_core_away_from_layer(
            domain,
            space,
            context,
            candidate,
            &edges,
            *distances.last().expect("layer total height"),
        )?;
        let mut rows = BTreeMap::<(PointKey, usize), PointKey>::new();
        for (&point, source) in &original {
            if endpoints.contains(&point) {
                candidate
                    .points
                    .get_mut(&point)
                    .expect("layer endpoint")
                    .protected = true;
                for row in 0..=key.layers {
                    rows.insert((point, row), point);
                }
                continue;
            }
            let direction = directions[&point];
            let direction_length = direction[0].hypot(direction[1]);
            if direction_length <= 1.0e-12 {
                return Err(layer_error(
                    domain,
                    "has an undefined inward normal at a sharp corner",
                ));
            }
            let direction = [
                direction[0] / direction_length,
                direction[1] / direction_length,
            ];
            for (row, &distance) in distances.iter().enumerate() {
                let position = if row == 0 {
                    *source
                } else {
                    layer_point(domain, space, *source, direction, distance)?
                };
                if row == key.layers {
                    candidate.points.insert(
                        point,
                        Point {
                            boundary: false,
                            protected: true,
                            ..position
                        },
                    );
                    rows.insert((point, row), point);
                } else {
                    let row_key = PointKey::Inserted(candidate.next_inserted);
                    candidate.next_inserted += 1;
                    candidate.points.insert(
                        row_key,
                        Point {
                            boundary: row == 0,
                            protected: true,
                            ..position
                        },
                    );
                    rows.insert((point, row), row_key);
                }
            }
        }
        for &point in original.keys() {
            for row in 0..key.layers {
                let a = rows[&(point, row)];
                let b = rows[&(point, row + 1)];
                if a != b {
                    candidate
                        .layer_edge_targets
                        .insert(ordered_pair(a, b), distances[row + 1] - distances[row]);
                }
            }
        }
        for (edge_index, edge) in edges.into_iter().enumerate() {
            if edge_index.is_multiple_of(128) {
                context.check()?;
            }
            for row in 0..key.layers {
                let mut polygon = vec![
                    rows[&(edge.points[0], row)],
                    rows[&(edge.points[1], row)],
                    rows[&(edge.points[1], row + 1)],
                    rows[&(edge.points[0], row + 1)],
                ];
                polygon.dedup();
                if polygon.first() == polygon.last() {
                    polygon.pop();
                }
                if polygon.len() < 3 {
                    return Err(layer_error(domain, "collapsed at an open-patch endpoint"));
                }
                if signed_area_polygon(&polygon, &candidate.points) < 0.0 {
                    polygon.reverse();
                }
                let leaf = candidate.cells[edge.cell].leaf;
                let cell = match polygon.as_slice() {
                    [a, b, c] => {
                        let mut cell = Cell::triangle([*a, *b, *c], leaf);
                        cell.protected = true;
                        cell
                    }
                    [a, b, c, d] => Cell::quad([*a, *b, *c, *d], leaf, true),
                    _ => unreachable!("layer strip cells are triangles or quads"),
                };
                candidate.cells.push(cell);
            }
        }
    }

    if layer_edges_cross(candidate) {
        return Err(layer_error(
            domain,
            "rows self-intersect or cross at a concave boundary feature",
        ));
    }
    *assessment = assess(domain, space, context, candidate)?;
    if !assessment.refine.is_empty() || assessment.score.hard_invalid != 0 {
        let location = assessment
            .location
            .unwrap_or_else(|| domain.bounds.center().to_array());
        return Err(layer_error(
            domain,
            &format!(
                "{} near ({:.6}, {:.6}, {:.6}); worst Scaled Jacobian={:.6e}",
                assessment
                    .reason
                    .as_deref()
                    .unwrap_or("produced invalid boundary-layer connectivity"),
                location[0],
                location[1],
                location[2],
                assessment.worst_quality,
            ),
        ));
    }
    Ok(())
}

fn layer_edges_cross(candidate: &Candidate) -> bool {
    let mut edges = BTreeMap::<(PointKey, PointKey), ([f64; 2], [f64; 2], bool)>::new();
    for cell in &candidate.cells {
        for index in 0..cell.points.len() {
            let a = cell.points[index];
            let b = cell.points[(index + 1) % cell.points.len()];
            edges
                .entry(ordered_pair(a, b))
                .and_modify(|entry| entry.2 |= cell.protected)
                .or_insert((
                    candidate.points[&a].uv,
                    candidate.points[&b].uv,
                    cell.protected,
                ));
        }
    }
    let mut edges = edges.into_iter().collect::<Vec<_>>();
    edges.sort_by(|first, second| {
        first.1 .0[0]
            .min(first.1 .1[0])
            .total_cmp(&second.1 .0[0].min(second.1 .1[0]))
            .then_with(|| first.0.cmp(&second.0))
    });
    for first in 0..edges.len() {
        let ((a_key, b_key), (a, b, a_layer)) = edges[first];
        for ((c_key, d_key), (c, d, c_layer)) in edges.iter().skip(first + 1).copied() {
            if c[0].min(d[0]) > a[0].max(b[0]) {
                break;
            }
            if !a_layer && !c_layer {
                continue;
            }
            if a_key == c_key || a_key == d_key || b_key == c_key || b_key == d_key {
                continue;
            }
            if c[0].max(d[0]) < a[0].min(b[0])
                || a[1].max(b[1]) < c[1].min(d[1])
                || c[1].max(d[1]) < a[1].min(b[1])
            {
                continue;
            }
            if segments_cross(a, b, c, d) {
                return true;
            }
        }
    }
    false
}

fn deform_core_away_from_layer(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    edges: &[BoundaryEdge],
    total_height: f64,
) -> MeshResult<()> {
    let segments = edges
        .iter()
        .map(|edge| {
            (
                Vec3::from_array(candidate.points[&edge.points[0]].world),
                Vec3::from_array(candidate.points[&edge.points[1]].world),
            )
        })
        .collect::<Vec<_>>();
    let influence = total_height + context.element_max_size.max(total_height) * 2.0;
    let movable = candidate
        .points
        .iter()
        .filter_map(|(key, point)| (!point.boundary && !point.protected).then_some((*key, *point)))
        .collect::<Vec<_>>();
    for (index, (key, old)) in movable.into_iter().enumerate() {
        if index.is_multiple_of(512) {
            context.check()?;
        }
        let world = Vec3::from_array(old.world);
        let distance = segments
            .iter()
            .map(|(a, b)| point_segment_distance(world, *a, *b))
            .fold(f64::INFINITY, f64::min);
        if distance >= influence {
            continue;
        }
        let offset = total_height * (1.0 - distance / influence);
        let old_sdf = domain.domain_sdf(&[world])[0];
        if !old_sdf.is_finite() || old_sdf >= 0.0 {
            continue;
        }
        let target = old_sdf - offset;
        let mut moved = world;
        let tolerance = root_tolerance(domain, offset).max(offset * 1.0e-6);
        for _ in 0..16 {
            let residual = domain.domain_sdf(&[moved])[0] - target;
            if residual.abs() <= tolerance {
                break;
            }
            let normal = domain.normals(&[moved])[0];
            if normal.length() <= f64::EPSILON {
                return Err(layer_error(
                    domain,
                    "has an undefined SDF normal in the core",
                ));
            }
            moved = moved - normal * residual;
            let coords = space.coords(moved);
            moved = space.point(coords[0], coords[1]);
        }
        if (domain.domain_sdf(&[moved])[0] - target).abs() > tolerance {
            continue;
        }
        let coords = space.coords(moved);
        candidate.points.insert(
            key,
            Point {
                uv: [coords[0], coords[1]],
                world: moved.to_array(),
                ..old
            },
        );
    }
    Ok(())
}

fn layer_point(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    boundary: Point,
    inward: [f64; 2],
    distance: f64,
) -> MeshResult<Point> {
    let mut point = space.point(
        boundary.uv[0] + inward[0] * distance,
        boundary.uv[1] + inward[1] * distance,
    );
    let tolerance = root_tolerance(domain, distance).max(distance * 1.0e-6);
    for _ in 0..16 {
        let sdf = domain.domain_sdf(&[point])[0];
        if !sdf.is_finite() {
            return Err(layer_error(domain, "encountered a non-finite SDF value"));
        }
        let residual = sdf + distance;
        if residual.abs() <= tolerance {
            break;
        }
        let normal = domain.normals(&[point])[0];
        if normal.length() <= f64::EPSILON {
            return Err(layer_error(domain, "has an undefined SDF normal"));
        }
        point = point - normal * residual;
        let coords = space.coords(point);
        point = space.point(coords[0], coords[1]);
    }
    let residual = (domain.domain_sdf(&[point])[0] + distance).abs();
    let displacement = (point - Vec3::from_array(boundary.world)).length();
    if residual > tolerance
        || displacement > distance * 2.0 + tolerance
        || domain.domain_sdf(&[point])[0] >= 0.0
    {
        return Err(layer_error(
            domain,
            "crosses another boundary or consumes a narrow feature",
        ));
    }
    let coords = space.coords(point);
    Ok(Point {
        uv: [coords[0], coords[1]],
        world: point.to_array(),
        boundary: false,
        protected: true,
    })
}

fn layer_error(domain: &MeshableDomain, reason: &str) -> MeshError {
    MeshError::InvalidInput(format!(
        "domain {:?} rejected the requested boundary layers: {reason}",
        domain.name
    ))
}

fn point_segment_distance(point: Vec3, a: Vec3, b: Vec3) -> f64 {
    let ab = b - a;
    let denominator = ab.dot(ab);
    if denominator <= f64::EPSILON {
        return (point - a).length();
    }
    let t = ((point - a).dot(ab) / denominator).clamp(0.0, 1.0);
    (point - (a + ab * t)).length()
}

fn optimize(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &mut Assessment,
) -> MeshResult<()> {
    let seed_cells = candidate.cells.len().max(1);
    let budget = action_budget(seed_cells, context.limits.max_cells);
    let mut trials = 0usize;
    let mut exhausted = BTreeSet::new();
    while trials < budget {
        if trials.is_multiple_of(64) {
            context.check()?;
        }
        let Some(violation) = assessment
            .violations
            .iter()
            .find(|violation| !exhausted.contains(&violation.entity))
            .copied()
        else {
            break;
        };
        let actions = actions_for(candidate, violation.entity);
        let mut best = None::<(Action, Candidate, Assessment)>;
        for action in actions {
            if trials >= budget {
                break;
            }
            trials += 1;
            let mut trial = candidate.clone();
            if !apply_action(domain, space, context, &mut trial, action)? {
                continue;
            }
            let trial_assessment = assess(domain, space, context, &trial)?;
            if !trial_assessment.refine.is_empty()
                || !boundary_ownership_preserved(action, candidate, assessment, &trial_assessment)
                || compare_scores(&trial_assessment.score, &assessment.score) != Ordering::Less
            {
                continue;
            }
            let replace = best
                .as_ref()
                .is_none_or(|(best_action, _, best_assessment)| {
                    compare_scores(&trial_assessment.score, &best_assessment.score)
                        .then_with(|| action.cmp(best_action))
                        == Ordering::Less
                });
            if replace {
                best = Some((action, trial, trial_assessment));
            }
        }
        if let Some((_, improved, improved_assessment)) = best {
            *candidate = improved;
            *assessment = improved_assessment;
            exhausted.clear();
        } else {
            exhausted.insert(violation.entity);
        }
    }
    merge_triangles(domain, space, context, candidate, assessment)
}

fn merge_triangles(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &mut Assessment,
) -> MeshResult<()> {
    let mut incidence = BTreeMap::<(PointKey, PointKey), Vec<usize>>::new();
    for (cell_index, cell) in candidate.cells.iter().enumerate() {
        if cell.points.len() != 3 || cell.protected {
            continue;
        }
        for edge in 0..3 {
            incidence
                .entry(ordered_pair(cell.points[edge], cell.points[(edge + 1) % 3]))
                .or_default()
                .push(cell_index);
        }
    }
    let mut used = BTreeSet::new();
    let mut replacements = BTreeMap::new();
    let mut removed = BTreeSet::new();
    for (trial_index, (edge, pair)) in incidence.into_iter().enumerate() {
        if trial_index.is_multiple_of(64) {
            context.check()?;
        }
        if pair.len() != 2 || pair.iter().any(|index| used.contains(index)) {
            continue;
        }
        let a = candidate.points[&edge.0].world;
        let b = candidate.points[&edge.1].world;
        let midpoint = midpoint3(a, b);
        let length = distance3(a, b);
        let probes = [
            Vec3::from_array(a),
            Vec3::from_array(b),
            Vec3::from_array(midpoint),
        ];
        let target = regional_target(
            context,
            &domain.name,
            Vec3::from_array(midpoint),
            length * 0.5,
            &probes,
        );
        let diagonal_error = (length / target).max(f64::MIN_POSITIVE).ln().powi(2);
        if diagonal_error + 1.0e-12 < assessment.score.mean_squared_log_size_error {
            continue;
        }
        let [first, second] = [pair[0], pair[1]];
        let Some(quad) = merged_cell(candidate, first, second, edge.0, edge.1) else {
            continue;
        };
        replacements.insert(first, quad);
        removed.insert(second);
        used.insert(first);
        used.insert(second);
    }
    if replacements.is_empty() {
        return Ok(());
    }
    let mut trial = candidate.clone();
    trial.cells = candidate
        .cells
        .iter()
        .enumerate()
        .filter_map(|(index, cell)| {
            if let Some(replacement) = replacements.get(&index) {
                Some(replacement.clone())
            } else {
                (!removed.contains(&index)).then_some(cell.clone())
            }
        })
        .collect();
    let trial_assessment = assess(domain, space, context, &trial)?;
    if trial_assessment.refine.is_empty()
        && boundary_owners(assessment) == boundary_owners(&trial_assessment)
        && compare_scores(&trial_assessment.score, &assessment.score) == Ordering::Less
    {
        *candidate = trial;
        *assessment = trial_assessment;
    }
    Ok(())
}

fn merged_cell(
    candidate: &Candidate,
    first: usize,
    second: usize,
    a: PointKey,
    b: PointKey,
) -> Option<Cell> {
    if candidate.cells[first].protected
        || candidate.cells[second].protected
        || candidate.cells[first].points.len() != 3
        || candidate.cells[second].points.len() != 3
    {
        return None;
    }
    let c = candidate.cells[first]
        .points
        .iter()
        .copied()
        .find(|point| *point != a && *point != b)?;
    let d = candidate.cells[second]
        .points
        .iter()
        .copied()
        .find(|point| *point != a && *point != b)?;
    if c == d {
        return None;
    }
    let mut quad = vec![c, a, d, b];
    if signed_area_polygon(&quad, &candidate.points) < 0.0 {
        quad.reverse();
    }
    if !polygon_is_strictly_convex(&quad, &candidate.points)
        || polygon_self_intersects(&quad, &candidate.points)
    {
        return None;
    }
    let triangle_quality = [first, second]
        .into_iter()
        .map(|index| {
            let positions = candidate.cells[index]
                .points
                .iter()
                .map(|key| candidate.points[key].world)
                .collect::<Vec<_>>();
            quality_score("tri3", &positions, QualityMetric::ScaledJacobian).unwrap_or(0.0)
        })
        .fold(1.0, f64::min);
    let positions = quad
        .iter()
        .map(|key| candidate.points[key].world)
        .collect::<Vec<_>>();
    let quad_quality =
        quality_score("quad4", &positions, QualityMetric::ScaledJacobian).unwrap_or(0.0);
    if quad_quality + 1.0e-12 < triangle_quality || quad_quality <= VALID_QUALITY {
        return None;
    }
    Some(Cell::quad(
        quad.try_into().expect("four quad points"),
        candidate.cells[first].leaf,
        false,
    ))
}

fn action_budget(seed_cells: usize, max_cells: u64) -> usize {
    let cell_limit = usize::try_from(max_cells).unwrap_or(usize::MAX);
    let headroom = cell_limit.saturating_sub(seed_cells);
    seed_cells
        .saturating_mul(2)
        .saturating_add(headroom.min(seed_cells))
        .clamp(32, 64)
}

fn compare_scores(a: &PatchScore, b: &PatchScore) -> Ordering {
    a.hard_invalid
        .cmp(&b.hard_invalid)
        .then_with(|| a.unmet_targets.cmp(&b.unmet_targets))
        .then_with(|| a.worst_violation.total_cmp(&b.worst_violation))
        .then_with(|| b.worst_quality.total_cmp(&a.worst_quality))
        .then_with(|| {
            a.mean_squared_log_size_error
                .total_cmp(&b.mean_squared_log_size_error)
        })
        .then_with(|| b.quad_count.cmp(&a.quad_count))
        .then_with(|| a.triangle_count.cmp(&b.triangle_count))
        .then_with(|| a.cell_count.cmp(&b.cell_count))
}

fn actions_for(candidate: &Candidate, entity: Entity) -> BTreeSet<Action> {
    let mut actions = BTreeSet::new();
    match entity {
        Entity::Cell(index) if index < candidate.cells.len() => {
            let cell = &candidate.cells[index];
            if cell.protected {
                return actions;
            }
            if cell.points.len() == 4 {
                actions.insert(Action::SplitQuad(index));
            } else {
                actions.insert(Action::Insert(index));
            }
            let points = &cell.points;
            for edge in 0..points.len() {
                add_edge_actions(
                    candidate,
                    &mut actions,
                    points[edge],
                    points[(edge + 1) % points.len()],
                );
            }
        }
        Entity::Edge(a, b) => {
            add_edge_actions(candidate, &mut actions, a, b);
            for index in edge_cells(candidate, a, b) {
                actions.insert(Action::Insert(index));
            }
        }
        Entity::Cell(_) => {}
    }
    actions
}

fn add_edge_actions(
    candidate: &Candidate,
    actions: &mut BTreeSet<Action>,
    a: PointKey,
    b: PointKey,
) {
    let (a, b) = ordered_pair(a, b);
    if candidate.points[&a].protected || candidate.points[&b].protected {
        return;
    }
    actions.insert(Action::Merge(a, b));
    actions.insert(Action::Flip(a, b));
    actions.insert(Action::Split(a, b));
    actions.insert(Action::Collapse(a, b));
    add_point_actions(candidate, actions, a);
    add_point_actions(candidate, actions, b);
}

fn add_point_actions(candidate: &Candidate, actions: &mut BTreeSet<Action>, key: PointKey) {
    if candidate.points[&key].protected {
        return;
    }
    if candidate.points[&key].boundary {
        for step in 0..5 {
            actions.insert(Action::RelocateBoundary(key, step));
        }
    } else {
        for step in 0..4 {
            actions.insert(Action::RelocateInterior(key, step));
        }
    }
}

fn apply_action(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    action: Action,
) -> MeshResult<bool> {
    match action {
        Action::Merge(a, b) => Ok(apply_merge(candidate, a, b)),
        Action::SplitQuad(index) => Ok(apply_split_quad(candidate, index)),
        Action::Flip(a, b) => Ok(apply_flip(candidate, a, b)),
        Action::RelocateInterior(key, step) => {
            apply_interior_relocation(domain, space, context, candidate, key, step)
        }
        Action::RelocateBoundary(key, step) => {
            apply_boundary_relocation(domain, space, context, candidate, key, step)
        }
        Action::Split(a, b) => apply_split(domain, space, context, candidate, a, b),
        Action::Insert(index) => Ok(apply_insert(domain, space, context, candidate, index)),
        Action::Collapse(a, b) => apply_collapse(domain, space, candidate, a, b),
    }
}

fn apply_merge(candidate: &mut Candidate, a: PointKey, b: PointKey) -> bool {
    let pair = edge_cells(candidate, a, b);
    if pair.len() != 2 {
        return false;
    }
    let first = pair[0];
    let second = pair[1];
    let Some(quad) = merged_cell(candidate, first, second, a, b) else {
        return false;
    };
    candidate.cells[first] = quad;
    candidate.cells.remove(second);
    true
}

fn apply_split_quad(candidate: &mut Candidate, index: usize) -> bool {
    let Some(cell) = candidate
        .cells
        .get(index)
        .filter(|cell| cell.points.len() == 4 && !cell.protected)
        .cloned()
    else {
        return false;
    };
    let p = &cell.points;
    let mut first = [[p[0], p[1], p[2]], [p[0], p[2], p[3]]];
    let mut second = [[p[0], p[1], p[3]], [p[1], p[2], p[3]]];
    for triangle in first.iter_mut().chain(second.iter_mut()) {
        orient_triangle(triangle, &candidate.points);
    }
    let chosen =
        if pair_quality(first, &candidate.points) >= pair_quality(second, &candidate.points) {
            first
        } else {
            second
        };
    candidate.cells[index] = Cell::triangle(chosen[0], cell.leaf);
    candidate.cells.push(Cell::triangle(chosen[1], cell.leaf));
    true
}

fn apply_flip(candidate: &mut Candidate, a: PointKey, b: PointKey) -> bool {
    let pair = edge_cells(candidate, a, b);
    if pair.len() != 2
        || pair.iter().any(|index| {
            candidate.cells[*index].protected || candidate.cells[*index].points.len() != 3
        })
    {
        return false;
    }
    let first = pair[0];
    let second = pair[1];
    let Some(c) = candidate.cells[first]
        .points
        .iter()
        .copied()
        .find(|point| *point != a && *point != b)
    else {
        return false;
    };
    let Some(d) = candidate.cells[second]
        .points
        .iter()
        .copied()
        .find(|point| *point != a && *point != b)
    else {
        return false;
    };
    if c == d || !edge_cells(candidate, c, d).is_empty() {
        return false;
    }
    let mut replacements = [[c, d, a], [d, c, b]];
    orient_triangle(&mut replacements[0], &candidate.points);
    orient_triangle(&mut replacements[1], &candidate.points);
    candidate.cells[first].points = replacements[0].into();
    candidate.cells[second].points = replacements[1].into();
    true
}

fn apply_interior_relocation(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    key: PointKey,
    step: u8,
) -> MeshResult<bool> {
    let Some(old) = candidate
        .points
        .get(&key)
        .copied()
        .filter(|point| !point.boundary && !point.protected)
    else {
        return Ok(false);
    };
    let incident = incident_cells(context, candidate, key)?;
    let neighbors = incident
        .iter()
        .flat_map(|index| candidate.cells[*index].points.iter().copied())
        .filter(|point| *point != key)
        .collect::<BTreeSet<_>>();
    if neighbors.len() < 3 {
        return Ok(false);
    }
    let target = neighbors
        .iter()
        .map(|neighbor| candidate.points[neighbor].uv)
        .fold([0.0; 2], |sum, uv| [sum[0] + uv[0], sum[1] + uv[1]])
        .map(|value| value / neighbors.len() as f64);
    let Some(fraction) = [1.0, 0.75, 0.5, 0.25].get(step as usize) else {
        return Ok(false);
    };
    let uv = [
        old.uv[0] + fraction * (target[0] - old.uv[0]),
        old.uv[1] + fraction * (target[1] - old.uv[1]),
    ];
    let world = space.point(uv[0], uv[1]);
    if domain.domain_sdf(&[world])[0] >= 0.0 {
        return Ok(false);
    }
    candidate.points.insert(
        key,
        Point {
            uv,
            world: world.to_array(),
            boundary: false,
            protected: old.protected,
        },
    );
    Ok(true)
}

fn apply_boundary_relocation(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    key: PointKey,
    step: u8,
) -> MeshResult<bool> {
    let Some(old) = candidate
        .points
        .get(&key)
        .copied()
        .filter(|point| point.boundary && !point.protected)
    else {
        return Ok(false);
    };
    let incident = incident_cells(context, candidate, key)?;
    let mut neighbors = BTreeSet::new();
    for (index, cell) in candidate.cells.iter().enumerate() {
        for edge in 0..cell.points.len() {
            let points = [
                cell.points[edge],
                cell.points[(edge + 1) % cell.points.len()],
            ];
            if points.contains(&key)
                && edge_cells(candidate, points[0], points[1]).as_slice() == [index]
            {
                neighbors.insert(if points[0] == key {
                    points[1]
                } else {
                    points[0]
                });
            }
        }
    }
    if neighbors.len() != 2 || incident.is_empty() {
        return Ok(false);
    }
    let mut iter = neighbors.into_iter();
    let a = candidate.points[&iter.next().expect("two boundary neighbors")];
    let b = candidate.points[&iter.next().expect("two boundary neighbors")];
    let target = [(a.uv[0] + b.uv[0]) * 0.5, (a.uv[1] + b.uv[1]) * 0.5];
    let Some(fraction) = [1.0, 0.75, 0.5, 0.25, 0.125].get(step as usize) else {
        return Ok(false);
    };
    let uv = [
        old.uv[0] + fraction * (target[0] - old.uv[0]),
        old.uv[1] + fraction * (target[1] - old.uv[1]),
    ];
    let boundary_world = space.point(uv[0], uv[1]);
    let center = incident
        .iter()
        .map(|index| Vec3::from_array(cell_centroid(candidate, *index)))
        .fold(Vec3::ZERO, |sum, point| sum + point)
        / incident.len() as f64;
    let seed = (1..=16)
        .map(|index| index as f64 / 16.0)
        .map(|weight| boundary_world + (center - boundary_world) * weight)
        .find(|point| domain.domain_sdf(&[*point])[0] < 0.0);
    let Some(seed) = seed else {
        return Ok(false);
    };
    let projection = domain
        .project_to_boundary(&[seed])
        .map_err(|error| MeshError::InvalidInput(error.to_string()))?[0];
    if !projection.converged {
        return Ok(false);
    }
    let coords = space.coords(projection.point);
    candidate.points.insert(
        key,
        Point {
            uv: [coords[0], coords[1]],
            world: projection.point.to_array(),
            boundary: true,
            protected: old.protected,
        },
    );
    Ok(true)
}

fn apply_insert(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    cell_index: usize,
) -> bool {
    if cell_index >= candidate.cells.len()
        || candidate.cells[cell_index].protected
        || candidate.cells[cell_index].points.len() != 3
        || candidate.cells.len().saturating_add(2)
            > usize::try_from(context.limits.max_cells).unwrap_or(usize::MAX)
    {
        return false;
    }
    let cell = candidate.cells[cell_index].clone();
    let points = cell
        .points
        .iter()
        .map(|key| candidate.points[key])
        .collect::<Vec<_>>();
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
    let key = PointKey::Inserted(candidate.next_inserted);
    candidate.next_inserted += 1;
    candidate.points.insert(
        key,
        Point {
            uv,
            world: world.to_array(),
            boundary: false,
            protected: false,
        },
    );
    let replacements = [
        Cell::triangle([cell.points[0], cell.points[1], key], cell.leaf),
        Cell::triangle([cell.points[1], cell.points[2], key], cell.leaf),
        Cell::triangle([cell.points[2], cell.points[0], key], cell.leaf),
    ];
    candidate.cells[cell_index] = replacements[0].clone();
    candidate.cells.push(replacements[1].clone());
    candidate.cells.push(replacements[2].clone());
    true
}

fn apply_split(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    a: PointKey,
    b: PointKey,
) -> MeshResult<bool> {
    let pair = edge_cells(candidate, a, b);
    if pair.is_empty()
        || pair.len() > 2
        || pair.iter().any(|index| {
            candidate.cells[*index].protected || candidate.cells[*index].points.len() != 3
        })
        || candidate.cells.len().saturating_add(pair.len())
            > usize::try_from(context.limits.max_cells).unwrap_or(usize::MAX)
    {
        return Ok(false);
    }
    let boundary = pair.len() == 1;
    if boundary && (!candidate.points[&a].boundary || !candidate.points[&b].boundary) {
        return Ok(false);
    }
    let mut world = Vec3::from_array(midpoint3(
        candidate.points[&a].world,
        candidate.points[&b].world,
    ));
    if boundary {
        let projection = domain
            .project_to_boundary(&[world])
            .map_err(|error| MeshError::InvalidInput(error.to_string()))?[0];
        if !projection.converged {
            return Ok(false);
        }
        world = projection.point;
    }
    let coords = space.coords(world);
    let key = PointKey::Inserted(candidate.next_inserted);
    candidate.next_inserted += 1;
    candidate.points.insert(
        key,
        Point {
            uv: [coords[0], coords[1]],
            world: world.to_array(),
            boundary,
            protected: false,
        },
    );
    for index in pair {
        let cell = candidate.cells[index].clone();
        let Some(c) = cell
            .points
            .iter()
            .copied()
            .find(|point| *point != a && *point != b)
        else {
            return Ok(false);
        };
        let mut first = [a, key, c];
        let mut second = [key, b, c];
        orient_triangle(&mut first, &candidate.points);
        orient_triangle(&mut second, &candidate.points);
        candidate.cells[index].points = first.into();
        candidate.cells.push(Cell::triangle(second, cell.leaf));
    }
    Ok(true)
}

fn apply_collapse(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    candidate: &mut Candidate,
    a: PointKey,
    b: PointKey,
) -> MeshResult<bool> {
    let pair = edge_cells(candidate, a, b);
    if candidate.points[&a].protected
        || candidate.points[&b].protected
        || candidate
            .cells
            .iter()
            .filter(|cell| cell.points.contains(&a) || cell.points.contains(&b))
            .any(|cell| cell.protected || cell.points.len() != 3)
    {
        return Ok(false);
    }
    let a_boundary = candidate.points[&a].boundary;
    let b_boundary = candidate.points[&b].boundary;
    let boundary_edge = a_boundary && b_boundary;
    if (boundary_edge && pair.len() != 1) || (!boundary_edge && pair.len() != 2) {
        return Ok(false);
    }
    let (keep, remove) = match (a_boundary, b_boundary) {
        (true, false) => (a, b),
        (false, true) => (b, a),
        _ => ordered_pair(a, b),
    };
    if boundary_edge {
        let midpoint = Vec3::from_array(midpoint3(
            candidate.points[&a].world,
            candidate.points[&b].world,
        ));
        let projection = domain
            .project_to_boundary(&[midpoint])
            .map_err(|error| MeshError::InvalidInput(error.to_string()))?[0];
        if !projection.converged {
            return Ok(false);
        }
        let coords = space.coords(projection.point);
        candidate.points.insert(
            keep,
            Point {
                uv: [coords[0], coords[1]],
                world: projection.point.to_array(),
                boundary: true,
                protected: false,
            },
        );
    }
    for cell in &mut candidate.cells {
        for point in &mut cell.points {
            if *point == remove {
                *point = keep;
            }
        }
    }
    candidate.cells.retain(|cell| {
        cell.points.iter().copied().collect::<BTreeSet<_>>().len() == cell.points.len()
    });
    candidate.points.remove(&remove);
    Ok(true)
}

fn boundary_ownership_preserved(
    action: Action,
    before_candidate: &Candidate,
    before: &Assessment,
    after: &Assessment,
) -> bool {
    let before_owners = boundary_owners(before);
    let mut after_owners = boundary_owners(after);
    if let Action::Split(a, b) = action {
        let edge = ordered_pair(a, b);
        if let Some(owner) = before_owners.get(&edge) {
            let inserted = PointKey::Inserted(before_candidate.next_inserted);
            if after_owners.remove(&ordered_pair(a, inserted)) != Some(owner.clone())
                || after_owners.remove(&ordered_pair(inserted, b)) != Some(owner.clone())
            {
                return false;
            }
            let mut unchanged = before_owners;
            unchanged.remove(&edge);
            return unchanged == after_owners;
        }
    } else if let Action::Collapse(a, b) = action {
        let edge = ordered_pair(a, b);
        if before_owners.contains_key(&edge) {
            let keep = edge.0;
            let before_incident = before_owners
                .iter()
                .filter(|((x, y), _)| *x == a || *y == a || *x == b || *y == b)
                .collect::<Vec<_>>();
            let after_incident = after_owners
                .iter()
                .filter(|((x, y), _)| *x == keep || *y == keep)
                .collect::<Vec<_>>();
            let Some(owner) = before_incident.first().map(|(_, owner)| *owner) else {
                return false;
            };
            if before_incident.iter().any(|(_, value)| *value != owner)
                || after_incident.iter().any(|(_, value)| *value != owner)
                || after_incident.len() + 1 != before_incident.len()
            {
                return false;
            }
            let before_unchanged = before_owners
                .into_iter()
                .filter(|((x, y), _)| *x != a && *y != a && *x != b && *y != b)
                .collect::<BTreeMap<_, _>>();
            let after_unchanged = after_owners
                .into_iter()
                .filter(|((x, y), _)| *x != keep && *y != keep)
                .collect::<BTreeMap<_, _>>();
            return before_unchanged == after_unchanged;
        }
    };
    before_owners == after_owners
}

fn boundary_owners(assessment: &Assessment) -> BTreeMap<(PointKey, PointKey), Option<String>> {
    assessment
        .boundary
        .iter()
        .map(|edge| {
            (
                ordered_pair(edge.points[0], edge.points[1]),
                edge.owner.clone(),
            )
        })
        .collect()
}

fn orient_triangle(triangle: &mut [PointKey; 3], points: &BTreeMap<PointKey, Point>) {
    if signed_area(*triangle, points) < 0.0 {
        triangle.swap(1, 2);
    }
}

fn edge_cells(candidate: &Candidate, a: PointKey, b: PointKey) -> Vec<usize> {
    candidate
        .cells
        .iter()
        .enumerate()
        .filter_map(|(index, cell)| {
            (cell.points.contains(&a) && cell.points.contains(&b)).then_some(index)
        })
        .collect()
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

fn cell_containment_residual(domain: &MeshableDomain, points: &[[f64; 3]]) -> f64 {
    domain
        .domain_sdf(&cell_containment_samples(points))
        .into_iter()
        .fold(f64::NEG_INFINITY, |worst, value| {
            if value.is_finite() {
                worst.max(value)
            } else {
                f64::INFINITY
            }
        })
}

fn cell_containment_samples(points: &[[f64; 3]]) -> Vec<Vec3> {
    let mut samples = Vec::with_capacity(points.len() + 1);
    samples.push(Vec3::from_array(centroid_slice(points)));
    samples.extend(
        (0..points.len())
            .map(|edge| midpoint3(points[edge], points[(edge + 1) % points.len()]))
            .map(Vec3::from_array),
    );
    samples
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
        for &point in &cell.points {
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
            .flat_map(|cell| cell.points.iter().copied())
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
            match cell.points.as_slice() {
                [a, b, c] => {
                    builder.tri3([ids[a], ids[b], ids[c]], catalog.zone, catalog.source)?;
                }
                [a, b, c, d] => {
                    builder.quad4(
                        [ids[a], ids[b], ids[c], ids[d]],
                        catalog.zone,
                        catalog.source,
                    )?;
                }
                _ => unreachable!("2D candidate cells are triangles or quads"),
            }
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

fn signed_area_polygon(polygon: &[PointKey], points: &BTreeMap<PointKey, Point>) -> f64 {
    (0..polygon.len())
        .map(|index| {
            let a = points[&polygon[index]].uv;
            let b = points[&polygon[(index + 1) % polygon.len()]].uv;
            a[0] * b[1] - a[1] * b[0]
        })
        .sum::<f64>()
        * 0.5
}

fn polygon_is_strictly_convex(polygon: &[PointKey], points: &BTreeMap<PointKey, Point>) -> bool {
    (0..polygon.len()).all(|index| {
        let a = points[&polygon[index]].uv;
        let b = points[&polygon[(index + 1) % polygon.len()]].uv;
        let c = points[&polygon[(index + 2) % polygon.len()]].uv;
        cross_2d(a, b, c) > 1.0e-14
    })
}

fn polygon_self_intersects(polygon: &[PointKey], points: &BTreeMap<PointKey, Point>) -> bool {
    polygon.len() == 4
        && (segments_cross(
            points[&polygon[0]].uv,
            points[&polygon[1]].uv,
            points[&polygon[2]].uv,
            points[&polygon[3]].uv,
        ) || segments_cross(
            points[&polygon[1]].uv,
            points[&polygon[2]].uv,
            points[&polygon[3]].uv,
            points[&polygon[0]].uv,
        ))
}

fn cross_2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
}

fn segments_cross(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let ab_c = cross_2d(a, b, c);
    let ab_d = cross_2d(a, b, d);
    let cd_a = cross_2d(c, d, a);
    let cd_b = cross_2d(c, d, b);
    ab_c * ab_d < 0.0 && cd_a * cd_b < 0.0
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
        .iter()
        .map(|key| candidate.points[key].world)
        .collect::<Vec<_>>();
    centroid_slice(&positions)
}

fn maximum_edge_2d(points: &[[f64; 3]]) -> f64 {
    (0..points.len())
        .map(|edge| distance3(points[edge], points[(edge + 1) % points.len()]))
        .fold(0.0, f64::max)
}

fn root_tolerance(domain: &MeshableDomain, local_size: f64) -> f64 {
    (domain.bounds.diagonal() * 1.0e-12)
        .max(local_size * 1.0e-6)
        .max(f64::EPSILON * domain.bounds.diagonal() * 64.0)
}

fn chord_tolerance(domain: &MeshableDomain, local_size: f64) -> f64 {
    root_tolerance(domain, local_size).max(local_size * 0.5)
}

fn boundary_tolerance(domain: &MeshableDomain, local_size: f64) -> f64 {
    root_tolerance(domain, local_size).max(local_size * 0.05)
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
        "domain {:?} could not produce a valid adaptive 2D mesh at the {:.6e} element-size floor near ({:.6}, {:.6}, {:.6}): {reason}; worst Scaled Jacobian={quality:.6e}, required > {VALID_QUALITY:.1e} and optimization target {QUALITY_TARGET:.2}",
        domain.name, context.element_min_size, location[0], location[1], location[2],
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

fn lerp3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn centroid_slice(points: &[[f64; 3]]) -> [f64; 3] {
    let mut result = [0.0; 3];
    for point in points {
        for axis in 0..3 {
            result[axis] += point[axis] / points.len() as f64;
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
        run_meshing, ControlRegion, ControlSet, GenerationLimits, JobControl, MemoryArtifact,
        MemoryStorage, MeshArtifact, MeshCatalog, MeshChunk, MeshFile, MeshSink, MeshingContext,
        MeshingRequest,
    };

    #[derive(Default)]
    struct TestSink {
        next: u32,
        cells: usize,
        chunks: Vec<MeshChunk>,
    }

    impl MeshSink for TestSink {
        fn allocate_chunk_id(&mut self) -> MeshResult<u32> {
            self.next += 1;
            Ok(self.next)
        }

        fn emit(&mut self, chunk: MeshChunk) -> MeshResult<()> {
            self.cells += chunk.cells.len();
            self.chunks.push(chunk);
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

    fn controlled_rectangle() -> (SceneDocument, String) {
        let mut document =
            primitive_document("rectangle", vec3(-1.0, -0.75, 0.0), vec3(1.0, 0.75, 0.0));
        let root = document.fluid_domain.as_ref().expect("fluid root").root;
        document
            .add_boundary_region(root, None, None, Some("wall"))
            .expect("whole-boundary region");
        let name = document
            .boundary_regions
            .last()
            .expect("boundary region")
            .name
            .clone();
        (document, name)
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

    fn mesh_chunks(
        document: &SceneDocument,
        element_min_size: f64,
        element_max_size: f64,
        controls: &ControlSet,
        limits: GenerationLimits,
    ) -> MeshResult<TestSink> {
        let domains = meshable_domains_from_document(document)
            .map_err(|error| MeshError::InvalidInput(error.to_string()))?;
        let job_control = JobControl::default();
        let catalog = MeshCatalog::from_domains(&domains, "advancing_front");
        let context = MeshingContext {
            domains: &domains,
            element_min_size,
            element_max_size,
            controls,
            job_control: &job_control,
            limits,
            catalog: &catalog,
        };
        let mut sink = TestSink::default();
        generate(&context, &mut sink)?;
        Ok(sink)
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
    fn common_score_and_action_ties_are_deterministic() {
        let baseline = PatchScore {
            hard_invalid: 0,
            unmet_targets: 2,
            worst_violation: 0.5,
            worst_quality: 0.2,
            mean_squared_log_size_error: 0.4,
            quad_count: 0,
            triangle_count: 4,
            cell_count: 4,
        };
        let improved = PatchScore {
            unmet_targets: 1,
            worst_quality: 0.1,
            ..baseline
        };
        assert_eq!(compare_scores(&improved, &baseline), Ordering::Less);

        let a = PointKey::Inserted(1);
        let b = PointKey::Inserted(2);
        let actions = BTreeSet::from([
            Action::Split(a, b),
            Action::Flip(a, b),
            Action::RelocateInterior(a, 0),
        ]);
        assert_eq!(actions.first(), Some(&Action::Flip(a, b)));
        assert_eq!(action_budget(1, 1), 32);
        assert_eq!(action_budget(10_000, u64::MAX), 64);
    }

    #[test]
    fn regional_target_detects_unsampled_regions_and_grades_outward() {
        let mut controls = ControlSet::default();
        controls
            .refinement(
                "plate",
                ControlRegion::sphere(vec3(0.23, 0.17, 0.0), 0.01).unwrap(),
                0.05,
                10.0,
            )
            .unwrap();
        let probes = [
            vec3(-1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(1.0, 1.0, 0.0),
            vec3(-1.0, 1.0, 0.0),
            vec3(0.0, -1.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(-1.0, 0.0, 0.0),
            vec3(0.0, 0.0, 0.0),
        ];
        assert!(probes
            .iter()
            .all(|point| { (controls.size_at("plate", *point, 1.0) - 1.0).abs() < 1.0e-12 }));
        let detected = regional_target_from_controls(
            &controls,
            "plate",
            vec3(0.0, 0.0, 0.0),
            2.0_f64.sqrt(),
            &probes,
            0.01,
            1.0,
        );
        assert!((detected - 0.05).abs() < 1.0e-12);

        let mut graded = ControlSet::default();
        graded
            .refinement(
                "plate",
                ControlRegion::sphere(vec3(0.0, 0.0, 0.0), 0.1).unwrap(),
                0.05,
                1.0,
            )
            .unwrap();
        let target = |x| {
            regional_target_from_controls(
                &graded,
                "plate",
                vec3(x, 0.0, 0.0),
                0.0,
                &[vec3(x, 0.0, 0.0)],
                0.01,
                1.0,
            )
        };
        assert!(target(0.0) < target(0.3));
        assert!(target(0.3) < target(1.0));
    }

    #[test]
    fn invalid_trials_leave_the_seed_connectivity_unchanged() {
        let a = PointKey::Inserted(1);
        let b = PointKey::Inserted(2);
        let c = PointKey::Inserted(3);
        let points = BTreeMap::from([
            (
                a,
                Point {
                    uv: [0.0, 0.0],
                    world: [0.0, 0.0, 0.0],
                    boundary: false,
                    protected: false,
                },
            ),
            (
                b,
                Point {
                    uv: [1.0, 0.0],
                    world: [1.0, 0.0, 0.0],
                    boundary: false,
                    protected: false,
                },
            ),
            (
                c,
                Point {
                    uv: [0.0, 1.0],
                    world: [0.0, 1.0, 0.0],
                    boundary: false,
                    protected: false,
                },
            ),
        ]);
        let mut trial = Candidate {
            points,
            cells: vec![Cell::triangle(
                [a, b, c],
                Leaf {
                    level: 0,
                    x: 0,
                    y: 0,
                },
            )],
            construction_failures: BTreeSet::new(),
            next_inserted: 4,
            layer_edge_targets: BTreeMap::new(),
        };
        let original = trial.cells[0].points.clone();
        assert!(!apply_flip(&mut trial, a, b));
        assert_eq!(trial.cells[0].points, original);
    }

    #[test]
    fn adjacent_triangles_merge_and_poor_quads_split_deterministically() {
        let keys = [
            PointKey::Inserted(1),
            PointKey::Inserted(2),
            PointKey::Inserted(3),
            PointKey::Inserted(4),
        ];
        let points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            .into_iter()
            .enumerate()
            .map(|(index, uv)| {
                (
                    keys[index],
                    Point {
                        uv,
                        world: [uv[0], uv[1], 0.0],
                        boundary: false,
                        protected: false,
                    },
                )
            })
            .collect();
        let leaf = Leaf {
            level: 0,
            x: 0,
            y: 0,
        };
        let mut candidate = Candidate {
            points,
            cells: vec![
                Cell::triangle([keys[0], keys[1], keys[2]], leaf),
                Cell::triangle([keys[0], keys[2], keys[3]], leaf),
            ],
            construction_failures: BTreeSet::new(),
            next_inserted: 5,
            layer_edge_targets: BTreeMap::new(),
        };
        assert!(apply_merge(&mut candidate, keys[0], keys[2]));
        assert_eq!(candidate.cells.len(), 1);
        assert_eq!(candidate.cells[0].points.len(), 4);
        assert!(apply_split_quad(&mut candidate, 0));
        assert_eq!(
            candidate
                .cells
                .iter()
                .map(|cell| cell.points.len())
                .collect::<Vec<_>>(),
            [3, 3]
        );

        candidate
            .cells
            .push(Cell::triangle([keys[0], keys[2], keys[1]], leaf));
        let before = candidate.cells.clone();
        assert!(!apply_merge(&mut candidate, keys[0], keys[2]));
        assert_eq!(
            candidate
                .cells
                .iter()
                .map(|cell| cell.points.clone())
                .collect::<Vec<_>>(),
            before
                .iter()
                .map(|cell| cell.points.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn feasible_fixture_meets_balanced_targets() {
        let fixtures = [(
            primitive_document("square", vec3(-0.75, -0.75, 0.0), vec3(0.75, 0.75, 0.0)),
            0.25,
        )];
        for (document, target) in fixtures {
            let sink = mesh_chunks(
                &document,
                target,
                target,
                &ControlSet::default(),
                GenerationLimits::default(),
            )
            .unwrap();
            let domains = meshable_domains_from_document(&document).unwrap();
            let domain = domains.iter().next().unwrap();
            let mut quads = 0;
            for chunk in &sink.chunks {
                let points = chunk
                    .points
                    .iter()
                    .map(|point| (point.id, point.position))
                    .collect::<BTreeMap<_, _>>();
                for cell in &chunk.cells {
                    let positions = cell
                        .point_ids
                        .iter()
                        .map(|id| points[id])
                        .collect::<Vec<_>>();
                    quads += usize::from(cell.element_type == "quad4");
                    let quality = quality_score(
                        &cell.element_type,
                        &positions,
                        QualityMetric::ScaledJacobian,
                    )
                    .unwrap();
                    assert!(
                        quality + 1.0e-12 >= QUALITY_TARGET,
                        "{} quality {quality}",
                        domain.name
                    );
                    for edge in 0..positions.len() {
                        let ratio =
                            distance3(positions[edge], positions[(edge + 1) % positions.len()])
                                / target;
                        assert!(
                            ratio + 1.0e-12 >= EDGE_RATIO_MIN && ratio <= EDGE_RATIO_MAX + 1.0e-12,
                            "{} edge ratio {ratio}: {:?} -> {:?}",
                            domain.name,
                            positions[edge],
                            positions[(edge + 1) % 3],
                        );
                    }
                }
                for edge in chunk.edges.iter().filter(|edge| edge.boundary) {
                    let midpoint =
                        midpoint3(points[&edge.point_ids[0]], points[&edge.point_ids[1]]);
                    let residual = domain.domain_sdf(&[Vec3::from_array(midpoint)])[0].abs();
                    assert!(residual <= boundary_tolerance(domain, target));
                }
            }
            assert!(quads > 0, "a rectangular domain should contain quad4 cells");
        }
    }

    #[test]
    fn uniform_and_growing_boundary_layers_emit_protected_quad_rows() {
        for growth in [1.0, 1.5] {
            let (document, region) = controlled_rectangle();
            let mut controls = ControlSet::default();
            controls
                .boundary_layer("rectangle", region, 0.04, 2, growth)
                .unwrap();
            let first =
                mesh_chunks(&document, 0.02, 0.2, &controls, GenerationLimits::default()).unwrap();
            let second =
                mesh_chunks(&document, 0.02, 0.2, &controls, GenerationLimits::default()).unwrap();
            assert_eq!(first.chunks, second.chunks);
            let cells = first
                .chunks
                .iter()
                .flat_map(|chunk| &chunk.cells)
                .collect::<Vec<_>>();
            assert!(cells.iter().any(|cell| cell.element_type == "quad4"));
            assert!(cells
                .iter()
                .all(|cell| { matches!(cell.element_type.as_str(), "tri3" | "quad4") }));
            assert!(first
                .chunks
                .iter()
                .flat_map(|chunk| &chunk.edges)
                .filter(|edge| edge.boundary)
                .all(|edge| !edge.tag_ids.is_empty()));
        }
    }

    #[test]
    fn partial_boundary_layer_tapers_with_valid_transition_triangles() {
        let mut document =
            primitive_document("rectangle", vec3(-1.0, -0.75, 0.0), vec3(1.0, 0.75, 0.0));
        let root = document.fluid_domain.as_ref().expect("fluid root").root;
        document
            .add_boundary_region(root, Some(0), None, Some("inlet"))
            .expect("direction boundary region");
        let region = document
            .boundary_regions
            .last()
            .expect("boundary region")
            .name
            .clone();
        let mut controls = ControlSet::default();
        controls
            .boundary_layer("rectangle", region, 0.035, 3, 1.2)
            .unwrap();
        let sink = mesh_chunks(
            &document,
            0.015,
            0.18,
            &controls,
            GenerationLimits::default(),
        )
        .unwrap();
        let types = sink
            .chunks
            .iter()
            .flat_map(|chunk| &chunk.cells)
            .map(|cell| cell.element_type.as_str())
            .collect::<BTreeSet<_>>();
        assert!(types.contains("tri3"));
        assert!(types.contains("quad4"));
    }

    #[test]
    fn curved_closed_boundary_layers_remain_valid() {
        let mut document = primitive_document("circle", vec3(-0.8, -0.8, 0.0), vec3(0.8, 0.8, 0.0));
        let root = document.fluid_domain.as_ref().expect("fluid root").root;
        document
            .add_boundary_region(root, None, None, Some("wall"))
            .expect("whole-circle region");
        let region = document
            .boundary_regions
            .last()
            .expect("boundary region")
            .name
            .clone();
        let mut controls = ControlSet::default();
        controls
            .boundary_layer("circle", region, 0.025, 2, 1.25)
            .unwrap();
        let sink = mesh_chunks(
            &document,
            0.0125,
            0.12,
            &controls,
            GenerationLimits::default(),
        )
        .unwrap();
        assert!(sink
            .chunks
            .iter()
            .flat_map(|chunk| &chunk.cells)
            .any(|cell| cell.element_type == "quad4"));
    }

    #[test]
    fn incompatible_adjacent_layers_and_excessive_thickness_are_rejected() {
        let mut document =
            primitive_document("rectangle", vec3(-1.0, -0.75, 0.0), vec3(1.0, 0.75, 0.0));
        let root = document.fluid_domain.as_ref().expect("fluid root").root;
        for direction in [0, 2] {
            document
                .add_boundary_region(root, Some(direction), None, Some("wall"))
                .expect("direction boundary region");
        }
        let regions = document
            .boundary_regions
            .iter()
            .map(|region| region.name.clone())
            .collect::<Vec<_>>();
        let mut incompatible = ControlSet::default();
        incompatible
            .boundary_layer("rectangle", &regions[0], 0.03, 2, 1.0)
            .unwrap();
        incompatible
            .boundary_layer("rectangle", &regions[1], 0.04, 3, 1.2)
            .unwrap();
        assert!(matches!(
            mesh_chunks(
                &document,
                0.015,
                0.18,
                &incompatible,
                GenerationLimits::default(),
            ),
            Err(MeshError::InvalidInput(message)) if message.contains("incompatible")
        ));

        let (document, region) = controlled_rectangle();
        let mut excessive = ControlSet::default();
        excessive
            .boundary_layer("rectangle", region, 0.6, 2, 1.0)
            .unwrap();
        assert!(mesh_chunks(
            &document,
            0.02,
            0.2,
            &excessive,
            GenerationLimits::default(),
        )
        .is_err());
    }

    #[test]
    fn local_refinement_adds_local_cells_without_refining_the_whole_domain() {
        let document = primitive_document("rectangle", vec3(-1.0, -1.0, 0.0), vec3(1.0, 1.0, 0.0));
        let background = mesh_chunks(
            &document,
            0.05,
            0.4,
            &ControlSet::default(),
            GenerationLimits::default(),
        )
        .unwrap();
        let mut controls = ControlSet::default();
        controls
            .refinement(
                "rectangle",
                ControlRegion::sphere(vec3(0.07, 0.03, 0.0), 0.04).unwrap(),
                0.05,
                4.0,
            )
            .unwrap();
        let refined =
            mesh_chunks(&document, 0.05, 0.4, &controls, GenerationLimits::default()).unwrap();
        assert!(refined.cells > background.cells);
        assert!(
            refined.cells < background.cells.saturating_mul(4),
            "a local control should not refine the full domain: {} background, {} refined",
            background.cells,
            refined.cells,
        );

        let local_mean = |sink: &TestSink| {
            let mut lengths = Vec::new();
            for chunk in &sink.chunks {
                let points = chunk
                    .points
                    .iter()
                    .map(|point| (point.id, point.position))
                    .collect::<BTreeMap<_, _>>();
                for cell in &chunk.cells {
                    let positions = cell
                        .point_ids
                        .iter()
                        .map(|id| points[id])
                        .collect::<Vec<_>>();
                    let center = centroid_slice(&positions);
                    if distance3(center, [0.07, 0.03, 0.0]) < 0.3 {
                        lengths.push(maximum_edge_2d(&positions));
                    }
                }
            }
            lengths.iter().sum::<f64>() / lengths.len() as f64
        };
        assert!(local_mean(&refined) < local_mean(&background));

        let limited = mesh_chunks(
            &document,
            0.05,
            0.1,
            &ControlSet::default(),
            GenerationLimits {
                max_cells: 4,
                ..GenerationLimits::default()
            },
        );
        assert!(matches!(limited, Err(MeshError::LimitExceeded(_))));
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

        let mut rotated = SceneDocument::new();
        let root = rotated
            .add_point_shape_from_world_points(
                "polygon",
                &[
                    vec3(0.0, -0.8, -0.5),
                    vec3(0.0, 0.8, -0.5),
                    vec3(0.0, 0.8, 0.5),
                    vec3(0.0, -0.8, 0.5),
                ],
                "yz",
            )
            .expect("rotated workplane polygon");
        rotated
            .set_domain_root(root, DomainKind::Fluid)
            .expect("mark rotated polygon");
        mesh_document_fast(&rotated, 0.025, 0.2).expect("mesh rotated workplane");
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
