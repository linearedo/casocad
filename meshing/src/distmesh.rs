use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use caso_kernel::meshing::{BoundaryBand, MeshableDomain, MeshableDomainSpace, MeshableInterface};
use caso_kernel::vec3::Vec3;
use spade::{ConstrainedDelaunayTriangulation, HasPosition, Point2, Triangulation};

use crate::algorithm::{
    MeshAlgorithm, MeshAlgorithmCapabilities, MeshAlgorithmDescriptor, MeshSink, MeshingContext,
    MeshingPhase, MeshingProgress, MeshingStatistics, QualityTermination,
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
const FORCE_SCALE: f64 = 1.2;
const EULER_STEP: f64 = 0.2;
const RETRIANGULATION_THRESHOLD: f64 = 0.1;
const CONVERGENCE_THRESHOLD: f64 = 0.001;
const MAX_RELAXATION_ITERATIONS: usize = 100;
const MAX_QUALITY_PASSES: usize = 1_000;
const MAX_OPTIMIZATION_BYTES: usize = 512 * 1024 * 1024;

pub static DISTMESH: DistMesh = DistMesh;
pub static DISTMESH_DESCRIPTOR: MeshAlgorithmDescriptor = MeshAlgorithmDescriptor {
    id: "distmesh",
    label: "DistMesh (Out-of-Core)",
    dimensions: &[2],
    capabilities: MeshAlgorithmCapabilities {
        refinement: false,
        boundary_layers: true,
    },
};

#[derive(Debug, Clone, Copy, Default)]
pub struct DistMesh;

impl MeshAlgorithm for DistMesh {
    fn descriptor(&self) -> &'static MeshAlgorithmDescriptor {
        &DISTMESH_DESCRIPTOR
    }

    fn generate(
        &self,
        context: &MeshingContext<'_>,
        sink: &mut dyn MeshSink,
    ) -> MeshResult<MeshingStatistics> {
        generate(context, sink)
    }
}

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
        let base =
            lengths.map(|length| ((length / context.target_size).ceil() as u32).clamp(1, 1 << 20));
        if u64::from(base[0]).saturating_mul(u64::from(base[1]))
            > context.limits.max_cells.saturating_mul(4)
        {
            return Err(MeshError::LimitExceeded(
                "adaptive 2D base grid exceeds the configured cell limit".into(),
            ));
        }
        // The core remains uniform. Extra dyadic levels are available only
        // to resolve SDF topology and boundary curvature.
        let max_depth = 8;
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

#[derive(Debug, Clone, Copy)]
struct SpadeVertex {
    position: Point2<f64>,
    key: PointKey,
}

#[derive(Debug, Clone, Copy)]
struct SharedBoundaryPoint {
    position: [f64; 3],
    id: MeshId,
}

#[derive(Debug, Clone)]
struct SharedInterface {
    source: String,
    target: String,
    segments: Vec<[[f64; 3]; 2]>,
}

impl HasPosition for SpadeVertex {
    type Scalar = f64;

    fn position(&self) -> Point2<Self::Scalar> {
        self.position
    }
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
    layer_end_targets: Vec<LayerEndTarget>,
}

#[derive(Debug, Clone, Copy)]
struct LayerEndTarget {
    a: [f64; 3],
    b: [f64; 3],
    edge_length: f64,
}

#[derive(Debug)]
struct BoundaryLayerStrip {
    cells: Vec<Cell>,
    constraints: BTreeSet<(PointKey, PointKey)>,
    front_edges: Vec<[PointKey; 2]>,
    end_columns: Vec<Vec<PointKey>>,
    levels: BTreeMap<PointKey, f64>,
}

#[derive(Debug, Clone)]
struct CoreQuality {
    objective: f64,
    minimum_scaled_jacobian: f64,
    worst_first: Vec<(usize, f64)>,
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
    Flip(PointKey, PointKey),
    RelocateInterior(PointKey, u8),
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct LayerContourKey {
    layer: LayerKey,
    tangential_size: u64,
    owner: Option<String>,
}

impl LayerKey {
    fn from_control(control: &BoundaryLayerControl) -> Self {
        Self {
            first_height: control.hwall_n.to_bits(),
            layers: control.layers,
            growth: control.ratio.to_bits(),
        }
    }

    fn first_height(self) -> f64 {
        f64::from_bits(self.first_height)
    }

    fn growth(self) -> f64 {
        f64::from_bits(self.growth)
    }
}

impl LayerContourKey {
    fn tangential_size(&self) -> f64 {
        f64::from_bits(self.tangential_size)
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
    let mut shared_boundary_points = Vec::new();
    let mut shared_interfaces: Vec<SharedInterface> = Vec::new();
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
        let has_shared_interface = shared_interfaces
            .iter()
            .any(|interface| interface.target == domain.name);
        install_shared_interfaces(
            domain,
            &space,
            context,
            &mut candidate,
            &mut assessment,
            &shared_interfaces,
        )?;
        let clipped_seed = candidate.clone();
        retriangulate_with_spade(domain, context, &mut candidate, &assessment)?;
        assessment = assess(domain, &space, context, &candidate)?;
        if !has_shared_interface
            && (!assessment.refine.is_empty() || assessment.score.hard_invalid != 0)
        {
            candidate = clipped_seed;
            assessment = assess(domain, &space, context, &candidate)?;
        }
        let has_layers = context
            .controls
            .boundary_layers
            .iter()
            .any(|control| control.domain == domain.name);
        if has_layers {
            prepare_layer_boundaries(domain, &space, context, &mut candidate, &mut assessment)?;
            apply_boundary_layers(domain, &space, context, &mut candidate, &mut assessment)?;
        }
        relax_distmesh(domain, &space, context, &mut candidate, &mut assessment)?;
        optimize(
            domain,
            &space,
            context,
            &mut candidate,
            &mut assessment,
            &mut statistics,
        )?;
        sort_cells_morton(&space, &mut candidate);
        assessment = assess(domain, &space, context, &candidate)?;
        capture_shared_interfaces(
            domain,
            context,
            &candidate,
            &assessment,
            &mut shared_interfaces,
        );

        emit(
            context,
            domain,
            &candidate,
            &assessment,
            sink,
            &mut statistics,
            &mut shared_boundary_points,
        )?;
    }
    Ok(statistics)
}

fn interface_edge(interface: &MeshableInterface, a: [f64; 3], b: [f64; 3]) -> bool {
    let points = [
        Vec3::from_array(a),
        Vec3::from_array(b),
        Vec3::from_array(midpoint3(a, b)),
    ];
    interface.contains(&points).into_iter().all(|hit| hit)
}

fn capture_shared_interfaces(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    candidate: &Candidate,
    assessment: &Assessment,
    shared: &mut Vec<SharedInterface>,
) {
    for interface in context.domains.interfaces_of(&domain.name) {
        let target = if interface.domain_a == domain.name {
            &interface.domain_b
        } else {
            &interface.domain_a
        };
        if shared.iter().any(|entry| entry.target == *target) {
            continue;
        }
        let segments = assessment
            .boundary
            .iter()
            .filter_map(|edge| {
                let a = candidate.points[&edge.points[0]].world;
                let b = candidate.points[&edge.points[1]].world;
                interface_edge(interface, a, b).then_some([a, b])
            })
            .collect::<Vec<_>>();
        if !segments.is_empty() {
            shared.push(SharedInterface {
                source: domain.name.clone(),
                target: target.clone(),
                segments,
            });
        }
    }
}

fn install_shared_interfaces(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &mut Assessment,
    shared: &[SharedInterface],
) -> MeshResult<()> {
    for entry in shared.iter().filter(|entry| entry.target == domain.name) {
        let interface = context
            .domains
            .interface_between(&entry.source, &entry.target)
            .map_err(|error| MeshError::InvalidInput(error.to_string()))?;
        let old_interface_points = candidate
            .points
            .iter()
            .filter_map(|(key, point)| {
                interface.contains(&[Vec3::from_array(point.world)])[0].then_some(*key)
            })
            .collect::<BTreeSet<_>>();
        let transition_points = assessment
            .boundary
            .iter()
            .filter(|edge| {
                let matches = edge
                    .points
                    .iter()
                    .filter(|point| old_interface_points.contains(point))
                    .count();
                matches == 1
            })
            .flat_map(|edge| edge.points)
            .collect::<BTreeSet<_>>();
        let removable = old_interface_points
            .difference(&transition_points)
            .copied()
            .collect::<BTreeSet<_>>();
        candidate.points.retain(|key, _| !removable.contains(key));
        assessment
            .boundary_vertices
            .retain(|key| !removable.contains(key));
        assessment.boundary.retain(|edge| {
            let a = candidate
                .points
                .get(&edge.points[0])
                .map(|point| point.world);
            let b = candidate
                .points
                .get(&edge.points[1])
                .map(|point| point.world);
            match (a, b) {
                (Some(a), Some(b)) => !interface_edge(interface, a, b),
                _ => false,
            }
        });

        let mut keys = BTreeMap::<[u64; 3], PointKey>::new();
        for segment in &entry.segments {
            for world in segment {
                let bits = world.map(f64::to_bits);
                if keys.contains_key(&bits) {
                    continue;
                }
                let coords = space.coords(Vec3::from_array(*world));
                let key = PointKey::Inserted(candidate.next_inserted);
                candidate.next_inserted += 1;
                candidate.points.insert(
                    key,
                    Point {
                        uv: [coords[0], coords[1]],
                        world: *world,
                        boundary: true,
                        protected: true,
                    },
                );
                assessment.boundary_vertices.insert(key);
                keys.insert(bits, key);
            }
        }
        for segment in &entry.segments {
            let points = segment.map(|world| keys[&world.map(f64::to_bits)]);
            assessment.boundary.push(BoundaryEdge {
                points,
                cell: 0,
                owner: None,
            });
        }
    }
    Ok(())
}

fn relax_distmesh(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &mut Assessment,
) -> MeshResult<()> {
    let original = candidate.clone();
    let original_minimum_edge = minimum_edge_ratio(candidate, context.target_size);
    let original_score = assessment.score;
    let mut retriangulated_at = candidate
        .points
        .iter()
        .map(|(key, point)| (*key, point.uv))
        .collect::<BTreeMap<_, _>>();
    for iteration in 0..MAX_RELAXATION_ITERATIONS {
        if iteration.is_multiple_of(4) {
            context.check()?;
        }
        let edges = candidate
            .cells
            .iter()
            .flat_map(|cell| {
                (0..cell.points.len()).map(|edge| {
                    ordered_pair(
                        cell.points[edge],
                        cell.points[(edge + 1) % cell.points.len()],
                    )
                })
            })
            .collect::<BTreeSet<_>>();
        let mut forces = BTreeMap::<PointKey, ([f64; 2], usize)>::new();
        for (a, b) in edges {
            let pa = candidate.points[&a].uv;
            let pb = candidate.points[&b].uv;
            let wa = candidate.points[&a].world;
            let wb = candidate.points[&b].world;
            let delta = [pb[0] - pa[0], pb[1] - pa[1]];
            let length = delta[0].hypot(delta[1]);
            if length <= f64::EPSILON {
                continue;
            }
            let midpoint = midpoint3(wa, wb);
            let probes = [
                Vec3::from_array(wa),
                Vec3::from_array(wb),
                Vec3::from_array(midpoint),
            ];
            let target = candidate
                .layer_edge_targets
                .get(&ordered_pair(a, b))
                .copied()
                .unwrap_or_else(|| {
                    local_target(
                        candidate,
                        context,
                        &domain.name,
                        Vec3::from_array(midpoint),
                        length * 0.5,
                        &probes,
                    )
                });
            let compression = (FORCE_SCALE * target - length).max(0.0);
            if compression == 0.0 {
                continue;
            }
            let force = [
                delta[0] * compression / length,
                delta[1] * compression / length,
            ];
            let entry = forces.entry(a).or_default();
            entry.0[0] -= force[0];
            entry.0[1] -= force[1];
            entry.1 += 1;
            let entry = forces.entry(b).or_default();
            entry.0[0] += force[0];
            entry.0[1] += force[1];
            entry.1 += 1;
        }

        let mut next = Vec::new();
        let mut maximum_move: f64 = 0.0;
        for (key, (force, count)) in forces {
            let old = candidate.points[&key];
            if old.boundary || old.protected || assessment.boundary_vertices.contains(&key) {
                continue;
            }
            let scale = EULER_STEP / count.max(1) as f64;
            let mut uv = [old.uv[0] + scale * force[0], old.uv[1] + scale * force[1]];
            let mut world = space.point(uv[0], uv[1]);
            if domain.domain_sdf(&[world])[0] >= 0.0 {
                let mut inside = Vec3::from_array(old.world);
                let mut outside = world;
                for _ in 0..48 {
                    let middle = (inside + outside) * 0.5;
                    if domain.domain_sdf(&[middle])[0] <= 0.0 {
                        inside = middle;
                    } else {
                        outside = middle;
                    }
                }
                world = inside;
                let coords = space.coords(world);
                uv = [coords[0], coords[1]];
            }
            maximum_move = maximum_move.max((uv[0] - old.uv[0]).hypot(uv[1] - old.uv[1]));
            next.push((key, uv, world.to_array()));
        }
        for (key, uv, world) in next {
            let point = candidate
                .points
                .get_mut(&key)
                .expect("moving DistMesh point");
            point.uv = uv;
            point.world = world;
        }
        if maximum_move / context.target_size <= CONVERGENCE_THRESHOLD {
            break;
        }
        let needs_retriangulation = candidate.points.iter().any(|(key, point)| {
            retriangulated_at.get(key).is_some_and(|old| {
                (point.uv[0] - old[0]).hypot(point.uv[1] - old[1])
                    > RETRIANGULATION_THRESHOLD * context.target_size
            })
        });
        if needs_retriangulation && !candidate.cells.iter().any(|cell| cell.protected) {
            retriangulate_with_spade(domain, context, candidate, assessment)?;
            *assessment = assess(domain, space, context, candidate)?;
            if assessment.score.hard_invalid != 0
                || compare_scores(&assessment.score, &original_score) == Ordering::Greater
                || minimum_edge_ratio(candidate, context.target_size)
                    < original_minimum_edge * (1.0 - 1.0e-12)
            {
                *candidate = original;
                *assessment = assess(domain, space, context, candidate)?;
                return Ok(());
            }
            retriangulated_at = candidate
                .points
                .iter()
                .map(|(key, point)| (*key, point.uv))
                .collect();
        }
    }
    let relaxed = assess(domain, space, context, candidate)?;
    if relaxed.refine.is_empty()
        && relaxed.score.hard_invalid == 0
        && compare_scores(&relaxed.score, &assessment.score) != Ordering::Greater
        && minimum_edge_ratio(candidate, context.target_size)
            >= original_minimum_edge * (1.0 - 1.0e-12)
    {
        *assessment = relaxed;
    } else {
        *candidate = original;
        *assessment = assess(domain, space, context, candidate)?;
    }
    Ok(())
}

fn minimum_edge_ratio(candidate: &Candidate, target_size: f64) -> f64 {
    candidate
        .cells
        .iter()
        .flat_map(|cell| {
            (0..cell.points.len()).map(|edge| {
                distance3(
                    candidate.points[&cell.points[edge]].world,
                    candidate.points[&cell.points[(edge + 1) % cell.points.len()]].world,
                ) / target_size
            })
        })
        .fold(f64::INFINITY, f64::min)
}

fn sort_cells_morton(space: &MeshableDomainSpace, candidate: &mut Candidate) {
    let bounds = space.bounds();
    let Candidate { points, cells, .. } = candidate;
    cells.sort_by_key(|cell| {
        let center = cell
            .points
            .iter()
            .map(|key| points[key].uv)
            .fold([0.0; 2], |sum, point| {
                [sum[0] + point[0], sum[1] + point[1]]
            })
            .map(|value| value / cell.points.len() as f64);
        let quantize = |value: f64, min: f64, max: f64| {
            (((value - min) / (max - min).max(f64::EPSILON)).clamp(0.0, 1.0) * u32::MAX as f64)
                as u32
        };
        morton2(
            quantize(center[0], bounds[0], bounds[1]),
            quantize(center[1], bounds[2], bounds[3]),
        )
    });
}

fn morton2(x: u32, y: u32) -> u64 {
    fn spread(mut value: u64) -> u64 {
        value &= 0x0000_0000_ffff_ffff;
        value = (value | value << 16) & 0x0000_ffff_0000_ffff;
        value = (value | value << 8) & 0x00ff_00ff_00ff_00ff;
        value = (value | value << 4) & 0x0f0f_0f0f_0f0f_0f0f;
        value = (value | value << 2) & 0x3333_3333_3333_3333;
        (value | value << 1) & 0x5555_5555_5555_5555
    }
    spread(u64::from(x)) | spread(u64::from(y)) << 1
}

fn retriangulate_with_spade(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &Assessment,
) -> MeshResult<()> {
    let mut triangulation = ConstrainedDelaunayTriangulation::<SpadeVertex>::new();
    let mut handles = BTreeMap::new();
    let mut leaves = BTreeMap::new();
    for cell in &candidate.cells {
        for key in &cell.points {
            leaves.entry(*key).or_insert(cell.leaf);
        }
    }
    for (&key, point) in &candidate.points {
        let handle = triangulation
            .insert(SpadeVertex {
                position: Point2::new(point.uv[0], point.uv[1]),
                key,
            })
            .map_err(|error| {
                MeshError::InvalidInput(format!(
                    "Spade rejected a DistMesh vertex in domain {:?}: {error:?}",
                    domain.name
                ))
            })?;
        handles.insert(key, handle);
    }
    for edge in &assessment.boundary {
        triangulation.try_add_constraint(handles[&edge.points[0]], handles[&edge.points[1]]);
    }

    let mut cells = Vec::new();
    for (index, face) in triangulation.inner_faces().enumerate() {
        if index.is_multiple_of(512) {
            context.check()?;
        }
        let points = face.vertices().map(|vertex| vertex.data().key);
        if points[0] == points[1] || points[1] == points[2] || points[2] == points[0] {
            continue;
        }
        let positions = points.map(|key| candidate.points[&key].world);
        let centroid = centroid_slice(&positions);
        if domain.domain_sdf(&[Vec3::from_array(centroid)])[0] >= 0.0
            || cell_containment_residual(domain, &positions)
                > chord_tolerance(domain, maximum_edge_2d(&positions))
        {
            continue;
        }
        let leaf = points
            .iter()
            .find_map(|key| leaves.get(key))
            .copied()
            .unwrap_or(Leaf {
                level: 0,
                x: 0,
                y: 0,
            });
        cells.push(Cell::triangle(points, leaf));
    }
    if cells.is_empty() {
        return Err(MeshError::InvalidInput(format!(
            "Spade produced no interior triangles for domain {:?}",
            domain.name
        )));
    }
    candidate.cells = cells;
    candidate.construction_failures.clear();
    Ok(())
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
    let _ = (domain, center, radius, probes);
    context.target_size
}

fn local_target(
    candidate: &Candidate,
    context: &MeshingContext<'_>,
    domain: &str,
    center: Vec3,
    radius: f64,
    probes: &[Vec3],
) -> f64 {
    candidate.layer_end_targets.iter().fold(
        regional_target(context, domain, center, radius, probes),
        |target, end| {
            target.min(
                end.edge_length
                    + point_segment_distance(
                        center,
                        Vec3::from_array(end.a),
                        Vec3::from_array(end.b),
                    ),
            )
        },
    )
}

#[cfg(test)]
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
        layer_end_targets: Vec::new(),
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
    let Candidate { points, cells, .. } = &mut candidate;
    cells.retain(|cell| {
        let positions = cell
            .points
            .iter()
            .map(|key| points[key].world)
            .collect::<Vec<_>>();
        let size = maximum_edge_2d(&positions);
        signed_area_polygon(&cell.points, points) > orientation_tolerance(size)
            && quality_score(
                cell.element_type(),
                &positions,
                QualityMetric::ScaledJacobian,
            )
            .unwrap_or(0.0)
                > VALID_QUALITY
    });
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
            continue;
        }
        let positions = triangle.map(|key| candidate.points[&key].world);
        if quality_score("tri3", &positions, QualityMetric::ScaledJacobian).unwrap_or(0.0)
            <= VALID_QUALITY
        {
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
    let world = if projection.converged && projection.distance_moved <= local_size {
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
        let containment_tolerance = topology_tolerance(domain, size);
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
                local_target(
                    candidate,
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
        let target = local_target(
            candidate,
            context,
            &domain.name,
            Vec3::from_array(midpoint),
            size * 0.5,
            &probes,
        );
        let topology_tolerance = topology_tolerance(domain, size);
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

fn rediscretize_layer_boundaries(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &mut Assessment,
) -> MeshResult<()> {
    let mut groups = BTreeMap::<LayerContourKey, Vec<BoundaryEdge>>::new();
    let mut replaced = BTreeSet::new();
    for (edge_index, edge) in assessment.boundary.iter().enumerate() {
        if edge_index.is_multiple_of(128) {
            context.check()?;
        }
        if edge
            .points
            .iter()
            .any(|point| candidate.points[point].protected)
        {
            continue;
        }
        let a = candidate.points[&edge.points[0]].world;
        let b = candidate.points[&edge.points[1]].world;
        let memberships = layer_memberships(
            domain,
            context,
            midpoint3(a, b),
            chord_tolerance(domain, distance3(a, b)),
        )?;
        let mut layers = memberships
            .iter()
            .map(|index| LayerKey::from_control(&context.controls.boundary_layers[*index]))
            .collect::<BTreeSet<_>>();
        if layers.len() > 1 {
            return Err(MeshError::InvalidInput(format!(
                "domain {:?} has overlapping boundary-layer controls with incompatible hwall_n, ratio, or derived layer count",
                domain.name
            )));
        }
        let Some(layer) = layers.pop_first() else {
            continue;
        };
        replaced.insert(ordered_pair(edge.points[0], edge.points[1]));
        groups
            .entry(LayerContourKey {
                layer,
                tangential_size: memberships
                    .iter()
                    .map(|index| context.controls.boundary_layers[*index].hwall_t)
                    .fold(f64::INFINITY, f64::min)
                    .to_bits(),
                owner: edge.owner.clone(),
            })
            .or_default()
            .push(edge.clone());
    }
    if groups.is_empty() {
        return Ok(());
    }

    let old_boundary_vertices = assessment.boundary_vertices.clone();
    let mut constraints = assessment
        .boundary
        .iter()
        .filter(|edge| !replaced.contains(&ordered_pair(edge.points[0], edge.points[1])))
        .cloned()
        .collect::<Vec<_>>();
    for (group_index, (key, edges)) in groups.into_iter().enumerate() {
        if group_index.is_multiple_of(16) {
            context.check()?;
        }
        for path in ordered_boundary_paths(domain, &edges)? {
            resample_boundary_path(
                domain,
                space,
                context,
                candidate,
                &key,
                &path,
                &mut constraints,
            )?;
        }
    }

    let retained = constraints
        .iter()
        .flat_map(|edge| edge.points)
        .collect::<BTreeSet<_>>();
    for point in old_boundary_vertices {
        if !retained.contains(&point) {
            candidate.points.remove(&point);
        }
    }
    assessment.boundary = constraints;
    assessment.boundary_vertices = retained;
    retriangulate_with_spade(domain, context, candidate, assessment)?;
    *assessment = assess(domain, space, context, candidate)?;
    if !assessment.refine.is_empty() || assessment.score.hard_invalid != 0 {
        return Err(layer_error(
            domain,
            "could not retriangulate the core after uniform boundary rediscretization",
        ));
    }
    if candidate.cells.len() > usize::try_from(context.limits.max_cells).unwrap_or(usize::MAX) {
        return Err(MeshError::LimitExceeded(format!(
            "uniform boundary rediscretization exceeds the configured {} cell limit",
            context.limits.max_cells
        )));
    }
    Ok(())
}

fn ordered_boundary_paths(
    domain: &MeshableDomain,
    edges: &[BoundaryEdge],
) -> MeshResult<Vec<Vec<PointKey>>> {
    let mut outgoing = BTreeMap::<PointKey, usize>::new();
    let mut incoming = BTreeMap::<PointKey, usize>::new();
    for (index, edge) in edges.iter().enumerate() {
        if outgoing.insert(edge.points[0], index).is_some()
            || incoming.insert(edge.points[1], index).is_some()
        {
            return Err(layer_error(
                domain,
                "controlled contour is not an oriented manifold path",
            ));
        }
    }
    let mut unused = (0..edges.len()).collect::<BTreeSet<_>>();
    let mut paths = Vec::new();
    while let Some(&fallback) = unused.first() {
        let start_edge = unused
            .iter()
            .copied()
            .find(|index| !incoming.contains_key(&edges[*index].points[0]))
            .unwrap_or(fallback);
        let start = edges[start_edge].points[0];
        let mut point = start;
        let mut path = vec![start];
        while let Some(index) = outgoing
            .get(&point)
            .copied()
            .filter(|index| unused.contains(index))
        {
            unused.remove(&index);
            point = edges[index].points[1];
            path.push(point);
            if point == start {
                break;
            }
        }
        if path.len() < 2 {
            return Err(layer_error(
                domain,
                "controlled contour contains no usable edge",
            ));
        }
        paths.push(path);
    }
    Ok(paths)
}

fn resample_boundary_path(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    contour: &LayerContourKey,
    path: &[PointKey],
    constraints: &mut Vec<BoundaryEdge>,
) -> MeshResult<()> {
    let closed = path.first() == path.last();
    let vertices = if closed {
        &path[..path.len() - 1]
    } else {
        path
    };
    if vertices.len() < usize::from(closed) + 2 {
        return Err(layer_error(
            domain,
            "controlled contour is too short to resample",
        ));
    }
    let mut fixed = vertices
        .iter()
        .enumerate()
        .filter_map(|(index, point)| candidate.points[point].protected.then_some(index))
        .collect::<BTreeSet<_>>();
    if !closed {
        fixed.insert(0);
        fixed.insert(vertices.len() - 1);
    }
    for index in 0..vertices.len() {
        if (!closed && (index == 0 || index + 1 == vertices.len()))
            || candidate.points[&vertices[index]].protected
        {
            continue;
        }
        let previous = vertices[(index + vertices.len() - 1) % vertices.len()];
        let next = vertices[(index + 1) % vertices.len()];
        let before = Vec3::from_array(candidate.points[&vertices[index]].world)
            - Vec3::from_array(candidate.points[&previous].world);
        let after = Vec3::from_array(candidate.points[&next].world)
            - Vec3::from_array(candidate.points[&vertices[index]].world);
        let lengths = before.length() * after.length();
        if lengths <= f64::EPSILON || before.dot(after) / lengths < 0.866_025_403_784_438_6 {
            fixed.insert(index);
        }
    }

    if closed && fixed.is_empty() {
        let mut cycle = vertices.to_vec();
        cycle.push(vertices[0]);
        let stations = resample_boundary_arc(
            domain,
            space,
            context,
            candidate,
            &cycle,
            true,
            contour.tangential_size(),
        )?;
        for index in 0..stations.len() {
            candidate.layer_edge_targets.insert(
                ordered_pair(stations[index], stations[(index + 1) % stations.len()]),
                contour.tangential_size(),
            );
            constraints.push(BoundaryEdge {
                points: [stations[index], stations[(index + 1) % stations.len()]],
                cell: 0,
                owner: contour.owner.clone(),
            });
        }
        return Ok(());
    }

    let mut linear = vertices.to_vec();
    if closed {
        let first_fixed = *fixed.first().expect("closed contour has a fixed point");
        linear.rotate_left(first_fixed);
        linear.push(linear[0]);
    }
    let fixed_points = linear
        .iter()
        .enumerate()
        .filter_map(|(index, point)| {
            let original = vertices.iter().position(|candidate| candidate == point)?;
            (fixed.contains(&original) || index == 0 || index + 1 == linear.len()).then_some(index)
        })
        .collect::<Vec<_>>();
    for pair in fixed_points.windows(2) {
        let stations = resample_boundary_arc(
            domain,
            space,
            context,
            candidate,
            &linear[pair[0]..=pair[1]],
            false,
            contour.tangential_size(),
        )?;
        for edge in stations.windows(2) {
            candidate
                .layer_edge_targets
                .insert(ordered_pair(edge[0], edge[1]), contour.tangential_size());
            constraints.push(BoundaryEdge {
                points: [edge[0], edge[1]],
                cell: 0,
                owner: contour.owner.clone(),
            });
        }
    }
    Ok(())
}

fn resample_boundary_arc(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    path: &[PointKey],
    closed: bool,
    tangential_size: f64,
) -> MeshResult<Vec<PointKey>> {
    context.check()?;
    let lengths = path
        .windows(2)
        .map(|edge| {
            distance3(
                candidate.points[&edge[0]].world,
                candidate.points[&edge[1]].world,
            )
        })
        .collect::<Vec<_>>();
    let total = lengths.iter().sum::<f64>();
    if !total.is_finite() || total <= f64::EPSILON {
        return Err(layer_error(
            domain,
            "controlled contour has zero arc length",
        ));
    }
    let edge_count = ((total / tangential_size).round() as usize).max(if closed { 3 } else { 1 });
    let station_count = if closed { edge_count } else { edge_count + 1 };
    let mut base = Vec::with_capacity(station_count);
    for station in 0..station_count {
        if station == 0 {
            base.push(path[0]);
            continue;
        }
        if !closed && station == edge_count {
            base.push(*path.last().expect("non-empty boundary arc"));
            continue;
        }
        let distance = total * station as f64 / edge_count as f64;
        let (seed, tangent) = point_on_boundary_polyline(candidate, path, &lengths, distance);
        let point = project_boundary_station(domain, space, tangential_size, seed, tangent)?;
        let key = PointKey::Inserted(candidate.next_inserted);
        candidate.next_inserted += 1;
        candidate.points.insert(key, point);
        base.push(key);
    }
    redistribute_boundary_stations(domain, space, tangential_size, candidate, &base, closed)?;

    let mut stations = vec![base[0]];
    for pair in base.windows(2) {
        append_boundary_chord(
            domain,
            space,
            tangential_size,
            candidate,
            pair[0],
            pair[1],
            0,
            &mut stations,
        )?;
    }
    if closed {
        append_boundary_chord(
            domain,
            space,
            tangential_size,
            candidate,
            *base.last().expect("closed boundary stations"),
            base[0],
            0,
            &mut stations,
        )?;
        if stations.last() == stations.first() {
            stations.pop();
        }
    }
    Ok(stations)
}

fn redistribute_boundary_stations(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    target_size: f64,
    candidate: &mut Candidate,
    stations: &[PointKey],
    closed: bool,
) -> MeshResult<()> {
    let edge_count = if closed {
        stations.len()
    } else {
        stations.len().saturating_sub(1)
    };
    if edge_count <= 1 {
        return Ok(());
    }
    for _ in 0..32 {
        let mut path = stations.to_vec();
        if closed {
            path.push(stations[0]);
        }
        let lengths = path
            .windows(2)
            .map(|edge| {
                distance3(
                    candidate.points[&edge[0]].world,
                    candidate.points[&edge[1]].world,
                )
            })
            .collect::<Vec<_>>();
        let total = lengths.iter().sum::<f64>();
        let movable_end = if closed {
            stations.len()
        } else {
            stations.len() - 1
        };
        let mut updates = Vec::with_capacity(movable_end.saturating_sub(1));
        for (station, &key) in stations.iter().enumerate().take(movable_end).skip(1) {
            let distance = total * station as f64 / edge_count as f64;
            let (seed, tangent) = point_on_boundary_polyline(candidate, &path, &lengths, distance);
            updates.push((
                key,
                project_boundary_station(domain, space, target_size, seed, tangent)?,
            ));
        }
        for (key, point) in updates {
            candidate.points.insert(key, point);
        }

        let lengths = (0..edge_count)
            .map(|index| {
                distance3(
                    candidate.points[&stations[index]].world,
                    candidate.points[&stations[(index + 1) % stations.len()]].world,
                )
            })
            .collect::<Vec<_>>();
        let shortest = lengths.iter().copied().fold(f64::INFINITY, f64::min);
        let longest = lengths.iter().copied().fold(0.0, f64::max);
        if longest / shortest <= 1.001 {
            break;
        }
    }
    Ok(())
}

fn point_on_boundary_polyline(
    candidate: &Candidate,
    path: &[PointKey],
    lengths: &[f64],
    mut distance: f64,
) -> ([f64; 3], [f64; 2]) {
    for (index, &length) in lengths.iter().enumerate() {
        if distance <= length || index + 1 == lengths.len() {
            let fraction = (distance / length.max(f64::MIN_POSITIVE)).clamp(0.0, 1.0);
            let a = candidate.points[&path[index]];
            let b = candidate.points[&path[index + 1]];
            return (
                lerp3(a.world, b.world, fraction),
                [b.uv[0] - a.uv[0], b.uv[1] - a.uv[1]],
            );
        }
        distance -= length;
    }
    let a = candidate.points[&path[path.len() - 2]];
    let b = candidate.points[path.last().expect("non-empty boundary polyline")];
    (b.world, [b.uv[0] - a.uv[0], b.uv[1] - a.uv[1]])
}

fn project_boundary_station(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    target_size: f64,
    seed: [f64; 3],
    tangent: [f64; 2],
) -> MeshResult<Point> {
    let tangent_length = tangent[0].hypot(tangent[1]);
    if tangent_length <= f64::EPSILON {
        return Err(layer_error(
            domain,
            "encountered a zero-length contour tangent during arc-length rediscretization",
        ));
    }
    let seed = Vec3::from_array(seed);
    let seed_value = domain.domain_sdf(&[seed])[0];
    if !seed_value.is_finite() {
        return Err(layer_error(
            domain,
            "returned a non-finite SDF value during contour rediscretization",
        ));
    }
    if seed_value == 0.0 {
        let coords = space.coords(seed);
        return Ok(Point {
            uv: [coords[0], coords[1]],
            world: seed.to_array(),
            boundary: true,
            protected: false,
        });
    }
    let inward = [-tangent[1] / tangent_length, tangent[0] / tangent_length];
    let (mut interior, mut exterior) = if seed_value < 0.0 {
        (
            seed,
            find_boundary_bracket_side(
                domain,
                space,
                seed,
                [-inward[0], -inward[1]],
                target_size,
                false,
            )
            .ok_or_else(|| {
                layer_error(
                    domain,
                    "could not find the non-negative SDF side of a contour station",
                )
            })?,
        )
    } else {
        (
            find_boundary_bracket_side(domain, space, seed, inward, target_size, true).ok_or_else(
                || {
                    layer_error(
                        domain,
                        "could not find the negative SDF side of a contour station",
                    )
                },
            )?,
            seed,
        )
    };
    let tolerance = root_tolerance(domain, target_size);
    for _ in 0..64 {
        if (interior - exterior).length() <= tolerance {
            break;
        }
        let midpoint = (interior + exterior) * 0.5;
        let value = domain.domain_sdf(&[midpoint])[0];
        if !value.is_finite() {
            return Err(layer_error(
                domain,
                "returned a non-finite SDF value while bracketing a contour station",
            ));
        }
        if value < 0.0 {
            interior = midpoint;
        } else {
            exterior = midpoint;
        }
    }
    let projection = domain
        .project_to_boundary(&[interior])
        .map_err(|error| MeshError::InvalidInput(error.to_string()))?[0];
    let point = if projection.converged {
        projection.point
    } else {
        // The sign bracket already locates the wall to root tolerance. The
        // Newton projector is useful at smooth points but is allowed to stop
        // at C0 SDF seams, so keep the certified interior-side limit there.
        interior
    };
    let coords = space.coords(point);
    Ok(Point {
        uv: [coords[0], coords[1]],
        world: point.to_array(),
        boundary: true,
        protected: false,
    })
}

fn find_boundary_bracket_side(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    seed: Vec3,
    preferred: [f64; 2],
    target_size: f64,
    negative: bool,
) -> Option<Vec3> {
    let uv = space.coords(seed);
    let base = root_tolerance(domain, target_size).max(target_size * 1.0e-7);
    let limit = (target_size * 4.0)
        .max(domain.boundary_tolerance() * 8.0)
        .min(domain.bounds.diagonal().max(base));
    let mut step = base;
    for _ in 0..32 {
        for turn in [0_i32, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8] {
            let angle = turn as f64 * std::f64::consts::PI / 8.0;
            let (sin, cos) = angle.sin_cos();
            let direction = [
                preferred[0] * cos - preferred[1] * sin,
                preferred[0] * sin + preferred[1] * cos,
            ];
            let point = space.point(uv[0] + direction[0] * step, uv[1] + direction[1] * step);
            let value = domain.domain_sdf(&[point])[0];
            if value.is_finite() && ((negative && value < 0.0) || (!negative && value >= 0.0)) {
                return Some(point);
            }
        }
        if step >= limit {
            break;
        }
        step = (step * 2.0).min(limit);
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn append_boundary_chord(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    target_size: f64,
    candidate: &mut Candidate,
    a: PointKey,
    b: PointKey,
    depth: u8,
    stations: &mut Vec<PointKey>,
) -> MeshResult<()> {
    let midpoint = midpoint3(candidate.points[&a].world, candidate.points[&b].world);
    let a_uv = candidate.points[&a].uv;
    let b_uv = candidate.points[&b].uv;
    let projection = project_boundary_station(
        domain,
        space,
        target_size,
        midpoint,
        [b_uv[0] - a_uv[0], b_uv[1] - a_uv[1]],
    )?;
    let residual = distance3(midpoint, projection.world)
        .max(domain.domain_sdf(&[Vec3::from_array(midpoint)])[0].abs());
    if residual <= chord_tolerance(domain, target_size) {
        stations.push(b);
        return Ok(());
    }
    if depth == 16 {
        return Err(layer_error(
            domain,
            "could not satisfy the SDF chord tolerance while rediscretizing the boundary",
        ));
    }
    let middle = PointKey::Inserted(candidate.next_inserted);
    candidate.next_inserted += 1;
    candidate.points.insert(middle, projection);
    append_boundary_chord(
        domain,
        space,
        target_size,
        candidate,
        a,
        middle,
        depth + 1,
        stations,
    )?;
    append_boundary_chord(
        domain,
        space,
        target_size,
        candidate,
        middle,
        b,
        depth + 1,
        stations,
    )
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
    rediscretize_layer_boundaries(domain, space, context, candidate, assessment)?;
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
                "domain {:?} has overlapping boundary-layer controls with incompatible hwall_n, ratio, or derived layer count",
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
                    "domain {:?} has adjacent boundary-layer controls with incompatible hwall_n, ratio, or derived layer count",
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

    let controlled = groups
        .values()
        .flatten()
        .map(|edge| ordered_pair(edge.points[0], edge.points[1]))
        .collect::<BTreeSet<_>>();
    let mut core_constraints = assessment
        .boundary
        .iter()
        .filter_map(|edge| {
            let ordered = ordered_pair(edge.points[0], edge.points[1]);
            (!controlled.contains(&ordered)).then_some(ordered)
        })
        .collect::<BTreeSet<_>>();
    let strip = build_boundary_layer_strip(domain, space, context, candidate, assessment, groups)?;
    validate_boundary_layer_strip(domain, context, candidate, &strip)?;
    core_constraints.extend(strip.constraints.iter().copied());
    rebuild_constrained_core(domain, space, context, candidate, strip, &core_constraints)?;
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
                    .unwrap_or("produced invalid constrained core connectivity"),
                location[0],
                location[1],
                location[2],
                assessment.worst_quality,
            ),
        ));
    }
    Ok(())
}

fn build_boundary_layer_strip(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &Assessment,
    groups: BTreeMap<LayerKey, Vec<BoundaryEdge>>,
) -> MeshResult<BoundaryLayerStrip> {
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

    let mut strip = BoundaryLayerStrip {
        cells: Vec::with_capacity(added_cells),
        constraints: BTreeSet::new(),
        front_edges: Vec::new(),
        end_columns: Vec::new(),
        levels: BTreeMap::new(),
    };
    candidate.layer_end_targets.clear();

    for (key, edges) in groups {
        let paths = ordered_boundary_paths(domain, &edges)?;
        let mut degree = BTreeMap::<PointKey, usize>::new();
        let mut directions = BTreeMap::<PointKey, [f64; 2]>::new();
        let mut tangential_sizes = BTreeMap::<PointKey, f64>::new();
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
            let tangential_size = candidate
                .layer_edge_targets
                .get(&ordered_pair(edge.points[0], edge.points[1]))
                .copied()
                .unwrap_or(context.target_size);
            for point in edge.points {
                *degree.entry(point).or_default() += 1;
                let sum = directions.entry(point).or_default();
                sum[0] += inward[0];
                sum[1] += inward[1];
                tangential_sizes
                    .entry(point)
                    .and_modify(|size| *size = size.min(tangential_size))
                    .or_insert(tangential_size);
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
        let controlled_edges = edges
            .iter()
            .map(|edge| ordered_pair(edge.points[0], edge.points[1]))
            .collect::<BTreeSet<_>>();
        for &point in &endpoints {
            for edge in assessment.boundary.iter().filter(|edge| {
                edge.points.contains(&point)
                    && !controlled_edges.contains(&ordered_pair(edge.points[0], edge.points[1]))
            }) {
                let a = candidate.points[&edge.points[0]].uv;
                let b = candidate.points[&edge.points[1]].uv;
                let delta = [b[0] - a[0], b[1] - a[1]];
                let length = delta[0].hypot(delta[1]);
                if length > f64::EPSILON {
                    let direction = directions.entry(point).or_default();
                    direction[0] -= delta[1] / length;
                    direction[1] += delta[0] / length;
                }
            }
        }

        let mut distances = Vec::with_capacity(key.layers + 1);
        distances.push(0.0);
        let mut height = key.first_height();
        for _ in 0..key.layers {
            distances.push(distances.last().copied().unwrap_or(0.0) + height);
            height *= key.growth();
        }

        let original = degree
            .keys()
            .map(|point| (*point, candidate.points[point]))
            .collect::<BTreeMap<_, _>>();
        let mut rows = BTreeMap::<(PointKey, usize), PointKey>::new();
        for (&point, source) in &original {
            let direction = directions[&point];
            let direction_length = direction[0].hypot(direction[1]);
            if direction_length <= 1.0e-12 {
                return Err(layer_error(
                    domain,
                    "has an undefined inward normal at a CAD corner",
                ));
            }
            let direction = [
                direction[0] / direction_length,
                direction[1] / direction_length,
            ];
            let mut source = project_boundary_station(
                domain,
                space,
                tangential_sizes[&point],
                source.world,
                [direction[1], -direction[0]],
            )?;
            source.protected = true;
            candidate.points.insert(point, source);
            rows.insert((point, 0), point);
            strip.levels.insert(point, 0.0);
            let mut previous = source;
            for row in 1..=key.layers {
                let step = distances[row] - distances[row - 1];
                let position = layer_point(
                    domain,
                    space,
                    previous,
                    source,
                    direction,
                    step,
                    distances[row],
                )?;
                let row_key = PointKey::Inserted(candidate.next_inserted);
                candidate.next_inserted += 1;
                candidate.points.insert(row_key, position);
                rows.insert((point, row), row_key);
                strip.levels.insert(row_key, distances[row]);
                previous = position;
            }
        }

        let fixed_columns = fixed_layer_columns(candidate, &paths);
        redistribute_layer_rows(
            domain,
            space,
            context,
            candidate,
            &paths,
            &rows,
            &distances,
            &fixed_columns,
            tangential_sizes
                .values()
                .copied()
                .fold(f64::INFINITY, f64::min),
        )?;

        for &point in original.keys() {
            for row in 0..key.layers {
                let a = rows[&(point, row)];
                let b = rows[&(point, row + 1)];
                candidate
                    .layer_edge_targets
                    .insert(ordered_pair(a, b), distances[row + 1] - distances[row]);
            }
        }

        for point in endpoints {
            let column = (0..=key.layers)
                .map(|row| rows[&(point, row)])
                .collect::<Vec<_>>();
            for pair in column.windows(2) {
                let a = candidate.points[&pair[0]].world;
                let b = candidate.points[&pair[1]].world;
                candidate.layer_end_targets.push(LayerEndTarget {
                    a,
                    b,
                    edge_length: distance3(a, b),
                });
            }
            strip.end_columns.push(column);
        }

        for edge in edges {
            let tangential_size = candidate
                .layer_edge_targets
                .get(&ordered_pair(edge.points[0], edge.points[1]))
                .copied()
                .unwrap_or(context.target_size);
            for row in 0..=key.layers {
                candidate.layer_edge_targets.insert(
                    ordered_pair(rows[&(edge.points[0], row)], rows[&(edge.points[1], row)]),
                    tangential_size,
                );
            }
            for row in 0..key.layers {
                let mut points = [
                    rows[&(edge.points[0], row)],
                    rows[&(edge.points[1], row)],
                    rows[&(edge.points[1], row + 1)],
                    rows[&(edge.points[0], row + 1)],
                ];
                if signed_area_polygon(&points, &candidate.points) < 0.0 {
                    points.reverse();
                }
                strip
                    .cells
                    .push(Cell::quad(points, candidate.cells[edge.cell].leaf, true));
            }
            strip.front_edges.push([
                rows[&(edge.points[0], key.layers)],
                rows[&(edge.points[1], key.layers)],
            ]);
        }
    }

    for cell in &strip.cells {
        for edge in 0..4 {
            strip
                .constraints
                .insert(ordered_pair(cell.points[edge], cell.points[(edge + 1) % 4]));
        }
    }
    for column in &strip.end_columns {
        for edge in column.windows(2) {
            strip.constraints.insert(ordered_pair(edge[0], edge[1]));
        }
    }
    Ok(strip)
}

fn fixed_layer_columns(candidate: &Candidate, paths: &[Vec<PointKey>]) -> BTreeSet<PointKey> {
    let mut fixed = BTreeSet::new();
    for path in paths {
        let closed = path.first() == path.last();
        let vertices = if closed {
            &path[..path.len() - 1]
        } else {
            fixed.insert(path[0]);
            fixed.insert(*path.last().expect("open path endpoint"));
            path
        };
        for index in 0..vertices.len() {
            if !closed && (index == 0 || index + 1 == vertices.len()) {
                continue;
            }
            let previous = vertices[(index + vertices.len() - 1) % vertices.len()];
            let current = vertices[index];
            let next = vertices[(index + 1) % vertices.len()];
            let before = Vec3::from_array(candidate.points[&current].world)
                - Vec3::from_array(candidate.points[&previous].world);
            let after = Vec3::from_array(candidate.points[&next].world)
                - Vec3::from_array(candidate.points[&current].world);
            let lengths = before.length() * after.length();
            if lengths <= f64::EPSILON || before.dot(after) / lengths < 0.866_025_403_784_438_6 {
                fixed.insert(current);
            }
        }
    }
    fixed
}

#[allow(clippy::too_many_arguments)]
fn redistribute_layer_rows(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    paths: &[Vec<PointKey>],
    rows: &BTreeMap<(PointKey, usize), PointKey>,
    distances: &[f64],
    fixed_columns: &BTreeSet<PointKey>,
    tangential_size: f64,
) -> MeshResult<()> {
    let movement_tolerance =
        CONVERGENCE_THRESHOLD * tangential_size.min(distances[1] - distances[0]);
    for iteration in 0_usize..32 {
        if iteration.is_multiple_of(4) {
            context.check()?;
        }
        let mut updates = Vec::new();
        let mut maximum_move: f64 = 0.0;
        for row in 1..distances.len() {
            for path in paths {
                let closed = path.first() == path.last();
                let vertices = if closed {
                    &path[..path.len() - 1]
                } else {
                    path.as_slice()
                };
                for index in 0..vertices.len() {
                    let column = vertices[index];
                    if fixed_columns.contains(&column)
                        || (!closed && (index == 0 || index + 1 == vertices.len()))
                    {
                        continue;
                    }
                    let previous = vertices[(index + vertices.len() - 1) % vertices.len()];
                    let next = vertices[(index + 1) % vertices.len()];
                    let key = rows[&(column, row)];
                    let old = candidate.points[&key];
                    let before = candidate.points[&rows[&(previous, row)]].uv;
                    let after = candidate.points[&rows[&(next, row)]].uv;
                    let seed = [
                        0.5 * old.uv[0] + 0.25 * (before[0] + after[0]),
                        0.5 * old.uv[1] + 0.25 * (before[1] + after[1]),
                    ];
                    let corrected = correct_sdf_level(domain, space, old, seed, distances[row])?;
                    maximum_move = maximum_move.max(distance3(old.world, corrected.world));
                    updates.push((key, corrected));
                }
            }
        }
        let previous = updates
            .iter()
            .map(|(key, _)| (*key, candidate.points[key]))
            .collect::<Vec<_>>();
        for (key, point) in updates {
            candidate.points.insert(key, point);
        }
        if !layer_rows_are_valid(candidate, paths, rows, distances.len() - 1) {
            for (key, point) in previous {
                candidate.points.insert(key, point);
            }
            break;
        }
        if maximum_move <= movement_tolerance {
            break;
        }
    }
    Ok(())
}

fn layer_rows_are_valid(
    candidate: &Candidate,
    paths: &[Vec<PointKey>],
    rows: &BTreeMap<(PointKey, usize), PointKey>,
    layers: usize,
) -> bool {
    paths.iter().all(|path| {
        path.windows(2).all(|edge| {
            (0..layers).all(|row| {
                let points = [
                    rows[&(edge[0], row)],
                    rows[&(edge[1], row)],
                    rows[&(edge[1], row + 1)],
                    rows[&(edge[0], row + 1)],
                ];
                let positions = points.map(|key| candidate.points[&key].world);
                signed_area_polygon(&points, &candidate.points)
                    > orientation_tolerance(maximum_edge_2d(&positions))
                    && !polygon_self_intersects(&points, &candidate.points)
                    && quality_score("quad4", &positions, QualityMetric::ScaledJacobian)
                        .is_some_and(|quality| quality > VALID_QUALITY)
            })
        })
    })
}

fn correct_sdf_level(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    old: Point,
    seed: [f64; 2],
    distance: f64,
) -> MeshResult<Point> {
    let tolerance = root_tolerance(domain, distance).max(distance * 1.0e-8);
    let mut point = space.point(seed[0], seed[1]);
    for _ in 0..24 {
        let value = domain.domain_sdf(&[point])[0];
        if !value.is_finite() {
            return Err(layer_error(domain, "encountered a non-finite SDF value"));
        }
        let residual = value + distance;
        if residual.abs() <= tolerance {
            let coords = space.coords(point);
            return Ok(Point {
                uv: [coords[0], coords[1]],
                world: point.to_array(),
                boundary: false,
                protected: true,
            });
        }
        let normal = domain.normals(&[point])[0];
        if normal.length() <= f64::EPSILON {
            break;
        }
        point = point - normal * residual;
        let coords = space.coords(point);
        point = space.point(coords[0], coords[1]);
    }
    Ok(old)
}

fn validate_boundary_layer_strip(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    candidate: &Candidate,
    strip: &BoundaryLayerStrip,
) -> MeshResult<()> {
    if strip.cells.is_empty()
        || strip
            .cells
            .iter()
            .any(|cell| cell.points.len() != 4 || !cell.protected)
    {
        return Err(layer_error(
            domain,
            "did not produce a complete protected quad4 strip",
        ));
    }
    for (index, cell) in strip.cells.iter().enumerate() {
        if index.is_multiple_of(256) {
            context.check()?;
        }
        let positions = cell
            .points
            .iter()
            .map(|key| candidate.points[key].world)
            .collect::<Vec<_>>();
        let size = maximum_edge_2d(&positions);
        let quality =
            quality_score("quad4", &positions, QualityMetric::ScaledJacobian).unwrap_or(0.0);
        if signed_area_polygon(&cell.points, &candidate.points) <= orientation_tolerance(size)
            || quality <= VALID_QUALITY
            || polygon_self_intersects(&cell.points, &candidate.points)
        {
            return Err(layer_error(
                domain,
                "contains an inverted, crossed, or degenerate quad",
            ));
        }
    }
    for (&key, &distance) in &strip.levels {
        let point = Vec3::from_array(candidate.points[&key].world);
        let residual = (domain.domain_sdf(&[point])[0] + distance).abs();
        let tolerance = root_tolerance(domain, distance.max(context.target_size));
        if residual > tolerance {
            return Err(layer_error(
                domain,
                &format!(
                    "could not preserve requested SDF height {distance:.6e} at {:?} (residual={residual:.6e}, tolerance={tolerance:.6e})",
                    point.to_array(),
                ),
            ));
        }
    }
    if cells_edges_cross(candidate, &strip.cells) {
        return Err(layer_error(
            domain,
            "rows or columns self-intersect at a concave boundary feature",
        ));
    }
    Ok(())
}

fn rebuild_constrained_core(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    strip: BoundaryLayerStrip,
    constraints: &BTreeSet<(PointKey, PointKey)>,
) -> MeshResult<()> {
    let fixed = constraints
        .iter()
        .flat_map(|edge| [edge.0, edge.1])
        .collect::<BTreeSet<_>>();
    let strip_points = strip
        .cells
        .iter()
        .flat_map(|cell| cell.points.iter().copied())
        .collect::<BTreeSet<_>>();
    let retained = candidate
        .points
        .iter()
        .filter_map(|(key, point)| {
            if fixed.contains(key) || strip_points.contains(key) {
                return Some(*key);
            }
            let in_strip = point_in_strip(point.uv, &strip.cells, &candidate.points);
            let near_front = strip.front_edges.iter().any(|edge| {
                let a = Vec3::from_array(candidate.points[&edge[0]].world);
                let b = Vec3::from_array(candidate.points[&edge[1]].world);
                point_segment_distance(Vec3::from_array(point.world), a, b) < 0.9 * (b - a).length()
            });
            (!in_strip && !near_front).then_some(*key)
        })
        .collect::<BTreeSet<_>>();
    candidate.points.retain(|key, _| retained.contains(key));
    candidate
        .layer_edge_targets
        .retain(|(a, b), _| candidate.points.contains_key(a) && candidate.points.contains_key(b));

    for (index, edge) in strip.front_edges.iter().enumerate() {
        if index.is_multiple_of(256) {
            context.check()?;
        }
        let a = candidate.points[&edge[0]];
        let b = candidate.points[&edge[1]];
        let midpoint_uv = [(a.uv[0] + b.uv[0]) * 0.5, (a.uv[1] + b.uv[1]) * 0.5];
        let midpoint = space.point(midpoint_uv[0], midpoint_uv[1]);
        let height = 0.5 * 3.0_f64.sqrt() * distance3(a.world, b.world);
        let normal = domain.normals(&[midpoint])[0];
        let world = if normal.length() > f64::EPSILON {
            midpoint - normal * (height / normal.length())
        } else {
            let delta = [b.uv[0] - a.uv[0], b.uv[1] - a.uv[1]];
            let length = delta[0].hypot(delta[1]);
            space.point(
                midpoint_uv[0] - delta[1] * height / length,
                midpoint_uv[1] + delta[0] * height / length,
            )
        };
        let coords = space.coords(world);
        let uv = [coords[0], coords[1]];
        let sdf = domain.domain_sdf(&[world])[0];
        if !sdf.is_finite()
            || sdf >= domain.domain_sdf(&[midpoint])[0]
            || point_in_strip(uv, &strip.cells, &candidate.points)
        {
            return Err(layer_error(
                domain,
                "has insufficient clearance for a target-sized triangular collar",
            ));
        }
        let duplicate_tolerance = root_tolerance(domain, context.target_size);
        if candidate
            .points
            .values()
            .any(|point| distance3(point.world, world.to_array()) <= duplicate_tolerance)
        {
            continue;
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
    }

    let mut leaves = BTreeMap::new();
    for cell in &candidate.cells {
        for key in &cell.points {
            leaves.entry(*key).or_insert(cell.leaf);
        }
    }
    let mut triangulation = ConstrainedDelaunayTriangulation::<SpadeVertex>::new();
    let mut handles = BTreeMap::new();
    for (&key, point) in &candidate.points {
        let handle = triangulation
            .insert(SpadeVertex {
                position: Point2::new(point.uv[0], point.uv[1]),
                key,
            })
            .map_err(|error| {
                MeshError::InvalidInput(format!(
                    "Spade rejected a constrained core vertex in domain {:?}: {error:?}",
                    domain.name
                ))
            })?;
        handles.insert(key, handle);
    }
    for &(a, b) in constraints {
        triangulation.try_add_constraint(handles[&a], handles[&b]);
    }

    let mut cells = Vec::new();
    for (index, face) in triangulation.inner_faces().enumerate() {
        if index.is_multiple_of(512) {
            context.check()?;
        }
        let points = face.vertices().map(|vertex| vertex.data().key);
        let positions = points.map(|key| candidate.points[&key].world);
        let centroid = centroid_slice(&positions);
        let centroid_uv = space.coords(Vec3::from_array(centroid));
        if point_in_strip(
            [centroid_uv[0], centroid_uv[1]],
            &strip.cells,
            &candidate.points,
        ) || domain.domain_sdf(&[Vec3::from_array(centroid)])[0] >= 0.0
            || cell_containment_residual(domain, &positions)
                > chord_tolerance(domain, maximum_edge_2d(&positions))
        {
            continue;
        }
        let leaf = points
            .iter()
            .find_map(|key| leaves.get(key))
            .copied()
            .unwrap_or(Leaf {
                level: 0,
                x: 0,
                y: 0,
            });
        cells.push(Cell::triangle(points, leaf));
    }
    if cells.is_empty() {
        return Err(layer_error(
            domain,
            "constrained triangulation produced no core triangles",
        ));
    }
    cells.extend(strip.cells);
    if cells.len() > usize::try_from(context.limits.max_cells).unwrap_or(usize::MAX) {
        return Err(MeshError::LimitExceeded(format!(
            "constrained boundary-layer mesh exceeds the configured {} cell limit",
            context.limits.max_cells
        )));
    }
    candidate.cells = cells;
    let used = candidate
        .cells
        .iter()
        .flat_map(|cell| cell.points.iter().copied())
        .collect::<BTreeSet<_>>();
    candidate.points.retain(|key, _| used.contains(key));
    candidate.construction_failures.clear();
    Ok(())
}

fn point_in_strip(uv: [f64; 2], cells: &[Cell], points: &BTreeMap<PointKey, Point>) -> bool {
    cells.iter().any(|cell| {
        (0..cell.points.len()).all(|edge| {
            cross_2d(
                points[&cell.points[edge]].uv,
                points[&cell.points[(edge + 1) % cell.points.len()]].uv,
                uv,
            ) >= -1.0e-12
        })
    })
}

fn cells_edges_cross(candidate: &Candidate, cells: &[Cell]) -> bool {
    let mut edges = BTreeMap::<(PointKey, PointKey), ([f64; 2], [f64; 2])>::new();
    for cell in cells {
        for index in 0..cell.points.len() {
            let a = cell.points[index];
            let b = cell.points[(index + 1) % cell.points.len()];
            edges
                .entry(ordered_pair(a, b))
                .or_insert((candidate.points[&a].uv, candidate.points[&b].uv));
        }
    }
    let edges = edges.into_iter().collect::<Vec<_>>();
    for first in 0..edges.len() {
        let ((a_key, b_key), (a, b)) = edges[first];
        for ((c_key, d_key), (c, d)) in edges.iter().skip(first + 1).copied() {
            if a_key == c_key || a_key == d_key || b_key == c_key || b_key == d_key {
                continue;
            }
            if segments_cross(a, b, c, d) {
                return true;
            }
        }
    }
    false
}

fn layer_point(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    previous: Point,
    boundary: Point,
    inward: [f64; 2],
    step: f64,
    cumulative_distance: f64,
) -> MeshResult<Point> {
    let tolerance = root_tolerance(domain, cumulative_distance).max(cumulative_distance * 1.0e-8);
    let target_value = |point: Vec3| domain.domain_sdf(&[point])[0] + cumulative_distance;
    let mut near = Vec3::from_array(previous.world);
    let mut near_value = target_value(near);
    if !near_value.is_finite() {
        return Err(layer_error(domain, "encountered a non-finite SDF value"));
    }
    if near_value <= 0.0 {
        near = Vec3::from_array(boundary.world);
        near_value = target_value(near);
    }

    let mut bracket = None;
    for factor in [1.0, 1.25, 1.5, 2.0, 3.0, 4.0] {
        for uv in [
            [
                previous.uv[0] + inward[0] * step * factor,
                previous.uv[1] + inward[1] * step * factor,
            ],
            [
                boundary.uv[0] + inward[0] * cumulative_distance * factor,
                boundary.uv[1] + inward[1] * cumulative_distance * factor,
            ],
        ] {
            let point = space.point(uv[0], uv[1]);
            if (point - Vec3::from_array(boundary.world)).length()
                > 2.0 * cumulative_distance + tolerance
            {
                continue;
            }
            let value = target_value(point);
            if !value.is_finite() {
                return Err(layer_error(domain, "encountered a non-finite SDF value"));
            }
            if value <= 0.0 {
                bracket = Some((point, value));
                break;
            }
        }
        if bracket.is_some() {
            break;
        }
    }
    let Some((mut far, mut far_value)) = bracket else {
        return Err(layer_error(
            domain,
            "collides with another boundary or requests excessive thickness",
        ));
    };

    for _ in 0..64 {
        let middle = (near + far) * 0.5;
        let value = target_value(middle);
        if !value.is_finite() {
            return Err(layer_error(domain, "encountered a non-finite SDF value"));
        }
        if value.abs() <= tolerance || (near - far).length() <= tolerance {
            near = middle;
            near_value = value;
            break;
        }
        if value > 0.0 {
            near = middle;
            near_value = value;
        } else {
            far = middle;
            far_value = value;
        }
    }
    let point = if near_value.abs() <= far_value.abs() {
        near
    } else {
        far
    };
    let residual = target_value(point).abs();
    if residual > tolerance || domain.domain_sdf(&[point])[0] >= 0.0 {
        return Err(layer_error(
            domain,
            "could not bracket an exact requested SDF height",
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

fn core_quality(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    candidate: &Candidate,
) -> CoreQuality {
    let mut combined = Vec::new();
    let mut minimum_scaled_jacobian: f64 = 1.0;
    for (index, cell) in candidate.cells.iter().enumerate() {
        if cell.protected || cell.points.len() != 3 {
            continue;
        }
        let positions = cell
            .points
            .iter()
            .map(|key| candidate.points[key].world)
            .collect::<Vec<_>>();
        let scaled_jacobian =
            quality_score("tri3", &positions, QualityMetric::ScaledJacobian).unwrap_or(0.0);
        minimum_scaled_jacobian = minimum_scaled_jacobian.min(scaled_jacobian);
        let mut squared_log_size = 0.0;
        for edge in 0..3 {
            let a = positions[edge];
            let b = positions[(edge + 1) % 3];
            let midpoint = midpoint3(a, b);
            let length = distance3(a, b);
            let probes = [
                Vec3::from_array(a),
                Vec3::from_array(b),
                Vec3::from_array(midpoint),
            ];
            let target = candidate
                .layer_edge_targets
                .get(&ordered_pair(
                    cell.points[edge],
                    cell.points[(edge + 1) % 3],
                ))
                .copied()
                .unwrap_or_else(|| {
                    local_target(
                        candidate,
                        context,
                        &domain.name,
                        Vec3::from_array(midpoint),
                        length * 0.5,
                        &probes,
                    )
                });
            squared_log_size += (length / target).max(f64::MIN_POSITIVE).ln().powi(2);
        }
        let shape_distortion = 1.0 - scaled_jacobian;
        let size_distortion = (squared_log_size / 3.0).sqrt();
        combined.push((index, shape_distortion.hypot(size_distortion)));
    }
    combined.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let objective = if combined.is_empty() {
        0.0
    } else {
        (combined
            .iter()
            .map(|(_, distortion)| distortion.powi(8))
            .sum::<f64>()
            / combined.len() as f64)
            .powf(0.125)
    };
    CoreQuality {
        objective,
        minimum_scaled_jacobian,
        worst_first: combined,
    }
}

fn quality_actions(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    candidate: &Candidate,
    cell_index: usize,
) -> BTreeSet<Action> {
    let mut actions = BTreeSet::new();
    let Some(cell) = candidate
        .cells
        .get(cell_index)
        .filter(|cell| !cell.protected && cell.points.len() == 3)
    else {
        return actions;
    };
    let mut oversized = false;
    for edge in 0..3 {
        let a = cell.points[edge];
        let b = cell.points[(edge + 1) % 3];
        actions.insert(Action::Flip(ordered_pair(a, b).0, ordered_pair(a, b).1));
        let aw = candidate.points[&a].world;
        let bw = candidate.points[&b].world;
        let midpoint = midpoint3(aw, bw);
        let length = distance3(aw, bw);
        let probes = [
            Vec3::from_array(aw),
            Vec3::from_array(bw),
            Vec3::from_array(midpoint),
        ];
        let target = candidate
            .layer_edge_targets
            .get(&ordered_pair(a, b))
            .copied()
            .unwrap_or_else(|| {
                local_target(
                    candidate,
                    context,
                    &domain.name,
                    Vec3::from_array(midpoint),
                    length * 0.5,
                    &probes,
                )
            });
        let ratio = length / target;
        if ratio < EDGE_RATIO_MIN {
            actions.insert(Action::Collapse(ordered_pair(a, b).0, ordered_pair(a, b).1));
        } else if ratio > EDGE_RATIO_MAX {
            oversized = true;
            actions.insert(Action::Split(ordered_pair(a, b).0, ordered_pair(a, b).1));
        }
    }
    for &point in &cell.points {
        if !candidate.points[&point].boundary && !candidate.points[&point].protected {
            for step in 0..4 {
                actions.insert(Action::RelocateInterior(point, step));
            }
        }
    }
    if oversized {
        actions.insert(Action::Insert(cell_index));
    }
    actions
}

fn estimated_optimization_bytes(candidate: &Candidate) -> usize {
    candidate
        .points
        .len()
        .saturating_mul(160)
        .saturating_add(candidate.cells.len().saturating_mul(128))
        .saturating_mul(2)
}

fn record_quality_termination(statistics: &mut MeshingStatistics, termination: QualityTermination) {
    if statistics.quality_termination == QualityTermination::NotRun
        || statistics.quality_termination == QualityTermination::Converged
            && termination != QualityTermination::Converged
    {
        statistics.quality_termination = termination;
    }
}

fn optimize(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &mut Assessment,
    statistics: &mut MeshingStatistics,
) -> MeshResult<()> {
    let initial = core_quality(domain, context, candidate);
    if initial.worst_first.is_empty() {
        record_quality_termination(statistics, QualityTermination::Converged);
        return Ok(());
    }
    let mut previous = initial.objective;
    for pass in 0..MAX_QUALITY_PASSES {
        context.check()?;
        if estimated_optimization_bytes(candidate) > MAX_OPTIMIZATION_BYTES {
            record_quality_termination(statistics, QualityTermination::MemoryBudget);
            return Ok(());
        }
        let current = core_quality(domain, context, candidate);
        let mut best = None::<(Candidate, Assessment, CoreQuality)>;
        let mut trials = 0usize;
        let mut max_cells_limited = false;
        'scan: for &(cell_index, _) in &current.worst_first {
            for action in quality_actions(domain, context, candidate, cell_index) {
                if trials >= action_budget(candidate.cells.len(), context.limits.max_cells) {
                    break;
                }
                trials += 1;
                if matches!(action, Action::Insert(_) | Action::Split(_, _))
                    && candidate.cells.len().saturating_add(2) as u64 > context.limits.max_cells
                {
                    max_cells_limited = true;
                    continue;
                }
                let mut trial = candidate.clone();
                if !apply_action(domain, space, context, &mut trial, action)? {
                    continue;
                }
                let trial_assessment = assess(domain, space, context, &trial)?;
                if !trial_assessment.refine.is_empty()
                    || trial_assessment.score.hard_invalid != 0
                    || boundary_owners(assessment) != boundary_owners(&trial_assessment)
                {
                    continue;
                }
                let trial_quality = core_quality(domain, context, &trial);
                let objective_decreased = trial_quality.objective
                    < current.objective - 1.0e-12 * current.objective.max(1.0);
                let minimum_preserved = trial_quality.minimum_scaled_jacobian + 1.0e-12
                    >= current.minimum_scaled_jacobian;
                if !objective_decreased || !minimum_preserved {
                    continue;
                }
                if best.as_ref().is_none_or(|(_, _, quality)| {
                    trial_quality.objective < quality.objective
                        || trial_quality.objective == quality.objective
                            && trial_quality.minimum_scaled_jacobian
                                > quality.minimum_scaled_jacobian
                }) {
                    best = Some((trial, trial_assessment, trial_quality));
                    break 'scan;
                }
            }
            if trials >= action_budget(candidate.cells.len(), context.limits.max_cells) {
                break;
            }
        }

        let Some((improved, improved_assessment, quality)) = best else {
            record_quality_termination(
                statistics,
                if max_cells_limited || candidate.cells.len() as u64 >= context.limits.max_cells {
                    QualityTermination::MaxCells
                } else {
                    QualityTermination::Converged
                },
            );
            return Ok(());
        };
        *candidate = improved;
        *assessment = improved_assessment;
        statistics.quality_passes += 1;
        let improvement = previous - quality.objective;
        let total_improvement = initial.objective - quality.objective;
        previous = quality.objective;
        if pass > 0 && improvement <= CONVERGENCE_THRESHOLD * total_improvement {
            record_quality_termination(statistics, QualityTermination::Converged);
            return Ok(());
        }
    }
    record_quality_termination(statistics, QualityTermination::IterationLimit);
    Ok(())
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

fn apply_action(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    action: Action,
) -> MeshResult<bool> {
    match action {
        Action::Flip(a, b) => Ok(apply_flip(candidate, a, b)),
        Action::RelocateInterior(key, step) => {
            apply_interior_relocation(domain, space, context, candidate, key, step)
        }
        Action::Split(a, b) => apply_split(domain, space, context, candidate, a, b),
        Action::Insert(index) => Ok(apply_insert(domain, space, context, candidate, index)),
        Action::Collapse(a, b) => apply_collapse(domain, space, candidate, a, b),
    }
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
    let mut uv = {
        let coords = space.coords(world);
        [coords[0], coords[1]]
    };
    if boundary {
        let a_uv = candidate.points[&a].uv;
        let b_uv = candidate.points[&b].uv;
        let projected = project_boundary_station(
            domain,
            space,
            context.target_size,
            world.to_array(),
            [b_uv[0] - a_uv[0], b_uv[1] - a_uv[1]],
        )?;
        world = Vec3::from_array(projected.world);
        uv = projected.uv;
    }
    let key = PointKey::Inserted(candidate.next_inserted);
    candidate.next_inserted += 1;
    candidate.points.insert(
        key,
        Point {
            uv,
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
    shared_boundary_points: &mut Vec<SharedBoundaryPoint>,
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
    let mut positions = BTreeMap::<PointKey, [f64; 3]>::new();
    let shared_tolerance = root_tolerance(domain, context.target_size);
    let previous_domain_points = shared_boundary_points.len();
    for (&key, chunks) in &uses {
        let point = candidate.points[&key];
        if assessment.boundary_vertices.contains(&key) {
            if let Some(shared) = shared_boundary_points[..previous_domain_points]
                .iter()
                .find(|shared| distance3(shared.position, point.world) <= shared_tolerance)
            {
                ids.insert(key, shared.id);
                positions.insert(key, shared.position);
                continue;
            }
        }
        let owner = *chunks.first().expect("used point has a chunk");
        let ordinal = ordinals[owner];
        ordinals[owner] = ordinal
            .checked_add(1)
            .ok_or_else(|| MeshError::LimitExceeded("2D point ID space exhausted".into()))?;
        let id = MeshId::from_raw((u64::from(chunk_ids[owner]) << 32) | u64::from(ordinal));
        ids.insert(key, id);
        positions.insert(key, point.world);
        if assessment.boundary_vertices.contains(&key) {
            shared_boundary_points.push(SharedBoundaryPoint {
                position: point.world,
                id,
            });
        }
    }
    let catalog = context.catalog.domain(&domain.name)?;
    for (chunk_index, &chunk_id) in chunk_ids.iter().enumerate() {
        context.check()?;
        let start = chunk_index * cells_per_chunk;
        let end = (start + cells_per_chunk).min(candidate.cells.len());
        let spade_tile = constrained_spade_tile(candidate, &candidate.cells[start..end])?;
        let used = candidate.cells[start..end]
            .iter()
            .flat_map(|cell| cell.points.iter().copied())
            .collect::<BTreeSet<_>>();
        let bounds = Bounds3::from_points(used.iter().map(|key| positions[key]))
            .expanded(root_tolerance(domain, context.target_size));
        let mut builder = MeshChunkBuilder::new(chunk_id, bounds)?;
        for key in &used {
            builder.point_copy(
                ids[key],
                positions[key],
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
        let active = (chunk.decoded_bytes() + estimated_spade_bytes(&spade_tile)) as u64;
        if active > context.limits.target_chunk_bytes as u64 {
            return Err(MeshError::LimitExceeded(format!(
                "DistMesh chunk {chunk_id} active Spade tile and writer batch require {active} bytes, exceeding the configured {} byte chunk target",
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
            phase_total: chunk_count as u64,
            completed_chunks: statistics.chunks,
            cells_committed: statistics.cells,
            active_bytes: active,
        });
    }
    Ok(())
}

fn constrained_spade_tile(
    candidate: &Candidate,
    cells: &[Cell],
) -> MeshResult<ConstrainedDelaunayTriangulation<Point2<f64>>> {
    let used = cells
        .iter()
        .flat_map(|cell| cell.points.iter().copied())
        .collect::<BTreeSet<_>>();
    let mut tile = ConstrainedDelaunayTriangulation::new();
    let mut vertices = BTreeMap::new();
    for key in used {
        let [a, b] = candidate.points[&key].uv;
        let vertex = tile.insert(Point2::new(a, b)).map_err(|error| {
            MeshError::InvalidInput(format!("Spade rejected a DistMesh tile vertex: {error:?}"))
        })?;
        vertices.insert(key, vertex);
    }
    for cell in cells {
        for edge in 0..cell.points.len() {
            let a = cell.points[edge];
            let b = cell.points[(edge + 1) % cell.points.len()];
            tile.try_add_constraint(vertices[&a], vertices[&b]);
        }
    }
    Ok(tile)
}

fn estimated_spade_bytes(tile: &ConstrainedDelaunayTriangulation<Point2<f64>>) -> usize {
    std::mem::size_of_val(tile)
        + tile.num_vertices() * 128
        + tile.num_undirected_edges() * 96
        + tile.num_inner_faces() * 64
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
            let positions = triangle.map(|point| points[&point].world);
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
    root_tolerance(domain, local_size).max((local_size * 0.05).min(domain.boundary_tolerance()))
}

fn topology_tolerance(domain: &MeshableDomain, local_size: f64) -> f64 {
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
        "domain {:?} could not produce a valid 2D mesh at target size {:.6e} near ({:.6}, {:.6}, {:.6}): {reason}; worst Scaled Jacobian={quality:.6e}, required > {VALID_QUALITY:.1e} and optimization target {QUALITY_TARGET:.2}",
        domain.name, context.target_size, location[0], location[1], location[2],
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
        _minimum_size: f64,
        target_size: f64,
    ) -> MemoryArtifact {
        let mut controls = ControlSet::default();
        controls.target_size(target_size).unwrap();
        let output = run_meshing(
            MeshingRequest {
                domains: meshable_domains_from_document(document).expect("meshable domains"),
                algorithm_id: "distmesh".into(),
                controls,
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
        _minimum_size: f64,
        target_size: f64,
    ) -> MeshResult<()> {
        let domains = meshable_domains_from_document(document)
            .map_err(|error| MeshError::InvalidInput(error.to_string()))?;
        let controls = ControlSet::default();
        let job_control = JobControl::default();
        let catalog = MeshCatalog::from_domains(&domains, "distmesh");
        let context = MeshingContext {
            domains: &domains,
            target_size,
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
        _minimum_size: f64,
        target_size: f64,
        controls: &ControlSet,
        limits: GenerationLimits,
    ) -> MeshResult<TestSink> {
        let domains = meshable_domains_from_document(document)
            .map_err(|error| MeshError::InvalidInput(error.to_string()))?;
        let job_control = JobControl::default();
        let catalog = MeshCatalog::from_domains(&domains, "distmesh");
        let context = MeshingContext {
            domains: &domains,
            target_size,
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
    fn quality_termination_is_backward_compatible_and_reported() {
        let legacy: MeshingStatistics = serde_json::from_str(
            r#"{"domains":1,"chunks":1,"points":3,"cells":1,"committed_batches":1,"peak_active_bytes":0,"elapsed_millis":0}"#,
        )
        .unwrap();
        assert_eq!(legacy.quality_passes, 0);
        assert_eq!(legacy.quality_termination, QualityTermination::NotRun);

        let document = primitive_document("square", vec3(-0.75, -0.75, 0.0), vec3(0.75, 0.75, 0.0));
        let mut controls = ControlSet::default();
        controls.target_size(0.3).unwrap();
        let output = run_meshing(
            MeshingRequest {
                domains: meshable_domains_from_document(&document).unwrap(),
                algorithm_id: "distmesh".into(),
                controls,
                limits: GenerationLimits::default(),
                job_control: JobControl::default(),
            },
            MemoryStorage::new(16 * 1024 * 1024).unwrap(),
        )
        .unwrap();
        assert_eq!(
            output.statistics.quality_termination,
            QualityTermination::Converged
        );
        assert!(output.statistics.quality_passes <= MAX_QUALITY_PASSES as u64);
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
            layer_end_targets: Vec::new(),
        };
        let original = trial.cells[0].points.clone();
        assert!(!apply_flip(&mut trial, a, b));
        assert_eq!(trial.cells[0].points, original);
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
            let mut cells = 0;
            for chunk in &sink.chunks {
                let points = chunk
                    .points
                    .iter()
                    .map(|point| (point.id, point.position))
                    .collect::<BTreeMap<_, _>>();
                for cell in &chunk.cells {
                    cells += 1;
                    assert_eq!(
                        cell.element_type, "tri3",
                        "an unlayered core should remain triangular"
                    );
                    let positions = cell
                        .point_ids
                        .iter()
                        .map(|id| points[id])
                        .collect::<Vec<_>>();
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
            assert!(cells > 0, "the domain should contain core triangles");
        }
    }

    #[test]
    fn uniform_and_growing_boundary_layers_emit_protected_quad_rows() {
        for growth in [1.0, 1.5] {
            let (document, region) = controlled_rectangle();
            let mut controls = ControlSet::default();
            controls
                .boundary_layer(
                    "rectangle",
                    region,
                    0.04,
                    0.2,
                    growth,
                    0.04 * (1.0 + growth),
                )
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
            .boundary_layer("rectangle", region, 0.035, 0.18, 1.2, 0.1274)
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
            .boundary_layer("circle", region, 0.025, 0.12, 1.25, 0.05625)
            .unwrap();
        let sink = mesh_chunks(
            &document,
            0.0125,
            0.12,
            &controls,
            GenerationLimits::default(),
        )
        .unwrap();
        let domains = meshable_domains_from_document(&document).expect("circle domain");
        let domain = domains.iter().next().expect("circle domain");
        assert!(sink
            .chunks
            .iter()
            .flat_map(|chunk| &chunk.cells)
            .any(|cell| cell.element_type == "quad4"));
        let mut tangential_lengths = Vec::new();
        for chunk in &sink.chunks {
            let points = chunk
                .points
                .iter()
                .map(|point| (point.id, point.position))
                .collect::<BTreeMap<_, _>>();
            tangential_lengths.extend(chunk.edges.iter().filter(|edge| edge.boundary).map(
                |edge| {
                    assert!(chunk.cells.iter().any(|cell| {
                        cell.element_type == "quad4"
                            && (0..cell.point_ids.len()).any(|index| {
                                ordered_pair(
                                    cell.point_ids[index],
                                    cell.point_ids[(index + 1) % cell.point_ids.len()],
                                ) == ordered_pair(edge.point_ids[0], edge.point_ids[1])
                            })
                    }));
                    distance3(points[&edge.point_ids[0]], points[&edge.point_ids[1]])
                },
            ));
        }
        let shortest = tangential_lengths
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let longest = tangential_lengths.iter().copied().fold(0.0, f64::max);
        let mean = tangential_lengths.iter().sum::<f64>() / tangential_lengths.len() as f64;
        assert!(tangential_lengths.len() > 20);
        assert!(
            longest / shortest < 1.001,
            "boundary spacing: {shortest}..{longest}"
        );
        assert!((mean - 0.12).abs() < 0.012, "mean boundary spacing: {mean}");

        let points = sink
            .chunks
            .iter()
            .flat_map(|chunk| &chunk.points)
            .map(|point| (point.id, point.position))
            .collect::<BTreeMap<_, _>>();
        let cells = sink
            .chunks
            .iter()
            .flat_map(|chunk| &chunk.cells)
            .collect::<Vec<_>>();
        let levels = [0.0, 0.025, 0.025 + 0.025 * 1.25];
        let tolerance = root_tolerance(domain, 0.12);
        let mut front_edges = BTreeSet::new();
        for cell in cells.iter().filter(|cell| cell.element_type == "quad4") {
            for &point in &cell.point_ids {
                let distance = -domain.domain_sdf(&[Vec3::from_array(points[&point])])[0];
                assert!(
                    levels
                        .iter()
                        .any(|level| (distance - level).abs() <= tolerance),
                    "quad point has unexpected SDF distance {distance}"
                );
            }
            for edge in 0..4 {
                let pair = ordered_pair(cell.point_ids[edge], cell.point_ids[(edge + 1) % 4]);
                if [pair.0, pair.1].into_iter().all(|point| {
                    (domain.domain_sdf(&[Vec3::from_array(points[&point])])[0] + levels[2]).abs()
                        <= tolerance
                }) {
                    front_edges.insert(pair);
                }
            }
        }
        assert!(!front_edges.is_empty());
        for edge in front_edges {
            let triangle = cells
                .iter()
                .find(|cell| {
                    cell.element_type == "tri3"
                        && cell.point_ids.contains(&edge.0)
                        && cell.point_ids.contains(&edge.1)
                })
                .expect("each layer-front edge has a core triangle");
            let shortest = (0..3)
                .map(|index| {
                    distance3(
                        points[&triangle.point_ids[index]],
                        points[&triangle.point_ids[(index + 1) % 3]],
                    )
                })
                .fold(f64::INFINITY, f64::min);
            assert!(
                shortest >= EDGE_RATIO_MIN * 0.12,
                "undersized front triangle edge {shortest}"
            );
        }
    }

    #[test]
    fn hole_boundary_stations_project_from_the_negative_sdf_side() {
        let mut document = SceneDocument::new();
        let outer = document
            .add_primitive_from_drag("rectangle", vec3(-1.5, -1.0, 0.0), vec3(1.5, 1.0, 0.0), 1.0)
            .expect("outer sea boundary");
        let wall = document
            .add_primitive_from_drag(
                "circle",
                vec3(-0.55, -0.55, 0.0),
                vec3(0.55, 0.55, 0.0),
                1.0,
            )
            .expect("circular wall");
        let sea = document
            .combine(outer, wall, "difference")
            .expect("sea minus wall");
        document.rename(sea, "sea").expect("rename sea");
        document
            .set_domain_root(sea, DomainKind::Fluid)
            .expect("mark sea domain");
        document
            .add_boundary_region(sea, None, None, Some("wall"))
            .expect("wall boundary region");
        let region = document
            .boundary_regions
            .last()
            .expect("wall region")
            .name
            .clone();
        let mut controls = ControlSet::default();
        controls
            .boundary_layer("sea", region, 0.025, 0.12, 1.25, 0.05625)
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
        let mut soft_tangential_targets = ControlSet::default();
        soft_tangential_targets
            .boundary_layer("rectangle", &regions[0], 0.03, 0.12, 1.0, 0.06)
            .unwrap();
        soft_tangential_targets
            .boundary_layer("rectangle", &regions[1], 0.03, 0.24, 1.0, 0.06)
            .unwrap();
        mesh_chunks(
            &document,
            0.015,
            0.18,
            &soft_tangential_targets,
            GenerationLimits::default(),
        )
        .expect("adjacent regions may use different soft hwall_t targets");

        let mut incompatible = ControlSet::default();
        incompatible
            .boundary_layer("rectangle", &regions[0], 0.03, 0.18, 1.0, 0.06)
            .unwrap();
        incompatible
            .boundary_layer("rectangle", &regions[1], 0.04, 0.18, 1.2, 0.1456)
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
            .boundary_layer("rectangle", region, 0.6, 0.2, 1.0, 1.2)
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
    fn deferred_refinement_does_not_change_the_uniform_private_core() {
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
        assert_eq!(refined.cells, background.cells);

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
            let mut controls = ControlSet::default();
            controls.target_size(0.01).unwrap();
            run_meshing(
                MeshingRequest {
                    domains,
                    algorithm_id: "distmesh".into(),
                    controls,
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
