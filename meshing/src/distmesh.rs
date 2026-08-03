use std::collections::{BTreeMap, BTreeSet};

use caso_kernel::meshing::{BoundaryBand, MeshableDomain, MeshableDomainSpace, MeshableInterface};
use caso_kernel::vec3::Vec3;
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

use crate::algorithm::{
    MeshAlgorithm, MeshAlgorithmCapabilities, MeshAlgorithmDescriptor, MeshSink, MeshingContext,
    MeshingPhase, MeshingProgress, MeshingStatistics, QualityTermination,
};
use crate::chunk::{MeshChunkBuilder, MeshId};
use crate::controls::BoundaryLayerControl;
use crate::error::{MeshError, MeshResult};
use crate::quality::{quality_score, QualityMetric};
use crate::schema::Bounds3;

mod audit;
mod cdt;
mod contour;
mod optimizer;

const QUALITY_TARGET: f64 = 0.40;
const VALID_QUALITY: f64 = 1.0e-8;
const EDGE_RATIO_MIN: f64 = 0.65;
const EDGE_RATIO_MAX: f64 = std::f64::consts::SQRT_2 * 1.0001;
const SNAP_RATIO: f64 = 0.06;
const ESTIMATED_CHUNK_BYTES_PER_CELL: usize = 2_048;
const MAX_QUALITY_PASSES: usize = 64;
const LAYER_TRANSITION_GROWTH: f64 = 1.30;
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
        self.sample_many(&[key])?;
        Ok(self.cache[&key])
    }

    fn sample_many(&mut self, keys: &[Lattice]) -> MeshResult<()> {
        let missing = keys
            .iter()
            .copied()
            .filter(|key| !self.cache.contains_key(key))
            .collect::<BTreeSet<_>>();
        if missing.is_empty() {
            return Ok(());
        }
        let samples = missing
            .iter()
            .map(|key| {
                let uv = self.grid.uv(*key);
                let world = self.space.point(uv[0], uv[1]);
                (*key, uv, world)
            })
            .collect::<Vec<_>>();
        let values = self
            .domain
            .domain_sdf(&samples.iter().map(|sample| sample.2).collect::<Vec<_>>());
        for ((key, uv, world), sdf) in samples.into_iter().zip(values) {
            if !sdf.is_finite() {
                return Err(MeshError::InvalidInput(format!(
                    "domain {:?} returned a non-finite SDF value",
                    self.domain.name
                )));
            }
            self.cache.insert(
                key,
                Sample {
                    key,
                    uv,
                    world: world.to_array(),
                    sdf,
                },
            );
        }
        Ok(())
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
        self.sample_many(&keys)?;
        Ok(keys.map(|key| self.cache[&key]))
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

type PointSpade = ConstrainedDelaunayTriangulation<Point2<f64>>;
type PointSpadeVertices = BTreeMap<PointKey, spade::handles::FixedVertexHandle>;

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
    layer_front_targets: Vec<LayerFrontTarget>,
    layer_end_targets: Vec<LayerEndTarget>,
    layer_refinement_limit: Option<QualityTermination>,
    protected_constraints: BTreeSet<(PointKey, PointKey)>,
    refine_layer_core: bool,
}

#[derive(Debug, Clone, Copy)]
struct LayerFrontTarget {
    a: [f64; 3],
    b: [f64; 3],
    edge_length: f64,
}

#[derive(Debug, Clone, Copy)]
struct LayerEndTarget {
    edge: (PointKey, PointKey),
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
struct CapQuality {
    average_scaled_jacobian: f64,
    minimum_scaled_jacobian: f64,
    transition_minimum_scaled_jacobian: f64,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct LayerKey {
    hwall_n: u64,
    layers: usize,
    ratio: u64,
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
            hwall_n: control.hwall_n.to_bits(),
            layers: control.layers,
            ratio: control.ratio.to_bits(),
        }
    }

    fn hwall_n(self) -> f64 {
        f64::from_bits(self.hwall_n)
    }

    fn ratio(self) -> f64 {
        f64::from_bits(self.ratio)
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
            if splittable.is_empty() {
                break (candidate, assessment);
            }
            refine_leaves(context, &mut leaves, &splittable)?;
            balance(context, grid, &mut leaves)?;
        };
        let mut canonical = candidate.clone();
        match contour::canonicalize_boundary_vertices(
            domain,
            context,
            &mut canonical,
            &assessment.boundary,
        ) {
            Ok(changed) => {
                candidate = canonical;
                if changed {
                    assessment = assess(domain, &space, context, &candidate)?;
                }
            }
            Err(MeshError::InvalidInput(_)) => {
                // Canonicalization improves the global CDT input but is not a
                // mandatory geometry operation. Keep the already valid SDF
                // construction transactionally when the inferred graph is
                // ambiguous instead of exposing an internal graph diagnostic.
            }
            Err(error) => return Err(error),
        }
        let before_interfaces = candidate.clone();
        install_shared_interfaces(
            domain,
            &space,
            context,
            &mut candidate,
            &mut assessment,
            &shared_interfaces,
        )?;
        let cdt_result =
            retriangulate_with_spade(domain, &space, context, &mut candidate, &assessment, true);
        let mut cdt_valid = cdt_result.is_ok();
        if let Err(error) = cdt_result {
            if !matches!(error, MeshError::InvalidInput(_)) {
                return Err(error);
            }
        }
        if cdt_valid {
            assessment = assess(domain, &space, context, &candidate)?;
            cdt_valid = assessment.refine.is_empty() && assessment.score.hard_invalid == 0;
        }
        if !cdt_valid {
            candidate = before_interfaces;
            assessment = assess(domain, &space, context, &candidate)?;
            if !assessment.refine.is_empty() || assessment.score.hard_invalid != 0 {
                return Err(minimum_size_error(
                    domain,
                    context,
                    assessment
                        .reason
                        .as_deref()
                        .unwrap_or("no topology-valid construction snapshot is available"),
                    assessment.location,
                    assessment.worst_quality,
                ));
            }
        }
        let has_layers = context
            .controls
            .boundary_layers
            .iter()
            .any(|control| control.domain == domain.name);
        let before_layers = has_layers.then(|| candidate.clone());
        let statistics_before_layers = has_layers.then(|| statistics.clone());
        if has_layers {
            prepare_layer_boundaries(domain, &space, context, &mut candidate, &mut assessment)?;
            apply_boundary_layers(domain, &space, context, &mut candidate, &mut assessment)?;
        }
        lock_constraint_vertices(&mut candidate, &assessment);
        optimizer::optimize(
            domain,
            &space,
            context,
            &mut candidate,
            &mut assessment,
            &mut statistics,
        )?;
        if has_layers && !optimizer::quality_gates_met(domain, context, &candidate) {
            let attempted_termination = statistics.quality_termination;
            candidate = before_layers.expect("layer fallback snapshot");
            candidate.refine_layer_core = false;
            statistics = statistics_before_layers.expect("layer statistics snapshot");
            if matches!(
                attempted_termination,
                QualityTermination::MaxCells | QualityTermination::MemoryBudget
            ) {
                statistics.quality_termination = attempted_termination;
            }
            assessment = assess(domain, &space, context, &candidate)?;
            prepare_layer_boundaries(domain, &space, context, &mut candidate, &mut assessment)?;
            apply_boundary_layers(domain, &space, context, &mut candidate, &mut assessment)?;
            lock_constraint_vertices(&mut candidate, &assessment);
            optimizer::optimize(
                domain,
                &space,
                context,
                &mut candidate,
                &mut assessment,
                &mut statistics,
            )?;
        }
        sort_cells_morton(&space, &mut candidate);
        assessment = assess(domain, &space, context, &candidate)?;
        audit::validate_candidate(domain, context, &candidate, &assessment)?;
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

fn lock_constraint_vertices(candidate: &mut Candidate, assessment: &Assessment) {
    for edge in &assessment.boundary {
        let constraint = ordered_pair(edge.points[0], edge.points[1]);
        candidate.protected_constraints.insert(constraint);
        for point in edge.points {
            if let Some(point) = candidate.points.get_mut(&point) {
                point.protected = true;
            }
        }
    }
    let protected_cells = candidate
        .cells
        .iter()
        .filter(|cell| cell.protected)
        .flat_map(|cell| {
            (0..cell.points.len()).map(|edge| {
                ordered_pair(
                    cell.points[edge],
                    cell.points[(edge + 1) % cell.points.len()],
                )
            })
        })
        .collect::<Vec<_>>();
    candidate.protected_constraints.extend(protected_cells);
}

fn interface_edge(interface: &MeshableInterface, a: [f64; 3], b: [f64; 3]) -> bool {
    interface
        .contains(&[Vec3::from_array(a), Vec3::from_array(b)])
        .into_iter()
        .all(|hit| hit)
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
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &Assessment,
    refine: bool,
) -> MeshResult<()> {
    for key in assessment
        .boundary
        .iter()
        .flat_map(|edge| edge.points)
        .collect::<BTreeSet<_>>()
    {
        let point = candidate.points[&key];
        let projected = domain.project_to_boundary_owner(&[Vec3::from_array(point.world)])[0];
        let replacement = (projected.converged
            && projected.distance_moved <= context.target_size
            && domain.domain_sdf(&[projected.point])[0].abs()
                <= chord_tolerance(domain, context.target_size))
        .then(|| {
            let coords = space.coords(projected.point);
            ([coords[0], coords[1]], projected.point.to_array())
        });
        let point = candidate
            .points
            .get_mut(&key)
            .expect("assessed boundary vertex");
        if !point.protected {
            if let Some((uv, world)) = replacement {
                point.uv = uv;
                point.world = world;
            }
        }
        point.boundary = true;
    }
    let graph = contour::PlanarConstraintGraph::from_boundary(
        domain,
        context,
        candidate,
        &assessment.boundary,
    )?;
    cdt::retriangulate(domain, space, context, candidate, &graph, refine)
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
    let target = candidate.layer_front_targets.iter().fold(
        regional_target(context, domain, center, radius, probes),
        |target, front| {
            target.min(
                front.edge_length
                    + point_segment_distance(
                        center,
                        Vec3::from_array(front.a),
                        Vec3::from_array(front.b),
                    ),
            )
        },
    );
    candidate
        .layer_end_targets
        .iter()
        .fold(target, |target, end| {
            target.min(
                end.edge_length
                    + point_segment_distance(
                        center,
                        Vec3::from_array(end.a),
                        Vec3::from_array(end.b),
                    ),
            )
        })
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
        layer_front_targets: Vec::new(),
        layer_end_targets: Vec::new(),
        layer_refinement_limit: None,
        protected_constraints: BTreeSet::new(),
        refine_layer_core: true,
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
        points.insert(key, exact_boundary_point(domain, space, local_size, a));
        crossings.insert(edge, key);
        return Ok(key);
    }
    if b.sdf == 0.0 {
        let key = PointKey::Lattice(b.key);
        points.insert(key, exact_boundary_point(domain, space, local_size, b));
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
    let mut world = if projection.converged && projection.distance_moved <= local_size {
        projection.point
    } else {
        Vec3::from_array(inside.world)
    };
    let owner_projection = domain.project_to_boundary_owner(&[world])[0];
    if owner_projection.converged
        && owner_projection.distance_moved <= local_size
        && domain.domain_sdf(&[owner_projection.point])[0].abs()
            <= chord_tolerance(domain, local_size)
    {
        world = owner_projection.point;
    }
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

fn exact_boundary_point(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    local_size: f64,
    sample: Sample,
) -> Point {
    let seed = Vec3::from_array(sample.world);
    let projection = domain.project_to_boundary_owner(&[seed])[0];
    let world = if projection.converged && projection.distance_moved <= local_size {
        projection.point
    } else {
        seed
    };
    let coords = space.coords(world);
    Point {
        uv: [coords[0], coords[1]],
        world: world.to_array(),
        boundary: true,
        protected: false,
    }
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
            let reason = format!(
                "cell is inverted, self-intersecting, or degenerate (area={area:.6e}, Scaled Jacobian={quality:.6e}, keys={:?}, points={positions:?})",
                cell.points
            );
            record_refinement(
                &mut assessment,
                cell.leaf,
                &reason,
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
        } else if let [(first_cell, first), (second_cell, second)] = entries.as_slice() {
            if first == second {
                for cell in [first_cell, second_cell] {
                    assessment.score.hard_invalid += 1;
                    record_refinement(
                        &mut assessment,
                        candidate.cells[*cell].leaf,
                        "adjacent cells traverse their shared edge in the same direction",
                        cell_centroid(candidate, *cell),
                    );
                }
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
        let a = candidate.points[&edge.points[0]].world;
        let b = candidate.points[&edge.points[1]].world;
        let memberships = layer_memberships(domain, context, midpoint3(a, b), distance3(a, b))?;
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
        // Project through the authored boundary region, not through the
        // target-size-dependent owner classification of the coarse chord.
        let owner = memberships
            .iter()
            .map(|index| {
                context.controls.boundary_layers[*index]
                    .boundary_region
                    .clone()
            })
            .next();
        replaced.insert(ordered_pair(edge.points[0], edge.points[1]));
        groups
            .entry(LayerContourKey {
                layer,
                tangential_size: memberships
                    .iter()
                    .map(|index| context.controls.boundary_layers[*index].hwall_t)
                    .fold(f64::INFINITY, f64::min)
                    .to_bits(),
                owner,
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
    retriangulate_with_spade(domain, space, context, candidate, assessment, false)?;
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
    let mut vertices = if closed {
        path[..path.len() - 1].to_vec()
    } else {
        path.to_vec()
    };
    if vertices.len() < usize::from(closed) + 2 {
        return Err(layer_error(
            domain,
            "controlled contour is too short to resample",
        ));
    }
    pin_curve_feature_points(
        domain,
        space,
        context,
        candidate,
        contour.owner.as_deref(),
        &mut vertices,
        closed,
    )?;
    let mut fixed = vertices
        .iter()
        .enumerate()
        .filter_map(|(index, point)| candidate.points[point].protected.then_some(index))
        .collect::<BTreeSet<_>>();
    if !closed {
        fixed.insert(0);
        fixed.insert(vertices.len() - 1);
    }
    // A coarse sampling of a smooth curve can have large turning angles.
    // Treating those angles as CAD corners makes hwall_t depend on target_size.
    // Region transitions were split above; only protected/open endpoints stay fixed.
    if closed && fixed.is_empty() {
        let mut cycle = vertices.clone();
        cycle.push(vertices[0]);
        let stations =
            resample_boundary_arc(domain, space, context, candidate, &cycle, true, contour)?;
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

    let mut linear = vertices.clone();
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
            contour,
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

#[allow(clippy::too_many_arguments)]
fn pin_curve_feature_points(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    owner: Option<&str>,
    vertices: &mut Vec<PointKey>,
    closed: bool,
) -> MeshResult<()> {
    let Some(owner) = owner else {
        return Ok(());
    };
    let features = domain
        .region_by_name(owner)
        .map_err(|error| MeshError::InvalidInput(error.to_string()))?
        .curve_feature_points()
        .map_err(|error| MeshError::InvalidInput(error.to_string()))?;
    let tolerance = chord_tolerance(domain, context.target_size).max(context.target_size * 0.25);
    for feature in features {
        context.check()?;
        let world = feature.to_array();
        let Some((vertex_index, vertex_distance)) = vertices
            .iter()
            .enumerate()
            .map(|(index, key)| (index, distance3(candidate.points[key].world, world)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
        else {
            continue;
        };
        if vertex_distance <= tolerance {
            let coords = space.coords(feature);
            let point = candidate
                .points
                .get_mut(&vertices[vertex_index])
                .expect("feature vertex");
            point.uv = [coords[0], coords[1]];
            point.world = world;
            point.boundary = true;
            point.protected = true;
            continue;
        }

        let edge_count = vertices.len() - usize::from(!closed);
        let nearest_edge = (0..edge_count)
            .map(|index| {
                let a = Vec3::from_array(candidate.points[&vertices[index]].world);
                let b = Vec3::from_array(
                    candidate.points[&vertices[(index + 1) % vertices.len()]].world,
                );
                (index, point_segment_distance(feature, a, b))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if nearest_edge.is_none_or(|(_, distance)| distance > tolerance) {
            continue;
        }
        let coords = space.coords(feature);
        let key = PointKey::Inserted(candidate.next_inserted);
        candidate.next_inserted += 1;
        candidate.points.insert(
            key,
            Point {
                uv: [coords[0], coords[1]],
                world,
                boundary: true,
                protected: true,
            },
        );
        vertices.insert(nearest_edge.expect("nearest feature edge").0 + 1, key);
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
    contour: &LayerContourKey,
) -> MeshResult<Vec<PointKey>> {
    let total = path
        .windows(2)
        .map(|edge| {
            distance3(
                candidate.points[&edge[0]].world,
                candidate.points[&edge[1]].world,
            )
        })
        .sum::<f64>();
    let minimum = if closed { 3 } else { 1 };
    let nominal = ((total / contour.tangential_size()).round() as usize).max(minimum);
    let mut counts = Vec::with_capacity(5);
    for delta in [0isize, -1, 1, -2, 2] {
        let edge_count = nominal.saturating_add_signed(delta).max(minimum);
        if !counts.contains(&edge_count) {
            counts.push(edge_count);
        }
    }
    let mut best = None::<(Candidate, Vec<PointKey>, (f64, f64, f64), usize)>;
    for edge_count in counts {
        context.check()?;
        let mut trial = candidate.clone();
        let stations = match resample_boundary_arc_with_count(
            domain,
            space,
            context,
            &mut trial,
            path,
            closed,
            contour.tangential_size(),
            contour.owner.as_deref(),
            edge_count,
        ) {
            Ok(stations) => stations,
            Err(MeshError::InvalidInput(_)) => continue,
            Err(error) => return Err(error),
        };
        let Some(score) =
            score_boundary_station_rows(domain, space, &trial, &stations, closed, contour)
        else {
            continue;
        };
        let better = best.as_ref().is_none_or(|(_, _, current, current_count)| {
            score
                .0
                .total_cmp(&current.0)
                .then_with(|| current.1.total_cmp(&score.1))
                .then_with(|| score.2.total_cmp(&current.2))
                .then_with(|| edge_count.cmp(current_count))
                .is_lt()
        });
        if better {
            best = Some((trial, stations, score, edge_count));
        }
    }
    let Some((best_candidate, stations, _, _)) = best else {
        return Err(layer_error(
            domain,
            &format!(
                "could not construct valid boundary stations for {:?}",
                contour.owner
            ),
        ));
    };
    *candidate = best_candidate;
    Ok(stations)
}

#[allow(clippy::too_many_arguments)]
fn resample_boundary_arc_with_count(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    path: &[PointKey],
    closed: bool,
    tangential_size: f64,
    owner: Option<&str>,
    edge_count: usize,
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
        let point = project_boundary_station(domain, space, owner, tangential_size, seed, tangent)?;
        let key = PointKey::Inserted(candidate.next_inserted);
        candidate.next_inserted += 1;
        candidate.points.insert(key, point);
        base.push(key);
    }
    redistribute_boundary_stations(
        domain,
        space,
        owner,
        tangential_size,
        candidate,
        &base,
        closed,
    )?;

    let mut stations = vec![base[0]];
    for pair in base.windows(2) {
        append_boundary_chord(
            domain,
            space,
            tangential_size,
            owner,
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
            owner,
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

fn score_boundary_station_rows(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    candidate: &Candidate,
    stations: &[PointKey],
    closed: bool,
    contour: &LayerContourKey,
) -> Option<(f64, f64, f64)> {
    let edge_count = if closed {
        stations.len()
    } else {
        stations.len().saturating_sub(1)
    };
    if edge_count == 0 {
        return None;
    }
    let mut distances = vec![0.0];
    let mut height = contour.layer.hwall_n();
    for _ in 0..contour.layer.layers {
        distances.push(distances.last().copied().unwrap_or(0.0) + height);
        height *= contour.layer.ratio();
    }
    let mut rows = Vec::<Vec<Point>>::with_capacity(stations.len());
    for index in 0..stations.len() {
        let previous = if index == 0 {
            if closed {
                stations.len() - 1
            } else {
                0
            }
        } else {
            index - 1
        };
        let next = if index + 1 == stations.len() {
            if closed {
                0
            } else {
                index
            }
        } else {
            index + 1
        };
        let before = candidate.points[&stations[previous]].uv;
        let after = candidate.points[&stations[next]].uv;
        let tangent = [after[0] - before[0], after[1] - before[1]];
        let length = tangent[0].hypot(tangent[1]);
        if length <= f64::EPSILON {
            return None;
        }
        let direction = [-tangent[1] / length, tangent[0] / length];
        let source = candidate.points[&stations[index]];
        let mut row = vec![source];
        let mut previous = source;
        for level in 1..distances.len() {
            let point = match layer_point(
                domain,
                space,
                previous,
                source,
                direction,
                distances[level] - distances[level - 1],
                distances[level],
            ) {
                Ok(point) => point,
                Err(_) => return None,
            };
            row.push(point);
            previous = point;
        }
        rows.push(row);
    }

    let mut quads = Vec::new();
    let mut tangential = Vec::new();
    for edge in 0..edge_count {
        let next = (edge + 1) % stations.len();
        tangential.push(distance3(rows[edge][0].world, rows[next][0].world));
        for level in 0..contour.layer.layers {
            quads.push(vec![
                rows[edge][level].world,
                rows[next][level].world,
                rows[next][level + 1].world,
                rows[edge][level + 1].world,
            ]);
        }
    }
    let mean = tangential.iter().sum::<f64>() / tangential.len() as f64;
    let mut worst_skewness: f64 = 0.0;
    let mut minimum_scaled_jacobian: f64 = 1.0;
    for quad in &quads {
        let jacobian = quality_score("quad4", quad, QualityMetric::ScaledJacobian)?;
        let skewness = quality_score("quad4", quad, QualityMetric::Skewness)?;
        if jacobian <= VALID_QUALITY || !skewness.is_finite() {
            return None;
        }
        worst_skewness = worst_skewness.max(skewness);
        minimum_scaled_jacobian = minimum_scaled_jacobian.min(jacobian);
    }
    (!quads.is_empty()).then_some((
        worst_skewness,
        minimum_scaled_jacobian,
        (mean / contour.tangential_size())
            .max(f64::MIN_POSITIVE)
            .ln()
            .abs(),
    ))
}

fn redistribute_boundary_stations(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    owner: Option<&str>,
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
                project_boundary_station(domain, space, owner, target_size, seed, tangent)?,
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
    owner: Option<&str>,
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
        let point = contour::project_to_owner(domain, owner, seed, target_size)?;
        let coords = space.coords(point);
        return Ok(Point {
            uv: [coords[0], coords[1]],
            world: point.to_array(),
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
    let root = if domain.domain_sdf(&[interior])[0].abs() <= domain.domain_sdf(&[exterior])[0].abs()
    {
        interior
    } else {
        exterior
    };
    let point = match contour::project_to_owner(domain, owner, root, target_size) {
        Ok(point) if domain.domain_sdf(&[point])[0].abs() <= tolerance => point,
        Ok(_) => root,
        Err(_) if owner.is_none() => {
            // The sign bracket already locates the wall to root tolerance.
            // The Newton projector is useful at smooth points but is allowed
            // to stop at C0 SDF seams, so keep the certified root there.
            root
        }
        Err(error) => return Err(error),
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
    owner: Option<&str>,
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
        owner,
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
        stations.push(b);
        return Ok(());
    }
    let middle = PointKey::Inserted(candidate.next_inserted);
    candidate.next_inserted += 1;
    candidate.points.insert(middle, projection);
    append_boundary_chord(
        domain,
        space,
        target_size,
        owner,
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
        owner,
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
            let a = Vec3::from_array(candidate.points[&edge.points[0]].world);
            let b = Vec3::from_array(candidate.points[&edge.points[1]].world);
            if let Some((parameter, owner)) =
                contour::first_layer_transition(domain, context, a, b)?
            {
                split = Some((edge.points, parameter, owner));
                break;
            }
        }
        let Some(([a, b], parameter, owner)) = split else {
            break;
        };
        if pass == 127
            || !split_boundary_transition(
                domain, space, context, candidate, a, b, parameter, &owner,
            )?
        {
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
        if !layer_memberships(domain, context, midpoint3(a, b), distance3(a, b))?.is_empty() {
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

#[allow(clippy::too_many_arguments)]
fn split_boundary_transition(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    a: PointKey,
    b: PointKey,
    parameter: f64,
    owner: &str,
) -> MeshResult<bool> {
    let pair = edge_cells(candidate, a, b);
    if pair.len() != 1
        || candidate.cells[pair[0]].protected
        || candidate.cells[pair[0]].points.len() != 3
    {
        return Ok(false);
    }
    let pa = candidate.points[&a];
    let pb = candidate.points[&b];
    let size = distance3(pa.world, pb.world);
    let mut point = project_boundary_station(
        domain,
        space,
        Some(owner),
        size,
        lerp3(pa.world, pb.world, parameter),
        [pb.uv[0] - pa.uv[0], pb.uv[1] - pa.uv[1]],
    )?;
    point.protected = true;
    let key = PointKey::Inserted(candidate.next_inserted);
    candidate.next_inserted += 1;
    candidate.points.insert(key, point);

    let index = pair[0];
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
    if signed_area(first, &candidate.points) <= orientation_tolerance(size)
        || signed_area(second, &candidate.points) <= orientation_tolerance(size)
        || candidate.cells.len().saturating_add(1) as u64 > context.limits.max_cells
    {
        candidate.points.remove(&key);
        return Ok(false);
    }
    candidate.cells[index].points = first.into();
    candidate.cells.push(Cell::triangle(second, cell.leaf));
    Ok(true)
}

fn layer_memberships(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    point: [f64; 3],
    trust_distance: f64,
) -> MeshResult<BTreeSet<usize>> {
    let point = Vec3::from_array(point);
    let mut memberships = BTreeSet::new();
    for (index, control) in context.controls.boundary_layers.iter().enumerate() {
        if control.domain != domain.name {
            continue;
        }
        let region = domain
            .region_by_name(&control.boundary_region)
            .map_err(|error| MeshError::InvalidInput(error.to_string()))?;
        let projection = region.project_to_owner(&[point])[0];
        // The midpoint of a coarse curved chord can be farther from the CAD
        // wall than hwall_t; chord length is the target-independent trust scale.
        let trust = control
            .hwall_t
            .max(trust_distance)
            .min(domain.bounds.diagonal());
        if !projection.converged || projection.distance_moved > trust {
            continue;
        }
        let contains = region
            .contains(&[projection.point])
            .map_err(|error| MeshError::InvalidInput(error.to_string()))?[0];
        if contains {
            memberships.insert(index);
        }
    }
    Ok(memberships)
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
        let memberships = layer_memberships(domain, context, midpoint3(a, b), distance3(a, b))?;
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
    candidate.layer_front_targets = strip
        .front_edges
        .iter()
        .map(|edge| {
            let a = candidate.points[&edge[0]].world;
            let b = candidate.points[&edge[1]].world;
            LayerFrontTarget {
                a,
                b,
                edge_length: distance3(a, b),
            }
        })
        .collect();
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
    if added_cells > usize::try_from(context.limits.max_cells).unwrap_or(usize::MAX) {
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
    candidate.layer_refinement_limit = None;

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
        let mut height = key.hwall_n();
        for _ in 0..key.layers {
            distances.push(distances.last().copied().unwrap_or(0.0) + height);
            height *= key.ratio();
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
            // The contour tessellation is allowed to approximate a curved CAD
            // edge, but a boundary-layer column must start on the exact domain
            // zero level.  Reproject through the domain SDF before constructing
            // the immutable normal rows; this is geometry-agnostic and also
            // prevents a CAD-owner projection at a Boolean seam from leaking a
            // small chord residual into every row of the strip.
            let mut source = correct_sdf_level(domain, space, *source, source.uv, 0.0)?;
            source.boundary = true;
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

        // Keep the exact normal columns. Tangential smoothing of individual
        // rows can shear an otherwise orthogonal quad into a high-skew cell.

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
                    edge: ordered_pair(pair[0], pair[1]),
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
                point_segment_distance(Vec3::from_array(point.world), a, b)
                    < LAYER_TRANSITION_GROWTH * context.target_size
            });
            (!in_strip && !near_front).then_some(*key)
        })
        .collect::<BTreeSet<_>>();
    candidate.points.retain(|key, _| retained.contains(key));
    candidate
        .layer_edge_targets
        .retain(|(a, b), _| candidate.points.contains_key(a) && candidate.points.contains_key(b));

    let mut leaves = BTreeMap::new();
    for cell in &candidate.cells {
        for key in &cell.points {
            leaves.entry(*key).or_insert(cell.leaf);
        }
    }
    let mut triangulation_constraints = constraints.clone();
    candidate.cells = triangulate_with_front_transition(
        domain,
        space,
        context,
        candidate,
        &strip,
        &triangulation_constraints,
        &leaves,
    )?;
    candidate.construction_failures.clear();

    seed_cap_boundary_transitions(
        domain,
        space,
        context,
        candidate,
        &strip,
        &mut triangulation_constraints,
        &leaves,
    )?;

    'segments: for end in candidate.layer_end_targets.clone() {
        context.check()?;
        if estimated_optimization_bytes(candidate) > MAX_OPTIMIZATION_BYTES {
            candidate.layer_refinement_limit = Some(QualityTermination::MemoryBudget);
            break;
        }
        let boundary_adjacent =
            candidate.points[&end.edge.0].boundary || candidate.points[&end.edge.1].boundary;
        let mut best_boundary_seed =
            None::<(Candidate, BTreeSet<(PointKey, PointKey)>, CapQuality)>;
        for scale in [1.0, 0.75, 0.5, 0.25] {
            let Some(points) =
                cap_seed_chain(domain, space, context, candidate, &strip, end, scale)
            else {
                continue;
            };
            for length in (1..=points.len()).rev() {
                let mut trial = candidate.clone();
                let keys = points[..length]
                    .iter()
                    .map(|point| {
                        let key = PointKey::Inserted(trial.next_inserted);
                        trial.next_inserted += 1;
                        trial.points.insert(key, *point);
                        key
                    })
                    .collect::<Vec<_>>();
                let mut trial_constraints = triangulation_constraints.clone();
                trial_constraints.insert(ordered_pair(end.edge.0, keys[0]));
                trial_constraints.insert(ordered_pair(end.edge.1, keys[0]));
                trial_constraints
                    .extend(keys.windows(2).map(|pair| ordered_pair(pair[0], pair[1])));
                let cells = match triangulate_constrained_core(
                    domain,
                    space,
                    context,
                    &mut trial,
                    &strip,
                    &trial_constraints,
                    &leaves,
                ) {
                    Ok(cells) => cells,
                    Err(MeshError::InvalidInput(_)) => continue,
                    Err(error) => return Err(error),
                };
                if cells.len() > usize::try_from(context.limits.max_cells).unwrap_or(usize::MAX) {
                    candidate.layer_refinement_limit = Some(QualityTermination::MaxCells);
                    continue;
                }
                trial.cells = cells;
                trial.construction_failures.clear();
                let trial_assessment = assess(domain, space, context, &trial)?;
                let attached =
                    edge_cells(&trial, end.edge.0, end.edge.1)
                        .into_iter()
                        .any(|index| {
                            trial.cells[index].points.len() == 3
                                && trial.cells[index].points.contains(&keys[0])
                        });
                if attached
                    && trial_assessment.refine.is_empty()
                    && trial_assessment.score.hard_invalid == 0
                {
                    if !boundary_adjacent {
                        triangulation_constraints = trial_constraints;
                        *candidate = trial;
                        continue 'segments;
                    }
                    let quality = edge_cap_quality(&trial, end.edge)
                        .expect("attached cap seed must have a local quality patch");
                    if best_boundary_seed
                        .as_ref()
                        .is_none_or(|(_, _, best)| cap_seed_quality_is_better(&quality, best))
                    {
                        best_boundary_seed = Some((trial, trial_constraints, quality));
                    }
                    break;
                }
            }
        }
        if let Some((best, constraints, _)) = best_boundary_seed {
            *candidate = best;
            triangulation_constraints = constraints;
        }
    }

    let used = candidate
        .cells
        .iter()
        .flat_map(|cell| cell.points.iter().copied())
        .collect::<BTreeSet<_>>();
    candidate.points.retain(|key, _| used.contains(key));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn triangulate_with_front_transition(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    strip: &BoundaryLayerStrip,
    constraints: &BTreeSet<(PointKey, PointKey)>,
    leaves: &BTreeMap<PointKey, Leaf>,
) -> MeshResult<Vec<Cell>> {
    let cell_limit = usize::try_from(context.limits.max_cells).unwrap_or(usize::MAX);
    if estimated_optimization_bytes(candidate) > MAX_OPTIMIZATION_BYTES {
        candidate.layer_refinement_limit = Some(QualityTermination::MemoryBudget);
        let cells = triangulate_constrained_core(
            domain,
            space,
            context,
            candidate,
            strip,
            constraints,
            leaves,
        )?;
        if cells.len() <= cell_limit {
            return Ok(cells);
        }
        return Err(MeshError::LimitExceeded(format!(
            "constrained boundary-layer mesh exceeds the configured {} cell limit",
            context.limits.max_cells
        )));
    }

    let fallback = candidate.clone();
    seed_front_transition_rings(domain, space, context, candidate, strip)?;
    match triangulate_constrained_core(
        domain,
        space,
        context,
        candidate,
        strip,
        constraints,
        leaves,
    ) {
        Ok(cells) if cells.len() <= cell_limit => return Ok(cells),
        Ok(_) => {
            candidate.layer_refinement_limit = Some(QualityTermination::MaxCells);
        }
        Err(MeshError::InvalidInput(_)) => {}
        Err(error) => return Err(error),
    }

    let termination = candidate.layer_refinement_limit;
    *candidate = fallback;
    candidate.layer_refinement_limit = termination;
    let cells = triangulate_constrained_core(
        domain,
        space,
        context,
        candidate,
        strip,
        constraints,
        leaves,
    )?;
    if cells.len() > cell_limit {
        return Err(MeshError::LimitExceeded(format!(
            "constrained boundary-layer mesh exceeds the configured {} cell limit",
            context.limits.max_cells
        )));
    }
    Ok(cells)
}

fn seed_front_transition_rings(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    strip: &BoundaryLayerStrip,
) -> MeshResult<()> {
    let edges = strip
        .front_edges
        .iter()
        .map(|points| BoundaryEdge {
            points: *points,
            cell: 0,
            owner: None,
        })
        .collect::<Vec<_>>();
    let paths = ordered_boundary_paths(domain, &edges)?;
    let mut inserted = Vec::<Point>::new();

    for path in paths {
        let closed = path.first() == path.last();
        let edge_count = path.len().saturating_sub(1);
        if edge_count == 0 {
            continue;
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
        let front_spacing = total / edge_count as f64;
        if !front_spacing.is_finite() || front_spacing <= f64::EPSILON {
            continue;
        }

        let mut desired_spacing = front_spacing;
        let mut previous_spacing = front_spacing;
        let mut offset = 0.0;
        for ring in 0..8 {
            context.check()?;
            let samples = if closed {
                (total / desired_spacing).round().max(3.0) as usize
            } else {
                (total / desired_spacing).round().max(1.0) as usize
            };
            let spacing = total / samples as f64;
            offset += 0.25 * 3.0_f64.sqrt() * (previous_spacing + spacing);
            for sample in 0..samples {
                let distance = total * (sample as f64 + 0.5) / samples as f64;
                let (front, tangent) =
                    point_on_boundary_polyline(candidate, &path, &lengths, distance.min(total));
                let front_uv = space.coords(Vec3::from_array(front));
                let direction = core_side_normal(
                    domain,
                    space,
                    [front_uv[0], front_uv[1]],
                    tangent,
                    front_spacing,
                );
                let uv = [
                    front_uv[0] + direction[0] * offset,
                    front_uv[1] + direction[1] * offset,
                ];
                let world = space.point(uv[0], uv[1]);
                let sdf = domain.domain_sdf(&[world])[0];
                let front_sdf = domain.domain_sdf(&[Vec3::from_array(front)])[0];
                let separation = SNAP_RATIO * spacing;
                if !sdf.is_finite()
                    || sdf >= front_sdf - root_tolerance(domain, spacing)
                    || point_in_strip(uv, &strip.cells, &candidate.points)
                    || candidate
                        .points
                        .values()
                        .any(|point| distance3(point.world, world.to_array()) <= separation)
                    || inserted
                        .iter()
                        .any(|point| distance3(point.world, world.to_array()) <= separation)
                {
                    continue;
                }
                inserted.push(Point {
                    uv,
                    world: world.to_array(),
                    boundary: false,
                    protected: false,
                });
            }
            if front_spacing + offset >= context.target_size {
                break;
            }
            previous_spacing = spacing;
            desired_spacing = (desired_spacing * LAYER_TRANSITION_GROWTH).min(context.target_size);
            if ring == 7 {
                candidate.layer_refinement_limit = Some(QualityTermination::IterationLimit);
            }
        }
    }

    for point in inserted {
        let key = PointKey::Inserted(candidate.next_inserted);
        candidate.next_inserted += 1;
        candidate.points.insert(key, point);
    }
    Ok(())
}

fn core_side_normal(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    uv: [f64; 2],
    tangent: [f64; 2],
    scale: f64,
) -> [f64; 2] {
    let length = tangent[0].hypot(tangent[1]);
    if length <= f64::EPSILON {
        return [0.0; 2];
    }
    let normal = [-tangent[1] / length, tangent[0] / length];
    let probe = 0.25 * scale;
    let left = space.point(uv[0] + probe * normal[0], uv[1] + probe * normal[1]);
    let right = space.point(uv[0] - probe * normal[0], uv[1] - probe * normal[1]);
    let values = domain.domain_sdf(&[left, right]);
    if values[0] <= values[1] {
        normal
    } else {
        [-normal[0], -normal[1]]
    }
}

fn seed_cap_boundary_transitions(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    strip: &BoundaryLayerStrip,
    constraints: &mut BTreeSet<(PointKey, PointKey)>,
    leaves: &BTreeMap<PointKey, Leaf>,
) -> MeshResult<()> {
    for column in &strip.end_columns {
        context.check()?;
        let [base, first, ..] = column.as_slice() else {
            continue;
        };
        let desired = (strip.levels[first] - strip.levels[base])
            .abs()
            .min(context.target_size);
        let adjacent_edge = |constraints: &BTreeSet<(PointKey, PointKey)>| {
            constraints.iter().copied().find(|edge| {
                (edge.0 == *base || edge.1 == *base) && !strip.constraints.contains(edge)
            })
        };
        let Some(initial_edge) = adjacent_edge(constraints) else {
            continue;
        };
        let mut edge = Some(initial_edge);
        loop {
            let Some(current_edge) = edge else {
                break;
            };
            let edge_length = distance3(
                candidate.points[&current_edge.0].world,
                candidate.points[&current_edge.1].world,
            );
            if edge_length >= EDGE_RATIO_MIN * desired
                || !merge_short_cap_boundary_edge(
                    domain,
                    space,
                    context,
                    candidate,
                    strip,
                    constraints,
                    leaves,
                    *base,
                    current_edge,
                )?
            {
                break;
            }
            context.check()?;
            edge = adjacent_edge(constraints);
        }
        let Some(edge) = edge else {
            continue;
        };
        let stations =
            graded_boundary_stations(domain, space, context, candidate, edge, *base, desired)?;
        if stations.is_empty() {
            continue;
        }
        if estimated_optimization_bytes(candidate) > MAX_OPTIMIZATION_BYTES {
            candidate.layer_refinement_limit = Some(QualityTermination::MemoryBudget);
            return Ok(());
        }

        let mut trial = candidate.clone();
        let mut chain = Vec::with_capacity(stations.len() + 2);
        chain.push(*base);
        for point in stations {
            let key = PointKey::Inserted(trial.next_inserted);
            trial.next_inserted += 1;
            trial.points.insert(key, point);
            chain.push(key);
        }
        let other = if edge.0 == *base { edge.1 } else { edge.0 };
        chain.push(other);

        let mut trial_constraints = constraints.clone();
        trial_constraints.remove(&edge);
        trial_constraints.extend(chain.windows(2).map(|pair| ordered_pair(pair[0], pair[1])));
        if let Some(target) = trial.layer_edge_targets.remove(&edge) {
            for pair in chain.windows(2) {
                trial
                    .layer_edge_targets
                    .insert(ordered_pair(pair[0], pair[1]), target);
            }
        }

        let cells = match triangulate_constrained_core(
            domain,
            space,
            context,
            &mut trial,
            strip,
            &trial_constraints,
            leaves,
        ) {
            Ok(cells) => cells,
            Err(MeshError::InvalidInput(_)) => continue,
            Err(error) => return Err(error),
        };
        if cells.len() > usize::try_from(context.limits.max_cells).unwrap_or(usize::MAX) {
            candidate.layer_refinement_limit = Some(QualityTermination::MaxCells);
            return Ok(());
        }
        trial.cells = cells;
        trial.construction_failures.clear();
        let assessment = assess(domain, space, context, &trial)?;
        if assessment.refine.is_empty() && assessment.score.hard_invalid == 0 {
            *constraints = trial_constraints;
            *candidate = trial;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn merge_short_cap_boundary_edge(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    strip: &BoundaryLayerStrip,
    constraints: &mut BTreeSet<(PointKey, PointKey)>,
    leaves: &BTreeMap<PointKey, Leaf>,
    base: PointKey,
    edge: (PointKey, PointKey),
) -> MeshResult<bool> {
    let remove = if edge.0 == base { edge.1 } else { edge.0 };
    let Some(point) = candidate.points.get(&remove) else {
        return Ok(false);
    };
    let incident = constraints
        .iter()
        .copied()
        .filter(|constraint| constraint.0 == remove || constraint.1 == remove)
        .collect::<Vec<_>>();
    if point.protected
        || !point.boundary
        || incident.len() != 2
        || incident.iter().any(|edge| strip.constraints.contains(edge))
    {
        return Ok(false);
    }
    let Some(next) = incident
        .iter()
        .flat_map(|edge| [edge.0, edge.1])
        .find(|point| *point != remove && *point != base)
    else {
        return Ok(false);
    };

    let mut trial = candidate.clone();
    trial.points.remove(&remove);
    trial
        .layer_edge_targets
        .retain(|(a, b), _| *a != remove && *b != remove);
    let mut trial_constraints = constraints.clone();
    trial_constraints.retain(|constraint| constraint.0 != remove && constraint.1 != remove);
    trial_constraints.insert(ordered_pair(base, next));
    let cells = match triangulate_constrained_core(
        domain,
        space,
        context,
        &mut trial,
        strip,
        &trial_constraints,
        leaves,
    ) {
        Ok(cells) => cells,
        Err(MeshError::InvalidInput(_)) => return Ok(false),
        Err(error) => return Err(error),
    };
    trial.cells = cells;
    trial.construction_failures.clear();
    let assessment = assess(domain, space, context, &trial)?;
    if !assessment.refine.is_empty() || assessment.score.hard_invalid != 0 {
        return Ok(false);
    }
    *candidate = trial;
    *constraints = trial_constraints;
    Ok(true)
}

fn graded_boundary_stations(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &Candidate,
    edge: (PointKey, PointKey),
    base: PointKey,
    first_size: f64,
) -> MeshResult<Vec<Point>> {
    if !first_size.is_finite() || first_size <= f64::EPSILON {
        return Ok(Vec::new());
    }
    let other = if edge.0 == base { edge.1 } else { edge.0 };
    let start = candidate.points[&base];
    let end = candidate.points[&other];
    let length = distance3(start.world, end.world);
    let midpoint_uv = [
        0.5 * (start.uv[0] + end.uv[0]),
        0.5 * (start.uv[1] + end.uv[1]),
    ];
    let tangent = interior_left_tangent(
        domain,
        space,
        midpoint_uv,
        [end.uv[0] - start.uv[0], end.uv[1] - start.uv[1]],
        first_size,
    );
    let tolerance = root_tolerance(domain, first_size);
    let mut stations = Vec::new();
    let mut previous = start.world;

    if length <= tolerance {
        return Ok(stations);
    }
    let desired_spacing = (LAYER_TRANSITION_GROWTH * first_size).min(context.target_size);
    let segments = (length / desired_spacing).ceil().clamp(1.0, 16.0) as usize;
    if segments < 2 {
        return Ok(stations);
    }
    let spacing = length / segments as f64;

    for station in 1..segments {
        context.check()?;
        let point = match project_boundary_station(
            domain,
            space,
            None,
            first_size,
            lerp3(start.world, end.world, station as f64 / segments as f64),
            tangent,
        ) {
            Ok(point) => point,
            Err(MeshError::Cancelled) => return Err(MeshError::Cancelled),
            Err(_) => return Ok(Vec::new()),
        };
        let projected_spacing = distance3(previous, point.world);
        if projected_spacing < spacing / LAYER_TRANSITION_GROWTH
            || projected_spacing > spacing * LAYER_TRANSITION_GROWTH
            || distance3(point.world, start.world) <= tolerance
            || distance3(point.world, end.world) <= tolerance
            || candidate
                .points
                .values()
                .any(|existing| distance3(existing.world, point.world) <= tolerance)
            || stations
                .iter()
                .any(|existing: &Point| distance3(existing.world, point.world) <= tolerance)
        {
            return Ok(Vec::new());
        }
        previous = point.world;
        stations.push(point);
    }
    let final_spacing = distance3(previous, end.world);
    if final_spacing < spacing / LAYER_TRANSITION_GROWTH
        || final_spacing > spacing * LAYER_TRANSITION_GROWTH
    {
        return Ok(Vec::new());
    }
    Ok(stations)
}

fn interior_left_tangent(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    uv: [f64; 2],
    tangent: [f64; 2],
    scale: f64,
) -> [f64; 2] {
    let length = tangent[0].hypot(tangent[1]);
    if length <= f64::EPSILON {
        return tangent;
    }
    let normal = [-tangent[1] / length, tangent[0] / length];
    let probe = 0.25 * scale;
    let left = space.point(uv[0] + probe * normal[0], uv[1] + probe * normal[1]);
    let right = space.point(uv[0] - probe * normal[0], uv[1] - probe * normal[1]);
    let values = domain.domain_sdf(&[left, right]);
    if values[0] <= values[1] {
        tangent
    } else {
        [-tangent[0], -tangent[1]]
    }
}

fn triangulate_constrained_core(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    strip: &BoundaryLayerStrip,
    constraints: &BTreeSet<(PointKey, PointKey)>,
    leaves: &BTreeMap<PointKey, Leaf>,
) -> MeshResult<Vec<Cell>> {
    cdt::triangulate_core(
        domain,
        space,
        context,
        candidate,
        strip,
        constraints,
        leaves,
    )
}

fn cap_seed_point(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &Candidate,
    strip: &BoundaryLayerStrip,
    end: LayerEndTarget,
    scale: f64,
) -> Option<Point> {
    let a = candidate.points.get(&end.edge.0)?;
    let b = candidate.points.get(&end.edge.1)?;
    let quad = strip
        .cells
        .iter()
        .find(|cell| cell.points.contains(&end.edge.0) && cell.points.contains(&end.edge.1))?;
    let strip_center = quad
        .points
        .iter()
        .map(|key| candidate.points[key].uv)
        .fold([0.0; 2], |sum, uv| [sum[0] + uv[0], sum[1] + uv[1]])
        .map(|value| value / quad.points.len() as f64);
    let midpoint_uv = [(a.uv[0] + b.uv[0]) * 0.5, (a.uv[1] + b.uv[1]) * 0.5];
    let delta = [b.uv[0] - a.uv[0], b.uv[1] - a.uv[1]];
    let length = delta[0].hypot(delta[1]);
    if length <= f64::EPSILON {
        return None;
    }
    let mut normal = [-delta[1] / length, delta[0] / length];
    if (strip_center[0] - midpoint_uv[0]) * normal[0]
        + (strip_center[1] - midpoint_uv[1]) * normal[1]
        > 0.0
    {
        normal = [-normal[0], -normal[1]];
    }
    let midpoint = midpoint3(a.world, b.world);
    let probes = [
        Vec3::from_array(a.world),
        Vec3::from_array(b.world),
        Vec3::from_array(midpoint),
    ];
    let target = local_target(
        candidate,
        context,
        &domain.name,
        Vec3::from_array(midpoint),
        0.5 * end.edge_length,
        &probes,
    );
    let height = 0.5 * 3.0_f64.sqrt() * target * scale;
    let uv = [
        midpoint_uv[0] + normal[0] * height,
        midpoint_uv[1] + normal[1] * height,
    ];
    let world = space.point(uv[0], uv[1]);
    let sdf = domain.domain_sdf(&[world])[0];
    let separation = SNAP_RATIO * target.min(end.edge_length);
    if !sdf.is_finite()
        || sdf >= 0.0
        || point_in_strip(uv, &strip.cells, &candidate.points)
        || candidate
            .points
            .values()
            .any(|point| distance3(point.world, world.to_array()) <= separation)
    {
        return None;
    }
    Some(Point {
        uv,
        world: world.to_array(),
        boundary: false,
        protected: false,
    })
}

fn cap_seed_chain(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &Candidate,
    strip: &BoundaryLayerStrip,
    end: LayerEndTarget,
    scale: f64,
) -> Option<Vec<Point>> {
    let first = cap_seed_point(domain, space, context, candidate, strip, end, scale)?;
    let a = candidate.points.get(&end.edge.0)?;
    let b = candidate.points.get(&end.edge.1)?;
    if a.boundary || b.boundary {
        return Some(vec![first]);
    }
    let midpoint_uv = [0.5 * (a.uv[0] + b.uv[0]), 0.5 * (a.uv[1] + b.uv[1])];
    let cap_direction = [first.uv[0] - midpoint_uv[0], first.uv[1] - midpoint_uv[1]];
    let cap_length = cap_direction[0].hypot(cap_direction[1]);
    let layer_a = space.coords(Vec3::from_array(end.a));
    let layer_b = space.coords(Vec3::from_array(end.b));
    let layer_direction = [layer_b[0] - layer_a[0], layer_b[1] - layer_a[1]];
    let layer_length = layer_direction[0].hypot(layer_direction[1]);
    if cap_length <= f64::EPSILON || layer_length <= f64::EPSILON {
        return None;
    }
    let direction = [
        cap_direction[0] / cap_length + layer_direction[0] / layer_length,
        cap_direction[1] / cap_length + layer_direction[1] / layer_length,
    ];
    let direction_length = direction[0].hypot(direction[1]);
    if direction_length <= f64::EPSILON {
        return None;
    }
    let direction = [
        direction[0] / direction_length,
        direction[1] / direction_length,
    ];
    let mut points = vec![first];
    let mut spacing = (end.edge_length * LAYER_TRANSITION_GROWTH * scale).min(context.target_size);

    for _ in 1..8 {
        let current = *points.last()?;
        let current_world = Vec3::from_array(current.world);
        let probes = [
            Vec3::from_array(a.world),
            Vec3::from_array(b.world),
            current_world,
        ];
        let target = local_target(
            candidate,
            context,
            &domain.name,
            current_world,
            spacing,
            &probes,
        );
        if target >= context.target_size * (1.0 - 1.0e-12) {
            break;
        }
        let uv = [
            current.uv[0] + direction[0] * spacing,
            current.uv[1] + direction[1] * spacing,
        ];
        let world = space.point(uv[0], uv[1]);
        let separation = SNAP_RATIO * spacing.min(target);
        if domain.domain_sdf(&[world])[0] >= 0.0
            || point_in_strip(uv, &strip.cells, &candidate.points)
            || candidate
                .points
                .values()
                .any(|point| distance3(point.world, world.to_array()) <= separation)
            || points
                .iter()
                .any(|point| distance3(point.world, world.to_array()) <= separation)
        {
            break;
        }
        points.push(Point {
            uv,
            world: world.to_array(),
            boundary: false,
            protected: false,
        });
        spacing = (spacing * LAYER_TRANSITION_GROWTH).min(context.target_size);
    }
    Some(points)
}

fn point_in_strip(uv: [f64; 2], cells: &[Cell], points: &BTreeMap<PointKey, Point>) -> bool {
    cells
        .iter()
        .any(|cell| point_in_cell(uv, cell, points, 1.0e-12))
}

fn point_in_cell(
    uv: [f64; 2],
    cell: &Cell,
    points: &BTreeMap<PointKey, Point>,
    tolerance: f64,
) -> bool {
    (0..cell.points.len()).all(|edge| {
        cross_2d(
            points[&cell.points[edge]].uv,
            points[&cell.points[(edge + 1) % cell.points.len()]].uv,
            uv,
        ) >= -tolerance
    })
}

fn cells_edges_cross(candidate: &Candidate, cells: &[Cell]) -> bool {
    crossing_cell_edges(candidate, cells).is_some()
}

fn crossing_cell_edges(
    candidate: &Candidate,
    cells: &[Cell],
) -> Option<((PointKey, PointKey), (PointKey, PointKey))> {
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
    let mut edges = edges.into_iter().collect::<Vec<_>>();
    edges.sort_by(|(_, (a, b)), (_, (c, d))| a[0].min(b[0]).total_cmp(&c[0].min(d[0])));
    for first in 0..edges.len() {
        let ((a_key, b_key), (a, b)) = edges[first];
        let first_max_x = a[0].max(b[0]);
        let first_min_y = a[1].min(b[1]);
        let first_max_y = a[1].max(b[1]);
        for ((c_key, d_key), (c, d)) in edges.iter().skip(first + 1).copied() {
            if c[0].min(d[0]) > first_max_x {
                break;
            }
            if c[1].min(d[1]) > first_max_y || c[1].max(d[1]) < first_min_y {
                continue;
            }
            if a_key == c_key || a_key == d_key || b_key == c_key || b_key == d_key {
                continue;
            }
            if segments_cross(a, b, c, d) {
                return Some(((a_key, b_key), (c_key, d_key)));
            }
        }
    }
    None
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
        let skewness = quality_score("tri3", &positions, QualityMetric::Skewness).unwrap_or(1.0);
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
        let shape_distortion = (1.0 - scaled_jacobian).max(skewness);
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

fn cap_quality(candidate: &Candidate) -> Option<CapQuality> {
    let direct = candidate
        .layer_end_targets
        .iter()
        .flat_map(|end| edge_cells(candidate, end.edge.0, end.edge.1))
        .filter(|index| {
            candidate.cells[*index].points.len() == 3 && !candidate.cells[*index].protected
        })
        .collect::<BTreeSet<_>>();
    cap_quality_from_direct(candidate, direct)
}

fn edge_cap_quality(candidate: &Candidate, edge: (PointKey, PointKey)) -> Option<CapQuality> {
    let direct = edge_cells(candidate, edge.0, edge.1)
        .into_iter()
        .filter(|index| {
            candidate.cells[*index].points.len() == 3 && !candidate.cells[*index].protected
        })
        .collect::<BTreeSet<_>>();
    cap_quality_from_direct(candidate, direct)
}

fn cap_quality_from_direct(candidate: &Candidate, direct: BTreeSet<usize>) -> Option<CapQuality> {
    if direct.is_empty() {
        return None;
    }
    let transition_minimum_scaled_jacobian = direct
        .iter()
        .map(|index| triangle_scaled_jacobian(candidate, *index))
        .fold(1.0, f64::min);
    let direct_edges = direct
        .iter()
        .flat_map(|index| {
            let points = &candidate.cells[*index].points;
            (0..3).map(|edge| ordered_pair(points[edge], points[(edge + 1) % 3]))
        })
        .collect::<BTreeSet<_>>();
    let mut cells = direct;
    cells.extend(
        candidate
            .cells
            .iter()
            .enumerate()
            .filter_map(|(index, cell)| {
                (!cell.protected
                    && cell.points.len() == 3
                    && (0..3).any(|edge| {
                        direct_edges.contains(&ordered_pair(
                            cell.points[edge],
                            cell.points[(edge + 1) % 3],
                        ))
                    }))
                .then_some(index)
            }),
    );
    let qualities = cells
        .iter()
        .map(|index| triangle_scaled_jacobian(candidate, *index))
        .collect::<Vec<_>>();
    Some(CapQuality {
        average_scaled_jacobian: qualities.iter().sum::<f64>() / qualities.len() as f64,
        minimum_scaled_jacobian: qualities.iter().copied().fold(1.0, f64::min),
        transition_minimum_scaled_jacobian,
    })
}

fn triangle_scaled_jacobian(candidate: &Candidate, index: usize) -> f64 {
    let positions = candidate.cells[index]
        .points
        .iter()
        .map(|key| candidate.points[key].world)
        .collect::<Vec<_>>();
    quality_score("tri3", &positions, QualityMetric::ScaledJacobian).unwrap_or(0.0)
}

fn cap_seed_quality_is_better(candidate: &CapQuality, current: &CapQuality) -> bool {
    let candidate_meets_target =
        candidate.transition_minimum_scaled_jacobian + 1.0e-12 >= QUALITY_TARGET;
    let current_meets_target =
        current.transition_minimum_scaled_jacobian + 1.0e-12 >= QUALITY_TARGET;
    candidate_meets_target && !current_meets_target
        || candidate_meets_target == current_meets_target
            && (candidate.minimum_scaled_jacobian > current.minimum_scaled_jacobian + 1.0e-12
                || (candidate.minimum_scaled_jacobian - current.minimum_scaled_jacobian).abs()
                    <= 1.0e-12
                    && candidate.average_scaled_jacobian
                        > current.average_scaled_jacobian + 1.0e-12)
}

fn cap_quality_preserved(before: Option<&CapQuality>, after: Option<&CapQuality>) -> bool {
    match (before, after) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some(before), Some(after)) => {
            after.average_scaled_jacobian + 1.0e-12 >= before.average_scaled_jacobian
                && after.minimum_scaled_jacobian + 1.0e-12 >= before.minimum_scaled_jacobian
                && after.transition_minimum_scaled_jacobian + 1.0e-12
                    >= before.transition_minimum_scaled_jacobian
        }
    }
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

#[allow(clippy::too_many_arguments)]
fn apply_split(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    a: PointKey,
    b: PointKey,
    pair: &[usize],
) -> MeshResult<bool> {
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
    if boundary && candidate.points[&a].protected && candidate.points[&b].protected {
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
            None,
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
    for &index in pair {
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
    if !assessment.refine.is_empty() || assessment.score.hard_invalid != 0 {
        return Err(minimum_size_error(
            domain,
            context,
            assessment
                .reason
                .as_deref()
                .unwrap_or("final mesh failed the topology and coverage audit"),
            assessment.location,
            assessment.worst_quality,
        ));
    }
    if let Some((first, second)) = crossing_cell_edges(candidate, &candidate.cells) {
        let reason = format!(
            "cell edges cross instead of forming a planar triangulation: {:?} -> {:?} crosses {:?} -> {:?}",
            candidate.points[&first.0].world,
            candidate.points[&first.1].world,
            candidate.points[&second.0].world,
            candidate.points[&second.1].world,
        );
        return Err(minimum_size_error(
            domain,
            context,
            &reason,
            candidate.cells.first().map(|_| cell_centroid(candidate, 0)),
            assessment.worst_quality,
        ));
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
        let spade_tile = spade_point_tile(candidate, &candidate.cells[start..end])?;
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

fn constrained_spade_tile(candidate: &Candidate, cells: &[Cell]) -> MeshResult<PointSpade> {
    let (mut tile, vertices) = spade_tile_vertices(candidate, cells)?;
    for cell in cells {
        for edge in 0..cell.points.len() {
            let a = cell.points[edge];
            let b = cell.points[(edge + 1) % cell.points.len()];
            if tile
                .try_add_constraint(vertices[&a], vertices[&b])
                .is_empty()
            {
                return Err(MeshError::InvalidInput(format!(
                    "DistMesh cells contain a crossing or overlapping edge {:?} -> {:?}",
                    candidate.points[&a].world, candidate.points[&b].world,
                )));
            }
        }
    }
    Ok(tile)
}

fn spade_point_tile(candidate: &Candidate, cells: &[Cell]) -> MeshResult<PointSpade> {
    spade_tile_vertices(candidate, cells).map(|(tile, _)| tile)
}

fn spade_tile_vertices(
    candidate: &Candidate,
    cells: &[Cell],
) -> MeshResult<(PointSpade, PointSpadeVertices)> {
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
    Ok((tile, vertices))
}

fn estimated_spade_bytes(tile: &PointSpade) -> usize {
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
    let extent = [a, b, c, d].into_iter().fold(
        [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ],
        |bounds, point| {
            [
                bounds[0].min(point[0]),
                bounds[1].max(point[0]),
                bounds[2].min(point[1]),
                bounds[3].max(point[1]),
            ]
        },
    );
    let scale = (extent[1] - extent[0]).max(extent[3] - extent[2]);
    let tolerance = f64::EPSILON * scale.powi(2) * 64.0;
    ab_c.abs() > tolerance
        && ab_d.abs() > tolerance
        && cd_a.abs() > tolerance
        && cd_b.abs() > tolerance
        && ab_c * ab_d < 0.0
        && cd_a * cd_b < 0.0
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
    root_tolerance(domain, local_size).max((local_size * 0.125).min(domain.boundary_tolerance()))
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
        "domain {:?} could not produce valid 2D topology at target size {:.6e} near ({:.6}, {:.6}, {:.6}): {reason}; worst Scaled Jacobian={quality:.6e}, validity requires > {VALID_QUALITY:.1e}; quality optimization is best-effort",
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
