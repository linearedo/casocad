use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::sync::Arc;

use arrow_array::{Array, LargeListArray, UInt64Array};

use crate::error::{MeshError, MeshResult};
use crate::quality::{quality_score, QualityMetric};
use crate::schema::{element_dimension, Bounds3, RowKind};
use crate::{BatchView, MeshFile};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub min: f64,
    pub max: f64,
}

impl Interval {
    pub const ALL: Self = Self {
        min: f64::NEG_INFINITY,
        max: f64::INFINITY,
    };

    pub const fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }

    pub fn contains(self, value: f64) -> bool {
        value >= self.min && value <= self.max
    }

    fn validate(self, name: &str) -> MeshResult<()> {
        if self.min.is_nan() || self.max.is_nan() || self.min > self.max {
            return Err(MeshError::InvalidInput(format!(
                "{name} interval must be ordered and non-NaN"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EntityKind {
    Point,
    Edge,
    Face,
    Cell,
}

impl EntityKind {
    pub const fn row_kind(self) -> RowKind {
        match self {
            Self::Point => RowKind::Point,
            Self::Edge => RowKind::Edge,
            Self::Face => RowKind::Face,
            Self::Cell => RowKind::Cell,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityFilter {
    pub metric: QualityMetric,
    pub interval: Interval,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshQuery {
    pub x: Interval,
    pub y: Interval,
    pub z: Interval,
    pub entity_kind: EntityKind,
    pub element_type: Option<String>,
    pub zone_ids: BTreeSet<u64>,
    pub tag_ids: BTreeSet<u64>,
    pub boundary_distance: Option<Interval>,
    pub quality: Option<QualityFilter>,
    pub formula: Option<TypedFormula>,
    pub display_limit: usize,
}

impl Default for MeshQuery {
    fn default() -> Self {
        Self {
            x: Interval::ALL,
            y: Interval::ALL,
            z: Interval::ALL,
            entity_kind: EntityKind::Cell,
            element_type: None,
            zone_ids: BTreeSet::new(),
            tag_ids: BTreeSet::new(),
            boundary_distance: None,
            quality: None,
            formula: None,
            display_limit: 1_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectedEntity {
    pub id: u64,
    pub kind: EntityKind,
    pub tile_id: u64,
    pub element_type: String,
    pub point_ids: Vec<u64>,
    pub points: Vec<[f64; 3]>,
    pub edge_ids: Vec<u64>,
    pub face_ids: Vec<u64>,
    pub tag_ids: Vec<u64>,
    pub zone_id: Option<u64>,
    pub source_id: Option<u64>,
    pub source_object_id: Option<u64>,
    pub boundary: bool,
    pub boundary_distance: Option<f64>,
    pub quality: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TileRenderData {
    pub tile_id: u64,
    pub entities: Vec<SelectedEntity>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshQueryResult {
    pub total_matching_count: u64,
    pub displayed_count: usize,
    pub selected_entity_ids: Vec<u64>,
    pub render_tiles: Vec<TileRenderData>,
}

#[derive(Debug)]
pub struct MeshQueryService {
    file: Arc<MeshFile>,
}

impl MeshQueryService {
    pub fn new(file: Arc<MeshFile>) -> Self {
        Self { file }
    }

    pub fn mesh_file(&self) -> &Arc<MeshFile> {
        &self.file
    }

    pub fn execute(&self, query: MeshQuery) -> MeshResult<MeshQueryResult> {
        self.execute_nodes(query, None)
    }

    pub(crate) fn execute_selected_nodes(
        &self,
        query: MeshQuery,
        nodes: &BTreeSet<u64>,
    ) -> MeshResult<MeshQueryResult> {
        self.execute_nodes(query, Some(nodes))
    }

    fn execute_nodes(
        &self,
        query: MeshQuery,
        selected_nodes: Option<&BTreeSet<u64>>,
    ) -> MeshResult<MeshQueryResult> {
        validate_query(&query)?;
        let query_bounds = Bounds3 {
            min: [query.x.min, query.y.min, query.z.min],
            max: [query.x.max, query.y.max, query.z.max],
        };
        let kind = query.entity_kind.row_kind();
        let mut candidate_tiles = self.file.candidate_leaf_tiles(query_bounds);
        if let Some(selected) = selected_nodes {
            candidate_tiles.retain(|tile| selected.contains(tile));
        }
        let mut total = 0u64;
        let mut heap = BinaryHeap::<(u64, u64)>::new();
        let mut selected = BTreeMap::<u64, SelectedEntity>::new();

        for entry in self.file.entity_batches(kind) {
            if entry
                .spatial_node_id
                .is_none_or(|tile_id| !candidate_tiles.contains(&tile_id))
                || entry
                    .bounds
                    .is_some_and(|bounds| !bounds.intersects(query_bounds))
                || query
                    .element_type
                    .as_ref()
                    .is_some_and(|value| !entry.element_types.iter().any(|found| found == value))
                || (!query.zone_ids.is_empty()
                    && entry.zone_ids.iter().all(|id| !query.zone_ids.contains(id)))
                || (!query.tag_ids.is_empty()
                    && entry.tag_ids.iter().all(|id| !query.tag_ids.contains(id)))
            {
                continue;
            }
            let tile_id = entry.spatial_node_id.ok_or_else(|| {
                MeshError::InvalidFile("entity batch has no owning tile in its directory".into())
            })?;
            let points = load_points(&self.file, tile_id)?;
            let boundary_segments = if query.boundary_distance.is_some() {
                load_boundary_segments(&self.file, tile_id, &points)?
            } else {
                Vec::new()
            };
            let batch = self.file.batch_view(entry.batch_index)?;
            for row in 0..batch.len() {
                let Some(entity) = entity_from_row(
                    &self.file,
                    &batch,
                    row,
                    query.entity_kind,
                    tile_id,
                    &points,
                    &boundary_segments,
                    query.quality.map(|quality| quality.metric),
                )?
                else {
                    continue;
                };
                if !matches_query(&query, &entity, &self.file)? {
                    continue;
                }
                total += 1;
                if query.display_limit == 0 {
                    continue;
                }
                let hash = stable_entity_hash(query.entity_kind, entity.id);
                if heap.len() < query.display_limit {
                    heap.push((hash, entity.id));
                    selected.insert(entity.id, entity);
                } else if heap.peek().is_some_and(|&(largest_hash, largest_id)| {
                    (hash, entity.id) < (largest_hash, largest_id)
                }) {
                    let (_, removed) = heap.pop().expect("heap is non-empty");
                    selected.remove(&removed);
                    heap.push((hash, entity.id));
                    selected.insert(entity.id, entity);
                }
            }
        }

        let mut ranked: Vec<_> = heap.into_vec();
        ranked.sort_unstable();
        let selected_entity_ids: Vec<u64> = ranked.iter().map(|(_, id)| *id).collect();
        let mut tiles = BTreeMap::<u64, Vec<SelectedEntity>>::new();
        for (_, id) in ranked {
            if let Some(entity) = selected.remove(&id) {
                tiles.entry(entity.tile_id).or_default().push(entity);
            }
        }
        let render_tiles = tiles
            .into_iter()
            .map(|(tile_id, entities)| TileRenderData { tile_id, entities })
            .collect::<Vec<_>>();
        Ok(MeshQueryResult {
            total_matching_count: total,
            displayed_count: selected_entity_ids.len(),
            selected_entity_ids,
            render_tiles,
        })
    }
}

fn validate_query(query: &MeshQuery) -> MeshResult<()> {
    query.x.validate("x")?;
    query.y.validate("y")?;
    query.z.validate("z")?;
    if let Some(interval) = query.boundary_distance {
        interval.validate("boundary distance")?;
        if interval.min < 0.0 {
            return Err(MeshError::InvalidInput(
                "boundary distance cannot be negative".into(),
            ));
        }
    }
    if let Some(quality) = query.quality {
        quality.interval.validate("quality")?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn entity_from_row(
    file: &MeshFile,
    batch: &BatchView,
    row: usize,
    kind: EntityKind,
    tile_id: u64,
    points: &BTreeMap<u64, [f64; 3]>,
    boundary_segments: &[([f64; 3], [f64; 3])],
    quality_metric: Option<QualityMetric>,
) -> MeshResult<Option<SelectedEntity>> {
    let ids = batch.u64s("entity_id")?;
    if ids.is_null(row) {
        return Ok(None);
    }
    if kind == EntityKind::Point {
        let ghosts = batch.bools("ghost")?;
        if ghosts.is_null(row) || ghosts.value(row) {
            return Ok(None);
        }
    }
    let element_types = batch.strings("element_type")?;
    let element_type = if kind == EntityKind::Point {
        "point1".to_string()
    } else if element_types.is_null(row) {
        return Err(MeshError::InvalidFile(
            "entity row has no element type".into(),
        ));
    } else {
        element_types.value(row).to_string()
    };
    let point_ids = if kind == EntityKind::Point {
        vec![ids.value(row)]
    } else {
        list_values(batch.lists("point_ids")?, row)?
    };
    let geometry: Option<Vec<_>> = point_ids.iter().map(|id| points.get(id).copied()).collect();
    let Some(geometry) = geometry else {
        return Err(MeshError::InvalidFile(format!(
            "{kind:?} {} references a point absent from tile {tile_id}",
            ids.value(row)
        )));
    };
    let tag_ids = list_values(batch.lists("tag_ids")?, row)?;
    let edge_ids = list_values(batch.lists("edge_ids")?, row)?;
    let face_ids = list_values(batch.lists("face_ids")?, row)?;
    let zones = batch.u64s("zone_id")?;
    let sources = batch.u64s("source_id")?;
    let source_id = (!sources.is_null(row)).then(|| sources.value(row));
    let boundary_flags = batch.bools("boundary")?;
    let boundary = !boundary_flags.is_null(row) && boundary_flags.value(row);
    let boundary_distance =
        (!boundary_segments.is_empty()).then(|| distance_to_boundary(&geometry, boundary_segments));
    let quality = quality_metric.and_then(|metric| quality_score(&element_type, &geometry, metric));
    Ok(Some(SelectedEntity {
        id: ids.value(row),
        kind,
        tile_id,
        element_type,
        point_ids,
        points: geometry,
        edge_ids,
        face_ids,
        tag_ids,
        zone_id: (!zones.is_null(row)).then(|| zones.value(row)),
        source_id,
        source_object_id: source_id.and_then(|id| file.catalog_source_object("source", id)),
        boundary,
        boundary_distance,
        quality,
    }))
}

fn matches_query(query: &MeshQuery, entity: &SelectedEntity, file: &MeshFile) -> MeshResult<bool> {
    if query
        .element_type
        .as_ref()
        .is_some_and(|element_type| entity.element_type != *element_type)
        || !entity.points.iter().all(|point| {
            query.x.contains(point[0]) && query.y.contains(point[1]) && query.z.contains(point[2])
        })
        || (!query.zone_ids.is_empty()
            && entity
                .zone_id
                .is_none_or(|zone| !query.zone_ids.contains(&zone)))
        || !query.tag_ids.iter().all(|tag| entity.tag_ids.contains(tag))
        || query.boundary_distance.is_some_and(|interval| {
            entity
                .boundary_distance
                .is_none_or(|d| !interval.contains(d))
        })
        || query
            .quality
            .is_some_and(|filter| entity.quality.is_none_or(|q| !filter.interval.contains(q)))
    {
        return Ok(false);
    }
    if let Some(formula) = &query.formula {
        let centroid = centroid(&entity.points);
        let context = FormulaContext {
            entity,
            centroid,
            dimension: element_dimension(&entity.element_type).unwrap_or(0),
            file,
        };
        if !formula.evaluate(&context)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn load_points(file: &MeshFile, tile_id: u64) -> MeshResult<BTreeMap<u64, [f64; 3]>> {
    let mut points = BTreeMap::new();
    for entry in file.tile_batches(tile_id, RowKind::Point) {
        let batch = file.batch_view(entry.batch_index)?;
        let ids = batch.u64s("entity_id")?;
        let x = batch.f64s("x")?;
        let y = batch.f64s("y")?;
        let z = batch.f64s("z")?;
        for row in 0..batch.len() {
            points.insert(ids.value(row), [x.value(row), y.value(row), z.value(row)]);
        }
    }
    Ok(points)
}

fn load_boundary_segments(
    file: &MeshFile,
    tile_id: u64,
    points: &BTreeMap<u64, [f64; 3]>,
) -> MeshResult<Vec<([f64; 3], [f64; 3])>> {
    let mut segments = Vec::new();
    for entry in file.tile_batches(tile_id, RowKind::Edge) {
        let batch = file.batch_view(entry.batch_index)?;
        let boundary = batch.bools("boundary")?;
        let connectivity = batch.lists("point_ids")?;
        for row in 0..batch.len() {
            if boundary.is_null(row) || !boundary.value(row) {
                continue;
            }
            let ids = list_values(connectivity, row)?;
            if ids.len() == 2 {
                if let (Some(a), Some(b)) = (points.get(&ids[0]), points.get(&ids[1])) {
                    segments.push((*a, *b));
                }
            }
        }
    }
    Ok(segments)
}

fn list_values(array: &LargeListArray, row: usize) -> MeshResult<Vec<u64>> {
    let values = array.value(row);
    values
        .as_any()
        .downcast_ref::<UInt64Array>()
        .map(|values| values.values().to_vec())
        .ok_or_else(|| MeshError::InvalidFile("list column must contain u64 values".into()))
}

fn centroid(points: &[[f64; 3]]) -> [f64; 3] {
    let mut result = [0.0; 3];
    for point in points {
        for axis in 0..3 {
            result[axis] += point[axis] / points.len() as f64;
        }
    }
    result
}

fn distance_to_boundary(points: &[[f64; 3]], segments: &[([f64; 3], [f64; 3])]) -> f64 {
    points
        .iter()
        .flat_map(|point| {
            segments
                .iter()
                .map(move |(a, b)| point_segment_distance(*point, *a, *b))
        })
        .reduce(f64::min)
        .unwrap_or(f64::INFINITY)
}

fn point_segment_distance(point: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = std::array::from_fn::<_, 3, _>(|axis| b[axis] - a[axis]);
    let ap = std::array::from_fn::<_, 3, _>(|axis| point[axis] - a[axis]);
    let denominator: f64 = ab.iter().map(|value| value * value).sum();
    let t = if denominator <= f64::EPSILON {
        0.0
    } else {
        (0..3).map(|axis| ap[axis] * ab[axis]).sum::<f64>() / denominator
    }
    .clamp(0.0, 1.0);
    (0..3)
        .map(|axis| (point[axis] - (a[axis] + t * ab[axis])).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn stable_entity_hash(kind: EntityKind, id: u64) -> u64 {
    let kind = match kind {
        EntityKind::Point => 1u64,
        EntityKind::Edge => 2,
        EntityKind::Face => 3,
        EntityKind::Cell => 4,
    };
    let mut value = id ^ kind.wrapping_mul(0x9e3779b97f4a7c15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedFormula {
    expression: BoolExpr,
}

impl TypedFormula {
    pub fn parse(source: &str) -> MeshResult<Self> {
        let mut parser = Parser::new(source)?;
        let expression = parser.parse_or()?;
        if parser.peek() != &Token::End {
            return Err(MeshError::InvalidInput(
                "unexpected token after formula expression".into(),
            ));
        }
        Ok(Self { expression })
    }

    fn evaluate(&self, context: &FormulaContext<'_>) -> MeshResult<bool> {
        self.expression.evaluate(context)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum BoolExpr {
    Literal(bool),
    Not(Box<Self>),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Compare(ValueExpr, CompareOp, ValueExpr),
    HasTag(TagArgument),
    IsSupported(Option<QualityMetric>),
}

impl BoolExpr {
    fn evaluate(&self, context: &FormulaContext<'_>) -> MeshResult<bool> {
        Ok(match self {
            Self::Literal(value) => *value,
            Self::Not(value) => !value.evaluate(context)?,
            Self::And(a, b) => a.evaluate(context)? && b.evaluate(context)?,
            Self::Or(a, b) => a.evaluate(context)? || b.evaluate(context)?,
            Self::Compare(a, operator, b) => {
                operator.evaluate(a.evaluate(context)?, b.evaluate(context)?)?
            }
            Self::HasTag(TagArgument::Id(id)) => context.entity.tag_ids.contains(id),
            Self::HasTag(TagArgument::Name(name)) => context
                .entity
                .tag_ids
                .iter()
                .any(|id| context.file.catalog_name("tag", *id) == Some(name.as_str())),
            Self::IsSupported(metric) => {
                let metric = metric.or(context
                    .entity
                    .quality
                    .map(|_| QualityMetric::ScaledJacobian));
                metric.is_some_and(|metric| {
                    quality_score(&context.entity.element_type, &context.entity.points, metric)
                        .is_some()
                })
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ValueExpr {
    Number(f64),
    Unsigned(u64),
    String(String),
    Field(FieldName),
}

impl ValueExpr {
    fn evaluate(&self, context: &FormulaContext<'_>) -> MeshResult<Value> {
        Ok(match self {
            Self::Number(value) => Value::Number(*value),
            Self::Unsigned(value) => Value::Unsigned(*value),
            Self::String(value) => Value::String(value.clone()),
            Self::Field(field) => match field {
                FieldName::Id => Value::Unsigned(context.entity.id),
                FieldName::X => Value::Number(context.centroid[0]),
                FieldName::Y => Value::Number(context.centroid[1]),
                FieldName::Z => Value::Number(context.centroid[2]),
                FieldName::TileId => Value::Unsigned(context.entity.tile_id),
                FieldName::ZoneId => context
                    .entity
                    .zone_id
                    .map(Value::Unsigned)
                    .unwrap_or(Value::Null),
                FieldName::Dimension => Value::Unsigned(u64::from(context.dimension)),
                FieldName::Quality => context
                    .entity
                    .quality
                    .map(Value::Number)
                    .unwrap_or(Value::Null),
                FieldName::BoundaryDistance => context
                    .entity
                    .boundary_distance
                    .map(Value::Number)
                    .unwrap_or(Value::Null),
                FieldName::ElementType => Value::String(context.entity.element_type.clone()),
            },
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldName {
    Id,
    X,
    Y,
    Z,
    TileId,
    ZoneId,
    Dimension,
    Quality,
    BoundaryDistance,
    ElementType,
}

#[derive(Debug, Clone)]
enum Value {
    Number(f64),
    Unsigned(u64),
    String(String),
    Null,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CompareOp {
    fn evaluate(self, left: Value, right: Value) -> MeshResult<bool> {
        match (left, right) {
            (Value::Null, Value::Null) => Ok(matches!(self, Self::Eq | Self::Le | Self::Ge)),
            (Value::Null, _) | (_, Value::Null) => Ok(matches!(self, Self::Ne)),
            (Value::Number(a), Value::Number(b)) => Ok(match self {
                Self::Eq => a == b,
                Self::Ne => a != b,
                Self::Lt => a < b,
                Self::Le => a <= b,
                Self::Gt => a > b,
                Self::Ge => a >= b,
            }),
            (Value::Unsigned(a), Value::Unsigned(b)) => Ok(match self {
                Self::Eq => a == b,
                Self::Ne => a != b,
                Self::Lt => a < b,
                Self::Le => a <= b,
                Self::Gt => a > b,
                Self::Ge => a >= b,
            }),
            (Value::Unsigned(a), Value::Number(b)) => {
                self.evaluate(Value::Number(a as f64), Value::Number(b))
            }
            (Value::Number(a), Value::Unsigned(b)) => {
                self.evaluate(Value::Number(a), Value::Number(b as f64))
            }
            (Value::String(a), Value::String(b)) => Ok(match self {
                Self::Eq => a == b,
                Self::Ne => a != b,
                Self::Lt => a < b,
                Self::Le => a <= b,
                Self::Gt => a > b,
                Self::Ge => a >= b,
            }),
            _ => Err(MeshError::InvalidInput(
                "formula comparison operands have different types".into(),
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TagArgument {
    Id(u64),
    Name(String),
}

struct FormulaContext<'a> {
    entity: &'a SelectedEntity,
    centroid: [f64; 3],
    dimension: u8,
    file: &'a MeshFile,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Identifier(String),
    Number(NumericLiteral),
    String(String),
    True,
    False,
    LeftParen,
    RightParen,
    Comma,
    Not,
    And,
    Or,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    End,
}

#[derive(Debug, Clone, PartialEq)]
struct NumericLiteral {
    value: f64,
    unsigned: Option<u64>,
}

struct Parser {
    tokens: Vec<Token>,
    index: usize,
}

impl Parser {
    fn new(source: &str) -> MeshResult<Self> {
        Ok(Self {
            tokens: tokenize(source)?,
            index: 0,
        })
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.index]
    }

    fn take(&mut self) -> Token {
        let token = self.tokens[self.index].clone();
        self.index += 1;
        token
    }

    fn parse_or(&mut self) -> MeshResult<BoolExpr> {
        let mut expression = self.parse_and()?;
        while self.peek() == &Token::Or {
            self.take();
            expression = BoolExpr::Or(Box::new(expression), Box::new(self.parse_and()?));
        }
        Ok(expression)
    }

    fn parse_and(&mut self) -> MeshResult<BoolExpr> {
        let mut expression = self.parse_not()?;
        while self.peek() == &Token::And {
            self.take();
            expression = BoolExpr::And(Box::new(expression), Box::new(self.parse_not()?));
        }
        Ok(expression)
    }

    fn parse_not(&mut self) -> MeshResult<BoolExpr> {
        if self.peek() == &Token::Not {
            self.take();
            return Ok(BoolExpr::Not(Box::new(self.parse_not()?)));
        }
        self.parse_primary_bool()
    }

    fn parse_primary_bool(&mut self) -> MeshResult<BoolExpr> {
        if self.peek() == &Token::LeftParen {
            self.take();
            let expression = self.parse_or()?;
            self.expect(Token::RightParen)?;
            return Ok(expression);
        }
        if self.peek() == &Token::True {
            self.take();
            return Ok(BoolExpr::Literal(true));
        }
        if self.peek() == &Token::False {
            self.take();
            return Ok(BoolExpr::Literal(false));
        }
        if let Token::Identifier(name) = self.peek().clone() {
            if name == "has_tag" {
                self.take();
                self.expect(Token::LeftParen)?;
                let argument = match self.take() {
                    Token::Number(NumericLiteral {
                        unsigned: Some(value),
                        ..
                    }) => TagArgument::Id(value),
                    Token::String(value) => TagArgument::Name(value),
                    _ => {
                        return Err(MeshError::InvalidInput(
                            "has_tag() expects an integer ID or string name".into(),
                        ))
                    }
                };
                self.expect(Token::RightParen)?;
                return Ok(BoolExpr::HasTag(argument));
            }
            if name == "is_supported" {
                self.take();
                self.expect(Token::LeftParen)?;
                let metric = if self.peek() == &Token::RightParen {
                    None
                } else {
                    let Token::String(value) = self.take() else {
                        return Err(MeshError::InvalidInput(
                            "is_supported() expects an optional metric string".into(),
                        ));
                    };
                    Some(QualityMetric::parse(&value).ok_or_else(|| {
                        MeshError::InvalidInput(format!("unknown quality metric {value:?}"))
                    })?)
                };
                self.expect(Token::RightParen)?;
                return Ok(BoolExpr::IsSupported(metric));
            }
        }
        let left = self.parse_value()?;
        let operator = match self.take() {
            Token::Eq => CompareOp::Eq,
            Token::Ne => CompareOp::Ne,
            Token::Lt => CompareOp::Lt,
            Token::Le => CompareOp::Le,
            Token::Gt => CompareOp::Gt,
            Token::Ge => CompareOp::Ge,
            _ => {
                return Err(MeshError::InvalidInput(
                    "formula value must be followed by a comparison operator".into(),
                ))
            }
        };
        let right = self.parse_value()?;
        Ok(BoolExpr::Compare(left, operator, right))
    }

    fn parse_value(&mut self) -> MeshResult<ValueExpr> {
        match self.take() {
            Token::Number(value) => Ok(value
                .unsigned
                .map(ValueExpr::Unsigned)
                .unwrap_or(ValueExpr::Number(value.value))),
            Token::String(value) => Ok(ValueExpr::String(value)),
            Token::Identifier(name) => Ok(ValueExpr::Field(match name.as_str() {
                "id" => FieldName::Id,
                "x" => FieldName::X,
                "y" => FieldName::Y,
                "z" => FieldName::Z,
                "tile_id" => FieldName::TileId,
                "zone_id" => FieldName::ZoneId,
                "dimension" => FieldName::Dimension,
                "quality" => FieldName::Quality,
                "boundary_distance" => FieldName::BoundaryDistance,
                "element_type" => FieldName::ElementType,
                _ => {
                    return Err(MeshError::InvalidInput(format!(
                        "unknown formula field {name:?}"
                    )))
                }
            })),
            _ => Err(MeshError::InvalidInput(
                "expected a formula field or literal".into(),
            )),
        }
    }

    fn expect(&mut self, token: Token) -> MeshResult<()> {
        if self.peek() == &token {
            self.take();
            Ok(())
        } else {
            Err(MeshError::InvalidInput(format!(
                "expected {token:?}, found {:?}",
                self.peek()
            )))
        }
    }
}

fn tokenize(source: &str) -> MeshResult<Vec<Token>> {
    let chars: Vec<char> = source.chars().collect();
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        let character = chars[index];
        if character.is_whitespace() {
            index += 1;
            continue;
        }
        let pair = chars.get(index + 1).map(|next| [character, *next]);
        if let Some(token) = match pair {
            Some(['&', '&']) => Some(Token::And),
            Some(['|', '|']) => Some(Token::Or),
            Some(['=', '=']) => Some(Token::Eq),
            Some(['!', '=']) => Some(Token::Ne),
            Some(['<', '=']) => Some(Token::Le),
            Some(['>', '=']) => Some(Token::Ge),
            _ => None,
        } {
            tokens.push(token);
            index += 2;
            continue;
        }
        match character {
            '(' => tokens.push(Token::LeftParen),
            ')' => tokens.push(Token::RightParen),
            ',' => tokens.push(Token::Comma),
            '!' => tokens.push(Token::Not),
            '<' => tokens.push(Token::Lt),
            '>' => tokens.push(Token::Gt),
            '"' | '\'' => {
                let quote = character;
                index += 1;
                let start = index;
                while index < chars.len() && chars[index] != quote {
                    index += 1;
                }
                if index == chars.len() {
                    return Err(MeshError::InvalidInput(
                        "unterminated formula string".into(),
                    ));
                }
                tokens.push(Token::String(chars[start..index].iter().collect()));
            }
            c if c.is_ascii_digit()
                || (c == '-'
                    && chars
                        .get(index + 1)
                        .is_some_and(|next| next.is_ascii_digit())) =>
            {
                let start = index;
                index += 1;
                while chars.get(index).is_some_and(|value| {
                    value.is_ascii_digit() || matches!(value, '.' | 'e' | 'E' | '+' | '-')
                }) {
                    index += 1;
                }
                let text: String = chars[start..index].iter().collect();
                let number: f64 = text.parse().map_err(|_| {
                    MeshError::InvalidInput(format!("invalid formula number {text:?}"))
                })?;
                if !number.is_finite() {
                    return Err(MeshError::InvalidInput(format!(
                        "formula number {text:?} is not finite"
                    )));
                }
                let unsigned = (!text.starts_with('-') && !text.contains(['.', 'e', 'E']))
                    .then(|| text.parse::<u64>().ok())
                    .flatten();
                tokens.push(Token::Number(NumericLiteral {
                    value: number,
                    unsigned,
                }));
                continue;
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = index;
                index += 1;
                while chars
                    .get(index)
                    .is_some_and(|value| value.is_ascii_alphanumeric() || *value == '_')
                {
                    index += 1;
                }
                let identifier: String = chars[start..index].iter().collect();
                tokens.push(match identifier.as_str() {
                    "true" => Token::True,
                    "false" => Token::False,
                    "and" => Token::And,
                    "or" => Token::Or,
                    "not" => Token::Not,
                    _ => Token::Identifier(identifier),
                });
                continue;
            }
            _ => {
                return Err(MeshError::InvalidInput(format!(
                    "unexpected formula character {character:?}"
                )))
            }
        }
        index += 1;
    }
    tokens.push(Token::End);
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formula_keeps_large_unsigned_literals_exact() {
        let formula = TypedFormula::parse("id == 9007199254740993").expect("formula");
        assert!(matches!(
            formula.expression,
            BoolExpr::Compare(
                ValueExpr::Field(FieldName::Id),
                CompareOp::Eq,
                ValueExpr::Unsigned(9_007_199_254_740_993)
            )
        ));
        assert!(TypedFormula::parse("has_tag(18446744073709551615)").is_ok());
        assert!(TypedFormula::parse("quality >= 0").is_ok());
    }
}
