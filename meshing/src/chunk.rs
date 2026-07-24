use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{MeshError, MeshResult};
use crate::schema::{element_dimension, expected_points, Bounds3, MAX_BATCH_BYTES, MAX_BATCH_ROWS};

const POINT_ORDINAL: u32 = 0x0000_0000;
const EDGE_ORDINAL: u32 = 0x4000_0000;
const FACE_ORDINAL: u32 = 0x8000_0000;
const CELL_ORDINAL: u32 = 0xc000_0000;
const ORDINAL_MASK: u32 = 0x3fff_ffff;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
pub struct MeshId(u64);

impl MeshId {
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u64 {
        self.0
    }

    pub const fn owner_chunk(self) -> u32 {
        (self.0 >> 32) as u32
    }

    fn new(chunk: u32, namespace: u32, ordinal: u32) -> MeshResult<Self> {
        if chunk == 0 || ordinal == 0 || ordinal > ORDINAL_MASK {
            return Err(MeshError::LimitExceeded(
                "mesh chunk/entity ID space exhausted".into(),
            ));
        }
        Ok(Self(
            (u64::from(chunk) << 32) | u64::from(namespace | ordinal),
        ))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChunkPoint {
    pub id: MeshId,
    pub owner_chunk_id: u32,
    pub position: [f64; 3],
    pub classification: String,
    pub tag_ids: Vec<u64>,
}

impl ChunkPoint {
    pub fn is_ghost_in(&self, chunk_id: u32) -> bool {
        self.owner_chunk_id != chunk_id
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChunkElement {
    pub id: MeshId,
    pub element_type: String,
    pub point_ids: Vec<MeshId>,
    pub edge_ids: Vec<MeshId>,
    pub face_ids: Vec<MeshId>,
    pub tag_ids: Vec<u64>,
    pub owner_cell_id: Option<MeshId>,
    pub neighbor_cell_id: Option<MeshId>,
    pub zone_id: Option<u64>,
    pub source_id: Option<u64>,
    pub boundary: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshChunk {
    pub id: u32,
    pub bounds: Bounds3,
    pub points: Vec<ChunkPoint>,
    pub edges: Vec<ChunkElement>,
    pub faces: Vec<ChunkElement>,
    pub cells: Vec<ChunkElement>,
}

impl MeshChunk {
    pub fn top_dimensional_elements(&self, dimension: u8) -> usize {
        match dimension {
            2 => self
                .cells
                .iter()
                .filter(|cell| element_dimension(&cell.element_type) == Some(2))
                .count(),
            3 => self
                .cells
                .iter()
                .filter(|cell| element_dimension(&cell.element_type) == Some(3))
                .count(),
            _ => 0,
        }
    }

    pub fn decoded_bytes(&self) -> usize {
        self.points.len() * std::mem::size_of::<ChunkPoint>()
            + self
                .points
                .iter()
                .map(|point| point.classification.len() + point.tag_ids.len() * 8)
                .sum::<usize>()
            + [&self.edges, &self.faces, &self.cells]
                .into_iter()
                .flatten()
                .map(|element| {
                    std::mem::size_of::<ChunkElement>()
                        + element.element_type.len()
                        + (element.point_ids.len()
                            + element.edge_ids.len()
                            + element.face_ids.len())
                            * 8
                        + element.tag_ids.len() * 8
                })
                .sum::<usize>()
    }

    pub fn validate(&self, dimension: u8) -> MeshResult<()> {
        if self.id == 0 || !self.bounds.is_valid() {
            return Err(MeshError::InvalidInput(
                "mesh chunk requires a nonzero ID and finite bounds".into(),
            ));
        }
        if self.top_dimensional_elements(dimension) > MAX_BATCH_ROWS {
            return Err(MeshError::LimitExceeded(format!(
                "chunk {} has more than {MAX_BATCH_ROWS} top-dimensional elements",
                self.id
            )));
        }
        if self.decoded_bytes() > MAX_BATCH_BYTES {
            return Err(MeshError::LimitExceeded(format!(
                "chunk {} exceeds the {} MiB decoded hard limit",
                self.id,
                MAX_BATCH_BYTES / 1024 / 1024
            )));
        }
        let mut ids = BTreeSet::new();
        let mut points = BTreeMap::new();
        for point in &self.points {
            if !ids.insert(point.id) || points.insert(point.id, point.position).is_some() {
                return Err(MeshError::InvalidInput(format!(
                    "chunk {} contains duplicate point {}",
                    self.id,
                    point.id.raw()
                )));
            }
            if point.position.into_iter().any(|value| !value.is_finite())
                || point.owner_chunk_id != point.id.owner_chunk()
                || !self.bounds.contains(point.position)
            {
                return Err(MeshError::InvalidInput(format!(
                    "chunk {} contains an invalid point {}",
                    self.id,
                    point.id.raw()
                )));
            }
        }
        for element in self.edges.iter().chain(&self.faces).chain(&self.cells) {
            if !ids.insert(element.id) {
                return Err(MeshError::InvalidInput(format!(
                    "duplicate entity ID {} in chunk {}",
                    element.id.raw(),
                    self.id
                )));
            }
            let Some(expected) = expected_points(&element.element_type) else {
                return Err(MeshError::InvalidInput(format!(
                    "unknown element type {:?}",
                    element.element_type
                )));
            };
            if !expected.contains(&element.point_ids.len())
                || element.point_ids.iter().any(|id| !points.contains_key(id))
            {
                return Err(MeshError::InvalidInput(format!(
                    "{} {} has invalid or non-local point connectivity",
                    element.element_type,
                    element.id.raw()
                )));
            }
            if element.element_type == "polyhedron" && element.face_ids.is_empty() {
                return Err(MeshError::InvalidInput(
                    "polyhedra require explicit face connectivity".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct MeshChunkBuilder {
    id: u32,
    bounds: Bounds3,
    points: BTreeMap<MeshId, ChunkPoint>,
    edges: Vec<ChunkElement>,
    faces: Vec<ChunkElement>,
    cells: Vec<ChunkElement>,
    next_point: u32,
    next_edge: u32,
    next_face: u32,
    next_cell: u32,
}

impl MeshChunkBuilder {
    pub fn new(id: u32, bounds: Bounds3) -> MeshResult<Self> {
        if id == 0 || !bounds.is_valid() {
            return Err(MeshError::InvalidInput(
                "mesh chunk builder requires a nonzero ID and finite bounds".into(),
            ));
        }
        Ok(Self {
            id,
            bounds,
            points: BTreeMap::new(),
            edges: Vec::new(),
            faces: Vec::new(),
            cells: Vec::new(),
            next_point: 1,
            next_edge: 1,
            next_face: 1,
            next_cell: 1,
        })
    }

    pub const fn id(&self) -> u32 {
        self.id
    }

    pub fn point(&mut self, position: [f64; 3]) -> MeshResult<MeshId> {
        let id = MeshId::new(self.id, POINT_ORDINAL, self.next_point)?;
        self.next_point += 1;
        self.point_copy(id, position, "interior", Vec::new())?;
        Ok(id)
    }

    pub fn classified_point(
        &mut self,
        position: [f64; 3],
        classification: impl Into<String>,
        tag_ids: Vec<u64>,
    ) -> MeshResult<MeshId> {
        let id = MeshId::new(self.id, POINT_ORDINAL, self.next_point)?;
        self.next_point += 1;
        self.point_copy(id, position, classification, tag_ids)?;
        Ok(id)
    }

    pub fn point_copy(
        &mut self,
        id: MeshId,
        position: [f64; 3],
        classification: impl Into<String>,
        tag_ids: Vec<u64>,
    ) -> MeshResult<()> {
        if position.into_iter().any(|value| !value.is_finite()) || !self.bounds.contains(position) {
            return Err(MeshError::InvalidInput(
                "chunk point must be finite and inside chunk bounds".into(),
            ));
        }
        let point = ChunkPoint {
            id,
            owner_chunk_id: id.owner_chunk(),
            position,
            classification: classification.into(),
            tag_ids,
        };
        if self.points.insert(id, point).is_some() {
            return Err(MeshError::InvalidInput(format!(
                "point {} is already present in chunk {}",
                id.raw(),
                self.id
            )));
        }
        if id.owner_chunk() == self.id {
            let ordinal = id.raw() as u32;
            if ordinal < EDGE_ORDINAL {
                self.next_point = self.next_point.max(ordinal.saturating_add(1));
            }
        }
        Ok(())
    }

    pub fn edge(
        &mut self,
        element_type: &str,
        point_ids: &[MeshId],
        tag_ids: Vec<u64>,
        boundary: bool,
    ) -> MeshResult<MeshId> {
        let id = MeshId::new(self.id, EDGE_ORDINAL, self.next_edge)?;
        self.next_edge += 1;
        self.edges.push(element(
            id,
            element_type,
            point_ids,
            tag_ids,
            boundary,
            None,
            None,
        )?);
        Ok(id)
    }

    pub fn boundary_edge(
        &mut self,
        point_ids: [MeshId; 2],
        tag_ids: Vec<u64>,
    ) -> MeshResult<MeshId> {
        self.edge("edge2", &point_ids, tag_ids, true)
    }

    pub fn face(
        &mut self,
        element_type: &str,
        point_ids: &[MeshId],
        tag_ids: Vec<u64>,
        boundary: bool,
    ) -> MeshResult<MeshId> {
        let id = MeshId::new(self.id, FACE_ORDINAL, self.next_face)?;
        self.next_face += 1;
        self.faces.push(element(
            id,
            element_type,
            point_ids,
            tag_ids,
            boundary,
            None,
            None,
        )?);
        Ok(id)
    }

    pub fn boundary_face(
        &mut self,
        element_type: &str,
        point_ids: &[MeshId],
        tag_ids: Vec<u64>,
    ) -> MeshResult<MeshId> {
        self.face(element_type, point_ids, tag_ids, true)
    }

    pub fn cell(
        &mut self,
        element_type: &str,
        point_ids: &[MeshId],
        zone_id: u64,
        source_id: u64,
    ) -> MeshResult<MeshId> {
        self.cell_with_faces(element_type, point_ids, &[], zone_id, source_id)
    }

    pub fn cell_with_faces(
        &mut self,
        element_type: &str,
        point_ids: &[MeshId],
        face_ids: &[MeshId],
        zone_id: u64,
        source_id: u64,
    ) -> MeshResult<MeshId> {
        let id = MeshId::new(self.id, CELL_ORDINAL, self.next_cell)?;
        self.next_cell += 1;
        let mut value = element(
            id,
            element_type,
            point_ids,
            Vec::new(),
            false,
            Some(zone_id),
            Some(source_id),
        )?;
        value.face_ids.extend_from_slice(face_ids);
        self.cells.push(value);
        Ok(id)
    }

    pub fn tri3(&mut self, points: [MeshId; 3], zone: u64, source: u64) -> MeshResult<MeshId> {
        self.cell("tri3", &points, zone, source)
    }

    pub fn quad4(&mut self, points: [MeshId; 4], zone: u64, source: u64) -> MeshResult<MeshId> {
        self.cell("quad4", &points, zone, source)
    }

    pub fn tet4(&mut self, points: [MeshId; 4], zone: u64, source: u64) -> MeshResult<MeshId> {
        self.cell("tet4", &points, zone, source)
    }

    pub fn pyramid5(&mut self, points: [MeshId; 5], zone: u64, source: u64) -> MeshResult<MeshId> {
        self.cell("pyramid5", &points, zone, source)
    }

    pub fn prism6(&mut self, points: [MeshId; 6], zone: u64, source: u64) -> MeshResult<MeshId> {
        self.cell("prism6", &points, zone, source)
    }

    pub fn hex8(&mut self, points: [MeshId; 8], zone: u64, source: u64) -> MeshResult<MeshId> {
        self.cell("hex8", &points, zone, source)
    }

    pub fn build(self, dimension: u8) -> MeshResult<MeshChunk> {
        let chunk = MeshChunk {
            id: self.id,
            bounds: self.bounds,
            points: self.points.into_values().collect(),
            edges: self.edges,
            faces: self.faces,
            cells: self.cells,
        };
        chunk.validate(dimension)?;
        Ok(chunk)
    }
}

fn element(
    id: MeshId,
    element_type: &str,
    point_ids: &[MeshId],
    tag_ids: Vec<u64>,
    boundary: bool,
    zone_id: Option<u64>,
    source_id: Option<u64>,
) -> MeshResult<ChunkElement> {
    let expected = expected_points(element_type)
        .ok_or_else(|| MeshError::InvalidInput(format!("unknown element type {element_type:?}")))?;
    if !expected.contains(&point_ids.len()) {
        return Err(MeshError::InvalidInput(format!(
            "{element_type} requires {:?} points, got {}",
            expected,
            point_ids.len()
        )));
    }
    Ok(ChunkElement {
        id,
        element_type: element_type.into(),
        point_ids: point_ids.to_vec(),
        edge_ids: Vec::new(),
        face_ids: Vec::new(),
        tag_ids,
        owner_cell_id: None,
        neighbor_cell_id: None,
        zone_id,
        source_id,
        boundary,
    })
}
