//! caso-meshing: solver mesh topology and Arrow IPC MeshIR artifacts.
//!
//! MeshIR v1 is a single Arrow entity table. Coordinates are world-space
//! meters; topology references shared point/edge/face ids.

#![forbid(unsafe_code)]

pub mod convert;
pub mod quality;
pub mod toolkit;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use arrow_array::builder::{Float64Builder, ListBuilder, StringBuilder, UInt64Builder};
use arrow_array::{
    Array, ArrayRef, Float64Array, ListArray, RecordBatch, StringArray, UInt64Array,
};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use arrow_schema::{Field, Schema};

pub use caso_kernel;

pub const MESH_IR_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct MeshPoint {
    pub id: u64,
    pub position: [f64; 3],
    pub tag_ids: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshEdge {
    pub id: u64,
    pub type_name: String,
    pub point_ids: Vec<u64>,
    pub owner_cell_id: Option<u64>,
    pub neighbor_cell_id: Option<u64>,
    pub tag_ids: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshFace {
    pub id: u64,
    pub type_name: String,
    pub point_ids: Vec<u64>,
    pub edge_ids: Vec<u64>,
    pub owner_cell_id: Option<u64>,
    pub neighbor_cell_id: Option<u64>,
    pub tag_ids: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshCell {
    pub id: u64,
    pub type_name: String,
    pub point_ids: Vec<u64>,
    pub edge_ids: Vec<u64>,
    pub face_ids: Vec<u64>,
    pub zone_id: Option<u64>,
    pub tag_ids: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshZone {
    pub id: u64,
    pub name: String,
    pub kind: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshTag {
    pub id: u64,
    pub name: String,
    pub kind: String,
    pub source_object_id: Option<u64>,
    pub source_region_id: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshAttribute {
    pub target_kind: String,
    pub target_id: u64,
    pub key: String,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct MeshIr {
    pub points: Vec<MeshPoint>,
    pub edges: Vec<MeshEdge>,
    pub faces: Vec<MeshFace>,
    pub cells: Vec<MeshCell>,
    pub zones: Vec<MeshZone>,
    pub tags: Vec<MeshTag>,
    pub attributes: Vec<MeshAttribute>,
}

#[derive(Debug, Clone, Default)]
pub struct MeshIrBuilder {
    mesh: MeshIr,
    next_point_id: u64,
    next_face_id: u64,
    next_cell_id: u64,
    next_zone_id: u64,
    next_tag_id: u64,
    pending_edge_tags: Vec<(Vec<u64>, u64)>,
    pending_face_tags: Vec<(Vec<u64>, u64)>,
}

impl MeshIrBuilder {
    pub fn new() -> Self {
        Self {
            next_point_id: 1,
            next_face_id: 1,
            next_cell_id: 1,
            next_zone_id: 1,
            next_tag_id: 1,
            ..Self::default()
        }
    }

    pub fn zone(&mut self, name: impl Into<String>, kind: impl Into<String>) -> u64 {
        let id = self.next_zone_id;
        self.next_zone_id += 1;
        self.mesh.zones.push(MeshZone {
            id,
            name: name.into(),
            kind: kind.into(),
        });
        id
    }

    pub fn tag(&mut self, name: impl Into<String>, kind: impl Into<String>) -> u64 {
        let id = self.next_tag_id;
        self.next_tag_id += 1;
        self.mesh.tags.push(MeshTag {
            id,
            name: name.into(),
            kind: kind.into(),
            source_object_id: None,
            source_region_id: None,
        });
        id
    }

    pub fn point(&mut self, x: f64, y: f64, z: f64) -> Result<u64, String> {
        for value in [x, y, z] {
            if !value.is_finite() {
                return Err("point coordinates must be finite".to_string());
            }
        }
        let id = self.next_point_id;
        self.next_point_id += 1;
        self.mesh.points.push(MeshPoint {
            id,
            position: [x, y, z],
            tag_ids: Vec::new(),
        });
        Ok(id)
    }

    pub fn face(
        &mut self,
        type_name: impl Into<String>,
        point_ids: Vec<u64>,
    ) -> Result<u64, String> {
        let type_name = type_name.into();
        validate_entity_type("face", &type_name, &point_ids)?;
        let id = self.next_face_id;
        self.next_face_id += 1;
        self.mesh.faces.push(MeshFace {
            id,
            type_name,
            point_ids,
            edge_ids: Vec::new(),
            owner_cell_id: None,
            neighbor_cell_id: None,
            tag_ids: Vec::new(),
        });
        Ok(id)
    }

    pub fn cell(
        &mut self,
        type_name: impl Into<String>,
        point_ids: Vec<u64>,
        zone_id: u64,
    ) -> Result<u64, String> {
        self.cell_with_faces(type_name, point_ids, Vec::new(), zone_id)
    }

    pub fn cell_with_faces(
        &mut self,
        type_name: impl Into<String>,
        point_ids: Vec<u64>,
        face_ids: Vec<u64>,
        zone_id: u64,
    ) -> Result<u64, String> {
        let type_name = type_name.into();
        validate_entity_type("cell", &type_name, &point_ids)?;
        let id = self.next_cell_id;
        self.next_cell_id += 1;
        self.mesh.cells.push(MeshCell {
            id,
            type_name,
            point_ids,
            edge_ids: Vec::new(),
            face_ids,
            zone_id: Some(zone_id),
            tag_ids: Vec::new(),
        });
        Ok(id)
    }

    pub fn tag_edge(&mut self, point_ids: Vec<u64>, tag_id: u64) {
        self.pending_edge_tags.push((point_ids, tag_id));
    }

    pub fn tag_face(&mut self, point_ids: Vec<u64>, tag_id: u64) {
        self.pending_face_tags.push((point_ids, tag_id));
    }

    pub fn attribute(
        &mut self,
        target_kind: impl Into<String>,
        target_id: u64,
        key: impl Into<String>,
        value: serde_json::Value,
    ) {
        self.mesh.attributes.push(MeshAttribute {
            target_kind: target_kind.into(),
            target_id,
            key: key.into(),
            value,
        });
    }

    pub fn build(mut self) -> Result<MeshIr, String> {
        self.mesh.derive_topology()?;
        self.apply_pending_tags()?;
        self.mesh.validate()?;
        Ok(self.mesh)
    }

    fn apply_pending_tags(&mut self) -> Result<(), String> {
        let edge_keys: BTreeMap<Vec<u64>, usize> = self
            .mesh
            .edges
            .iter()
            .enumerate()
            .map(|(index, edge)| (topology_key(&edge.point_ids), index))
            .collect();
        for (point_ids, tag_id) in &self.pending_edge_tags {
            let Some(index) = edge_keys.get(&topology_key(point_ids)) else {
                return Err(format!(
                    "tag_edge references no edge with points {point_ids:?}"
                ));
            };
            push_unique(&mut self.mesh.edges[*index].tag_ids, *tag_id);
        }
        let face_keys: BTreeMap<Vec<u64>, usize> = self
            .mesh
            .faces
            .iter()
            .enumerate()
            .map(|(index, face)| (topology_key(&face.point_ids), index))
            .collect();
        for (point_ids, tag_id) in &self.pending_face_tags {
            let Some(index) = face_keys.get(&topology_key(point_ids)) else {
                return Err(format!(
                    "tag_face references no face with points {point_ids:?}"
                ));
            };
            push_unique(&mut self.mesh.faces[*index].tag_ids, *tag_id);
        }
        Ok(())
    }
}

impl MeshIr {
    pub fn entity_count(&self) -> usize {
        self.points.len()
            + self.edges.len()
            + self.faces.len()
            + self.cells.len()
            + self.zones.len()
            + self.tags.len()
            + self.attributes.len()
    }

    pub fn tag_name(&self, id: u64) -> Option<&str> {
        self.tags
            .iter()
            .find(|tag| tag.id == id)
            .map(|tag| tag.name.as_str())
    }

    pub fn zone_name(&self, id: u64) -> Option<&str> {
        self.zones
            .iter()
            .find(|zone| zone.id == id)
            .map(|zone| zone.name.as_str())
    }

    pub fn point(&self, id: u64) -> Option<&MeshPoint> {
        self.points.iter().find(|point| point.id == id)
    }

    pub fn derive_topology(&mut self) -> Result<(), String> {
        let mut edge_map: BTreeMap<Vec<u64>, usize> = self
            .edges
            .iter()
            .enumerate()
            .map(|(index, edge)| (topology_key(&edge.point_ids), index))
            .collect();
        let mut face_map: BTreeMap<Vec<u64>, usize> = self
            .faces
            .iter()
            .enumerate()
            .map(|(index, face)| (topology_key(&face.point_ids), index))
            .collect();
        let mut next_edge_id = next_id(self.edges.iter().map(|edge| edge.id));
        let mut next_face_id = next_id(self.faces.iter().map(|face| face.id));

        for face_index in 0..self.faces.len() {
            if self.faces[face_index].edge_ids.is_empty() {
                let edge_ids = ensure_face_edges(
                    &mut self.edges,
                    &mut edge_map,
                    &mut next_edge_id,
                    &self.faces[face_index].type_name,
                    &self.faces[face_index].point_ids,
                )?;
                self.faces[face_index].edge_ids = edge_ids;
            }
        }

        for cell_index in 0..self.cells.len() {
            let cell_id = self.cells[cell_index].id;
            let type_name = self.cells[cell_index].type_name.clone();
            let point_ids = self.cells[cell_index].point_ids.clone();
            let dimension = element_dimension(&type_name)
                .ok_or_else(|| format!("unknown cell type {type_name:?}"))?;

            match dimension {
                1 => {
                    if self.cells[cell_index].edge_ids.is_empty() {
                        let edge_id = ensure_edge(
                            &mut self.edges,
                            &mut edge_map,
                            &mut next_edge_id,
                            &type_name,
                            &point_ids,
                        )?;
                        attach_cell_to_edge(&mut self.edges, edge_id, cell_id)?;
                        self.cells[cell_index].edge_ids = vec![edge_id];
                    }
                }
                2 => {
                    if self.cells[cell_index].edge_ids.is_empty() {
                        let mut edge_ids = Vec::new();
                        for (edge_type, ids) in cell_edges(&type_name, &point_ids)? {
                            let edge_id = ensure_edge(
                                &mut self.edges,
                                &mut edge_map,
                                &mut next_edge_id,
                                edge_type,
                                &ids,
                            )?;
                            attach_cell_to_edge(&mut self.edges, edge_id, cell_id)?;
                            edge_ids.push(edge_id);
                        }
                        self.cells[cell_index].edge_ids = edge_ids;
                    }
                }
                3 => {
                    if self.cells[cell_index].face_ids.is_empty() {
                        let faces = cell_faces(&type_name, &point_ids).ok_or_else(|| {
                            format!("{type_name} cells require explicit face_ids in MeshIR v1")
                        })?;
                        let mut face_ids = Vec::new();
                        let mut cell_edge_ids = Vec::new();
                        for (face_type, ids) in faces {
                            let face_id = ensure_face(
                                &mut self.faces,
                                &mut face_map,
                                &mut next_face_id,
                                face_type,
                                &ids,
                            )?;
                            let face_index = face_index_by_id(&self.faces, face_id)?;
                            if self.faces[face_index].edge_ids.is_empty() {
                                self.faces[face_index].edge_ids = ensure_face_edges(
                                    &mut self.edges,
                                    &mut edge_map,
                                    &mut next_edge_id,
                                    &self.faces[face_index].type_name,
                                    &self.faces[face_index].point_ids,
                                )?;
                            }
                            attach_cell_to_face(&mut self.faces, face_id, cell_id)?;
                            cell_edge_ids.extend(self.faces[face_index].edge_ids.iter().copied());
                            face_ids.push(face_id);
                        }
                        cell_edge_ids.sort_unstable();
                        cell_edge_ids.dedup();
                        self.cells[cell_index].face_ids = face_ids;
                        self.cells[cell_index].edge_ids = cell_edge_ids;
                    } else {
                        let face_ids = self.cells[cell_index].face_ids.clone();
                        let mut cell_edge_ids = Vec::new();
                        let mut cell_points = self.cells[cell_index].point_ids.clone();
                        for face_id in face_ids {
                            let face_index = face_index_by_id(&self.faces, face_id)?;
                            if self.faces[face_index].edge_ids.is_empty() {
                                self.faces[face_index].edge_ids = ensure_face_edges(
                                    &mut self.edges,
                                    &mut edge_map,
                                    &mut next_edge_id,
                                    &self.faces[face_index].type_name,
                                    &self.faces[face_index].point_ids,
                                )?;
                            }
                            attach_cell_to_face(&mut self.faces, face_id, cell_id)?;
                            cell_edge_ids.extend(self.faces[face_index].edge_ids.iter().copied());
                            cell_points.extend(self.faces[face_index].point_ids.iter().copied());
                        }
                        cell_points.sort_unstable();
                        cell_points.dedup();
                        cell_edge_ids.sort_unstable();
                        cell_edge_ids.dedup();
                        if self.cells[cell_index].point_ids.is_empty() {
                            self.cells[cell_index].point_ids = cell_points;
                        }
                        if self.cells[cell_index].edge_ids.is_empty() {
                            self.cells[cell_index].edge_ids = cell_edge_ids;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        unique_ids("point", self.points.iter().map(|point| point.id))?;
        unique_ids("edge", self.edges.iter().map(|edge| edge.id))?;
        unique_ids("face", self.faces.iter().map(|face| face.id))?;
        unique_ids("cell", self.cells.iter().map(|cell| cell.id))?;
        unique_ids("zone", self.zones.iter().map(|zone| zone.id))?;
        unique_ids("tag", self.tags.iter().map(|tag| tag.id))?;

        let points: BTreeSet<u64> = self.points.iter().map(|point| point.id).collect();
        let edges: BTreeSet<u64> = self.edges.iter().map(|edge| edge.id).collect();
        let faces: BTreeSet<u64> = self.faces.iter().map(|face| face.id).collect();
        let cells: BTreeSet<u64> = self.cells.iter().map(|cell| cell.id).collect();
        let zones: BTreeSet<u64> = self.zones.iter().map(|zone| zone.id).collect();
        let tags: BTreeSet<u64> = self.tags.iter().map(|tag| tag.id).collect();

        for point in &self.points {
            for value in point.position {
                if !value.is_finite() {
                    return Err(format!("point {} has a non-finite coordinate", point.id));
                }
            }
            require_ids("point tag", point.id, &point.tag_ids, &tags)?;
        }
        for edge in &self.edges {
            validate_entity_type("edge", &edge.type_name, &edge.point_ids)?;
            require_ids("edge point", edge.id, &edge.point_ids, &points)?;
            require_ids("edge tag", edge.id, &edge.tag_ids, &tags)?;
            require_optional_id("edge owner", edge.id, edge.owner_cell_id, &cells)?;
            require_optional_id("edge neighbor", edge.id, edge.neighbor_cell_id, &cells)?;
        }
        for face in &self.faces {
            validate_entity_type("face", &face.type_name, &face.point_ids)?;
            require_ids("face point", face.id, &face.point_ids, &points)?;
            require_ids("face edge", face.id, &face.edge_ids, &edges)?;
            require_ids("face tag", face.id, &face.tag_ids, &tags)?;
            require_optional_id("face owner", face.id, face.owner_cell_id, &cells)?;
            require_optional_id("face neighbor", face.id, face.neighbor_cell_id, &cells)?;
        }
        for cell in &self.cells {
            validate_entity_type("cell", &cell.type_name, &cell.point_ids)?;
            if cell.type_name == "polyhedron" && cell.face_ids.is_empty() {
                return Err(format!("cell {} polyhedron requires face_ids", cell.id));
            }
            require_ids("cell point", cell.id, &cell.point_ids, &points)?;
            require_ids("cell edge", cell.id, &cell.edge_ids, &edges)?;
            require_ids("cell face", cell.id, &cell.face_ids, &faces)?;
            require_ids("cell tag", cell.id, &cell.tag_ids, &tags)?;
            require_optional_id("cell zone", cell.id, cell.zone_id, &zones)?;
        }
        for attribute in &self.attributes {
            let exists = match attribute.target_kind.as_str() {
                "point" => points.contains(&attribute.target_id),
                "edge" => edges.contains(&attribute.target_id),
                "face" => faces.contains(&attribute.target_id),
                "cell" => cells.contains(&attribute.target_id),
                "zone" => zones.contains(&attribute.target_id),
                "tag" => tags.contains(&attribute.target_id),
                _ => false,
            };
            if !exists {
                return Err(format!(
                    "attribute {:?} references missing {} {}",
                    attribute.key, attribute.target_kind, attribute.target_id
                ));
            }
        }
        Ok(())
    }
}

fn next_id(ids: impl Iterator<Item = u64>) -> u64 {
    ids.max().unwrap_or(0) + 1
}

fn push_unique(ids: &mut Vec<u64>, id: u64) {
    if !ids.contains(&id) {
        ids.push(id);
    }
}

fn topology_key(ids: &[u64]) -> Vec<u64> {
    let mut key = ids.to_vec();
    key.sort_unstable();
    key.dedup();
    key
}

fn ensure_edge(
    edges: &mut Vec<MeshEdge>,
    edge_map: &mut BTreeMap<Vec<u64>, usize>,
    next_edge_id: &mut u64,
    type_name: &str,
    point_ids: &[u64],
) -> Result<u64, String> {
    validate_entity_type("edge", type_name, point_ids)?;
    let key = topology_key(point_ids);
    if let Some(index) = edge_map.get(&key) {
        return Ok(edges[*index].id);
    }
    let id = *next_edge_id;
    *next_edge_id += 1;
    let index = edges.len();
    edges.push(MeshEdge {
        id,
        type_name: type_name.to_string(),
        point_ids: point_ids.to_vec(),
        owner_cell_id: None,
        neighbor_cell_id: None,
        tag_ids: Vec::new(),
    });
    edge_map.insert(key, index);
    Ok(id)
}

fn ensure_face(
    faces: &mut Vec<MeshFace>,
    face_map: &mut BTreeMap<Vec<u64>, usize>,
    next_face_id: &mut u64,
    type_name: &str,
    point_ids: &[u64],
) -> Result<u64, String> {
    validate_entity_type("face", type_name, point_ids)?;
    let key = topology_key(point_ids);
    if let Some(index) = face_map.get(&key) {
        return Ok(faces[*index].id);
    }
    let id = *next_face_id;
    *next_face_id += 1;
    let index = faces.len();
    faces.push(MeshFace {
        id,
        type_name: type_name.to_string(),
        point_ids: point_ids.to_vec(),
        edge_ids: Vec::new(),
        owner_cell_id: None,
        neighbor_cell_id: None,
        tag_ids: Vec::new(),
    });
    face_map.insert(key, index);
    Ok(id)
}

fn ensure_face_edges(
    edges: &mut Vec<MeshEdge>,
    edge_map: &mut BTreeMap<Vec<u64>, usize>,
    next_edge_id: &mut u64,
    face_type: &str,
    point_ids: &[u64],
) -> Result<Vec<u64>, String> {
    let mut edge_ids = Vec::new();
    for (a, b) in element_wire_edges(face_type, point_ids) {
        edge_ids.push(ensure_edge(
            edges,
            edge_map,
            next_edge_id,
            "edge2",
            &[a, b],
        )?);
    }
    Ok(edge_ids)
}

fn attach_cell_to_edge(edges: &mut [MeshEdge], edge_id: u64, cell_id: u64) -> Result<(), String> {
    let Some(edge) = edges.iter_mut().find(|edge| edge.id == edge_id) else {
        return Err(format!("missing edge {edge_id}"));
    };
    attach_owner_neighbor(
        "edge",
        edge.id,
        &mut edge.owner_cell_id,
        &mut edge.neighbor_cell_id,
        cell_id,
    )
}

fn attach_cell_to_face(faces: &mut [MeshFace], face_id: u64, cell_id: u64) -> Result<(), String> {
    let index = face_index_by_id(faces, face_id)?;
    attach_owner_neighbor(
        "face",
        faces[index].id,
        &mut faces[index].owner_cell_id,
        &mut faces[index].neighbor_cell_id,
        cell_id,
    )
}

fn attach_owner_neighbor(
    entity: &str,
    entity_id: u64,
    owner: &mut Option<u64>,
    neighbor: &mut Option<u64>,
    cell_id: u64,
) -> Result<(), String> {
    if *owner == Some(cell_id) || *neighbor == Some(cell_id) {
        return Ok(());
    }
    if owner.is_none() {
        *owner = Some(cell_id);
        return Ok(());
    }
    if neighbor.is_none() {
        *neighbor = Some(cell_id);
        return Ok(());
    }
    Err(format!(
        "{entity} {entity_id} is non-manifold: more than two owner cells"
    ))
}

fn face_index_by_id(faces: &[MeshFace], face_id: u64) -> Result<usize, String> {
    faces
        .iter()
        .position(|face| face.id == face_id)
        .ok_or_else(|| format!("missing face {face_id}"))
}

fn unique_ids(name: &str, ids: impl Iterator<Item = u64>) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for id in ids {
        if id == 0 {
            return Err(format!("{name} id 0 is reserved"));
        }
        if !seen.insert(id) {
            return Err(format!("duplicate {name} id {id}"));
        }
    }
    Ok(())
}

fn require_ids(
    label: &str,
    owner_id: u64,
    ids: &[u64],
    allowed: &BTreeSet<u64>,
) -> Result<(), String> {
    for id in ids {
        if !allowed.contains(id) {
            return Err(format!(
                "{label} reference {id} on entity {owner_id} does not exist"
            ));
        }
    }
    Ok(())
}

fn require_optional_id(
    label: &str,
    owner_id: u64,
    id: Option<u64>,
    allowed: &BTreeSet<u64>,
) -> Result<(), String> {
    if let Some(id) = id {
        if !allowed.contains(&id) {
            return Err(format!(
                "{label} reference {id} on entity {owner_id} does not exist"
            ));
        }
    }
    Ok(())
}

fn element_dimension(type_name: &str) -> Option<u8> {
    Some(match type_name {
        "point1" => 0,
        "edge2" | "edge3" => 1,
        "tri3" | "tri6" | "quad4" | "quad8" | "quad9" | "polygon" => 2,
        "tet4" | "tet10" | "hex8" | "hex20" | "hex27" | "prism6" | "prism15" | "pyramid5"
        | "pyramid13" | "polyhedron" => 3,
        _ => return None,
    })
}

fn validate_entity_type(
    entity_kind: &str,
    type_name: &str,
    point_ids: &[u64],
) -> Result<(), String> {
    let Some(dimension) = element_dimension(type_name) else {
        return Err(format!("unknown {entity_kind} type {type_name:?}"));
    };
    match entity_kind {
        "edge" if dimension != 1 => {
            return Err(format!("edge cannot use {type_name:?}"));
        }
        "face" if dimension != 2 => {
            return Err(format!("face cannot use {type_name:?}"));
        }
        "cell" => {}
        _ => {}
    }
    let ok = match type_name {
        "point1" => point_ids.len() == 1,
        "edge2" => point_ids.len() == 2,
        "edge3" => point_ids.len() == 3,
        "tri3" => point_ids.len() == 3,
        "tri6" => point_ids.len() == 6,
        "quad4" => point_ids.len() == 4,
        "quad8" => point_ids.len() == 8,
        "quad9" => point_ids.len() == 9,
        "polygon" => point_ids.len() >= 3,
        "tet4" => point_ids.len() == 4,
        "tet10" => point_ids.len() == 10,
        "hex8" => point_ids.len() == 8,
        "hex20" => point_ids.len() == 20,
        "hex27" => point_ids.len() == 27,
        "prism6" => point_ids.len() == 6,
        "prism15" => point_ids.len() == 15,
        "pyramid5" => point_ids.len() == 5,
        "pyramid13" => point_ids.len() == 13,
        "polyhedron" => point_ids.len() >= 4 || point_ids.is_empty(),
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(format!(
            "{type_name} has invalid point count {}; expected MeshIR v1 ordering",
            point_ids.len()
        ))
    }
}

fn cell_edges(type_name: &str, point_ids: &[u64]) -> Result<Vec<(&'static str, Vec<u64>)>, String> {
    validate_entity_type("cell", type_name, point_ids)?;
    let edges = match type_name {
        "edge2" => vec![("edge2", point_ids.to_vec())],
        "edge3" => vec![("edge3", point_ids.to_vec())],
        "tri3" => cycle_edges(&point_ids[..3], "edge2"),
        "tri6" => vec![
            ("edge3", vec![point_ids[0], point_ids[1], point_ids[3]]),
            ("edge3", vec![point_ids[1], point_ids[2], point_ids[4]]),
            ("edge3", vec![point_ids[2], point_ids[0], point_ids[5]]),
        ],
        "quad4" => cycle_edges(&point_ids[..4], "edge2"),
        "quad8" | "quad9" => vec![
            ("edge3", vec![point_ids[0], point_ids[1], point_ids[4]]),
            ("edge3", vec![point_ids[1], point_ids[2], point_ids[5]]),
            ("edge3", vec![point_ids[2], point_ids[3], point_ids[6]]),
            ("edge3", vec![point_ids[3], point_ids[0], point_ids[7]]),
        ],
        "polygon" => cycle_edges(point_ids, "edge2"),
        _ => Vec::new(),
    };
    Ok(edges)
}

fn cycle_edges(points: &[u64], type_name: &'static str) -> Vec<(&'static str, Vec<u64>)> {
    (0..points.len())
        .map(|index| {
            (
                type_name,
                vec![points[index], points[(index + 1) % points.len()]],
            )
        })
        .collect()
}

fn cell_faces(type_name: &str, point_ids: &[u64]) -> Option<Vec<(&'static str, Vec<u64>)>> {
    let faces = match type_name {
        "tet4" => vec![
            ("tri3", vec![point_ids[0], point_ids[2], point_ids[1]]),
            ("tri3", vec![point_ids[0], point_ids[1], point_ids[3]]),
            ("tri3", vec![point_ids[1], point_ids[2], point_ids[3]]),
            ("tri3", vec![point_ids[2], point_ids[0], point_ids[3]]),
        ],
        "hex8" => vec![
            (
                "quad4",
                vec![point_ids[0], point_ids[3], point_ids[2], point_ids[1]],
            ),
            (
                "quad4",
                vec![point_ids[0], point_ids[1], point_ids[5], point_ids[4]],
            ),
            (
                "quad4",
                vec![point_ids[1], point_ids[2], point_ids[6], point_ids[5]],
            ),
            (
                "quad4",
                vec![point_ids[2], point_ids[3], point_ids[7], point_ids[6]],
            ),
            (
                "quad4",
                vec![point_ids[3], point_ids[0], point_ids[4], point_ids[7]],
            ),
            (
                "quad4",
                vec![point_ids[4], point_ids[5], point_ids[6], point_ids[7]],
            ),
        ],
        "prism6" => vec![
            ("tri3", vec![point_ids[0], point_ids[2], point_ids[1]]),
            ("tri3", vec![point_ids[3], point_ids[4], point_ids[5]]),
            (
                "quad4",
                vec![point_ids[0], point_ids[1], point_ids[4], point_ids[3]],
            ),
            (
                "quad4",
                vec![point_ids[1], point_ids[2], point_ids[5], point_ids[4]],
            ),
            (
                "quad4",
                vec![point_ids[2], point_ids[0], point_ids[3], point_ids[5]],
            ),
        ],
        "pyramid5" => vec![
            (
                "quad4",
                vec![point_ids[0], point_ids[3], point_ids[2], point_ids[1]],
            ),
            ("tri3", vec![point_ids[0], point_ids[1], point_ids[4]]),
            ("tri3", vec![point_ids[1], point_ids[2], point_ids[4]]),
            ("tri3", vec![point_ids[2], point_ids[3], point_ids[4]]),
            ("tri3", vec![point_ids[3], point_ids[0], point_ids[4]]),
        ],
        _ => return None,
    };
    Some(faces)
}

pub fn element_wire_edges(type_name: &str, point_ids: &[u64]) -> Vec<(u64, u64)> {
    match type_name {
        "edge2" if point_ids.len() == 2 => vec![(point_ids[0], point_ids[1])],
        "edge3" if point_ids.len() == 3 => {
            vec![(point_ids[0], point_ids[2]), (point_ids[2], point_ids[1])]
        }
        "tri3" | "tri6" if point_ids.len() >= 3 => {
            vec![
                (point_ids[0], point_ids[1]),
                (point_ids[1], point_ids[2]),
                (point_ids[2], point_ids[0]),
            ]
        }
        "quad4" | "quad8" | "quad9" if point_ids.len() >= 4 => vec![
            (point_ids[0], point_ids[1]),
            (point_ids[1], point_ids[2]),
            (point_ids[2], point_ids[3]),
            (point_ids[3], point_ids[0]),
        ],
        "polygon" if point_ids.len() >= 3 => (0..point_ids.len())
            .map(|index| (point_ids[index], point_ids[(index + 1) % point_ids.len()]))
            .collect(),
        _ => Vec::new(),
    }
}

fn mesh_ir_schema(metadata_json: &str, columns: &[ArrayRef]) -> Schema {
    let names = [
        "entity_kind",
        "id",
        "type_name",
        "name",
        "x",
        "y",
        "z",
        "point_ids",
        "edge_ids",
        "face_ids",
        "owner_cell_id",
        "neighbor_cell_id",
        "zone_id",
        "tag_ids",
        "source_object_id",
        "source_region_id",
        "target_kind",
        "target_id",
        "key",
        "value_json",
    ];
    let fields: Vec<Field> = names
        .iter()
        .zip(columns)
        .map(|(name, column)| {
            Field::new(
                *name,
                column.data_type().clone(),
                *name != "entity_kind" && *name != "id",
            )
        })
        .collect();
    let mut map = HashMap::new();
    map.insert(
        "casocad.mesh_ir.schema".to_string(),
        MESH_IR_SCHEMA_VERSION.to_string(),
    );
    map.insert("metadata".to_string(), metadata_json.to_string());
    Schema::new(fields).with_metadata(map)
}

pub fn write_mesh_ir(mesh: &MeshIr, metadata: &serde_json::Value) -> Result<Vec<u8>, String> {
    let mut mesh = mesh.clone();
    mesh.derive_topology()?;
    mesh.validate()?;

    let mut entity_kind = StringBuilder::new();
    let mut id = UInt64Builder::new();
    let mut type_name = StringBuilder::new();
    let mut name = StringBuilder::new();
    let mut x = Float64Builder::new();
    let mut y = Float64Builder::new();
    let mut z = Float64Builder::new();
    let mut point_ids = ListBuilder::new(UInt64Builder::new());
    let mut edge_ids = ListBuilder::new(UInt64Builder::new());
    let mut face_ids = ListBuilder::new(UInt64Builder::new());
    let mut owner_cell_id = UInt64Builder::new();
    let mut neighbor_cell_id = UInt64Builder::new();
    let mut zone_id = UInt64Builder::new();
    let mut tag_ids = ListBuilder::new(UInt64Builder::new());
    let mut source_object_id = UInt64Builder::new();
    let mut source_region_id = UInt64Builder::new();
    let mut target_kind = StringBuilder::new();
    let mut target_id = UInt64Builder::new();
    let mut key = StringBuilder::new();
    let mut value_json = StringBuilder::new();

    let mut row = |kind: &str,
                   row_id: u64,
                   row_type: Option<&str>,
                   row_name: Option<&str>,
                   row_xyz: Option<[f64; 3]>,
                   row_point_ids: Option<&[u64]>,
                   row_edge_ids: Option<&[u64]>,
                   row_face_ids: Option<&[u64]>,
                   row_owner_cell_id: Option<u64>,
                   row_neighbor_cell_id: Option<u64>,
                   row_zone_id: Option<u64>,
                   row_tag_ids: Option<&[u64]>,
                   row_source_object_id: Option<u64>,
                   row_source_region_id: Option<u64>,
                   row_target_kind: Option<&str>,
                   row_target_id: Option<u64>,
                   row_key: Option<&str>,
                   row_value_json: Option<&str>| {
        entity_kind.append_value(kind);
        id.append_value(row_id);
        append_string(&mut type_name, row_type);
        append_string(&mut name, row_name);
        append_xyz(row_xyz, &mut x, &mut y, &mut z);
        append_id_list(&mut point_ids, row_point_ids);
        append_id_list(&mut edge_ids, row_edge_ids);
        append_id_list(&mut face_ids, row_face_ids);
        append_u64(&mut owner_cell_id, row_owner_cell_id);
        append_u64(&mut neighbor_cell_id, row_neighbor_cell_id);
        append_u64(&mut zone_id, row_zone_id);
        append_id_list(&mut tag_ids, row_tag_ids);
        append_u64(&mut source_object_id, row_source_object_id);
        append_u64(&mut source_region_id, row_source_region_id);
        append_string(&mut target_kind, row_target_kind);
        append_u64(&mut target_id, row_target_id);
        append_string(&mut key, row_key);
        append_string(&mut value_json, row_value_json);
    };

    for point in &mesh.points {
        row(
            "point",
            point.id,
            Some("point1"),
            None,
            Some(point.position),
            None,
            None,
            None,
            None,
            None,
            None,
            Some(&point.tag_ids),
            None,
            None,
            None,
            None,
            None,
            None,
        );
    }
    for edge in &mesh.edges {
        row(
            "edge",
            edge.id,
            Some(&edge.type_name),
            None,
            None,
            Some(&edge.point_ids),
            None,
            None,
            edge.owner_cell_id,
            edge.neighbor_cell_id,
            None,
            Some(&edge.tag_ids),
            None,
            None,
            None,
            None,
            None,
            None,
        );
    }
    for face in &mesh.faces {
        row(
            "face",
            face.id,
            Some(&face.type_name),
            None,
            None,
            Some(&face.point_ids),
            Some(&face.edge_ids),
            None,
            face.owner_cell_id,
            face.neighbor_cell_id,
            None,
            Some(&face.tag_ids),
            None,
            None,
            None,
            None,
            None,
            None,
        );
    }
    for cell in &mesh.cells {
        row(
            "cell",
            cell.id,
            Some(&cell.type_name),
            None,
            None,
            Some(&cell.point_ids),
            Some(&cell.edge_ids),
            Some(&cell.face_ids),
            None,
            None,
            cell.zone_id,
            Some(&cell.tag_ids),
            None,
            None,
            None,
            None,
            None,
            None,
        );
    }
    for zone in &mesh.zones {
        row(
            "zone",
            zone.id,
            Some(&zone.kind),
            Some(&zone.name),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
    }
    for tag in &mesh.tags {
        row(
            "tag",
            tag.id,
            Some(&tag.kind),
            Some(&tag.name),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            tag.source_object_id,
            tag.source_region_id,
            None,
            None,
            None,
            None,
        );
    }
    for (index, attribute) in mesh.attributes.iter().enumerate() {
        let value = to_sorted_compact_json(&attribute.value);
        row(
            "attribute",
            index as u64 + 1,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(&attribute.target_kind),
            Some(attribute.target_id),
            Some(&attribute.key),
            Some(&value),
        );
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(entity_kind.finish()),
        Arc::new(id.finish()),
        Arc::new(type_name.finish()),
        Arc::new(name.finish()),
        Arc::new(x.finish()),
        Arc::new(y.finish()),
        Arc::new(z.finish()),
        Arc::new(point_ids.finish()),
        Arc::new(edge_ids.finish()),
        Arc::new(face_ids.finish()),
        Arc::new(owner_cell_id.finish()),
        Arc::new(neighbor_cell_id.finish()),
        Arc::new(zone_id.finish()),
        Arc::new(tag_ids.finish()),
        Arc::new(source_object_id.finish()),
        Arc::new(source_region_id.finish()),
        Arc::new(target_kind.finish()),
        Arc::new(target_id.finish()),
        Arc::new(key.finish()),
        Arc::new(value_json.finish()),
    ];
    let metadata_json = to_sorted_compact_json(metadata);
    let schema = Arc::new(mesh_ir_schema(&metadata_json, &columns));
    let batch = RecordBatch::try_new(schema.clone(), columns).map_err(|error| error.to_string())?;
    let mut bytes = Vec::new();
    {
        let mut writer =
            FileWriter::try_new(&mut bytes, &schema).map_err(|error| error.to_string())?;
        writer.write(&batch).map_err(|error| error.to_string())?;
        writer.finish().map_err(|error| error.to_string())?;
    }
    Ok(bytes)
}

fn append_string(builder: &mut StringBuilder, value: Option<&str>) {
    match value {
        Some(value) => builder.append_value(value),
        None => builder.append_null(),
    }
}

fn append_u64(builder: &mut UInt64Builder, value: Option<u64>) {
    match value {
        Some(value) => builder.append_value(value),
        None => builder.append_null(),
    }
}

fn append_xyz(
    value: Option<[f64; 3]>,
    x: &mut Float64Builder,
    y: &mut Float64Builder,
    z: &mut Float64Builder,
) {
    match value {
        Some([vx, vy, vz]) => {
            x.append_value(vx);
            y.append_value(vy);
            z.append_value(vz);
        }
        None => {
            x.append_null();
            y.append_null();
            z.append_null();
        }
    }
}

fn append_id_list(builder: &mut ListBuilder<UInt64Builder>, ids: Option<&[u64]>) {
    match ids {
        Some(ids) => {
            for id in ids {
                builder.values().append_value(*id);
            }
            builder.append(true);
        }
        None => builder.append(false),
    }
}

pub fn read_mesh_ir(bytes: &[u8]) -> Result<(MeshIr, serde_json::Value), String> {
    let cursor = std::io::Cursor::new(bytes);
    let reader = FileReader::try_new(cursor, None).map_err(|error| error.to_string())?;
    let metadata_json = reader
        .schema()
        .metadata()
        .get("metadata")
        .cloned()
        .unwrap_or_else(|| "{}".to_string());
    let metadata: serde_json::Value =
        serde_json::from_str(&metadata_json).map_err(|error| error.to_string())?;
    let schema_version = reader
        .schema()
        .metadata()
        .get("casocad.mesh_ir.schema")
        .ok_or("missing casocad.mesh_ir.schema metadata")?
        .parse::<u32>()
        .map_err(|error| error.to_string())?;
    if schema_version != MESH_IR_SCHEMA_VERSION {
        return Err(format!(
            "unsupported MeshIR schema {schema_version}; expected {MESH_IR_SCHEMA_VERSION}"
        ));
    }

    let mut mesh = MeshIr::default();
    for batch in reader {
        let batch = batch.map_err(|error| error.to_string())?;
        let entity_kind = string_column(&batch, "entity_kind")?;
        let id = u64_column(&batch, "id")?;
        let type_name = string_column(&batch, "type_name")?;
        let name = string_column(&batch, "name")?;
        let x = f64_column(&batch, "x")?;
        let y = f64_column(&batch, "y")?;
        let z = f64_column(&batch, "z")?;
        let point_ids = list_u64_column(&batch, "point_ids")?;
        let edge_ids = list_u64_column(&batch, "edge_ids")?;
        let face_ids = list_u64_column(&batch, "face_ids")?;
        let owner_cell_id = u64_column(&batch, "owner_cell_id")?;
        let neighbor_cell_id = u64_column(&batch, "neighbor_cell_id")?;
        let zone_id = u64_column(&batch, "zone_id")?;
        let tag_ids = list_u64_column(&batch, "tag_ids")?;
        let source_object_id = u64_column(&batch, "source_object_id")?;
        let source_region_id = u64_column(&batch, "source_region_id")?;
        let target_kind = string_column(&batch, "target_kind")?;
        let target_id = u64_column(&batch, "target_id")?;
        let key = string_column(&batch, "key")?;
        let value_json = string_column(&batch, "value_json")?;

        for row in 0..batch.num_rows() {
            let row_kind = required_string(entity_kind, row, "entity_kind")?;
            let row_id = required_u64(id, row, "id")?;
            match row_kind {
                "point" => mesh.points.push(MeshPoint {
                    id: row_id,
                    position: [
                        required_f64(x, row, "x")?,
                        required_f64(y, row, "y")?,
                        required_f64(z, row, "z")?,
                    ],
                    tag_ids: optional_ids(tag_ids, row).unwrap_or_default(),
                }),
                "edge" => mesh.edges.push(MeshEdge {
                    id: row_id,
                    type_name: required_string(type_name, row, "type_name")?.to_string(),
                    point_ids: required_ids(point_ids, row, "point_ids")?,
                    owner_cell_id: optional_u64(owner_cell_id, row),
                    neighbor_cell_id: optional_u64(neighbor_cell_id, row),
                    tag_ids: optional_ids(tag_ids, row).unwrap_or_default(),
                }),
                "face" => mesh.faces.push(MeshFace {
                    id: row_id,
                    type_name: required_string(type_name, row, "type_name")?.to_string(),
                    point_ids: required_ids(point_ids, row, "point_ids")?,
                    edge_ids: optional_ids(edge_ids, row).unwrap_or_default(),
                    owner_cell_id: optional_u64(owner_cell_id, row),
                    neighbor_cell_id: optional_u64(neighbor_cell_id, row),
                    tag_ids: optional_ids(tag_ids, row).unwrap_or_default(),
                }),
                "cell" => mesh.cells.push(MeshCell {
                    id: row_id,
                    type_name: required_string(type_name, row, "type_name")?.to_string(),
                    point_ids: required_ids(point_ids, row, "point_ids")?,
                    edge_ids: optional_ids(edge_ids, row).unwrap_or_default(),
                    face_ids: optional_ids(face_ids, row).unwrap_or_default(),
                    zone_id: optional_u64(zone_id, row),
                    tag_ids: optional_ids(tag_ids, row).unwrap_or_default(),
                }),
                "zone" => mesh.zones.push(MeshZone {
                    id: row_id,
                    name: required_string(name, row, "name")?.to_string(),
                    kind: required_string(type_name, row, "type_name")?.to_string(),
                }),
                "tag" => mesh.tags.push(MeshTag {
                    id: row_id,
                    name: required_string(name, row, "name")?.to_string(),
                    kind: required_string(type_name, row, "type_name")?.to_string(),
                    source_object_id: optional_u64(source_object_id, row),
                    source_region_id: optional_u64(source_region_id, row),
                }),
                "attribute" => mesh.attributes.push(MeshAttribute {
                    target_kind: required_string(target_kind, row, "target_kind")?.to_string(),
                    target_id: required_u64(target_id, row, "target_id")?,
                    key: required_string(key, row, "key")?.to_string(),
                    value: serde_json::from_str(required_string(value_json, row, "value_json")?)
                        .map_err(|error| error.to_string())?,
                }),
                other => return Err(format!("unknown MeshIR entity_kind {other:?}")),
            }
        }
    }
    mesh.validate()?;
    Ok((mesh, metadata))
}

fn string_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray, String> {
    batch
        .column_by_name(name)
        .and_then(|array| array.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| format!("missing {name} column"))
}

fn u64_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt64Array, String> {
    batch
        .column_by_name(name)
        .and_then(|array| array.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| format!("missing {name} column"))
}

fn f64_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float64Array, String> {
    batch
        .column_by_name(name)
        .and_then(|array| array.as_any().downcast_ref::<Float64Array>())
        .ok_or_else(|| format!("missing {name} column"))
}

fn list_u64_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ListArray, String> {
    batch
        .column_by_name(name)
        .and_then(|array| array.as_any().downcast_ref::<ListArray>())
        .ok_or_else(|| format!("missing {name} column"))
}

fn required_string<'a>(array: &'a StringArray, row: usize, name: &str) -> Result<&'a str, String> {
    if array.is_null(row) {
        return Err(format!("row {row} missing {name}"));
    }
    Ok(array.value(row))
}

fn required_u64(array: &UInt64Array, row: usize, name: &str) -> Result<u64, String> {
    optional_u64(array, row).ok_or_else(|| format!("row {row} missing {name}"))
}

fn required_f64(array: &Float64Array, row: usize, name: &str) -> Result<f64, String> {
    if array.is_null(row) {
        return Err(format!("row {row} missing {name}"));
    }
    Ok(array.value(row))
}

fn optional_u64(array: &UInt64Array, row: usize) -> Option<u64> {
    if array.is_null(row) {
        None
    } else {
        Some(array.value(row))
    }
}

fn required_ids(array: &ListArray, row: usize, name: &str) -> Result<Vec<u64>, String> {
    optional_ids(array, row).ok_or_else(|| format!("row {row} missing {name}"))
}

fn optional_ids(array: &ListArray, row: usize) -> Option<Vec<u64>> {
    if array.is_null(row) {
        return None;
    }
    let values = array.value(row);
    let ids = values.as_any().downcast_ref::<UInt64Array>()?;
    let mut out = Vec::with_capacity(ids.len());
    for index in 0..ids.len() {
        if ids.is_valid(index) {
            out.push(ids.value(index));
        }
    }
    Some(out)
}

fn to_sorted_compact_json(value: &serde_json::Value) -> String {
    fn write(value: &serde_json::Value, out: &mut String) {
        match value {
            serde_json::Value::Object(map) => {
                out.push('{');
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                for (index, key) in keys.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    out.push_str(&serde_json::to_string(key).expect("string"));
                    out.push(':');
                    write(&map[*key], out);
                }
                out.push('}');
            }
            serde_json::Value::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    write(item, out);
                }
                out.push(']');
            }
            other => out.push_str(&other.to_string()),
        }
    }
    let mut out = String::new();
    write(value, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn mesh_ir_round_trips_shared_triangles() {
        let mut builder = MeshIrBuilder::new();
        let zone = builder.zone("fluid", "fluid");
        let wall = builder.tag("wall", "boundary");
        let p0 = builder.point(0.0, 0.0, 0.0).unwrap();
        let p1 = builder.point(1.0, 0.0, 0.0).unwrap();
        let p2 = builder.point(1.0, 1.0, 0.0).unwrap();
        let p3 = builder.point(0.0, 1.0, 0.0).unwrap();
        builder.cell("tri3", vec![p0, p1, p2], zone).unwrap();
        builder.cell("tri3", vec![p0, p2, p3], zone).unwrap();
        builder.tag_edge(vec![p0, p1], wall);
        let mesh = builder.build().unwrap();

        assert_eq!(mesh.points.len(), 4);
        assert_eq!(mesh.cells.len(), 2);
        assert_eq!(mesh.edges.len(), 5);
        assert_eq!(
            mesh.edges
                .iter()
                .filter(|edge| edge.owner_cell_id.is_some() && edge.neighbor_cell_id.is_some())
                .count(),
            1
        );

        let bytes = write_mesh_ir(&mesh, &json!({"source": "test"})).unwrap();
        let (read, metadata) = read_mesh_ir(&bytes).unwrap();
        assert_eq!(metadata, json!({"source": "test"}));
        assert_eq!(read, mesh);
        assert_eq!(read.tag_name(wall), Some("wall"));
    }

    #[test]
    fn tetrahedra_derive_shared_faces() {
        let mut builder = MeshIrBuilder::new();
        let zone = builder.zone("solid", "solid");
        let p0 = builder.point(0.0, 0.0, 0.0).unwrap();
        let p1 = builder.point(1.0, 0.0, 0.0).unwrap();
        let p2 = builder.point(0.0, 1.0, 0.0).unwrap();
        let p3 = builder.point(0.0, 0.0, 1.0).unwrap();
        let p4 = builder.point(0.0, 0.0, -1.0).unwrap();
        builder.cell("tet4", vec![p0, p1, p2, p3], zone).unwrap();
        builder.cell("tet4", vec![p0, p2, p1, p4], zone).unwrap();
        let mesh = builder.build().unwrap();

        assert_eq!(mesh.cells.len(), 2);
        assert_eq!(mesh.faces.len(), 7);
        assert_eq!(
            mesh.faces
                .iter()
                .filter(|face| face.owner_cell_id.is_some() && face.neighbor_cell_id.is_some())
                .count(),
            1
        );
    }

    #[test]
    fn polyhedron_round_trips_with_explicit_faces() {
        let mut builder = MeshIrBuilder::new();
        let zone = builder.zone("poly", "unknown");
        let ids: Vec<u64> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
        .iter()
        .map(|p| builder.point(p[0], p[1], p[2]).unwrap())
        .collect();
        let faces = vec![
            builder
                .face("quad4", vec![ids[0], ids[1], ids[2], ids[3]])
                .unwrap(),
            builder
                .face("quad4", vec![ids[4], ids[5], ids[6], ids[7]])
                .unwrap(),
            builder
                .face("quad4", vec![ids[0], ids[1], ids[5], ids[4]])
                .unwrap(),
            builder
                .face("quad4", vec![ids[1], ids[2], ids[6], ids[5]])
                .unwrap(),
            builder
                .face("quad4", vec![ids[2], ids[3], ids[7], ids[6]])
                .unwrap(),
            builder
                .face("quad4", vec![ids[3], ids[0], ids[4], ids[7]])
                .unwrap(),
        ];
        builder
            .cell_with_faces("polyhedron", Vec::new(), faces, zone)
            .unwrap();
        let mesh = builder.build().unwrap();

        assert_eq!(mesh.cells[0].point_ids.len(), 8);
        let bytes = write_mesh_ir(&mesh, &json!({"schema": 1})).unwrap();
        let (read, _) = read_mesh_ir(&bytes).unwrap();
        assert_eq!(read, mesh);
    }

    #[test]
    fn invalid_references_fail() {
        let mut mesh = MeshIr::default();
        mesh.cells.push(MeshCell {
            id: 1,
            type_name: "tri3".to_string(),
            point_ids: vec![1, 2, 3],
            edge_ids: Vec::new(),
            face_ids: Vec::new(),
            zone_id: None,
            tag_ids: Vec::new(),
        });
        assert!(mesh.validate().unwrap_err().contains("does not exist"));
    }
}
