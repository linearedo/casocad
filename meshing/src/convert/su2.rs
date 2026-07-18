use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::MeshIr;

use super::{marker_name, mesh_cell_dimension, normalized_mesh, point_index, point_indices};

const SU2_LINE: u8 = 3;
const SU2_TRIANGLE: u8 = 5;
const SU2_QUAD: u8 = 9;
const SU2_TETRA: u8 = 10;
const SU2_HEX: u8 = 12;
const SU2_PRISM: u8 = 13;
const SU2_PYRAMID: u8 = 14;

struct Su2Element {
    code: u8,
    points: Vec<usize>,
}

pub fn write(mesh: &MeshIr) -> Result<Vec<u8>, String> {
    let mesh = normalized_mesh(mesh)?;
    let dimension = mesh_cell_dimension(&mesh, "SU2")?;
    let point_indices = point_indices(&mesh.points);
    let cells = cell_elements(&mesh, dimension, &point_indices)?;
    let markers = marker_elements(&mesh, dimension, &point_indices)?;

    let mut out = String::new();
    writeln!(out, "NDIME= {dimension}").map_err(|error| error.to_string())?;
    writeln!(out, "NPOIN= {}", mesh.points.len()).map_err(|error| error.to_string())?;
    for point in &mesh.points {
        if dimension == 2 {
            writeln!(out, "{:.17} {:.17}", point.position[0], point.position[1])
                .map_err(|error| error.to_string())?;
        } else {
            writeln!(
                out,
                "{:.17} {:.17} {:.17}",
                point.position[0], point.position[1], point.position[2]
            )
            .map_err(|error| error.to_string())?;
        }
    }
    writeln!(out, "NELEM= {}", cells.len()).map_err(|error| error.to_string())?;
    for cell in &cells {
        write_element(&mut out, cell)?;
    }
    writeln!(out, "NMARK= {}", markers.len()).map_err(|error| error.to_string())?;
    for (name, elements) in markers {
        writeln!(out, "MARKER_TAG= {name}").map_err(|error| error.to_string())?;
        writeln!(out, "MARKER_ELEMS= {}", elements.len()).map_err(|error| error.to_string())?;
        for element in &elements {
            write_element(&mut out, element)?;
        }
    }
    Ok(out.into_bytes())
}

fn cell_elements(
    mesh: &MeshIr,
    dimension: u8,
    point_indices: &BTreeMap<u64, usize>,
) -> Result<Vec<Su2Element>, String> {
    mesh.cells
        .iter()
        .map(|cell| {
            Ok(Su2Element {
                code: cell_code(&cell.type_name, dimension)?,
                points: convert_points(&cell.point_ids, point_indices)?,
            })
        })
        .collect()
}

fn marker_elements(
    mesh: &MeshIr,
    dimension: u8,
    point_indices: &BTreeMap<u64, usize>,
) -> Result<BTreeMap<String, Vec<Su2Element>>, String> {
    let mut markers: BTreeMap<String, Vec<Su2Element>> = BTreeMap::new();
    if dimension == 2 {
        for edge in &mesh.edges {
            if edge.owner_cell_id.is_some() && edge.neighbor_cell_id.is_none() {
                if edge.type_name != "edge2" {
                    return Err(format!(
                        "SU2 export does not support boundary edge type {:?}",
                        edge.type_name
                    ));
                }
                markers
                    .entry(marker_name(mesh, &edge.tag_ids))
                    .or_default()
                    .push(Su2Element {
                        code: SU2_LINE,
                        points: convert_points(&edge.point_ids, point_indices)?,
                    });
            }
        }
    } else {
        for face in &mesh.faces {
            if face.owner_cell_id.is_some() && face.neighbor_cell_id.is_none() {
                markers
                    .entry(marker_name(mesh, &face.tag_ids))
                    .or_default()
                    .push(Su2Element {
                        code: boundary_face_code(&face.type_name)?,
                        points: convert_points(&face.point_ids, point_indices)?,
                    });
            }
        }
    }
    Ok(markers)
}

fn cell_code(type_name: &str, dimension: u8) -> Result<u8, String> {
    match (dimension, type_name) {
        (2, "tri3") => Ok(SU2_TRIANGLE),
        (2, "quad4") => Ok(SU2_QUAD),
        (3, "tet4") => Ok(SU2_TETRA),
        (3, "hex8") => Ok(SU2_HEX),
        (3, "prism6") => Ok(SU2_PRISM),
        (3, "pyramid5") => Ok(SU2_PYRAMID),
        _ => Err(format!(
            "SU2 export does not support cell type {type_name:?}; supported linear types are tri3, quad4, tet4, hex8, prism6, pyramid5"
        )),
    }
}

fn boundary_face_code(type_name: &str) -> Result<u8, String> {
    match type_name {
        "tri3" => Ok(SU2_TRIANGLE),
        "quad4" => Ok(SU2_QUAD),
        _ => Err(format!(
            "SU2 export does not support boundary face type {type_name:?}"
        )),
    }
}

fn convert_points(
    point_ids: &[u64],
    point_indices: &BTreeMap<u64, usize>,
) -> Result<Vec<usize>, String> {
    point_ids
        .iter()
        .map(|point_id| point_index(point_indices, *point_id))
        .collect()
}

fn write_element(out: &mut String, element: &Su2Element) -> Result<(), String> {
    write!(out, "{}", element.code).map_err(|error| error.to_string())?;
    for point in &element.points {
        write!(out, " {point}").map_err(|error| error.to_string())?;
    }
    out.push('\n');
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeshIrBuilder;

    #[test]
    fn writes_2d_linear_cells_and_boundary_markers() {
        let mut builder = MeshIrBuilder::new();
        let zone = builder.zone("fluid", "fluid");
        let wall = builder.tag("wall", "boundary");
        let p0 = builder.point(0.0, 0.0, 0.0).unwrap();
        let p1 = builder.point(1.0, 0.0, 0.0).unwrap();
        let p2 = builder.point(1.0, 1.0, 0.0).unwrap();
        let p3 = builder.point(0.0, 1.0, 0.0).unwrap();
        let p4 = builder.point(2.0, 0.0, 0.0).unwrap();
        let p5 = builder.point(3.0, 0.0, 0.0).unwrap();
        let p6 = builder.point(2.0, 1.0, 0.0).unwrap();
        builder.cell("quad4", vec![p0, p1, p2, p3], zone).unwrap();
        builder.cell("tri3", vec![p4, p5, p6], zone).unwrap();
        builder.tag_edge(vec![p0, p1], wall);
        let text = String::from_utf8(write(&builder.build().unwrap()).unwrap()).unwrap();

        assert!(text.contains("NDIME= 2\n"));
        assert!(text.contains("NPOIN= 7\n"));
        assert!(text.contains("NELEM= 2\n"));
        assert!(text.contains("9 0 1 2 3\n"));
        assert!(text.contains("5 4 5 6\n"));
        assert!(text.contains("MARKER_TAG= wall\n"));
        assert!(text.contains("3 0 1\n"));
    }

    #[test]
    fn writes_3d_linear_cells() {
        let mut builder = MeshIrBuilder::new();
        let zone = builder.zone("solid", "solid");
        let ids: Vec<u64> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 1.0],
            [3.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
        ]
        .iter()
        .map(|point| builder.point(point[0], point[1], point[2]).unwrap())
        .collect();
        builder
            .cell("tet4", vec![ids[0], ids[1], ids[2], ids[3]], zone)
            .unwrap();
        builder
            .cell(
                "hex8",
                vec![
                    ids[4], ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11],
                ],
                zone,
            )
            .unwrap();
        let text = String::from_utf8(write(&builder.build().unwrap()).unwrap()).unwrap();

        assert!(text.contains("NDIME= 3\n"));
        assert!(text.contains("0.00000000000000000 0.00000000000000000 1.00000000000000000\n"));
        assert!(text.contains("10 0 1 2 3\n"));
        assert!(text.contains("12 4 5 6 7 8 9 10 11\n"));
    }

    #[test]
    fn rejects_high_order_cells() {
        let mut builder = MeshIrBuilder::new();
        let zone = builder.zone("fluid", "fluid");
        let ids: Vec<u64> = (0..6)
            .map(|index| builder.point(index as f64, 0.0, 0.0).unwrap())
            .collect();
        builder.cell("tri6", ids, zone).unwrap();
        let error = write(&builder.build().unwrap()).unwrap_err();

        assert!(error.contains("tri6"));
    }

    #[test]
    fn rejects_polyhedron_cells() {
        let mut builder = MeshIrBuilder::new();
        let zone = builder.zone("poly", "solid");
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
        .map(|point| builder.point(point[0], point[1], point[2]).unwrap())
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
        let error = write(&builder.build().unwrap()).unwrap_err();

        assert!(error.contains("polyhedron"));
    }

    #[test]
    fn rejects_mixed_dimension_cells() {
        let mut builder = MeshIrBuilder::new();
        let zone = builder.zone("mixed", "unknown");
        let p0 = builder.point(0.0, 0.0, 0.0).unwrap();
        let p1 = builder.point(1.0, 0.0, 0.0).unwrap();
        let p2 = builder.point(0.0, 1.0, 0.0).unwrap();
        let p3 = builder.point(0.0, 0.0, 1.0).unwrap();
        builder.cell("tri3", vec![p0, p1, p2], zone).unwrap();
        builder.cell("tet4", vec![p0, p1, p2, p3], zone).unwrap();
        let error = write(&builder.build().unwrap()).unwrap_err();

        assert!(error.contains("mixed 2D/3D"));
    }
}
