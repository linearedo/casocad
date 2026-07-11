//! Interop helper: write a sample .arrow artifact for the Python check.
use caso_meshing::{write_mesh_artifact, MeshElement};
fn main() {
    let path = std::env::args().nth(1).expect("usage: write_sample <out.arrow>");
    let elements = vec![
        MeshElement { element_type: "point".into(), vertices: vec![[0.0, 0.0, 0.0]], tag_name: "fluid_internal".into() },
        MeshElement { element_type: "triangle".into(), vertices: vec![[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]], tag_name: "wall".into() },
    ];
    let bytes = write_mesh_artifact(&elements, &serde_json::json!({"source": "casowasm", "dx": 0.08})).unwrap();
    std::fs::write(&path, bytes).unwrap();
    println!("wrote {path}");
}
