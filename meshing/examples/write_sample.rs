//! Interop helper: write a sample MeshIR .arrow artifact.
use caso_meshing::{write_mesh_ir, MeshIrBuilder};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: write_sample <out.arrow>");
    let mut builder = MeshIrBuilder::new();
    let zone = builder.zone("fluid", "fluid");
    let wall = builder.tag("wall", "boundary");
    let p0 = builder.point(0.0, 0.0, 0.0).unwrap();
    let p1 = builder.point(1.0, 0.0, 0.0).unwrap();
    let p2 = builder.point(0.0, 1.0, 0.0).unwrap();
    builder.cell("tri3", vec![p0, p1, p2], zone).unwrap();
    builder.tag_edge(vec![p0, p1], wall);
    let mesh = builder.build().unwrap();
    let bytes = write_mesh_ir(
        &mesh,
        &serde_json::json!({"source": "casowasm", "dx": 0.08}),
    )
    .unwrap();
    std::fs::write(&path, bytes).unwrap();
    println!("wrote {path}");
}
