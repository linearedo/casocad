//! Headless drag benchmark: measures where time goes during a simulated
//! Move/Rotate gizmo drag (document edit vs node build vs surface build at
//! the coarse tier). Ignored by default; run manually with:
//!
//! ```bash
//! cargo test -p caso-app --test drag_bench -- --ignored --nocapture
//! ```

use std::time::Instant;

use caso_kernel::scene::SceneDocument;
use caso_kernel::sdf::node::RotationAxis;
use caso_kernel::vec3::{vec3, Vec3};
use caso_surfaces::ViewportSurfaceCache;

const FRAMES: usize = 60;
const COARSE_TIER: u32 = 12;

fn box_scene() -> (SceneDocument, u32) {
    let mut document = SceneDocument::new();
    let id = document.add_primitive("box", 1.0).expect("box");
    (document, id)
}

fn demo_scene() -> (SceneDocument, u32) {
    let document = SceneDocument::default_scene().expect("scene");
    let root = *document.primary_roots().first().expect("root");
    (document, root)
}

fn boolean_scene() -> (SceneDocument, u32) {
    let mut document = SceneDocument::new();
    let a = document.add_primitive("sphere", 1.0).expect("sphere");
    let b = document.add_primitive("sphere", 1.0).expect("sphere");
    document
        .move_object(b, vec3(0.4, 0.0, 0.0))
        .expect("offset");
    let id = document.combine(a, b, "union").expect("union");
    (document, id)
}

fn polygon_scene() -> (SceneDocument, u32) {
    let mut document = SceneDocument::new();
    let points: Vec<Vec3> = (0..200)
        .map(|i| {
            let a = i as f64 / 200.0 * std::f64::consts::TAU;
            vec3(
                a.cos() * (2.0 + 0.3 * (7.0 * a).sin()),
                a.sin() * (2.0 + 0.3 * (5.0 * a).cos()),
                0.0,
            )
        })
        .collect();
    let id = document
        .add_point_shape_from_world_points("polygon", &points, "xy")
        .expect("polygon");
    (document, id)
}

/// One simulated drag: returns (edit_ms, node_ms, surface_ms) totals.
fn run_drag(
    document: &mut SceneDocument,
    mut target: u32,
    rotate: bool,
) -> (f64, f64, f64) {
    let mut cache = ViewportSurfaceCache::default();
    cache.resolution = COARSE_TIER;
    let (mut edit_ms, mut node_ms, mut surface_ms) = (0.0, 0.0, 0.0);
    for frame in 0..FRAMES {
        let start = Instant::now();
        if rotate {
            document
                .rotate_object(target, RotationAxis::Z, 0.5, Some(vec3(0.0, 0.0, 0.0)))
                .expect("rotate");
        } else {
            target = document
                .move_object(target, vec3(0.01, 0.0, 0.0))
                .expect("move");
        }
        edit_ms += start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        let mut components = Vec::new();
        for root in document.primary_roots() {
            components.push(document.build_node(root).expect("node"));
        }
        node_ms += start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        let scene = caso_surfaces::build_viewport_surface_scene(
            &components,
            document.version,
            &mut cache,
        );
        surface_ms += start.elapsed().as_secs_f64() * 1000.0;
        assert!(!scene.surfaces.is_empty(), "frame {frame} built no surfaces");
    }
    (edit_ms, node_ms, surface_ms)
}

#[test]
#[ignore = "manual benchmark; run with --ignored --nocapture"]
fn drag_frame_breakdown() {
    println!(
        "\n{:<22} {:>6}  {:>10} {:>10} {:>12} {:>12}",
        "scene", "op", "edit ms/f", "node ms/f", "surface ms/f", "total ms/f"
    );
    type SceneBuilder = fn() -> (SceneDocument, u32);
    let scenes: [(&str, SceneBuilder); 4] = [
        ("box", box_scene),
        ("default demo (bool)", demo_scene),
        ("union of 2 spheres", boolean_scene),
        ("200-pt polygon", polygon_scene),
    ];
    for (name, build) in scenes {
        for rotate in [false, true] {
            let (mut document, target) = build();
            let (edit, node, surface) = run_drag(&mut document, target, rotate);
            let n = FRAMES as f64;
            println!(
                "{:<22} {:>6}  {:>10.3} {:>10.3} {:>12.3} {:>12.3}",
                name,
                if rotate { "rotate" } else { "move" },
                edit / n,
                node / n,
                surface / n,
                (edit + node + surface) / n
            );
        }
    }

    // Signature-string cost for the heavy polygon node.
    let (document, id) = polygon_scene();
    let node = document.build_node(id).expect("node");
    let start = Instant::now();
    let mut total = 0usize;
    for _ in 0..1000 {
        total += format!("{node:?}").len();
    }
    println!(
        "\nsignature format!() of 200-pt polygon: {:.4} ms/call ({} chars)",
        start.elapsed().as_secs_f64(),
        total / 1000
    );
}
