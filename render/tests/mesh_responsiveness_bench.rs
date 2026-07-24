#![cfg(not(target_arch = "wasm32"))]

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use caso_kernel::meshing::meshable_domains_from_document;
use caso_kernel::roles::DomainKind;
use caso_kernel::scene::SceneDocument;
use caso_kernel::vec3::vec3;
use caso_meshing::{
    ControlSet, GenerationLimits, JobControl, MemoryStorage, MeshArtifact, MeshFile, MeshQuery,
    MeshRendererCache, MeshingRequest, RendererBudgets,
};
use caso_render::{ViewportRenderer, MESH_TILE_UPLOAD_BUDGET_BYTES};

fn large_mesh() -> Arc<MeshFile> {
    let mut document = SceneDocument::new();
    let rectangle = document
        .add_primitive_from_drag("rectangle", vec3(0.0, 0.0, 0.0), vec3(130.0, 1.0, 0.0), 1.0)
        .expect("rectangle");
    document
        .set_domain_root(rectangle, DomainKind::Fluid)
        .expect("domain");
    let output = caso_meshing::run_meshing(
        MeshingRequest {
            domains: meshable_domains_from_document(&document).expect("meshable"),
            algorithm_id: "uniform_2d".into(),
            element_min_size: 0.1,
            element_max_size: 0.1,
            controls: ControlSet::default(),
            limits: GenerationLimits::default(),
            job_control: JobControl::default(),
        },
        MemoryStorage::new(128 * 1024 * 1024).expect("storage"),
    )
    .expect("large mesh");
    let MeshArtifact::Memory(bytes) = output.artifact else {
        panic!("expected memory mesh");
    };
    Arc::new(MeshFile::from_memory(bytes).expect("mesh file"))
}

#[test]
#[ignore = "manual settle-to-activation benchmark; run with --ignored --nocapture"]
fn settle_to_activation_breakdown() {
    let file = large_mesh();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("headless adapter");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("headless device");
    let mut samples = (0..7)
        .map(|_| run_once(file.clone(), &device, &queue))
        .collect::<Vec<_>>();
    samples.sort_by(|a, b| a.total_ms.total_cmp(&b.total_ms));
    let median = samples[samples.len() / 2];

    println!(
        "settle_to_activation_ms={:.2} decode_ms={:.2} \
         line_build_ms={:.2} upload_ms={:.2} \
         frames={} tiles={} upload_budget_mib={} samples=7",
        median.total_ms,
        median.decode_ms,
        median.line_build_ms,
        median.upload_ms,
        median.frames,
        median.tiles,
        MESH_TILE_UPLOAD_BUDGET_BYTES / (1024 * 1024),
    );
}

#[derive(Clone, Copy)]
struct Sample {
    total_ms: f64,
    decode_ms: f32,
    line_build_ms: f32,
    upload_ms: f32,
    frames: usize,
    tiles: usize,
}

fn run_once(file: Arc<MeshFile>, device: &wgpu::Device, queue: &wgpu::Queue) -> Sample {
    let mut preparation = MeshRendererCache::new(file.clone(), RendererBudgets::default());
    let bounds = file.manifest().bounds;
    let target = preparation
        .update_lod_focus(bounds.center())
        .expect("LOD target");
    let target_keys = target.tiles.into_iter().collect::<BTreeSet<_>>();
    assert!(!target_keys.is_empty());
    let mut renderer = ViewportRenderer::new(device);

    let started = Instant::now();
    let deadline = started + Duration::from_secs(30);
    let mut sample = Sample {
        total_ms: 0.0,
        decode_ms: 0.0,
        line_build_ms: 0.0,
        upload_ms: 0.0,
        frames: 0,
        tiles: target_keys.len(),
    };
    loop {
        sample.frames += 1;
        let update = preparation
            .prepare_lod_incremental(
                MeshQuery::default(),
                &BTreeSet::new(),
                &BTreeSet::new(),
                1.0,
            )
            .expect("prepare");
        sample.decode_ms += update.stats.decode_ms;
        sample.line_build_ms += update.stats.line_build_ms;
        renderer.set_mesh_tile_target(
            update.selection.generation,
            update.selection.tiles.iter().copied(),
        );
        for tile in update.prepared {
            renderer.upsert_mesh_tile(tile.generation, tile.key, tile.lines);
        }
        renderer.upload_pending_mesh_tiles(device, queue, MESH_TILE_UPLOAD_BUDGET_BYTES);
        sample.upload_ms += renderer.mesh_tile_stats().upload_ms;
        if renderer.active_mesh_tiles() == &target_keys {
            sample.total_ms = started.elapsed().as_secs_f64() * 1_000.0;
            return sample;
        }
        assert!(Instant::now() < deadline, "preview activation timed out");
        std::thread::sleep(Duration::from_millis(16));
    }
}
