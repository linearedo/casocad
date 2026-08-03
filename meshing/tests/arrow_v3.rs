use std::sync::Arc;

use arrow_array::{Array, BooleanArray};
use caso_kernel::boundary_ops::{surface_patches_for_root, CurvePatchKind};
use caso_kernel::meshing::{meshable_domains_from_document, MeshableDomains};
use caso_kernel::roles::DomainKind;
use caso_kernel::scene::SceneDocument;
use caso_kernel::vec3::vec3;
use caso_meshing::quality::{quality_score, QualityMetric};
use caso_meshing::{
    Bounds3, ControlRegion, ControlSet, EntityKind, GenerationLimits, JobControl, MemoryArtifact,
    MemoryStorage, MeshArtifact, MeshChunkBuilder, MeshError, MeshFile, MeshId, MeshQuery,
    MeshQueryService, MeshingRequest, RowKind, MESH_SCHEMA_NAME, MESH_SCHEMA_VERSION,
};

fn rectangle(width: f64, height: f64) -> MeshableDomains {
    let mut document = SceneDocument::new();
    let root = document
        .add_primitive_from_drag(
            "rectangle",
            vec3(0.0, 0.0, 0.0),
            vec3(width, height, 0.0),
            1.0,
        )
        .unwrap();
    document.rename(root, "sea").unwrap();
    document.set_domain_root(root, DomainKind::Fluid).unwrap();
    meshable_domains_from_document(&document).unwrap()
}

fn rectangle_with_curved_hole() -> (MeshableDomains, String) {
    let mut document = SceneDocument::new();
    let outer = document
        .add_primitive_from_drag(
            "rectangle",
            vec3(-1.0, -0.75, 0.0),
            vec3(1.0, 0.75, 0.0),
            1.0,
        )
        .unwrap();
    let hole = document
        .add_primitive_from_drag("circle", vec3(-0.3, -0.3, 0.0), vec3(0.3, 0.3, 0.0), 1.0)
        .unwrap();
    let root = document.combine(outer, hole, "difference").unwrap();
    document.rename(root, "sea").unwrap();
    document.set_domain_root(root, DomainKind::Fluid).unwrap();
    let node = document.build_node(root).unwrap();
    let patches = surface_patches_for_root(&node);
    let curved = patches
        .iter()
        .find(|patch| {
            patch.patch_id.starts_with("cut_surface.")
                && matches!(patch.curve, Some(CurvePatchKind::Outline))
        })
        .unwrap();
    document
        .add_boundary_region(
            curved.owner_object_id,
            curved.outside_direction,
            Some(&curved.patch_id),
            Some(&curved.patch_type),
        )
        .unwrap();
    let region = document.boundary_regions.last().unwrap().name.clone();
    (meshable_domains_from_document(&document).unwrap(), region)
}

fn nested_planar_domains() -> MeshableDomains {
    let mut document = SceneDocument::new();
    let outer = document
        .add_primitive_from_drag("rectangle", vec3(-1.5, -1.0, 0.0), vec3(1.5, 1.0, 0.0), 1.0)
        .unwrap();
    let inner = document
        .add_primitive_from_drag(
            "circle",
            vec3(-0.45, -0.45, 0.0),
            vec3(0.45, 0.45, 0.0),
            1.0,
        )
        .unwrap();
    document.rename(inner, "solid").unwrap();
    document.set_domain_root(inner, DomainKind::Solid).unwrap();
    let fluid = document.combine(outer, inner, "difference").unwrap();
    document.rename(fluid, "fluid").unwrap();
    document.set_domain_root(fluid, DomainKind::Fluid).unwrap();
    meshable_domains_from_document(&document).unwrap()
}

fn controls(target_size: f64) -> ControlSet {
    let mut controls = ControlSet::default();
    controls.target_size(target_size).unwrap();
    controls
}

fn request(domains: MeshableDomains, target_size: f64) -> MeshingRequest {
    MeshingRequest {
        domains,
        algorithm_id: "distmesh".into(),
        controls: controls(target_size),
        limits: GenerationLimits::default(),
        job_control: JobControl::default(),
    }
}

fn memory(artifact: MeshArtifact) -> MemoryArtifact {
    match artifact {
        MeshArtifact::Memory(bytes) => bytes,
        #[cfg(not(target_arch = "wasm32"))]
        MeshArtifact::Native(path) => panic!("expected memory artifact, got {}", path.display()),
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn artifacts_are_deterministic_auditable_queryable_and_storage_neutral() {
    let first = memory(
        caso_meshing::run_meshing(
            request(rectangle(2.0, 1.0), 0.3),
            MemoryStorage::new(32 * 1024 * 1024).unwrap(),
        )
        .unwrap()
        .artifact,
    );
    let second = memory(
        caso_meshing::run_meshing(
            request(rectangle(2.0, 1.0), 0.3),
            MemoryStorage::new(32 * 1024 * 1024).unwrap(),
        )
        .unwrap()
        .artifact,
    );
    assert_eq!(first, second);

    let path = std::env::temp_dir().join(format!(
        "casocad-public-contract-{}-{}.arrow",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let native = caso_meshing::run_meshing(
        request(rectangle(2.0, 1.0), 0.3),
        caso_meshing::NativeFileStorage::new(&path).unwrap(),
    )
    .unwrap();
    assert!(matches!(native.artifact, MeshArtifact::Native(ref value) if value == &path));
    assert_eq!(std::fs::read(&path).unwrap(), first.as_ref());

    let file = Arc::new(MeshFile::from_memory(first).unwrap());
    assert_eq!(file.manifest().schema_name, MESH_SCHEMA_NAME);
    assert_eq!(file.manifest().schema_version, MESH_SCHEMA_VERSION);
    assert!(file.manifest().counts.points > 0);
    assert!(file.manifest().counts.cells >= 3);
    let query = MeshQueryService::new(file.clone())
        .execute(MeshQuery {
            entity_kind: EntityKind::Cell,
            display_limit: 3,
            ..MeshQuery::default()
        })
        .unwrap();
    assert_eq!(query.displayed_count, 3);
    assert_eq!(
        file.full_audit(&JobControl::default()).unwrap().entities,
        file.manifest().counts.entity_count()
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn quality_scales_rendering_and_worst_selection_agree() {
    let triangle = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 3.0_f64.sqrt() / 2.0, 0.0],
    ];
    let square = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    assert!(
        (quality_score("tri3", &triangle, QualityMetric::ScaledJacobian).unwrap() - 1.0).abs()
            < 1.0e-12
    );
    assert!((quality_score("quad4", &square, QualityMetric::Skewness).unwrap()).abs() < 1.0e-12);
    assert!(
        (quality_score("quad4", &square, QualityMetric::AspectRatio).unwrap() - 1.0).abs()
            < 1.0e-12
    );

    for (metric, ideal, poor) in [
        (QualityMetric::ScaledJacobian, 1.0, -0.5),
        (QualityMetric::Skewness, 0.0, 1.0),
        (QualityMetric::AspectRatio, 1.0, 10.0),
        (QualityMetric::Compactness, 1.0, 0.0),
        (QualityMetric::Orthogonality, 1.0, 0.0),
    ] {
        assert_eq!(metric.rendering_goodness(ideal), 1.0);
        assert!(metric.rendering_goodness(poor) < 0.11);
    }
    assert_eq!(QualityMetric::ScaledJacobian.worst(0.8, 0.3), 0.3);
    assert_eq!(QualityMetric::Skewness.worst(0.2, 0.7), 0.7);
    assert_eq!(QualityMetric::AspectRatio.worst(1.2, 3.0), 3.0);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn cancellation_and_limits_never_publish_artifacts() {
    let path = std::env::temp_dir().join(format!(
        "casocad-failed-publication-{}.arrow",
        std::process::id()
    ));
    std::fs::write(&path, b"previous artifact").unwrap();

    let cancelled = JobControl::default();
    cancelled.cancel();
    let mut cancelled_request = request(rectangle(2.0, 1.0), 0.25);
    cancelled_request.job_control = cancelled;
    assert!(matches!(
        caso_meshing::run_meshing(
            cancelled_request,
            caso_meshing::NativeFileStorage::new(&path).unwrap()
        ),
        Err(MeshError::Cancelled)
    ));
    assert_eq!(std::fs::read(&path).unwrap(), b"previous artifact");

    assert!(caso_meshing::run_meshing(
        request(rectangle(2.0, 1.0), 0.25),
        MemoryStorage::new(128).unwrap()
    )
    .is_err());

    let mut cell_limited = request(rectangle(2.0, 1.0), 0.25);
    cell_limited.limits.max_cells = 1;
    assert!(matches!(
        caso_meshing::run_meshing(
            cell_limited,
            caso_meshing::NativeFileStorage::new(&path).unwrap()
        ),
        Err(MeshError::LimitExceeded(_))
    ));
    assert_eq!(std::fs::read(&path).unwrap(), b"previous artifact");

    let mut chunk_limited = request(rectangle(2.0, 1.0), 0.25);
    chunk_limited.limits.target_chunk_bytes = 1;
    assert!(matches!(
        caso_meshing::run_meshing(
            chunk_limited,
            caso_meshing::NativeFileStorage::new(&path).unwrap()
        ),
        Err(MeshError::LimitExceeded(message)) if message.contains("chunk target")
    ));
    assert_eq!(std::fs::read(&path).unwrap(), b"previous artifact");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn registry_capabilities_and_dimension_neutral_storage_are_explicit() {
    let descriptor = &caso_meshing::descriptors()[0];
    assert_eq!(descriptor.id, "distmesh");
    assert_eq!(descriptor.dimensions, &[2]);
    assert!(!descriptor.capabilities.refinement);
    assert!(descriptor.capabilities.boundary_layers);

    let mut missing = request(rectangle(1.0, 1.0), 0.25);
    missing.algorithm_id = "not_installed".into();
    assert!(matches!(
        caso_meshing::run_meshing(missing, MemoryStorage::new(8 * 1024 * 1024).unwrap()),
        Err(MeshError::InvalidInput(message)) if message.contains("not compiled in")
    ));

    let mut refined = request(rectangle(1.0, 1.0), 0.25);
    refined
        .controls
        .refinement(
            "sea",
            ControlRegion::sphere(vec3(0.5, 0.5, 0.0), 0.2).unwrap(),
            0.1,
            0.3,
        )
        .unwrap();
    assert!(matches!(
        caso_meshing::run_meshing(refined, MemoryStorage::new(8 * 1024 * 1024).unwrap()),
        Err(MeshError::Capability(message)) if message.contains("does not support refinement")
    ));

    let document = SceneDocument::default_scene().unwrap();
    assert!(matches!(
        caso_meshing::run_meshing(
            request(meshable_domains_from_document(&document).unwrap(), 0.2),
            MemoryStorage::new(8 * 1024 * 1024).unwrap()
        ),
        Err(MeshError::UnsupportedDimension { dimension: 3, .. })
    ));

    let bounds = Bounds3 {
        min: [0.0; 3],
        max: [1.0; 3],
    };
    let mut builder = MeshChunkBuilder::new(7, bounds).unwrap();
    let points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    .map(|point| builder.point(point).unwrap());
    builder
        .tet4([points[0], points[1], points[2], points[4]], 1, 1)
        .unwrap();
    builder.hex8(points, 1, 1).unwrap();
    assert_eq!(builder.build(3).unwrap().cells.len(), 2);

    let mut invalid = MeshChunkBuilder::new(8, bounds).unwrap();
    let local = invalid.point([0.0; 3]).unwrap();
    invalid
        .tri3([local, MeshId::from_raw(1), MeshId::from_raw(2)], 1, 1)
        .unwrap();
    assert!(invalid.build(3).is_err());
}

#[test]
fn smaller_target_size_produces_a_denser_mesh() {
    let coarse = caso_meshing::run_meshing(
        request(rectangle(2.0, 1.0), 0.4),
        MemoryStorage::new(32 * 1024 * 1024).unwrap(),
    )
    .unwrap();
    let fine = caso_meshing::run_meshing(
        request(rectangle(2.0, 1.0), 0.2),
        MemoryStorage::new(32 * 1024 * 1024).unwrap(),
    )
    .unwrap();
    assert!(fine.statistics.cells > coarse.statistics.cells);
}

#[test]
fn straight_and_curved_boundaries_produce_valid_tagged_quad_layers() {
    let (domains, region) = rectangle_with_curved_hole();
    let mut generation = request(domains, 0.25);
    generation
        .controls
        .boundary_layer("sea", region, 0.04, 0.2, 1.2, 0.089)
        .unwrap();
    let output =
        caso_meshing::run_meshing(generation, MemoryStorage::new(64 * 1024 * 1024).unwrap())
            .unwrap();
    let file = Arc::new(MeshFile::from_memory(memory(output.artifact)).unwrap());
    file.full_audit(&JobControl::default()).unwrap();
    let service = MeshQueryService::new(file);
    let cells = service
        .execute(MeshQuery {
            entity_kind: EntityKind::Cell,
            display_limit: usize::MAX,
            ..MeshQuery::default()
        })
        .unwrap();
    assert!(cells
        .render_tiles
        .iter()
        .flat_map(|tile| &tile.entities)
        .any(|entity| entity.element_type == "quad4"));
    let boundary = service
        .execute(MeshQuery {
            entity_kind: EntityKind::Edge,
            display_limit: usize::MAX,
            ..MeshQuery::default()
        })
        .unwrap();
    assert!(boundary
        .render_tiles
        .iter()
        .flat_map(|tile| &tile.entities)
        .any(|entity| !entity.tag_ids.is_empty()));
}

#[test]
fn nested_domains_reuse_interface_points_and_pass_the_full_audit() {
    let output = caso_meshing::run_meshing(
        request(nested_planar_domains(), 0.2),
        MemoryStorage::new(64 * 1024 * 1024).unwrap(),
    )
    .unwrap();
    assert_eq!(output.statistics.domains, 2);
    let file = MeshFile::from_memory(memory(output.artifact)).unwrap();
    file.full_audit(&JobControl::default()).unwrap();
    let shared_ghosts = file
        .entity_batches(RowKind::Point)
        .map(|entry| file.batch_view(entry.batch_index).unwrap())
        .map(|view| {
            let ghosts = view
                .record_batch()
                .column_by_name("ghost")
                .unwrap()
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap();
            (0..ghosts.len()).filter(|row| ghosts.value(*row)).count()
        })
        .sum::<usize>();
    assert!(shared_ghosts > 0);
}
