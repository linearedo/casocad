use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow_array::{Array, BooleanArray};
use caso_kernel::meshing::meshable_domains_from_document;
use caso_kernel::roles::DomainKind;
use caso_kernel::scene::SceneDocument;
use caso_kernel::vec3::vec3;
use caso_meshing::{
    Bounds3, ControlRegion, ControlSet, EntityKind, GenerationLimits, IncrementalLodPreparation,
    JobControl, MemoryArtifact, MemoryStorage, MeshArtifact, MeshChunkBuilder, MeshError, MeshFile,
    MeshId, MeshQuery, MeshQueryService, MeshRenderStyle, MeshRendererCache, MeshTileDetail,
    MeshView, MeshingPhase, MeshingRequest, QueryBudget, QueryMeasures, RenderLineColor,
    RendererBudgets, RowKind, TagFilter, TagScope, TypedFormula, MESH_SCHEMA_NAME,
    MESH_SCHEMA_VERSION,
};

fn rectangle(width: f64, height: f64) -> caso_kernel::meshing::MeshableDomains {
    let mut document = SceneDocument::new();
    let rectangle = document
        .add_primitive_from_drag(
            "rectangle",
            vec3(0.0, 0.0, 0.0),
            vec3(width, height, 0.0),
            1.0,
        )
        .unwrap();
    document.rename(rectangle, "sea").unwrap();
    document
        .set_domain_root(rectangle, DomainKind::Fluid)
        .unwrap();
    meshable_domains_from_document(&document).unwrap()
}

fn layered_rectangle() -> (caso_kernel::meshing::MeshableDomains, String) {
    let mut document = SceneDocument::new();
    let rectangle = document
        .add_primitive_from_drag("rectangle", vec3(0.0, 0.0, 0.0), vec3(2.0, 1.0, 0.0), 1.0)
        .unwrap();
    document.rename(rectangle, "sea").unwrap();
    document
        .set_domain_root(rectangle, DomainKind::Fluid)
        .unwrap();
    document
        .add_boundary_region(rectangle, None, None, Some("wall"))
        .unwrap();
    let region = document.boundary_regions.last().unwrap().name.clone();
    (meshable_domains_from_document(&document).unwrap(), region)
}

fn nested_planar_domains() -> caso_kernel::meshing::MeshableDomains {
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

fn request(algorithm: &str) -> MeshingRequest {
    MeshingRequest {
        domains: rectangle(2.0, 1.0),
        algorithm_id: algorithm.into(),
        controls: control_set(0.25),
        limits: GenerationLimits::default(),
        job_control: JobControl::default(),
    }
}

fn control_set(target_size: f64) -> ControlSet {
    let mut controls = ControlSet::default();
    controls.target_size(target_size).unwrap();
    controls
}

fn memory(output: caso_meshing::MeshingOutput) -> MemoryArtifact {
    match output.artifact {
        MeshArtifact::Memory(bytes) => bytes,
        #[cfg(not(target_arch = "wasm32"))]
        MeshArtifact::Native(path) => panic!("expected memory, got {}", path.display()),
    }
}

#[test]
fn distmesh_v3_is_deterministic_lazy_queryable_and_auditable() {
    let first = memory(
        caso_meshing::run_meshing(
            request("distmesh"),
            MemoryStorage::new(64 * 1024 * 1024).unwrap(),
        )
        .unwrap(),
    );
    let second = memory(
        caso_meshing::run_meshing(
            request("distmesh"),
            MemoryStorage::new(64 * 1024 * 1024).unwrap(),
        )
        .unwrap(),
    );
    assert_eq!(first, second);
    let file = Arc::new(MeshFile::from_memory(first).unwrap());
    assert_eq!(file.manifest().schema_name, MESH_SCHEMA_NAME);
    assert_eq!(file.manifest().schema_version, MESH_SCHEMA_VERSION);
    assert!(file.manifest().counts.points > 0);
    assert!(file.manifest().counts.cells > 0);
    assert!(file.manifest().counts.preview_elements > 0);
    assert!(file.manifest().exact_batches.end <= file.manifest().preview_batches.start);
    assert!(file.manifest().preview_batches.end <= file.manifest().spatial_batches.start);
    assert!(file.manifest().spatial_batches.end <= file.manifest().directory_batches.start);

    let result = MeshQueryService::new(file.clone())
        .execute(MeshQuery {
            entity_kind: EntityKind::Cell,
            display_limit: 3,
            ..MeshQuery::default()
        })
        .unwrap();
    assert!(result.total_matching_count >= 3);
    assert_eq!(result.displayed_count, 3);
    let report = file.full_audit(&JobControl::default()).unwrap();
    assert_eq!(report.entities, file.manifest().counts.entity_count());
}

#[test]
fn audit_steps_match_the_blocking_report_and_generation_reports_finalization() {
    let phases = Arc::new(Mutex::new(Vec::new()));
    let reported = phases.clone();
    let mut generation = request("distmesh");
    generation.job_control = JobControl::default().with_progress(move |progress| {
        let mut phases = reported.lock().unwrap();
        if phases.last() != Some(&progress.phase) {
            phases.push(progress.phase);
        }
    });
    let file = Arc::new(
        MeshFile::from_memory(memory(
            caso_meshing::run_meshing(generation, MemoryStorage::new(64 * 1024 * 1024).unwrap())
                .unwrap(),
        ))
        .unwrap(),
    );
    assert_eq!(
        *phases.lock().unwrap(),
        [
            MeshingPhase::Generating,
            MeshingPhase::BuildingSpatialIndex,
            MeshingPhase::WritingPreviews,
            MeshingPhase::Finalizing,
        ]
    );

    let expected = file.full_audit(&JobControl::default()).unwrap();
    let mut cursor = file.audit_cursor();
    let actual = loop {
        let before = cursor.progress().completed_leaves;
        let step = file
            .audit_step(&mut cursor, 1, &JobControl::default())
            .unwrap();
        assert!(step.progress.completed_leaves <= before + 1);
        if let Some(report) = step.report {
            break report;
        }
    };
    assert_eq!(actual, expected);
}

#[test]
fn decoded_chunk_target_is_enforced() {
    let mut limited = request("distmesh");
    limited.limits.target_chunk_bytes = 1;
    assert!(matches!(
        caso_meshing::run_meshing(
            limited,
            MemoryStorage::new(64 * 1024 * 1024).unwrap()
        ),
        Err(MeshError::LimitExceeded(message)) if message.contains("chunk target")
    ));
}

#[test]
fn planned_cursor_measures_statistics_and_adjacent_tags_share_exact_rows() {
    let file = Arc::new(
        MeshFile::from_memory(memory(
            caso_meshing::run_meshing(
                request("distmesh"),
                MemoryStorage::new(64 * 1024 * 1024).unwrap(),
            )
            .unwrap(),
        ))
        .unwrap(),
    );
    let service = MeshQueryService::new(file.clone());
    let query = MeshQuery {
        measures: QueryMeasures {
            quality: Some(caso_meshing::quality::QualityMetric::ScaledJacobian),
            boundary_distance: true,
            adjacent_boundary_tags: true,
        },
        display_limit: usize::MAX,
        ..MeshQuery::default()
    };
    let plan = service.plan(query.clone()).unwrap();
    assert_eq!(plan.measures, query.measures);
    assert!(plan.candidate_rows > 0);
    assert!(plan.candidate_batches() > 0);

    let mut cursor = service.cursor(plan);
    let mut rows = Vec::new();
    loop {
        let step = cursor
            .step(QueryBudget::new(7, Duration::from_secs(1)))
            .unwrap();
        assert!(step.progress.scanned_rows <= step.progress.candidate_rows);
        rows.extend(step.rows);
        if step.progress.complete {
            break;
        }
    }
    assert!(rows
        .iter()
        .all(|row| { row.quality.is_some() && row.boundary_distance.is_some_and(f64::is_finite) }));
    assert!(rows
        .iter()
        .any(|row| !row.adjacent_boundary_tag_ids.is_empty()));
    let synchronous = service.execute(query.clone()).unwrap();
    assert_eq!(synchronous.total_matching_count, rows.len() as u64);
    assert_eq!(
        synchronous
            .selected_entity_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>(),
        rows.iter().map(|row| row.id).collect()
    );

    let statistics = service.statistics(query.clone()).unwrap();
    assert_eq!(statistics.filtered_cells, rows.len() as u64);
    assert_eq!(statistics.supported, statistics.filtered_cells);
    assert_eq!(statistics.unsupported, 0);
    assert!(statistics.minimum.is_some());
    assert!(statistics.maximum_boundary_distance.is_some());
    assert!(statistics.progress.complete);

    let tag = rows
        .iter()
        .flat_map(|row| row.adjacent_boundary_tag_ids.iter().copied())
        .next()
        .unwrap();
    let tagged = service
        .execute(MeshQuery {
            tag_filter: Some(TagFilter::any([tag], TagScope::AdjacentBoundary)),
            display_limit: usize::MAX,
            ..MeshQuery::default()
        })
        .unwrap();
    assert!(tagged.total_matching_count > 0);
    assert!(tagged.total_matching_count <= rows.len() as u64);

    let mut renderer = MeshRendererCache::new(file.clone(), RendererBudgets::default());
    renderer
        .update_lod_focus(file.manifest().bounds.center())
        .unwrap();
    let boundary_query = MeshQuery {
        entity_kind: EntityKind::Edge,
        tag_filter: Some(TagFilter::any([tag], TagScope::Entity)),
        measures: QueryMeasures {
            quality: Some(caso_meshing::quality::QualityMetric::ScaledJacobian),
            ..Default::default()
        },
        display_limit: usize::MAX,
        ..MeshQuery::default()
    };
    let mut colored_lines = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let update = renderer
            .prepare_lod_incremental_styled(
                boundary_query.clone(),
                MeshRenderStyle::SelectedBoundaryQuality,
                &BTreeSet::new(),
                &BTreeSet::new(),
                1.0,
            )
            .unwrap();
        colored_lines.extend(
            update
                .prepared
                .iter()
                .flat_map(|tile| tile.lines.iter().copied()),
        );
        if update.stats.pending_tiles == 0 {
            break;
        }
        assert!(Instant::now() < deadline, "quality renderer timed out");
        std::thread::yield_now();
    }
    assert!(!colored_lines.is_empty());
    assert!(colored_lines.iter().all(|line| {
        matches!(line.color, RenderLineColor::Quality(Some(score)) if (0.0..=1.0).contains(&score))
    }));

    let formula_plan = service
        .plan(MeshQuery {
            formula: Some(TypedFormula::parse("quality >= 0 and boundary_distance >= 0").unwrap()),
            ..MeshQuery::default()
        })
        .unwrap();
    assert_eq!(
        formula_plan.measures.quality,
        Some(caso_meshing::quality::QualityMetric::ScaledJacobian)
    );
    assert!(formula_plan.measures.boundary_distance);

    let aspect_ratio = service
        .execute(MeshQuery {
            quality: Some(caso_meshing::QualityFilter {
                metric: caso_meshing::quality::QualityMetric::AspectRatio,
                interval: caso_meshing::Interval::new(1.0, f64::INFINITY),
            }),
            formula: Some(TypedFormula::parse("quality >= 1").unwrap()),
            display_limit: usize::MAX,
            ..MeshQuery::default()
        })
        .unwrap();
    let aspect_values = aspect_ratio
        .render_tiles
        .iter()
        .flat_map(|tile| &tile.entities)
        .filter_map(|entity| entity.quality)
        .collect::<Vec<_>>();
    assert!(!aspect_values.is_empty());
    assert!(aspect_values.iter().all(|quality| {
        quality.metric == caso_meshing::quality::QualityMetric::AspectRatio && quality.value >= 1.0
    }));

    let cancelled = cursor.cancellation();
    cancelled.cancel();
    assert!(matches!(
        cursor.step(QueryBudget::default()),
        Err(MeshError::Cancelled)
    ));
}

#[test]
fn registry_and_capability_errors_use_stable_ids() {
    assert_eq!(
        caso_meshing::descriptors()
            .iter()
            .map(|descriptor| descriptor.id)
            .collect::<Vec<_>>(),
        ["distmesh"]
    );
    assert!(!caso_meshing::descriptors()[0].capabilities.refinement);
    assert!(caso_meshing::descriptors()[0].capabilities.boundary_layers);
    let missing = request("not_installed");
    assert!(matches!(
        caso_meshing::run_meshing(
            missing.clone(),
            MemoryStorage::new(8 * 1024 * 1024).unwrap()
        ),
        Err(MeshError::InvalidInput(message)) if message.contains("not compiled in")
    ));
}

#[test]
fn cancellation_and_memory_cap_fail_without_artifacts() {
    let control = JobControl::default();
    control.cancel();
    let cancelled = MeshingRequest {
        job_control: control,
        ..request("distmesh")
    };
    assert!(matches!(
        caso_meshing::run_meshing(cancelled, MemoryStorage::new(8 * 1024 * 1024).unwrap()),
        Err(MeshError::Cancelled)
    ));
    assert!(
        caso_meshing::run_meshing(request("distmesh"), MemoryStorage::new(128).unwrap()).is_err()
    );
}

#[test]
fn chunk_builder_supports_mixed_v3_families_and_enforces_local_points() {
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
        .tri3([points[0], points[1], points[2]], 1, 1)
        .unwrap();
    builder
        .quad4([points[0], points[1], points[2], points[3]], 1, 1)
        .unwrap();
    builder
        .tet4([points[0], points[1], points[2], points[4]], 1, 1)
        .unwrap();
    builder
        .pyramid5(
            [points[0], points[1], points[2], points[3], points[4]],
            1,
            1,
        )
        .unwrap();
    builder
        .prism6(
            [
                points[0], points[1], points[2], points[4], points[5], points[6],
            ],
            1,
            1,
        )
        .unwrap();
    builder.hex8(points, 1, 1).unwrap();
    let chunk = builder.build(3).unwrap();
    assert_eq!(chunk.cells.len(), 6);

    let mut invalid = MeshChunkBuilder::new(8, bounds).unwrap();
    let local = invalid.point([0.0; 3]).unwrap();
    assert!(invalid
        .tri3([local, MeshId::from_raw(1), MeshId::from_raw(2)], 1, 1)
        .is_ok());
    assert!(invalid.build(3).is_err());
}

#[test]
fn distmesh_rejects_refinement_and_generates_2d_boundary_layers() {
    let mut refined = request("distmesh");
    refined
        .controls
        .refinement(
            "sea",
            ControlRegion::sphere(vec3(1.0, 0.5, 0.0), 0.25).unwrap(),
            0.1,
            0.3,
        )
        .unwrap();
    assert!(matches!(
        caso_meshing::run_meshing(refined, MemoryStorage::new(64 * 1024 * 1024).unwrap()),
        Err(MeshError::Capability(message)) if message.contains("does not support refinement")
    ));

    let (domains, region) = layered_rectangle();
    let mut layered = request("distmesh");
    layered.domains = domains;
    layered
        .controls
        .boundary_layer("sea", region, 0.04, 0.2, 1.2, 0.088)
        .unwrap();
    let layered =
        caso_meshing::run_meshing(layered, MemoryStorage::new(64 * 1024 * 1024).unwrap()).unwrap();
    assert!(layered.statistics.cells > 0);
    MeshFile::from_memory(memory(layered))
        .unwrap()
        .full_audit(&JobControl::default())
        .unwrap();
}

#[test]
fn distmesh_rejects_3d_generation_explicitly() {
    let document = SceneDocument::default_scene().unwrap();
    let domains = meshable_domains_from_document(&document).unwrap();
    let domain = domains.iter().next().unwrap();
    let region = domain.boundary_regions.first().unwrap().name.clone();
    let mut controls = ControlSet::default();
    controls.target_size(0.1).unwrap();
    controls
        .boundary_layer(&domain.name, region, 0.01, 0.1, 1.2, 0.022)
        .unwrap();
    let result = caso_meshing::run_meshing(
        MeshingRequest {
            domains,
            algorithm_id: "distmesh".into(),
            controls,
            limits: GenerationLimits::default(),
            job_control: JobControl::default(),
        },
        MemoryStorage::new(64 * 1024 * 1024).unwrap(),
    );
    assert!(matches!(
        result,
        Err(MeshError::UnsupportedDimension { dimension: 3, .. })
    ));
}

#[test]
fn declared_large_rectangle_never_succeeds_with_zero_cells() {
    let output = caso_meshing::run_meshing(
        MeshingRequest {
            domains: rectangle(200.0, 200.0),
            algorithm_id: "distmesh".into(),
            controls: control_set(20.0),
            limits: GenerationLimits::default(),
            job_control: JobControl::default(),
        },
        MemoryStorage::new(64 * 1024 * 1024).unwrap(),
    )
    .unwrap();
    assert!(output.statistics.cells > 0);
}

#[test]
fn uniform_density_increases_as_target_size_decreases() {
    let mut coarse = request("distmesh");
    coarse.controls = control_set(0.4);
    let coarse =
        caso_meshing::run_meshing(coarse, MemoryStorage::new(64 * 1024 * 1024).unwrap()).unwrap();
    let mut fine = request("distmesh");
    fine.controls = control_set(0.2);
    let fine =
        caso_meshing::run_meshing(fine, MemoryStorage::new(64 * 1024 * 1024).unwrap()).unwrap();
    assert!(fine.statistics.cells > coarse.statistics.cells);
}

#[test]
fn many_chunks_keep_spade_and_writer_memory_within_the_target() {
    let mut generation = request("distmesh");
    generation.controls = control_set(0.08);
    generation.limits.target_chunk_bytes = 32 * 1024;
    let output =
        caso_meshing::run_meshing(generation, MemoryStorage::new(64 * 1024 * 1024).unwrap())
            .unwrap();
    assert!(output.statistics.chunks > 1);
    assert!(output.statistics.peak_active_bytes <= 32 * 1024);
}

#[test]
fn directly_nested_planar_domains_generate_auditable_interfaces() {
    let output = caso_meshing::run_meshing(
        MeshingRequest {
            domains: nested_planar_domains(),
            algorithm_id: "distmesh".into(),
            controls: control_set(0.12),
            limits: GenerationLimits::default(),
            job_control: JobControl::default(),
        },
        MemoryStorage::new(64 * 1024 * 1024).unwrap(),
    )
    .unwrap();
    assert_eq!(output.statistics.domains, 2);
    let file = MeshFile::from_memory(memory(output)).unwrap();
    file.full_audit(&JobControl::default()).unwrap();
    let point_batches = file
        .entity_batches(RowKind::Point)
        .map(|entry| entry.batch_index)
        .collect::<Vec<_>>();
    let shared_ghosts = point_batches
        .into_iter()
        .map(|batch| file.batch_view(batch).unwrap())
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
    assert!(
        shared_ghosts > 0,
        "the second interface side must reuse owner point IDs"
    );
}

#[test]
fn distmesh_does_not_generate_3d() {
    let document = SceneDocument::default_scene().unwrap();
    let result = caso_meshing::run_meshing(
        MeshingRequest {
            domains: meshable_domains_from_document(&document).unwrap(),
            algorithm_id: "distmesh".into(),
            controls: control_set(0.1),
            limits: GenerationLimits::default(),
            job_control: JobControl::default(),
        },
        MemoryStorage::new(256 * 1024 * 1024).unwrap(),
    );
    assert!(matches!(
        result,
        Err(MeshError::UnsupportedDimension { dimension: 3, .. })
    ));
}

#[test]
fn lod_uses_internal_previews_when_zoomed_out_and_exact_leaves_when_close() {
    let output = caso_meshing::run_meshing(
        MeshingRequest {
            domains: rectangle(130.0, 1.0),
            algorithm_id: "distmesh".into(),
            controls: control_set(0.1),
            limits: GenerationLimits::default(),
            job_control: JobControl::default(),
        },
        MemoryStorage::new(128 * 1024 * 1024).unwrap(),
    )
    .unwrap();
    let file = Arc::new(MeshFile::from_memory(memory(output)).unwrap());
    file.full_audit(&JobControl::default()).unwrap();
    let bounds = file.manifest().bounds;
    let mut renderer = MeshRendererCache::new(file, RendererBudgets::default());
    let overview_target = renderer
        .update_lod_view(view_for_bounds(bounds, 64.0))
        .unwrap();
    assert!(overview_target
        .tiles
        .iter()
        .all(|key| key.detail == MeshTileDetail::Preview));
    let first_overview = renderer
        .prepare_lod_incremental(
            MeshQuery::default(),
            &Default::default(),
            &Default::default(),
            1.0,
        )
        .unwrap();
    #[cfg(not(target_arch = "wasm32"))]
    {
        assert!(first_overview.prepared.is_empty());
        assert!(first_overview.stats.worker_active);
    }
    let overview = if first_overview.prepared.is_empty() {
        prepare_until_progress(
            &mut renderer,
            &MeshQuery::default(),
            &BTreeSet::new(),
            &BTreeSet::new(),
        )
    } else {
        first_overview
    };
    assert_eq!(overview.prepared.len(), 1);
    assert!(!overview.prepared[0].lines.is_empty());
    assert_eq!(overview.stats.pending_tiles, 0);

    #[cfg(not(target_arch = "wasm32"))]
    {
        // Styling invalidates lines but not decoded entities. The rebuild is
        // still dispatched, returns no synchronous tile, and records no decode.
        let decode_p95 = overview.stats.decode_p95_ms;
        let selected = BTreeSet::from([u64::MAX]);
        let dispatched = renderer
            .prepare_lod_incremental(MeshQuery::default(), &selected, &BTreeSet::new(), 1.0)
            .unwrap();
        assert!(dispatched.prepared.is_empty());
        let rebuilt = prepare_until_progress(
            &mut renderer,
            &MeshQuery::default(),
            &selected,
            &BTreeSet::new(),
        );
        assert_eq!(rebuilt.stats.decode_ms, 0.0);
        assert_eq!(rebuilt.stats.decode_p95_ms, decode_p95);
        let restored = renderer
            .prepare_lod_incremental(
                MeshQuery::default(),
                &BTreeSet::new(),
                &BTreeSet::new(),
                1.0,
            )
            .unwrap();
        assert!(restored.prepared.is_empty());
        let restored = prepare_until_progress(
            &mut renderer,
            &MeshQuery::default(),
            &BTreeSet::new(),
            &BTreeSet::new(),
        );
        assert_eq!(restored.stats.decode_ms, 0.0);
    }

    // A branch that was collapsed below 192 px stays collapsed throughout
    // the 192..256 hysteresis band.
    assert!(renderer
        .update_lod_view(view_for_bounds(bounds, 220.0))
        .is_none());
    let expanded = renderer
        .update_lod_view(view_for_bounds(bounds, 300.0))
        .unwrap();
    assert_ne!(expanded.tiles, overview_target.tiles);
    assert!(renderer
        .update_lod_view(view_for_bounds(bounds, 220.0))
        .is_none());
    assert_eq!(
        renderer
            .update_lod_view(view_for_bounds(bounds, 180.0))
            .unwrap()
            .tiles,
        overview_target.tiles
    );

    let close_view = view_for_bounds(bounds, 100_000.0);
    let close_target = renderer.update_lod_view(close_view).unwrap();
    assert!(close_target
        .tiles
        .iter()
        .all(|key| key.detail == MeshTileDetail::Exact));
    assert!(renderer.update_lod_view(close_view).is_none());
    let mut prepared = 0;
    loop {
        let close = prepare_until_progress(
            &mut renderer,
            &MeshQuery {
                display_limit: 16,
                ..MeshQuery::default()
            },
            &BTreeSet::new(),
            &BTreeSet::new(),
        );
        assert!(close.prepared.len() <= 4);
        prepared += close.prepared.len();
        if close.stats.pending_tiles == 0 {
            break;
        }
    }
    assert_eq!(prepared, close_target.tiles.len());
    assert!(renderer.decoded_bytes() <= RendererBudgets::default().decoded_bytes);
    assert!(renderer.gpu_bytes() <= RendererBudgets::default().gpu_bytes);

    // A rapid reversal cancels the intermediate queue. Only the newest
    // generation is returned, and its root preview is decoded once.
    let stale = renderer
        .update_lod_view(view_for_bounds(bounds, 300.0))
        .unwrap();
    let current = renderer
        .update_lod_view(view_for_bounds(bounds, 64.0))
        .unwrap();
    assert!(current.generation > stale.generation);
    let reversed = prepare_until_progress(
        &mut renderer,
        &MeshQuery {
            display_limit: 16,
            ..MeshQuery::default()
        },
        &BTreeSet::new(),
        &BTreeSet::new(),
    );
    assert!(reversed
        .prepared
        .iter()
        .all(|tile| tile.generation == current.generation && current.tiles.contains(&tile.key)));

    // Exact tiles decoded above survive camera-only target changes.
    let restored = renderer.update_lod_view(close_view).unwrap();
    let cached = renderer
        .prepare_lod_incremental(
            MeshQuery {
                display_limit: 16,
                ..MeshQuery::default()
            },
            &Default::default(),
            &Default::default(),
            1.0,
        )
        .unwrap();
    assert_eq!(cached.stats.decode_ms, 0.0);
    assert_eq!(cached.stats.pending_tiles, 0);
    assert_eq!(cached.prepared.len(), restored.tiles.len());

    let width = bounds.max[0] - bounds.min[0];
    assert!(renderer
        .update_lod_view(view_for_bounds_at(
            bounds,
            100_000.0,
            bounds.min[0] + 0.25 * width,
        ))
        .is_some());
    assert!(renderer
        .update_lod_view(view_for_bounds_at(
            bounds,
            100_000.0,
            bounds.min[0] + 0.75 * width,
        ))
        .is_some());

    let center_y = 0.5 * (bounds.min[1] + bounds.max[1]);
    let left_focus = [bounds.min[0], center_y, 0.0];
    let left = renderer.update_lod_focus(left_focus).unwrap();
    assert_eq!(left.tiles.len(), 4);
    assert!(left
        .tiles
        .iter()
        .all(|tile| tile.detail == MeshTileDetail::Exact));
    assert!(renderer.update_lod_focus(left_focus).is_none());

    let right = renderer
        .update_lod_focus([bounds.max[0], center_y, 0.0])
        .unwrap();
    assert_eq!(right.tiles.len(), 4);
    assert_ne!(right.tiles, left.tiles);
}

fn prepare_until_progress(
    renderer: &mut MeshRendererCache,
    query: &MeshQuery,
    selected_ids: &BTreeSet<u64>,
    highlighted_ids: &BTreeSet<u64>,
) -> IncrementalLodPreparation {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let update = renderer
            .prepare_lod_incremental(query.clone(), selected_ids, highlighted_ids, 1.0)
            .unwrap();
        if !update.prepared.is_empty() || update.stats.pending_tiles == 0 {
            return update;
        }
        assert!(Instant::now() < deadline, "mesh preview worker timed out");
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::yield_now();
    }
}

fn view_for_bounds(bounds: Bounds3, projected_pixels: f64) -> MeshView {
    view_for_bounds_at(
        bounds,
        projected_pixels,
        0.5 * (bounds.min[0] + bounds.max[0]),
    )
}

fn view_for_bounds_at(bounds: Bounds3, projected_pixels: f64, center_x: f64) -> MeshView {
    const VIEWPORT: u32 = 1024;
    let extent = (bounds.max[0] - bounds.min[0])
        .max(bounds.max[1] - bounds.min[1])
        .max(f64::MIN_POSITIVE);
    let scale = 2.0 * projected_pixels / (f64::from(VIEWPORT) * extent);
    let center_y = 0.5 * (bounds.min[1] + bounds.max[1]);
    MeshView::new(
        [
            scale as f32,
            0.0,
            0.0,
            0.0,
            0.0,
            scale as f32,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (-scale * center_x) as f32,
            (-scale * center_y) as f32,
            0.5,
            1.0,
        ],
        VIEWPORT,
        VIEWPORT,
    )
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn native_and_memory_storage_are_byte_identical_and_replace_atomically() {
    let path = std::env::temp_dir().join(format!(
        "casocad-arrow-v3-{}.casomesh.arrow",
        std::process::id()
    ));
    let memory = memory(
        caso_meshing::run_meshing(
            request("distmesh"),
            MemoryStorage::new(64 * 1024 * 1024).unwrap(),
        )
        .unwrap(),
    );
    let native = caso_meshing::run_meshing(
        request("distmesh"),
        caso_meshing::NativeFileStorage::new(&path).unwrap(),
    )
    .unwrap();
    assert!(matches!(native.artifact, MeshArtifact::Native(ref value) if value == &path));
    assert_eq!(std::fs::read(&path).unwrap(), memory.as_ref());
    MeshFile::open_native(&path).unwrap();

    let before = std::fs::read(&path).unwrap();
    let control = JobControl::default();
    control.cancel();
    assert!(caso_meshing::run_meshing(
        MeshingRequest {
            job_control: control,
            ..request("distmesh")
        },
        caso_meshing::NativeFileStorage::new(&path).unwrap(),
    )
    .is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    std::fs::remove_file(path).unwrap();
}

#[test]
#[ignore = "release scale probe: set CASOCAD_SCALE_SIZE for a >=20 GiB artifact"]
#[cfg(not(target_arch = "wasm32"))]
fn native_20_gib_scale_probe() {
    let size = std::env::var("CASOCAD_SCALE_SIZE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .expect("set CASOCAD_SCALE_SIZE explicitly");
    let path = std::env::temp_dir().join("casocad-scale-20gib.casomesh.arrow");
    let output = caso_meshing::run_meshing(
        MeshingRequest {
            domains: rectangle(200.0, 200.0),
            algorithm_id: "distmesh".into(),
            controls: control_set(size),
            limits: GenerationLimits {
                max_cells: u64::MAX,
                max_chunks: u64::MAX,
                ..GenerationLimits::default()
            },
            job_control: JobControl::default(),
        },
        caso_meshing::NativeFileStorage::new(&path).unwrap(),
    )
    .unwrap();
    assert!(output.statistics.cells > 0);
    assert!(std::fs::metadata(path).unwrap().len() >= 20 * 1024 * 1024 * 1024);
}
