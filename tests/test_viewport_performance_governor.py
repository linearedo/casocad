from __future__ import annotations

from time import perf_counter

import pytest

from app.artifacts import RenderSceneSnapshot, build_render_artifact
from app.viewport.performance_governor import (
    ViewportBudgetConfig,
    ViewportPerformanceGovernor,
    ViewportRenderBudget,
)
from core.sdf import (
    Extrude,
    PlacedSDF2D,
    QuadraticBezierSurfaceProfile,
    Revolve,
    SDFNode,
    SDFTree,
    Sphere,
    Union,
)


def _union_all(nodes: list[SDFNode]) -> SDFNode:
    if len(nodes) == 1:
        return nodes[0]
    midpoint = len(nodes) // 2
    return Union(
        name="test_union",
        left=_union_all(nodes[:midpoint]),
        right=_union_all(nodes[midpoint:]),
    )


def _tree(count: int) -> SDFTree:
    components = tuple(
        Sphere(
            name=f"sphere_{idx}",
            object_id=idx,
            center=(float(idx * 3), 0.0, 0.0),
            radius=0.5,
        )
        for idx in range(1, count + 1)
    )
    return SDFTree(root=_union_all(list(components)), components=components)


def _swept_bezier_tree(kind: str) -> SDFTree:
    section = PlacedSDF2D(
        name="bezier_surface_2d",
        object_id=10,
        profile=QuadraticBezierSurfaceProfile(),
    )
    if kind == "extrude":
        solid = Extrude(
            name="swept_bezier",
            object_id=1,
            section=section,
            height=1.0,
        )
    elif kind == "revolve":
        solid = Revolve(
            name="swept_bezier",
            object_id=1,
            section=section,
            axis="u",
            angle_degrees=180.0,
        )
    else:
        raise ValueError(kind)
    return SDFTree(root=solid, components=(solid,))


def test_governor_enters_large_scene_from_object_threshold() -> None:
    governor = ViewportPerformanceGovernor(
        ViewportBudgetConfig(
            enter_object_count=4,
            exit_object_count=1,
            exact_object_budget=2,
        )
    )

    budget = governor.budget_for_tree(_tree(4))

    assert budget.large_scene_mode
    assert budget.no_blur
    assert "objects>=4" in budget.reason
    assert len(budget.exact_object_ids) == 2


def test_budget_config_rejects_invalid_perf_contract() -> None:
    with pytest.raises(ValueError, match="target_fps"):
        ViewportBudgetConfig(target_fps=0.0)
    with pytest.raises(ValueError, match="severe_exact_object_budget"):
        ViewportBudgetConfig(severe_exact_object_budget=0)


def test_budget_config_normalizes_partial_budget_overrides() -> None:
    config = ViewportBudgetConfig(
        exact_object_budget=8,
        degraded_exact_object_budget=16,
        severe_exact_object_budget=10,
    )

    assert config.degraded_exact_object_budget == 8
    assert config.severe_exact_object_budget == 8


def test_governor_enters_large_scene_from_rolling_frame_budget() -> None:
    governor = ViewportPerformanceGovernor(
        ViewportBudgetConfig(
            enter_object_count=99,
            enter_performance_object_count=1,
            exact_object_budget=2,
        )
    )
    tree = _tree(3)

    assert not governor.budget_for_tree(tree).large_scene_mode
    governor.record_frame_ms(40.0)
    budget = governor.budget_for_tree(tree)

    assert budget.large_scene_mode
    assert "frame_p95=40.0ms" in budget.reason


def test_governor_ignores_idle_frame_gaps() -> None:
    governor = ViewportPerformanceGovernor(
        ViewportBudgetConfig(
            enter_object_count=99,
            max_frame_sample_ms=250.0,
        )
    )

    governor.record_frame_ms(1200.0)
    budget = governor.budget_for_tree(_tree(1))

    assert not budget.large_scene_mode
    assert "frame_p95" not in budget.reason


def test_governor_does_not_enter_large_scene_for_one_slow_object() -> None:
    governor = ViewportPerformanceGovernor(
        ViewportBudgetConfig(
            enter_object_count=99,
            enter_performance_object_count=8,
        )
    )

    governor.record_frame_ms(153.2)
    budget = governor.budget_for_tree(_tree(1))

    assert not budget.large_scene_mode
    assert "frame_p95" not in budget.reason


@pytest.mark.parametrize("kind", ["extrude", "revolve"])
def test_swept_quadratic_bezier_surface_stays_on_surface_viewport_path(
    kind: str,
) -> None:
    tree = _swept_bezier_tree(kind)
    governor = ViewportPerformanceGovernor()

    budget = governor.budget_for_tree(tree, selected_object_id=1)

    assert not budget.large_scene_mode
    assert budget.exact_object_ids == frozenset()
    assert "complex_proxy=swept_quadratic_bezier_surface" not in budget.reason


def test_over_budget_telemetry_reduces_exact_object_budget() -> None:
    governor = ViewportPerformanceGovernor(
        ViewportBudgetConfig(
            enter_object_count=99,
            enter_performance_object_count=1,
            exact_object_budget=5,
            degraded_exact_object_budget=2,
            severe_exact_object_budget=1,
            recent_object_budget=5,
        )
    )
    tree = _tree(5)

    governor.record_artifact_ms(40.0, object_count=5)
    budget = governor.budget_for_tree(tree)

    assert budget.large_scene_mode
    assert len(budget.exact_object_ids) == 2
    assert governor.state().exact_object_budget == 2


def test_recovered_telemetry_progressively_restores_exact_budget() -> None:
    governor = ViewportPerformanceGovernor(
        ViewportBudgetConfig(
            telemetry_window=2,
            enter_object_count=99,
            enter_performance_object_count=1,
            exit_object_count=0,
            exact_object_budget=5,
            degraded_exact_object_budget=2,
            severe_exact_object_budget=1,
            recent_object_budget=5,
        )
    )
    tree = _tree(5)

    governor.record_artifact_ms(80.0, object_count=5)
    assert len(governor.budget_for_tree(tree).exact_object_ids) == 1
    governor.record_artifact_ms(10.0, object_count=5)
    assert len(governor.budget_for_tree(tree).exact_object_ids) == 1
    governor.record_artifact_ms(10.0, object_count=5)
    assert len(governor.budget_for_tree(tree).exact_object_ids) == 5


def test_recovered_idle_telemetry_restores_exact_budget_until_interaction_resumes() -> None:
    governor = ViewportPerformanceGovernor(
        ViewportBudgetConfig(
            telemetry_window=2,
            enter_object_count=99,
            enter_performance_object_count=1,
            exit_object_count=0,
            exact_object_budget=6,
            degraded_exact_object_budget=2,
            severe_exact_object_budget=1,
            recent_object_budget=6,
            idle_refine_step=2,
        )
    )
    tree = _tree(6)

    governor.record_artifact_ms(80.0, object_count=6)
    governor.end_interaction()
    assert len(governor.budget_for_tree(tree).exact_object_ids) == 1
    assert not governor.can_refine_idle(tree)
    governor.record_artifact_ms(10.0, object_count=6)
    governor.record_artifact_ms(10.0, object_count=6)
    assert len(governor.budget_for_tree(tree).exact_object_ids) == 6
    assert not governor.advance_idle_refinement(tree)
    governor.begin_interaction()
    assert not governor.can_refine_idle(tree)
    assert len(governor.budget_for_tree(tree).exact_object_ids) == 6


def test_idle_refinement_stays_degraded_while_perf_p95_is_over_budget() -> None:
    governor = ViewportPerformanceGovernor(
        ViewportBudgetConfig(
            enter_object_count=1,
            exact_object_budget=96,
            degraded_exact_object_budget=8,
            severe_exact_object_budget=8,
            recent_object_budget=8,
            idle_refine_step=16,
        )
    )
    tree = _tree(1000)

    governor.record_artifact_ms(71.1, object_count=1000)
    governor.record_render_wait_ms(71.8)
    governor.record_frame_ms(120.1)
    budget = governor.budget_for_tree(tree)

    assert budget.large_scene_mode
    assert len(budget.exact_object_ids) == 8
    assert not governor.can_refine_idle(tree)
    assert not governor.advance_idle_refinement(tree)
    assert len(governor.budget_for_tree(tree).exact_object_ids) == 8


def test_object_threshold_uses_degraded_budget_before_bad_frames() -> None:
    tree = _tree(1000)
    governor = ViewportPerformanceGovernor(
        ViewportBudgetConfig(
            enter_object_count=96,
            exact_object_budget=96,
            degraded_exact_object_budget=8,
            recent_object_budget=8,
        )
    )

    budget = governor.budget_for_tree(tree)

    assert budget.large_scene_mode
    assert "objects>=96" in budget.reason
    assert len(budget.exact_object_ids) == 8


def test_priority_keeps_active_objects_before_recent_and_budget_fill() -> None:
    governor = ViewportPerformanceGovernor(
        ViewportBudgetConfig(
            enter_object_count=1,
            exact_object_budget=2,
            recent_object_budget=2,
        )
    )
    tree = _tree(5)

    budget = governor.budget_for_tree(
        tree,
        selected_object_id=2,
        hovered_object_id=3,
        camera_position=(0.0, 0.0, 8.0),
    )

    assert budget.large_scene_mode
    assert budget.active_object_ids == frozenset({2, 3})
    assert budget.exact_object_ids == frozenset({2, 3})


def test_artifact_timings_default_to_surface_viewport_contract() -> None:
    tree = _tree(3)
    budget = ViewportRenderBudget(
        large_scene_mode=True,
        target_frame_ms=33.0,
        exact_object_ids=frozenset({1, 2}),
        reason="artifact_p95=40.0ms",
    )

    artifact = build_render_artifact(
        RenderSceneSnapshot(
            version=7,
            tree=tree,
            budget=budget,
            requested_at=perf_counter() - 0.01,
        )
    )

    assert artifact.version == 7
    assert artifact.surface_scene is not None
    assert artifact.surface_scene.has_geometry
    assert artifact.timings.large_scene_mode
    assert artifact.timings.total_object_count == 3
    assert artifact.timings.exact_object_count == 3
    assert artifact.timings.no_blur
    assert artifact.timings.render_wait_ms >= artifact.timings.total_ms


def test_viewport_artifact_builds_swept_surface_scene() -> None:
    tree = _swept_bezier_tree("extrude")
    budget = ViewportRenderBudget(
        large_scene_mode=True,
        target_frame_ms=33.0,
        reason="artifact_p95=40.0ms",
    )

    artifact = build_render_artifact(
        RenderSceneSnapshot(
            version=8,
            tree=tree,
            budget=budget,
        )
    )

    assert artifact.surface_scene is not None
    assert artifact.surface_scene.has_geometry
    assert artifact.timings.surface_ms >= 0.0
    assert artifact.timings.large_scene_mode
    assert artifact.timings.exact_object_count == 1
