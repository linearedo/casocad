from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from core.sdf import (
    BoundingBox3D,
    SDFNode,
    SDFTree,
)

@dataclass(frozen=True)
class ViewportBudgetConfig:
    target_fps: float = 30.0
    telemetry_window: int = 60
    enter_object_count: int = 96
    enter_artifact_ms: float = 33.0
    enter_cull_grid_ms: float = 33.0
    enter_render_call_ms: float = 33.0
    enter_render_wait_ms: float = 33.0
    enter_frame_ms: float = 33.0
    enter_performance_object_count: int = 8
    max_frame_sample_ms: float = 250.0
    exit_object_count: int = 48
    exit_artifact_ms: float = 24.0
    exact_object_budget: int = 96
    degraded_exact_object_budget: int = 8
    severe_exact_object_budget: int = 8
    recent_object_budget: int = 16
    idle_refine_step: int = 16
    idle_refine_interval_ms: int = 120

    def __post_init__(self) -> None:
        if self.target_fps <= 0.0 or not math.isfinite(self.target_fps):
            raise ValueError("target_fps must be finite and positive")
        for name in (
            "exact_object_budget",
            "degraded_exact_object_budget",
            "severe_exact_object_budget",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        degraded_exact = min(
            int(self.exact_object_budget),
            int(self.degraded_exact_object_budget),
        )
        severe_exact = min(degraded_exact, int(self.severe_exact_object_budget))
        object.__setattr__(self, "degraded_exact_object_budget", degraded_exact)
        object.__setattr__(self, "severe_exact_object_budget", severe_exact)

    @property
    def target_frame_ms(self) -> float:
        return 1000.0 / max(self.target_fps, 1.0)


@dataclass(frozen=True)
class ViewportRenderBudget:
    large_scene_mode: bool
    target_frame_ms: float
    exact_object_ids: frozenset[int] = frozenset()
    active_object_ids: frozenset[int] = frozenset()
    reason: str = ""
    no_blur: bool = True


@dataclass(frozen=True)
class GovernorState:
    large_scene_mode: bool
    reason: str
    exact_object_budget: int
    recent_object_ids: tuple[int, ...]
    interacting: bool = False
    idle_refining: bool = False
    no_blur: bool = True


@dataclass
class _RollingTelemetry:
    maxlen: int
    frame_ms: deque[float] = field(init=False)
    artifact_ms: deque[float] = field(init=False)
    cull_grid_ms: deque[float] = field(init=False)
    render_call_ms: deque[float] = field(init=False)
    render_wait_ms: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.frame_ms = deque(maxlen=self.maxlen)
        self.artifact_ms = deque(maxlen=self.maxlen)
        self.cull_grid_ms = deque(maxlen=self.maxlen)
        self.render_call_ms = deque(maxlen=self.maxlen)
        self.render_wait_ms = deque(maxlen=self.maxlen)


def _percentile(values: Iterable[float], p: float) -> float:
    items = sorted(float(value) for value in values)
    if not items:
        return 0.0
    index = min(len(items) - 1, max(0, math.ceil((len(items) - 1) * p)))
    return items[index]


def _object_ids(tree: SDFTree | None) -> tuple[int, ...]:
    if tree is None:
        return ()
    return tuple(
        int(getattr(node, "object_id", 0) or 0)
        for node in tree.components
        if int(getattr(node, "object_id", 0) or 0) > 0
    )


def _node_by_object_id(tree: SDFTree) -> dict[int, SDFNode]:
    return {
        int(node.object_id): node
        for node in tree.components
        if int(getattr(node, "object_id", 0) or 0) > 0
    }


def _bbox(node: SDFNode) -> BoundingBox3D | None:
    try:
        return node.bounding_box()
    except Exception:  # noqa: BLE001 - viewport prioritization is best-effort.
        return None


def _bbox_center_radius(box: BoundingBox3D) -> tuple[np.ndarray, float]:
    center = np.array(
        [
            (box.x_min + box.x_max) * 0.5,
            (box.y_min + box.y_max) * 0.5,
            (box.z_min + box.z_max) * 0.5,
        ],
        dtype=np.float64,
    )
    half = np.array(
        [
            max((box.x_max - box.x_min) * 0.5, 0.01),
            max((box.y_max - box.y_min) * 0.5, 0.01),
            max((box.z_max - box.z_min) * 0.5, 0.01),
        ],
        dtype=np.float64,
    )
    return center, float(np.linalg.norm(half))


class ViewportPerformanceGovernor:
    def __init__(self, config: ViewportBudgetConfig | None = None) -> None:
        self.config = config or ViewportBudgetConfig()
        self._telemetry = _RollingTelemetry(self.config.telemetry_window)
        self._large_scene_mode = False
        self._large_scene_reason = ""
        self._known_object_ids: set[int] = set()
        self._recent_object_ids: deque[int] = deque(
            maxlen=self.config.recent_object_budget
        )
        self._last_object_count = 0
        self._interacting = False
        self._idle_refine_budget = 0

    @property
    def large_scene_mode(self) -> bool:
        return self._large_scene_mode

    def state(self) -> GovernorState:
        return GovernorState(
            large_scene_mode=self._large_scene_mode,
            reason=self._large_scene_reason,
            exact_object_budget=self._current_exact_object_budget(),
            recent_object_ids=tuple(self._recent_object_ids),
            interacting=self._interacting,
            idle_refining=self._idle_refine_budget > 0,
        )

    def begin_interaction(self) -> None:
        self._interacting = True
        self._idle_refine_budget = 0

    def end_interaction(self) -> None:
        self._interacting = False
        self._idle_refine_budget = 0

    def can_refine_idle(self, tree: SDFTree | None) -> bool:
        if self._interacting or tree is None or not self._large_scene_mode:
            return False
        object_count = len(_object_ids(tree))
        if object_count <= 0:
            return False
        if not self._idle_refinement_allowed():
            return False
        return self._current_exact_object_budget() < min(
            object_count,
            max(1, int(self.config.exact_object_budget)),
        )

    def advance_idle_refinement(self, tree: SDFTree | None) -> bool:
        if not self.can_refine_idle(tree):
            return False
        object_count = len(_object_ids(tree))
        maximum = min(object_count, max(1, int(self.config.exact_object_budget)))
        base = self._telemetry_exact_object_budget()
        current = max(base, self._idle_refine_budget)
        next_budget = min(
            maximum,
            current + max(1, int(self.config.idle_refine_step)),
        )
        if next_budget <= current:
            return False
        self._idle_refine_budget = next_budget
        return True

    def record_frame_ms(self, frame_ms: float) -> None:
        sample = float(frame_ms)
        if sample <= 0.0 or sample > self.config.max_frame_sample_ms:
            return
        self._telemetry.frame_ms.append(sample)
        self._update_mode(0)

    def record_render_call_ms(self, render_call_ms: float) -> None:
        self._telemetry.render_call_ms.append(float(render_call_ms))
        self._update_mode(0)

    def record_cull_grid_ms(self, cull_grid_ms: float) -> None:
        self._telemetry.cull_grid_ms.append(float(cull_grid_ms))
        self._update_mode(0)

    def record_render_wait_ms(self, render_wait_ms: float) -> None:
        self._telemetry.render_wait_ms.append(float(render_wait_ms))
        self._update_mode(0)

    def record_artifact_ms(self, artifact_ms: float, object_count: int) -> None:
        self._telemetry.artifact_ms.append(float(artifact_ms))
        self._update_mode(int(object_count))

    def note_tree(self, tree: SDFTree | None) -> None:
        current = set(_object_ids(tree))
        for object_id in sorted(current - self._known_object_ids):
            self._recent_object_ids.appendleft(object_id)
        self._known_object_ids = current
        self._recent_object_ids = deque(
            (object_id for object_id in self._recent_object_ids if object_id in current),
            maxlen=self.config.recent_object_budget,
        )

    def budget_for_tree(
        self,
        tree: SDFTree | None,
        *,
        selected_object_id: int = 0,
        edited_object_ids: Iterable[int] = (),
        hovered_object_id: int = 0,
        camera_position: tuple[float, float, float] | None = None,
    ) -> ViewportRenderBudget:
        self.note_tree(tree)
        object_count = len(_object_ids(tree))
        self._update_mode(object_count)
        if tree is None or not self._large_scene_mode:
            return ViewportRenderBudget(
                large_scene_mode=False,
                target_frame_ms=self.config.target_frame_ms,
                reason=self._large_scene_reason,
            )
        exact_ids = self._choose_exact_ids(
            tree,
            selected_object_id=selected_object_id,
            edited_object_ids=edited_object_ids,
            hovered_object_id=hovered_object_id,
            camera_position=camera_position,
            exact_object_budget=self._current_exact_object_budget(),
        )
        return ViewportRenderBudget(
            large_scene_mode=True,
            target_frame_ms=self.config.target_frame_ms,
            exact_object_ids=frozenset(exact_ids),
            active_object_ids=frozenset(
                object_id
                for object_id in (
                    int(selected_object_id or 0),
                    int(hovered_object_id or 0),
                    *(int(value) for value in edited_object_ids),
                )
                if object_id > 0
            ),
            reason=self._large_scene_reason,
        )

    def _performance_p95(self) -> float:
        return max(
            _percentile(self._telemetry.artifact_ms, 0.95),
            _percentile(self._telemetry.cull_grid_ms, 0.95),
            _percentile(self._telemetry.render_call_ms, 0.95),
            _percentile(self._telemetry.render_wait_ms, 0.95),
            _percentile(self._telemetry.frame_ms, 0.95),
        )

    def _idle_refinement_allowed(self) -> bool:
        return (
            self._last_object_count < self.config.enter_object_count
            and self._performance_p95() <= self.config.target_frame_ms
        )

    def _object_count_exact_budget(self) -> int:
        maximum = max(1, int(self.config.exact_object_budget))
        if self._last_object_count >= self.config.enter_object_count:
            return min(
                maximum,
                max(1, int(self.config.degraded_exact_object_budget)),
            )
        return maximum

    def _telemetry_exact_object_budget(self) -> int:
        maximum = self._object_count_exact_budget()
        degraded = min(
            maximum,
            max(1, int(self.config.degraded_exact_object_budget)),
        )
        severe = min(
            degraded,
            max(1, int(self.config.severe_exact_object_budget)),
        )
        perf_p95 = self._performance_p95()
        if perf_p95 > self.config.target_frame_ms * 2.0:
            return severe
        if perf_p95 > self.config.target_frame_ms:
            return degraded
        return maximum

    def _current_exact_object_budget(self) -> int:
        base = self._telemetry_exact_object_budget()
        if self._interacting:
            return base
        if not self._idle_refinement_allowed():
            self._idle_refine_budget = 0
            return base
        if self._idle_refine_budget > base:
            return min(
                max(1, int(self.config.exact_object_budget)),
                self._idle_refine_budget,
            )
        return base

    def _update_mode(self, object_count: int) -> None:
        if object_count > 0:
            self._last_object_count = int(object_count)
        else:
            object_count = self._last_object_count
        artifact_p95 = _percentile(self._telemetry.artifact_ms, 0.95)
        cull_grid_p95 = _percentile(self._telemetry.cull_grid_ms, 0.95)
        render_call_p95 = _percentile(self._telemetry.render_call_ms, 0.95)
        wait_p95 = _percentile(self._telemetry.render_wait_ms, 0.95)
        frame_p95 = _percentile(self._telemetry.frame_ms, 0.95)
        reasons = []
        if object_count >= self.config.enter_object_count:
            reasons.append(f"objects>={self.config.enter_object_count}")
        perf_reasons_enabled = (
            object_count >= max(1, int(self.config.enter_performance_object_count))
        )
        if perf_reasons_enabled:
            if artifact_p95 > self.config.enter_artifact_ms:
                reasons.append(f"artifact_p95={artifact_p95:.1f}ms")
            if cull_grid_p95 > self.config.enter_cull_grid_ms:
                reasons.append(f"cull_grid_p95={cull_grid_p95:.1f}ms")
            if render_call_p95 > self.config.enter_render_call_ms:
                reasons.append(f"render_call_p95={render_call_p95:.1f}ms")
            if wait_p95 > self.config.enter_render_wait_ms:
                reasons.append(f"render_wait_p95={wait_p95:.1f}ms")
            if frame_p95 > self.config.enter_frame_ms:
                reasons.append(f"frame_p95={frame_p95:.1f}ms")
        if reasons:
            self._large_scene_mode = True
            self._large_scene_reason = ",".join(reasons)
            return
        if (
            self._large_scene_mode
            and object_count <= self.config.exit_object_count
            and artifact_p95 <= self.config.exit_artifact_ms
            and cull_grid_p95 <= self.config.exit_artifact_ms
            and render_call_p95 <= self.config.exit_artifact_ms
            and wait_p95 <= self.config.exit_artifact_ms
            and frame_p95 <= self.config.exit_artifact_ms
        ):
            self._large_scene_mode = False
            self._large_scene_reason = ""

    def _choose_exact_ids(
        self,
        tree: SDFTree,
        *,
        selected_object_id: int,
        edited_object_ids: Iterable[int],
        hovered_object_id: int,
        camera_position: tuple[float, float, float] | None,
        exact_object_budget: int,
    ) -> set[int]:
        nodes = _node_by_object_id(tree)
        active_ids = {
            object_id
            for object_id in (
                int(selected_object_id or 0),
                int(hovered_object_id or 0),
                *(int(value) for value in edited_object_ids),
            )
            if object_id in nodes
        }
        exact_ids: set[int] = set(active_ids)
        for object_id in self._recent_object_ids:
            if len(exact_ids) >= exact_object_budget:
                return exact_ids
            if object_id in nodes:
                exact_ids.add(object_id)
            if len(exact_ids) >= exact_object_budget:
                return exact_ids

        cam = (
            np.asarray(camera_position, dtype=np.float64)
            if camera_position is not None
            else None
        )
        scored: list[tuple[float, float, int]] = []
        for object_id, node in nodes.items():
            if object_id in exact_ids:
                continue
            box = _bbox(node)
            if box is None:
                continue
            center, radius = _bbox_center_radius(box)
            distance = (
                float(np.linalg.norm(center - cam))
                if cam is not None
                else math.inf
            )
            scored.append((distance, -radius, object_id))
        scored.sort()
        for _distance, _neg_radius, object_id in scored:
            if len(exact_ids) >= exact_object_budget:
                break
            exact_ids.add(object_id)
        return exact_ids


__all__ = [
    "GovernorState",
    "ViewportBudgetConfig",
    "ViewportPerformanceGovernor",
    "ViewportRenderBudget",
]
