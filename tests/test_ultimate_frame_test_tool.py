from __future__ import annotations

import importlib.util
import logging
import random
import sys
from pathlib import Path


_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "ultimate_frame_test.py"
_SPEC = importlib.util.spec_from_file_location("ultimate_frame_test", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_ultimate = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ultimate
_SPEC.loader.exec_module(_ultimate)


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        "test",
        logging.INFO,
        __file__,
        1,
        message,
        (),
        None,
    )


def test_ultimate_loop_does_not_insert_boolean_operations() -> None:
    runner = object.__new__(_ultimate.UltimateFrameRunner)
    runner.rng = random.Random(1)

    todo = runner._build_loop_todo(0)
    kinds = [kind for _loop, kind in todo]

    assert len(todo) == len(_ultimate.ALL_KINDS)
    assert set(kinds) == set(_ultimate.ALL_KINDS)
    assert all("boolean" not in kind for kind in kinds)


def test_metrics_parser_accepts_surface_artifact_logs() -> None:
    metrics = _ultimate.Metrics()

    metrics.record_log(
        _record(
            "Render artifact built: total=11.2 ms, surface=11.2 ms, "
            "render_wait=12.0 ms, tree_nodes=21, surface_resolution=14, "
            "surface_vertices=128, surface_triangles=64, large_scene=no, "
            "objects=3, exact=3, no_blur=True, reason=none"
        )
    )

    assert metrics.artifact_ms == [11.2]
    assert metrics.surface_ms == [11.2]
    assert metrics.artifact_wait_ms == [12.0]
    assert metrics.events[-1]["surface_ms"] == 11.2
    assert metrics.events[-1]["surface_vertices"] == 128
    assert not metrics.events[-1]["large_scene"]


def test_metrics_parser_accepts_surface_qrhi_init_log() -> None:
    metrics = _ultimate.Metrics()

    metrics.record_log(
        _record(
            "viewport surface qrhi: initialize backend=OpenGL "
            "clip_y_sign=-1 fb_y_up=1 depth_zero_to_one=False"
        )
    )

    assert metrics.backend_info == {
        "backend": "OpenGL",
        "fb_y_up": 1,
        "clip_y_sign": -1,
    }
