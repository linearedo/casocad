from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "qrhi_compile_log_summary.py"
_SPEC = importlib.util.spec_from_file_location("qrhi_compile_log_summary", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
format_summary = _MODULE.format_summary
parse_events = _MODULE.parse_events
summarize_by_signature = _MODULE.summarize_by_signature


def test_qrhi_compile_log_summary_parses_bake_and_pipeline_events() -> None:
    text = """
2026-06-26 01:31:11,888 INFO app.viewport.renderers.qrhi.renderer: qrhi: prewarm bake start tool=quadratic_bezier_surface pipeline=no kinds=['box', 'placed_quadratic_bezier_surface_2d'] cap=1 profile_mode=simple cull_mode=flat selector_mode=no_selectors carve_mode=no_carves source_bytes=9887
2026-06-26 01:31:12,308 INFO app.viewport.renderers.qrhi.renderer: qrhi: async bake done reason=prewarm kinds=['box', 'placed_quadratic_bezier_surface_2d'] cap=1 profile_mode=simple cull_mode=flat selector_mode=no_selectors carve_mode=no_carves qsb=417.4 ms source_bytes=9887
2026-06-26 01:31:16,356 INFO app.viewport.renderers.qrhi.renderer: qrhi: pipeline driver-compiled kinds=['box', 'placed_quadratic_bezier_surface_2d'] cap=1 profile_mode=simple cull_mode=flat selector_mode=no_selectors carve_mode=no_carves backend=Vulkan source_bytes=9887 in 2.01s
"""

    events = parse_events(text)
    summary = format_summary(events)

    assert len(events) == 3
    assert events[1].qsb_ms == 417.4
    assert events[2].pipeline_s == 2.01
    assert "pipeline driver-compiled" in summary
    assert "source_bytes" in summary
    assert "9887" in summary
    assert "signature summary" in summary


def test_qrhi_compile_log_summary_groups_by_signature() -> None:
    text = """
2026-06-26 01:31:11,888 INFO app.viewport.renderers.qrhi.renderer: qrhi: async bake done reason=prewarm kinds=['box', 'placed_quadratic_bezier_surface_2d'] cap=1 profile_mode=simple cull_mode=flat selector_mode=no_selectors carve_mode=no_carves qsb=300.0 ms source_bytes=7042
2026-06-26 01:31:12,888 INFO app.viewport.renderers.qrhi.renderer: qrhi: async bake done reason=prewarm kinds=['box', 'placed_quadratic_bezier_surface_2d'] cap=1 profile_mode=simple cull_mode=flat selector_mode=no_selectors carve_mode=no_carves qsb=420.0 ms source_bytes=7042
2026-06-26 01:31:16,356 INFO app.viewport.renderers.qrhi.renderer: qrhi: pipeline driver-compiled kinds=['box', 'placed_quadratic_bezier_surface_2d'] cap=1 profile_mode=simple cull_mode=flat selector_mode=no_selectors carve_mode=no_carves backend=Vulkan source_bytes=7042 in 1.75s
2026-06-26 01:31:18,356 INFO app.viewport.renderers.qrhi.renderer: qrhi: pipeline driver-compiled kinds=['box'] cap=1 profile_mode=simple cull_mode=flat selector_mode=no_selectors carve_mode=no_carves backend=OpenGL source_bytes=4000 in 0.10s
"""

    summaries = summarize_by_signature(parse_events(text))
    summary_text = format_summary(parse_events(text))

    assert summaries[0].source_bytes == 7042
    assert summaries[0].max_qsb_ms == 420.0
    assert summaries[0].max_pipeline_s == 1.75
    assert summaries[0].backends == ("Vulkan",)
    assert "max_pipeline_s" in summary_text
    assert "1.75" in summary_text
    assert "420.0" in summary_text
