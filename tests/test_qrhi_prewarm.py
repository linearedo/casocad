from __future__ import annotations

from app.viewport.renderers.qrhi import renderer as renderer_module
from app.viewport.renderers.qrhi.renderer import (
    QRhiInterpreterRenderer,
    _prewarm_render_ir,
)
from core.gpu_codegen import (
    group_capacity,
    has_carves,
    profiles_are_simple,
    scene_structure_signature,
    selector_indices,
    supported,
    uses_spatial_cull,
    viewport_leaf_signature,
)
from core.render_ir import RenderIR, RenderIRNode


def _box_render_ir() -> RenderIR:
    return RenderIR(
        nodes=(
            RenderIRNode(
                kind="box",
                object_id=1,
                dimension=3,
                children=(),
                params=(0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.5, 0.5, 0.5),
            ),
        ),
        root_indices=(0,),
        component_indices=(),
    )


def _render_sig(render_ir: RenderIR) -> tuple[object, int | None, bool, bool, bool, bool]:
    return (
        viewport_leaf_signature(render_ir),
        group_capacity(render_ir),
        profiles_are_simple(render_ir),
        uses_spatial_cull(render_ir),
        bool(selector_indices(render_ir)),
        has_carves(render_ir),
    )


def test_prewarm_render_ir_adds_quadratic_bezier_surface_signature() -> None:
    render_ir = _box_render_ir()

    prewarm = _prewarm_render_ir(render_ir, "quadratic_bezier_surface")

    assert prewarm is not None
    assert supported(prewarm)
    assert scene_structure_signature(prewarm) == {
        "box",
        "placed_quadratic_bezier_surface_2d",
    }


def test_prewarm_render_ir_adds_quadratic_bezier_polycurve_signature() -> None:
    render_ir = _box_render_ir()

    prewarm = _prewarm_render_ir(render_ir, "quadratic_bezier_polycurve")

    assert prewarm is not None
    assert supported(prewarm)
    assert scene_structure_signature(prewarm) == {
        "box",
        "placed_quadratic_bezier_polycurve_1d",
    }


def test_prewarm_render_ir_ignores_unknown_tools() -> None:
    assert _prewarm_render_ir(None, "quadratic_bezier_tube") is None


def test_tool_prewarm_can_skip_gui_thread_pipeline_compile(monkeypatch) -> None:
    render_ir = _box_render_ir()
    prewarm = _prewarm_render_ir(render_ir, "quadratic_bezier_surface")
    assert prewarm is not None
    sig = _render_sig(prewarm)
    shader = object()
    calls = []
    renderer = QRhiInterpreterRenderer()
    renderer._baked = True
    renderer._rpd = object()
    renderer._cg_srb = object()
    renderer._cg_frag_cache[sig] = shader

    monkeypatch.setattr(
        renderer,
        "_prewarm_pipeline",
        lambda prewarm_sig, frag: calls.append((prewarm_sig, frag)),
    )

    renderer.prewarm_for_tool(render_ir, "quadratic_bezier_surface", compile_pipeline=False)
    assert calls == []

    renderer.prewarm_for_tool(render_ir, "quadratic_bezier_surface", compile_pipeline=True)
    assert calls == [(sig, shader)]


def test_tool_pipeline_prewarm_policy_is_backend_agnostic() -> None:
    class FakeRhi:
        def __init__(self, backend: str) -> None:
            self._backend = backend

        def backendName(self) -> str:
            return self._backend

    renderer = QRhiInterpreterRenderer()
    assert not renderer.should_prewarm_tool_pipeline()

    for backend in ("OpenGL", "Vulkan", "Metal", "D3D11"):
        renderer._rhi = FakeRhi(backend)
        assert renderer.should_prewarm_tool_pipeline()


def test_shader_cached_pipeline_cold_scene_finalizes_deferred(monkeypatch) -> None:
    render_ir = _box_render_ir()
    sig = _render_sig(render_ir)
    shader = object()
    renderer = QRhiInterpreterRenderer()
    renderer._baked = True
    renderer._rpd = object()
    renderer._cg_frag_cache[sig] = shader
    activated = []
    callbacks = []
    updates = []

    monkeypatch.setattr(
        renderer_module.QTimer,
        "singleShot",
        lambda _delay_ms, callback: callbacks.append(callback),
    )
    monkeypatch.setattr(
        renderer,
        "_activate_codegen_scene",
        lambda active_render_ir: activated.append(active_render_ir),
    )
    renderer.set_update_callback(lambda: updates.append(True))

    renderer.set_scene(render_ir)

    assert activated == []
    assert callbacks
    assert renderer._cg_deferred_sig == sig

    callbacks[0]()

    assert activated == [render_ir]
    assert updates == [True]
    assert renderer._cg_deferred_sig is None


def test_codegen_signature_tracks_selector_shader_variant() -> None:
    geom = RenderIRNode(
        kind="sphere",
        object_id=1,
        dimension=3,
        children=(),
        params=(0.0, 0.0, 0.0, 1.0),
    )
    selector_volume = RenderIRNode(
        kind="sphere",
        object_id=2,
        dimension=3,
        children=(),
        params=(0.0, 0.0, 0.0, 0.5),
    )
    selector = RenderIRNode(
        kind="region_selector",
        object_id=1,
        dimension=3,
        children=(1,),
        params=(0.1,),
        flags=7,
    )
    plain = RenderIR(nodes=(geom,), root_indices=(0,), component_indices=())
    selected = RenderIR(
        nodes=(geom, selector_volume, selector),
        root_indices=(0,),
        component_indices=(2,),
    )
    renderer = QRhiInterpreterRenderer()

    assert renderer._codegen_sig(plain) != renderer._codegen_sig(selected)
    assert renderer._codegen_sig(plain)[4] is False
    assert renderer._codegen_sig(selected)[4] is True


def test_codegen_signature_does_not_track_direct_leaf_kind_growth() -> None:
    box = _box_render_ir()
    with_curve = _prewarm_render_ir(box, "quadratic_bezier_polycurve")
    with_surface = _prewarm_render_ir(box, "quadratic_bezier_surface")
    assert with_curve is not None
    assert with_surface is not None
    renderer = QRhiInterpreterRenderer()

    assert scene_structure_signature(with_curve) != scene_structure_signature(with_surface)
    assert renderer._codegen_sig(with_curve) == renderer._codegen_sig(with_surface)
