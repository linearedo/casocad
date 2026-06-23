from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from app.viewport.renderers.interpreter_glsl.shader_assembly import (
    build_program_source,
)
from app.viewport.renderers.qrhi.vulkanize import (
    IMAGE_BINDING,
    UBO_BINDING,
    uniform_block_members,
    vulkanize,
)

_SHADERS = Path(
    "app/viewport/renderers/interpreter_glsl/shaders"
)


def _interpreter_source(features=frozenset()) -> str:
    comp = (_SHADERS / "raymarch_interpreter.comp").read_text()
    return build_program_source(features, comp)


# --- the transform ----------------------------------------------------------


def test_loose_uniforms_collapse_into_one_std140_block() -> None:
    gl = _interpreter_source()
    vk = vulkanize(gl)
    # No loose uniforms remain (except the opaque image2D, which is not "loose").
    assert vk.count("uniform _Globals") == 1
    assert f"binding = {UBO_BINDING}) uniform _Globals" in vk
    # Every collected member appears inside the block.
    for _type, name, _arr in uniform_block_members(gl):
        assert name in vk


def test_output_image_moved_off_the_ssbo_range() -> None:
    vk = vulkanize(_interpreter_source())
    assert f"binding = {IMAGE_BINDING}" in vk
    # The SSBOs keep their original unique bindings (0..3 for the core).
    assert "binding = 0) readonly buffer Nodes" in vk


def test_version_bumped_to_vulkan_baseline() -> None:
    vk = vulkanize(_interpreter_source(), version="#version 450")
    assert vk.splitlines()[0] == "#version 450"


def test_raises_without_loose_uniforms() -> None:
    with pytest.raises(ValueError):
        vulkanize("#version 460\nvoid main() {}\n")


# --- qsb bake (the real proof; skipped if the baker is unavailable) ----------

_QSB = os.path.join(os.path.dirname(sys.executable), "pyside6-qsb")


@pytest.mark.skipif(
    not (os.path.exists(_QSB) or shutil.which("pyside6-qsb")),
    reason="pyside6-qsb not available",
)
@pytest.mark.parametrize("features", [frozenset(), None], ids=["core", "full"])
def test_vulkanized_interpreter_bakes_through_qsb(features) -> None:
    if features is None:
        from core.gpu_features import FULL_FEATURES

        features = FULL_FEATURES
    vk = vulkanize(_interpreter_source(features))
    qsb = _QSB if os.path.exists(_QSB) else shutil.which("pyside6-qsb")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "rm.comp"
        out = Path(tmp) / "rm.comp.qsb"
        src.write_text(vk)
        result = subprocess.run(
            [qsb, "--glsl", "430", "-o", str(out), str(src)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists() and out.stat().st_size > 0
