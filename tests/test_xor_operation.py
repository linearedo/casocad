from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.scene import SceneDocument
from core.serialization import load_scene, save_scene
from core.sdf import (
    BinaryProfile,
    BinaryProfile1D,
    Box,
    CircleProfile,
    Difference,
    SegmentProfile,
    Sphere,
    Xor,
)
from core.sdf.roles import Exactness, exactness_violations, node_exactness


def test_xor_operator_uses_exact_sdf_formula() -> None:
    left = Sphere(name="left", center=(-0.45, 0.0, 0.0), radius=0.6)
    right = Sphere(name="right", center=(0.45, 0.0, 0.0), radius=0.6)
    node = Xor(name="xor", left=left, right=right)

    x = np.asarray([-0.45, 0.0, 2.0], dtype=np.float64)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    field = node.to_numpy(x, y, z)

    left_field = left.to_numpy(x, y, z)
    right_field = right.to_numpy(x, y, z)
    expected = np.maximum(
        np.minimum(left_field, right_field),
        -np.maximum(left_field, right_field),
    )
    np.testing.assert_allclose(field, expected)
    assert field[0] < 0.0
    assert field[1] > 0.0
    assert field[2] > 0.0


def test_xor_profiles_are_supported_for_1d_and_2d() -> None:
    first = SegmentProfile(center=-0.25, half_length=0.5)
    second = SegmentProfile(center=0.25, half_length=0.5)
    profile_1d = BinaryProfile1D(first, second, operation="xor")
    t = np.asarray([-0.6, 0.0, 0.6], dtype=np.float64)

    assert profile_1d.to_numpy(t)[0] < 0.0
    assert profile_1d.to_numpy(t)[1] > 0.0
    assert profile_1d.to_numpy(t)[2] < 0.0

    first_2d = CircleProfile(center=(-0.25, 0.0), radius=0.5)
    second_2d = CircleProfile(center=(0.25, 0.0), radius=0.5)
    profile_2d = BinaryProfile(
        left=first_2d,
        right=second_2d,
        operation="xor",
    )
    u = np.asarray([-0.6, 0.0, 0.6], dtype=np.float64)
    v = np.zeros_like(u)

    assert profile_2d.to_numpy(u, v)[0] < 0.0
    assert profile_2d.to_numpy(u, v)[1] > 0.0
    assert profile_2d.to_numpy(u, v)[2] < 0.0


def test_scene_combine_xor_saves_and_loads(tmp_path: Path) -> None:
    document = SceneDocument()
    first = document.add_primitive("sphere")
    second = document.add_primitive("box")

    result = document.combine(first, second, "xor")

    assert isinstance(document.node(result), Xor)
    path = tmp_path / "xor.json"
    save_scene(document, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["objects"]["xor_1"]["type"] == "xor"

    loaded = load_scene(path)
    assert isinstance(loaded.objects[0], Xor)


def test_xor_is_not_in_exact_compiler_grammar() -> None:
    node = Xor(
        name="xor",
        left=Box(name="box", half_size=(1.0, 1.0, 1.0)),
        right=Sphere(name="sphere", radius=0.3),
    )
    assert node_exactness(node) is Exactness.NONE
    violations = exactness_violations(node)
    assert any("XOR 'xor' is not accepted for solver-ready Domains yet" in v for v in violations)
    assert any("free SDF modeling" in v for v in violations)

    region_only = Difference(
        name="region",
        left=Box(name="outer", half_size=(1.0, 1.0, 1.0)),
        right=Sphere(name="hole", radius=0.3),
    )
    invalid = Xor(name="invalid", left=region_only, right=Sphere(name="s"))
    assert exactness_violations(invalid)
