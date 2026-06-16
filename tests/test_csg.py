from __future__ import annotations

import numpy as np

from core.sdf import Box, Cylinder, Difference, SDFTree, Sphere, Union


def test_tree_registers_leaves_and_generates_function() -> None:
    sphere = Sphere(name="sphere", radius=1.0)
    box = Box(name="box", half_size=(0.5, 0.5, 0.5))
    union = Union(name="union", left=sphere, right=box)
    tree = SDFTree(union, components=(union,))
    assert [node.object_id for node in tree.nodes] == [1, 2, 3]
    assert tree.to_glsl().startswith("float sceneSDF(vec3 p)")
    assert "int sceneBoundaryOwnerId(vec3 p)" in tree.to_glsl()
    assert "bool sceneSelectionOwnsBoundary(" in tree.to_glsl()
    assert "float sceneSelectedObjectSDF(" in tree.to_glsl()
    assert "int sceneSelectedObjectDimension(" in tree.to_glsl()
    assert (
        f"if (selected_object_id == {union.object_id})" in tree.to_glsl()
    )
    assert f"boundary_owner_id == {sphere.object_id}" in tree.to_glsl()
    assert f"boundary_owner_id == {box.object_id}" in tree.to_glsl()
    components = tree.components_to_glsl()
    assert "const int COMPONENT_COUNT = 1;" in components
    assert "component == 0" in components
    assert "int componentObjectId(int component)" in components
    assert f"if (component == 0) return {union.object_id};" in components
    assert "component == 1" not in components


def test_difference_removes_center() -> None:
    outer = Sphere(name="outer", radius=2.0)
    inner = Sphere(name="inner", radius=1.0)
    shell = Difference(name="shell", left=outer, right=inner)
    coordinate = np.asarray([0.0], dtype=np.float64)
    assert shell.to_numpy(coordinate, coordinate, coordinate)[0] == 1.0


def test_through_cutter_removes_coincident_front_face() -> None:
    box = Box(name="box", half_size=(1.0, 1.0, 0.5))
    cutter = Cylinder(name="cutter", radius=0.25, half_height=0.6)
    cut = Difference(name="cut", left=box, right=cutter)
    zero = np.asarray([0.0], dtype=np.float64)
    front = np.asarray([0.5], dtype=np.float64)
    assert cut.to_numpy(zero, zero, front)[0] > 0.0


def test_scene_glsl_assigns_surface_colors_by_stable_object_id() -> None:
    box = Box(
        name="box",
        object_id=11,
        half_size=(1.0, 1.0, 1.0),
    )
    sphere = Sphere(name="sphere", object_id=23, radius=0.5)
    tree = SDFTree(
        Difference(name="cut", object_id=31, left=box, right=sphere)
    )
    source = tree.to_glsl()
    assert "int sceneObjectId(vec3 p)" in source
    assert "object_id = 31" in source
    assert "object_id = 11" not in source
    assert "object_id = 23" not in source
