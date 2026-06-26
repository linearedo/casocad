from __future__ import annotations

from core.scene import SceneDocument
from core.sdf import Box


def test_default_names_use_per_kind_counter_not_object_id() -> None:
    document = SceneDocument([Box(name="existing", object_id=5)])

    first = document.node(document.add_primitive("sphere"))
    second = document.node(document.add_primitive("sphere"))

    assert first.name == "sphere_1"
    assert second.name == "sphere_2"
    assert first.object_id == 6
    assert second.object_id == 7


def test_boolean_default_name_is_operation_counter() -> None:
    document = SceneDocument()
    box_handle = document.add_primitive("box")
    sphere_handle = document.add_primitive("sphere")

    result = document.node(document.combine(box_handle, sphere_handle, "difference"))

    assert result.name == "difference_1"
