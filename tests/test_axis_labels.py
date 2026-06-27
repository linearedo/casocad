from __future__ import annotations

from app.axis_labels import world_axis_label


def test_world_axis_label_names_cardinal_axes() -> None:
    assert world_axis_label((1.0, 0.0, 0.0)) == "X axis"
    assert world_axis_label((0.0, -2.0, 0.0)) == "Y axis"
    assert world_axis_label((0.0, 0.0, 0.5)) == "Z axis"


def test_world_axis_label_keeps_oblique_vectors_explicit() -> None:
    assert world_axis_label((1.0, 1.0, 0.0)).startswith("vector")
