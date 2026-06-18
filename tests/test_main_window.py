from __future__ import annotations

import numpy as np

from app.main_window import (
    CLEAR_SELECTION_SHORTCUT,
    DEFAULT_BACKGROUND_HEX,
    DUPLICATE_SHORTCUT,
    FRAME_SHORTCUTS,
    FRAME_VIEW_KEY,
    MainWindow,
    REDO_SHORTCUTS,
    RENAME_SHORTCUT,
    SELECT_ALL_SHORTCUT,
    SNAP_TOGGLE_SHORTCUT,
    color_to_rgb_tuple,
    rgb_tuple_to_hex,
    scene_item_handles,
    selected_sdf_bounding_box,
    viewport_shape_created_message,
)
from app.panels.properties import CadDimensionSpinBox, property_dimension_value
from core.scene import SceneDocument
from PySide6.QtGui import QColor


def test_empty_document_publish_sends_compilable_scene_source() -> None:
    class FakeDocument:
        objects: list[object] = []

    class FakeViewport:
        def __init__(self) -> None:
            self.scene_source = ""
            self.mode = ""
            self.default_grid_configured = False
            self.default_grid_framed = False

        def set_scene_artifact(self, tree: object, scene_source: str) -> None:
            assert tree is None
            self.scene_source = scene_source

        def set_mode(self, mode: str) -> None:
            self.mode = mode

        def configure_default_grid(self) -> None:
            self.default_grid_configured = True

        def frame_default_grid(self) -> None:
            self.default_grid_framed = True

    class FakeAction:
        def __init__(self) -> None:
            self.checked = False

        def setChecked(self, checked: bool) -> None:
            self.checked = checked

    class FakeWindow:
        def __init__(self) -> None:
            self.document = FakeDocument()
            self.viewport = FakeViewport()
            self._sdf_action = FakeAction()
            self.grid_synced = False

        def _sync_grid_spacing_control(self) -> None:
            self.grid_synced = True

    window = FakeWindow()

    MainWindow._publish_document(window)

    assert "float sceneSDF(vec3 p)" in window.viewport.scene_source
    assert "const int COMPONENT_COUNT = 0;" in window.viewport.scene_source
    assert window._sdf_action.checked
    assert window.viewport.mode == "sdf"
    assert window.viewport.default_grid_configured
    assert window.viewport.default_grid_framed
    assert window.grid_synced


def test_grid_spacing_control_uses_cad_dimension_parser() -> None:
    assert CadDimensionSpinBox.__name__ == "CadDimensionSpinBox"
    assert property_dimension_value("1m/4") == 0.25
    assert property_dimension_value("50mm*2") == 0.1


def test_viewport_background_color_helpers_use_default_background() -> None:
    color = QColor(DEFAULT_BACKGROUND_HEX)

    assert rgb_tuple_to_hex(color_to_rgb_tuple(color)) == DEFAULT_BACKGROUND_HEX
    assert rgb_tuple_to_hex((2.0, -1.0, 0.5)) == "#ff0080"


def test_viewport_shape_created_message_explains_sticky_draw_tool() -> None:
    assert viewport_shape_created_message("box") == (
        "Created box. Draw tool remains active; press Esc to finish."
    )


def test_boundary_region_viewport_entries_follow_rotated_owner_axes() -> None:
    document = SceneDocument.default()
    assert document.fluid_domain is not None
    root_handle = document.handle_for(document.fluid_domain.root)
    document.rotate_object(root_handle, "z", 45.0, (0.0, 0.0, 0.0))
    inlet = next(
        tag
        for tag in document.boundary_regions
        if tag.name == "inlet"
    )
    window = type("FakeWindow", (), {"document": document})()
    window._viewport_outside_direction_normal = (
        MainWindow._viewport_outside_direction_normal.__get__(
            window,
            type(window),
        )
    )

    selectors, normals = MainWindow._viewport_boundary_region_entries(
        window,
        [inlet],
    )

    assert selectors == ((inlet.owner_object_id, 0),)
    np.testing.assert_allclose(
        normals[0],
        (-np.sqrt(0.5), -np.sqrt(0.5), 0.0),
        atol=1e-12,
    )


def test_snap_toggle_shortcut_is_single_key_for_viewport_precision() -> None:
    assert SNAP_TOGGLE_SHORTCUT == "G"


def test_redo_shortcuts_support_common_cad_and_desktop_conventions() -> None:
    assert REDO_SHORTCUTS == ("Ctrl+Y", "Ctrl+Shift+Z")


def test_select_all_shortcut_matches_desktop_selection_convention() -> None:
    assert SELECT_ALL_SHORTCUT == "Ctrl+A"


def test_duplicate_shortcut_matches_desktop_duplicate_convention() -> None:
    assert DUPLICATE_SHORTCUT == "Ctrl+D"


def test_rename_shortcut_matches_scene_item_rename_convention() -> None:
    assert RENAME_SHORTCUT == "F2"


def test_frame_shortcuts_match_cad_view_conventions() -> None:
    assert FRAME_SHORTCUTS == ("Home",)
    assert FRAME_VIEW_KEY == "F"


def test_clear_selection_shortcut_matches_escape_convention() -> None:
    assert CLEAR_SELECTION_SHORTCUT == "Esc"


def test_scene_item_handles_follow_document_walk_order() -> None:
    document = SceneDocument.default()

    assert scene_item_handles(document) == [
        handle for handle, _node, _parent in document.walk()
    ]


def test_selected_sdf_bounding_box_unions_selected_scene_items() -> None:
    document = SceneDocument()
    sphere = document.add_primitive("sphere")
    box = document.add_primitive("box")

    bounds = selected_sdf_bounding_box(document, [sphere, box])

    assert bounds is not None
    assert bounds.x_min == -0.5
    assert bounds.x_max == 0.75
    assert bounds.y_min == -0.5
    assert bounds.y_max == 0.5


def test_frame_scene_falls_back_to_reference_grid_without_selection_or_domain() -> None:
    class FakeSceneTree:
        def selected_handles(self) -> list[int]:
            return []

    class FakeDocument:
        fluid_domain = None

    class FakeViewport:
        def __init__(self) -> None:
            self.framed_default_grid = False

        def frame_default_grid(self) -> None:
            self.framed_default_grid = True

    class FakeStatusBar:
        def __init__(self) -> None:
            self.message = ""

        def showMessage(self, message: str, _timeout: int) -> None:
            self.message = message

    class FakeWindow:
        def __init__(self) -> None:
            self.scene_tree = FakeSceneTree()
            self.document = FakeDocument()
            self.viewport = FakeViewport()
            self.status_bar = FakeStatusBar()

        def statusBar(self) -> FakeStatusBar:
            return self.status_bar

    window = FakeWindow()

    MainWindow._frame_scene(window)

    assert window.viewport.framed_default_grid
    assert window.status_bar.message == "Framed reference grid"


def test_clear_selection_clears_scene_tree_selection() -> None:
    class FakeSceneTree:
        def __init__(self) -> None:
            self.selected = [1, 2]
            self.selected_after_clear = None

        def selected_handles(self) -> list[int]:
            return self.selected

        def select_handles(self, handles: list[int]) -> None:
            self.selected_after_clear = handles
            self.selected = handles

    class FakeStatusBar:
        def __init__(self) -> None:
            self.message = ""

        def showMessage(self, message: str, _timeout: int) -> None:
            self.message = message

    class FakeWindow:
        def __init__(self) -> None:
            self.scene_tree = FakeSceneTree()
            self.status_bar = FakeStatusBar()

        def statusBar(self) -> FakeStatusBar:
            return self.status_bar

    window = FakeWindow()

    MainWindow._clear_selection(window)

    assert window.scene_tree.selected_after_clear == []
    assert window.status_bar.message == "Selection cleared"


def test_clear_selection_shortcut_cancels_active_viewport_tool_first() -> None:
    class FakeViewport:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel_active_interaction_tool(self) -> bool:
            self.cancelled = True
            return True

    class FakeSceneTree:
        def __init__(self) -> None:
            self.selected_after_clear = None

        def select_handles(self, handles: list[int]) -> None:
            self.selected_after_clear = handles

    class FakeStatusBar:
        def __init__(self) -> None:
            self.message = ""

        def showMessage(self, message: str, _timeout: int) -> None:
            self.message = message

    class FakeWindow:
        def __init__(self) -> None:
            self.viewport = FakeViewport()
            self.scene_tree = FakeSceneTree()
            self.status_bar = FakeStatusBar()

        def statusBar(self) -> FakeStatusBar:
            return self.status_bar

    window = FakeWindow()

    MainWindow._clear_selection(window)

    assert window.viewport.cancelled
    assert window.scene_tree.selected_after_clear is None
    assert window.status_bar.message == "Viewport tool cancelled"


def test_delete_shortcut_cancels_active_viewport_tool_first() -> None:
    class FakeViewport:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel_active_interaction_tool(self) -> bool:
            self.cancelled = True
            return True

    class FakeSceneTree:
        def selected_handles(self) -> list[int]:
            return [1]

    class FakeStatusBar:
        def __init__(self) -> None:
            self.message = ""

        def showMessage(self, message: str, _timeout: int) -> None:
            self.message = message

    class FakeWindow:
        def __init__(self) -> None:
            self.viewport = FakeViewport()
            self.scene_tree = FakeSceneTree()
            self.status_bar = FakeStatusBar()
            self.deleted = None

        def _on_delete_nodes(self, handles: list[int]) -> None:
            self.deleted = handles

        def statusBar(self) -> FakeStatusBar:
            return self.status_bar

    window = FakeWindow()

    MainWindow._delete_selection(window)

    assert window.viewport.cancelled
    assert window.deleted is None
    assert window.status_bar.message == "Viewport tool cancelled"


def test_duplicate_selection_copies_and_pastes_selected_nodes() -> None:
    class FakeSceneTree:
        def __init__(self) -> None:
            self.selected = [11]
            self.selected_after_paste = []

        def selected_handles(self) -> list[int]:
            return self.selected

        def select_handles(self, handles: list[int]) -> None:
            self.selected_after_paste = handles

    class FakeDocument:
        def __init__(self) -> None:
            self.copied_handles = []
            self.pasted_nodes = []
            self.paste_offset = None

        def copy_nodes(self, handles: list[int]) -> list[object]:
            self.copied_handles = handles
            return ["node"]

        def paste_nodes(
            self,
            nodes: list[object],
            offset: tuple[float, float, float],
        ) -> list[int]:
            self.pasted_nodes = nodes
            self.paste_offset = offset
            return [42]

    class FakeViewport:
        def paste_offset(self) -> tuple[float, float, float]:
            return (0.1, 0.2, 0.0)

    class FakeStatusBar:
        def __init__(self) -> None:
            self.message = ""

        def showMessage(self, message: str, _timeout: int) -> None:
            self.message = message

    class FakeWindow:
        def __init__(self) -> None:
            self.scene_tree = FakeSceneTree()
            self.document = FakeDocument()
            self.viewport = FakeViewport()
            self.status_bar = FakeStatusBar()
            self.undo_recorded = False
            self.published = False

        def _history_snapshot(self) -> object:
            return object()

        def _record_undo_snapshot(self, _snapshot: object) -> None:
            self.undo_recorded = True

        def _publish_document(self, clear_selection: bool = True) -> None:
            self.published = not clear_selection

        def _paste_nodes(self, nodes: list[object], action_name: str) -> list[int]:
            return MainWindow._paste_nodes(self, nodes, action_name)

        def statusBar(self) -> FakeStatusBar:
            return self.status_bar

    window = FakeWindow()

    MainWindow._duplicate_selection(window)

    assert window.document.copied_handles == [11]
    assert window.document.pasted_nodes == ["node"]
    assert window.document.paste_offset == (0.1, 0.2, 0.0)
    assert window.scene_tree.selected_after_paste == [42]
    assert window.undo_recorded
    assert window.published
    assert window.status_bar.message == "Duplicated 1 SDF object"


def test_viewport_combined_transform_moves_then_rotates_once() -> None:
    class FakeDocument:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def move_object(
            self,
            handle: int,
            delta: tuple[float, float, float],
        ) -> int:
            self.calls.append(("move", handle, delta))
            return 43

        def rotate_object(
            self,
            handle: int,
            axis: str,
            angle: float,
            pivot: tuple[float, float, float],
        ) -> int:
            self.calls.append(("rotate", handle, axis, angle, pivot))
            return 44

        def refresh_derived_geometry(self) -> None:
            self.calls.append(("refresh",))

    class FakeSceneTree:
        def __init__(self) -> None:
            self.selected = 0

        def select_handle(self, handle: int) -> None:
            self.selected = handle

    class FakeStatusBar:
        def __init__(self) -> None:
            self.message = ""

        def showMessage(self, message: str, _timeout: int) -> None:
            self.message = message

    class FakeWindow:
        def __init__(self) -> None:
            self.document = FakeDocument()
            self.scene_tree = FakeSceneTree()
            self.status_bar = FakeStatusBar()
            self.undo_recorded = False
            self.published = False

        def _history_snapshot(self) -> object:
            return object()

        def _record_undo_snapshot(self, _snapshot: object) -> None:
            self.undo_recorded = True

        def _publish_document(
            self,
            clear_selection: bool = True,
            render: bool = False,
        ) -> None:
            self.published = not clear_selection and render

        def statusBar(self) -> FakeStatusBar:
            return self.status_bar

    window = FakeWindow()

    MainWindow._on_viewport_transform_requested(
        window,
        42,
        (1.0, 2.0, 3.0),
        (("z", 15.0, (1.5, 2.5, 3.5)),),
    )

    assert window.document.calls == [
        ("move", 42, (1.0, 2.0, 3.0)),
        ("rotate", 43, "z", 15.0, (1.5, 2.5, 3.5)),
        ("refresh",),
    ]
    assert window.scene_tree.selected == 44
    assert window.undo_recorded
    assert window.published
    assert window.status_bar.message == "Object transform applied"


def test_rename_selection_focuses_properties_name_editor() -> None:
    class FakeSceneTree:
        def selected_handles(self) -> list[int]:
            return [11]

    class FakeProperties:
        def __init__(self) -> None:
            self.focused = False

        def focus_name_editor(self) -> bool:
            self.focused = True
            return True

    class FakeStatusBar:
        def __init__(self) -> None:
            self.message = ""

        def showMessage(self, message: str, _timeout: int) -> None:
            self.message = message

    class FakeWindow:
        def __init__(self) -> None:
            self.scene_tree = FakeSceneTree()
            self.properties = FakeProperties()
            self.status_bar = FakeStatusBar()

        def statusBar(self) -> FakeStatusBar:
            return self.status_bar

    window = FakeWindow()

    MainWindow._rename_selection(window)

    assert window.properties.focused
    assert window.status_bar.message == "Rename selected scene item"
