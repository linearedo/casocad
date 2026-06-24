from __future__ import annotations

from typing import Protocol, Sequence

from core.render_ir import RenderIR
from core.sdf import SDFTree

from .renderer import SceneUpdateStats


class ViewportRenderer(Protocol):
    def clear_scene(self) -> None: ...

    def upload_render_ir(self, render_ir: RenderIR | None) -> bool: ...

    def upload_preview_render_ir(self, render_ir: RenderIR | None) -> bool: ...

    def clear_preview_render_ir(self) -> None: ...

    def update_render_ir_object_parameters(
        self,
        render_ir: RenderIR | None,
        object_ids: tuple[int, ...],
    ) -> bool: ...

    def last_scene_update_stats(self) -> SceneUpdateStats | None: ...

    def bind_framebuffer(self, framebuffer_glo: int) -> None: ...

    def has_scene_program(self) -> bool: ...

    def render(
        self,
        width: int,
        height: int,
        camera_position: tuple[float, float, float],
        camera_target: tuple[float, float, float],
        focal_length: float,
        view_projection,
        mode: str,
        grid_visible: bool,
        components_visible: bool,
        sdf_opacity: float,
        background_color: tuple[float, float, float],
        view_rotation,
        gizmo_visible: bool,
        grid_spacing: float,
        grid_plane: int,
        boundary_selection_active: bool,
        boundary_hover_owner_id: int,
        boundary_hover_direction: int,
        boundary_hover_normal: tuple[float, float, float],
        scene_hover_object_id: int,
        scene_selected_object_id: int,
        selected_boundary_regions: Sequence[tuple[int, int]],
        selected_boundary_normals: Sequence[tuple[float, float, float]],
        preview_point_count: int,
        preview_points: Sequence[tuple[float, float, float]],
        rotation_gizmo_visible: bool,
        rotation_gizmo_center: tuple[float, float, float],
        rotation_gizmo_radius: float,
    ) -> None: ...

    def release(self) -> None: ...


__all__ = [
    "ViewportRenderer",
]
