from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import moderngl
import numpy as np
from core.render_ir import RenderIR, RenderIRNode

X_AXIS_COLOR = (1.0, 0.0, 0.0)
Y_AXIS_COLOR = (0.0, 1.0, 0.0)
Z_AXIS_COLOR = (0.1, 0.45, 1.0)
WORLD_AXIS_LENGTH = 50.0
ROTATION_GIZMO_SEGMENTS = 72
MAX_SELECTED_BOUNDARY_OWNERS = 16
POINT_VERTEX_WIDTH = 7
SQUARE_INSTANCE_WIDTH = 12
IR_MAX_NODES = 64
IR_MAX_PARAM_VEC4S = 512
IR_MAX_COMPONENTS = 32
IR_PARAMETER_PROGRAM_CACHE_LIMIT = 64
IR_PARAM_DATA_BINDING = 2
IR_PARAMETER_MAX_FLOATS = IR_MAX_PARAM_VEC4S * 4


@dataclass(frozen=True)
class SceneUpdateStats:
    path: str
    total_ms: float
    shader_build_ms: float
    program_compile_ms: float
    vao_build_ms: float
    render_ir_nodes: int
    reused_program: bool




@dataclass
class _RenderIRLayerState:
    program: moderngl.Program | None = None
    vao: moderngl.VertexArray | None = None
    topology_signature: tuple[object, ...] | None = None
    values: tuple[float, ...] = ()
    offsets_by_object: dict[tuple[int, str, int], int] | None = None
    render_ir: RenderIR | None = None




class SDFRenderer:
    def __init__(self, context: moderngl.Context) -> None:
        self.context = context
        self._shader_dir = Path(__file__).parent / "renderers" / "opengl" / "shaders"
        vertices = np.asarray(
            (-1.0, -1.0, 3.0, -1.0, -1.0, 3.0), dtype=np.float32
        )
        self._vertex_buffer = context.buffer(vertices.tobytes())
        self._ir_param_buffer: moderngl.Buffer | None = None
        self._scene_layer = _RenderIRLayerState()
        self._preview_layer = _RenderIRLayerState()
        self._last_scene_update_stats: SceneUpdateStats | None = None
        self._points_program = self._load_program(
            "lattice_points.vert", "lattice_cells.frag"
        )
        self._squares_program = self._load_program(
            "lattice_squares.vert", "lattice_cells.frag"
        )
        self._grid_program = self._load_program(
            "grid_overlay.vert", "grid_overlay.frag"
        )
        self._grid_vao = context.vertex_array(
            self._grid_program, [(self._vertex_buffer, "2f", "in_position")]
        )
        self._world_axis_program = self._load_program(
            "world_axis.vert", "lattice_cells.frag"
        )
        world_axis_vertices = self._build_world_axis_vertices()
        self._world_axis_buffer = context.buffer(world_axis_vertices.tobytes())
        self._world_axis_vao = context.vertex_array(
            self._world_axis_program,
            [(self._world_axis_buffer, "3f 3f", "in_position", "in_color")],
        )
        self._world_axis_vertex_count = world_axis_vertices.shape[0]
        self._rotation_gizmo_buffer = context.buffer(
            reserve=3 * ROTATION_GIZMO_SEGMENTS * 2 * 6 * 4
        )
        self._rotation_gizmo_vao = context.vertex_array(
            self._world_axis_program,
            [(self._rotation_gizmo_buffer, "3f 3f", "in_position", "in_color")],
        )
        self._rotation_gizmo_vertex_count = 0
        self._gizmo_program = self._load_program(
            "orientation_gizmo.vert", "orientation_gizmo.frag"
        )
        self._gizmo_label_program = self._load_program(
            "orientation_labels.vert", "orientation_gizmo.frag"
        )
        self._point_buffer: moderngl.Buffer | None = None
        self._points_vao: moderngl.VertexArray | None = None
        self._point_count = 0
        self._preview_point_buffer: moderngl.Buffer | None = None
        self._preview_points_vao: moderngl.VertexArray | None = None
        self._preview_line_buffer: moderngl.Buffer | None = None
        self._preview_lines_vao: moderngl.VertexArray | None = None
        self._preview_point_count = 0
        self._preview_line_vertex_count = 0
        self._stream_point_chunks: list[
            tuple[moderngl.Buffer, moderngl.VertexArray, int]
        ] = []
        square_edges = np.asarray(
            (
                (-0.5, -0.5), (0.5, -0.5),
                (0.5, -0.5), (0.5, 0.5),
                (0.5, 0.5), (-0.5, 0.5),
                (-0.5, 0.5), (-0.5, -0.5),
            ),
            dtype=np.float32,
        )
        self._square_edge_buffer = context.buffer(square_edges.tobytes())
        self._square_instance_buffer: moderngl.Buffer | None = None
        self._squares_vao: moderngl.VertexArray | None = None
        self._square_count = 0
        self._stream_square_chunks: list[
            tuple[moderngl.Buffer, moderngl.VertexArray, int]
        ] = []
        self._cell_size = 1.0
        gizmo_vertices = np.asarray(
            (
                0, 0, 0, *X_AXIS_COLOR,
                1, 0, 0, *X_AXIS_COLOR,
                0, 0, 0, *Y_AXIS_COLOR,
                0, 1, 0, *Y_AXIS_COLOR,
                0, 0, 0, *Z_AXIS_COLOR,
                0, 0, 1, *Z_AXIS_COLOR,
            ),
            dtype=np.float32,
        ).reshape(-1, 6)
        self._gizmo_buffer = context.buffer(gizmo_vertices.tobytes())
        self._gizmo_vao = context.vertex_array(
            self._gizmo_program,
            [(self._gizmo_buffer, "3f 3f", "in_position", "in_color")],
        )
        self._gizmo_vertex_count = gizmo_vertices.shape[0]
        label_vertices = self._build_gizmo_labels()
        self._gizmo_label_buffer = context.buffer(label_vertices.tobytes())
        self._gizmo_label_vao = context.vertex_array(
            self._gizmo_label_program,
            [
                (
                    self._gizmo_label_buffer,
                    "3f 2f 3f",
                    "in_anchor",
                    "in_offset",
                    "in_color",
                )
            ],
        )
        self._gizmo_label_vertex_count = label_vertices.shape[0]
        self._framebuffer: moderngl.Framebuffer | None = None
        self._framebuffer_glo: int | None = None

    @staticmethod
    def _build_world_axis_vertices() -> np.ndarray:
        return np.asarray(
            (
                (0.0, 0.0, -WORLD_AXIS_LENGTH, *Z_AXIS_COLOR),
                (0.0, 0.0, WORLD_AXIS_LENGTH, *Z_AXIS_COLOR),
            ),
            dtype=np.float32,
        )

    @staticmethod
    def build_rotation_gizmo_vertices(
        center: tuple[float, float, float],
        radius: float,
    ) -> np.ndarray:
        center_array = np.asarray(center, dtype=np.float32)
        radius = max(float(radius), 1.0e-6)
        rings = (
            (X_AXIS_COLOR, 1, 2),
            (Y_AXIS_COLOR, 0, 2),
            (Z_AXIS_COLOR, 0, 1),
        )
        vertices: list[tuple[float, ...]] = []
        for color, first_axis, second_axis in rings:
            for index in range(ROTATION_GIZMO_SEGMENTS):
                first_angle = 2.0 * np.pi * index / ROTATION_GIZMO_SEGMENTS
                second_angle = 2.0 * np.pi * (index + 1) / ROTATION_GIZMO_SEGMENTS
                for angle in (first_angle, second_angle):
                    point = center_array.copy()
                    point[first_axis] += radius * np.cos(angle)
                    point[second_axis] += radius * np.sin(angle)
                    vertices.append((*point, *color))
        return np.asarray(vertices, dtype=np.float32)

    def _load_program(
        self, vertex_name: str, fragment_name: str
    ) -> moderngl.Program:
        return self.context.program(
            vertex_shader=(self._shader_dir / vertex_name).read_text(encoding="utf-8"),
            fragment_shader=(self._shader_dir / fragment_name).read_text(
                encoding="utf-8"
            ),
        )



    @staticmethod
    def _build_gizmo_labels() -> np.ndarray:
        vertices: list[tuple[float, ...]] = []

        def segment(
            anchor: tuple[float, float, float],
            first: tuple[float, float],
            second: tuple[float, float],
            color: tuple[float, float, float],
        ) -> None:
            vertices.extend(
                (
                    (*anchor, *first, *color),
                    (*anchor, *second, *color),
                )
            )

        x_anchor = (1.18, 0.0, 0.0)
        segment(x_anchor, (-0.5, -0.6), (0.5, 0.6), X_AXIS_COLOR)
        segment(x_anchor, (-0.5, 0.6), (0.5, -0.6), X_AXIS_COLOR)

        y_anchor = (0.0, 1.18, 0.0)
        segment(y_anchor, (-0.5, 0.6), (0.0, 0.0), Y_AXIS_COLOR)
        segment(y_anchor, (0.5, 0.6), (0.0, 0.0), Y_AXIS_COLOR)
        segment(y_anchor, (0.0, 0.0), (0.0, -0.65), Y_AXIS_COLOR)

        z_anchor = (0.0, 0.0, 1.18)
        segment(z_anchor, (-0.5, 0.6), (0.5, 0.6), Z_AXIS_COLOR)
        segment(z_anchor, (0.5, 0.6), (-0.5, -0.6), Z_AXIS_COLOR)
        segment(z_anchor, (-0.5, -0.6), (0.5, -0.6), Z_AXIS_COLOR)
        return np.asarray(vertices, dtype=np.float32)

    def clear_scene(self) -> None:
        total_start = perf_counter()
        self._scene_layer = _RenderIRLayerState()
        self._last_scene_update_stats = SceneUpdateStats(
            path="render_ir_empty",
            total_ms=(perf_counter() - total_start) * 1000.0,
            shader_build_ms=0.0,
            program_compile_ms=0.0,
            vao_build_ms=0.0,
            render_ir_nodes=0,
            reused_program=False,
        )

    def upload_render_ir(self, render_ir: RenderIR | None) -> bool:
        # codegen removed; the interpreter backend overrides this.
        return False

    def upload_preview_render_ir(self, render_ir: RenderIR | None) -> bool:
        return False

    def clear_preview_render_ir(self) -> None:
        self._preview_layer = _RenderIRLayerState()



    def update_render_ir_object_parameters(
        self, render_ir: RenderIR | None, object_ids: tuple[int, ...]
    ) -> bool:
        return False

    def last_scene_update_stats(self) -> SceneUpdateStats | None:
        return self._last_scene_update_stats

    def _write_parameter_values(self, params: tuple[float, ...]) -> None:
        if self._ir_param_buffer is None:
            self._ir_param_buffer = self.context.buffer(
                reserve=IR_MAX_PARAM_VEC4S * 4 * 4
            )
        data = np.zeros((max(1, (len(params) + 3) // 4), 4), dtype=np.float32)
        for index, value in enumerate(params):
            data[index // 4, index % 4] = float(value)
        self._ir_param_buffer.write(data.tobytes())
        self._ir_param_buffer.bind_to_uniform_block(IR_PARAM_DATA_BINDING)

    def _upload_preview_points(
        self,
        points: tuple[tuple[float, float, float], ...],
    ) -> None:
        point_count = min(len(points), 32)
        if point_count <= 0:
            self._preview_point_count = 0
            self._preview_line_vertex_count = 0
            return
        point_vertices: list[tuple[float, ...]] = []
        line_vertices: list[tuple[float, ...]] = []
        anchor_color = (0.15, 0.92, 1.0)
        control_color = (1.0, 0.76, 0.16)
        line_color = (0.15, 0.92, 1.0)
        for index, point in enumerate(points[:point_count]):
            color = anchor_color if index % 2 == 0 else control_color
            size = 11.0 if index % 2 == 0 else 9.0
            point_vertices.append((*point, *color, size))
            if index + 1 < point_count:
                line_vertices.append((*point, *line_color))
                line_vertices.append((*points[index + 1], *line_color))
        point_data = np.asarray(point_vertices, dtype=np.float32)
        if self._preview_point_buffer is None:
            self._preview_point_buffer = self.context.buffer(
                reserve=32 * POINT_VERTEX_WIDTH * 4
            )
            self._preview_points_vao = self.context.vertex_array(
                self._points_program,
                [
                    (
                        self._preview_point_buffer,
                        "3f 3f 1f",
                        "in_position",
                        "in_color",
                        "in_point_size",
                    )
                ],
            )
        self._preview_point_buffer.write(point_data.tobytes())
        self._preview_point_count = point_data.shape[0]
        if line_vertices:
            line_data = np.asarray(line_vertices, dtype=np.float32)
            if self._preview_line_buffer is None:
                self._preview_line_buffer = self.context.buffer(
                    reserve=64 * 6 * 4
                )
                self._preview_lines_vao = self.context.vertex_array(
                    self._world_axis_program,
                    [
                        (
                            self._preview_line_buffer,
                            "3f 3f",
                            "in_position",
                            "in_color",
                        )
                    ],
                )
            self._preview_line_buffer.write(line_data.tobytes())
            self._preview_line_vertex_count = line_data.shape[0]
        else:
            self._preview_line_vertex_count = 0

    def _write_parameterized_scene_metadata(
        self,
        program: moderngl.Program | None,
        render_ir: RenderIR,
    ) -> None:
        if program is None:
            return
        material_object_ids = [
            int(ref.object_id) for ref in render_ir.material_refs[:IR_MAX_COMPONENTS]
        ]
        material_node_indices = [
            int(ref.node_index) for ref in render_ir.material_refs[:IR_MAX_COMPONENTS]
        ]
        if "u_scene_material_count" in program:
            program["u_scene_material_count"].value = len(material_object_ids)
        if "u_scene_material_object_ids" in program:
            padded = tuple(
                material_object_ids + [0] * (IR_MAX_COMPONENTS - len(material_object_ids))
            )
            program["u_scene_material_object_ids"].value = padded
        if "u_scene_material_node_indices" in program:
            padded = tuple(
                material_node_indices + [0] * (IR_MAX_COMPONENTS - len(material_node_indices))
            )
            program["u_scene_material_node_indices"].value = padded

    def _write_boundary_selection_uniforms(
        self,
        program: moderngl.Program | None,
        render_ir: RenderIR | None,
        boundary_selection_active: bool,
        boundary_hover_owner_id: int,
        boundary_hover_normal: tuple[float, float, float],
        selected_boundary_regions: tuple[tuple[int, int], ...],
        selected_boundary_normals: tuple[tuple[float, float, float], ...],
    ) -> None:
        if program is None:
            return
        node_indices_by_object_id = (
            {
                int(node.object_id): index
                for index, node in enumerate(render_ir.nodes)
                if node.object_id > 0
            }
            if render_ir is not None
            else {}
        )
        count = min(len(selected_boundary_regions), MAX_SELECTED_BOUNDARY_OWNERS)
        owner_ids = [
            int(region[0])
            for region in selected_boundary_regions[:count]
        ]
        node_indices = [
            node_indices_by_object_id.get(owner_id, -1)
            for owner_id in owner_ids
        ]
        whole_flags = [
            int(region[1])
            for region in selected_boundary_regions[:count]
        ]
        normals = [
            tuple(float(value) for value in normal)
            for normal in selected_boundary_normals[:count]
        ]
        normals.extend(
            (0.0, 0.0, 0.0)
            for _index in range(MAX_SELECTED_BOUNDARY_OWNERS - len(normals))
        )
        if "u_boundary_selection_active" in program:
            program["u_boundary_selection_active"].value = boundary_selection_active
        if "u_boundary_hover_owner_id" in program:
            program["u_boundary_hover_owner_id"].value = int(boundary_hover_owner_id)
        if "u_boundary_hover_node_index" in program:
            program["u_boundary_hover_node_index"].value = (
                node_indices_by_object_id.get(int(boundary_hover_owner_id), -1)
            )
        if "u_boundary_hover_normal" in program:
            program["u_boundary_hover_normal"].value = boundary_hover_normal
        if "u_selected_boundary_count" in program:
            program["u_selected_boundary_count"].value = count
        if "u_selected_boundary_owner_ids" in program:
            padded = tuple(
                owner_ids + [0] * (MAX_SELECTED_BOUNDARY_OWNERS - len(owner_ids))
            )
            program["u_selected_boundary_owner_ids"].value = padded
        if "u_selected_boundary_node_indices" in program:
            padded = tuple(
                node_indices
                + [-1] * (MAX_SELECTED_BOUNDARY_OWNERS - len(node_indices))
            )
            program["u_selected_boundary_node_indices"].value = padded
        if "u_selected_boundary_whole_flags" in program:
            padded = tuple(
                whole_flags
                + [0] * (MAX_SELECTED_BOUNDARY_OWNERS - len(whole_flags))
            )
            program["u_selected_boundary_whole_flags"].value = padded
        if "u_selected_boundary_normals" in program:
            program["u_selected_boundary_normals"].write(
                np.asarray(normals, dtype=np.float32).tobytes()
            )

    def has_scene_program(self) -> bool:
        return self._scene_layer.program is not None and self._scene_layer.vao is not None

    def bind_framebuffer(self, framebuffer_glo: int) -> None:
        if (
            self._framebuffer is None
            or self._framebuffer_glo != framebuffer_glo
        ):
            if self._framebuffer is not None:
                self._framebuffer.release()
            self._framebuffer = self.context.detect_framebuffer(
                glo=framebuffer_glo
            )
            self._framebuffer_glo = framebuffer_glo
        self._framebuffer.use()

    def upload_lattice(
        self,
        positions: np.ndarray,
        node_types: np.ndarray,
        boundary_faces: np.ndarray,
        source_object_ids: np.ndarray,
        primary_tag_ids: np.ndarray,
        tag_axis_u: np.ndarray,
        tag_axis_v: np.ndarray,
        cell_size: float,
        dimension: int = 3,
        axis_i: tuple[float, float, float] = (1.0, 0.0, 0.0),
        axis_j: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> None:
        point_vertices, square_instances = self.prepare_lattice_upload(
            positions,
            node_types,
            boundary_faces,
            source_object_ids,
            primary_tag_ids,
            cell_size,
            dimension=dimension,
            axis_i=axis_i,
            axis_j=axis_j,
        )
        self.begin_lattice_upload(
            point_vertices.shape[0],
            square_instances.shape[0],
            cell_size,
        )
        self.write_lattice_points(0, point_vertices)
        self.write_lattice_squares(0, square_instances)

    def clear_lattice(self) -> None:
        if self._points_vao is not None:
            self._points_vao.release()
        if self._point_buffer is not None:
            self._point_buffer.release()
        if self._preview_points_vao is not None:
            self._preview_points_vao.release()
        if self._preview_point_buffer is not None:
            self._preview_point_buffer.release()
        if self._preview_lines_vao is not None:
            self._preview_lines_vao.release()
        if self._preview_line_buffer is not None:
            self._preview_line_buffer.release()
        if self._squares_vao is not None:
            self._squares_vao.release()
        if self._square_instance_buffer is not None:
            self._square_instance_buffer.release()
        self.clear_lattice_stream()
        self._point_buffer = None
        self._points_vao = None
        self._square_instance_buffer = None
        self._squares_vao = None
        self._point_count = 0
        self._square_count = 0

    def clear_lattice_stream(self) -> None:
        for buffer, vao, _count in self._stream_point_chunks:
            vao.release()
            buffer.release()
        for buffer, vao, _count in self._stream_square_chunks:
            vao.release()
            buffer.release()
        self._stream_point_chunks.clear()
        self._stream_square_chunks.clear()

    def append_lattice_preview_chunk(
        self,
        positions: np.ndarray,
        node_types: np.ndarray,
        boundary_faces: np.ndarray,
        source_object_ids: np.ndarray,
        primary_tag_ids: np.ndarray,
        cell_size: float,
        dimension: int = 3,
        axis_i: tuple[float, float, float] = (1.0, 0.0, 0.0),
        axis_j: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> None:
        point_vertices, square_instances = self.prepare_lattice_upload(
            positions,
            node_types,
            boundary_faces,
            source_object_ids,
            primary_tag_ids,
            cell_size,
            dimension=dimension,
            axis_i=axis_i,
            axis_j=axis_j,
        )
        self._cell_size = float(cell_size)
        if point_vertices.size:
            point_buffer = self.context.buffer(point_vertices.tobytes())
            point_vao = self.context.vertex_array(
                self._points_program,
                [
                    (
                        point_buffer,
                        "3f 3f 1f",
                        "in_position",
                        "in_color",
                        "in_point_size",
                    )
                ],
            )
            self._stream_point_chunks.append(
                (point_buffer, point_vao, point_vertices.shape[0])
            )
        if square_instances.size:
            square_buffer = self.context.buffer(square_instances.tobytes())
            square_vao = self.context.vertex_array(
                self._squares_program,
                [
                    (self._square_edge_buffer, "2f", "in_offset"),
                    (
                        square_buffer,
                        "3f 3f 3f 3f /i",
                        "in_center",
                        "in_color",
                        "in_axis_u",
                        "in_axis_v",
                    ),
                ],
            )
            self._stream_square_chunks.append(
                (square_buffer, square_vao, square_instances.shape[0])
            )

    def begin_lattice_upload(
        self,
        point_count: int,
        square_count: int,
        cell_size: float,
    ) -> None:
        self.clear_lattice()
        self._cell_size = float(cell_size)
        if point_count > 0:
            self._point_buffer = self.context.buffer(
                reserve=point_count * POINT_VERTEX_WIDTH * 4
            )
            self._points_vao = self.context.vertex_array(
                self._points_program,
                [
                    (
                        self._point_buffer,
                        "3f 3f 1f",
                        "in_position",
                        "in_color",
                        "in_point_size",
                    )
                ],
            )
        if square_count > 0:
            self._square_instance_buffer = self.context.buffer(
                reserve=square_count * SQUARE_INSTANCE_WIDTH * 4
            )
            self._squares_vao = self.context.vertex_array(
                self._squares_program,
                [
                    (self._square_edge_buffer, "2f", "in_offset"),
                    (
                        self._square_instance_buffer,
                        "3f 3f 3f 3f /i",
                        "in_center",
                        "in_color",
                        "in_axis_u",
                        "in_axis_v",
                    ),
                ],
            )

    def write_lattice_points(
        self,
        start: int,
        point_vertices: np.ndarray,
    ) -> None:
        if self._point_buffer is None or point_vertices.size == 0:
            return
        vertices = point_vertices.astype(np.float32, copy=False)
        self._point_buffer.write(
            vertices.tobytes(),
            offset=start * POINT_VERTEX_WIDTH * 4,
        )
        self._point_count = max(self._point_count, start + vertices.shape[0])

    def write_lattice_squares(
        self,
        start: int,
        square_instances: np.ndarray,
    ) -> None:
        if self._square_instance_buffer is None or square_instances.size == 0:
            return
        instances = square_instances.astype(np.float32, copy=False)
        self._square_instance_buffer.write(
            instances.tobytes(),
            offset=start * SQUARE_INSTANCE_WIDTH * 4,
        )
        self._square_count = max(self._square_count, start + instances.shape[0])

    @classmethod
    def prepare_lattice_upload(
        cls,
        positions: np.ndarray,
        node_types: np.ndarray,
        boundary_faces: np.ndarray,
        source_object_ids: np.ndarray,
        primary_tag_ids: np.ndarray,
        cell_size: float,
        *,
        dimension: int = 3,
        axis_i: tuple[float, float, float] = (1.0, 0.0, 0.0),
        axis_j: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> tuple[np.ndarray, np.ndarray]:
        if positions.size == 0:
            return (
                np.empty((0, POINT_VERTEX_WIDTH), dtype=np.float32),
                np.empty((0, SQUARE_INSTANCE_WIDTH), dtype=np.float32),
            )
        boundary = node_types == 1
        colors = cls._lattice_colors(
            node_types,
            source_object_ids,
            primary_tag_ids,
        )
        point_vertices = cls._lattice_point_vertices(
            positions,
            boundary,
            colors,
        )
        square_instances = cls._build_boundary_square_instances(
            positions,
            boundary_faces,
            colors,
            cell_size,
            dimension=dimension,
            axis_i=axis_i,
            axis_j=axis_j,
        )
        return point_vertices, square_instances

    @staticmethod
    def _lattice_point_vertices(
        positions: np.ndarray,
        boundary: np.ndarray,
        colors: np.ndarray,
    ) -> np.ndarray:
        point_sizes = np.where(boundary, 5.0, 3.0).astype(np.float32)
        return np.column_stack((positions, colors, point_sizes)).astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _lattice_colors(
        node_types: np.ndarray,
        source_object_ids: np.ndarray,
        primary_tag_ids: np.ndarray,
    ) -> np.ndarray:
        """Assign stable colors to fluid interior, boundary owners, and tags."""
        palette = np.asarray(
            (
                (0.12, 0.42, 1.00),
                (1.00, 0.35, 0.18),
                (0.25, 0.90, 0.38),
                (1.00, 0.78, 0.18),
                (0.95, 0.30, 0.80),
                (0.20, 0.88, 0.88),
                (0.72, 0.42, 1.00),
                (0.95, 0.55, 0.62),
                (0.55, 0.85, 0.20),
                (1.00, 0.52, 0.12),
            ),
            dtype=np.float32,
        )
        colors = np.full(
            (node_types.size, 3),
            (0.12, 0.42, 1.00),
            dtype=np.float32,
        )
        attributed = source_object_ids != 0
        source_ids = source_object_ids[attributed].astype(np.int64)
        colors[attributed] = palette[(source_ids - 1) % len(palette)]
        tagged = primary_tag_ids != 0
        tag_ids = primary_tag_ids[tagged].astype(np.int64)
        colors[tagged] = palette[(tag_ids - 1) % len(palette)]
        return colors

    @staticmethod
    def _build_boundary_square_instances(
        positions: np.ndarray,
        boundary_faces: np.ndarray,
        colors: np.ndarray,
        cell_size: float,
        *,
        dimension: int = 3,
        axis_i: tuple[float, float, float] = (1.0, 0.0, 0.0),
        axis_j: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> np.ndarray:
        """Build preview cell outlines for boundary lattice nodes."""
        if positions.size == 0:
            return np.empty((0, 12), dtype=np.float32)
        if dimension == 2:
            return np.empty((0, 12), dtype=np.float32)
        origin = np.min(positions, axis=0)
        indices = np.rint((positions - origin) / cell_size).astype(np.int64)
        index_by_key = {
            tuple(int(value) for value in key): index
            for index, key in enumerate(indices)
        }
        face_frames = (
            ((0, 1, 0), (0, 0, 1)),
            ((0, 1, 0), (0, 0, 1)),
            ((1, 0, 0), (0, 0, 1)),
            ((1, 0, 0), (0, 0, 1)),
            ((1, 0, 0), (0, 1, 0)),
            ((1, 0, 0), (0, 1, 0)),
        )
        instances: list[np.ndarray] = []
        for bit, (axis_u, axis_v) in enumerate(face_frames):
            face_bit = np.uint8(1 << bit)
            axis_u_array = np.asarray(axis_u, dtype=np.int64)
            axis_v_array = np.asarray(axis_v, dtype=np.int64)
            world_axis_u = axis_u_array.astype(np.float32)
            world_axis_v = axis_v_array.astype(np.float32)
            for first_index, key in enumerate(indices):
                if boundary_faces[first_index] & face_bit == 0:
                    continue
                vertex_indices = [first_index]
                for offset in (
                    axis_u_array,
                    axis_v_array,
                    axis_u_array + axis_v_array,
                ):
                    neighbor = index_by_key.get(
                        tuple(int(value) for value in key + offset)
                    )
                    if (
                        neighbor is None
                        or boundary_faces[neighbor] & face_bit == 0
                    ):
                        break
                    vertex_indices.append(neighbor)
                if len(vertex_indices) != 4:
                    continue
                center = np.mean(positions[vertex_indices], axis=0)
                color = colors[vertex_indices[0]]
                instances.append(
                    np.concatenate(
                        (center, color, world_axis_u, world_axis_v)
                    )
                )
        return (
            np.asarray(instances, dtype=np.float32)
            if instances
            else np.empty((0, 12), dtype=np.float32)
        )

    def render(
        self,
        width: int,
        height: int,
        camera_position: tuple[float, float, float],
        camera_target: tuple[float, float, float],
        focal_length: float,
        view_projection: np.ndarray,
        mode: str,
        grid_visible: bool,
        components_visible: bool,
        sdf_opacity: float,
        background_color: tuple[float, float, float],
        view_rotation: np.ndarray,
        gizmo_visible: bool,
        grid_spacing: float,
        grid_plane: int,
        boundary_selection_active: bool,
        boundary_hover_owner_id: int,
        boundary_hover_direction: int,
        boundary_hover_normal: tuple[float, float, float],
        scene_hover_object_id: int,
        scene_selected_object_id: int,
        selected_boundary_regions: tuple[tuple[int, int], ...],
        selected_boundary_normals: tuple[tuple[float, float, float], ...],
        preview_point_count: int,
        preview_points: tuple[tuple[float, float, float], ...],
        rotation_gizmo_visible: bool,
        rotation_gizmo_center: tuple[float, float, float],
        rotation_gizmo_radius: float,
    ) -> None:
        self.context.viewport = (0, 0, width, height)
        self.context.clear(*background_color, 1.0)
        self.context.enable(moderngl.DEPTH_TEST)
        self.context.enable(moderngl.PROGRAM_POINT_SIZE)
        matrix_bytes = view_projection.T.astype(np.float32).tobytes()
        scene_program = self._scene_layer.program
        scene_vao = self._scene_layer.vao
        preview_program = self._preview_layer.program
        preview_vao = self._preview_layer.vao
        preview_active = (
            preview_program is not None
            and preview_vao is not None
            and self._preview_layer.render_ir is not None
            and bool(self._preview_layer.values)
        )
        if mode == "sdf" and (
            (scene_program is not None and scene_vao is not None)
            or preview_active
        ):
            self.context.disable(moderngl.DEPTH_TEST)
            uniform_values = {
                "u_resolution": (float(width), float(height)),
                "u_camera_position": camera_position,
                "u_camera_target": camera_target,
                "u_camera_right": tuple(float(value) for value in view_rotation[0]),
                "u_camera_up": tuple(float(value) for value in view_rotation[1]),
                "u_focal_length": focal_length,
                "u_show_components": components_visible,
                "u_surface_opacity": sdf_opacity,
                "u_background_color": background_color,
                "u_show_grid": grid_visible,
                "u_render_preview_layer": False,
                "u_grid_spacing": grid_spacing,
                "u_grid_plane": grid_plane,
                "u_scene_selected_object_id": int(scene_selected_object_id),
            }
            if scene_program is not None and scene_vao is not None:
                for name, value in uniform_values.items():
                    if name in scene_program:
                        scene_program[name].value = value
                self._write_boundary_selection_uniforms(
                    scene_program,
                    self._scene_layer.render_ir,
                    boundary_selection_active,
                    boundary_hover_owner_id,
                    boundary_hover_normal,
                    selected_boundary_regions,
                    selected_boundary_normals,
                )
                if self._scene_layer.render_ir is not None:
                    self._write_parameterized_scene_metadata(
                        scene_program,
                        self._scene_layer.render_ir,
                    )
                self._write_parameter_values(self._scene_layer.values)
                scene_vao.render(mode=moderngl.TRIANGLES)
            elif preview_active:
                background_uniform_values = dict(uniform_values)
                background_uniform_values.update(
                    {
                        "u_show_grid": grid_visible,
                        "u_render_preview_layer": False,
                        "u_surface_opacity": 0.0,
                    }
                )
                for name, value in background_uniform_values.items():
                    if name in preview_program:
                        preview_program[name].value = value
                self._write_boundary_selection_uniforms(
                    preview_program,
                    self._preview_layer.render_ir,
                    boundary_selection_active,
                    boundary_hover_owner_id,
                    boundary_hover_normal,
                    selected_boundary_regions,
                    selected_boundary_normals,
                )
                assert self._preview_layer.render_ir is not None
                self._write_parameterized_scene_metadata(
                    preview_program,
                    self._preview_layer.render_ir,
                )
                self._write_parameter_values(self._preview_layer.values)
                preview_vao.render(mode=moderngl.TRIANGLES)
            if preview_active:
                preview_uniform_values = dict(uniform_values)
                preview_uniform_values.update(
                    {
                        "u_show_grid": False,
                        "u_render_preview_layer": True,
                        "u_surface_opacity": 1.0,
                    }
                )
                for name, value in preview_uniform_values.items():
                    assert preview_program is not None
                    if name in preview_program:
                        preview_program[name].value = value
                self._write_boundary_selection_uniforms(
                    preview_program,
                    self._preview_layer.render_ir,
                    False,
                    0,
                    (0.0, 0.0, 0.0),
                    (),
                    (),
                )
                self._write_parameterized_scene_metadata(
                    preview_program,
                    self._preview_layer.render_ir,
                )
                self._write_parameter_values(self._preview_layer.values)
                self.context.enable(moderngl.BLEND)
                self.context.blend_func = (
                    moderngl.SRC_ALPHA,
                    moderngl.ONE_MINUS_SRC_ALPHA,
                )
                preview_vao.render(mode=moderngl.TRIANGLES)
                self.context.disable(moderngl.BLEND)
                self._write_parameter_values(self._scene_layer.values)
            if preview_point_count > 0 and preview_points:
                self._upload_preview_points(preview_points)
                self._points_program["u_view_projection"].write(matrix_bytes)
                if (
                    self._preview_lines_vao is not None
                    and self._preview_line_vertex_count > 0
                ):
                    self._world_axis_program["u_view_projection"].write(matrix_bytes)
                    self._preview_lines_vao.render(
                        mode=moderngl.LINES,
                        vertices=self._preview_line_vertex_count,
                    )
                if (
                    self._preview_points_vao is not None
                    and self._preview_point_count > 0
                ):
                    self._preview_points_vao.render(
                        mode=moderngl.POINTS,
                        vertices=self._preview_point_count,
                    )
            self.context.enable(moderngl.DEPTH_TEST)
        elif mode == "lattice":
            if grid_visible:
                self.context.disable(moderngl.DEPTH_TEST)
                self._grid_program["u_resolution"].value = (
                    float(width),
                    float(height),
                )
                self._grid_program["u_camera_position"].value = camera_position
                self._grid_program["u_camera_target"].value = camera_target
                self._grid_program["u_camera_right"].value = tuple(
                    float(value) for value in view_rotation[0]
                )
                self._grid_program["u_camera_up"].value = tuple(
                    float(value) for value in view_rotation[1]
                )
                self._grid_program["u_focal_length"].value = focal_length
                self._grid_program["u_grid_spacing"].value = grid_spacing
                self._grid_program["u_grid_plane"].value = grid_plane
                self._grid_program["u_background_color"].value = background_color
                self._grid_vao.render(mode=moderngl.TRIANGLES)
                self.context.enable(moderngl.DEPTH_TEST)
            if self._points_vao is not None:
                self._points_program["u_view_projection"].write(matrix_bytes)
                self._points_vao.render(
                    mode=moderngl.POINTS,
                    vertices=self._point_count,
                )
            if self._stream_point_chunks:
                self._points_program["u_view_projection"].write(matrix_bytes)
                for _buffer, vao, count in self._stream_point_chunks:
                    vao.render(mode=moderngl.POINTS, vertices=count)
        if mode == "lattice" and self._squares_vao is not None:
            self._squares_program["u_view_projection"].write(matrix_bytes)
            self._squares_program["u_cell_size"].value = self._cell_size
            self._squares_vao.render(
                mode=moderngl.LINES,
                vertices=8,
                instances=self._square_count,
            )
        if mode == "lattice" and self._stream_square_chunks:
            self._squares_program["u_view_projection"].write(matrix_bytes)
            self._squares_program["u_cell_size"].value = self._cell_size
            for _buffer, vao, count in self._stream_square_chunks:
                vao.render(mode=moderngl.LINES, vertices=8, instances=count)
        if grid_visible:
            self.context.disable(moderngl.DEPTH_TEST)
            self._world_axis_program["u_view_projection"].write(matrix_bytes)
            self._world_axis_vao.render(
                mode=moderngl.LINES,
                vertices=self._world_axis_vertex_count,
            )
        if rotation_gizmo_visible:
            vertices = self.build_rotation_gizmo_vertices(
                rotation_gizmo_center,
                rotation_gizmo_radius,
            )
            self._rotation_gizmo_buffer.write(vertices.tobytes())
            self._rotation_gizmo_vertex_count = vertices.shape[0]
            self.context.disable(moderngl.DEPTH_TEST)
            self._world_axis_program["u_view_projection"].write(matrix_bytes)
            self._rotation_gizmo_vao.render(
                mode=moderngl.LINES,
                vertices=self._rotation_gizmo_vertex_count,
            )
        if gizmo_visible:
            self.context.disable(moderngl.DEPTH_TEST)
            self._gizmo_program["u_view_rotation"].write(
                view_rotation.T.astype(np.float32).tobytes()
            )
            self._gizmo_program["u_origin"].value = (-0.84, -0.78)
            self._gizmo_program["u_scale"].value = (
                0.12 * height / max(width, 1),
                0.12,
            )
            self._gizmo_vao.render(
                mode=moderngl.LINES, vertices=self._gizmo_vertex_count
            )
            self._gizmo_program["u_point_size"].value = 7.0
            self._gizmo_vao.render(
                mode=moderngl.POINTS, vertices=self._gizmo_vertex_count
            )
            self._gizmo_label_program["u_view_rotation"].write(
                view_rotation.T.astype(np.float32).tobytes()
            )
            self._gizmo_label_program["u_origin"].value = (-0.84, -0.78)
            self._gizmo_label_program["u_scale"].value = (
                0.12 * height / max(width, 1),
                0.12,
            )
            self._gizmo_label_program["u_label_scale"].value = (
                0.018 * height / max(width, 1),
                0.018,
            )
            self._gizmo_label_vao.render(
                mode=moderngl.LINES,
                vertices=self._gizmo_label_vertex_count,
            )
            self.context.enable(moderngl.DEPTH_TEST)

    def release(self) -> None:
        released_vaos: set[int] = set()
        released_programs: set[int] = set()

        def release_vao(vao: moderngl.VertexArray | None) -> None:
            if vao is None:
                return
            vao_id = id(vao)
            if vao_id in released_vaos:
                return
            vao.release()
            released_vaos.add(vao_id)

        def release_program(program: moderngl.Program | None) -> None:
            if program is None:
                return
            program_id = id(program)
            if program_id in released_programs:
                return
            program.release()
            released_programs.add(program_id)

        if self._ir_param_buffer is not None:
            self._ir_param_buffer.release()
        release_vao(self._scene_layer.vao)
        release_program(self._scene_layer.program)
        release_vao(self._preview_layer.vao)
        release_program(self._preview_layer.program)
        if self._points_vao is not None:
            self._points_vao.release()
        if self._point_buffer is not None:
            self._point_buffer.release()
        if self._squares_vao is not None:
            self._squares_vao.release()
        if self._square_instance_buffer is not None:
            self._square_instance_buffer.release()
        self.clear_lattice_stream()
        self._grid_vao.release()
        self._world_axis_vao.release()
        self._world_axis_buffer.release()
        self._rotation_gizmo_vao.release()
        self._rotation_gizmo_buffer.release()
        self._world_axis_program.release()
        if self._framebuffer is not None:
            self._framebuffer.release()
        self._gizmo_vao.release()
        self._gizmo_buffer.release()
        self._gizmo_program.release()
        self._gizmo_label_vao.release()
        self._gizmo_label_buffer.release()
        self._gizmo_label_program.release()
        self._square_edge_buffer.release()
        self._points_program.release()
        self._squares_program.release()
        self._grid_program.release()
        self._vertex_buffer.release()
