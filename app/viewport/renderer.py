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
MAX_SELECTED_BOUNDARY_OWNERS = 128
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
class _ParameterizedProgramCacheEntry:
    program: moderngl.Program
    vao: moderngl.VertexArray
    offsets_by_object: dict[tuple[int, str, int], int]


@dataclass
class _RenderIRLayerState:
    program: moderngl.Program | None = None
    vao: moderngl.VertexArray | None = None
    topology_signature: tuple[object, ...] | None = None
    values: tuple[float, ...] = ()
    offsets_by_object: dict[tuple[int, str, int], int] | None = None
    render_ir: RenderIR | None = None


class _ParameterizedIRSceneSource:
    def __init__(self, render_ir: RenderIR) -> None:
        self.render_ir = render_ir
        self.param_offsets: dict[int, int] = {}
        offset = 0
        for index, node in enumerate(render_ir.nodes):
            self.param_offsets[index] = offset
            offset += len(node.params)
        self.param_vec4_count = max(1, (offset + 3) // 4)

    def build(self) -> str:
        lines = [
            "uniform int u_scene_material_count;",
            f"uniform int u_scene_material_object_ids[{IR_MAX_COMPONENTS}];",
            f"uniform int u_scene_material_node_indices[{IR_MAX_COMPONENTS}];",
            "",
            "layout(std140) uniform SceneParamData {",
            f"    vec4 u_scene_param_vec4s[{self.param_vec4_count}];",
            "};",
            "",
            "vec3 irOrientedLocal(",
            "    vec3 p, vec3 center, vec3 axis_u, vec3 axis_v, vec3 axis_w",
            ") {",
            "    vec3 local = p - center;",
            "    return vec3(",
            "        dot(local, axis_u),",
            "        dot(local, axis_v),",
            "        dot(local, axis_w)",
            "    );",
            "}",
            "",
            "float irTubeSDF(",
            "    float centerline_distance, float radius, float inner_radius",
            ") {",
            "    float outer = centerline_distance - radius;",
            "    if (inner_radius <= 0.0) return outer;",
            "    return max(outer, inner_radius - centerline_distance);",
            "}",
            "",
            "float irFlatTubeSDF(",
            "    float outer, float centerline_distance, float inner_radius",
            ") {",
            "    if (inner_radius <= 0.0) return outer;",
            "    return max(outer, inner_radius - centerline_distance);",
            "}",
            "",
            "float irSegmentDistance2D(vec2 p, vec2 a, vec2 b) {",
            "    vec2 pa = p - a;",
            "    vec2 ba = b - a;",
            "    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1.0e-12), 0.0, 1.0);",
            "    return length(pa - ba * h);",
            "}",
            "",
            "vec3 irSafeDirection(vec3 preferred, vec3 fallback) {",
            "    float preferred_length = length(preferred);",
            "    if (preferred_length > 1.0e-12) return preferred / preferred_length;",
            "    float fallback_length = max(length(fallback), 1.0e-12);",
            "    return fallback / fallback_length;",
            "}",
            "",
        ]
        for index, node in enumerate(self.render_ir.nodes):
            if self._is_profile_1d_node(node):
                lines.append(self._profile_1d_sdf_function(index, node))
        for index, node in enumerate(self.render_ir.nodes):
            if self._is_profile_2d_node(node):
                lines.append(self._profile_sdf_function(index, node))
        for index, node in enumerate(self.render_ir.nodes):
            lines.append(self._sdf_function(index, node))
        lines.append(self._node_dispatch_sdf_function())
        lines.append(self._scene_contract())
        return "\n".join(lines)

    @staticmethod
    def _sdf_name(index: int) -> str:
        return f"irNodeSDF_{index}"

    @staticmethod
    def _profile_sdf_name(index: int) -> str:
        return f"irProfileSDF_{index}"

    @staticmethod
    def _profile_1d_sdf_name(index: int) -> str:
        return f"irProfile1DSDF_{index}"

    @staticmethod
    def _is_profile_1d_node(node: RenderIRNode) -> bool:
        return node.kind.startswith("profile_") and node.kind.endswith("_1d")

    @staticmethod
    def _is_profile_2d_node(node: RenderIRNode) -> bool:
        return node.kind.startswith("profile_") and node.kind.endswith("_2d")

    @staticmethod
    def _minimum(expressions: list[str]) -> str:
        expression = expressions[0]
        for item in expressions[1:]:
            expression = f"min({expression}, {item})"
        return expression

    def _param(self, index: int) -> str:
        components = ("x", "y", "z", "w")
        return f"u_scene_param_vec4s[{index // 4}].{components[index % 4]}"

    def _vec3(self, index: int) -> str:
        return (
            f"vec3({self._param(index)}, {self._param(index + 1)},"
            f" {self._param(index + 2)})"
        )

    def _node_param(self, node_index: int, relative_index: int) -> str:
        return self._param(self.param_offsets[node_index] + relative_index)

    def _node_vec3(self, node_index: int, relative_index: int) -> str:
        return self._vec3(self.param_offsets[node_index] + relative_index)

    def _node_vec2(self, node_index: int, relative_index: int) -> str:
        return (
            f"vec2({self._node_param(node_index, relative_index)},"
            f" {self._node_param(node_index, relative_index + 1)})"
        )

    def _oriented_local(self, node_index: int) -> str:
        return (
            "irOrientedLocal("
            f"p, {self._node_vec3(node_index, 0)},"
            f" {self._node_vec3(node_index, 3)},"
            f" {self._node_vec3(node_index, 6)},"
            f" {self._node_vec3(node_index, 9)})"
        )

    def _placed_2d_profile_coordinates(
        self,
        node_index: int,
    ) -> tuple[str, str, str]:
        local = f"(p - {self._node_vec3(node_index, 0)})"
        u = f"dot({local}, {self._node_vec3(node_index, 3)})"
        v = f"dot({local}, {self._node_vec3(node_index, 6)})"
        plane = f"dot({local}, {self._node_vec3(node_index, 9)})"
        return u, v, plane

    def _point(self, node_index: int, point_index: int) -> str:
        return self._node_vec3(node_index, point_index * 3)

    def _polyline_centerline(self, node_index: int, point_count: int) -> str:
        distances = [
            "segmentDistance3D("
            f"p, {self._point(node_index, index)},"
            f" {self._point(node_index, index + 1)})"
            for index in range(point_count - 1)
        ]
        return self._minimum(distances)

    def _bezier_centerline(self, node_index: int, point_count: int) -> str:
        distances = [
            "quadraticBezierDistance3D("
            f"p, {self._point(node_index, index)},"
            f" {self._point(node_index, index + 1)},"
            f" {self._point(node_index, index + 2)})"
            for index in range(0, point_count - 2, 2)
        ]
        return self._minimum(distances)

    def _point_2d(self, node_index: int, point_index: int, offset: int = 0) -> str:
        return self._node_vec2(node_index, offset + point_index * 2)

    def _polyline_distance_2d(
        self,
        node_index: int,
        point_count: int,
        *,
        q_var: str,
        offset: int = 0,
        closed: bool = False,
    ) -> str:
        if point_count < 2:
            return "1000000.0"
        end = point_count if closed else point_count - 1
        distances = [
            "irSegmentDistance2D("
            f"{q_var}, {self._point_2d(node_index, point_index, offset)},"
            f" {self._point_2d(node_index, (point_index + 1) % point_count, offset)})"
            for point_index in range(end)
        ]
        return self._minimum(distances)

    def _bezier_distance_2d(
        self,
        node_index: int,
        point_count: int,
        *,
        q_var: str,
        offset: int = 0,
    ) -> str:
        if point_count < 3:
            return "1000000.0"
        distances = [
            "quadraticBezierDistance("
            f"{q_var}, {self._point_2d(node_index, point_index, offset)},"
            f" {self._point_2d(node_index, point_index + 1, offset)},"
            f" {self._point_2d(node_index, point_index + 2, offset)})"
            for point_index in range(0, point_count - 2, 2)
        ]
        return self._minimum(distances) if distances else "1000000.0"

    def _profile_1d_sdf_function(self, index: int, node: RenderIRNode) -> str:
        header = f"float {self._profile_1d_sdf_name(index)}(float t) {{"
        if node.kind == "profile_segment_1d":
            body = [
                f"    return abs(t - {self._node_param(index, 0)})"
                f" - {self._node_param(index, 1)};"
            ]
        elif node.kind == "profile_union_1d":
            left, right = node.children
            body = [
                f"    float a = {self._profile_1d_sdf_name(left)}(t);",
                f"    float b = {self._profile_1d_sdf_name(right)}(t);",
                "    return min(a, b);",
            ]
        elif node.kind == "profile_intersection_1d":
            left, right = node.children
            body = [
                f"    float a = {self._profile_1d_sdf_name(left)}(t);",
                f"    float b = {self._profile_1d_sdf_name(right)}(t);",
                "    return max(a, b);",
            ]
        elif node.kind == "profile_difference_1d":
            left, right = node.children
            body = [
                f"    float a = {self._profile_1d_sdf_name(left)}(t);",
                f"    float b = {self._profile_1d_sdf_name(right)}(t);",
                "    return max(a, -b);",
            ]
        elif node.kind == "profile_smooth_union_1d":
            left, right = node.children
            body = [
                f"    float a = {self._profile_1d_sdf_name(left)}(t);",
                f"    float b = {self._profile_1d_sdf_name(right)}(t);",
                f"    float k = {self._node_param(index, 0)};",
                "    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);",
                "    return mix(b, a, h) - k * h * (1.0 - h);",
            ]
        else:
            body = ["    return 1000000.0;"]
        return "\n".join((header, *body, "}"))

    def _profile_sdf_function(self, index: int, node: RenderIRNode) -> str:
        header = f"float {self._profile_sdf_name(index)}(vec2 q) {{"
        if node.kind == "profile_circle_2d":
            body = [
                f"    return length(q - vec2({self._node_param(index, 0)},"
                f" {self._node_param(index, 1)})) - {self._node_param(index, 2)};"
            ]
        elif node.kind == "profile_rectangle_2d":
            body = [
                f"    vec2 center = vec2({self._node_param(index, 0)},"
                f" {self._node_param(index, 1)});",
                f"    vec2 half_size = vec2({self._node_param(index, 2)},"
                f" {self._node_param(index, 3)});",
                "    vec2 delta = abs(q - center) - half_size;",
                "    return length(max(delta, vec2(0.0)))"
                " + min(max(delta.x, delta.y), 0.0);",
            ]
        elif node.kind == "profile_square_2d":
            body = [
                f"    vec2 center = vec2({self._node_param(index, 0)},"
                f" {self._node_param(index, 1)});",
                f"    float half_size = {self._node_param(index, 2)};",
                "    vec2 delta = abs(q - center) - vec2(half_size);",
                "    return length(max(delta, vec2(0.0)))"
                " + min(max(delta.x, delta.y), 0.0);",
            ]
        elif node.kind == "profile_rounded_rectangle_2d":
            body = [
                f"    vec2 center = vec2({self._node_param(index, 0)},"
                f" {self._node_param(index, 1)});",
                f"    vec2 half_size = vec2({self._node_param(index, 2)},"
                f" {self._node_param(index, 3)});",
                f"    float corner_radius = {self._node_param(index, 4)};",
                "    vec2 inner = half_size - vec2(corner_radius);",
                "    vec2 delta = abs(q - center) - inner;",
                "    return length(max(delta, vec2(0.0)))"
                " + min(max(delta.x, delta.y), 0.0) - corner_radius;",
            ]
        elif node.kind == "profile_ellipse_2d":
            body = [
                "    return exactEllipseDistance(",
                f"        q - vec2({self._node_param(index, 0)},"
                f" {self._node_param(index, 1)}),",
                f"        vec2({self._node_param(index, 2)},"
                f" {self._node_param(index, 3)})",
                "    );",
            ]
        elif node.kind == "profile_polygon_2d":
            point_count = len(node.params) // 2
            if point_count < 3:
                body = ["    return 1000000.0;"]
            else:
                body = [
                    "    float distance_to_edge = 1000000.0;",
                    "    bool inside = false;",
                ]
                for point_index in range(point_count):
                    next_index = (point_index + 1) % point_count
                    first = (
                        f"vec2({self._node_param(index, point_index * 2)},"
                        f" {self._node_param(index, point_index * 2 + 1)})"
                    )
                    second = (
                        f"vec2({self._node_param(index, next_index * 2)},"
                        f" {self._node_param(index, next_index * 2 + 1)})"
                    )
                    body.extend(
                        (
                            f"    vec2 a_{point_index} = {first};",
                            f"    vec2 b_{point_index} = {second};",
                            "    distance_to_edge = min("
                            f"distance_to_edge, irSegmentDistance2D(q, a_{point_index}, b_{point_index}));",
                            "    inside = inside != segmentRayCrosses("
                            f"q, a_{point_index}, b_{point_index});",
                        )
                    )
                body.append("    return inside ? -distance_to_edge : distance_to_edge;")
        elif node.kind == "profile_bezier_surface_2d":
            point_count = len(node.params) // 2
            if point_count < 3:
                body = ["    return 1000000.0;"]
            else:
                distance = self._bezier_distance_2d(
                    index,
                    point_count,
                    q_var="q",
                )
                closed = (
                    abs(float(node.params[0]) - float(node.params[-2])) <= 1.0e-12
                    and abs(float(node.params[1]) - float(node.params[-1])) <= 1.0e-12
                )
                if not closed:
                    distance = (
                        f"min({distance}, irSegmentDistance2D(q,"
                        f" {self._point_2d(index, point_count - 1)},"
                        f" {self._point_2d(index, 0)}))"
                    )
                body = [
                    f"    float distance_to_edge = {distance};",
                    "    float crossings = 0.0;",
                ]
                for point_index in range(0, point_count - 2, 2):
                    body.append(
                        "    crossings += quadraticBezierRayCrossValue("
                        f"q, {self._point_2d(index, point_index)},"
                        f" {self._point_2d(index, point_index + 1)},"
                        f" {self._point_2d(index, point_index + 2)});"
                    )
                if not closed:
                    body.append(
                        "    crossings += segmentRayCrossValue("
                        f"q, {self._point_2d(index, point_count - 1)},"
                        f" {self._point_2d(index, 0)});"
                    )
                body.extend(
                    (
                        "    float parity = mod(crossings, 2.0);",
                        "    return (1.0 - 2.0 * parity) * distance_to_edge;",
                    )
                )
        elif node.kind == "profile_offset_2d":
            child = node.children[0]
            body = [
                f"    return {self._profile_sdf_name(child)}("
                f"q - vec2({self._node_param(index, 0)},"
                f" {self._node_param(index, 1)}));"
            ]
        elif node.kind == "profile_union_2d":
            left, right = node.children
            body = [
                f"    float a = {self._profile_sdf_name(left)}(q);",
                f"    float b = {self._profile_sdf_name(right)}(q);",
                "    return min(a, b);",
            ]
        elif node.kind == "profile_intersection_2d":
            left, right = node.children
            body = [
                f"    float a = {self._profile_sdf_name(left)}(q);",
                f"    float b = {self._profile_sdf_name(right)}(q);",
                "    return max(a, b);",
            ]
        elif node.kind == "profile_difference_2d":
            left, right = node.children
            body = [
                f"    float a = {self._profile_sdf_name(left)}(q);",
                f"    float b = {self._profile_sdf_name(right)}(q);",
                "    return max(a, -b);",
            ]
        elif node.kind == "profile_smooth_union_2d":
            left, right = node.children
            body = [
                f"    float a = {self._profile_sdf_name(left)}(q);",
                f"    float b = {self._profile_sdf_name(right)}(q);",
                f"    float k = {self._node_param(index, 0)};",
                "    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);",
                "    return mix(b, a, h) - k * h * (1.0 - h);",
            ]
        else:
            body = ["    return 1000000.0;"]
        return "\n".join((header, *body, "}"))

    def _sdf_function(self, index: int, node: RenderIRNode) -> str:
        header = f"float {self._sdf_name(index)}(vec3 p) {{"
        if node.kind == "sphere":
            body = [
                f"    return length(p - {self._node_vec3(index, 0)})"
                f" - {self._node_param(index, 3)};"
            ]
        elif node.kind == "box":
            body = [
                f"    vec3 local = {self._oriented_local(index)};",
                f"    vec3 q = abs(local) - {self._node_vec3(index, 12)};",
                "    return length(max(q, vec3(0.0)))",
                "        + min(max(q.x, max(q.y, q.z)), 0.0);",
            ]
        elif node.kind == "cylinder":
            body = [
                f"    vec3 local = {self._oriented_local(index)};",
                "    vec2 d = abs(vec2(length(local.xy), local.z))",
                f"        - vec2({self._node_param(index, 12)},"
                f" {self._node_param(index, 13)});",
                "    return min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0)));",
            ]
        elif node.kind == "cone":
            body = [
                f"    vec3 local = {self._oriented_local(index)};",
                f"    float radius = {self._node_param(index, 12)};",
                f"    float half_height = {self._node_param(index, 13)};",
                "    float height = 2.0 * half_height;",
                "    vec2 q = vec2(radius, -height);",
                "    vec2 w = vec2(length(local.xy), local.z - half_height);",
                "    vec2 a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);",
                "    vec2 b = w - q * vec2(clamp(w.x / q.x, 0.0, 1.0), 1.0);",
                "    float d = min(dot(a, a), dot(b, b));",
                "    float s = max(-(w.x * q.y - w.y * q.x), -(w.y - q.y));",
                "    return sqrt(d) * sign(s);",
            ]
        elif node.kind == "capped_cone":
            body = [
                f"    vec3 local = {self._oriented_local(index)};",
                "    return cappedConeSDF(",
                "        local,",
                f"        {self._node_param(index, 14)},",
                f"        {self._node_param(index, 12)},",
                f"        {self._node_param(index, 13)}",
                "    );",
            ]
        elif node.kind == "box_frame":
            body = [
                f"    vec3 local = {self._oriented_local(index)};",
                "    return boxFrameSDF(",
                f"        local, {self._node_vec3(index, 12)},"
                f" {self._node_param(index, 15)}",
                "    );",
            ]
        elif node.kind == "pyramid":
            body = [
                f"    vec3 local = {self._oriented_local(index)};",
                f"    float base_half_size = {self._node_param(index, 12)};",
                f"    float half_height = {self._node_param(index, 13)};",
                "    float scale = 2.0 * base_half_size;",
                "    float height = (2.0 * half_height) / scale;",
                "    vec3 q = vec3(",
                "        local.x / scale,",
                "        (local.z + half_height) / scale,",
                "        local.y / scale",
                "    );",
                "    return scale * pyramidUnitSDF(q, height);",
            ]
        elif node.kind == "torus":
            body = [
                f"    vec3 local = {self._oriented_local(index)};",
                "    return length(vec2(",
                f"        length(local.xy) - {self._node_param(index, 12)},",
                "        local.z",
                f"    )) - {self._node_param(index, 13)};",
            ]
        elif node.kind == "union":
            body = self._binary_sdf_body(index, "min(a, b)")
        elif node.kind == "intersection":
            body = self._binary_sdf_body(index, "max(a, b)")
        elif node.kind == "difference":
            body = self._binary_sdf_body(index, "max(a, -b)")
        elif node.kind == "smooth_union":
            left, right = node.children
            body = [
                f"    float a = {self._sdf_name(left)}(p);",
                f"    float b = {self._sdf_name(right)}(p);",
                f"    float k = {self._node_param(index, 0)};",
                "    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);",
                "    return mix(b, a, h) - k * h * (1.0 - h);",
            ]
        elif node.kind == "polyline_tube":
            body = self._polyline_tube_body(index, node)
        elif node.kind == "bezier_tube":
            body = self._bezier_tube_body(index, node)
        elif node.kind == "placed_circle_2d":
            u, v, plane = self._placed_2d_profile_coordinates(index)
            body = [
                f"    vec2 q = vec2({u}, {v}) - vec2("
                f"{self._node_param(index, 12)}, {self._node_param(index, 13)});",
                f"    float profile = length(q) - {self._node_param(index, 14)};",
                "    return max(profile, abs("
                f"{plane}) - 0.002);",
            ]
        elif node.kind == "placed_rectangle_2d":
            u, v, plane = self._placed_2d_profile_coordinates(index)
            body = [
                f"    vec2 center = vec2({self._node_param(index, 12)},"
                f" {self._node_param(index, 13)});",
                f"    vec2 half_size = vec2({self._node_param(index, 14)},"
                f" {self._node_param(index, 15)});",
                f"    vec2 q = abs(vec2({u}, {v}) - center) - half_size;",
                "    float profile = length(max(q, vec2(0.0)))"
                " + min(max(q.x, q.y), 0.0);",
                "    return max(profile, abs("
                f"{plane}) - 0.002);",
            ]
        elif node.kind == "placed_square_2d":
            u, v, plane = self._placed_2d_profile_coordinates(index)
            body = [
                f"    vec2 center = vec2({self._node_param(index, 12)},"
                f" {self._node_param(index, 13)});",
                f"    float half_size = {self._node_param(index, 14)};",
                f"    vec2 q = abs(vec2({u}, {v}) - center) - vec2(half_size);",
                "    float profile = length(max(q, vec2(0.0)))"
                " + min(max(q.x, q.y), 0.0);",
                "    return max(profile, abs("
                f"{plane}) - 0.002);",
            ]
        elif node.kind == "placed_rounded_rectangle_2d":
            u, v, plane = self._placed_2d_profile_coordinates(index)
            body = [
                f"    vec2 center = vec2({self._node_param(index, 12)},"
                f" {self._node_param(index, 13)});",
                f"    vec2 half_size = vec2({self._node_param(index, 14)},"
                f" {self._node_param(index, 15)});",
                f"    float corner_radius = {self._node_param(index, 16)};",
                "    vec2 inner = half_size - vec2(corner_radius);",
                f"    vec2 q = abs(vec2({u}, {v}) - center) - inner;",
                "    float profile = length(max(q, vec2(0.0)))"
                " + min(max(q.x, q.y), 0.0) - corner_radius;",
                "    return max(profile, abs("
                f"{plane}) - 0.002);",
            ]
        elif node.kind == "placed_ellipse_2d":
            u, v, plane = self._placed_2d_profile_coordinates(index)
            body = [
                "    float profile = exactEllipseDistance(",
                f"        vec2({u}, {v}) - vec2({self._node_param(index, 12)},"
                f" {self._node_param(index, 13)}),",
                f"        vec2({self._node_param(index, 14)},"
                f" {self._node_param(index, 15)})",
                "    );",
                "    return max(profile, abs("
                f"{plane}) - 0.002);",
            ]
        elif node.kind == "placed_profile_2d":
            child = node.children[0]
            u, v, plane = self._placed_2d_profile_coordinates(index)
            body = [
                f"    float profile = {self._profile_sdf_name(child)}("
                f"vec2({u}, {v}));",
                f"    return max(profile, abs({plane}) - 0.002);",
            ]
        elif node.kind == "placed_profile_1d":
            child = node.children[0]
            local = f"(p - {self._node_vec3(index, 0)})"
            t = f"dot({local}, {self._node_vec3(index, 3)})"
            body = [
                f"    float t = {t};",
                f"    vec3 radial = {local} - t * {self._node_vec3(index, 3)};",
                f"    float profile = {self._profile_1d_sdf_name(child)}(t);",
                "    return max(profile, length(radial) - 0.004);",
            ]
        elif node.kind == "placed_polyline_2d":
            point_count = (len(node.params) - 12) // 2
            u, v, plane = self._placed_2d_profile_coordinates(index)
            centerline = self._polyline_distance_2d(
                index,
                point_count,
                q_var="q",
                offset=12,
            )
            body = [
                f"    vec2 q = vec2({u}, {v});",
                f"    float profile = {centerline} - 0.004;",
                f"    return max(profile, abs({plane}) - 0.002);",
            ]
        elif node.kind == "placed_bezier_curve_2d":
            point_count = (len(node.params) - 12) // 2
            u, v, plane = self._placed_2d_profile_coordinates(index)
            centerline = self._bezier_distance_2d(
                index,
                point_count,
                q_var="q",
                offset=12,
            )
            body = [
                f"    vec2 q = vec2({u}, {v});",
                f"    float profile = {centerline} - 0.004;",
                f"    return max(profile, abs({plane}) - 0.002);",
            ]
        elif node.kind == "extrude_profile_2d":
            child = node.children[0]
            u, v, plane = self._placed_2d_profile_coordinates(index)
            body = [
                f"    float profile = {self._profile_sdf_name(child)}("
                f"vec2({u}, {v}));",
                f"    float axial = abs({plane} - {self._node_param(index, 13)})"
                f" - {self._node_param(index, 12)} * 0.5;",
                "    vec2 pair = vec2(profile, axial);",
                "    return length(max(pair, vec2(0.0)))"
                " + min(max(profile, axial), 0.0);",
            ]
        elif node.kind == "revolve_profile_2d":
            child = node.children[0]
            axis_origin = self._node_vec3(index, 12)
            axis_direction = self._node_vec3(index, 15)
            radial_direction = self._node_vec3(index, 18)
            tangent_direction = self._node_vec3(index, 21)
            local = f"(p - {axis_origin})"
            axial = f"dot({local}, {axis_direction})"
            radial_x = f"dot({local}, {radial_direction})"
            radial_y = f"dot({local}, {tangent_direction})"
            radial = (
                f"sqrt(max(({radial_x}) * ({radial_x})"
                f" + ({radial_y}) * ({radial_y}), 0.0))"
            )
            sample_point = (
                f"({axis_origin}"
                f" + ({axial}) * {axis_direction}"
                f" + ({radial}) * {radial_direction})"
            )
            section_local = f"({sample_point} - {self._node_vec3(index, 0)})"
            section_u = f"dot({section_local}, {self._node_vec3(index, 3)})"
            section_v = f"dot({section_local}, {self._node_vec3(index, 6)})"
            body = [
                f"    float profile = {self._profile_sdf_name(child)}("
                f"vec2({section_u}, {section_v}));",
                f"    if ({self._node_param(index, 24)} >= 6.283184307179586)"
                " return profile;",
                "    float angular = angularSectorSDF(",
                f"        vec2({radial_x}, {radial_y}), {self._node_param(index, 24)}",
                "    );",
                "    vec2 pair = vec2(profile, angular);",
                "    return length(max(pair, vec2(0.0)))"
                " + min(max(profile, angular), 0.0);",
            ]
        else:
            body = ["    return 1000000.0;"]
        return "\n".join((header, *body, "}"))

    def _binary_sdf_body(self, index: int, expression: str) -> list[str]:
        left, right = self.render_ir.nodes[index].children
        return [
            f"    float a = {self._sdf_name(left)}(p);",
            f"    float b = {self._sdf_name(right)}(p);",
            f"    return {expression};",
        ]

    def _polyline_tube_body(self, index: int, node: RenderIRNode) -> list[str]:
        point_count = (len(node.params) - 3) // 3
        radius = self._node_param(index, len(node.params) - 3)
        inner_radius = self._node_param(index, len(node.params) - 2)
        centerline = self._polyline_centerline(index, point_count)
        if node.params[-1] <= 0.5:
            return [
                f"    float centerline = {centerline};",
                f"    return irTubeSDF(centerline, {radius}, {inner_radius});",
            ]
        distances = [
            "flatCappedSegmentTubeSDF3D("
            f"p, {self._point(index, segment)},"
            f" {self._point(index, segment + 1)}, {radius})"
            for segment in range(point_count - 1)
        ]
        return [
            f"    float outer = {self._minimum(distances)};",
            f"    float centerline = {centerline};",
            f"    return irFlatTubeSDF(outer, centerline, {inner_radius});",
        ]

    def _bezier_tube_body(self, index: int, node: RenderIRNode) -> list[str]:
        point_count = (len(node.params) - 3) // 3
        radius = self._node_param(index, len(node.params) - 3)
        inner_radius = self._node_param(index, len(node.params) - 2)
        centerline = self._bezier_centerline(index, point_count)
        if node.params[-1] <= 0.5:
            return [
                f"    float centerline = {centerline};",
                f"    return irTubeSDF(centerline, {radius}, {inner_radius});",
            ]
        start = self._point(index, 0)
        first_control = self._point(index, 1)
        first_end = self._point(index, 2)
        last_start = self._point(index, point_count - 3)
        last_control = self._point(index, point_count - 2)
        end = self._point(index, point_count - 1)
        return [
            f"    float centerline = {centerline};",
            f"    vec3 start_tangent = irSafeDirection({first_control} - {start},"
            f" {first_end} - {start});",
            f"    vec3 end_tangent = irSafeDirection({end} - {last_control},"
            f" {end} - {last_start});",
            f"    float start_plane = dot({start} - p, start_tangent);",
            f"    float end_plane = dot(p - {end}, end_tangent);",
            "    float outer = max(max(centerline"
            f" - {radius}, start_plane), end_plane);",
            f"    return irFlatTubeSDF(outer, centerline, {inner_radius});",
        ]

    def _node_dispatch_sdf_function(self) -> str:
        branches = "\n".join(
            f"    if (node_index == {index}) return {self._sdf_name(index)}(p);"
            for index in range(len(self.render_ir.nodes))
        )
        return (
            "float irSceneNodeSDFByIndex(int node_index, vec3 p) {\n"
            f"{branches}\n"
            "    return 1000000.0;\n"
            "}"
        )

    def _scene_contract(self) -> str:
        root = self.render_ir.root_indices[0]
        scene_node_indices = tuple(
            dict.fromkeys(
                ref.node_index
                for ref in self.render_ir.material_refs
                if 0 <= ref.node_index < len(self.render_ir.nodes)
            )
        ) or (root,)
        scene_distance = f"{self._sdf_name(scene_node_indices[0])}(p)"
        for node_index in scene_node_indices[1:]:
            scene_distance = (
                f"min({scene_distance}, {self._sdf_name(node_index)}(p))"
            )
        return (
            "float sceneSDF(vec3 p) {\n"
            f"    return {scene_distance};\n"
            "}\n"
            "int sceneObjectId(vec3 p) {\n"
            "    float best_distance = 1000000.0;\n"
            "    int object_id = 0;\n"
            f"    for (int index = 0; index < {IR_MAX_COMPONENTS}; ++index) {{\n"
            "        if (index >= u_scene_material_count) break;\n"
            "        int node_index = u_scene_material_node_indices[index];\n"
            "        float material_distance = abs(irSceneNodeSDFByIndex(node_index, p));\n"
            "        if (material_distance < best_distance) {\n"
            "            best_distance = material_distance;\n"
            "            object_id = u_scene_material_object_ids[index];\n"
            "        }\n"
            "    }\n"
            "    return object_id;\n"
            "}"
        )


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
        self._parameter_program_cache: dict[
            tuple[object, ...], _ParameterizedProgramCacheEntry
        ] = {}
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

    def _remember_parameter_program(
        self,
        topology_signature: tuple[object, ...],
        program: moderngl.Program,
        vao: moderngl.VertexArray,
        offsets_by_object: dict[tuple[int, str, int], int],
    ) -> None:
        self._parameter_program_cache.pop(topology_signature, None)
        self._parameter_program_cache[topology_signature] = (
            _ParameterizedProgramCacheEntry(
                program=program,
                vao=vao,
                offsets_by_object=offsets_by_object,
            )
        )
        while len(self._parameter_program_cache) > IR_PARAMETER_PROGRAM_CACHE_LIMIT:
            oldest_signature = next(iter(self._parameter_program_cache))
            if oldest_signature in {
                self._scene_layer.topology_signature,
                self._preview_layer.topology_signature,
            }:
                active_entry = self._parameter_program_cache.pop(oldest_signature)
                self._parameter_program_cache[oldest_signature] = active_entry
                continue
            evicted = self._parameter_program_cache.pop(oldest_signature)
            evicted.vao.release()
            evicted.program.release()

    def _scene_fragment_source(
        self,
        scene_function: str,
        *,
        preview_layer: bool,
    ) -> str:
        fragment_template = (self._shader_dir / "raymarch.frag").read_text(
            encoding="utf-8"
        )
        prefix, _suffix = fragment_template.split("/*__SCENE_SDF__*/", 1)
        suffix_name = (
            "raymarch_fast_scene.glsl"
            if preview_layer
            else "raymarch_static_scene.glsl"
        )
        suffix = (self._shader_dir / suffix_name).read_text(encoding="utf-8")
        return f"{prefix}{scene_function}\n{suffix}"

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
        self._last_scene_update_stats = None
        try:
            return self._upload_parameterized_render_ir(
                self._scene_layer,
                render_ir,
                record_stats=True,
                preview_layer=False,
            )
        except moderngl.Error:
            self._last_scene_update_stats = None
            return False

    def upload_preview_render_ir(self, render_ir: RenderIR | None) -> bool:
        if render_ir is None:
            self.clear_preview_render_ir()
            return True
        try:
            return self._upload_parameterized_render_ir(
                self._preview_layer,
                render_ir,
                record_stats=False,
                preview_layer=True,
            )
        except moderngl.Error:
            self.clear_preview_render_ir()
            return False

    def clear_preview_render_ir(self) -> None:
        self._preview_layer = _RenderIRLayerState()

    def _upload_parameterized_render_ir(
        self,
        layer: _RenderIRLayerState,
        render_ir: RenderIR | None,
        *,
        record_stats: bool,
        preview_layer: bool,
    ) -> bool:
        if (
            render_ir is None
            or not render_ir.supported
            or len(render_ir.nodes) > IR_MAX_NODES
            or len(render_ir.component_indices) > IR_MAX_COMPONENTS
            or len(render_ir.root_indices) != 1
        ):
            return False
        params = render_ir.parameter_values
        if len(params) > IR_PARAMETER_MAX_FLOATS:
            return False
        topology_signature = (
            "preview" if preview_layer else "scene",
            render_ir.topology_signature,
        )
        total_start = perf_counter()
        cached_entry = self._parameter_program_cache.get(topology_signature)
        reused_program = (
            layer.program is not None
            and layer.vao is not None
            and layer.topology_signature == topology_signature
        )
        if not reused_program and cached_entry is not None:
            layer.program = cached_entry.program
            layer.vao = cached_entry.vao
            layer.topology_signature = topology_signature
            layer.offsets_by_object = dict(cached_entry.offsets_by_object)
            self._remember_parameter_program(
                topology_signature,
                cached_entry.program,
                cached_entry.vao,
                cached_entry.offsets_by_object,
            )
            reused_program = True
        shader_build_ms = 0.0
        program_compile_ms = 0.0
        vao_build_ms = 0.0
        if (
            layer.program is None
            or layer.vao is None
            or layer.topology_signature != topology_signature
        ):
            shader_build_start = perf_counter()
            vertex_source = (self._shader_dir / "raymarch.vert").read_text(
                encoding="utf-8"
            )
            source_builder = _ParameterizedIRSceneSource(render_ir)
            scene_source = source_builder.build()
            fragment_source = self._scene_fragment_source(
                scene_source,
                preview_layer=preview_layer,
            )
            shader_build_ms = (perf_counter() - shader_build_start) * 1000.0
            program_compile_start = perf_counter()
            program = self.context.program(
                vertex_shader=vertex_source,
                fragment_shader=fragment_source,
            )
            program_compile_ms = (perf_counter() - program_compile_start) * 1000.0
            vao_build_start = perf_counter()
            vao = self.context.vertex_array(
                program,
                [(self._vertex_buffer, "2f", "in_position")],
            )
            vao_build_ms = (perf_counter() - vao_build_start) * 1000.0
            if "SceneParamData" in program:
                program["SceneParamData"].binding = IR_PARAM_DATA_BINDING
            layer.program = program
            layer.vao = vao
            layer.topology_signature = topology_signature
            offsets_by_object = self._parameter_object_offsets(
                render_ir,
                source_builder.param_offsets,
            )
            layer.offsets_by_object = offsets_by_object
            self._remember_parameter_program(
                topology_signature,
                program,
                vao,
                offsets_by_object,
            )
        layer.values = params
        layer.render_ir = render_ir
        if record_stats:
            self._last_scene_update_stats = SceneUpdateStats(
                path="render_ir_upload",
                total_ms=(perf_counter() - total_start) * 1000.0,
                shader_build_ms=shader_build_ms,
                program_compile_ms=program_compile_ms,
                vao_build_ms=vao_build_ms,
                render_ir_nodes=len(render_ir.nodes),
                reused_program=reused_program,
            )
        return True

    @staticmethod
    def _parameter_object_offsets(
        render_ir: RenderIR,
        param_offsets: dict[int, int],
    ) -> dict[tuple[int, str, int], int]:
        offsets: dict[tuple[int, str, int], int] = {}
        for index, node in enumerate(render_ir.nodes):
            if node.object_id <= 0 or not node.params:
                continue
            offsets[(int(node.object_id), node.kind, len(node.params))] = (
                param_offsets[index]
            )
        return offsets

    def update_render_ir_object_parameters(
        self,
        render_ir: RenderIR | None,
        object_ids: tuple[int, ...],
    ) -> bool:
        total_start = perf_counter()
        if (
            render_ir is None
            or self._scene_layer.program is None
            or self._scene_layer.vao is None
            or not self._scene_layer.values
            or self._scene_layer.offsets_by_object is None
        ):
            return False
        targets = {int(object_id) for object_id in object_ids if object_id > 0}
        if not targets:
            return False
        values = list(self._scene_layer.values)
        updated: set[int] = set()
        for node in render_ir.nodes:
            object_id = int(node.object_id)
            if object_id not in targets or not node.params:
                continue
            key = (object_id, node.kind, len(node.params))
            offset = self._scene_layer.offsets_by_object.get(key)
            if offset is None or offset + len(node.params) > len(values):
                return False
            values[offset:offset + len(node.params)] = node.params
            updated.add(object_id)
        if not targets.issubset(updated):
            return False
        self._scene_layer.values = tuple(values)
        self._write_parameter_values(self._scene_layer.values)
        self._last_scene_update_stats = SceneUpdateStats(
            path="render_ir_parameter_update",
            total_ms=(perf_counter() - total_start) * 1000.0,
            shader_build_ms=0.0,
            program_compile_ms=0.0,
            vao_build_ms=0.0,
            render_ir_nodes=len(render_ir.nodes),
            reused_program=True,
        )
        return True

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
            }
            if scene_program is not None and scene_vao is not None:
                for name, value in uniform_values.items():
                    if name in scene_program:
                        scene_program[name].value = value
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
        for entry in self._parameter_program_cache.values():
            release_vao(entry.vao)
            release_program(entry.program)
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
