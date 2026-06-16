from __future__ import annotations

from dataclasses import dataclass, field

from .base import FloatArray, SDFNode, glsl_float, glsl_vec3
from .csg import Difference, Intersection, SmoothUnion, Union
from .transforms import Rotate, Scale, Translate


@dataclass
class SDFTree:
    root: SDFNode
    components: tuple[SDFNode, ...] = ()
    _nodes: dict[int, SDFNode] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        nodes = (
            *self._walk(self.root),
            *(
                descendant
                for component in self.components
                for descendant in self._walk(component)
            ),
        )
        for node in (item for item in nodes if item.object_id > 0):
            self.register(node)
        for node in (item for item in nodes if item.object_id == 0):
            self.register(node)

    def _walk(self, node: SDFNode) -> tuple[SDFNode, ...]:
        return (node, *(
            descendant
            for child in node.children()
            for descendant in self._walk(child)
        ))

    def register(self, node: SDFNode) -> int:
        if node.object_id == 0:
            node.object_id = max(self._nodes, default=0) + 1
        if not 1 <= node.object_id <= 65_535:
            raise ValueError("object_id must be in the range 1..65535")
        if node.object_id in self._nodes and self._nodes[node.object_id] is not node:
            raise ValueError(f"duplicate object_id {node.object_id}")
        self._nodes[node.object_id] = node
        return node.object_id

    @property
    def nodes(self) -> tuple[SDFNode, ...]:
        return tuple(self._nodes[key] for key in sorted(self._nodes))

    def to_glsl(self) -> str:
        materials = tuple(
            node for node in self.components if node.dimension == 3
        ) or (self.root,)
        branches = "\n".join(
            (
                f"    float material_distance_{index} = "
                f"abs({node.to_glsl('p')});\n"
                f"    if (material_distance_{index} < best_distance) {{\n"
                f"        best_distance = material_distance_{index};\n"
                f"        object_id = {node.object_id};\n"
                "    }"
            )
            for index, node in enumerate(materials)
        )
        selection_branches = "\n".join(
            (
                f"    if (selected_object_id == {node.object_id}) "
                f"return {self._selection_owner_glsl(node)};"
            )
            for node in self.nodes
            if node.dimension == 3
        )
        selected_sdf_branches = "\n".join(
            (
                f"    if (selected_object_id == {node.object_id}) "
                f"return {node.to_glsl('p')};"
            )
            for node in self.nodes
        )
        selected_dimension_branches = "\n".join(
            (
                f"    if (selected_object_id == {node.object_id}) "
                f"return {node.dimension};"
            )
            for node in self.nodes
        )
        return (
            f"float sceneSDF(vec3 p) {{\n    return {self.root.to_glsl('p')};\n}}\n"
            "float sceneMovedSDF(\n"
            "    vec3 p,\n"
            "    int selected_object_id,\n"
            "    vec3 preview_offset\n"
            ") {\n"
            f"    return {self._moved_sdf_glsl(self.root, 'p')};\n"
            "}\n"
            f"int sceneBoundaryOwnerId(vec3 p) {{\n"
            f"    return {self._boundary_owner_glsl(self.root, 'p')};\n"
            "}\n"
            "int sceneObjectId(vec3 p) {\n"
            "    float best_distance = 1000000.0;\n"
            "    int object_id = 0;\n"
            f"{branches}\n"
            "    return object_id;\n"
            "}\n"
            "bool sceneSelectionOwnsBoundary(\n"
            "    int selected_object_id,\n"
            "    int boundary_owner_id\n"
            ") {\n"
            f"{selection_branches}\n"
            "    return false;\n"
            "}\n"
            "float sceneSelectedObjectSDF(vec3 p, int selected_object_id) {\n"
            f"{selected_sdf_branches}\n"
            "    return 1000000.0;\n"
            "}\n"
            "int sceneSelectedObjectDimension(int selected_object_id) {\n"
            f"{selected_dimension_branches}\n"
            "    return 0;\n"
            "}"
        )

    def _moved_sdf_glsl(self, node: SDFNode, p_var: str) -> str:
        moved_point = f"({p_var} - preview_offset)"
        if isinstance(node, Translate):
            assert node.child is not None
            child_point = f"({p_var} - {glsl_vec3(node.offset)})"
            moved_child_point = f"({moved_point} - {glsl_vec3(node.offset)})"
            normal = self._moved_sdf_glsl(node.child, child_point)
            moved = self._moved_sdf_glsl(node.child, moved_child_point)
            return (
                f"(selected_object_id == {node.object_id}"
                f" ? {moved} : {normal})"
            )
        if isinstance(node, Scale):
            assert node.child is not None
            factor = glsl_float(node.factor)
            child_point = f"({p_var} / {factor})"
            moved_child_point = f"({moved_point} / {factor})"
            normal = f"({self._moved_sdf_glsl(node.child, child_point)} * {factor})"
            moved = f"({self._moved_sdf_glsl(node.child, moved_child_point)} * {factor})"
            return (
                f"(selected_object_id == {node.object_id}"
                f" ? {moved} : {normal})"
            )
        if isinstance(node, Rotate):
            assert node.child is not None
            components = node._inverse_components(
                f"{p_var}.x", f"{p_var}.y", f"{p_var}.z"
            )
            moved_components = node._inverse_components(
                f"{moved_point}.x", f"{moved_point}.y", f"{moved_point}.z"
            )
            normal = self._moved_sdf_glsl(
                node.child,
                f"vec3({', '.join(components)})",
            )
            moved = self._moved_sdf_glsl(
                node.child,
                f"vec3({', '.join(moved_components)})",
            )
            return (
                f"(selected_object_id == {node.object_id}"
                f" ? {moved} : {normal})"
            )
        if isinstance(node, (Union, Intersection, Difference, SmoothUnion)):
            assert node.left is not None and node.right is not None
            if isinstance(node, Union):
                normal = (
                    f"min({self._moved_sdf_glsl(node.left, p_var)}, "
                    f"{self._moved_sdf_glsl(node.right, p_var)})"
                )
                moved = f"min({node.left.to_glsl(moved_point)}, {node.right.to_glsl(moved_point)})"
            elif isinstance(node, Intersection):
                normal = (
                    f"max({self._moved_sdf_glsl(node.left, p_var)}, "
                    f"{self._moved_sdf_glsl(node.right, p_var)})"
                )
                moved = f"max({node.left.to_glsl(moved_point)}, {node.right.to_glsl(moved_point)})"
            elif isinstance(node, Difference):
                normal = (
                    f"max({self._moved_sdf_glsl(node.left, p_var)}, "
                    f"-({self._moved_sdf_glsl(node.right, p_var)}))"
                )
                moved = f"max({node.left.to_glsl(moved_point)}, -({node.right.to_glsl(moved_point)}))"
            else:
                smoothing = glsl_float(node.smoothing)
                left = self._moved_sdf_glsl(node.left, p_var)
                right = self._moved_sdf_glsl(node.right, p_var)
                h = (
                    f"clamp(0.5 + 0.5 * (({right}) - ({left}))"
                    f" / {smoothing}, 0.0, 1.0)"
                )
                normal = (
                    f"(mix(({right}), ({left}), {h})"
                    f" - {smoothing} * {h} * (1.0 - {h}))"
                )
                moved_left = node.left.to_glsl(moved_point)
                moved_right = node.right.to_glsl(moved_point)
                moved_h = (
                    f"clamp(0.5 + 0.5 * (({moved_right}) - ({moved_left}))"
                    f" / {smoothing}, 0.0, 1.0)"
                )
                moved = (
                    f"(mix(({moved_right}), ({moved_left}), {moved_h})"
                    f" - {smoothing} * {moved_h} * (1.0 - {moved_h}))"
                )
            return (
                f"(selected_object_id == {node.object_id}"
                f" ? {moved} : {normal})"
            )
        return (
            f"(selected_object_id == {node.object_id}"
            f" ? {node.to_glsl(moved_point)} : {node.to_glsl(p_var)})"
        )

    def _selection_owner_glsl(self, node: SDFNode) -> str:
        owner_ids = sorted(self._boundary_owner_ids(node))
        return " || ".join(
            f"boundary_owner_id == {object_id}" for object_id in owner_ids
        )

    def _boundary_owner_ids(self, node: SDFNode) -> set[int]:
        if isinstance(node, (Translate, Rotate, Scale)):
            assert node.child is not None
            return self._boundary_owner_ids(node.child)
        if isinstance(node, (Union, Intersection, Difference, SmoothUnion)):
            assert node.left is not None and node.right is not None
            return self._boundary_owner_ids(node.left) | self._boundary_owner_ids(
                node.right
            )
        return {node.object_id}

    def _boundary_owner_glsl(self, node: SDFNode, p_var: str) -> str:
        if isinstance(node, (Translate, Rotate, Scale)):
            assert node.child is not None
            if isinstance(node, Translate):
                child_point = f"({p_var} - {glsl_vec3(node.offset)})"
            elif isinstance(node, Scale):
                child_point = f"({p_var} / {glsl_float(node.factor)})"
            else:
                components = node._inverse_components(
                    f"{p_var}.x", f"{p_var}.y", f"{p_var}.z"
                )
                child_point = f"vec3({', '.join(components)})"
            return self._boundary_owner_glsl(node.child, child_point)
        if isinstance(node, (Union, Intersection, Difference, SmoothUnion)):
            assert node.left is not None and node.right is not None
            left_distance = node.left.to_glsl(p_var)
            right_distance = node.right.to_glsl(p_var)
            left_owner = self._boundary_owner_glsl(node.left, p_var)
            right_owner = self._boundary_owner_glsl(node.right, p_var)
            if isinstance(node, Union):
                condition = f"(({left_distance}) <= ({right_distance}))"
            elif isinstance(node, Intersection):
                condition = f"(({left_distance}) >= ({right_distance}))"
            elif isinstance(node, Difference):
                condition = f"(({left_distance}) >= -({right_distance}))"
            else:
                smoothing = glsl_float(node.smoothing)
                blend = (
                    f"clamp(0.5 + 0.5 * (({right_distance})"
                    f" - ({left_distance})) / {smoothing}, 0.0, 1.0)"
                )
                condition = f"(({blend}) >= 0.5)"
            return f"({condition} ? {left_owner} : {right_owner})"
        return str(node.object_id)

    def _material_nodes(self, node: SDFNode) -> tuple[SDFNode, ...]:
        if isinstance(node, (Union, Intersection, Difference, SmoothUnion)):
            assert node.left is not None and node.right is not None
            return (
                *self._material_nodes(node.left),
                *self._material_nodes(node.right),
            )
        if isinstance(node, (Translate, Rotate, Scale)):
            return (node,)
        return (node,) if node.dimension == 3 else ()

    def components_to_glsl(self, max_components: int = 32) -> str:
        candidates = self.components
        unique: list[SDFNode] = []
        seen: set[int] = set()
        for node in candidates:
            if id(node) in seen:
                continue
            seen.add(id(node))
            unique.append(node)
        nodes = tuple(unique[:max_components])
        branches = "\n".join(
            f"    if (component == {index}) return {node.to_glsl('p')};"
            for index, node in enumerate(nodes)
        )
        object_id_branches = "\n".join(
            f"    if (component == {index}) return {node.object_id};"
            for index, node in enumerate(nodes)
        )
        return (
            f"const int COMPONENT_COUNT = {len(nodes)};\n"
            "float componentSDF(vec3 p, int component) {\n"
            f"{branches}\n"
            "    return 1000000.0;\n"
            "}\n"
            "int componentObjectId(int component) {\n"
            f"{object_id_branches}\n"
            "    return 0;\n"
            "}"
        )

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        return self.root.to_numpy(X, Y, Z)
