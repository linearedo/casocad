from __future__ import annotations

from dataclasses import dataclass, field

from .base import FloatArray, SDFNode


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

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        return self.root.to_numpy(X, Y, Z)
