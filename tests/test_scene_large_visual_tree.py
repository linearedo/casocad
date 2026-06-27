from __future__ import annotations

from core.scene import SceneDocument
from core.sdf import Box, SDFNode, Union


def _union_depth(node: SDFNode) -> int:
    if not isinstance(node, Union):
        return 1
    assert node.left is not None
    assert node.right is not None
    return 1 + max(_union_depth(node.left), _union_depth(node.right))


def test_large_visual_snapshot_uses_balanced_union_tree() -> None:
    document = SceneDocument(
        objects=[
            Box(
                name=f"box_{idx}",
                object_id=idx + 1,
                center=(float(idx), 0.0, 0.0),
                half_size=(0.1, 0.1, 0.1),
            )
            for idx in range(1000)
        ]
    )

    _version, tree = document.visual_snapshot()

    assert tree is not None
    assert len(tree.components) == 1000
    assert _union_depth(tree.root) <= 12
