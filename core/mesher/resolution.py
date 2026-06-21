from __future__ import annotations

from core.sdf.base import SDFNode
from core.sdf.operators import BinarySDFOperator, SmoothUnion
from core.sdf.placed_1d import PlacedSDF1D
from core.sdf.placed_2d import PlacedSDF2D
from core.sdf.primitives_3d import Box, BoxFrame, CappedCone, Cone, Cylinder, Pyramid, Sphere, Torus
from core.sdf.solid_from_2d import Extrude, Revolve
from core.sdf.transforms import Scale, UnaryTransform
from core.sdf.tubes import BezierTube, PolylineTube

FEATURE_INTERVALS = 6.0


def _profile_feature_size(section: PlacedSDF2D) -> float:
    assert section.profile is not None
    u_min, u_max, v_min, v_max = section.profile.bounds()
    return min(u_max - u_min, v_max - v_min)


def minimum_feature_size(node: SDFNode) -> float:
    """Return a conservative parameter-based geometric feature estimate."""
    if isinstance(node, PlacedSDF1D):
        assert node.profile is not None
        minimum, maximum = node.profile.bounds()
        return maximum - minimum
    if isinstance(node, PlacedSDF2D):
        return _profile_feature_size(node)
    if isinstance(node, Sphere):
        return 2.0 * node.radius
    if isinstance(node, Box):
        return 2.0 * min(node.half_size)
    if isinstance(node, BoxFrame):
        return node.thickness
    if isinstance(node, Cylinder):
        return 2.0 * min(node.radius, node.half_height)
    if isinstance(node, CappedCone):
        return 2.0 * min(node.radius_a, node.radius_b, node.half_height)
    if isinstance(node, Cone):
        return 2.0 * min(node.radius, node.half_height)
    if isinstance(node, Pyramid):
        return 2.0 * min(node.base_half_size, node.half_height)
    if isinstance(node, Torus):
        return 2.0 * node.minor_radius
    if isinstance(node, Scale):
        assert node.child is not None
        return node.factor * minimum_feature_size(node.child)
    if isinstance(node, UnaryTransform):
        assert node.child is not None
        return minimum_feature_size(node.child)
    if isinstance(node, BinarySDFOperator):
        assert node.left is not None and node.right is not None
        feature = min(
            minimum_feature_size(node.left),
            minimum_feature_size(node.right),
        )
        if isinstance(node, SmoothUnion):
            feature = min(feature, 2.0 * node.smoothing)
        return feature
    if isinstance(node, Extrude):
        assert node.section is not None
        return min(node.height, _profile_feature_size(node.section))
    if isinstance(node, (PolylineTube, BezierTube)):
        wall = (
            node.radius - node.inner_radius
            if node.inner_radius > 0.0
            else 2.0 * node.radius
        )
        return min(2.0 * node.radius, wall)
    if isinstance(node, Revolve):
        assert node.section is not None
        return _profile_feature_size(node.section)
    box = node.bounding_box()
    spans = (
        box.x_max - box.x_min,
        box.y_max - box.y_min,
        box.z_max - box.z_min,
    )
    positive_spans = [span for span in spans if span > 0.0]
    if not positive_spans:
        raise ValueError("cannot estimate dx for zero-size geometry")
    return min(positive_spans)


def recommended_max_dx(root: SDFNode) -> float:
    """Recommend at least six lattice intervals across the smallest feature."""
    return minimum_feature_size(root) / FEATURE_INTERVALS
