from __future__ import annotations

from app.viewport.renderer import (
    ROTATION_GIZMO_SEGMENTS,
    X_AXIS_COLOR,
    Y_AXIS_COLOR,
    WORLD_AXIS_LENGTH,
    Z_AXIS_COLOR,
    SDFRenderer,
)
from app.viewport.renderer import (
    MAX_SELECTED_BOUNDARY_OWNERS,
    POINT_VERTEX_WIDTH,
    SQUARE_INSTANCE_WIDTH,
)


class OpenGLRenderer(SDFRenderer):
    """OpenGL backend adapter.

    For now, this adapter points at the existing renderer implementation to keep
    behavior exact while formalizing the backend boundary for future backends.
    """


__all__ = [
    "OpenGLRenderer",
    "ROTATION_GIZMO_SEGMENTS",
    "X_AXIS_COLOR",
    "Y_AXIS_COLOR",
    "WORLD_AXIS_LENGTH",
    "Z_AXIS_COLOR",
    "MAX_SELECTED_BOUNDARY_OWNERS",
    "POINT_VERTEX_WIDTH",
    "SQUARE_INSTANCE_WIDTH",
]
