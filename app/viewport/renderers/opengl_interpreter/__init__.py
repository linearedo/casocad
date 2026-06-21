"""OpenGL SDF interpreter backend (design §7.1).

A fixed interpreter shader driven by GPU node/param/child/bytecode buffers, so a
topology change is a buffer upload rather than a per-scene GLSL recompile.
"""

from .renderer import InterpreterRenderer, InterpreterUpdateStats
from .scene_buffers import SceneBuffers
from .sdf_evaluator import SdfEvaluator, SdfEvalResult

__all__ = [
    "SdfEvaluator",
    "SdfEvalResult",
    "InterpreterRenderer",
    "InterpreterUpdateStats",
    "SceneBuffers",
]
