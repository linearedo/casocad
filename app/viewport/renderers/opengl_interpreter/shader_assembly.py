from __future__ import annotations

"""Assemble the interpreter GLSL from feature modules (design §13.3).

One codebase, sized per GPU: the core chunk is always included; profiles, sweeps
and selectors are appended only when their feature is enabled. The host prepends
the generated node-type/opcode/capacity defines and the FEATURE_* block.
"""

from pathlib import Path

from core.gpu_features import CULL, PROFILES, SELECTORS, SWEEPS, emit_feature_defines
from core.gpu_node_types import emit_glsl_defines

_DIR = Path(__file__).parent / "shaders"


def _read(name: str) -> str:
    return (_DIR / name).read_text()


def interpreter_chunks(
    features: frozenset[str],
    *,
    stack_capacity: int | None = None,
) -> str:
    """Concatenated #defines + core + the enabled optional feature chunks.

    ``stack_capacity`` overrides both value-stack caps (the fragment backend uses
    a reduced cap; compute/validation keeps the generated 64).
    """

    defines = emit_glsl_defines()
    if stack_capacity is not None:
        defines = defines.replace(
            "#define IR_STACK_CAPACITY 64",
            f"#define IR_STACK_CAPACITY {stack_capacity}",
        ).replace(
            "#define IR_PROFILE_STACK_CAPACITY 64",
            f"#define IR_PROFILE_STACK_CAPACITY {stack_capacity}",
        )

    parts = [defines, emit_feature_defines(features), _read("sdf_core.glsl")]
    if PROFILES in features:
        parts.append(_read("sdf_profiles.glsl"))
    if SWEEPS in features:
        parts.append(_read("sdf_sweeps.glsl"))
    if SELECTORS in features:
        parts.append(_read("sdf_selectors.glsl"))
    if CULL in features:
        parts.append(_read("sdf_cull.glsl"))
    return "\n".join(parts)


def build_program_source(
    features: frozenset[str],
    main_chunk: str,
    *,
    stack_capacity: int | None = None,
    version: str = "#version 460",
) -> str:
    """Full shader source: version + interpreter chunks + a stage main chunk."""

    return "\n".join(
        (version, "", interpreter_chunks(features, stack_capacity=stack_capacity), main_chunk)
    )


__all__ = ["interpreter_chunks", "build_program_source"]
