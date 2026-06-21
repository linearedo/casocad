from __future__ import annotations

"""Interpreter feature tiers (design §13.3).

The interpreter shader is one codebase assembled per-GPU from feature modules:
the core (primitives + SDF operators + the value-stack VM) is always present;
profiles, sweeps and selectors are optional chunks a weak driver may omit
because it cannot compile them (Mesa Intel hangs on the full shader; §13.2).

This module is backend-agnostic — it defines the feature set, the
``kind -> required feature`` map, and the GLSL ``#define`` block for a chosen
feature set, so the host can decide whether a scene is renderable on the active
tier without importing anything GL-specific.
"""

from .render_ir import RenderIR

# Optional feature flags (the core — primitives + operators — is implicit).
PROFILES = "profiles"    # placed-2D sections, extrude/revolve, profile sub-VMs
SWEEPS = "sweeps"        # polyline / bezier tubes
SELECTORS = "selectors"  # Layer 2 region selectors (evalSubtreeSDF)

# Optional features that gate which node KINDS a tier can render.
OPTIONAL_FEATURES = (PROFILES, SWEEPS, SELECTORS)

# CULL is a rendering optimization (world-grid DDA), not a node-kind gate, so it
# is NOT in OPTIONAL_FEATURES (it never affects scene_fits_tier). It only adds
# the sdf_cull.glsl chunk + FEATURE_CULL define to the assembled shader.
CULL = "cull"
_ASSEMBLY_FEATURES = (PROFILES, SWEEPS, SELECTORS, CULL)

# Tier presets.
LEAN_FEATURES: frozenset[str] = frozenset()                  # core only
FULL_FEATURES: frozenset[str] = frozenset(OPTIONAL_FEATURES)

# Which optional feature each node kind needs (absent => core, always available).
_KIND_FEATURE: dict[str, str] = {
    # placed 2D sections + sweeps-by-profile + profile sub-graph nodes
    "placed_circle_2d": PROFILES,
    "placed_rectangle_2d": PROFILES,
    "placed_square_2d": PROFILES,
    "placed_rounded_rectangle_2d": PROFILES,
    "placed_ellipse_2d": PROFILES,
    "placed_profile_2d": PROFILES,
    "placed_polyline_2d": PROFILES,
    "placed_bezier_curve_2d": PROFILES,
    "placed_profile_1d": PROFILES,
    "extrude_profile_2d": PROFILES,
    "revolve_profile_2d": PROFILES,
    # tubes
    "polyline_tube": SWEEPS,
    "bezier_tube": SWEEPS,
    # Layer 2
    "region_selector": SELECTORS,
}


def required_feature(kind: str) -> str | None:
    """Optional feature a node kind needs, or ``None`` if it is core."""

    if kind.startswith("profile_"):  # profile sub-graph nodes ride with PROFILES
        return PROFILES
    return _KIND_FEATURE.get(kind)


def features_for_render_ir(render_ir: RenderIR | None) -> set[str]:
    """The optional features a scene actually uses."""

    if render_ir is None:
        return set()
    needed: set[str] = set()
    for node in render_ir.nodes:
        feature = required_feature(node.kind)
        if feature is not None:
            needed.add(feature)
    return needed


def scene_fits_tier(render_ir: RenderIR | None, features: frozenset[str]) -> bool:
    """True if every node in the scene is supported by ``features``."""

    return features_for_render_ir(render_ir).issubset(features)


def emit_feature_defines(features: frozenset[str]) -> str:
    """GLSL ``#define FEATURE_*`` block for the chosen optional features."""

    lines = ["// Feature tier (core/gpu_features.py)."]
    for feature in _ASSEMBLY_FEATURES:
        if feature in features:
            lines.append(f"#define FEATURE_{feature.upper()}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "PROFILES",
    "SWEEPS",
    "SELECTORS",
    "CULL",
    "OPTIONAL_FEATURES",
    "LEAN_FEATURES",
    "FULL_FEATURES",
    "required_feature",
    "features_for_render_ir",
    "scene_fits_tier",
    "emit_feature_defines",
]
