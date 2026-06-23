from __future__ import annotations

"""GL-GLSL -> Vulkan-GLSL transform for the QRhi renderer (migration D-R2).

The interpreter shaders are written in OpenGL-style GLSL: storage buffers, images
and loose ``uniform`` scalars each live in *separate* binding namespaces, so e.g.
``Nodes`` and the output image can both be ``binding = 0``. Vulkan/QRhi (and
therefore ``qsb``) put **all** resources in one namespace and forbid loose
uniforms. This module mechanically rewrites an assembled GL interpreter source
into Vulkan-style GLSL so ``qsb`` can bake it, without touching the original GL
chunks (the ModernGL path keeps working during the migration):

* bump ``#version`` to a Vulkan baseline,
* move the output ``image2D`` to a binding that does not collide with the SSBOs,
* collect every loose ``uniform <scalar/vector> name;`` into a single ``std140``
  uniform block (unnamed, so member references in the shader body are unchanged).

SSBO bindings are already mutually unique (core 0-3, eval 4-7, cull 8-13) and are
left as-is. The host packs the uniform block with std140 layout (see the renderer).
"""

import re

# Bindings chosen above the SSBO range (0-13) so nothing collides in Vulkan.
IMAGE_BINDING = 14
UBO_BINDING = 15

_VERSION_RE = re.compile(r"^#version\s+\d+.*$", re.MULTILINE)
# A loose uniform of a basic type: `uniform vec3 u_foo;` / `uniform int u_bar[16];`
# Excludes opaque types (image*/sampler*) which must stay as resource bindings.
_LOOSE_UNIFORM_RE = re.compile(
    r"^[ \t]*uniform[ \t]+"
    r"(?!image|sampler)"          # not an opaque type
    r"([A-Za-z_][A-Za-z0-9_]*)"   # type
    r"[ \t]+"
    r"([A-Za-z_][A-Za-z0-9_]*)"   # name
    r"((?:\[[0-9]+\])?)"          # optional array suffix
    r"[ \t]*;[ \t]*$",
    re.MULTILINE,
)
_IMAGE_BINDING_RE = re.compile(
    r"(layout\s*\(\s*[^)]*?binding\s*=\s*)(\d+)(\s*\)[^;]*\buniform\b[^;]*\bimage2D\b)"
)


def vulkanize(source: str, *, version: str = "#version 450") -> str:
    """Return a Vulkan-style copy of an assembled GL interpreter shader.

    Raises ValueError if no loose uniforms are found (a sign the source was not
    the expected GL interpreter shape).
    """

    # 1) version
    src, n = _VERSION_RE.subn(version, source, count=1)
    if n == 0:
        src = version + "\n" + src

    # 2) move the output image off the SSBO binding range
    src = _IMAGE_BINDING_RE.sub(lambda m: f"{m.group(1)}{IMAGE_BINDING}{m.group(3)}", src)

    # 3) collect loose uniforms into one std140 block
    members: list[str] = []

    def _collect(match: re.Match[str]) -> str:
        gl_type, name, array = match.group(1), match.group(2), match.group(3)
        members.append(f"    {gl_type} {name}{array};")
        return ""  # drop the loose declaration

    src = _LOOSE_UNIFORM_RE.sub(_collect, src)
    if not members:
        raise ValueError("vulkanize: no loose uniforms found — unexpected source")

    block = (
        f"layout(std140, binding = {UBO_BINDING}) uniform _Globals {{\n"
        + "\n".join(members)
        + "\n};\n"
    )

    # Insert the UBO block right after the #version line (members are basic types,
    # so no forward-declaration issues).
    lines = src.split("\n")
    insert_at = next(
        (i + 1 for i, ln in enumerate(lines) if ln.startswith("#version")), 0
    )
    lines.insert(insert_at, "\n" + block)
    return "\n".join(lines)


def uniform_block_members(source: str) -> list[tuple[str, str, int]]:
    """Return ``(glsl_type, name, array_len)`` for each loose uniform, in source
    order — the std140 packing order the host must follow."""

    out: list[tuple[str, str, int]] = []
    for m in _LOOSE_UNIFORM_RE.finditer(source):
        array = m.group(3)
        length = int(array[1:-1]) if array else 0
        out.append((m.group(1), m.group(2), length))
    return out


__all__ = ["vulkanize", "uniform_block_members", "IMAGE_BINDING", "UBO_BINDING"]
