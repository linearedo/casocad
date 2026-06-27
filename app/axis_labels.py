from __future__ import annotations

import math


def world_axis_label(
    vector: tuple[float, float, float] | object,
    *,
    suffix: str = " axis",
) -> str:
    try:
        x, y, z = (float(value) for value in vector)  # type: ignore[operator]
    except (TypeError, ValueError):
        return f"custom{suffix}"
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 1.0e-12 or not math.isfinite(length):
        return f"custom{suffix}"
    unit = (x / length, y / length, z / length)
    index = max(range(3), key=lambda item: abs(unit[item]))
    if (
        abs(abs(unit[index]) - 1.0) <= 1.0e-6
        and all(abs(unit[item]) <= 1.0e-6 for item in range(3) if item != index)
    ):
        return f"{('X', 'Y', 'Z')[index]}{suffix}"
    return f"vector ({unit[0]:.3g}, {unit[1]:.3g}, {unit[2]:.3g})"
