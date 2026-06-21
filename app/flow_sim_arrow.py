"""
flow_sim_arrow.py – Load a casoCAD .arrow lattice file into FlowLatticeData.

Kept deliberately thin: only file I/O and field extraction happen here.
No Qt, no simulation math.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.io import read_lattice


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_FLOW_DIR: tuple[float, float, float] = (1.0, 0.0, 0.0)
_AXIS_DIRECTIONS: tuple[tuple[float, float, float], ...] = (
    (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, -1.0), (0.0, 0.0, 1.0),
)


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------

def _float3(v: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    try:
        a = np.asarray(v, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return default
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        return default
    return (float(a[0]), float(a[1]), float(a[2]))


def _col(table: Any, *names: str) -> str | None:
    existing = {str(n).lower(): str(n) for n in getattr(table, "column_names", ())}
    for name in names:
        if name.lower() in existing:
            return existing[name.lower()]
    return None


def _fcol(table: Any, name: str | None) -> NDArray[np.float64]:
    if name is None:
        return np.empty(0, dtype=np.float64)
    return np.asarray(table.column(name), dtype=np.float64)


def _u8col(table: Any, name: str | None, size: int) -> NDArray[np.uint8]:
    if name is None:
        return np.zeros(size, dtype=np.uint8)
    return np.asarray(table.column(name), dtype=np.uint8)


def _to_int(v: Any, field: str) -> int:
    try:
        return int(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"missing or invalid metadata field: {field}") from e


def _to_int_or_none(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _norm_tags(v: Any) -> tuple[int, ...]:
    if v is None:
        return ()
    items = v if isinstance(v, (list, tuple)) else (v,)
    out: list[int] = []
    for item in items:
        try:
            n = int(item)
            if n > 0:
                out.append(n)
        except (TypeError, ValueError):
            pass
    return tuple(out)


def _axis_matrix(grid: dict[str, Any]) -> NDArray[np.float64]:
    return np.array(
        [_float3(grid.get("axis_i"), (1.0, 0.0, 0.0)),
         _float3(grid.get("axis_j"), (0.0, 1.0, 0.0)),
         _float3(grid.get("axis_k"), (0.0, 0.0, 1.0))],
        dtype=np.float64,
    ).T


def _infer_ijk(
    positions: NDArray[np.float64],
    grid: dict[str, Any],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    empty = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64))
    if not positions.size:
        return empty
    dx = float(grid.get("dx", 1.0))
    if dx <= 0 or not np.isfinite(dx):
        raise ValueError("arrow metadata has invalid grid.dx")
    origin = np.asarray(_float3(grid.get("lattice_origin"), (0.0, 0.0, 0.0)))
    basis = _axis_matrix(grid)
    try:
        inv = np.linalg.inv(basis)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(basis)
    comp = ((positions - origin) @ inv) / dx
    idx = np.rint(comp).astype(np.int64)
    return idx[:, 0], idx[:, 1], idx[:, 2]


def _best_flow_direction(metadata: dict[str, Any]) -> tuple[float, float, float]:
    objects = metadata.get("object_directory")
    if not isinstance(objects, list):
        return _DEFAULT_FLOW_DIR
    candidates: list[tuple[tuple[float, float, float], str]] = []
    for item in objects:
        if not isinstance(item, dict) or item.get("kind") != "boundary_region":
            continue
        idx = _to_int_or_none(item.get("outside_direction"))
        vec = np.asarray(
            _AXIS_DIRECTIONS[idx] if idx is not None and 0 <= idx < len(_AXIS_DIRECTIONS)
            else _DEFAULT_FLOW_DIR
        )
        candidates.append((tuple(float(x) for x in vec), str(item.get("name", "")).lower()))
    for vec, name in candidates:
        if "inlet" in name:
            return tuple(-float(x) for x in vec)
    if candidates:
        return tuple(-float(x) for x in candidates[0][0])
    return _DEFAULT_FLOW_DIR


# ---------------------------------------------------------------------------
# Data object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlowLatticeData:
    path: Path
    positions: NDArray[np.float64]    # (N, 3)
    i: NDArray[np.int64]              # (N,)
    j: NDArray[np.int64]
    k: NDArray[np.int64]
    node_type: NDArray[np.uint8]      # 0 = fluid, 1 = boundary
    tag_ids: tuple[tuple[int, ...], ...]
    object_names: dict[int, str]
    lattice_origin: tuple[float, float, float]
    axis_i: tuple[float, float, float]
    axis_j: tuple[float, float, float]
    axis_k: tuple[float, float, float]
    dx: float
    nx: int
    ny: int
    nz: int
    dimension: int
    flow_direction: tuple[float, float, float]

    @property
    def boundary_mask(self) -> NDArray[np.bool_]:
        return self.node_type == 1

    @property
    def fluid_mask(self) -> NDArray[np.bool_]:
        return self.node_type != 1

    @property
    def tagged_mask(self) -> NDArray[np.bool_]:
        return np.array([bool(tags) for tags in self.tag_ids], dtype=np.bool_)

    @property
    def untagged_boundary_mask(self) -> NDArray[np.bool_]:
        return self.boundary_mask & ~self.tagged_mask


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_arrow_lattice(path: str | Path) -> FlowLatticeData:
    path = Path(path)
    table, metadata = read_lattice(path)
    if not isinstance(metadata, dict):
        raise ValueError("arrow metadata is missing or invalid")
    grid = metadata.get("grid", metadata)
    if not isinstance(grid, dict):
        raise ValueError("arrow metadata is missing grid section")

    x = _col(table, "x", "coord_x")
    y = _col(table, "y", "coord_y")
    z = _col(table, "z", "coord_z")
    nt = _col(table, "node_type", "type")
    missing = [n for n, v in (("x", x), ("y", y), ("z", z), ("node_type", nt)) if v is None]
    if missing:
        raise ValueError(f"missing arrow columns: {', '.join(missing)}")

    positions = np.column_stack((_fcol(table, x), _fcol(table, y), _fcol(table, z)))
    N = positions.shape[0]
    node_type = _u8col(table, nt, N)

    # Grid indices
    ic = _col(table, "i", "index_i")
    jc = _col(table, "j", "index_j")
    kc = _col(table, "k", "index_k")
    if ic and jc and kc:
        i = np.asarray(table.column(ic), dtype=np.int64)
        j = np.asarray(table.column(jc), dtype=np.int64)
        k = np.asarray(table.column(kc), dtype=np.int64)
    else:
        i, j, k = _infer_ijk(positions, grid)

    # Tag ids
    tc = _col(table, "tag_ids", "tags", "tag_id")
    if tc is not None:
        tag_ids = tuple(_norm_tags(v) for v in table.column(tc).to_pylist())
    else:
        tag_ids = tuple(() for _ in range(N))

    # Object names
    object_names: dict[int, str] = {}
    for item in (metadata.get("object_directory") or []):
        if not isinstance(item, dict):
            continue
        oid = _to_int_or_none(item.get("object_id"))
        if oid and oid > 0:
            object_names[oid] = str(item.get("name", f"Object {oid}")).strip() or f"Object {oid}"

    return FlowLatticeData(
        path=path,
        positions=positions,
        i=i, j=j, k=k,
        node_type=node_type,
        tag_ids=tag_ids,
        object_names=object_names,
        lattice_origin=_float3(grid.get("lattice_origin"), (0.0, 0.0, 0.0)),
        axis_i=_float3(grid.get("axis_i"), (1.0, 0.0, 0.0)),
        axis_j=_float3(grid.get("axis_j"), (0.0, 1.0, 0.0)),
        axis_k=_float3(grid.get("axis_k"), (0.0, 0.0, 1.0)),
        dx=float(grid.get("dx", 1.0)),
        nx=_to_int(grid.get("nx"), "grid.nx"),
        ny=_to_int(grid.get("ny"), "grid.ny"),
        nz=_to_int(grid.get("nz"), "grid.nz"),
        dimension=_to_int(grid.get("dimension"), "grid.dimension"),
        flow_direction=_best_flow_direction(metadata),
    )
