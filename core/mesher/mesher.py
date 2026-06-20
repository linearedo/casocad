from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from core.boundary import BoundaryRegion
from core.boundary_patches import SURFACE_SELECTOR_TYPES, surface_selector_values
from core.boundary_direction import (
    owner_outside_direction_vector,
    world_axis_direction_from_vector,
)
from core.io.arrow_writer import ArrowWriter
from core.sdf.placed_1d import PlacedSDF1D
from core.sdf.placed_2d import PlacedPolyline2D, PlacedSDF2D
from core.sdf.base import SDFNode
from core.sdf.primitives_2d import (
    CircleProfile,
    EllipseProfile,
    RectangleProfile,
    SquareProfile,
)

from .classifier import (
    evaluate_volume_attribution,
    evaluate_with_attribution,
    nearest_tag_mask,
    retained_mask,
    sample_boundary_faces,
)
from .domain import FluidDomain, MesherConfig
from .grid import GridSpec, derive_lattice_grid, generate_chunks
from .resolution import recommended_max_dx

ProgressCallback = Callable[[int, int], None]
PreviewCallback = Callable[["LatticePreviewChunk"], None]


def _preview_priorities(
    i: NDArray[np.uint64],
    j: NDArray[np.uint64],
    k: NDArray[np.uint64],
) -> NDArray[np.uint64]:
    """Return deterministic pseudo-random priorities for spatial sampling."""
    values = (
        i * np.uint64(0x9E3779B185EBCA87)
        ^ j * np.uint64(0xC2B2AE3D27D4EB4F)
        ^ k * np.uint64(0x165667B19E3779F9)
    )
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return values


def _lowest_priority_indices(
    priorities: NDArray[np.uint64],
    limit: int,
) -> NDArray[np.intp]:
    if limit <= 0:
        return np.empty(0, dtype=np.intp)
    if priorities.size <= limit:
        return np.arange(priorities.size, dtype=np.intp)
    return np.argpartition(priorities, limit - 1)[:limit]


@dataclass(frozen=True)
class LatticePreviewChunk:
    dimension: int
    preview_cell_size: float
    preview_positions: NDArray[np.float32]
    preview_node_types: NDArray[np.uint8]
    preview_boundary_faces: NDArray[np.uint8]
    preview_source_object_ids: NDArray[np.uint16]
    preview_primary_tag_ids: NDArray[np.uint16]
    preview_tag_ids: tuple[tuple[int, ...], ...]
    preview_tag_axis_u: NDArray[np.float32]
    preview_tag_axis_v: NDArray[np.float32]
    preview_axis_i: tuple[float, float, float]
    preview_axis_j: tuple[float, float, float]
    preview_axis_k: tuple[float, float, float]


@dataclass(frozen=True)
class LatticeResult:
    dimension: int
    path: Path
    row_count: int
    grid_node_count: int
    file_size: int
    preview_cell_size: float
    preview_positions: NDArray[np.float32]
    preview_node_types: NDArray[np.uint8]
    preview_boundary_faces: NDArray[np.uint8]
    preview_source_object_ids: NDArray[np.uint16]
    preview_primary_tag_ids: NDArray[np.uint16]
    preview_tag_ids: tuple[tuple[int, ...], ...]
    preview_tag_axis_u: NDArray[np.float32]
    preview_tag_axis_v: NDArray[np.float32]
    preview_axis_i: tuple[float, float, float]
    preview_axis_j: tuple[float, float, float]
    preview_axis_k: tuple[float, float, float]
    boundary_sample_indices: NDArray[np.uint64]
    boundary_sample_directions: NDArray[np.uint8]
    boundary_sample_positions: NDArray[np.float64]
    boundary_sample_normals: NDArray[np.float64]
    boundary_sample_owner_object_ids: NDArray[np.uint16]
    boundary_sample_region_ids: tuple[tuple[int, ...], ...]
    boundary_sample_errors: NDArray[np.float64]
    boundary_error_maximum: float
    boundary_error_mean: float
    boundary_error_rms: float
    boundary_error_percentile_95: float
    requested_boundary_error_tolerance: float | None
    boundary_error_tolerance_met: bool
    refinement_count: int


def _walk_unique(root: SDFNode) -> tuple[SDFNode, ...]:
    result: list[SDFNode] = []
    seen: set[int] = set()

    def visit(node: SDFNode) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        result.append(node)
        for child in node.children():
            visit(child)

    visit(root)
    return tuple(result)


def _object_id_directory(root: SDFNode) -> dict[int, SDFNode]:
    return {
        node.object_id: node
        for node in _walk_unique(root)
        if node.object_id > 0
    }


def _boundary_interval_mask(
    tag: BoundaryRegion,
    owner: PlacedSDF2D,
    positions: NDArray[np.float64],
) -> NDArray[np.bool_]:
    profile = owner.profile
    if isinstance(profile, SquareProfile):
        profile = profile._rectangle()
    assert tag.selector_start is not None and tag.selector_end is not None
    u, v, _plane = owner.project_numpy(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
    )
    if isinstance(profile, CircleProfile) and tag.patch_id == "curve":
        cu, cv = profile.center
        parameter = np.mod(np.arctan2(v - cv, u - cu), 2.0 * np.pi) / (2.0 * np.pi)
        return _interval_parameter_mask(parameter, tag.selector_start, tag.selector_end)
    if isinstance(profile, EllipseProfile) and tag.patch_id == "curve":
        cu, cv = profile.center
        au, av = profile.semi_axes
        parameter = (
            np.mod(np.arctan2((v - cv) / av, (u - cu) / au), 2.0 * np.pi)
            / (2.0 * np.pi)
        )
        return _interval_parameter_mask(parameter, tag.selector_start, tag.selector_end)
    if not isinstance(profile, RectangleProfile):
        return np.ones(positions.shape[0], dtype=np.bool_)
    if tag.patch_id not in {"-U", "+U", "-V", "+V"}:
        return np.ones(positions.shape[0], dtype=np.bool_)
    start, end = sorted((tag.selector_start, tag.selector_end))
    cu, cv = profile.center
    hu, hv = profile.half_size
    if tag.patch_id in {"-U", "+U"}:
        parameter = (v - (cv - hv)) / (2.0 * hv)
    else:
        parameter = (u - (cu - hu)) / (2.0 * hu)
    return np.asarray(
        (parameter >= start - 1.0e-9) & (parameter <= end + 1.0e-9),
        dtype=np.bool_,
    )


def _interval_parameter_mask(
    parameter: NDArray[np.float64],
    start: float,
    end: float,
) -> NDArray[np.bool_]:
    parameter = np.where(np.isclose(parameter, 1.0, atol=1.0e-12), 0.0, parameter)
    start = float(np.mod(start, 1.0))
    end = float(np.mod(end, 1.0))
    if start <= end:
        return np.asarray(
            (parameter >= start - 1.0e-9) & (parameter <= end + 1.0e-9),
            dtype=np.bool_,
        )
    return np.asarray(
        (parameter >= start - 1.0e-9) | (parameter <= end + 1.0e-9),
        dtype=np.bool_,
    )


def _selector_object_from_id(
    selector_id: str,
    selector_by_id: dict[int, SDFNode],
) -> SDFNode | None:
    prefix = "selector:"
    if not selector_id.startswith(prefix):
        return None
    try:
        object_id = int(selector_id[len(prefix):])
    except ValueError:
        return None
    return selector_by_id.get(object_id)


def _surface_split_selector_mask(
    selector_id: str,
    selector_by_id: dict[int, SDFNode],
    root: SDFNode,
    positions: NDArray[np.float64],
    *,
    region: BoundaryRegion | None = None,
    side: str = "inside",
    tolerance: float,
) -> NDArray[np.bool_]:
    selector = _selector_object_from_id(selector_id, selector_by_id)
    if selector is None:
        return np.zeros(positions.shape[0], dtype=np.bool_)
    values = surface_selector_values(
        root,
        selector,
        positions,
        scope_region=region,
    )
    inside = np.asarray(values <= tolerance, dtype=np.bool_)
    if side == "outside":
        return np.asarray(~inside, dtype=np.bool_)
    return inside


class LatticeMesher:
    def __init__(
        self,
        domain: FluidDomain,
        config: MesherConfig,
        preview_limit: int = 200_000,
    ) -> None:
        self.domain = domain
        self.config = config
        self.preview_limit = max(0, preview_limit)

    def _metadata(self, grid: GridSpec) -> dict[str, Any]:
        objects = {
            node.object_id: node
            for node in (
                *_walk_unique(self.domain.root),
                *self.domain.tag_objects,
                *self.domain.selector_objects,
            )
        }
        return {
            "sdf_cad_version": "0.2.0",
            "unit_label": self.config.unit_label,
            "dimension": self.domain.root.dimension,
            "boundary_rule": (
                "retained node with at least one outside "
                f"{4 if self.domain.root.dimension == 2 else 6}-neighbor"
            ),
            "tag_rule": (
                "PlacedSDF1D: refined 2D boundary crossings on the placed "
                "filled segment; PlacedSDF2D: nearest lattice layer and exact "
                "filled profile in 3D; "
                "BoundaryRegion: refined per-edge SDF zero crossing with matching "
                "boundary owner and optional exposed direction; node tag_ids are "
                "the union of matching exposed directions"
            ),
            "boundary_sampling": {
                "method": "linear sign-crossing estimate plus 8 bisection steps",
                "ownership": "evaluated per exposed lattice direction",
                "normal": "central SDF gradient at refined crossing",
                "error_metric": (
                    "distance from each retained boundary node to its refined "
                    "SDF crossing along an exposed lattice edge"
                ),
                "requested_maximum_error": (
                    self.config.boundary_error_tolerance
                ),
            },
            "fluid_domain": {
                "root_object_id": self.domain.root.object_id,
                "tag_object_ids": [
                    tag.object_id for tag in self.domain.tag_objects
                ],
                "selector_object_ids": [
                    selector.object_id
                    for selector in self.domain.selector_objects
                ],
            },
            "grid": {
                "nx": grid.nx,
                "ny": grid.ny,
                "nz": grid.nz,
                "dimension": grid.dimension,
                "dx": self.config.dx,
                "n_levels": 0,
                "recommended_max_dx": recommended_max_dx(self.domain.root),
                "lattice_origin": grid.lattice_origin,
                "axis_i": grid.axis_i,
                "axis_j": grid.axis_j,
                "axis_k": grid.axis_k,
            },
            "object_directory": [
                {
                    "object_id": object_id,
                    "name": node.name,
                    "kind": node.kind,
                    "dimension": node.dimension,
                    **(
                        {
                            "owner_object_id": node.owner_object_id,
                            "outside_direction": node.outside_direction,
                        }
                        if isinstance(node, BoundaryRegion)
                        else {}
                    ),
                }
                for object_id, node in sorted(objects.items())
            ],
        }

    def mesh(
        self,
        path: str | Path,
        progress: ProgressCallback | None = None,
        preview: PreviewCallback | None = None,
    ) -> LatticeResult:
        output_path = Path(path)
        tolerance = self.config.boundary_error_tolerance
        dx = self.config.dx
        result: LatticeResult | None = None
        for refinement in range(self.config.max_error_refinements + 1):
            grid = derive_lattice_grid(self.domain.root, dx)
            if grid.node_count > self.config.max_candidate_nodes:
                output_path.unlink(missing_ok=True)
                raise ValueError(
                    "boundary error target requires "
                    f"{grid.node_count:,} candidate nodes, exceeding the "
                    f"configured limit of {self.config.max_candidate_nodes:,}"
                )
            attempt_config = replace(
                self.config,
                dx=dx,
                boundary_error_tolerance=tolerance,
            )
            attempt = LatticeMesher(
                self.domain,
                attempt_config,
                preview_limit=self.preview_limit,
            )
            try:
                result = attempt._mesh_once(path, progress, preview)
            except Exception:
                output_path.unlink(missing_ok=True)
                raise
            maximum = result.boundary_error_maximum
            tolerance_met = tolerance is None or maximum <= tolerance
            result = replace(
                result,
                requested_boundary_error_tolerance=tolerance,
                boundary_error_tolerance_met=tolerance_met,
                refinement_count=refinement,
            )
            if tolerance_met:
                return result
            dx *= 0.5
        assert result is not None
        output_path.unlink(missing_ok=True)
        raise ValueError(
            "maximum boundary error "
            f"{result.boundary_error_maximum:.6g} m exceeds the requested "
            f"{tolerance:.6g} m after "
            f"{self.config.max_error_refinements} refinements"
        )

    def _mesh_once(
        self,
        path: str | Path,
        progress: ProgressCallback | None = None,
        preview: PreviewCallback | None = None,
    ) -> LatticeResult:
        output_path = Path(path)
        grid = derive_lattice_grid(self.domain.root, self.config.dx)
        boundary_offsets = self.domain.boundary_offsets()
        row_count = 0
        processed = 0
        preview_positions: list[NDArray[np.float32]] = []
        preview_node_types: list[NDArray[np.uint8]] = []
        preview_boundary_faces: list[NDArray[np.uint8]] = []
        preview_source_object_ids: list[NDArray[np.uint16]] = []
        preview_primary_tag_ids: list[NDArray[np.uint16]] = []
        preview_tag_ids: list[tuple[int, ...]] = []
        preview_tag_axis_u: list[NDArray[np.float32]] = []
        preview_tag_axis_v: list[NDArray[np.float32]] = []
        boundary_sample_indices: list[NDArray[np.uint64]] = []
        boundary_sample_directions: list[NDArray[np.uint8]] = []
        boundary_sample_positions: list[NDArray[np.float64]] = []
        boundary_sample_normals: list[NDArray[np.float64]] = []
        boundary_sample_owner_ids: list[NDArray[np.uint16]] = []
        boundary_sample_region_ids: list[tuple[int, ...]] = []
        boundary_sample_errors: list[NDArray[np.float64]] = []
        interior_priorities = np.empty(0, dtype=np.uint64)
        interior_positions = np.empty((0, 3), dtype=np.float32)
        interior_node_types = np.empty(0, dtype=np.uint8)
        interior_boundary_faces = np.empty(0, dtype=np.uint8)
        interior_source_object_ids = np.empty(0, dtype=np.uint16)
        interior_primary_tag_ids = np.empty(0, dtype=np.uint16)
        interior_tag_ids: tuple[tuple[int, ...], ...] = ()
        interior_tag_axis_u = np.empty((0, 3), dtype=np.float32)
        interior_tag_axis_v = np.empty((0, 3), dtype=np.float32)
        interior_count = 0
        tag_by_id = {tag.object_id: tag for tag in self.domain.tag_objects}
        owner_by_id = _object_id_directory(self.domain.root)
        selector_by_id = {
            selector.object_id: selector for selector in self.domain.selector_objects
        }

        def tag_axis(object_id: int, attribute: str) -> tuple[float, float, float]:
            tag = tag_by_id.get(object_id)
            if isinstance(tag, (PlacedSDF1D, PlacedSDF2D)):
                return getattr(tag, attribute, (0.0, 0.0, 0.0))
            return (0.0, 0.0, 0.0)

        with ArrowWriter(output_path, self._metadata(grid)) as writer:
            for chunk in generate_chunks(grid, self.config.chunk_size):
                sdf, source_object_ids = evaluate_with_attribution(
                    self.domain.root, chunk.x, chunk.y, chunk.z
                )
                keep = retained_mask(sdf)
                retained_sdf = sdf[keep]
                retained_source_ids = source_object_ids[keep]
                volume_source_ids = evaluate_volume_attribution(
                    self.domain.root,
                    chunk.x,
                    chunk.y,
                    chunk.z,
                )[keep]
                face_samples = sample_boundary_faces(
                    self.domain.root,
                    chunk.x,
                    chunk.y,
                    chunk.z,
                    keep,
                    self.config.dx,
                    offsets=boundary_offsets,
                )
                boundary_faces = face_samples.boundary_faces
                node_type = np.where(
                    boundary_faces != 0,
                    np.uint8(1),
                    np.uint8(0),
                ).astype(np.uint8, copy=False)
                boundary_owner_ids = np.zeros(
                    retained_sdf.shape,
                    dtype=np.uint16,
                )
                for direction in range(len(boundary_offsets)):
                    direction_samples = face_samples.directions == direction
                    nodes = face_samples.node_indices[direction_samples]
                    unset = boundary_owner_ids[nodes] == 0
                    boundary_owner_ids[nodes[unset]] = (
                        face_samples.owner_object_ids[direction_samples][unset]
                    )
                retained_source_ids = np.where(
                    node_type == 1,
                    boundary_owner_ids,
                    volume_source_ids,
                ).astype(np.uint16, copy=False)
                retained_x = chunk.x[keep]
                retained_y = chunk.y[keep]
                retained_z = chunk.z[keep]
                tags: list[list[int]] = [[] for _ in range(retained_sdf.size)]
                sample_tags: list[list[int]] = [
                    [] for _ in range(face_samples.node_indices.size)
                ]
                for tag in self.domain.tag_objects:
                    if isinstance(tag, BoundaryRegion):
                        matched_samples = (
                            face_samples.owner_object_ids
                            == tag.owner_object_id
                        )
                        if tag.outside_direction is not None:
                            owner = owner_by_id.get(tag.owner_object_id)
                            direction = (
                                owner_outside_direction_vector(
                                    owner,
                                    tag.outside_direction,
                                )
                                if owner is not None
                                else None
                            )
                            if direction is None:
                                matched_samples &= (
                                    face_samples.directions
                                    == tag.outside_direction
                                )
                            else:
                                lattice_direction = (
                                    world_axis_direction_from_vector(direction)
                                )
                                if lattice_direction is not None:
                                    matched_samples &= (
                                        face_samples.directions
                                        == lattice_direction
                                    )
                                else:
                                    alignment = np.einsum(
                                        "ij,j->i",
                                        face_samples.normals,
                                        direction,
                                    )
                                    matched_samples &= alignment >= 0.90
                        if (
                            tag.selector_start is not None
                            and tag.selector_end is not None
                            and self.domain.root.dimension == 2
                        ):
                            owner = owner_by_id.get(tag.owner_object_id)
                            if isinstance(owner, PlacedSDF2D):
                                matched_samples &= _boundary_interval_mask(
                                    tag,
                                    owner,
                                    face_samples.positions,
                                )
                        if (
                            tag.selector_id is not None
                            and tag.selector_type
                            in SURFACE_SELECTOR_TYPES
                            and self.domain.root.dimension == 3
                        ):
                            matched_samples &= _surface_split_selector_mask(
                                tag.selector_id,
                                selector_by_id,
                                self.domain.root,
                                face_samples.positions,
                                region=tag,
                                side=tag.selector_side,
                                tolerance=max(self.config.dx * 0.5, 1e-9),
                            )
                        matched_sample_indices = np.flatnonzero(
                            matched_samples
                        )
                        matched_nodes = np.unique(
                            face_samples.node_indices[matched_sample_indices]
                        )
                        for sample_index in matched_sample_indices:
                            sample_tags[int(sample_index)].append(tag.object_id)
                    elif isinstance(tag, (PlacedSDF1D, PlacedPolyline2D)):
                        matched_samples = tag.contains_points(
                            face_samples.positions,
                            tolerance=max(self.config.dx / 128.0, 1e-9),
                        )
                        matched_sample_indices = np.flatnonzero(
                            matched_samples
                        )
                        matched_nodes = np.unique(
                            face_samples.node_indices[matched_sample_indices]
                        )
                        for sample_index in matched_sample_indices:
                            sample_tags[int(sample_index)].append(tag.object_id)
                    else:
                        matched_nodes = np.flatnonzero(nearest_tag_mask(
                            tag,
                            grid,
                            chunk.i[keep],
                            chunk.j[keep],
                            chunk.k[keep],
                            retained_x,
                            retained_y,
                            retained_z,
                        ))
                    for index in matched_nodes:
                        tags[int(index)].append(tag.object_id)
                normalized_tags = [sorted(set(items)) for items in tags]
                normalized_sample_tags = [
                    tuple(sorted(set(items))) for items in sample_tags
                ]
                retained_i = chunk.i[keep]
                retained_j = chunk.j[keep]
                retained_k = chunk.k[keep]
                boundary_sample_indices.append(
                    np.column_stack(
                        (
                            retained_i[face_samples.node_indices],
                            retained_j[face_samples.node_indices],
                            retained_k[face_samples.node_indices],
                        )
                    ).astype(np.uint64, copy=False)
                )
                boundary_sample_directions.append(face_samples.directions)
                boundary_sample_positions.append(face_samples.positions)
                boundary_sample_normals.append(face_samples.normals)
                boundary_sample_owner_ids.append(
                    face_samples.owner_object_ids
                )
                boundary_sample_errors.append(
                    face_samples.approximation_errors
                )
                boundary_sample_region_ids.extend(normalized_sample_tags)
                writer.write_batch(
                    x=retained_x,
                    y=retained_y,
                    z=retained_z,
                    i=chunk.i[keep],
                    j=chunk.j[keep],
                    k=chunk.k[keep],
                    node_type=node_type,
                    tag_ids=normalized_tags,
                    level=np.zeros(retained_sdf.shape, dtype=np.uint8),
                )

                positions = np.column_stack(
                    (retained_x, retained_y, retained_z)
                ).astype(np.float32, copy=False)

                boundary_indices = np.flatnonzero(node_type == 1)
                if boundary_indices.size:
                    boundary_tags = tuple(
                        tuple(normalized_tags[int(index)])
                        for index in boundary_indices
                    )
                    boundary_primary_ids = np.asarray(
                        [items[0] if items else 0 for items in boundary_tags],
                        dtype=np.uint16,
                    )
                    boundary_axis_u = np.asarray(
                        [
                            tag_axis(int(object_id), "axis_u")
                            if object_id
                            else (0.0, 0.0, 0.0)
                            for object_id in boundary_primary_ids
                        ],
                        dtype=np.float32,
                    )
                    boundary_axis_v = np.asarray(
                        [
                            tag_axis(int(object_id), "axis_v")
                            if object_id
                            else (0.0, 0.0, 0.0)
                            for object_id in boundary_primary_ids
                        ],
                        dtype=np.float32,
                    )
                    preview_positions.append(positions[boundary_indices])
                    preview_node_types.append(node_type[boundary_indices])
                    preview_boundary_faces.append(
                        boundary_faces[boundary_indices]
                    )
                    preview_source_object_ids.append(
                        retained_source_ids[boundary_indices]
                    )
                    preview_primary_tag_ids.append(boundary_primary_ids)
                    preview_tag_ids.extend(boundary_tags)
                    preview_tag_axis_u.append(boundary_axis_u)
                    preview_tag_axis_v.append(boundary_axis_v)
                    if preview is not None:
                        preview(
                            LatticePreviewChunk(
                                dimension=self.domain.root.dimension,
                                preview_cell_size=self.config.dx,
                                preview_positions=positions[boundary_indices],
                                preview_node_types=node_type[boundary_indices],
                                preview_boundary_faces=boundary_faces[
                                    boundary_indices
                                ],
                                preview_source_object_ids=retained_source_ids[
                                    boundary_indices
                                ],
                                preview_primary_tag_ids=boundary_primary_ids,
                                preview_tag_ids=boundary_tags,
                                preview_tag_axis_u=boundary_axis_u,
                                preview_tag_axis_v=boundary_axis_v,
                                preview_axis_i=grid.axis_i,
                                preview_axis_j=grid.axis_j,
                                preview_axis_k=grid.axis_k,
                            )
                        )

                if self.preview_limit > 0:
                    interior_indices = np.flatnonzero(node_type == 0)
                    interior_count += int(interior_indices.size)
                    if interior_indices.size:
                        priorities = _preview_priorities(
                            retained_i[interior_indices],
                            retained_j[interior_indices],
                            retained_k[interior_indices],
                        )
                        local = _lowest_priority_indices(
                            priorities, self.preview_limit
                        )
                        selected = interior_indices[local]
                        selected_tags = tuple(
                            tuple(normalized_tags[int(index)])
                            for index in selected
                        )
                        selected_primary_ids = np.asarray(
                            [
                                items[0] if items else 0
                                for items in selected_tags
                            ],
                            dtype=np.uint16,
                        )
                        selected_axis_u = np.asarray(
                            [
                                tag_axis(int(object_id), "axis_u")
                                if object_id
                                else (0.0, 0.0, 0.0)
                                for object_id in selected_primary_ids
                            ],
                            dtype=np.float32,
                        )
                        selected_axis_v = np.asarray(
                            [
                                tag_axis(int(object_id), "axis_v")
                                if object_id
                                else (0.0, 0.0, 0.0)
                                for object_id in selected_primary_ids
                            ],
                            dtype=np.float32,
                        )
                        merged_priorities = np.concatenate(
                            (interior_priorities, priorities[local])
                        )
                        merged_positions = np.concatenate(
                            (interior_positions, positions[selected])
                        )
                        merged_node_types = np.concatenate(
                            (interior_node_types, node_type[selected])
                        )
                        merged_boundary_faces = np.concatenate(
                            (
                                interior_boundary_faces,
                                boundary_faces[selected],
                            )
                        )
                        merged_source_ids = np.concatenate(
                            (
                                interior_source_object_ids,
                                retained_source_ids[selected],
                            )
                        )
                        merged_primary_ids = np.concatenate(
                            (
                                interior_primary_tag_ids,
                                selected_primary_ids,
                            )
                        )
                        merged_tags = interior_tag_ids + selected_tags
                        merged_axis_u = np.concatenate(
                            (interior_tag_axis_u, selected_axis_u)
                        )
                        merged_axis_v = np.concatenate(
                            (interior_tag_axis_v, selected_axis_v)
                        )
                        keep_interior = _lowest_priority_indices(
                            merged_priorities, self.preview_limit
                        )
                        interior_priorities = merged_priorities[keep_interior]
                        interior_positions = merged_positions[keep_interior]
                        interior_node_types = merged_node_types[keep_interior]
                        interior_boundary_faces = merged_boundary_faces[
                            keep_interior
                        ]
                        interior_source_object_ids = merged_source_ids[
                            keep_interior
                        ]
                        interior_primary_tag_ids = merged_primary_ids[
                            keep_interior
                        ]
                        interior_tag_ids = tuple(
                            merged_tags[int(index)] for index in keep_interior
                        )
                        interior_tag_axis_u = merged_axis_u[keep_interior]
                        interior_tag_axis_v = merged_axis_v[keep_interior]

                row_count += int(np.count_nonzero(keep))
                processed += chunk.x.size
                if progress is not None:
                    progress(processed, grid.node_count)

        interior_limit = min(
            self.preview_limit,
            round(interior_count * self.config.internal_preview_density),
        )
        keep_interior = _lowest_priority_indices(
            interior_priorities, interior_limit
        )
        if keep_interior.size:
            preview_positions.append(interior_positions[keep_interior])
            preview_node_types.append(interior_node_types[keep_interior])
            preview_boundary_faces.append(
                interior_boundary_faces[keep_interior]
            )
            preview_source_object_ids.append(
                interior_source_object_ids[keep_interior]
            )
            preview_primary_tag_ids.append(
                interior_primary_tag_ids[keep_interior]
            )
            preview_tag_ids.extend(
                interior_tag_ids[int(index)] for index in keep_interior
            )
            preview_tag_axis_u.append(interior_tag_axis_u[keep_interior])
            preview_tag_axis_v.append(interior_tag_axis_v[keep_interior])

        all_boundary_errors = (
            np.concatenate(boundary_sample_errors)
            if boundary_sample_errors
            else np.empty(0, dtype=np.float64)
        )
        return LatticeResult(
            dimension=self.domain.root.dimension,
            path=output_path,
            row_count=row_count,
            grid_node_count=grid.node_count,
            file_size=output_path.stat().st_size,
            preview_cell_size=self.config.dx,
            preview_positions=(
                np.concatenate(preview_positions)
                if preview_positions
                else np.empty((0, 3), dtype=np.float32)
            ),
            preview_node_types=(
                np.concatenate(preview_node_types)
                if preview_node_types
                else np.empty(0, dtype=np.uint8)
            ),
            preview_boundary_faces=(
                np.concatenate(preview_boundary_faces)
                if preview_boundary_faces
                else np.empty(0, dtype=np.uint8)
            ),
            preview_source_object_ids=(
                np.concatenate(preview_source_object_ids)
                if preview_source_object_ids
                else np.empty(0, dtype=np.uint16)
            ),
            preview_primary_tag_ids=(
                np.concatenate(preview_primary_tag_ids)
                if preview_primary_tag_ids
                else np.empty(0, dtype=np.uint16)
            ),
            preview_tag_ids=tuple(preview_tag_ids),
            preview_tag_axis_u=(
                np.concatenate(preview_tag_axis_u)
                if preview_tag_axis_u
                else np.empty((0, 3), dtype=np.float32)
            ),
            preview_tag_axis_v=(
                np.concatenate(preview_tag_axis_v)
                if preview_tag_axis_v
                else np.empty((0, 3), dtype=np.float32)
            ),
            preview_axis_i=grid.axis_i,
            preview_axis_j=grid.axis_j,
            preview_axis_k=grid.axis_k,
            boundary_sample_indices=(
                np.concatenate(boundary_sample_indices)
                if boundary_sample_indices
                else np.empty((0, 3), dtype=np.uint64)
            ),
            boundary_sample_directions=(
                np.concatenate(boundary_sample_directions)
                if boundary_sample_directions
                else np.empty(0, dtype=np.uint8)
            ),
            boundary_sample_positions=(
                np.concatenate(boundary_sample_positions)
                if boundary_sample_positions
                else np.empty((0, 3), dtype=np.float64)
            ),
            boundary_sample_normals=(
                np.concatenate(boundary_sample_normals)
                if boundary_sample_normals
                else np.empty((0, 3), dtype=np.float64)
            ),
            boundary_sample_owner_object_ids=(
                np.concatenate(boundary_sample_owner_ids)
                if boundary_sample_owner_ids
                else np.empty(0, dtype=np.uint16)
            ),
            boundary_sample_region_ids=tuple(boundary_sample_region_ids),
            boundary_sample_errors=all_boundary_errors,
            boundary_error_maximum=(
                float(np.max(all_boundary_errors))
                if all_boundary_errors.size
                else 0.0
            ),
            boundary_error_mean=(
                float(np.mean(all_boundary_errors))
                if all_boundary_errors.size
                else 0.0
            ),
            boundary_error_rms=(
                float(np.sqrt(np.mean(all_boundary_errors**2)))
                if all_boundary_errors.size
                else 0.0
            ),
            boundary_error_percentile_95=(
                float(np.percentile(all_boundary_errors, 95.0))
                if all_boundary_errors.size
                else 0.0
            ),
            requested_boundary_error_tolerance=(
                self.config.boundary_error_tolerance
            ),
            boundary_error_tolerance_met=True,
            refinement_count=0,
        )
