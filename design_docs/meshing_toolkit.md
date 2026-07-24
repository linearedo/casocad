# Exact SDF queries for mesh producers

This document describes the geometry-side contract implemented by
`kernel/src/meshing.rs`. It is independent of any particular mesh storage or
generator.

## Interior exactness

Domain fields are exact signed distances on the negative (interior) side.
Positive magnitudes classify exterior points but must not be consumed as
distances. Boundary roots therefore use sign-preserving bracketing.

## Differential queries

`MeshableDomain` exposes boundary normals, interior-start projection,
curvature, and the scale-derived boundary tolerance. A placed 2D domain also
exposes a local orthonormal `MeshableDomainSpace` with bounds, point/coordinate
conversion, and an in-plane SDF.

## Boundary classification

`BoundaryBand::UnprojectedSamples` uses the display classifier tolerance and
accepts straight chords approximating a curved wall.
`BoundaryBand::ProjectedVertices` is the tight zero-set band.
`BoundaryBand::Custom` supplies an explicit absolute tolerance.

`classify_boundary` is total: it reports whether the point is on the domain
boundary, the controlling leaf, and the winning named region. Region
precedence prefers more cuts, then patch-scoped regions, then later creation.

## Interfaces

Nested marked domains expose paired interface sides without changing either
domain's sign convention. Consumers seed from the inner domain's interior
band and preserve each side's own zone and boundary classification.

The Arrow-native storage and producer contract is documented in
`docs/casomesh_arrow_v3.md`.
