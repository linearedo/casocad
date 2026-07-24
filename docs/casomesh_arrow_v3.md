# casoCAD Arrow mesh v3

A casoCAD mesh is one immutable, uncompressed Arrow IPC **File** with schema
identity `casocad.casomesh.arrow.v3`. Version 2 and MeshIR artifacts are
rejected; v3 has no compatibility rows or converter.

All batches use the nullable superset schema returned by
`caso_meshing::arrow_schema()`. Each batch contains exactly one `row_kind`,
and sections appear in this order:

1. `catalog`
2. exact self-contained spatial chunks (`point`, `edge`, `face`, `cell`)
3. `preview_point` and `preview_element`
4. `spatial_node`
5. `batch_directory`
6. one final `manifest`

Exact chunk batches have no more than 65,536 top-dimensional elements and a
32 MiB decoded estimate. Every element references point copies in the same
chunk. A point has one owner row and may have ghost copies; only its owner row
contributes to the logical point count. Entity IDs are opaque `u64` values
allocated from the owning chunk and are not contiguous.

The directory records batch index, row kind, spatial-node ID, bounds, row
count, decoded estimate, element types, zones, and tags. It is the sole batch
directory; the manifest does not duplicate it as JSON. The manifest stores
schema identity, dimension, logical counts, generator ID, settings/control
metadata, bounds, spatial root, and the batch range of every section. It does
not contain a self-referential file length.

Every exact chunk is a quadtree/octree leaf. Nodes persist deterministic
bottom-k boundary and volume/surface previews (up to 256 of each class), so
zoomed-out rendering reads preview batches rather than exact leaves.

`MeshReadSession` uses Arrow's `FileReader` and indexed record-batch access.
Native input is a `File`; browser input is a `Cursor<Arc<[u8]>>`. Opening
validates only the Arrow schema/footer, final manifest, directory, catalog,
and spatial tree. Exact entity data is validated on first access. A full
cancellable audit is available separately.

Generation calls the same `run_meshing(request, storage)` pipeline on both
platforms. `NativeFileStorage` fsyncs and validates a sibling candidate before
atomic replacement. `MemoryStorage` writes through a hard-capped buffer and
publishes `Arc<[u8]>`.
