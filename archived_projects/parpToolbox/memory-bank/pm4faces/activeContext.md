# Active Context

Focus: Object-first assembly for both tiles and merged render mesh; optional snap-to-plane preserved.

- Status: Implemented per-object assembly using `MSUR.IndexCount` as the object id.
  - Tiles: `ExportTiles()` assembles from per-object meshes to avoid cross-object dedup.
  - Merged render mesh: `ExportRenderMeshMerged()` builds a single OBJ using the same per-object method.
  - Vertex mapping: fresh map per object; with `--snap-to-plane`, mapping keys include surface id to avoid cross-surface mixing.
  - Stackalloc removed in hot paths to reduce stack pressure.
- Flags guidance: preserve anchors with `--ck-min-tris 0`; use `--render-mesh-merged` to emit unified scene alongside objects/tiles; `--no-mscn-remap` recommended for face completeness.
- Snap-to-plane: optional; `--height-scale` retained (default 1.0) for MSUR Height experiments.
- Next: Region-scale validation on real data; keep docs in sync; no extra diagnostics unless requested.
