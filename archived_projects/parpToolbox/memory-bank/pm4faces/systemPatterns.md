# System Patterns

- Grouping Modes: composite-instance (default), type-instance, type-attr-instance, surface, groupkey, composite, render-mesh/surfaces-all.
- Object Identity: `MSUR.IndexCount` is the stable object id for PM4 objects.
- Assembly Strategy:
  - Tiles: assembled from per-object meshes (object-first) to avoid cross-object vertex dedup.
  - Merged render mesh: `render_mesh.obj` built with the same object-first method.
- Vertex Mapping:
  - Fresh map per object to prevent cross-object aliasing.
  - With `--snap-to-plane`, mapping key includes surface id to prevent cross-surface mixing.
- Parity & Centering:
  - Tiles always X-flip; objects follow `--legacy-parity`.
  - `--project-local` recenters geometry consistently across OBJ/glTF/GLB.
- Minimal Flags Guidance:
  - Preserve anchors with `--ck-min-tris 0`.
  - Prefer `--no-mscn-remap` for face completeness.
  - Use `--render-mesh-merged` to emit unified scene alongside other modes.
