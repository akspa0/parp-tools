# Progress

- Implemented: Object-first assembly using `MSUR.IndexCount` as object id for both tiles and merged `render_mesh.obj`.
- Implemented: `--snap-to-plane` (default off), `--height-scale` (default 1.0), corrected projection formula using |N|^2.
- Implemented: Removed stackalloc in hot paths; per-object vertex maps to avoid cross-object dedup; with snapping, per-surface keys used.
- Guidance: Preserve anchors with `--ck-min-tris 0`; recommend `--no-mscn-remap` for face completeness; `--render-mesh-merged` to emit unified scene.
- Validation: Real data runs confirm coherent faces for tiles and merged mesh; quick checks on `surface_coverage.csv` and `objects_index.json` pass.
- Pending: Empirical validation of height scaling (1.0, 1/36, 1/16) on broader regions; keep changes minimal.
