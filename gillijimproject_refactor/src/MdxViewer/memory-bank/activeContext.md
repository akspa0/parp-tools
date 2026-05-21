# Active Context — MdxViewer / AlphaWoW Viewer

This file is intentionally compressed. Keep only the current compatibility role, the latest validated behavior, and the live boundaries here.

## Role

- `MdxViewer` is the legacy or compatibility host.
- Do not use it as the design owner for new `wow-viewer` architecture unless the task explicitly targets the old viewer, terrain archaeology, or extraction of working behavior into `wow-viewer` libraries.

## Current Validated Snapshot

### Weak-signal terrain restore

- The viewer has explicit whole-tile and per-chunk restore modes.
- Per-chunk restore can target selected chunks and use texture-tied sub-cell guidance.
- The shadow-lift heuristic remains opt-in and experimental.
- Proof level is compile validation only; broad real-data runtime signoff is still missing.

### Runtime-backed M2 viewer path

- The pure `wow-viewer`-backed M2 renderer is the default successful runtime route in `MdxViewer`.
- Viewer-side skeletal animation playback is proven for the bounded wolf repro.
- Standalone character customization and projected-heavy doodad cull fixes landed for the runtime path.

### PM4 compatibility surface

- `MdxViewer` now consumes a `wow-viewer` PM4 `MSHD.Field04` region-id seam for overlay coloring, workbench diagnostics, collection/export metadata, and PM4 cache payloads.
- The PM4 selection workbench now also exposes selected-region peer summaries and can export an LLM-oriented evidence bundle (`json` + `md` + `svg`) derived from the currently visible overlay objects.
- This is a grouping/debug aid only. It does not change PM4 placement math or match scoring.

## Important Boundaries

- Do not claim full M2 parity yet.
- Remaining M2 gaps still include broader character-family coverage, projected or additive material behavior, particles, ribbons, and fuller native-material parity.
- Do not treat terrain-restore heuristics as broadly proven until real-data runtime validation exists.

## wow-viewer Terrain Adapter Status

- `AlphaTerrainAdapter` (in `wow-viewer/src/core/WowViewer.Core.IO/Maps/`) now implements `ITerrainAdapter` and bridges `AlphaWdtReader` output → per-chunk `TerrainChunkData`.
- `AlphaTileData.ToTileLoadResult()` converts flat 257×257 heightmaps and alpha packs to `TileLoadResult` with `MddfPlacement`/`ModfPlacement`.
- `TerrainTileTensorPack.ToTileLoadResult()` converts LK flat-array format to the same `TileLoadResult` shape.
- `AlphaWdtReader` and `AlphaWdtWriter` in `wow-viewer` are now the canonical alphaWDT read/write owners. Future MdxViewer alphaWDT work should consume those shared contracts, not add another parser or writer.
- `AlphaEmbeddedAdtReader` is compatibility-only. Keep it aligned with the shared reader until consumers move off it, but do not deepen it as the design owner for alphaWDT semantics.
- If MdxViewer needs new alphaWDT behavior later, land it in `wow-viewer` shared I/O first and then bridge it into the viewer.

## Routing Reminder

- If the task is new renderer or runtime ownership, move it into `wow-viewer` libraries.
- If the task is a bounded compatibility hotfix or archaeology pass, `MdxViewer` is the right surface.
