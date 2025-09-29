# Viewer & Diff Plan

## Purpose
Build a portable visualization to explore per-ADT-tile sediment layers (versions) and placed objects (UniqueIDs), with a version switch (default 0.5.3) and a robust diff (added/removed/moved/changed). Keep implementation modular with small files.

## UX Goals
- Select a tile on a map and toggle versions (sediment layers) to reveal placed objects.
- Diff between two versions to see added/removed/moved/changed objects.
- Group/Filter by kit and subkit; show tooltips with UID, asset path, kit, subkit.
- Annotations (read-only MVP) to store notes/tags per artifact.

## Data Model Extensions (Core)
- PlacementAsset: add float X, Y (Z optional).
- AssetTimelineDetailedEntry: add X, Y.
- UniqueIdAssetEntry: add X, Y.
- Source: MDDF/MODF records; pipe through VersionComparisonService.

## Coordinate System & Transform
- WoW right-handed; +X north, +Y west; Z is up.
- ADT grid: 64x64 tiles; side length S = 533.33333 yards.
- Block index from world axis: block = floor(32 - (axis / S)).
- Local normalized in-tile: localX = (32 - (X / S)) - blockX; localY = (32 - (Y / S)) - blockY.
- Pixels (W x H): px = localX * W; py = (1 - localY) * H.

## Output Artifacts
- Minimap PNG per tile: `viewer/minimap/<Map>/tile_r<Row>_c<Col>.png`.
- Overlay JSON per tile: `viewer/overlays/<Map>/tile_r<Row>_c<Col>.json` containing layers per version and points.
- Diff JSON per tile: `viewer/diffs/<Map>/tile_r<Row>_c<Col>.json` with added/removed/moved/changed.
- Index JSON: `viewer/index.json` (maps, tiles, versions, defaults).
- Config JSON (optional): thresholds, default version.

## Diff Engine
- Match across versions beyond UID:
  - Primary: asset_path match (case-insensitive).
  - Proximity threshold D for positional match (configurable; start ~10 world units).
  - Ties broken by nearest neighbor; future: orientation/scale if available.
- Classification:
  - Added: in B only
  - Removed: in A only
  - Moved: same asset_path matched, position delta > ε (0.5% tile width)
  - Changed: same asset_path matched, metadata (kit/subkit) changed

## Static Web Viewer
- `viewer/` folder copied into comparison output.
- index.html: maps/tiles grid with search.
- tile.html: minimap canvas + right sidebar with Layers (versions) and Diff controls.
- JS modules (<=150 LOC ea): state.js, tileCanvas.js, overlayLoader.js, legend.js.
- Colors: distinct HSL per version; diff colors: Added=green, Removed=red, Moved=orange, Changed=purple.

## CLI
- `compare-versions`:
  - `--viewer-report` generate viewer assets (minimaps, overlays, diffs, static files)
  - `--default-version <ver>` default selection (default: earliest available, prefer 0.5.3)
  - `--diff <A,B>` specify versions to diff (else pick two from provided set)

## Modules & Files (Tiny LOC)
- Core/Services/
  - VersionOrderService.cs
  - CoordinateTransformer.cs
  - MinimapComposer.cs
  - BlpReaderAdapter.cs (Imaging/)
  - OverlayBuilder.cs
  - OverlayDiffBuilder.cs
  - ViewerReportWriter.cs (thin orchestrator)
  - AnnotationsStore.cs (MVP read-only)
- Cli/Program.cs: parse flags and invoke report writer

## Phases
1) Add X/Y capture to models/service; validate on small dataset.
2) Minimap composer (BLP decode via lib/wow.tools.local); world→pixel tests.
3) Overlay JSON per tile grouped by version and kit/subkit.
4) Diff JSON per tile (A vs B); tune thresholds.
5) Static viewer HTML/JS; version switch defaulting to 0.5.3; diff toggles.
6) CLI integration; build end-to-end; smoke test on Arathi & Dun Morogh.
7) Optional annotations.json (read-only display); future edit support.

## Tests
- Golden point transform test for a known placement.
- Diff classification with crafted fixtures covering UID change, position shift, kit change.
- File count sanity checks; empty tiles still generate valid JSON.

## Risks & Mitigations
- BLP variance: prefer local decoder; cache PNGs.
- Coordinate confusion: verify with known tiles; add debug overlay crosshair.
- Large outputs: lazy-load overlays per tile in viewer.

## Defaults
- Minimap tile size: 512x512 pixels (tunable).
- Default version: 0.5.3 if present; else earliest chronologically.
- Proximity D: 10 world units initial; ε (moved) = 0.5% tile width.
