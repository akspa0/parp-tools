# Pipeline Status: Unknown Coordinate Transform

## Current Flow
- **[WDT scan]** `AlphaWdtAnalyzer.AnalyzeAlphaWdt()` gathers placements via `AdtScanner.Scan()`, then optionally replaces coordinates with LK ADT data through `BuildCoordinateLookup()`.
- **[CSV export]** `RangeCsvWriter.WritePerMapCsv()` emits `assets_alpha_<map>.csv` from the analyzer’s `PlacementAsset` list.
- **[Viewer data]** `ViewerReportWriter.Generate()` funnels the same `PlacementAsset` data into `AssetTimelineDetailedEntry`, `OverlayBuilder.BuildOverlayJson()`, and finally the static viewer JSON under `viewer/overlays/...`.
- **[Tile projection]** `CoordinateTransformer.ComputeLocalCoordinates()` maps world coordinates to tile-relative positions used by the Leaflet viewer at `ViewerAssets/js/main.js`.

## Verified Facts
- **[Cached placements OK]** `cached_maps/analysis/.../placements.csv` contains the raw Alpha coordinates we expect.
- **[CLI CSVs OK when raw]** When `--prefer-raw` flows into `AlphaWdtAnalyzer.AnalyzeAlphaWdt()`, freshly generated `assets_alpha_<map>.csv` matches the cached placements (aside from intentional flips).
- **[Viewer still off]** Regenerated overlays show world coordinates shifted by ~17,066 and mirrored axes, implying the analyzer still supplied LK-adjusted numbers during generation.

## Pain Points
- **[Flag propagation]** `rebuild-and-regenerate.ps1` only forwards extra CLI switches via `-ExtraArgs`. Running with `--prefer-raw` directly does **not** reach the analyzer, so the LK override remains active.
- **[Axis flips disagree]** Despite toggling the flip logic (X vs Y) in `AlphaWdtAnalyzer.AnalyzeAlphaWdt()`, viewer overlays did not reflect the changes—suggesting cached or LK coordinates persisted.
- **[Limited validation]** We lack an automated diff between `placements.csv`, `assets_alpha_<map>.csv`, and `viewer/overlays/.../tile_*.json`, making it hard to prove where the wrong values enter.

## Unknowns
- **[Analyzer input]** Whether `AdtScanner.Scan()` already applies a hidden transform before `PlacementAsset` construction.
- **[Viewer caching]** Whether old overlays remain in `rollback_outputs/` or browser cache, hiding the latest regeneration.
- **[CoordinateTransformer expectations]** Whether the tile projection needs a 17,066 offset to align with minimaps (if yes, we should not bake it into world coordinates).

## Next Checks
1. **Instrument analyzer output**: Log one `PlacementAsset` per tile after the optional LK override to confirm the raw vs LK values actually used.
2. **Force raw mode**: Update `rebuild-and-regenerate.ps1` with a `-PreferRaw` switch that always appends `--prefer-raw` when set.
3. **UID trace**: Pick a single `unique_id` and follow it across cached CSV → generated CSV → overlay JSON → viewer popup.
4. **Overlay diff tool**: Build a small script to compare overlay world coords with placements.csv to detect offsets instantly.
5. **Minimap alignment test**: Render a synthetic overlay (single point with known coords) to verify `CoordinateTransformer` math independent of analyzer data.

## Blockers
- No automated guarantee that regeneration uses raw coordinates.
- Lack of unit tests covering the analyzer/viewer pipeline.
- Manual verification is error-prone due to large data volumes.
