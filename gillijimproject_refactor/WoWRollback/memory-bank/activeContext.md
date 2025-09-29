# Active Context - WoWRollback Archaeological Tool

## Current Focus
**Phase 1: Alpha WDT Analysis Implementation**
- Implementing Alpha WDT file parsing to extract UniqueID ranges 
- Building archaeological analysis capabilities to identify "volumes of work" by artists
- Preserving singleton IDs and outliers as important historical artifacts

## Recent Discoveries
- Empty CSV output indicates the tool was incorrectly analyzing LK ADT files instead of Alpha WDT source files
- The tool architecture needs dual analysis modes:
  1. `analyze-alpha-wdt` - Extract ranges from source Alpha WDT files
  2. `analyze-lk-adt` - Extract ranges from converted LK ADT files (existing functionality)

## Current Architecture Issues
- `AdtPlacementAnalyzer` currently parses LK ADT files using `GillijimProject.WowFiles.LichKing` classes
- Need to add `AlphaWdtAnalyzer` that can parse Alpha WDT files directly
- Must leverage existing UniqueID clustering logic from `AlphaWDTAnalysisTool.UniqueIdClusterer`

## Next Steps
1. **Implement Alpha WDT Parser**: Create `AlphaWdtAnalyzer` service that can read raw Alpha WDT files
2. **Preserve Archaeological Artifacts**: Ensure singleton IDs and small gaps are captured, not filtered out
3. **Add CLI Commands**: Implement both analysis modes (`analyze-alpha-wdt`, `analyze-lk-adt`)
4. **Generate Per-Map CSVs**: Output `alpha_ranges_by_map_<map>.csv` and `lk_ranges_by_map_<map>.csv`

## Key Technical Requirements
- Parse Alpha WDT files directly (not converted ADT files)
- Use clustering logic similar to `UniqueIdClusterer.FindClusters()` but preserve outliers
- Each UniqueID range represents a "volume of work" by ancient developers
- Singleton IDs are precious artifacts showing experiments/tests/enhancements

## Archaeological Perspective
We're digital archaeologists uncovering fossilized remains of 20+ year old game development. Every UniqueID tells a story of work performed long ago, and the patterns reveal the sedimentary layers of WoW's evolution from 2000 to present day.

## Update: Version Comparison + YAML Sediment Reports (2025-09-29)
- Implemented per-ADT tile YAML reports under `rollback_outputs/comparisons/<key>/yaml/`.
- Each tile report includes:
  - `sediment_layers` per version (ranges with min/max UniqueID and source file)
  - Kit and subkit tallies derived from asset paths
  - `unique_ids` section correlating assets to matched ranges
  - Aggregate `stats` per tile
- Added CLI flag `--yaml-report` to `compare-versions` to generate YAML alongside CSVs.
- Purpose: enable interactive exploration (sorting/filtering by kit/subkit, tracing borrowed assets across maps/versions).

## Next Focus: Visualization & Diff Viewer (2025-09-29)
- Objective: Build a static web viewer to visualize per-tile sediment layers and diffs between versions.
- Defaults:
  - Default selected version: `0.5.3` (if present; else earliest chronologically)
  - Minimap tile size: 512x512 px
  - Diff thresholds: proximity D=10 world units; moved epsilon=0.5% of tile width
- CLI Additions:
  - `--viewer-report` to generate viewer assets (minimaps, overlays, diffs)
  - `--default-version <ver>` to set initial selection
  - `--diff <A,B>` to compute diff overlays
- Pipeline (tiny modules):
  - `CoordinateTransformer` (world→tile→pixel using ADT v18)
  - `MinimapComposer` (BLP decode via lib/wow.tools.local)
  - `OverlayBuilder` (per-tile per-version JSON)
  - `OverlayDiffBuilder` (Added/Removed/Moved/Changed)
  - `ViewerReportWriter` (copies static viewer, writes assets)
- Matching for diffs (UIDs can change): asset_path equality + spatial proximity, with sensible tie-breakers.
