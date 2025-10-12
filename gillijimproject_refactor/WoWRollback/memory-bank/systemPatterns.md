# System Patterns — WoWRollback (Rewrite)

## Session layout
- **[structure]** `parp_out/session_YYYYMMDD_HHMMSS/`
- **[subdirs]** `01_dbcs/`, `02_crosswalks/`, `03_adts/`, `04_analysis/`, `05_viewer/`, `logs/`, `manifest.json`.

## Overlay conventions
 - **[paths]** `05_viewer/overlays/<version>/<map>/combined/tile_r{row}_c{col}.json`.
```json
{
  "layers": [
    {
      "version": "<version>",
      "kinds": [
        { "kind": "M2",  "points": [ { "world": {"x":0,"y":0,"z":0}, "pixel": {"x":0,"y":0}, "assetPath":"...", "fileName":"...", "uniqueId":123 } ] },
        { "kind": "WMO", "points": [ /* same shape */ ] }
      ]
    }
  ]
}
```

## Viewer index
 - **[file]** Root `05_viewer/index.json` is built by scanning overlays on disk.
 - **[shape]** `{ comparisonKey, versions, defaultVersion, maps[{ map, tiles[{ row, col, versions }] }] }`.

## Placements probing
1. `04_analysis/<version>/objects/<map>_placements.csv`
2. `<adtOutputDir>/analysis/csv/<map>_placements.csv`
3. `<adtOutputDir>/analysis/csv/placements.csv` (legacy)

## Overlay sources (fallback)
 - **[prefer]** Use `04_analysis/<version>/master/<map>_master_index.json` when present.
 - **[else]** Use `<map>_placements.csv` and group rows by tile to emit layered schema.

## Viewer fetch paths
 - **[state.js]** Viewer requests `overlays/<version>/<map>/<variant>/tile_r{row}_c{col}.json` with `variant='combined'`.

## World→pixel transform
 - **[constants]** `TILE_SIZE = 533.33333`, `MAP_HALF_SIZE = 32 * TILE_SIZE`.
 - **[anchor]** NW corner per tile; `worldXNW = MAP_HALF_SIZE - (col * TILE_SIZE)`, `worldYNW = MAP_HALF_SIZE - (row * TILE_SIZE)`.
 - **[scale]** `(dx,dy)` to 512x512 pixels and clamp.

## Error handling
 - **[skip-if-missing]** Optional inputs may be absent; proceed with what exists.
 - **[log-and-continue]** Parse failures logged per-tile; do not fail the entire stage.
 - **[deterministic outputs]** Paths and filenames are stable; never shift targets.

## Archaeological Data Model
The WoWRollback tool treats game data as archaeological layers:

```
Modern WoW (LK ADT files)
    ↑ Evolution/Preservation
Alpha WoW (Alpha WDT files)
    ↑ Original Development
```

## UniqueID as Historical Artifacts
- **Volume of Work**: Each contiguous range represents a discrete work session
- **Singleton Artifacts**: Isolated IDs showing experiments, tests, or technological changes
- **Gap Analysis**: Spaces between ranges indicate development phases or technological shifts
- **Clustering Strategy**: Preserve outliers rather than filtering them (unlike typical clustering)

## Core Services Architecture

### Alpha Analysis Path
```
Alpha WDT Files → AlphaWdtAnalyzer → UniqueID Extraction → Archaeological Clustering → alpha_ranges_by_map_<map>.csv
```

### LK Analysis Path (Existing)
```
LK ADT Files → AdtPlacementAnalyzer → UniqueID Extraction → Range Analysis → lk_ranges_by_map_<map>.csv
```

### Comparison Path (Future)
```
Alpha CSV + LK CSV → EvolutionAnalyzer → Content Preservation Analysis → evolution_report_<map>.csv
```

## Data Flow Patterns

### Input Sources
- **Alpha WDT Files**: Raw game client files from ~2000-2004
- **Converted LK ADT Files**: Output from AlphaWDTAnalysisTool conversion process
- **Configuration Files**: User-defined range filters for rollback functionality

### Output Artifacts
- **Per-Map Range CSVs**: Detailed UniqueID range data by map
- **Archaeological Reports**: Analysis of content preservation/evolution
- **Rollback Configurations**: JSON/YAML configs for selective content removal

## Error Handling Patterns
- **Missing Files**: Skip-if-missing approach for optional analysis
- **Parse Failures**: Log and continue rather than abort entire analysis
- **Invalid Ranges**: Preserve all data, mark questionable ranges for manual review

## Processing Patterns
- **Batch Processing**: Handle entire map collections
- **Session Management**: Timestamped output directories
- **Incremental Analysis**: Support partial re-runs on updated data

## Integration Patterns
- **Reuse AlphaWDTAnalysisTool Logic**: Leverage existing WDT parsing capabilities
- **Standalone Operation**: Independent tool that doesn't modify source tools
- **Configurable Thresholds**: Archaeological significance parameters (gap sizes, minimum cluster sizes)

## Visualization & Diff Pipeline (2025-09-29)
- **Overlays**: Per-tile JSON grouped by version (sediment layer), containing plotted points with normalized and pixel coords, plus kit/subkit.
- **Diffs**: Per-tile JSON between two versions with Added/Removed/Moved/Changed sets. Matching by `asset_path` + spatial proximity; UIDs may change.
- **Minimap**: Compose PNGs from BLP via `lib/wow.tools.local` adapter. World→tile→pixel transform per ADT v18.
- **Static Viewer**: Portable HTML/JS/CSS copied under comparison output; loads overlays/diffs on demand; version switch defaults to 0.5.3 when available.
- **Tiny-File Rule**: Each service in its own small file (≤~150 LOC): `CoordinateTransformer`, `MinimapComposer`, `OverlayBuilder`, `OverlayDiffBuilder`, `ViewerReportWriter`.
