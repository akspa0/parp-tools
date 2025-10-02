# WoWRollback

## Overview
WoWRollback is a digital archaeology toolkit for exploring the evolution of World of Warcraft map content. It ingests Alpha-era WDT/ADT exports alongside converted Wrath (LK) ADTs, catalogs placement `UniqueID` ranges, compares versions, and optionally produces rollback-ready assets and a portable viewer for visual analysis.

## Prerequisites
- .NET SDK 9.0 (64-bit)
- Real WoW Alpha/LK data exports produced by `AlphaWDTAnalysisTool` (or equivalent) using the same directory layout described below
- Optional: CASC or filesystem access to minimap BLP/PNG tiles for richer viewer imagery

## Project Layout
```
WoWRollback/
├── WoWRollback.Core/       # Core services and models
├── WoWRollback.Cli/        # Command-line entry point
├── docs/                   # Design notes & plans
├── memory-bank/            # Persistent project context
└── rollback_outputs/       # Default output root (timestamped sessions & comparisons)
```

## Building
```
dotnet build WoWRollback/WoWRollback.sln
```
All commands below assume execution from the repository root via `dotnet run --project WoWRollback/WoWRollback.Cli`.

## Command Reference

### analyze-alpha-wdt
Extract Alpha placement ranges from a single WDT.
```
dotnet run --project WoWRollback/WoWRollback.Cli -- \
  analyze-alpha-wdt \
  --wdt-file path/to/AlphaMap.wdt \
  --out rollback_outputs
```
Outputs land in `rollback_outputs/session_*/<map>/alpha_<map>_ranges.csv` plus supporting ledgers/timelines when available.

### analyze-lk-adt
Analyze converted LK ADTs for a specific map.
```
dotnet run --project WoWRollback/WoWRollback.Cli -- \
  analyze-lk-adt \
  --map Arathi \
  --input-dir path/to/lk/adts \
  --out rollback_outputs
```
Generates `lk_<map>_ranges.csv` and related summaries under the session directory.

### compare-versions
Compare multiple version roots (e.g., Alpha vs LK) and emit CSV/YAML summaries, optional viewer artifacts.
```
dotnet run --project WoWRollback/WoWRollback.Cli -- \
  compare-versions \
  --versions alpha_053,lk_335 \
  --root rollback_inputs \
  --maps Arathi,DunMorogh \
  --yaml-report \
  --viewer-report \
  --default-version 0.5.3 \
  --diff alpha_053,lk_335
```
Key flags:
- `--versions`: Comma-separated list of version folder names under `--root`
- `--maps`: Optional comma-separated map filter
- `--yaml-report`: Emit per-tile YAML summaries (`.../comparisons/<key>/yaml/`)
- `--viewer-report`: Produce minimaps, overlays, diffs, and static viewer config under `.../comparisons/<key>/viewer/`
- `--default-version`: Preferred default layer in the viewer (falls back to earliest version)
- `--diff`: Explicit baseline/comparison pair for diff JSON (defaults to earliest→latest)

Viewer output structure (`.../viewer/`):
- `minimap/<Version>/<Map>/<Map>_<Col>_<Row>.png`
- `overlays/<Version>/<Map>/<Variant>/tile_r<Row>_c<Col>.json`
- `diffs/<Map>/tile_r<Row>_c<Col>.json`
- `index.json`, `config.json`
- Static viewer bundle (HTML/CSS/JS) copied from `ViewerAssets/`

Overlay variants:
- `combined` – all placements for the selected version
- `m2` – MDX/M2 doodads only
- `wmo` – WMO placements only

The viewer UI now exposes:
- Version, map, and overlay dropdowns in `index.html`
- Overlay selector in `tile.html`
- Per-variant marker colors/radii

Regeneration tip: `rebuild-and-regenerate.ps1` writes the new directory layout. Ensure no files under `rollback_outputs/comparisons/<comparison-key>/` are open before running, otherwise CSV locks will abort the CLI.

### dry-run
Simulate rollback filtering using keep/drop configs without writing ADTs.
```
dotnet run --project WoWRollback/WoWRollback.Cli -- \
  dry-run \
  --map Arathi \
  --input-dir path/to/lk/adts \
  --config configs/arathi_keep_ranges.json \
  --out rollback_outputs
```
Reports counts of placements that would be removed per tile and overall.

## Typical Workflow
1. Run `analyze-alpha-wdt` for each Alpha map of interest.
2. Run `analyze-lk-adt` for the converted LK outputs of the same maps.
3. Populate `rollback_inputs/<version>/<map>/...` with the session CSVs or preprocessed data.
4. Call `compare-versions --viewer-report` to generate comparison CSVs, YAML (optional), and viewer assets.
5. Open the viewer bundle (once static assets are in place) to inspect sediment layers, diffs, and annotations.
6. Iterate with `dry-run` (and future rollback commands) to plan selective removals.

## Outputs & Layout
- Sessions: `rollback_outputs/session_YYYYMMDD_HHmmss/`
- Comparisons: `rollback_outputs/comparisons/<comparisonKey>/`
  - `csv/`: Core range, timeline, design kit reports
  - `yaml/`: Optional per-tile YAML when `--yaml-report` is used
  - `viewer/`: Viewer JSON/PNG assets when `--viewer-report` is used

## Troubleshooting
- **ImageSharp vulnerability warnings**: The build presently references `SixLabors.ImageSharp 2.1.9` via `Warcraft.NET`. Upgrade when upstream packages allow, or suppress if running in controlled environments.
- **Missing minimaps**: Ensure the source data contains BLP/PNG tiles or accept placeholder crosshairs.
- **Empty viewer output**: If `AssetTimelineDetailed` is empty for a comparison, the viewer generator returns early—confirm inputs produced placement data.

## Future Enhancements
- Automated minimap sourcing via CASC/file lookup
- Static viewer bundle deployment from `docs/viewer/`
- Rollback APPLY command that rewrites ADTs using keep/drop configurations
- Integration tests on real data fixtures (skip-if-missing)
