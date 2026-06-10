# Implementation Plan: WMO Minimap BLP Harvest and Asset Signal

**Branch**: `029-wmo-minimap-signal` | **Date**: 2026-05-30 | **Spec**: `specs/029-wmo-minimap-signal/spec.md`

**Input**: Feature specification from `/specs/029-wmo-minimap-signal/spec.md`

## Summary

Harvest WMO minimap BLPs from staged game client MPQ archives using the naming pattern confirmed via Ghidra RE (`<WMOName>_<groupIdx>_<quadY>_<quadX>.blp` under `Textures\Minimap\`), decode them to RGB, resolve asset provenance, and store results in a Zarr-compatible structure alongside the existing object roof library. This replaces the DBC-chain approach from the initial draft with the correct filename-pattern discovery approach.

## Technical Context

**Language/Version**: C# / .NET 10 (harvester tool), Python 3.11+ / uv (downstream consumption)

**Primary Dependencies**: WowViewer.Core.IO (MPQ reader, BLP decoder, WMO readers), WowViewer.Tool.Harvest (existing harvest infrastructure), SereniaBLPLib (BLP decode), Zarr v3 / Parquet (output storage)

**Storage**: Zarr arrays + Parquet metadata at `wow-viewer/output/datasets/object_roof_library/<build>/wmo_minimap/`

**Testing**: `dotnet test` for C# unit tests; Python inspection scripts for Zarr/Parquet validation

**Target Platform**: Windows x64 (MPQ reading requires StormLib native DLL)

**Performance Goals**: Complete harvest for one build in under 10 minutes

**Constraints**: No GPU required (pure archive reads); must work with staged clients under `output/tmp/wowarchive-clients/`

**Scale/Scope**: ~6 target builds, potentially hundreds to thousands of WMO minimap BLPs per build

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Repo Independence | PASS | All code under `wow-viewer/`, reads from staged clients |
| II. Library-First | PASS | Core logic in `WowViewer.Core.IO`, tool is thin wrapper |
| III. Real-Data Validation | PASS | Validates against staged client data |
| IV. Residual Model Chain | N/A | No model training in this spec |
| V. Streaming-First | N/A | Not a terrain dataset pipeline; separate WMO asset store |
| VI. No H:\CLIENTS | PASS | Uses `output/tmp/wowarchive-clients/` only |

## Project Structure

### Documentation (this feature)

```text
specs/029-wmo-minimap-signal/
├── spec.md
├── plan.md
└── tasks.md
```

### Source Code (repository root)

```text
wow-viewer/src/core/WowViewer.Core.IO/
├── Wmo/
│   ├── WmoMinimapBlpReader.cs          # NEW: filename parser + BLP enumeration
│   └── WmoMinimapAssetResolver.cs      # NEW: stem→asset path resolution
wow-viewer/src/tools/harvest/WowViewer.Tool.Harvest/
│   └── Commands/WmoMinimapHarvestCommand.cs  # NEW: harvest CLI command
wow-viewer/data-harvester/
├── scripts/
│   └── inspect_wmo_minimap_harvest.py  # NEW: Zarr/Parquet QA script
```

**Structure Decision**: The BLP enumeration and filename parsing logic goes in `WowViewer.Core.IO/Wmo/` (library-first). The harvest command is a thin CLI wrapper in the existing harvest tool. The Python inspection script goes in the data-harvester scripts directory.

## Implementation Phases

### Phase 1: WMO Minimap BLP Discovery and Filename Parsing

**Goal**: Given a staged client root, enumerate all MPQ entries matching `Textures\Minimap\*_*_??_??.blp`, parse each filename into `(wmo_stem, group_index, quad_y, quad_x)`, and produce a list of discovered entries.

**Approach**:
- Add `WmoMinimapBlpReader` to `WowViewer.Core.IO/Wmo/`
- Use existing MPQ file enumeration (StormLib `SFileFindFirstFile` / `SFileFindNextFile`) to scan `Textures\Minimap\` prefix
- Regex or split-based filename parsing for the `<stem>_<groupIdx>_<quadY>_<quadX>.blp` pattern
- Handle build-specific variations (3-digit group index padding seen in Ghidra for build 3368)
- Unit test against known patterns

**Validation**: Run discovery on staged `3_3_5_12340`, confirm at least 100 entries found.

---

### Phase 2: BLP Decode and Asset Path Resolution

**Goal**: For each discovered entry, decode the BLP to RGB and resolve the WMO stem to a full asset path.

**Approach**:
- Use existing `SereniaBLPLib` or `WowViewer.Core.IO` BLP reader to decode each entry
- Add `WmoMinimapAssetResolver` that searches the MPQ file list for `.wmo` roots matching the stem name
- Build a stem→asset_path mapping table from the file list before iterating BLPs
- Collect image dimensions and RGB data per entry
- Handle decode failures gracefully (log, skip, continue)

**Validation**: Decode at least 10 BLPs from `3_3_5_12340`, verify non-zero RGB content and correct asset path resolution.

---

### Phase 3: Zarr Output and Metadata Parquet

**Goal**: Write harvested data to a Zarr-compatible store with image arrays and a queryable Parquet metadata table.

**Approach**:
- Extend the harvest tool with a `wmo-minimap-harvest` command
- Write images to Zarr array `wmo_minimap_rgb` (N, H_max, W_max, 3) uint8 with padding for variable-size BLPs
- Write metadata Parquet with columns: `asset_path`, `wmo_stem`, `group_index`, `quad_y`, `quad_x`, `blp_path`, `image_width`, `image_height`, `source`, `build`
- Write per-group aggregation: `wmo_group_composites.parquet` with columns: `asset_path`, `group_index`, `quad_count`, `quad_blp_paths` (list), `total_width`, `total_height`
- Output root: `wow-viewer/output/datasets/object_roof_library/<build>/wmo_minimap/`

**Validation**: Load the Parquet with Python/pandas, verify join with `placements.parquet` on `asset_path`.

---

### Phase 4: QA Inspection Script and Cross-Build Validation

**Goal**: Provide a Python inspection script that renders WMO minimap BLPs alongside placement data for visual QA, and run the harvest across all 6 target builds.

**Approach**:
- Add `inspect_wmo_minimap_harvest.py` to `wow-viewer/data-harvester/scripts/`
- Script renders: sampled WMO minimap BLP images, group composite mosaics, overlap with terrain minimap tiles
- Run harvest across all 6 builds: `0_5_3_3368`, `0_5_5_3494`, `0_7_0_3694`, `3_0_1_8303`, `3_3_5_12340`, `4_0_0_11927`
- Collect per-build summary statistics: total BLPs found, unique WMOs, decode success rate

**Validation**: QA script produces visual output for at least 3 WMOs across 2 builds. Per-build summaries show non-zero discovery counts for at least interior-dungeon WMOs.

## Complexity Tracking

No constitution violations. All phases are straightforward archive-read operations with no GPU dependency and no model training.
