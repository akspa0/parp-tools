# Quickstart: PM4 Asset Matching

## Goal

Run the future automation lane end-to-end:

1. export PM4 object segments
2. build or reuse staged-asset signal corpora
3. rank WMO/M2 candidates automatically
4. synthesize replacement placement proposals for missing tiles

## Prerequisites

- staged client roots under `I:/parp/parp-tools/output/tmp/wowarchive-clients/`
- PM4 development corpus or another trusted PM4 input root
- `wow-viewer` buildable with `.NET 10`
- Python tooling run only through `wow-viewer/data-harvester` with `uv`

## Current Commands

### 1. Export PM4 Segment Report

The current export report is no longer a thin identity dump. Each segment now
includes PM4-derived bounds, footprint hull, area, height stats, anchor
signals, surface-family histogram, confidence flags, and the current
WMO/M2-eligibility prediction.

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- `
  pm4 export-segments `
  --input i:/parp/parp-tools/wow-viewer/test_data/development/World/Maps/development `
  --output i:/parp/parp-tools/wow-viewer/output/datasets/pm4_asset_matching/development.pm4-segments.json
```

Bounded smoke proof now exists for the real tile:

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- `
  pm4 export-segments `
  --input i:/parp/parp-tools/wow-viewer/test_data/development/World/Maps/development/development_00_00.pm4 `
  --output i:/parp/parp-tools/wow-viewer/output/tmp/pm4-export-segments-rich-smoke.json
```

### 2. Run Validation-Tile Automated Matching

The first implemented US2 slice uses a real `_obj0.adt` placement file as a
bounded validation asset-reference surface. This is enough to score PM4
segments automatically against known WMO/M2 placements and inspect the ranking
breakdown before the later Zarr corpus lane is finished.

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- `
  pm4 match-assets `
  --input i:/parp/parp-tools/wow-viewer/test_data/development/World/Maps/development/development_00_00.pm4 `
  --placements i:/parp/parp-tools/wow-viewer/test_data/development/World/Maps/development/development_0_0_obj0.adt `
  --archive-root i:/parp/parp-tools/output/tmp/wowarchive-clients/3_3_5_12340 `
  --max-candidates 5 `
  --output i:/parp/parp-tools/wow-viewer/output/tmp/pm4-match-assets-smoke.json
```

Bounded smoke proof now exists for that command:

- output: `wow-viewer/output/tmp/pm4-match-assets-smoke.json`
- current run shape:
  - `4110` PM4 segments scored
  - `25` validation WMO/M2 references
  - each segment now carries:
    - expected asset kind (`wmo` or `m2`) when matchable
    - explicit `matched` / `ambiguous` / `unresolved` / `ineligible` status
    - score breakdown per candidate
    - rationale explaining why a segment is unresolved or ambiguous

### 3. Build Durable Asset Signal Corpus

```powershell
cd i:/parp/parp-tools/wow-viewer/data-harvester
uv run scripts/build_pm4_asset_signal_corpus.py `
  --client-root ../output/tmp/wowarchive-clients/<build> `
  --output ../output/datasets/pm4_asset_matching/<build>.asset-signals.zarr
```

### 4. Run Corpus-Scale Automated Matching

```powershell
cd i:/parp/parp-tools/wow-viewer/data-harvester
uv run scripts/validate_pm4_asset_matching.py `
  --pm4-zarr ../output/datasets/pm4_asset_matching/development.pm4-segments.zarr `
  --asset-zarr ../output/datasets/pm4_asset_matching/<build>.asset-signals.zarr `
  --output ../output/datasets/pm4_asset_matching/development.match-report.json
```

### 5. Synthesize Replacement Placements

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- `
  pm4 synthesize-placements `
  --match-report i:/parp/parp-tools/wow-viewer/output/datasets/pm4_asset_matching/development.match-report.json `
  --target-tiles 30_48,30_49 `
  --output i:/parp/parp-tools/wow-viewer/output/datasets/pm4_asset_matching/development.replacement-placements.json
```

## Validation Expectations

- segment export completes without freezing the primary viewer shell
- the current exported report is JSON-backed and already carries usable PM4 geometry/matchability evidence
- bounded validation matching can rank real `_obj0.adt` WMO/M2 placements before the Zarr corpus lane is finished
- every eligible PM4 segment has a ranked candidate list or an explicit unresolved state
- replacement placement proposals carry provenance back to PM4 segments and chosen candidates
- validation runs against known placed tiles can measure whether the expected asset appears in the top-ranked candidate set
