# Quickstart: PM4 Asset Matching

## Goal

Run the automation lane end-to-end:

1. export PM4 object segments
2. build or reuse a durable staged-asset signal corpus
3. rank WMO/M2 candidates automatically without requiring a corresponding `_obj0.adt`
4. use `_obj0.adt` matching only as a bounded validation mode when ground truth exists
5. synthesize replacement placement proposals for missing tiles

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

### 2. Build Durable Asset Signal Corpus

The first missing-ADT slice must build a durable candidate corpus keyed by
asset identity rather than placement `UniqueID`. The initial implementation is
a JSON-backed staged-client export; the later Python/Zarr lane can wrap the
same contract for larger corpus runs.

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- `
  pm4 export-asset-signals `
  --archive-root "i:/parp/parp-tools/output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft" `
  --listfile i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt `
  --limit 120 `
  --output i:/parp/parp-tools/wow-viewer/output/tmp/pm4-asset-signals-smoke.json
```

Current bounded smoke proof for this command:

- output: `wow-viewer/output/tmp/pm4-asset-signals-smoke.json`
- current run shape:
  - `102` durable asset signals exported
  - staged client root used: `i:/parp/parp-tools/output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft`
  - current bounded sample is biased by the first validated listfile-backed assets, so it proves the non-ADT workflow mechanically but not candidate quality yet

### 3. Run Durable-Corpus Automated Matching

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- `
  pm4 match-assets `
  --input i:/parp/parp-tools/wow-viewer/test_data/development/World/Maps/development/development_00_00.pm4 `
  --asset-corpus i:/parp/parp-tools/wow-viewer/output/tmp/pm4-asset-signals-smoke.json `
  --max-candidates 10 `
  --output i:/parp/parp-tools/wow-viewer/output/tmp/pm4-match-assets-corpus-smoke.json
```

Current bounded smoke proof for this command:

- output: `wow-viewer/output/tmp/pm4-match-assets-corpus-smoke.json`
- current run shape:
  - `4110` PM4 segments scored
  - `102` durable asset references
  - `0 matched`, `0 ambiguous`, `4095 unresolved`, `15 ineligible`
  - this proves the missing-ADT command path runs without `_obj0.adt`; it does not yet prove the bounded corpus is a good candidate pool

### 4. Run Validation-Tile Matching When Ground Truth Exists

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- `
  pm4 match-assets `
  --input i:/parp/parp-tools/wow-viewer/test_data/development/World/Maps/development/development_00_00.pm4 `
  --placements i:/parp/parp-tools/wow-viewer/test_data/development/World/Maps/development/development_0_0_obj0.adt `
  --archive-root "i:/parp/parp-tools/output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft" `
  --max-candidates 5 `
  --output i:/parp/parp-tools/wow-viewer/output/tmp/pm4-match-assets-validation-smoke.json
```

This is still useful for measuring score quality against known placed assets,
but it is no longer the primary missing-ADT workflow.

### 5. Scale To Zarr Corpus Tooling

```powershell
cd i:/parp/parp-tools/wow-viewer/data-harvester
uv run scripts/build_pm4_asset_signal_corpus.py `
  --client-root "../output/tmp/wowarchive-clients/<build>/World of Warcraft" `
  --asset-zarr ../output/datasets/pm4_asset_matching/<build>.asset-signals.zarr `
  --source-json ../output/tmp/pm4-asset-signals-smoke.json
```

### 6. Synthesize Replacement Placements

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
- durable-corpus matching can run without any corresponding `_obj0.adt` placement file
- bounded validation matching can still rank real `_obj0.adt` WMO/M2 placements before the Zarr corpus lane is finished
- every eligible PM4 segment has a ranked candidate list or an explicit unresolved state
- replacement placement proposals carry provenance back to PM4 segments and chosen candidates
- validation runs against known placed tiles can measure whether the expected asset appears in the top-ranked candidate set
