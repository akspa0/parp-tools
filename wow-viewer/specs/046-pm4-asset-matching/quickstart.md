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

## Planned Commands

### 1. Export PM4 Segment Signals

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- `
  pm4 export-segments `
  --input i:/parp/parp-tools/wow-viewer/test_data/development/World/Maps/development `
  --output i:/parp/parp-tools/wow-viewer/output/datasets/pm4_asset_matching/development.pm4-segments.zarr
```

### 2. Build Asset Reference Signal Corpus

```powershell
cd i:/parp/parp-tools/wow-viewer/data-harvester
uv run scripts/build_pm4_asset_signal_corpus.py `
  --client-root ../output/tmp/wowarchive-clients/<build> `
  --output ../output/datasets/pm4_asset_matching/<build>.asset-signals.zarr
```

### 3. Run Automated Matching

```powershell
cd i:/parp/parp-tools/wow-viewer/data-harvester
uv run scripts/validate_pm4_asset_matching.py `
  --pm4-zarr ../output/datasets/pm4_asset_matching/development.pm4-segments.zarr `
  --asset-zarr ../output/datasets/pm4_asset_matching/<build>.asset-signals.zarr `
  --output ../output/datasets/pm4_asset_matching/development.match-report.json
```

### 4. Synthesize Replacement Placements

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- `
  pm4 synthesize-placements `
  --match-report i:/parp/parp-tools/wow-viewer/output/datasets/pm4_asset_matching/development.match-report.json `
  --target-tiles 30_48,30_49 `
  --output i:/parp/parp-tools/wow-viewer/output/datasets/pm4_asset_matching/development.replacement-placements.json
```

## Validation Expectations

- segment export completes without freezing the primary viewer shell
- every eligible PM4 segment has a ranked candidate list or an explicit unresolved state
- replacement placement proposals carry provenance back to PM4 segments and chosen candidates
- validation runs against known placed tiles can measure whether the expected asset appears in the top-ranked candidate set
