# AI Assistant Guidelines for GillijimProject — v0.4.9 Branch

## Branch Status

- Current branch: `v0.4.9`, based at commit `ced5899`
- Last working commit reference: `bd585dd`
- v0.4.9 is a CLEAN RESTART from before the broken archive-backed extraction work

## What Works (filesystem mode only)

- dataset-build-v10-stage1 --input-dir <adt_dir> --minimap-root <minimap_dir> --output-dir <out_dir>
- extract-v10-tensors --input <root.adt> --minimap-root <dir> --output <npz>
- mine-v10-brushes, mine-v10-mcly, mine-v10-mcal-compositions, mine-v10-mcal-brushes, mine-v10-height-profiles, mine-v10-prefab-cells
- label-v10-mcly
- All training scripts: train_v10_stage1_minimap2height, train_v10_stage2_terrain_synth, train_v10_minimap_to_mclay, train_v10_minimap_to_mclay_grid
- curate_v10_training_shards.py

## What Does NOT Work (archive-backed mode, DO NOT USE)

- dataset-build-v10-stage1 --client-root <path> (hangs, MpqArchiveCatalog probe bug)
- build_v10_2_dataset.py (calls the broken archive extraction)
- train_v10_2_terrain_synth.py (unproven, depends on archive-backed extraction)
- list-maps command for archive tile discovery (hash table probing inconsistent with ReadFile)

## Known Bug In MpqArchiveCatalog (needs fix before archive-backed can work)

FindFileInArchive and TryFindBlockByName in MpqArchiveCatalog.cs stop probing on HashEntryEmpty.
Blizzard's MPQ tools sometimes leave empty slots mid-chain — files behind them are invisible to ReadFile().

Fix: continue past empty slots with 256-entry probe limit.
Diagnostic: MpqProbePastEmptyHitCount counter in MpqDiagnostics.

## Working Data Paths

- Development ADTs: `test_data/original_development/World/Maps/development`
- Development minimap PNGs: `datasets/original_development/development/images/`
- Staged clients (for reference, NOT for archive-backed extraction): `output/tmp/wowarchive-clients/`
- Fixed local clients: `H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft`

## Build

```powershell
dotnet build .\wow-viewer\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -c Debug
```

## Priority

1. Get a trained model checkpoint — use filesystem development shards (64 tiles)
2. Fix MpqArchiveCatalog probe bug so archive-backed extraction works
3. Pre-extract minimap PNGs from client MPQs for broader corpus
4. Run extraction against 3.3.5, 3.0.1, 0.7.0, 0.5.3 clients
