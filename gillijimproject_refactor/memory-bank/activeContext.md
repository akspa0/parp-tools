# Active Context — Clean Restart at v0.4.9

## Current Branch & Base

- Branch: `v0.4.9`
- Base commit: `ced5899` — "fix: add new training arguments and U-Net decoder implementation in terrain synthesis script"
- Last working commit reference: `bd585dd` — "feat: Add NPZ serialization for TerrainTileTensorPack"
- 25 commits between bd585dd and ced5899: all Wave 1 + Wave 2 infrastructure (dataset-build-v10-stage1, MCAL/MCLY readers, miners, trainers)

## What Works (filesystem mode)

- `wowviewer-converter dataset-build-v10-stage1 --input-dir <adt_dir> --minimap-root <minimap_dir>` — proven to extract NPZ shards with minimap_rgb_256
- `wowviewer-converter extract-v10-tensors` — single-tile extraction
- `wowviewer-converter mine-v10-brushes` — anchor-aware brush mining
- `wowviewer-converter mine-v10-mcly` — MCLY combination dictionary
- `wowviewer-converter mine-v10-mcal-compositions` — MCAL composition mining
- `wowviewer-converter mine-v10-mcal-brushes` — MCAL brush-stroke mining
- `wowviewer-converter mine-v10-height-profiles` — height clustering
- `wowviewer-converter mine-v10-prefab-cells` — prefab detection
- `wowviewer-converter label-v10-mcly` — MCLY label manifest
- `train_v10_stage1_minimap2height.py` — Stage 1 trainer
- `train_v10_stage2_terrain_synth.py` — Stage 2 multi-resolution synthesis trainer (proven CUDA runs)
- `train_v10_minimap_to_mclay.py` — Wave 2 classifier
- `train_v10_minimap_to_mclay_grid.py` — Wave 2 chunk-grid classifier
- `curate_v10_training_shards.py` — shard curation

## What BROKE (added after ced5899, DO NOT USE)

- build_v10_2_dataset.py — calls list-maps + dataset-build-v10-stage1 --client-root; hangs on archive-backed extraction
- train_v10_2_terrain_synth.py — separate v10.2 trainer; unproven, no shards with minimap_rgb_256 extracted
- --client-root mode on dataset-build-v10-stage1 — archive-backed minimap loading has MpqArchiveCatalog probe-chain bugs and hangs
- Archive catalog session cache for minimap loading — unproven at scale
- list-maps command for archive tile discovery — hash table probing broke tile enumeration

## Root Cause Of Breakage

Archive-backed minimap BLP loading from MPQ archives was added to dataset-build-v10-stage1 without end-to-end validation. The custom MPQ reader (MpqArchiveCatalog) has hash table probing bugs:
- FindFileInArchive stops probing on HashEntryEmpty — files behind empty probe slots are invisible
- No probe limit when skipping empty slots — caused hangs on large MPQs
- StormLibPatchArchiveReader fallback is a dead codepath (StormLib.dll not present)

The fix (not yet applied to this branch):
- Continue past empty slots with 256-entry probe limit
- Diagnostic counter MpqProbePastEmptyHitCount tracks recoveries

## Filesystem Extraction (the working approach)

Development minimap PNGs: `I:\parp\parp-tools\datasets\original_development\development\images\`
Development ADTs: `I:\parp\parp-tools\gillijimproject_refactor\test_data\original_development\World\Maps\development`

Extraction command:
```
wowviewer-converter dataset-build-v10-stage1 --input-dir <adt_dir> --minimap-root <minimap_dir> --output-dir <out_dir> --limit 64
```

This produces NPZ shards with: minimap_rgb_256, height_257, height_65, height_17, mcal_alpha_pack_256, mccv_rgb, mcnr_normal_xyz, mcly_layer_mask, mcly_texture_ids, hole_mask_16, metadata.json

## Next Steps

1. Build converter: `dotnet build wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug`
2. Extract development shards with minimaps (64 tiles)
3. Train Stage 2 model with those shards
4. For broader client coverage: pre-extract minimap PNGs to disk first, then use filesystem mode
