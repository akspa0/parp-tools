# Progress — v0.4.9 Clean Branch

## Current Position

- Branch `v0.4.9` created from `ced5899` — clean restart after archive-backed extraction broke everything
- v10 terrain AI pipeline works in filesystem mode only
- All MCAL/MCLY/Wave 2 infrastructure is present and validated on development tiles
- v10.2 pipeline (build_v10_2_dataset.py, train_v10_2_terrain_synth.py, archive-backed minimap loading) was added after ced5899 and is BAD STATE — do NOT use

## Validated Milestones (at bd585dd through ced5899 commit range)

- NPZ serialization for TerrainTileTensorPack (bd585dd)
- Object-anchored 3D brush pattern extraction (f125fa5)
- ArchiveCatalogSession for reusable archive reads (02c4ff4)
- Minimap RGB extraction and Stage 1 trainer (e05cd46)
- dataset-build-v10-stage1 command (2c423ae) — filesystem mode ONLY
- MCLY texture-layer combination mining with texture names (b191030)
- MCAL composition mining (fcc60a8)
- MCAL brush-stroke mining (44d3688)
- Height profile clustering (c49ea8c)
- Prefab-cell clone detection (50509dc)
- MCLY label manifest generation (919f0a6)
- Stage 2 terrain synthesis trainer (4cc347c)
- PM4 placeholder tile support (e667775)
- U-Net decoder training arguments (ced5899)

## Known Working Extraction Command

```
wowviewer-converter dataset-build-v10-stage1 --input-dir <adt_dir> --minimap-root <minimap_dir> --output-dir <out_dir> --limit 64
```



## What Broke (Post-ced5899, NOT inherited by v0.4.9)

- Archive-backed tile enumeration (EnumerateArchiveTileSources) returns 0 tiles for --client-root mode
- MpqArchiveCatalog.FindFileInArchive stops probing on empty hash slots — files behind empty slots invisible
- StormLibPatchArchiveReader fallback is dead code (DllNotFoundException)
- build_v10_2_dataset.py orchestrates broken archive extraction
- train_v10_2_terrain_synth.py cannot train without minimap shards

## Open Boundaries

- No all-client/all-map corpus exists yet
- Only development-map extraction (64 tiles) is currently proven
- Alpha-era client (0.5.3) list-maps returns 0 maps — monolithic WDT format not handled
- Cata 4.0.0.11927 has known MPQ format issues (pre-beta build with few minimaps)
- Wrath 3.3.5 archive-backed extraction hangs — needs MpqArchiveCatalog probe fix
- Only filesystem minimap mode works reliably — needs pre-extracted PNGs
