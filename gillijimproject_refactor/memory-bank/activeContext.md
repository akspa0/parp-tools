# ACTIVE CONTEXT — V16 + Multi-Model Architecture

## Branch: `v0.5.0-dev`

## V16 Consolidated Zarr Dataset (2026-05-16)

Single Zarr store per client build. Data flows from C# harvester via binary pipe → Python Zarr writer. No intermediate NPZ files on disk.

### Pipeline
- `WowViewer.Tool.Harvest harvest-stream` → NPZB length-prefixed binary blobs on stdout
- `build_v16_dataset.py build --build <key>` → reads pipe, writes Zarr + Parquet index + placements Parquet
- `train_v16.py --builds <keys>` → V16Dataset reads Zarr, trains V15Model arch (~27.4M params)

### Zarr Arrays (per tile)
height_257, normal_xyz, normal_mask, alpha_256, holes_16, liquid_mask, liquid_height, **object_mask**, **object_precise_mask**, **object_instance_mask** (NEW), minimap_rgb, shadow_mask, mcly_texture_ids, mcly_layer_mask

### New: `object_instance_mask_257`
Per-pixel instance label: 0=terrain, 1+=placement index (MDDF first, then MODF). Each placement's footprint is stamped with its unique instance ID. This enables per-object segmentation training.

### New: `placements.parquet`
Companion table mapping tile_id → per-placement rows with columns: nameId, uniqueId, posX-Y-Z, rotX-Y-Z, scale, bbMin-Max, instance_type, instance_idx, asset_path. Links instance mask IDs to real model paths.

### Data Flow (full)
```
C# harvester → NPZB pipe → build_v16_dataset.py
  ├── Zarr arrays (14 fixed-shape arrays per tile)
  ├── index.parquet (tile_id, build, map, tile_x/y, height_mean/std, has_* flags, n_mddf, n_modf)
  └── placements.parquet (per-placement rows with asset_path linkage)
```

### Key Files
| File | Purpose |
|------|---------|
| `scripts/build_v16_dataset.py` | V16 build pipeline (streaming → Zarr + placements) |
| `scripts/train_v16.py` | V16 training script |
| `src/harvester/v16_dataset.py` | PyTorch Dataset from Zarr |
| `src/harvester/v15_model.py` | V15Model (= V16 model arch) |
| `WowViewer.Core.IO/Maps/AdtTensorPackBuilder.cs` | C# instance mask generation |
| `WowViewer.Core.IO/Maps/AlphaTensorPackBuilder.cs` | Alpha instance mask generation |
| `WowViewer.Core/Maps/TerrainTileTensorPack.cs` | ObjectInstanceMask257 property |
| `WowViewer.Core.IO/Maps/NpzTileSerializer.cs` | Writes `object_instance_mask_257` as int32 |
| `docs/architecture/v16-terrain-model-spec-2026-05-16.md` | V16 full spec |
| `docs/architecture/multi-model-terrain-reconstruction-2026-05-16.md` | Multi-model architecture |

## Multi-Model Architecture (2026-05-16)

Six independent models, each training on ground truth only:

1. **V16 Terrain** (current): minimap → height/normals/alpha/holes/liquid. Uses object_mask for downweighting.
2. **Model A (Object Seg)**: minimap → per-pixel object mask + instance IDs. Ground truth: individually projected `placement_mddf_data`/`placement_modf_data`. **Unblocked now** — instance mask C# code landed.
3. **Model B (Liquid Seg)**: De facto V16 liquid head. Already training.
4. **Model D (Asset Attr)**: instance crop → asset path classification. Ground truth: metadata name tables. **Needs**: global asset vocabulary scan (Gap 2).
5. **Model F (Terrain V2)**: Clean minimap → terrain. Retrains with inpainted objects. **Needs**: Model A + clean minimap pipeline.
6. **PM4 Cross-Ref**: Use Model D predictions on PM4-only tiles to build CK24 → asset mapping. **Needs**: trained Model D.

### Data Gaps
- Gap 1: ✅ Per-instance object mask — **LANDED** in C# harvester
- Gap 2: Global asset vocabulary — needs scan of all tiles, not yet built
- Gap 3: PM4-to-object mapping — deferred, needs Model D
- Gap 4: Clean minimap generation — needs object inpainting pipeline

## C# Changes (This Session)

- `WowViewer.Tool.Harvest` archive-backed extraction now stages `_obj0.adt` beside the root ADT and `_tex0.adt`, so V16/archive harvest output no longer drops placements, object masks, or instance masks on split ADT builds like `3_3_5_12340`
- `AdtTensorPackBuilder.BuildUnifiedLiquid` now uses explicit liquid-presence masks from `MH2O`/`MCLQ` instead of treating `height == 0` as "no liquid", fixing sea-level water loss in unified liquid masks
- `AdtTensorPackBuilder.ExtractMapName` now falls back to the staged tile stem so archive temp extraction still records the real map name in NPZ metadata
- `WowViewer.Tool.Harvest` WL fallback no longer synthesizes fake `World\Maps\<map>\<map>.wl*` paths. It now enumerates actual `*.wlw/*.wlm/*.wlq/*.wll` virtual files from the loaded MPQ listfiles under `World\Maps\<map>\`, caches and parses them once per staged client/map, and reuses them across tiles. Focused `harvest-stream --limit 1` smoke on staged `3_3_5_12340 / Azeroth` now reports `no WL* files found in loaded archives for Azeroth`, so the fallback is currently a real archive-backed no-op for that build/map rather than a naming bug
- `AdtTensorPackBuilder.BuildObjectMasks` → returns `(float[,], float[,], int[,])` tuple with instance mask
- `AlphaTensorPackBuilder.BuildObjectMasks` → same pattern, added `PaintIntCircle`/`PaintIntRect` overloads
- `TerrainTileTensorPack.ObjectInstanceMask257` → new `int[,]?` property
- `NpzTileSerializer` → writes `object_instance_mask_257` as `<i4`
- Both builders assign instance IDs starting at 1 (0=terrain), MDDF first then MODF

## Python Changes (This Session)

- `wow-viewer/data-harvester/scripts/run-data-harvester-python.ps1` remains available as a repo-local fallback when sandboxed agent sessions cannot reach the uv-managed AppData paths, but elevated proof on 2026-05-16 showed both `.venv\Scripts\python.exe` and `uv run` work correctly in a real shell and remain the canonical operator path
- `build_v16_dataset.py` now forwards `harvest-stream` stderr live, prints per-map progress early enough for small maps, and raises explicit errors on truncated headers, bad magic, invalid blob lengths, NPZ decode failures, non-zero harvester exit codes, missing `ENDS`, and zero-tile maps instead of silently `break`ing
- V16 builds now stage into `wow-viewer/output/datasets/v16/<build>.zarr.partial` and only replace the final `.zarr` store after successful finalization; failed runs preserve the partial store and no longer silently leave a poisoned final dataset path with preallocated `50000`-tile arrays
- `build_v16_dataset.py stats` now warns when `index.parquet` is missing or when array length does not match finalized index rows, making interrupted/incomplete V16 stores obvious
- `WowViewer.Tool.Harvest` now exposes `discover-maps --client-root <staged client>` and filters map candidates using the real V16 contract instead of a bootstrap hard-coded map list: pure WMO-only maps (`MWMO/MONM` present, no terrain tiles), zero-tile maps, missing-WDT transport entries, and "terrain but no V16-usable probe tile" maps are skipped, where "usable" currently means the archive probe path can produce both `height_257` and `minimap_rgb_256`
- `build_v16_dataset.py` no longer aborts the whole build when one discovered map produces zero usable V16 tiles at stream time; it now warns and skips that map, while still failing loud if the entire requested build produces zero usable tiles
- `build_v16_dataset.py` → now carries 14 Zarr arrays (was 12), adds `object_precise_mask` and `object_instance_mask`, writes `placements.parquet` companion table with per-placement rows + asset_path linkage, index includes `n_mddf`/`n_modf` counts
- `v16_dataset.py` → reads `object_instance_mask` from Zarr, returns int64 `instance_mask` tensor and `has_instance` flag
- `train_v16.py` → unchanged (V16 model doesn't use instance mask yet; will be used by future Model A)

## NOT YET (Blocked on User)
- Full V16 builds for all client builds (rebuild harvester binary first)
- V16 training run
- Object segmentation Model A training script
- Asset vocabulary build
- PM4 cross-reference analysis
- Rebuild `3_3_5_12340`; `stats` now confirms the current final store is an interrupted pre-finalization output (`50000` preallocated rows, no `index.parquet`). The new builder will replace it atomically on success, while failures stay in `.zarr.partial`
- Full V16 rebuilds can use canonical `uv run` again; the remaining environment caveat is sandbox/AppData access during agent-run validation, not a broken repo-local `.venv`
