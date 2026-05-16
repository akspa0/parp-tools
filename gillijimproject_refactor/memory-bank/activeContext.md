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

- `AdtTensorPackBuilder.BuildObjectMasks` → returns `(float[,], float[,], int[,])` tuple with instance mask
- `AlphaTensorPackBuilder.BuildObjectMasks` → same pattern, added `PaintIntCircle`/`PaintIntRect` overloads
- `TerrainTileTensorPack.ObjectInstanceMask257` → new `int[,]?` property
- `NpzTileSerializer` → writes `object_instance_mask_257` as `<i4`
- Both builders assign instance IDs starting at 1 (0=terrain), MDDF first then MODF

## Python Changes (This Session)

- `build_v16_dataset.py` → now carries 14 Zarr arrays (was 12), adds `object_precise_mask` and `object_instance_mask`, writes `placements.parquet` companion table with per-placement rows + asset_path linkage, index includes `n_mddf`/`n_modf` counts
- `v16_dataset.py` → reads `object_instance_mask` from Zarr, returns int64 `instance_mask` tensor and `has_instance` flag
- `train_v16.py` → unchanged (V16 model doesn't use instance mask yet; will be used by future Model A)

## NOT YET (Blocked on User)
- Full V16 builds for all client builds (rebuild harvester binary first)
- V16 training run
- Object segmentation Model A training script
- Asset vocabulary build
- PM4 cross-reference analysis