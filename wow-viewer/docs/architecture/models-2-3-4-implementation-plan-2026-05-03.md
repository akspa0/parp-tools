# Models 2–4 Implementation Plans

## Model 2: MapTexture Compositor

**Goal:** Deterministic renderer. Given MCAL alpha + MCLY texture IDs + tileset BLPs,
produce a flat, unshaded terrain texture (256×256×3 uint8).

### Architecture

```
for each chunk [cy, cx] in 16×16 grid:
  for each layer [0..3]:
    if mcly_ids[cy, cx, layer] >= 0:
      texture = tileset_cache[mcly_names[id]]
      for each pixel in chunk (64×64 sub-region):
        weight = mcal_alpha[global_y, global_x, layer]
        output[global_y, global_x] += texture[sample_y, sample_x] × weight
```

No parameters. No training. Pure compositing.

### Dependencies

1. **Tileset harvesting** — needed BEFORE Model 2 can run
   - Input: NPZ cache manifest → extract unique `mcly_texture_names`
   - For each unique texture path, read BLP from client MPQ via NativeMpqService
   - Decode BLP → PNG using SereniaBLPLib
   - Output: `tilesets/` directory of decoded PNGs

2. **MTEX path resolution**
   - `mcly_texture_names` are relative paths like `World\Texture\...\file.blp`
   - Need to map to actual MPQ paths or harvested files
   - Some textures may have multiple variants (split ADTs vs monolithic)

### Implementation

`scripts/composite_maptexture.py`
- Takes: NPZ shard + tileset PNG directory
- Produces: `maptexture_256.png` (flat texture) per tile
- Produces: `residual_256.png` (real minimap − MapTexture) per tile
- Batch processing via ProcessPoolExecutor

### Validation

- Compare MapTexture vs real minimap side-by-side
- MapTexture should look like the minimap minus shadows, objects, lighting
- Residual should highlight objects (buildings, trees) and shadows

---

## Model 3: Texture Decomposition

**Goal:** Decompose a minimap into its component textures.
Given minimap + heightmap, predict MCAL alpha, MCLY texture IDs, and residual detail.

### Architecture

```
Input:  minimap_rgb (256×256×3) + height_257 (256×256×1, from Model 1)
        = 4 channels

Encoder: Same ConvNeXt V2 Tiny + overlapping stem as Model 1
         (pretrained from Model 1 checkpoint)

Decoder: Same U-Net as Model 1

Heads (from 64ch @ 256×256 decoder features):
  mcal_head:    [Conv3×3→GELU]×2 → Conv3×3 → sigmoid → 4ch @ 256×256
  residual_head: [Conv3×3→GELU]×2 → Conv3×3 → sigmoid → 3ch @ 256×256
  mcly_head:    AdaptivePool→16×16 → Conv1×1→GELU → Conv3×3 → N classes @ 16×16
```

### Training Data

Same NPZ shards as Model 1, augmented with:
1. `maptexture_256` — composited flat texture (from Model 2)
2. `residual_256` — real minimap − MapTexture (from Model 2)

Targets:
- `mcal_alpha_pack_256` — ground truth from NPZ
- `mcly_texture_ids` — ground truth from NPZ
- `residual_256` — ground truth from Model 2 output

### Loss

```
L_total = L_mcal_l1 + 0.5 × L_mcly_ce + 0.3 × L_residual_l1

L_mcal_l1:    L1 loss on alpha weights (only where MCAL present)
L_mcly_ce:    Cross-entropy per 16×16 chunk (ignore unknown textures)
L_residual_l1: L1 loss on residual (objects, shadows, detail)
```

No uncertainty weighting (simpler, since all targets are well-scaled).
No frequency banding (not predicting height).

### Training Schedule

- Pretrained encoder weights from Model 1 checkpoint (freeze first 10 epochs)
- Batch size 16, LR 2e-4, cosine schedule
- 200 epochs (simpler task than height prediction)
- Signal dropout: 0 (no dropout — all inputs always available)

### Inference

```
minimap + heightmap → Model 3 → (MCAL, MCLY, residual)
MCAL + MCLY → Model 4 → tileset BLPs
MCAL + tileset BLPs → Model 2 → MapTexture
MapTexture + residual → final texture
```

### Output Quality Check

- MapTexture should look like correct terrain texturing
- Residual should contain only objects/shadows (not terrain colors)
- Final texture = MapTexture + residual should approximate real minimap

---

## Model 4: Tileset Resolver

**Goal:** Map MCLY texture ID → tileset BLP file on disk.

### Primary: Lookup Table

```
MCLY texture ID → mcly_texture_names[id] → MTEX path → harvested BLP PNG
```

The MTEX paths are stored in the NPZ metadata. At harvest time, we build an index:
```
{
    "texture_name": "World\\Texture\\...\\file.blp",
    "harvested_path": "tilesets/file.png",
    "size": [256, 256],
    "design_kit": "azeroth",
    "era_tag": "335"
}
```

### Secondary: ML Resolver (optional)

For tiles where MCLY data is missing (alpha-era clients, etc.):
- Input: 16×16 pixel region from minimap at chunk position
- Output: softmax over known texture classes
- Architecture: tiny ConvNet (4 conv layers + FC head)
- Training: pairs of (chunk_minimap_pixels, mcly_texture_id) from tiles WITH MCLY
- This is essentially the `train_v10_minimap_to_mclay_grid.py` script's function

### Implementation Order

1. Harvest tileset BLPs (new converter command using NativeMpqService)
2. Build `tileset_index.json` (texture → file mapping)
3. Implement lookup resolver (no ML needed for most tiles)
4. Optional: ML resolver for alpha tiles

---

## End-to-End Training Plan

### Phase 1: Harvest Tilesets (1-2 hours)

```
wowviewer-converter harvest-tilesets `
  --cache-manifest output/tmp/v11_cache/v9_tensor_cache_manifest.json `
  --output-dir output/ml-training/tilesets
```

Reads all unique texture names from the NPZ manifest, extracts BLPs from client
MPQs via NativeMpqService, decodes to PNG.

### Phase 2: Composite MapTextures (overnight)

```
python scripts/composite_maptexture.py output/tmp/v11_cache/ --tilesets tilesets/ --output output/tmp/maptextures
```

For every NPZ shard with MCAL/MCLY data, composites the flat MapTexture and
computes the residual. Adds `maptexture_256` and `residual_256` to each NPZ.

### Phase 3: Train Model 3 (2-3 days)

```
python scripts/train_model3.py output/tmp/v11_cache/ `
  --model1-checkpoint runs/v11.1_prod5/last.pt `
  --output-dir runs/model3_prod1 `
  --epochs 200 --batch-size 16
```

Encoder initialized from Model 1 weights. Decoder trained from scratch.
Simpler task than height prediction — should converge faster.

### Phase 4: End-to-End Inference

```
python scripts/infer_full_pipeline.py `
  --model1 runs/v11.1_prod5/last.pt `
  --model3 runs/model3_prod1/last.pt `
  --tilesets output/ml-training/tilesets/ `
  --input <minimap_or_shard> `
  --output-dir <output> --export-obj
```

Produces: OBJ mesh with composited terrain texture from decomposition.
