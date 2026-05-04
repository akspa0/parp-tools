# Terrain Regeneration Pipeline — Architecture

## Model 1: Heightmap (v11) — TRAINING NOW

```
minimap_rgb (3) + optional signals (23, 70% dropout) → height_17/65/257
                                                   → mcal_alpha (side output)
                                                   → mcly_class (side output)
```

Trained with 70% signal dropout — only minimap guaranteed at inference.
Produces: 257×257 heightmap per tile, OBJ mesh, sidecar MCAL/MCLY.

---

## Model 2: MapTexture Compositor (deterministic, no training)

```
mcal_alpha (4ch, 256×256) + mcly_texture_ids (16×16×4) + tileset BLPs
    → MapTexture (256×256×3, uint8)
```

For each chunk (16×16 grid per tile), for each active layer:
  `output[pixel] += tileset_texture[pixel] × mcal_alpha[pixel, layer]`

This is NOT a model. It's a deterministic compositor — the same operation Blizzard
uses to generate in-game map textures. The result is what the terrain looks like
pre-shading: flat texturing without lighting or geometry deformation.

**Training data source for Model 3:** 
  `residual = real_minimap − MapTexture`

The residual contains: geometry shading, baked lighting, objects, shadows, holes.
Model 3 learns to predict this from minimap + heightmap alone.

---

## Model 3: Texture Decomposition

```
minimap_rgb (256×256×3) + heightmap (256×256×1, from Model 1)
    → mcal_alpha (256×256×4, sigmoid)        — supervised by ground truth MCAL
    → mcly_labels (16×16, CE over N classes)   — supervised by ground truth MCLY
    → texture_residual (256×256×3, sigmoid)    — supervised by MapTexture residual
```

**Architecture:** Same ConvNeXt V2 backbone + U-Net decoder as Model 1.
Different heads, shared encoder pretrained from Model 1.

**Loss:**
- MCAL L1 (masked, only when MCAL present in training tile)
- MCLY cross-entropy (per-chunk, ignore unknown IDs)
- Residual L1 (always)

**Training data:** Same NPZ shards as Model 1, with two additions:
1. Harvested tileset BLPs (PNGs from client MPQs)
2. Pre-computed MapTextures (from Model 2 compositor)
3. Pre-computed residuals (real_minimap − MapTexture)

**Inference flow:**
1. Model 1: minimap → heightmap
2. Model 3: (minimap, heightmap) → (MCAL, MCLY, residual)
3. Look up tileset BLPs from MCLY class predictions
4. Model 2: (MCAL, tileset_BLPs) → MapTexture
5. Final texture = MapTexture + residual
6. UV-map onto Model 1's OBJ mesh

---

## Model 4: Tileset Resolver (lookup + optional model)

```
mcly_class + tile_name (map, x, y) → tileset_BLP_path
```

**Simple approach:** Map MCAL texture IDs → MTEX paths → BLP files.
MTEX paths are stored in the NPZ metadata.
BLP files harvested from client MPQs via V10TilesetHarvestCommand.

**Smart approach (optional):** A small classifier that maps pixel regions of the
minimap to tileset textures, for cases where MCLY data is missing or corrupted.

---

## Data Flow At Full Inference

```
minimap (input)
    │
    ├─→ Model 1 (height) → heightmap → OBJ mesh
    │
    └─→ Model 3 (texture decomposition) → MCAL, MCLY, residual
                  │
                  ├─→ Model 4 (tileset lookup) → BLP textures
                  │
                  └─→ Model 2 (MapTexture compositor) → flat texture
                              │
                              + residual
                              │
                              → final texture → UV-map on OBJ
```

---

## Self-Supervised Training Loop

Once Models 1 and 3 are trained:

1. Take ANY minimap tile
2. Model 1 predicts heightmap
3. Model 3 predicts MCAL + MCLY
4. Look up tileset BLPs from MCLY
5. Composite MapTexture from MCAL + BLPs
6. Feed predicted signals BACK as input — the model trains on its own outputs
7. This bootstraps: the model improves by decomposing and recomposing

Essentially an autoencoder over the texture domain, grounded in real MCAL/MCLY
supervision from client data.

---

## Implementation Order

1. **Now:** Let Model 1 finish training (minimap → height, with 70% dropout)
2. **Next:** Harvest tileset BLPs for textures referenced by the training set
3. **Then:** Implement Model 2 (MapTexture compositor) as a Python script
4. **Then:** Pre-compute MapTextures + residuals for all training tiles
5. **Then:** Train Model 3 (texture decomposition) using pre-computed targets
6. **Finally:** End-to-end inference pipeline
