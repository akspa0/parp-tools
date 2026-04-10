# WoW Height Regressor V7.1

## Scope

V7.1 is a multichannel terrain regressor, not a pure minimap-to-height model.

It is also not the right long-term owner for texture-layer decomposition.
Alpha-mask recovery and tileset separation are a second problem: taking a
rendered minimap tile and inferring the underlying base texture plus the three
overlay alpha masks that produced it. We already have the supervision needed for
that job in the exported terrain data, but it should be trained as a separate
model rather than folded into the terrain-height path.

That distinction matters. The minimap does not directly encode enough elevation
signal to reconstruct valid terrain on its own, especially once terrain has been
flattened or visually overwritten by roads, buildings, doodads, liquids, or
other object-driven surface changes. The model works by combining the minimap
with auxiliary terrain signals that disambiguate those losses.

The most important auxiliary source is WDL. For many maps, the practical data
surface is not "full ADTs or nothing". We often have minimaps and WDL files even
when complete ADT-derived heightmaps are missing. That means the model must be
able to learn both:

1. the correlation between minimap tiles and WDL-derived low-resolution terrain
2. the correlation between minimap tiles and full heightmap terrain where the
  higher-resolution teacher exists

That is what closes the reconstruction loop. If we only train against full
heightmaps, then maps with only minimap+WDL data remain dead ends. If we train
with WDL as an explicit prior and correlation target, then low-resolution terrain
is always recoverable and the higher-resolution terrain path becomes an extension
of that same learned relationship rather than a separate dead-end hack.

## Input Contract

V7.1 uses 11 input channels:

| Ch | Name | Source | Purpose |
|----|------|--------|---------|
| 0-2 | Minimap RGB | `images/{tile}.png` | Surface color, roads, painted terrain, water tint |
| 3-5 | Normal Map RGB | `terrain_data.normalmap` | Local slope and terrain orientation |
| 6 | WDL Height | `terrain_data.wdl_heights` | Low-resolution global elevation prior and fallback teacher when full ADT terrain is unavailable |
| 7 | H_Min Mask | `terrain_data.height_min` | Tile minimum-altitude hint |
| 8 | H_Max Mask | `terrain_data.height_max` | Tile maximum-altitude hint |
| 9 | Water Mask | `terrain_data.liquid_mask` | Flat-water zones and liquid suppression cues |
| 10 | Object Footprint | `terrain_data.objects` + bounds | Terrain flattened or obscured by placed objects |

The object-footprint input is deliberate. If buildings or other placed assets
erase visible terrain cues from the minimap, the model needs an explicit signal
for those regions or it will hallucinate invalid height from painted surface data.

The liquid and object channels are not accidental complexity. They encode known
losses in the minimap image:

- liquids overwrite terrain appearance and often imply broad flat or smoothly
  varying support surfaces
- objects obscure the terrain entirely, so those pixels cannot be interpreted as
  direct terrain evidence

V7.1 treats those areas as explicit "missing or corrupted terrain observation"
regions rather than pretending the minimap still contains raw terrain truth there.

## Output Contract

The intended terrain outputs are:

| Output | Purpose |
|--------|---------|
| Global heightmap | Main world-space terrain surface |
| Local heightmap | Tile-relative detail surface |
| Bounds head | Tile/global range hints |

The primary mesh export path still uses the predicted global height channel.

Current code status: the terrain trainer now predicts only the global/local
height channels plus bounds. Alpha-mask recovery has been split out into the
separate `train_texture_v1.py` workflow.

## Model Shape

- Architecture: 5-level U-Net
- Input resolution: `512x512`
- Output resolution: `512x512`
- Input channels: `11`
- Output channels: `2`
- Auxiliary head: 4-value bounds predictor

## Loss Stack

V7.1 is trained against a weighted combination of:

- Global height L1
- Local height L1
- Bounds MSE
- Gradient consistency
- SSIM structure loss
- Edge-preservation loss

This stack is part of why the model was usable. The goal is not just a smooth
height field; it is a terrain surface that stays coherent around object-driven
flattening, sharper terrain breaks, and paint-layer transitions.

## Reconstruction Strategy

The intended terrain-rebuild strategy is layered:

1. Use minimap + WDL to guarantee a low-resolution terrain reconstruction path
  on maps where ADT-derived full terrain is missing.
2. Use minimap + WDL + known-loss channels to learn how full heightmaps differ
  from that low-resolution prior when complete teacher data exists.
3. Carry liquid and object-placement signals explicitly so the model can reason
  about minimap regions that are visually dominated by non-terrain content.

That is why WDL and the known-loss masks belong in the model. They are not
cheating; they are the exact data-surface bridge that lets the system generalize
from "we have everything" to "we only have minimap plus fallback terrain".

## Separate Texture-Decomposition Model

The alpha-mask and texture job should be separated from V7 terrain recovery.

That second model should learn correlations between:

1. the rendered minimap tile
2. the terrain texture palette used on the tile
3. the grayscale alpha-mask bits that blend those textures together

The exported ML dataset already provides the necessary supervision surfaces:

- `terrain_data.textures` tells us which terrain textures were assigned
- `terrain_data.chunk_layers` exposes the layer metadata per chunk
- `terrain_data.alpha_masks` gives explicit grayscale blend masks
- `terrain_data.alpha_atlas` gives compact packed alpha transport
- `tilesets/` provides decoded texture sources that can be correlated against
  the minimap appearance

For the current development-map work, that is a good second-stage target because
the texture palette available in the `4.0.0.11927` client is relatively limited.
That constrained palette should make it easier to learn the correlation between
minimap color or pattern and the underlying terrain-texture stack before trying
to generalize to broader later-era palettes.

In other words:

1. V7 terrain model: minimap + WDL + known-loss channels -> terrain heights.
2. Texture model: minimap + tileset palette context -> base texture + alpha-mask
   reconstruction.
3. Shadow-scar object model: minimap + `MCSH` evidence + surviving placements ->
  missing-object candidates and restored placement hypotheses.

Keeping those jobs separate should make both models easier to reason about,
train, and validate. The third model matters because some minimap-era object
state may survive only as terrain-shadow evidence after placements were edited
or removed.

## Development-Map Dataset Guidance

For the current development-map target, the preferred training mix remains:

1. `3.0.1` Northrend first
2. `4.0.0.11927` LostIsles as a supplement

The current trainer supports explicit dataset-root selection as well as the
`development-map` auto-discovery profile used by the viewer training UI.

### Explicit roots

```bash
cd src/WoWMapConverter/scripts
python train_v7.py \
  --dataset-root "PATH_TO_301_NORTHREND_DATASET" \
  --dataset-root "PATH_TO_400_LOSTISLES_DATASET" \
  --include-map Northrend \
  --include-map LostIsles \
  --output-dir ./vlm_output_v7_devmap
```

### Auto-discovery profile

```bash
cd src/WoWMapConverter/scripts
python train_v7.py --profile development-map
```

That profile looks for dataset roots whose folder names indicate:

- `3.0.1` plus `Northrend`
- `4.0.0.11927` plus `LostIsles`

If your local folder names do not encode those markers cleanly, use explicit
`--dataset-root` arguments instead.

## Training Outputs

Training writes:

- `best.pt`
- `checkpoint.pt`
- `training_log.json`
- preview images under `previews/`

The viewer training window reads `training_log.json` directly, so the in-viewer
loss plots and the notebook export both track the real V7.1 loss components.

## Inference Outputs

`infer_v7.py` writes:

- `tile.obj`
- `tile.mtl`
- `tile_height.png`
- `tile_height_meta.json`
- optional `tile_debug.png`
- optional `tile_bounds.json`

The main OBJ export still comes from the predicted global height channel after
optional smoothing or post-scale adjustments.

## Boundary

This restores the real V7.1 contract instead of the broken minimap-only reset,
and the terrain trainer no longer mixes alpha prediction into that core model.
It does not by itself prove runtime quality on the development map. Real proof
still requires running training against the intended dataset roots and checking
the exported terrain results on real tiles.