# wowviewer-data-harvester

Dataset generation, model training, and inference for WowViewer V14 terrain AI.

## Quick Start

```powershell
cd wow-viewer/data-harvester
uv sync
```

## Shard Format

Input NPZ shards from `WowViewer.Tool.Harvest` contain these arrays:

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `minimap_rgb_256` | 256×256×3 | uint8 | Ground-truth minimap pixels |
| `height_257` | 257×257 | float32 | MCVT vertex heights |
| `mcal_alpha_pack_256` | 256×256×4 | float32 | Alpha blend weights (0-1) |
| `mcly_texture_ids` | 16×16×4 | int32 | Texture IDs per chunk layer |
| `mcly_layer_mask` | 16×16×4 | bool | Active layer flags |

## Models

| Model | Input | Output | Purpose |
|-------|-------|--------|---------|
| D1 (Decompose) | minimap_rgb_256 | tileset + alpha + residual | Decompose minimap into known tilesets + residual |
| R1 (Reconstruct) | residual | height_257 + hole + liquid | Predict terrain from residual |

See `docs/architecture/v14-model-and-refactor-plan-2026-05-06.md` for full model specs.