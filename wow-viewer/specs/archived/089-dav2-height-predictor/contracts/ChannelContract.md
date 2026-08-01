# Contract: V23 Channel Tensor

**Phase 1 contract. Companion to `plan.md`, `research.md`, and `data-model.md`.**

This contract defines the packed encoder input tensor for V23.

## Shape

```text
float32[C, 256, 256]
```

- `C = 15` in `InputMode.full`
- `C = 3` in `InputMode.minimap_only`
- `C = 7` in `InputMode.minimap_alpha`
- `C = 10` in `InputMode.minimap_alpha_normal`

## Canonical Full Order

| Indices | Source | Meaning |
|---|---|---|
| `0..2` | `minimap_rgb` | normalized minimap RGB |
| `3..6` | `alpha_256` | layer alpha weights |
| `7..10` | `mcly_tileset_ids` + prune table | tileset one-hot channels |
| `11..13` | `normal_xyz` | resampled normals |
| `14` | derived | terrain-valid mask |

## Invariants

- Channel order is fixed per checkpoint.
- `InputMode.full` is the only default mode.
- Missing optional source arrays zero-fill the affected channels and mark their `channel_valid_mask` entry false.
- `mcly_texture_ids` are never consumed directly; they must first be resolved through the V22 build-wide tileset ids and the V23 prune table.

## Producer

- `harvester.v23.channels.build_channel_tensor`
- `harvester.v23.dataset.V23HeightDataset`

## Consumer

- `harvester.v23.encoder.DepthAnythingV2SmallEncoder`

## Failure Modes

- Missing `minimap_rgb` is fatal for the sample.
- Missing `alpha_256` or `normal_xyz` is allowed only in degraded modes or explicit fallback paths.
- A prune table mismatch with the selected builds is a configuration error.
