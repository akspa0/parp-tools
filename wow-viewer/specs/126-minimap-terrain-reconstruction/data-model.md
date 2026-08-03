# Data Model: Minimap-to-Terrain Reconstruction Stack

**Feature**: 126-minimap-terrain-reconstruction | **Date**: 2026-08-02

Signals that already exist are listed with their real manifest shapes so this feature reuses them
rather than re-deriving them. New artifacts are marked **NEW**.

## Existing signals reused (v50 store, 0_5_3_3368)

| Signal | Row shape | dtype | Role in this feature |
|--------|-----------|-------|----------------------|
| `minimap_rgb` | [256, 256, 3] | uint8 | Model input (pristine variant, for ablation) |
| `height_257` | [257, 257] | float32 | Height ground truth (MCVT) |
| `normal_xyz` | [257, 257, 3] | float32 | MCNR normals — ground truth for E1's shading law |
| `normal_mask` / `mcnr_mask_257` | [257, 257] | bool | Validity of the normal field |
| `alpha_256` | [256, 256, 4] | float32 | MCAL layer weights — texture decode target |
| `mcly_texture_ids` | [16, 16, 4] | int32 | Layer slot texture identity — decode target |
| `mcly_layer_mask` | [16, 16, 4] | float32 | Which slots are occupied |
| `mcly_tileset_ids` | [16, 16, 4] | int32 | Tileset grouping for tier-2 identity |
| `shadow_mask` | [256, 256] | float32 | MCSH. **Not a target** — measured absent from minimaps (r = -0.006) |
| `liquid_mask` | [256, 256] | float32 | Excludes water from terrain relief scoring |
| `liquid_height` / `liquid_type_256` | [256, 256] | float32 / uint8 | Liquid handling |
| `wdl_outer_17` / `wdl_inner_16` | [17, 17] / [16, 16] | float32 | Absolute elevation source |
| `wdl_outer_present` / `wdl_inner_present` | [17, 17] / [16, 16] | bool | Whether a prior exists for this tile |
| `object_geometry_visible_mask_257` | [257, 257] | float32 | **Occlusion-correct** visible-terrain mask |
| `object_geometry_visible_instance_257` | [257, 257] | int32 | Per-object attribution |
| `object_geometry_visible_source_257` | [257, 257] | uint8 | Which object class occluded |
| `object_precise_mask` | [257, 257] | float32 | Full ground footprint. **Over-masks — do not substitute** |

## New artifacts

### Render passes (C#)

| Artifact | Form | Source | Notes |
|----------|------|--------|-------|
| Textureless residual | 256x256 PNG | exists (`--textureless-residuals`) | Shading only, white albedo, no objects |
| **NEW** Unlit albedo | 256x256 PNG | `--albedo-only` (FR-001) | Real textures, flat lighting, no objects. Symmetric to the residual. |
| DXT1 parity companion | 256x256 PNG | exists (`--dxt1-parity`) | Codec-degraded minimap |

**Invariant**: `albedo (*) shading` must reproduce the full synthetic minimap within tolerance, and
the albedo pass must be *invariant to sun direction*. Both are checked in E2; the second is the one
that is easy to skip and fatal to omit.

### Stores

#### Per-object capture library — **EXISTS, outside the v50 contract**

Built under the spec 077 lineage. One row per capture variant, not per tile.

| Array / table | Role |
|---------------|------|
| `capture_rgb` (N, H, W, 3) uint8 | Object appearance |
| `capture_mask` (N, H, W) uint8 | Object silhouette |
| `capture_alpha` (N, H, W) uint8 | Optional, present only when captured |
| `assets.parquet` | One row per library entry |
| `index.parquet` | One row per capture variant |
| `metadata.json` | Group-level provenance |

Jobs with missing artifacts are retained as `capture_status=not_attempted` rather than dropped —
consistent with the partition-not-filter rule.

**Two open items** (R8, R9):

- Nothing in the v50 store references this library. It must be bound as a **sidecar** — different
  grain (capture variant vs tile), different regeneration cadence — providing a join from
  `object_geometry_visible_instance_257` values to library asset identity. Whether instance IDs and
  asset keys already share a vocabulary is **unverified**; a mapping table may be required.
- Captures predate the lighting corrections. Objects are lit by the same model as terrain, and they
  appear in the model **input**, so stale captures are an input-domain gap. E5 measures whether it
  exceeds the codec noise floor; if it does, a re-render precedes Phase 3.

#### `v126-reconstruction-curriculum-v1` — **NEW**

Row-aligned with `index.parquet`, one row per tile.

| Array | Shape | dtype | Role |
|-------|-------|-------|------|
| `minimap_rgb` | [256, 256, 3] | float32 | Pristine input (ablation only) |
| `minimap_rgb_dxt1` | [256, 256, 3] | float32 | **Default input** — matches authored domain |
| `albedo_256` | [256, 256, 3] | float32 | Decomposition target |
| `residual_256` | [256, 256] | float32 | Shading target — the crux signal |
| `height_257` | [257, 257] | float32 | Height target |
| `visible_mask_257` | [257, 257] | float32 | Loss mask (occlusion-correct) |
| `alpha_256` | [256, 256, 4] | float32 | Texture decode target |
| `mcly_texture_ids` | [16, 16, 4] | int32 | Texture identity target |
| `wdl_outer_17` / `wdl_inner_16` | [17, 17] / [16, 16] | float32 | Absolute datum |

Index fields: `map`, `tile_x`, `tile_y`, `source_group_id`, `split`, `bucket`, `occluded_fraction`,
`has_wdl_prior`, `is_flat`, `is_empty`.

**Store attrs** (top level, because the release gate reads them there — a nested identity dict is
invisible to it): `model_family`, `release`, `schema`, `split_mode`, plus the codec degradation
record.

**Curation**: `bucket` partitions rather than filters. Tiles that are empty, near-fully occluded,
flat, or failed decomposition stay in the store and stay queryable. No row is dropped.

## Entity relationships

```text
minimap_rgb ──DXT1 round-trip──> minimap_rgb_dxt1        [model input]
                                        │
                                        ▼
                            ┌───── decomposition ─────┐
                            ▼                         ▼
                       albedo_256                residual_256
                            │                         │
                    texture decode              height inversion
                            │                         │
                            ▼                         ▼
              alpha_256 / mcly_texture_ids       height_257 (relative)
                            │                         │
                            └──── feeds back ────►    │
                                                      ▼
                                          + wdl_* ──> absolute height ──> mesh
```

Every arrow is supervised: the forward model produces ground truth for each intermediate, so no stage
is trained on another stage's guess unless that is the deliberate experiment.

## Validation rules

- Height targets use the versioned relative-height contract; adding a constant to a tile's heights
  leaves the target unchanged by construction.
- Absolute elevation is composed from WDL, never predicted. Tiles without a prior emit relative-only
  and say so.
- The height loss is masked by `visible_mask_257`. The excluded fraction is recorded per tile.
- Liquid regions are excluded from terrain relief scoring.
- Empty tiles (single unique colour) are excluded from aggregates but retained in the store.
- Train/val split is by `source_group_id` with no group spanning both sides.
- Evaluation uses Kalimdor and Azeroth. PVPZone02 and Kalidar are never validation targets.
- Every reported metric names its evaluation set.

## Model outputs

| Head | Grid | Baseline it must beat | Target |
|------|------|----------------------|--------|
| Residual | 256x256 | Per-tile mean residual | r >= 0.85 |
| Height (relative) | 257x257 | Tile-mean height | r >= 0.85, target 0.92 |
| Albedo | 256x256 | Per-tile mean colour | Tier 1 threshold |
| Layer alpha | 256x256x4 | Dominant-layer-everywhere | Tier 2-3 IoU |
| Texture identity | 16x16x4 | Most-frequent-texture prior | Per-class recall |
| Confidence | 257x257 | — | Calibration, not accuracy |

Per FR-023, **no head may sit at its baseline** in a run reported as successful, regardless of the
aggregate. Per FR-024, each head must be droppable or freezable without retraining the others.
