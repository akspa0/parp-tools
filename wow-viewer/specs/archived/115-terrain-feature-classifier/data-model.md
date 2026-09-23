# Phase 1 Data Model: Terrain Feature Classification

## Canonical feature families

Fixed ordinal encoding. The ordinal **is** the channel index in every predicted feature map, so it is
part of the contract and may not be reordered without a taxonomy revision bump.

| Ordinal | Family | Meaning |
|---|---|---|
| 0 | `unknown` | No rule matched, or excluded/unusable source data. Never a silent default for real terrain. |
| 1 | `terrain` | Natural ground: grass, dirt, sand, rock, snow. The default "this is real geometry" class. |
| 2 | `road` | Authored flat travel surfaces: road, path, trail, cobble, brick, pavement. **The confound this feature exists to isolate.** |
| 3 | `water` | Water/liquid-adjacent surfaces: water, river, lake, ocean, swamp beds. |
| 4 | `structure` | Building/object-adjacent surfaces: floors, foundations, walls, tiled bases. |

`TAXONOMY_REVISION` is a string constant hashed into label-set provenance. Changing any rule, the
family list, or their order requires bumping it.

## Entities

### TextureNameDump (JSON side-car, produced by C# `dump-texture-names`)

| Field | Type | Notes |
|---|---|---|
| `build` | string | Client build fingerprint, e.g. `0.5.3.3368`. |
| `map` | string | Map name; joins to `index.parquet.map`. |
| `tiles[]` | array | One record per occupied tile. |
| `tiles[].tile_x` / `tile_y` | int 0..63 | Joins to `index.parquet.tile_x` / `tile_y`. |
| `tiles[].texture_names[]` | string[] | **Ordered** MTEX table. Position = the local index stored in `mcly_texture_ids`. Order is load-bearing. |

### TerrainFeatureRuleSet (in-code, versioned)

Ordered list of `(substring, family_ordinal)`. First case-insensitive match on the texture path
wins; no match ⇒ `unknown`. Order is load-bearing: specific tokens precede generic ones.

### TerrainFeatureLabelMap (derived, per curriculum row)

| Field | Type | Notes |
|---|---|---|
| `labels` | `(256, 256) uint8` | Family ordinal per pixel, aligned pixel-for-pixel to `minimap_rgb`. |
| `valid` | `(256, 256) bool` | False where the source data could not produce an honest label. Loss and metrics ignore these pixels; they are never relabelled to a real class. |

Derivation per pixel `(y, x)` in chunk `(cy, cx) = (y // 16, x // 16)`:

1. Dominant layer `L` = highest `k` in 3..1 with `alpha_256[y, x, k] > DOMINANT_ALPHA_THRESHOLD`
   (0.5) **and** `mcly_layer_mask[cy, cx, k]` set; else 0.
2. Local texture index `t = mcly_texture_ids[cy, cx, L]`.
3. Name = `texture_names[t]` from the dump for this tile; out-of-range ⇒ `valid = False`.
4. Family = first matching rule against that name, else `unknown`.

Rows with no dump entry, or no populated `mcly_texture_ids`, are excluded wholesale from training
supervision (spec FR-004) rather than emitted as all-`unknown`.

### TerrainFeatureLabelSet (persisted derived store)

Zarr store beside the curriculum: `labels (N,256,256) uint8`, `valid (N,256,256) bool`, and attrs
carrying `taxonomy_revision`, `dominant_alpha_threshold`, `rule_set_sha256`, source curriculum
identity, and reconciled counts (`rows_labelled`, `rows_excluded`, per-family pixel coverage).
Row order matches the curriculum exactly, so existing split assignments apply unchanged.

### PredictedFeatureMap (generated — the only form downstream models may consume)

`(K, 256, 256) float32` class probabilities, `K = 5`. Produced only by a classifier checkpoint.
Never derived from ground-truth texture IDs at inference.

### Deconfounded geometry input

`(3 + K, 256, 256)` = `minimap_rgb` ⊕ `PredictedFeatureMap`. `in_channels` is folded into the
architecture config hash, so a deconfounded checkpoint can never be silently loaded as the RGB-only
baseline or vice-versa.

## Invariants

1. Ground-truth texture IDs/names appear only in label derivation and classifier training. They
   never enter any model's inference input path.
2. Label ordinals are contract-stable; the taxonomy revision must change if they do.
3. `valid = False` pixels are excluded from loss and metrics, never coerced to a class.
4. A row missing usable ground truth is excluded, never zero-filled.
5. The derived label store's row order is identical to the curriculum's, preserving the frozen split.
