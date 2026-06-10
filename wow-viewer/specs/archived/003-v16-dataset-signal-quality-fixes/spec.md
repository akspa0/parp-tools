# Feature Specification: V16 Dataset Signal Quality Fixes

**Feature Branch**: `003-v16-dataset-signal-quality-fixes`

**Created**: 2026-05-20

**Status**: Draft

**Input**: User description: "MCNK liquid flags must be included in the dataset. Object masks must separate MDDF/MODF with smart MDDF filtering. WL* data must produce smooth liquid planes. Liquid priority chain must be MCNK > MCLQ > MH2O > WL* > none."

## Problem Statement

Four bugs degrade V16 dataset quality across all client builds:

1. **MCNK liquid flags missing**: `AdtRawChunkBlobCollector` skips the MCNK top-level chunk header (offset 0x00-0x7F), only collecting subchunks from 0x80. The 128-byte header contains liquid type flags at offset 0x00 (bits 2-5: `0x04`=water, `0x08`=ocean, `0x10`=magma, `0x20`=slime). These flags are the authoritative per-chunk liquid classification but never reach the dataset.

2. **MDDF+MODF merged into one mask**: `BuildObjectMasks` paints both doodads (MDDF, circles) and WMOs (MODF, bounding rects) into the same `ObjectMask257` array. The training loop computes `weight = 1.0 - object_mask`. WMOs appear in minimaps (valid loss signal), but MDDFs like trees do not (false loss signal). Every tree pixel incorrectly suppresses terrain loss across the entire corpus.

3. **WL* produces blocky squares**: `ReadWlFiles` averages 16 vertices per block to a single scalar, then stamps a hard 3x3 neighborhood. Each WL block covers ~16 pixels on the 257 grid but only writes to 9. Adjacent blocks don't overlap or blend, producing "subdivided squares" artifacts.

4. **Liquid priority chain is backwards**: The user requires MCNK flags as primary source, but the current C# chain has MH2O first and MCNK is not available (Bug 1). The Python chain has MCNK last but it always returns None for LK builds.

## User Scenarios & Testing

### User Story 1 — MCNK Liquid Flags Present in Dataset (Priority: P1)

A terrain researcher opens a V16 Zarr store and finds `mcnk_flags_16` array with per-chunk liquid type flags for every tile. Alpha builds show MCNK-sourced liquid; LK builds show MH2O-sourced liquid. The signal validation reports non-zero `has_liquid_source_mcnk` for Alpha builds.

**Why this priority**: MCNK flags are the most granular liquid classification available. Without them, the dataset loses per-chunk liquid type information that cannot be recovered from unified masks.

**Independent Test**: Build or patch-liquids a 0.5.x build, then verify `mcnk_flags_16` exists in Zarr and `signal_validation.json` shows `has_liquid_source_mcnk > 0`.

**Acceptance Scenarios**:

1. **Given** a staged 0.5.x Alpha client, **When** `harvest-stream` extracts a tile, **Then** the NPZ blob contains `mcnk_flags_16` with non-zero values for tiles with liquid chunks.
2. **Given** a finalized V16 store for a 0.5.x build, **When** `signal_validation.json` is generated, **Then** `has_liquid_source_mcnk` count is > 0 for tiles with MCNK liquid flags.
3. **Given** a staged 3.3.5 client, **When** `harvest-stream` extracts a tile, **Then** the NPZ blob contains `mcnk_flags_16` (may be all-zero for LK builds where MCNK flags are not the primary liquid source).

---

### User Story 2 — Object Mask Split with MDDF Filtering (Priority: P1)

A terrain researcher opens a validation image and sees that tree doodads no longer create false loss signals, while WMO footprints are no longer inflated into coarse projected bounds rectangles. The dataset stores raw MDDF mask, raw MODF mask, and a filtered combined mask. Training uses only the filtered mask for terrain loss weighting.

**Why this priority**: MDDF false loss signals are the largest source of incorrect terrain supervision. Trees cover significant terrain area but never appear in minimaps. Fixing this alone should measurably improve model terrain understanding.

**Independent Test**: Build a V16 store, inspect the validation overview, verify that tree-heavy tiles no longer show full object mask coverage and that WMO-heavy tiles no longer show oversized projected rectangle masks. Run a smoke train and verify loss computation uses the filtered mask.

**Acceptance Scenarios**:

1. **Given** a V16 store with tiles containing trees, **When** the filtered object mask is inspected, **Then** tree footprints are excluded while rock/WMO footprints are retained.
2. **Given** a tile with MDDF objects matching the clutter exclusion regex against the normalized asset path, **When** `object_filtered_mask_257` is generated, **Then** excluded vegetation and clutter assets have mask value 0.0 at their footprint pixels.
3. **Given** a tile with MDDF objects whose resolved model bounds classify them as tiny clutter or tall clutter, **When** `object_filtered_mask_257` is generated, **Then** those doodads are excluded from the filtered loss mask even if they remain present in the raw MDDF mask.
4. **Given** a tile with a resolvable WMO asset, **When** `modf_mask_257` is generated during archive-backed harvest, **Then** the raster follows transformed WMO mesh triangles instead of a projected placement AABB.
5. **Given** a V16 store, **When** `train_v16.py` runs, **Then** terrain loss is weighted by `1.0 - object_filtered_mask_257` (not the raw merged mask).

---

### User Story 3 — WL* Smooth Liquid Planes (Priority: P2)

A terrain researcher visualizes WL*-sourced liquid data and sees smooth liquid surfaces instead of blocky "subdivided squares". Adjacent WL blocks blend seamlessly.

**Why this priority**: WL* is the last-resort liquid source for builds without MH2O or MCLQ. Blocky output produces incorrect supervision for any tile relying on WL* data.

**Independent Test**: Build a V16 store for a build that uses WL* liquid, inspect liquid height visualization, verify smooth interpolation between blocks.

**Acceptance Scenarios**:

1. **Given** a tile with multiple adjacent WL blocks, **When** the liquid height is visualized, **Then** heights blend smoothly between blocks with no visible grid artifacts.
2. **Given** a single WL block with 4x4 vertices, **When** the liquid mask is generated, **Then** all 16 vertex heights are used (not averaged to a scalar).
3. **Given** adjacent WL blocks, **When** the liquid mask is generated, **Then** there are no hard edges or gaps between blocks.

---

### User Story 4 — Liquid Priority Chain Correct (Priority: P1)

A terrain researcher checks `signal_validation.json` and sees liquid source provenance following the correct priority: MCNK > MCLQ > MH2O > WL*. For 0.5.x builds, MCNK is the dominant source. For 3.3.5 builds, MH2O is the dominant source. No build uses WL* when a higher-priority source is available.

**Why this priority**: Wrong priority chain means the dataset may use inferior liquid data when better data exists.

**Independent Test**: Run `patch-liquids` on multiple builds across eras, verify `liquid_patch_report.json` shows correct source dominance per era.

**Acceptance Scenarios**:

1. **Given** a 0.5.x build with MCNK liquid flags, **When** liquid supervision is derived, **Then** the source tag is `mcnk` (not `unified` or `wl`).
2. **Given** a 3.3.5 build with MH2O data, **When** liquid supervision is derived, **Then** the source tag is `mh2o` (not `unified` or `wl`).
3. **Given** a build with both MCNK flags and MH2O data, **When** liquid supervision is derived, **Then** MCNK takes priority per the defined chain.
4. **Given** a build with no liquid sources, **When** liquid supervision is derived, **Then** `has_liquid_mask` is false and the tile is marked non-usable for liquid supervision.

---

### Edge Cases

- Sea-level water at `0.0f` must not be treated as "no liquid" (MCNK flags determine presence, not height value).
- Alpha MCLQ type `-1` sentinel means "not present" — must not be confused with type `0` (valid water).
- MDDF objects with no bounding box data (fallback to position-based circle) must still be filtered by name regex.
- WL* blocks with zero vertices (corrupted data) must be skipped, not produce NaN heights.
- Builds where MCNK flags are all-zero but MH2O has data must fall through correctly to MH2O.

## Requirements

### Functional Requirements

- **FR-001**: `AdtTensorPackBuilder` MUST extract MCNK header flags (uint32 at offset 0x00 of each chunk payload) and store them as `int[16,16]` in `TerrainTileTensorPack.McnkFlags16`.
- **FR-002**: `NpzTileSerializer` MUST write `mcnk_flags_16` as `<i4` dtype with shape `(16, 16)`.
- **FR-003**: `AdtTensorPackBuilder.BuildObjectMasks` MUST produce separate `MddfMask257`, `ModfMask257`, and `ObjectFilteredMask257` arrays.
- **FR-004**: `ObjectFilteredMask257` MUST exclude MDDF objects matching a clutter exclusion regex against the normalized asset path, including vegetation and small-prop families such as trees, shrubs, grass, stones, pebbles, logs, and stumps.
- **FR-005**: `ObjectFilteredMask257` MUST exclude MDDF objects whose resolved model bounds classify them as tiny planar clutter or tall clutter; the filter must use real model bounding boxes when the asset can be opened from the active asset source and only fall back to coarse placement heuristics when bounds cannot be resolved.
- **FR-006**: `ObjectFilteredMask257` MUST include all MODF (WMO) placements unconditionally.
- **FR-006a**: `AdtTensorPackBuilder.BuildObjectMasks` MUST prefer geometry-derived MODF footprints when a WMO render document can be opened from the active asset source, and MAY fall back to projected placement bounds only when geometry cannot be resolved.
- **FR-007**: `NpzTileSerializer` MUST write `mddf_mask_257`, `modf_mask_257`, and `object_filtered_mask_257` as `<f4` dtype with shape `(257, 257)`.
- **FR-008**: `ReadWlFiles` MUST map each WL block's 4x4 vertex grid to a ~16x16 patch on the 257 grid using per-vertex heights with bilinear interpolation between adjacent blocks.
- **FR-009**: `build_v16_dataset.py` `_derive_liquid_supervision` MUST use priority chain: MCNK flags > MCLQ > MH2O > WL* > none.
- **FR-010**: `build_v16_dataset.py` MUST map `mcnk_flags_16` from NPZ to Zarr, and derive liquid presence from `(flags & 0x3C) != 0`.
- **FR-011**: `v16_dataset.py` MUST load `object_filtered_mask_257` and expose it as the `weight` tensor for terrain loss.
- **FR-012**: `train_v16.py` MUST compute terrain loss weight as `1.0 - object_filtered_mask` (not the raw merged mask).
- **FR-013**: `validate_v16_training_ready.py` MUST validate shapes and dtypes of all new arrays.
- **FR-014**: `signal_validation.json` MUST report `has_liquid_source_mcnk` counts for builds where MCNK flags are available.

### Key Entities

- **MCNK Header Flags**: uint32 at offset 0x00 of each MCNK chunk payload. Bits 2-5 indicate liquid type: `0x04`=water, `0x08`=ocean/deep, `0x10`=magma, `0x20`=slime.
- **ObjectFilteredMask257**: 257x257 float32 mask where 1.0 = object that appears in minimap and should suppress terrain loss. Built from raw MDDF after clutter filtering based on normalized asset-path tokens plus resolved model bounds, and from MODF (all).
- **WL Block**: 4x4 vertex grid with world position, spanning ~33m x ~33m on the terrain.
- **Liquid Priority Chain**: MCNK > MCLQ > MH2O > WL* > none. First source with data wins.

## Success Criteria

### Measurable Outcomes

- **SC-001**: `mcnk_flags_16` exists in all V16 stores. Alpha builds have > 0 tiles with non-zero MCNK flags.
- **SC-002**: `object_filtered_mask_257` excludes tree-class MDDF objects. Validation images show no tree footprints in the filtered mask.
- **SC-003**: WL*-sourced liquid masks show smooth interpolation with no blocky grid artifacts in validation images.
- **SC-004**: `signal_validation.json` for 0.5.x builds shows `has_liquid_source_mcnk > 0`. For 3.3.5 builds shows `has_liquid_source_mh2o > 0`. No build shows `has_liquid_source_wl > 0` when higher-priority sources exist.
- **SC-005**: Smoke train run completes without shape/dtype errors using the new filtered mask.
- **SC-006**: Existing passing tests remain green.

## Assumptions

- MCNK header flags are reliable for liquid classification across all WoW client eras.
- Normalized asset-path tokens plus resolved model bounds are sufficient to remove the current dominant MDDF false-loss cases, even if later tuning is still needed for edge assets.
- WL* files exist primarily for pre-LK builds; LK+ builds use MH2O.
- Existing Zarr stores can be patched in place via `patch-liquids` and a new `patch-object-masks` command.

## Implementation Status (2026-05-20)

### Completed

- [x] C# `TerrainTileTensorPack`: added `McnkFlags16`, `MddfMask257`, `ModfMask257`, `ObjectFilteredMask257` properties
- [x] C# `AdtTensorPackBuilder.ReadMcnkFlags`: new method extracts MCNK header flags from chunk payloads
- [x] C# `AdtTensorPackBuilder.BuildObjectMasks`: returns 6-tuple with separate MDDF/MODF/filtered masks; filtered mask uses regex exclusion + height gate
- [x] C# `AdtTensorPackBuilder.ReadWlFiles`: rewritten to use per-vertex heights with weighted blending (was averaged 3x3 stamp)
- [x] C# `NpzTileSerializer`: serializes `mcnk_flags_16`, `mddf_mask_257`, `modf_mask_257`, `object_filtered_mask_257`
- [x] C# `TerrainTileTensorPack.ToTileLoadResult`: uses real `McnkFlags16` instead of hardcoded `0x3C`
- [x] Python `build_v16_dataset.py`: new array mappings in `OUTPUT_ARRAY_NAMES`, `SHAPES`, `DTYPES`, `CHUNK_SIZES`, `ALL_ARRAY_KEYS`; `_derive_mcnk_liquid_flags` reads `mcnk_flags_16` directly; `_derive_object_supervision` returns 12-tuple with MDDF/MODF/filtered masks
- [x] Python `v16_dataset.py`: uses `object_filtered_mask` for training weight when available, falls back to merged mask for legacy shards
- [x] Python `inspect_v16_dataset.py`: fixed tile_x/tile_y truthiness bug (`0 or -1` → proper `is not None` check)
- [x] Python `inspect_v16_harvest_samples.py`: fixed same truthiness bug
- [x] Python `train_v16.py`: validation alpha QA now renders painted alpha instead of raw channel `0`
- [x] Python `train_v16.py`: `train-max-tiles` now defines a persistent run-level train pool and `train-epoch-tiles` can rotate fresh per-epoch train subsets from that pool
- [x] Python `train_v16.py`: CUDA-oriented loader defaults are less conservative (`--num-workers=-1` auto mode, persistent workers default on, `prefetch-factor=4`)
- [x] Build: 0 errors, all Python files syntax-valid

### Not Yet Validated

- [ ] Real-data harvest stream on staged `3_3_5_12340` with new C# binary
- [ ] Verify `mcnk_flags_16` appears in NPZ blobs for Alpha builds
- [ ] Verify `object_filtered_mask` excludes tree-class MDDF objects
- [ ] Verify WL* liquid heights are smooth (not blocky)
- [ ] Verify `patch-liquids` picks up `mcnk_flags_16` from fresh harvest
- [x] Smoke train run with filtered weight mask
- [x] Smoke proof that epoch-rotating train subsets change per epoch when `train-epoch-tiles < train-max-tiles`
- [ ] Production-oriented epoch-rotation run outcome (`v16_full_corpus_epoch_rotation`) still pending final training results
