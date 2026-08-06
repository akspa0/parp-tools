# Feature Specification: V60 Unified Dataset and Shadow-First Terrain Model

**Feature Branch**: `134-v60-unified-dataset-model`

**Created**: 2026-08-05

**Status**: Draft

**Input**: User direction: consolidate all scattered v50.zarr stores and archaeology outputs into a single unified v60 Zarr datastore, implement the terrain_shadow_256→height_257 model (spec 133 US3), apply the surviving-height-levels curation fix, and release v0.5.2 of the viewer.

## Context

### The current state

After the full v50.1 dataset pipeline (specs 109-112), the curation refactor (spec 122), the archaeology pipeline (spec 127), the brush-signature classification (spec 132), and the unbaked minimap decomposition (spec 133), the data is scattered across multiple output formats and locations:

- **Per-build Zarr stores** under `output/datasets/v50/v50.1/` — one per build+map (e.g., `0_5_3_3368-Azeroth.zarr`)
- **Archaeology Zarr stores** under `output/archaeology/<build_id>/store/` — per-build with per-map sub-stores
- **Archaeology NPZ shards** under `output/archaeology/<build_id>/npz/` — raw shards from harvest
- **Archaeology classification output** under `output/archaeology/<build_id>/classify/` — three-tier classification
- **Residual-extractor curricula** under `output/datasets/azeroth-residual-extractor-curriculum/`
- **Spec-specific Zarr stores** under `output/datasets/spec116/`, `spec117/`, etc.
- **Model checkpoints** under `output/v50/v50.1/direct_geometry/`, `output/runs/`, etc.

The dataset was labelled v50.1 but the actual signal set has grown significantly since: `terrain_shadow_256` (spec 133), the three-tier `signal_class` (spec 132), and the curation improvements (spec 122) were all added after the v50.1 manifest was frozen. A version bump to v60 is warranted.

### The model problem

No direct minimap→height model beats the tile-mean baseline. The root cause identified in spec 133 is that `minimap_rgb = albedo × lighting` blends texture, shadow, and normals into one RGB signal. The fix is:
1. Emit `terrain_shadow_256` as a separate signal (spec 133 US1 — **done**, C# side committed)
2. Build a curriculum that includes the decomposed signal (spec 133 US2)
3. Train a model that takes `terrain_shadow_256 → height_257` (spec 133 US3)

### The curation fix

The workstream-terrain-ml.md identifies a concrete curation improvement: `surviving_height_levels` should gate curation in both directions. ~127 tiles currently classified as usable hold ≤64 distinct heights (some with only 2 values across a 516-unit range); 26 compressed-rich tiles are excluded from curation today when their target is already correct. This is CPU-only work that changes what the model sees.

### The release

The viewer has accumulated significant improvements since v0.5.2: PM4 scene graph (spec 131), three-tier classification (spec 132), unbaked minimap decomposition (spec 133). The tag v0.5.2 exists but hasn't been released as a GitHub Release. The repo needs a clean release, branch merge to main, and a new dev branch.

## Signal catalog for v60

The v60 consolidated store carries every signal from the v50.1 frozen catalog plus the new additions:

| Signal | Shape | Source | Since |
|--------|-------|--------|-------|
| height_257 | 257x257 float32 | MCVT harvest | v50.1 |
| normal_xyz | 257x257x3 float32 | MCNR harvest | v50.1 |
| minimap_rgb | 256x256x3 uint8 | synthesis | v50.1 |
| minimap_rgb_authored | 256x256x3 uint8 | MPQ BLP | v50.1 |
| mcal_alpha_pack | 256x256x4 float32 | MCAL harvest | v50.1 |
| mcly_texture_ids | 16x16x4 int32 | MCLY harvest | v50.1 |
| mcly_layer_mask | 16x16x4 bool | MCLY harvest | v50.1 |
| mcsh_shadow_mask | 256x256 float32 | MCSH harvest | v50.1 |
| mcnk_flags_16 | 16x16 int32 | MCNK flags | v50.1 |
| liquid_mask | 256x256 float32 | MH2O/MCLQ | v50.1 |
| liquid_height | 256x256 float32 | MH2O/MCLQ | v50.1 |
| ... | ... | ... | ... |
| **terrain_shadow_256** | **256x256 float32** | **ComposeShadowArray** | **v60 NEW** |
| **signal_class** | **string** | **classify.py** | **v60 NEW** |
| **surviving_height_levels** | **int32** | **tile_inventory.py** | **v60 NEW** |

## User Scenarios & Testing

### User Story 1 - Unified v60 dataset (Priority: P1)

A dataset operator can build a single v60 Zarr store that consolidates every per-build/per-map store into a unified format with a single index, manifest, and signal catalog — all new signals included.

**Why this priority**: Every downstream consumer (archaeology, training, inference) currently reads from scattered stores with different schemas. A single store eliminates this friction.

**Acceptance Scenarios**:

1. **Given** the old v50.1 stores, **When** the v60 builder runs, **Then** every tile from every store is present in the unified store with all signals.
2. **Given** a v60 store, **When** `terrain_shadow_256` is queried, **Then** every tile has a non-null shadow array.
3. **Given** a v60 store, **When** `signal_class` is queried, **Then** every tile has a valid three-tier classification.
4. **Given** the v60 builder run twice with the same inputs, **When** the outputs are compared, **Then** they are bit-identical (deterministic).

---

### User Story 2 - Curation improvement: surviving_height_levels gating (Priority: P1)

A dataset operator can rebuild the training curriculum with the surviving_height_levels curation fix applied — excluding ≤64-level tiles that teach wrong relationships, admitting compressed-rich tiles that were incorrectly excluded.

**Why this priority**: The curation fix is CPU-only and changes what the model sees. Running it before any GPU spend avoids wasting training time on the wrong data.

**Acceptance Scenarios**:

1. **Given** the old curriculum, **When** the curation fix is applied, **Then** tiles with ≤64 surviving levels are excluded from the training set.
2. **Given** the old curriculum, **When** the curation fix is applied, **Then** tiles with `information_class=rich_terrain` that were previously excluded are now admitted.
3. **Given** the curation fix, **When** the curriculum is rebuilt, **Then** the train/val split is deterministic.

---

### User Story 3 - Shadow→height model (Priority: P1)

A researcher can train a model that takes `terrain_shadow_256` (single-channel, 256x256) as input and predicts `height_257` (257x257) as output, learning the physical relationship between terrain shadow and terrain height without the confounding texture signal.

**Why this priority**: This is the milestone — a model that beats the tile-mean baseline by learning the shadow→height relationship directly.

**Independent Test**: Train on v60 tiles with intact shadow signals, evaluate on held-out tiles, measure val_mae against the tile-mean baseline.

**Acceptance Scenarios**:

1. **Given** a trained shadow→height model, **When** evaluated on held-out tiles, **Then** it beats the tile-mean baseline by at least 5% relative.
2. **Given** a trained shadow→height model, **When** fed a tile with re-textured albedo, **Then** the height prediction is unchanged (the model learned shadow, not texture).
3. **Given** a trained shadow→height model, **When** fed a tile with no shadow (flat lighting), **Then** the model reports lower confidence.

---

### User Story 4 - Synthesized control tiles in the dataset (Priority: P2)

A dataset operator can generate synthetic control tiles — minimaps and signals baked from
known ground truth via the compositor — and include them in the v60 store as a control group
for model training.

**Why this priority**: Now that the compositor can bake minimaps perfectly, we can generate
control tiles with exact known ground truth (height, normals, shadow, texture). These are
invaluable for model training: they provide a clean, fully-supervised control group to measure
against real-client tiles, and let us tweak lighting/reliability/precision later.

**Acceptance Scenarios**:

1. **Given** a set of synthetic terrain heightmaps, **When** the compositor bakes them, **Then**
   each produces a minimap, terrain_shadow_256, normal_xyz, and height_257 with exact known
   ground truth.
2. **Given** the synthetic control tiles, **When** added to the v60 store, **Then** they are
   tagged with a `source_kind=synthetic` index column so they can be selected or excluded from
   training.
3. **Given** a synthetic control tile, **When** the shadow→height model is evaluated on it,
   **Then** the prediction can be compared against the exact known height (perfect ground truth).

---

### User Story 5 - Deduplicated, versioned unified store (Priority: P2)

A dataset operator can build a single v60 Zarr store that packs all data for all builds,
deduplicating signals that are identical across builds for the same map, and storing
versioned data per map.

**Why this priority**: Many signals (height_257, normal_xyz, minimap_rgb, terrain_shadow_256)
are byte-identical across builds for the same map when the terrain wasn't changed between
builds. Storing a full copy per build wastes enormous space. Deduplicating identical signal
arrays and storing one canonical copy per map, with per-build version pointers, gives a
gigantic space savings while keeping every build's data queryable.

**Acceptance Scenarios**:

1. **Given** two builds with identical terrain for the same map, **When** the dedup pass runs,
   **Then** the shared signal arrays are stored once, not twice.
2. **Given** a deduplicated store, **When** a specific build's tile is queried, **Then** the
   correct versioned data is returned (the dedup is transparent to consumers).
3. **Given** a map that changed between builds, **When** deduplicated, **Then** only the changed
   signals are stored per build; unchanged ones point to the canonical copy.
4. **Given** the deduplicated store, **When** its size is compared to the naive per-build store,
   **Then** it is smaller (the savings are measurable).

---

### User Story 6 - v0.5.2 release and branch management (Priority: P2)

A maintainer can tag and publish v0.5.2, merge the current feature branches into main, and start a new dev branch for continued work.

**Why this priority**: The current branches have accumulated ~3 commits of unmerged work. A clean release resets the branch topology.

**Acceptance Scenarios**:

1. **Given** the current branch state, **When** branches are merged, **Then** main contains all committed work from 131, 132, and 133.
2. **Given** a v0.5.2 tag, **When** the CI pipeline runs, **Then** it publishes release binaries.
3. **Given** the release, **When** the README and docs are updated, **Then** they reflect the current state.

## Requirements

### Functional Requirements

- **FR-001**: The v60 builder MUST produce a single Zarr store with a unified index across all builds and maps.
- **FR-002**: The v60 store MUST carry `terrain_shadow_256`, `signal_class`, and `surviving_height_levels` as first-class signals.
- **FR-003**: The curation fix MUST gate on `surviving_height_levels` — exclude ≤64 levels, admit compressed-rich tiles.
- **FR-004**: The shadow→height model MUST accept a single-channel 256x256 input and produce a 257x257 height field.
- **FR-005**: The shadow→height model MUST be trainable on a single GPU in under 24 hours.
- **FR-006**: The v0.5.2 release MUST publish the viewer binary via GitHub Actions.
- **FR-007**: The v60 builder MUST support adding synthetic control tiles (baked from known ground
  truth via the compositor) tagged with a `source_kind=synthetic` index column.
- **FR-008**: Synthetic control tiles MUST carry exact known ground truth for height, normals,
  shadow, and texture so they can serve as a fully-supervised control group.
- **FR-009**: The v60 store MUST deduplicate signal arrays that are byte-identical across builds
  for the same map, storing one canonical copy and per-build version pointers.
- **FR-010**: Deduplication MUST be transparent to consumers — querying a specific build's tile
  returns the correct versioned data.

### Non-Functional Requirements

- **NFR-001**: The v60 builder must complete in under 30 minutes (consolidating existing stores, no re-harvesting).
- **NFR-002**: The model must be reproducible — same seed, same data, same checkpoint.
- **NFR-003**: The README and userguide must be updated before the release tag.

## Success Criteria

1. **Unified store**: A single v60 Zarr store exists with all signals from all builds, deterministic.
2. **Curation fix**: Training curriculum with surviving_height_levels gating applied.
3. **Model beats baseline**: Shadow→height model achieves val_mae < 0.142 (5% below 0.1493 baseline).
4. **v0.5.2 released**: GitHub Release published, branches merged, README/userguide updated.

## Key Entities

### V60Store
- `store_id`: string — "v60-unified"
- `builds`: list of build IDs included
- `signals`: list of signal names with shapes and dtypes
- `row_count`: int
- `index`: parquet table with per-row metadata

### ShadowHeightModel
- `input_channels`: 1 (terrain_shadow_256)
- `output`: 257x257 relative height
- `architecture`: direct_cnn_v112 (1-channel) or mit_b0_regression (1-channel)
- `target_contract`: v112.1 (relative height, min-max normalized)

## Assumptions

1. The existing v50.1 stores have valid data — no re-harvesting is needed for the consolidation.
2. The `terrain_shadow_256` signal requires re-harvesting with the new C# code; the v60 builder can produce it by running the harvest tool or by calling the compositor from Python.
3. The shadow→height model reuses the existing `direct_cnn_v112` architecture with `in_channels=1`.
4. The user runs all training and harvest commands (Rule 0).
5. The v0.5.2 release is a tag push on the main branch after merge.