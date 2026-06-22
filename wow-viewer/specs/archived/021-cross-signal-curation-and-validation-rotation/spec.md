# Feature Specification: Cross-Signal Curation and Validation Rotation

**Feature Branch**: `021-cross-signal-curation-and-validation-rotation`

**Created**: 2026-05-24

**Status**: Draft

**Input**: Curated tiles pass individual signal-quality checks but can still have cross-signal mismatches (e.g., minimap shows ocean but heightmap has terrain, or a developer-island tile's height/normal data paired with an ocean minimap from a different tile). Validation tiles are fixed per run, so a single bad tile pollutes every epoch's preview. The training pool also never rotates, limiting the total concepts the model sees even when "less is more" per-epoch sampling is the intent.

## Problem Statement

The V16 curation manifest (`normal_terrain_full_corpus_v16_1_1`) checks individual signal health — minimap variance, normal edge coverage, alpha coverage, etc. — and rejects clearly bad tiles (empty minimap, empty normals, whiteplate flats). But it never checks whether signals *agree with each other* within the same tile.

This allows three classes of bad tiles through:

1. **Cross-signal mismatch**: minimap is an ocean tile (low texture variance, blue-green), but the heightmap shows terrain relief (high std, cliffs, hills). The minimap and heightmap came from different tiles — a data harvesting or indexing bug.
2. **Developer island bleed**: height/normal data from a developer test island is paired with a minimap from a different map area. The signals are individually valid but describe completely different terrain.
3. **Partial payload corruption**: one signal array has valid data, another has stale default values from a failed harvest.

These tiles are invisible in single-signal curation but glaringly obvious in training: the model sees conflicting signals and learns nothing useful from that tile. Worse, if such a tile lands in the validation set, it stares back at the researcher for hundreds of epochs making every validation image look wrong.

Separately, the validation tile pool is fixed per run — the same tiles every epoch. This means:
- A single bad tile in validation poisons all `best_epoch` comparisons
- The model can memorize validation tiles
- There is no statistical validation coverage across the corpus

## User Scenarios & Testing

### User Story 1 — Cross-Signal Mismatch Tiles Are Rejected (Priority: P1)

A curation profile adds a cross-signal check: if `corr(minimap_gray, height_257)` is very low (<0.1) when both signals have reasonable variance (>0.01), the tile is flagged as data-mismatched and rejected. The developer-island/ocean-minimap tile no longer passes curation.

**Why this priority**: These tiles actively harm training by providing contradictory supervision.

**Independent Test**: Run curation on the current corpus with the new profile and verify the known bad tile (developer island minimap + ocean heightmap) is rejected.

**Acceptance Scenarios**:

1. **Given** the current V16 corpus, **When** curation runs with cross-signal checks, **Then** at least one previously-kept tile is now rejected with reason `signal_cross_mismatch`.
2. **Given** a rejected cross-mismatch tile, **When** inspecting its minimap/height/normal arrays, **Then** the minimap visually disagrees with the height/normal structure.
3. **Given** the cross-signal check, **When** a correctly-matched tile is evaluated, **Then** it is not rejected (low false-positive rate).

---

### User Story 2 — Validation Pool Rotates Across Epochs (Priority: P1)

Each epoch samples a random subset of the validation pool (e.g., 80% fixed anchor tiles + 20% rotating fresh tiles). A single bad tile only affects 1-in-5 epoch previews instead of all of them. The `best_epoch` comparison uses the full fixed-anchor subset for consistency.

**Why this priority**: Validation rotation prevents a single bad tile from polluting every epoch while keeping the `best_epoch` metric stable.

**Independent Test**: Run a 10-epoch training session and verify that the validation preview tiles change between epochs, while the `best_val` loss uses a consistent anchor set.

**Acceptance Scenarios**:

1. **Given** a training run with validation rotation, **When** comparing epoch 1 and epoch 2 preview images, **Then** at least one tile differs (rotation happened).
2. **Given** validation rotation, **When** `best_val` is computed, **Then** it uses only the fixed anchor subset so epoch-to-epoch comparisons are valid.
3. **Given** a bad tile in the full pool, **When** it appears in a rotating preview, **Then** it does not affect the `best_val` loss (which uses anchors only).

---

### User Story 3 — Training Pool Rotates via Bucket Sampling (Priority: P2)

The per-epoch training pool is not hard-limited to N fixed tiles. Instead, the bucket sampling profile (`v16_1_1_normal`) draws a subset of tiles from the full curated pool each epoch, weighted by difficulty. Over many epochs, the model sees all kept tiles.

**Why this priority**: The "less is more" bottleneck is about per-epoch constraint, not permanent exclusion of tiles the model never sees.

**Independent Test**: Run a 100-epoch training session and verify the cumulative set of unique training tiles grows beyond the per-epoch limit.

**Acceptance Scenarios**:

1. **Given** a training run with `--train-max-tiles 400 --train-epoch-tiles 128`, **When** accumulating unique tiles across 10 epochs, **Then** the count exceeds 128 (new tiles appear each epoch).
2. **Given** the full curation pool is 12K tiles, **When** training runs for 1000 epochs, **Then** the model has seen at least 1000 unique tiles (linear growth with epochs).

---

### Edge Cases

- What if a build has very few curated tiles? Cross-signal checks must gracefully handle small per-build pools.
- What if `corr(minimap, height)` is low but legitimate (e.g., flat terrain with highly textured minimap)? The threshold must account for signal variance, not just raw correlation.
- What if the rotating validation set introduces a tile that happens to be much easier/harder than the anchors? The `best_val` comparison should be anchor-only.

## Requirements

### Functional Requirements

- **FR-001**: The curation script MUST add a cross-signal check that rejects tiles where `corr(minimap_gray, height_257)` is below a threshold when both signals have non-trivial variance.
- **FR-002**: The cross-signal check MUST also compare minimap edge structure with normal edge structure: tiles with high normal edges but zero minimap edges should be rejected.
- **FR-003**: A new reject reason `signal_cross_mismatch` MUST be added to the curation profile.
- **FR-004**: The validation loop MUST split the validation pool into an anchor subset (~80%) and a rotating subset (~20%).
- **FR-005**: The `best_val` loss MUST use only the anchor subset for epoch-to-epoch comparison.
- **FR-006**: The rotating validation subset MUST be resampled each epoch.
- **FR-007**: The training bucket sampler MUST draw from the full curated pool each epoch, not a pre-fixed subset.
- **FR-008**: The `DeterministicEpochSampler` MUST support resampling without resetting epoch order for tiles already seen this run.

### Key Entities

- **Cross-Signal Correlation Check**: Computes Pearson correlation between minimap grayscale and height_257, plus edge-overlap between minimap edges and normal edges.
- **Validation Anchor Subset**: The fixed portion of the validation pool used for consistent `best_val` tracking.
- **Validation Rotating Subset**: The variable portion resampled each epoch for broader preview coverage.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Cross-signal curation rejects at least one previously-kept tile from the current corpus.
- **SC-002**: Validation preview tiles differ across epochs in a 10-epoch run.
- **SC-003**: `best_val` loss is reproducible (±5%) when re-running the same epoch with the same anchor subset.
- **SC-004**: Cumulative unique training tiles grows monotonically across epochs.

## Relationship to Other Specs

- **Depends on**: `007-v16-1-1-curated-normal-acceleration` (curation manifest pipeline)
- **Extends**: training loop in `train_v16_1_common.py`
- **Enables**: V16.1.4 combined model training without mismatched-tile pollution
