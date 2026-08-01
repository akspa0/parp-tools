# Feature Specification: Weak Signal Amplifier — Cross-Signal Terrain Data Restoration

**Feature Branch**: `061-weak-signal-terrain-restoration`

**Created**: 2026-06-13

**Status**: Draft

**Input**: User description: "get the weak signal amplifier working so it also can reconstruct mismatched terrain data like tiles with proper normals but improper heightmap data. It exists all over the data corpus, we train on it in our models — of which, we need to explore improving too."

## User Scenarios & Testing

### User Story 1 — Detect Height-Normal Mismatch Tiles (Priority: P1)

As a dataset operator, I want to scan the V18 Zarr corpus and identify every tile where the normal data encodes significant terrain variation but the height data is flat or within a suspiciously narrow band, so I can quantify the scope of poisoned supervision data before training.

**Why this priority**: Cannot fix what isn't measured. Every mismatched tile silently degrades height model training. Quantifying the scope is the prerequisite for any repair or model improvement.

**Independent Test**: Run a new `detect_height_normal_mismatch.py` script against the focused V18 stores. It writes a `mismatch_report.parquet` listing every tile where:
- `normal_relief_mean > 0.02` (normals show non-trivial terrain variation)
- `height_range < 3.0` (height data is suspiciously flat)
- The tile is in the kept curation manifest

The report includes per-tile mismatch metrics and a summary count. The script succeeds if it produces a valid parquet file with at least one mismatch tile (we know they exist in the corpus).

**Acceptance Scenarios**:

1. **Given** the focused V18 Zarr stores (`0_5_3_3368.zarr`, `3_3_5_12340.zarr`) and the `v18_focus_terrain_v1` curation manifest, **When** `detect_height_normal_mismatch.py --dataset-dir wow-viewer/output/datasets/v18 --curation-manifest v18_focus_terrain_v1 --output v18_mismatch_report.parquet` runs, **Then** a parquet file is written with columns `[build, tile_id, tile_x, tile_y, map, height_range, height_std, normal_relief_mean, normal_cov, normal_edge_frac, mismatch_severity, mismatch_reason]`.
2. **Given** a tile where normals show a steep hillside (`normal_relief_mean = 0.12`) but heights are all within 0.5m (`height_range = 0.48`), **When** the detector evaluates it, **Then** it is flagged with `mismatch_reason = "height_flat_vs_normal_varied"` and `mismatch_severity = "high"`.
3. **Given** a tile where both normals and heights are flat (ocean tile), **When** the detector evaluates it, **Then** it is NOT flagged — it is correctly blank, not mismatched.

---

### User Story 2 — Reconstruct Heights from Normals for Mismatched Tiles (Priority: P1)

As a dataset operator, I want to reconstruct plausible height values for mismatched tiles by integrating the normal data into a height field, so the poison is removed from the training corpus.

**Why this priority**: Detection without repair leaves the corpus degraded. Height-from-normals integration produces approximate but geometrically consistent heights. Combined with the existing height data as a baseline offset, this yields a usable corrected height field.

**Independent Test**: Run `reconstruct_heights_from_normals.py` on mismatch tiles. For each flagged tile:
- Compute surface gradients from `normal_xyz`: `dz/dx = -nx/nz`, `dz/dy = -ny/nz`
- Integrate gradients to height field via Frankot-Chellappa (Fourier domain)
- Anchor the reconstructed field to the original height data's mean Z
- Write the corrected `height_257` back into a `height_corrected_257` array in a sidecar repair store
- Visually compare original vs corrected height in a before/after preview

The script succeeds if:
1. `python -m py_compile` passes
2. A sidecar Zarr store `v18_mismatch_repair.zarr` is written with `height_corrected_257` matching the index of the original stores
3. A before/after preview PNG exists for at least one repaired tile
4. The reconstructed height field shows visibly more terrain variation than the flat original

**Acceptance Scenarios**:

1. **Given** a mismatched tile with `normal_xyz` encoding a ridge and `height_257` nearly flat at Z=0, **When** the reconstructor runs, **Then** `height_corrected_257` shows a ridge shape matching the normal-derived slopes, with heights anchored around Z=0.
2. **Given** a tile where `normal_mask` only covers 40% of the grid, **When** the reconstructor runs, **Then** only the masked region is integrated; unmasked regions retain their original heights.
3. **Given** a tile where all normals point straight up (`nz ≈ 1.0`, `nx ≈ 0`, `ny ≈ 0`), **When** the reconstructor runs, **Then** heights are unchanged — there is no slope information to recover.

---

### User Story 3 — Patch V18 Zarr Stores with Corrected Heights (Priority: P2)

As a dataset operator, I want to apply the reconstructed heights directly into the V18 Zarr stores so that a refined curation manifest can be rebuilt from corrected data.

**Why this priority**: The goal is training on clean data. Patching the stores completes the data-side of the repair.

**Independent Test**: Run the V18 Zarr repair subcommand `build_v18_dataset.py repair-heights --mismatch-report v18_mismatch_report.parquet --sidecar-repair v18_mismatch_repair.zarr --build 0_5_3_3368`. For each mismatched tile in the report, the store's `height_257[tile_id]` is replaced with the corrected height from the sidecar. The original height is backed up to `height_uncorrected_257`. The `index.parquet` gains a `height_corrected` boolean column. Run `python -m py_compile` and a smoke test on 3 tiles.

**Acceptance Scenarios**:

1. **Given** a V18 Zarr store with 3 known mismatched tiles and their corrected heights, **When** `repair-heights` runs, **Then** `height_257` for those 3 tiles now contains the corrected values and `height_uncorrected_257` contains the originals.
2. **Given** a previously repaired tile where heights were already corrected, **When** `repair-heights` runs again with the same sidecar, **Then** the operation is idempotent — `height_uncorrected_257` is not overwritten a second time.

---

### User Story 4 — Explore Model Improvements Using Normal-Height Correlation (Priority: P3)

As a model developer, I want to explore whether the V18 height model can be improved by adding an auxiliary normal-consistency loss or by training on corrected data, so the model learns to use minimap signals more effectively for height prediction.

**Why this priority**: Data repair comes first. Model architecture changes are only valuable after we have clean supervision. This story is scouting and measurement, not architectural commitment.

**Independent Test**: Two scouting experiments:
1. **Corrected-data retrain**: Run `train_v18_focus.py height` against the tiny manifest but with corrected-heights for mismatched tiles. Compare `val_loss` curves against the uncorrected baseline run.
2. **Normal-consistency auxiliary loss**: Add `_height_normal_consistency_loss()` to the height trainer that penalizes the difference between `_normals_from_height(pred_height)` and the ground-truth `normal_xyz` on terrain-valid regions. Run a tiny-manifest scout and compare val_loss.

Success if either route shows measurable improvement (lower val_loss, or visually better height previews) without regressing on the other.

**Acceptance Scenarios**:

1. **Given** the tiny manifest with 3 corrected-height tiles in the train split, **When** a 5-epoch height scout runs, **Then** `val_loss` over the mismatched tiles is lower than the uncorrected baseline by at least 0.02.
2. **Given** the normal-consistency loss is added at weight 0.1, **When** a 5-epoch height scout runs on uncorrected data, **Then** height predictions on mismatched tiles show visually improved terrain variation compared to the baseline model.

---

### Edge Cases

- **Fully blank tile (ocean/void)**: `normal_cov < 0.05` and `height_range < 0.5`. Should NOT be flagged as mismatched — the blank is correct.
- **WMO-dominated tile**: Object masks cover >70% of the tile. Normal data under objects may be unreliable or flat. Flag with lower severity.
- **Normal data missing**: `has_normals = False`. Cannot reconstruct. Skip entirely.
- **Normal nz near zero**: Where normals are nearly horizontal (`nz < 0.05`), the gradient `-nx/nz` diverges. These pixels should be masked out before integration.
- **Integration artifacts at boundaries**: Fourier integration of truncated gradient fields can produce ringing. Apply a Hann window to the gradient field before integration.
- **Already-corrected tiles**: The mismatch detector must exclude tiles already marked `height_corrected = True` in `index.parquet`.
- **Stitching discontinuity**: Corrected tile heights may not perfectly align with uncorrected neighbors. Accept this for v1 — quilt-level stitching is downstream follow-through (spec 047).

## Requirements

### Functional Requirements

- **FR-001**: System MUST provide a `detect_height_normal_mismatch.py` script that reads V18 Zarr stores and a curation manifest, and writes a mismatch report as parquet with per-tile metrics.
- **FR-002**: The mismatch detector MUST compute `normal_relief_mean` (mean of `sqrt(nx² + ny²)` over normal-masked pixels) as the primary normal-variation metric.
- **FR-003**: The mismatch detector MUST compute `height_range` (max - min of `height_257`) as the primary height-flatness metric.
- **FR-004**: A tile MUST be flagged as mismatched when `normal_relief_mean >= threshold` AND `height_range < threshold` AND `normal_cov >= min_coverage`.
- **FR-005**: Default thresholds MUST be `normal_relief_mean >= 0.02`, `height_range < 3.0`, `normal_cov >= 0.10`. All thresholds MUST be CLI-overridable.
- **FR-006**: System MUST provide a `reconstruct_heights_from_normals.py` script that integrates normal vectors into a height field via Frankot-Chellappa Fourier-domain integration.
- **FR-007**: The reconstructor MUST mask out pixels where `nz < 0.05` (near-horizontal normals) before integration.
- **FR-008**: The reconstructor MUST mask out pixels where `normal_mask == False` before integration.
- **FR-009**: Reconstructed heights MUST be anchored such that their mean Z equals the original height data's mean Z (preserving absolute world-space position).
- **FR-010**: System MUST write corrected heights to a sidecar Zarr store (`height_corrected_257`) with index parity to the source store.
- **FR-011**: The `build_v16_dataset.py` repair path MUST accept `repair-heights` subcommand that patches `height_257` arrays from a sidecar store.
- **FR-012**: The repair operation MUST back up original heights to `height_uncorrected_257` before overwriting.
- **FR-013**: The repair operation MUST be idempotent — re-running with the same report+sidecar produces no additional changes.
- **FR-014**: All new Python modules MUST live in `wow-viewer/data-harvester/src/harvester/` and all new scripts in `wow-viewer/data-harvester/scripts/`.
- **FR-015**: All new code MUST pass `uv run python -m py_compile` and associated pytests.
- **FR-016**: The mismatch detector MUST skip tiles where `has_normals = False` or `normal_cov < 0.05`.
- **FR-017**: The mismatch report MUST include a `mismatch_severity` column with values `low`, `medium`, `high` based on the ratio of `normal_relief_mean` to `height_range`.

### Key Entities

- **MismatchReport**: A parquet file with one row per audited tile. Columns: `build`, `tile_id`, `tile_x`, `tile_y`, `map`, `height_range`, `height_std`, `normal_relief_mean`, `normal_cov`, `normal_edge_frac`, `minimap_gray_std`, `mismatch_severity`, `mismatch_reason`, `object_cov`. Only mismatched tiles are written (non-mismatched tiles are skipped in output).
- **SidecarRepairStore**: A minimal Zarr store containing only `height_corrected_257` and an `index.parquet` matching the source store's tile IDs. Written by the reconstructor, consumed by the repair subcommand.
- **HeightCorrectionRecord**: Per-tile metadata in the sidecar store's index: `tile_id`, `correction_method` (always `frankot_chellappa` for v1), `integration_rms_error`, `normal_nz_masked_frac`, `was_corrected` (bool).

## Success Criteria

- **SC-001**: The mismatch detector identifies at least 10 tiles across the focused V18 corpus where `normal_relief_mean > 0.02` and `height_range < 3.0`.
- **SC-002**: For at least one mismatched tile, the reconstructed height field shows a qualitatively correct terrain shape (ridge/valley/slope matches what the normals encode), validated by visual inspection of before/after preview PNGs.
- **SC-003**: The repair operation is idempotent — a second run on the same store produces zero additional writes.
- **SC-004**: At least 3 pytest tests pass covering: mismatch detection logic, normal-to-height integration on synthetic data, and repair idempotency.
- **SC-005**: `uv run python -m py_compile` passes on all new scripts and library modules.
- **SC-006**: A tiny-manifest height scouting run (21 tiles, 5 epochs) against corrected data shows `val_loss` improvement of at least 0.02 compared to the uncorrected baseline on the same split.

## Assumptions

- The V18 focused Zarr stores (`0_5_3_3368.zarr`, `3_3_5_12340.zarr`) exist at `wow-viewer/output/datasets/v18/` with populated `height_257` and `normal_xyz` arrays.
- The `v18_focus_terrain_v1` curation manifest exists at `wow-viewer/output/datasets/v18/curation/v18_focus_terrain_v1/kept_tiles.parquet`.
- Normal vectors in the Zarr stores are unit-length and correctly oriented (Z-up world space). If normals are malformed for a given tile, the mismatch detector may produce false positives — that is a separate data quality issue.
- Frankot-Chellappa Fourier integration is sufficient for v1. More sophisticated integrators (Poisson, screened Poisson) are deferred to v2 if artifacts are unacceptable.
- Height anchoring to mean Z is sufficient for v1. Per-tile stitching continuity with neighbors is explicitly deferred to downstream quilt-level work (spec 047).
- The `height_uncorrected_257` backup array may increase Zarr store size. This is acceptable for focused two-build scope.
- Model improvement scouting (US4) is measurement-only for this spec. Any architecture change that passes the scout becomes a separate spec.
- All Python work uses `uv` and the existing `wow-viewer/data-harvester/` environment. No new `.venv` or `requirements.txt` outside that root.
