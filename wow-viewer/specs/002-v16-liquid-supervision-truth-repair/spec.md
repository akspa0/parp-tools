# Feature Specification: V16 Liquid Supervision Truth Repair

**Feature Branch**: `002-v16-liquid-supervision-truth-repair`

**Created**: 2026-05-20

**Status**: Draft

**Input**: User description: "Fix the liquid-supervision truth for V16 so the raw harvest, Zarr repair path, and human validation images agree. 0.7.0 and 3.0.1 still look wrong or collapse to unified-only. We need human-verifiable evidence, fail-loud sampling, and a bounded fix plan for a fresh chat."

## Problem Statement

The V16 dataset repair path is partly working but still not trustworthy across all eras:

- `0_5_3_3368` and `0_5_5_3494` now patch into strong `mcnk`-backed liquid masks.
- `3_3_5_12340` and `4_0_0_11927` now patch into explicit `mh2o`-backed liquid masks.
- `0_7_0_3694` still remains `unified`-only.
- `3_0_1_8303` still remains `unified`-only, and raw harvest sampling for `mh2o`/`mcnk_liquid` wrote `0 samples`, which is not acceptable as a silent outcome.

This means the current pipeline still allows one of the worst failure modes:

1. harvest raw data
2. silently collapse rich source signals into `unified`
3. produce plausible-looking final masks
4. waste training time because the provenance is wrong or missing

The fix must start from truth surfaces, not from model training:

- raw `harvest-stream` NPZ contents
- finalized Zarr arrays + `index.parquet`
- labeled human validation images
- fail-loud sample commands that do not pretend success when no qualifying tiles were found

## User Scenarios & Testing

### User Story 1 — Human Validation Must Show Real Liquid Provenance (Priority: P1)

A terrain researcher opens the validation image for a build and needs to know whether a tile's liquid mask came from `mcnk`, `mclq`, `mh2o`, `unified`, or `wl`. If the source is missing or downgraded, the image and JSON must say so plainly.

**Why this priority**: Human inspection is the only fast way to catch orientation, mirroring, and missing-signal failures before wasting long training runs.

**Independent Test**: Run the finalized-store inspection command on `0_7_0_3694`, `3_0_1_8303`, `3_3_5_12340`, and `4_0_0_11927`, then verify each overview tile label includes `liquid_src=<source>` and that the sample JSON reflects the same source flags.

**Acceptance Scenarios**:

1. **Given** a finalized V16 Zarr store, **When** `inspect_v16_dataset.py --write-overview` is run, **Then** the overview PNG labels each tile with `build`, `tile_id`, `map`, `tile_x`, `tile_y`, and `liquid_src`.
2. **Given** a build with only `unified` liquid provenance, **When** the overview is generated, **Then** tiles are labeled `liquid_src=unified` and the summary JSON includes exact per-source counts.
3. **Given** a tile with no liquid signal at all, **When** the overview is generated, **Then** it is labeled `liquid_src=none` rather than inheriting an assumed source.

---

### User Story 2 — Raw Harvest Sampling Must Fail Loud When a Claimed Source Is Missing (Priority: P1)

Today, the raw-harvest sample inspector can finish with `Wrote 0 samples` for a requested category such as `mh2o` or `mcnk_liquid`. That is an unacceptable success signal.

The raw-harvest inspector must explicitly fail or mark the build/category as missing when the requested source does not appear in the streamed NPZ data.

**Why this priority**: The `3_0_1_8303` problem only became obvious because the raw inspector returned zero samples for `mh2o`/`mcnk_liquid`. That should have been a hard signal, not a subtle clue in console output.

**Independent Test**: Run `inspect_v16_harvest_samples.py --build 3_0_1_8303 --kinds mh2o mcnk_liquid ...` and verify it exits non-zero or writes an explicit failure summary when no matching samples are found.

**Acceptance Scenarios**:

1. **Given** a requested kind such as `mh2o`, **When** the inspector finds at least one sample, **Then** it writes the sample NPZs/PNGs and a non-empty samples JSON.
2. **Given** a requested kind such as `mh2o`, **When** the inspector finds zero samples, **Then** it exits non-zero or writes `status=missing` in a machine-readable summary instead of reporting normal success.
3. **Given** mixed requested kinds, **When** one kind succeeds and another fails, **Then** the output clearly separates the successful and missing categories.

---

### User Story 3 — Era-Specific Liquid Sources Must Survive Into Final Stores (Priority: P1)

For builds that genuinely contain richer liquid supervision, the final repaired store must preserve that provenance instead of collapsing everything into `unified`.

**Why this priority**: The training contract depends on knowing whether a liquid mask is `mcnk`-backed, `mclq`-backed, `mh2o`-backed, or only `unified`.

**Independent Test**: Run `patch-liquids` on the repaired stores, then verify `liquid_patch_report.json` and `signal_validation.json` reflect the richer source for known-good builds.

**Acceptance Scenarios**:

1. **Given** `0_5_3_3368` and `0_5_5_3494`, **When** `patch-liquids` completes, **Then** the final store reports dominant `mcnk` liquid provenance.
2. **Given** `3_3_5_12340` and `4_0_0_11927`, **When** `patch-liquids` completes, **Then** the final store reports explicit `mh2o` liquid provenance.
3. **Given** `0_7_0_3694` and `3_0_1_8303`, **When** the repair workflow is rerun after the fix, **Then** they either gain explicit era-appropriate provenance or emit a focused failure that proves the richer source is absent at harvest time.

---

### User Story 4 — One-Tile Trace Mode for Known Bad Liquid Tiles (Priority: P2)

When a build still looks wrong, a researcher needs a single-tile trace command that dumps the raw NPZ arrays, decoded metadata, and derived masks for one exact `(map, tile_x, tile_y)` tile.

**Why this priority**: Broad sampling is useful, but repair work on `0_7.0` / `3_0.1` now needs a deterministic per-tile audit seam.

**Independent Test**: Run a single-tile trace command or trace mode against one known-wet `3_0_1_8303` tile and verify it writes the raw arrays, metadata, and derived-source summary.

**Acceptance Scenarios**:

1. **Given** a targeted tile coordinate, **When** trace mode is run, **Then** it writes raw arrays and decoded metadata to a dedicated output folder.
2. **Given** a tile with no explicit source arrays, **When** trace mode is run, **Then** the summary states which source arrays were absent and which fallback path produced the final mask.
3. **Given** a tile with a source orientation issue, **When** trace mode is run, **Then** the saved visual panels make the mismatch visible without retraining anything.

## Edge Cases

- Sea-level water at `0.0f` must not be treated as "no liquid".
- `0 samples` for a requested raw source must not be treated as success.
- Orientation fixes must not be applied globally if only one era/source needs them.
- A build may legitimately end up `unified`-only, but only after the pipeline proves that richer source arrays are absent.
- Existing Zarr stores must be repairable in place where possible; full re-harvest is a last resort, not the default.

## Requirements

### Functional Requirements

- **FR-001**: `inspect_v16_dataset.py` MUST produce labeled validation images that include explicit liquid provenance per sample tile.
- **FR-002**: `inspect_v16_harvest_samples.py` MUST fail loud or emit explicit `missing` status when requested liquid source categories produce zero samples.
- **FR-003**: The repair workflow MUST preserve explicit liquid provenance (`mcnk`, `mclq`, `mh2o`, `unified`, `wl`) in finalized V16 stores.
- **FR-004**: `signal_validation.json` MUST report per-source liquid counts and warn when a build collapses to `unified` unexpectedly.
- **FR-005**: The workflow MUST support deterministic single-tile tracing for one exact bad tile.
- **FR-006**: The fix path MUST prefer patching existing stores in place over full dataset rebuilds when the missing truth can be recovered from raw streamed data.

### Key Entities

- **Raw Harvest Sample**: A streamed NPZ blob plus its decoded `metadata.json`, raw chunk payloads, and derived source masks.
- **Final Store Validation Overview**: Labeled PNG showing `input/minimap`, `height`, `liquid mask`, and `object mask` for sample tiles.
- **Liquid Provenance**: One of `mcnk`, `mclq`, `mh2o`, `unified`, `wl`, or `none`.
- **Known Bad Tile**: A tile that visually should contain water but currently arrives as `unified`-only or `none`.

## Success Criteria

### Measurable Outcomes

- **SC-001**: `inspect_v16_harvest_samples.py` no longer reports silent success with `0 samples` for requested liquid categories.
- **SC-002**: `3_0_1_8303` raw harvest sampling yields explicit evidence of why `mh2o` / `mcnk_liquid` are missing, not just an empty result.
- **SC-003**: `0_7_0_3694` and `3_0_1_8303` no longer remain ambiguous in final validation; they either gain explicit source provenance or fail with focused diagnostics.
- **SC-004**: Human validation images exist for the holdout builds and clearly label liquid provenance per sample tile.
- **SC-005**: Existing good cases (`0_5_3_3368`, `0_5_5_3494`, `3_3_5_12340`, `4_0_0_11927`) remain good after the truth-repair work.

## Assumptions

- The current `patch-liquids` workflow is a valid in-place repair seam once the correct source arrays are exposed.
- `3_3_5_12340` and `4_0_0_11927` are the reference later-era builds for sane `MH2O` truth.
- `0_7_0_3694` and `3_0_1_8303` are the active holdouts that need a dedicated repair lane.
- This feature is about dataset truth and validation, not terrain model architecture changes or liquid-height refinement training.
