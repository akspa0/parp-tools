# Feature Specification: V16 Inference Output Contract & Patch Pipeline

**Feature Branch**: `001-v16-inference-output-contract-and-patch-pipeline`

**Created**: 2026-05-18

**Status**: Draft

**Input**: User description: "Lock input→output dataset pairing contract for inference; define patch-ready outputs for LK ADT patching; define alphaWDT creation handoff boundaries; include provenance/evidence artifacts for train/val/infer."

## User Scenarios & Testing

### User Story 1 — Direct `.pred.zarr` Consumption (Priority: P1)

A terrain-AI researcher runs `infer_v16.py` and gets a `.pred.zarr` prediction store. Today, to actually patch ADTs, they must first run the per-tile summary staging step (which writes thousands of `inference_summary.json` + `.npy` files), then point `terrain-patch-adt` at that staging directory. This is slow, fragile, and wastes disk.

The researcher should be able to point `terrain-patch-adt` directly at the `.pred.zarr` store and have it read predictions from Zarr arrays, skipping the per-tile summary staging entirely.

**Why this priority**: This is the most common friction point in the current inference→patch loop. Eliminating the staging step makes the pipeline dramatically faster and simpler.

**Independent Test**: Run `infer_v16.py` to produce a `.pred.zarr`, then run `terrain-patch-adt --pred-zarr <path> --input-adt-dir <dir> --output-dir <dir>` and verify patched ADTs are produced without any intermediate staging files.

**Acceptance Scenarios**:

1. **Given** a completed `.pred.zarr` store, **When** `terrain-patch-adt` is invoked with `--pred-zarr`, **Then** it reads predictions directly from Zarr arrays and patches ADTs without writing staging files.
2. **Given** a `.pred.zarr` store with `index.parquet`, **When** `terrain-patch-adt` reads it, **Then** it can resolve tile coordinates from the index and match them to input ADT files by `(map, tile_x, tile_y)`.
3. **Given** a `.pred.zarr` store missing required arrays, **When** `terrain-patch-adt` reads it, **Then** it reports clear error messages about which arrays are missing.

---

### User Story 2 — Liquid Chunk Patching (Priority: P1)

The current `terrain-patch-adt` command patches height (MCVT) and normals (MCNR) but ignores liquid data entirely. The V16 model predicts `liquid_pred_mask_256` and the dataset carries `liquid_height`. Patched ADTs should carry predicted liquid data in their `MH2O` (LK 3.3.5+) or `MCLQ` (Alpha/early) chunks.

**Why this priority**: Liquid is a first-class terrain signal. Without it, patched ADTs have no water, which breaks rendering and gameplay validation.

**Independent Test**: Run inference on a tile known to have water (e.g., Azeroth coastal tile), patch the ADT, then inspect the output ADT's liquid chunks and verify they contain the predicted liquid mask and height.

**Acceptance Scenarios**:

1. **Given** a `.pred.zarr` with `liquid_pred_mask_256` and `liquid_height` arrays, **When** `terrain-patch-adt` patches an LK ADT, **Then** the output `_obj0.adt` contains an `MH2O` chunk with the predicted liquid data.
2. **Given** a tile with no predicted liquid (mask all zeros), **When** `terrain-patch-adt` patches it, **Then** no `MH2O` chunk is written (or an empty one is written, matching the source tile's behavior).
3. **Given** the source ADT already has `MH2O` data, **When** `--replace-liquid` is set, **Then** the predicted liquid replaces the original; **When** `--replace-liquid` is not set, **Then** the original liquid is preserved.

---

### User Story 3 — One-Shot Pipeline Command (Priority: P2)

A researcher should be able to run a single command that: loads a trained checkpoint, runs inference on a build, patches the ADTs, and optionally converts to Alpha WDT. Today this requires 3-4 manual sequential commands.

**Why this priority**: Reduces the cognitive overhead and script-writing burden for the most common validation workflow.

**Independent Test**: Run `WowViewer.Tool.Converter infer-and-patch --build 3_3_5_12340 --checkpoint <path> --client-root <dir> --map Azeroth --output-dir <dir>` and verify it produces both patched LK ADTs and (optionally) an Alpha WDT.

**Acceptance Scenarios**:

1. **Given** a trained checkpoint and a staged client, **When** `infer-and-patch` is invoked, **Then** it runs inference, patches ADTs, and writes a patch report.
2. **Given** `--alpha-output` is set, **When** `infer-and-patch` completes patching, **Then** it also runs `convert-lk-to-alpha` on the patched output.
3. **Given** any failure during the pipeline, **When** `infer-and-patch` encounters it, **Then** it reports which stage failed and exits with a non-zero code.

---

### User Story 4 — Provenance & Evidence Artifacts (Priority: P2)

Every inference run should produce a `_inference_run.json` sidecar recording: model version, checkpoint hash, input index hash, seed, device, timestamps, and array shapes. This already exists in `infer_v16.py`. The gap is that `terrain-patch-adt` does not record what it did: which tiles were patched, what channels were replaced, old/new content hashes.

**Why this priority**: Without patch provenance, there is no way to audit what a patched ADT actually contains or reproduce a specific result.

**Independent Test**: Run `terrain-patch-adt` and verify the output directory contains a `patch_report.json` with per-tile entries showing replaced channels, source hashes, and prediction hashes.

**Acceptance Scenarios**:

1. **Given** a completed patch run, **When** the output directory is inspected, **Then** a `patch_report.json` exists with per-tile entries.
2. **Each patch report entry** contains: `tile_name`, `map`, `tile_x`, `tile_y`, `replaced_channels` (list), `source_root_hash`, `source_obj_hash`, `pred_height_hash`, `pred_liquid_hash`.
3. **Given** `--report-path <path>` is set, **When** patching completes, **Then** the report is written to the specified path instead of the default location.

---

### User Story 5 — Zarr Index Coordination (Priority: P3)

The input `.zarr` store's `index.parquet` and the output `.pred.zarr`'s `index.parquet` must be row-aligned with identical `tile_id` values. This is already enforced by `infer_v16.py`. But there is no validation command that proves two stores are properly paired. A `validate-inference-pair` command would check: same row count, same `tile_id` values, same order, matching `tile_x`/`tile_y`.

**Why this priority**: Prevents silent misalignment bugs that produce wrong patches.

**Independent Test**: Run `validate-inference-pair --input <build>.zarr --output <build>.pred.zarr` on a valid pair and verify it passes; run on a mismatched pair and verify it fails with a clear error.

**Acceptance Scenarios**:

1. **Given** a valid input/output pair, **When** `validate-inference-pair` is run, **Then** it reports "PASS" with row count and hash match.
2. **Given** a mismatched pair (different row count), **When** run, **Then** it reports "FAIL" with the specific mismatch.
3. **Given** a mismatched pair (same row count, different tile_ids), **When** run, **Then** it reports "FAIL" with the first mismatched row.

---

### Edge Cases

- What happens when `terrain-patch-adt` encounters a tile in `.pred.zarr` that has no matching ADT in the input directory? (Should warn and skip, not abort.)
- What happens when a predicted height is NaN or Inf? (Should abort that tile and report in the patch report.)
- What happens when `--replace-liquid` is set but the predicted liquid mask is all zeros? (Should write an empty/no-liquid state, not preserve the original.)
- What happens when the `.pred.zarr` store was built from a different build than the input ADTs? (Should abort with a build mismatch error.)

## Requirements

### Functional Requirements

- **FR-001**: `terrain-patch-adt` MUST accept `--pred-zarr <path>` to read predictions directly from a Zarr store.
- **FR-002**: `terrain-patch-adt` MUST resolve tile coordinates from the `.pred.zarr`'s `index.parquet` when `--pred-zarr` is used.
- **FR-003**: `terrain-patch-adt` MUST patch `MH2O` (LK) or `MCLQ` (Alpha) liquid chunks from predicted `liquid_pred_mask_256` and `liquid_height`.
- **FR-004**: `terrain-patch-adt` MUST support `--replace-liquid` / `--no-replace-liquid` flags (default: replace when liquid prediction exists).
- **FR-005**: `terrain-patch-adt` MUST write a `patch_report.json` to the output directory (or `--report-path`).
- **FR-006**: A new `infer-and-patch` command MUST chain inference + patching + optional alpha conversion in one invocation.
- **FR-007**: A new `validate-inference-pair` command MUST verify input/output store alignment.
- **FR-008**: All commands MUST produce clear error messages for missing arrays, build mismatches, and NaN/Inf predictions.
- **FR-009**: The `_inference_run.json` sidecar MUST be copied or referenced by the patch report for traceability.

### Key Entities

- **Prediction Store** (`.pred.zarr`): Zarr v3 store with predicted arrays + `index.parquet` + `_inference_run.json`.
- **Patch Report** (`patch_report.json`): Per-tile record of what was patched, source hashes, and prediction hashes.
- **Inference Pair**: Input `.zarr` + output `.pred.zarr` with aligned `index.parquet` rows.

## Success Criteria

### Measurable Outcomes

- **SC-001**: `terrain-patch-adt --pred-zarr` produces identical output to the current staging-based flow for the same checkpoint + build.
- **SC-002**: Liquid chunks are present in patched ADTs for tiles with predicted water.
- **SC-003**: The one-shot `infer-and-patch` command completes the full pipeline in a single invocation.
- **SC-004**: `validate-inference-pair` catches row-count mismatches, tile_id mismatches, and order mismatches.
- **SC-005**: Patch report is generated for every `terrain-patch-adt` run.

## Assumptions

- The V16 model architecture and training pipeline are stable (V16 spec is implemented).
- `infer_v16.py` already produces correct `.pred.zarr` stores with `_inference_run.json`.
- The `terrain-patch-adt` command already handles MCVT/MCNR patching correctly.
- LK 3.3.5 ADT format is the primary patch target; Alpha liquid patching through MCLQ is secondary.
- The `convert-lk-to-alpha` command is stable and can consume patched LK output.
