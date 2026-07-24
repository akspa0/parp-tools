# Data Model: Object-Library Segmentation & Classifier (Spec 119)

**Date**: 2026-07-23 | **Spec**: [spec.md](spec.md) | **Research**: [research.md](research.md)

This defines the entities Spec 119 reads, derives, and writes. It does not redefine the
object-library zarr itself (that contract is owned by `build_object_library.py` /
`harvester.object_library`); Spec 119 is a read-only consumer of the library plus a writer of
derived artifacts.

## Input entities (read-only, from the object-library zarr)

### ObjectLibraryCaptureImage
- **Source**: `<store>/capture_rgb` array, shape `(N, H, W, 3)`, dtype `uint8`.
- **Meaning**: the textured top-down render of one asset. `H=W=target_size` (default 128).
- **Index**: row `i` corresponds to `assets.parquet` row `i` and `index.parquet` row `i`.

### ObjectLibraryCaptureMask
- **Source**: `<store>/capture_mask` array, shape `(N, H, W)`, dtype `uint8`.
- **Meaning**: the occlusion-free silhouette (255 = object, 0 = background) for the same asset/row.
- **Coverage**: `mean(mask > 0)` per row; rows below `--blank-threshold` (default 0.01) are
  "empty" (D-04).

### ObjectLibraryAssetRow (from `assets.parquet`)
- `library_id`: str — deterministic SHA1-prefix id (`objlib_<14hex>`).
- `original_asset_path`: str — the client-internal path as captured.
- `normalized_asset_path`: str — lowercase, slash-normalized.
- `asset_type`: str — `wmo` | `mdx` | `m2` (the classifier's primary label).
- `visibility_class`: str — `roof_visible` | `likely_visible` | ... (informational; not a label).
- `capture_status`: str — `captured` | `failed` | ... (only `captured` rows are trainable).

## Derived entities (computed by Spec 119)

### AssetFamily (the split key — D-01)
- **Derivation**: `family = parent_directory(normalized_asset_path)`. E.g.
  `world/wmo/azeroth/buildings/castle/castle01.wmo` → family
  `world/wmo/azeroth/buildings/castle`.
- **Role**: the held-out split isolates by family — entire families move to held-out together, so
  numbered variants (`castle01`/`castle02`) never straddle the split (FR-004).
- **Validation**: a leakage check enumerates numeric-suffix variant pairs and asserts none straddle
  train/held-out; `verified_violation_count` must be 0.

### CoarseClassLabel (the classifier target — D-03)
- **Values**: `wmo` | `mdx` | `m2` | `empty` (the `empty` class is assigned to blank captures, D-04).
- **Source**: `asset_type` from `assets.parquet`, plus the blank-threshold relabel.
- **Index**: stable int mapping `{"empty":0, "m2":1, "mdx":2, "wmo":3}` (empty=0 so it is the
  default/background class; order is alphabetical-minus-empty for readability).

### FineFamilyLabel (heuristic, secondary — D-03)
- **Derivation**: `grandparent_directory(normalized_asset_path)` as a string token (e.g. `castle`).
- **Status**: heuristic, not authoritative; used only in an optional `--fine-labels` training cut,
  never as the primary success metric (SC-001 is on the coarse split).

### SegmentationTarget (the segmenter target — D-04)
- **Derivation**: `capture_mask > 0` → binary `(H,W)` int64 {0 background, 1 foreground}.
- **Exclusion**: rows with coverage < `--blank-threshold` are excluded from segmenter training
  (D-04: an all-background target teaches the trivial baseline SC-002 must beat).

### HeldOutSplit
- **Schema**: a JSON file (`<split-name>.json`) listing `train_families` and `held_out_families`
  (lists of family strings), plus `train_row_count`/`held_out_row_count` and
  `verified_violation_count` (must be 0).
- **Determinism**: derived deterministically from the library's `assets.parquet` + a seed; the same
  library + seed yields the same split (FR-009 spirit).

## Output entities (written by Spec 119, never mutating the source zarr — FR-011)

### ClassifierCheckpoint
- **Path**: `<output-root>/<run-name>/classifier.pt`.
- **Contents**: `state_dict`, `architecture` (config hash + `base`), `class_index` (the
  `CoarseClassLabel` int→str map), `config` (lr, epochs, blank-threshold, split name/hash).
- **Reconstruction**: constructable from `base` alone (D-02), so inference rebuilds the exact net.

### SegmenterCheckpoint
- **Path**: `<output-root>/<run-name>/segmenter.pt`.
- **Contents**: `state_dict`, `architecture` (config hash + `base`), `config`. Binary single-channel.

### ModelStageRun (per model — D-05)
- **Schema**: `v50-model-stage-run-v1` (reused verbatim from `model_stage_contract`).
- **Stage**: `"object_library_classifier"` (classifier) or `"object_library_segmenter"` (segmenter) —
  two new entries in `STAGES`.
- **Fields of note**: `baselines` carries the majority-class baseline (classifier) or the
  all-foreground/all-background trivial IoU (segmenter); `metrics` carries held-out
  accuracy/IoU + per-class precision/recall; `promotion_verdict` starts `pending`.

### AssetEmbedding (US3)
- **Path**: `<output-root>/<run-name>/embeddings.parquet`.
- **Schema**: `library_id` (str), `embedding` (list[float], fixed length = classifier penultimate
  dim), `predicted_class` (str), `labeled_class` (str), `agreement` (bool).
- **Determinism**: `torch.no_grad()`, `model.eval()`, no stochastic ops → byte-identical for a
  frozen checkpoint (FR-009).

### QualityReport (US3 — FR-008)
- **Path**: `<output-root>/<run-name>/quality_report.json`.
- **Schema**:
  - `mislabels`: list of `{library_id, asset_path, labeled_class, predicted_class, confidence}`,
    sorted by confidence in the wrong class descending.
  - `near_duplicates`: list of `{library_id_a, library_id_b, cosine_similarity}` pairs above a
    threshold (default 0.95), top-K (default 200).
  - `low_coverage`: list of `{library_id, asset_path, coverage}` below the blank threshold.
  - `summary`: counts (`total`, `mislabel_count`, `near_duplicate_pair_count`, `low_coverage_count`).
