# Research: Object-Library Segmentation & Classifier (Spec 119)

**Date**: 2026-07-23 | **Spec**: [spec.md](spec.md)

This resolves the technical unknowns raised in the spec's Technical Context before design.

## D-01: Held-out split strategy — family-isolated, not spatial

**Decision**: The held-out split groups assets by **family** = the asset's parent directory in the
normalized path (e.g., `world\wmo\azeroth\buildings\castle\` is one family; `castle01`/`castle02`/
`castle03` all share it). The split holds out entire families, so near-duplicate numbered variants
never straddle train/held-out (FR-004). This is a **different** split from Spec 116's
spatially-isolated terrain-tile split — the object library has no spatial coordinates, so spatial
isolation does not apply.

**Rationale**: The spec's highest-risk requirement (FR-004) is leakage. Numbered variants
(`_000`/`_001`, `castle01`/`castle02`) are visually near-identical; a row-random split would put
`castle01` in train and `castle02` in held-out, inflating every metric. Directory-isolation is the
coarsest grouping that provably keeps variants together without requiring a manual
near-duplicate detector (which would itself be a research task). It mirrors the spirit of Spec 116's
8-neighbour isolation (isolate by the natural locality unit) applied to the library's locality unit
(the directory).

**Alternatives considered**:
- *Row-random split*: rejected — leaks near-duplicates, invalidates metrics (the exact failure FR-004
  exists to prevent).
- *Perceptual-hash clustering split*: rejected for v1 — correct but adds a perceptual-hash dependency
  and a clustering step before any model exists; defer to a v2 refinement if directory-isolation
  proves too coarse (e.g., one giant directory holding many unrelated assets).
- *Asset-path-prefix split* (first N path components): equivalent to directory-isolation for the
  common case; directory-isolation is simpler to explain and verify.

**Leakage check**: after building the split, enumerate all family pairs where ≥2 assets share a
name stem differing only by a numeric suffix (`_\d{3}`, `\d+$`); assert no such pair straddles
train/held-out. Report `verified_violation_count` (must be 0), mirroring Spec 116's
`held_out_split.py` contract.

## D-02: Model architecture — two small from-scratch specialists

**Decision**: Two **independent** models, not one multi-task net (Rule 7: no multi-task training,
each model predicts ONE signal):
- **Classifier** (US1): a small conv encoder + global pool + linear head. Input 128×128×3 (the
  library's `--target-size` default), output = class logits. ~100–300K params at `--base 16`.
- **Segmenter** (US2): a U-Net-lite (strided double-conv encoder 128→64→32→16 + skip decoder back
  to 128), single binary foreground channel. Mirrors `ObjectSegmentNet`'s shape (Spec 118) but at
  128×128 and binary, not 256×256 3-class. ~200–500K params at `--base 16`.

Both are constructable from `base` alone (so inference reconstructs the exact architecture from the
checkpoint's config, mirroring the Spec 117/118 bridge pattern). No pretrained backbone
(FR-003/SC-005).

**Rationale**: Rule 7 forbids multi-task training and shared weights between models. A classifier
and a segmenter predict different signals (class label vs pixel mask), so they must be separate
checkpoints. The U-Net-lite for segmentation is the proven shape from Spec 118; the plain conv
encoder for classification is the minimal shape that can separate object families from a clean
top-down crop. 128×128 matches the library's default `--target-size`, avoiding a resize mismatch.

**Alternatives considered**:
- *One multi-task net with two heads*: rejected — violates Rule 7 (no multi-task, no shared weights).
- *Pretrained backbone (e.g., ResNet-18)*: rejected — FR-003/SC-005 mandate from-scratch; also
  violates the tiny-modular-specialist constraint.
- *Shared encoder, separate heads, separately trained*: rejected — still shares weights at training
  time even if checkpoints differ; Rule 7 is about the training graph, not just the checkpoint file.

## D-03: Label taxonomy — coarse-first, finer-as-heuristic

**Decision**: The classifier's primary label is the coarse `asset_type` from `assets.parquet`
(`wmo`/`mdx`/`m2`). A **finer family label** is derived heuristically from the asset path's
grandparent directory (e.g., `world\wmo\azeroth\buildings\castle\` → family `castle`), and is
explicitly labeled as heuristic, not authoritative. The first training cut uses the coarse 3-class
split; the finer taxonomy is an extensible layer validated only if the coarse classifier succeeds
(SC-001 is defined on the coarse split).

**Rationale**: The coarse label is already in `assets.parquet` and is authoritative (it's the file
extension). A finer taxonomy requires path parsing that is heuristic (directory names are not a
controlled vocabulary) — making it the primary label would let a path-parsing bug masquerade as a
model failure. Coarse-first gives an unambiguous learnability verdict; finer families are a
follow-on that the spec explicitly allows as "extensible, explicitly-labeled-as-heuristic."

**Alternatives considered**:
- *Finer taxonomy only*: rejected — heuristic labels, noisy, would confound the learnability verdict.
- *Manual family labeling*: rejected — does not scale to thousands of assets; the heuristic is the
  scalable path, kept as a secondary layer.

## D-04: Blank-capture handling — threshold + explicit empty class

**Decision**: Captures with mask coverage below a documented threshold (default 1.0%, configurable
via `--blank-threshold`) are treated as **empty**. For the classifier, they are assigned an explicit
`empty` class (so the model can learn to flag them rather than confidently misclassify a blank
image). For the segmenter, they are **excluded** from training (an all-background mask teaches the
model to predict all-background, which is the trivial baseline SC-002 guards against). The threshold
and the count of excluded/relabeled captures are recorded in the run record.

**Rationale**: FR-006 mandates blank captures not be silently trained on. The two treatments differ
because the two tasks differ: a classifier *should* learn to say "empty" (it's a real class at
inference time — a failed capture is a real input), but a segmenter trained on all-background masks
converges to the all-background trivial baseline, which SC-002 must beat. Excluding blanks from the
segmenter keeps the held-out IoU honest.

**Alternatives considered**:
- *Exclude blanks from both*: rejected for the classifier — "empty" is a useful inference-time class
  (FR-013: runs on loose images; a blank loose image should be flagged, not force-classified).
- *Include blanks in both with no relabel*: rejected — violates FR-006 and corrupts the segmenter.

## D-05: Reuse of existing v50 machinery

**Decision**: Spec 119 reuses (imports, does not reimplement):
- `harvester.v50.lr_schedule.make_onecycle_scheduler` + `warmup_complete`/`warmup_epochs_for`
  (the Spec 117 scheduling fix — stale counter suppressed until warmup completes).
- `harvester.v50.model_stage_contract` — widened with a new stage
  `"object_library_classifier"` and `"object_library_segmenter"` (two stages, one per model, since
  they are independent checkpoints per Rule 7). The `v50-model-stage-run-v1` schema is reused
  verbatim; only `STAGES` grows.
- `harvester.object_library` — the existing schema module (`ObjectLibraryEntry`,
  `detect_asset_type`, `normalize_asset_path`, `library_id_from_asset_path`) is the label/path
  source of truth; no duplication.
- The `build_object_library.py` zarr layout (`capture_rgb`/`capture_mask`/`assets.parquet`/
  `index.parquet`/`metadata.json`) is the input contract; Spec 119 reads it, never mutates it
  (FR-011).

**Rationale**: Rule 3/6 — do not reimplement what exists. The lr-schedule fix, the stage-run
schema, and the object-library schema are all validated; reusing them keeps Spec 119 small and
keeps the run records comparable to Specs 114–118.

## D-06: Quality lens (US3) — embedding + disagreement report

**Decision**: The quality lens runs the **frozen classifier** over the full library and emits:
1. A per-asset **embedding** = the classifier's penultimate-layer output (a fixed-length float
   vector), written to a parquet (`embeddings.parquet`: `library_id`, `embedding` list).
2. A **disagreement report** (`quality_report.json`): captures where predicted class ≠ labeled
   class, sorted by classifier confidence in the wrong class; near-duplicate clusters via
   cosine-similarity nearest-neighbor on the embeddings (top-K pairs above a threshold).
3. A **low-coverage flag** list (captures below the blank threshold), separate from mislabels.

Embeddings are deterministic from a frozen checkpoint (FR-009): `torch.no_grad()`, `model.eval()`,
fixed seed for any stochastic op (there are none at eval time).

**Rationale**: US3's payoff is turning the library into a queryable, audited corpus. The embedding
is the natural byproduct of the classifier (no new model needed). The disagreement report is the
direct realization of FR-008. Near-duplicate clustering via embedding cosine similarity is the
cheap, dependency-free path (no separate perceptual-hash or clustering library).

**Alternatives considered**:
- *Separate embedding model (e.g., a contrastive net)*: rejected — scope creep; the classifier's
  penultimate layer is a valid embedding for the quality-lens use case (US3 does not require
  retrieval-grade embeddings; FR-012 defers retrieval).
- *External clustering (DBSCAN/HDBSCAN)*: rejected for v1 — cosine-threshold top-K is simpler and
  sufficient for "are near-duplicates grouped"; a clustering library can be added in a v2 if the
  threshold approach proves too coarse.
