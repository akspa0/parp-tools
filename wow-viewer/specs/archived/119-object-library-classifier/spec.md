# Feature Specification: Object-Library Segmentation & Classifier

**Feature Branch**: `119-object-library-classifier`

**Created**: 2026-07-23

**Status**: Draft

**Input**: User request: "let's move on to the object library segmentation model/classifier work." The new `capture-objects` harvest pipeline (Spec 118's harvest command + `build_object_library.py`) now produces a zarr object library with `capture_rgb` (N,H,W,3) + `capture_mask` (N,H,W) per asset, plus `assets.parquet` (asset_type: m2/mdx/wmo, normalized_asset_path, library_id, visibility_class). We want to train a small from-scratch model on this object library that classifies and segments the captured object images themselves — understanding object data well enough to identify class and potentially the entire object from the top-down capture. This is the v50-era successor to the completed Spec 025 object-roof-mask-library model.

## Overview *(context, not a template section — kept brief)*

The object library (`<run-name>.zarr`) is now complete and WMO-inclusive: every M2/MDX/WMO asset in a client is captured as a top-down textured image paired with a clean occlusion mask, with asset-type and path provenance in `assets.parquet`. That library is a self-contained, labeled image dataset — but nothing yet *learns* from it. This feature trains a small, from-scratch specialist that reads a single captured object image and predicts (a) the object's class and (b) a per-pixel segmentation of the object versus background. It is the object-library analogue of Spec 118 US3's minimap segmenter, but operates on the captured object crops, not minimap tiles — a distinct input surface with a distinct supervision source (the library's own `capture_mask` + `asset_type` labels, not terrain-tile ground truth).

The trained model has two downstream uses: (1) a quality/clustering lens over the library itself (which captures are mislabeled, which assets look near-duplicate, which classes are confusable), and (2) a building block for object-identity retrieval that can later feed the minimap object-aware chain. It is an independently checkpointed specialist whose output feeds downstream stages by output, not shared weights — consistent with the project's tiny-modular-specialist constraint (Rule 7).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Object class classifier from a captured image (Priority: P1)

Train a small from-scratch classifier that takes a single object-library capture image and predicts the object's class (at minimum: wmo vs mdx/m2; finer families where the library's path taxonomy readily provides them). The classifier is supervised by the library's own `asset_type` (and derived family) labels, not by minimap terrain data.

**Why this priority**: Classification is the cheapest, most directly verifiable learning task on the library — the labels are already in `assets.parquet`, no new harvest is needed, and a held-out accuracy number is an unambiguous "did it learn anything" verdict. It is the foundation: a classifier that cannot separate wmo from mdx/m2 on clean captured images signals a data problem before any segmentation work is worthwhile.

**Independent Test**: Split the library into train/held-out by asset family (not random-by-row, to avoid near-duplicate-asset leakage). Train the classifier; report held-out top-1 accuracy and per-class precision/recall. Verify: (a) held-out accuracy is well above the majority-class baseline; (b) confusion is concentrated in genuinely confusable families, not spread uniformly (which would indicate a labeling or feature problem); (c) the classifier runs on a loose captured image with no store present.

**Acceptance Scenarios**:

1. **Given** a held-out WMO capture, **When** the classifier predicts, **Then** it predicts `wmo` (or the finer WMO family) above a defined accuracy threshold, not a random class.
2. **Given** two near-duplicate assets (e.g., `castle01` and `castle02`), **When** both are in the held-out set, **Then** the classifier's predictions are stable and consistent with their labels, not flickering between classes — or, if they are genuinely indistinguishable from the top-down capture alone, that is reported as a measured limitation, not a silent error.
3. **Given** a capture with a blank or near-blank mask (a failed/empty capture that slipped through), **When** classified, **Then** it is flagged as low-confidence or an explicit "empty/unknown" class rather than confidently misclassified.

---

### User Story 2 - Per-pixel object segmentation from a captured image (Priority: P2)

Train a small from-scratch segmentation model that takes a single object-library capture image and predicts, per pixel, object-vs-background, supervised by the library's own `capture_mask`. This is the segmentation analogue of US1: the mask is already the ground truth, so this is a "can the model reproduce the renderer's silhouette from the textured image alone" learnability test.

**Why this priority**: Segmentation is strictly harder than classification and depends on the masks being clean (proven by the capture pipeline's validation). It delivers standalone value: a segmenter that reproduces silhouettes from RGB alone can denoise/impute masks for captures where the mask pass was partial or noisy, and is a prerequisite for any instance-separation or retrieval work. It is prioritized after classification because a classifier that fails makes a segmenter pointless.

**Independent Test**: Train on the library's masks with the same family-isolated held-out split; report held-out per-pixel IoU and boundary F1. Verify: (a) held-out IoU is well above the "predict-all-foreground" and "predict-all-background" trivial baselines; (b) failure cases are concentrated on captures with thin/extreme aspect ratios or very low mask coverage, not uniformly distributed; (c) the segmenter runs on a loose captured image with no store present.

**Acceptance Scenarios**:

1. **Given** a held-out capture with a clear, high-coverage object, **When** the segmenter predicts, **Then** predicted foreground overlaps the ground-truth mask above a defined IoU threshold.
2. **Given** a capture with a very small or thin object (low mask coverage), **When** the segmenter predicts, **Then** it either reproduces the thin silhouette acceptably or is reported as a measured low-IoU failure case — not a confident wrong blob.
3. **Given** a blank capture (all-background mask), **When** the segmenter predicts, **Then** it predicts all-background, not spurious foreground.

---

### User Story 3 - Library quality lens + downstream retrieval hook (Priority: P3)

Use the trained classifier + segmenter as a quality and clustering lens over the library: flag captures whose predicted class disagrees with the labeled class (candidate mislabels), flag near-duplicate assets by embedding proximity, and expose a per-asset embedding that can later drive object-identity retrieval (find the library asset closest to a minimap-detected object crop). The retrieval wiring itself is out of scope for this feature; this story delivers the embedding + the quality report.

**Why this priority**: This is the payoff that makes the model useful beyond a held-out number — it turns the library from a flat image folder into a queryable, audited corpus. It depends on US1/US2 being trained. The retrieval integration is explicitly deferred to avoid scope creep into the minimap chain (Spec 118's territory).

**Independent Test**: Run the quality lens over the full library; verify: (a) flagged mislabels, when manually inspected, are genuinely mislabeled or genuinely confusable (not random noise); (b) near-duplicate clusters group assets a human agrees are visually similar; (c) the per-asset embedding is deterministic and reproducible from a frozen checkpoint.

**Acceptance Scenarios**:

1. **Given** the trained classifier and the full library, **When** the quality lens runs, **Then** it emits a report listing captures whose predicted class disagrees with the labeled class, with confidence, sorted by disagreement severity.
2. **Given** the per-asset embeddings, **When** nearest-neighbor search is run, **Then** near-duplicate assets cluster together (verifiable on a known-similar pair like `castle01`/`castle02`).
3. **Given** a frozen checkpoint, **When** embeddings are recomputed for the same library, **Then** they are byte-identical (deterministic).

---

### Edge Cases

- **Near-duplicate assets** (`castle01`/`castle02`, `_000`/`_001` numbered variants): the held-out split must isolate by asset family/directory, not by row, or near-duplicates leak across train/held-out and inflate every metric. A leakage check is mandatory.
- **Blank or near-blank captures** (failed renders, empty masks): must not corrupt training. They are either filtered with a documented threshold or assigned an explicit "empty" class; silently training on all-blank masks teaches the model to predict all-background.
- **Extreme aspect ratios** (a long bridge or wall WMO): the square top-down capture pads heavily, leaving the object a thin strip. The segmenter may legitimately struggle; this is a measured limitation, not a bug. The quality lens should flag low-coverage captures separately.
- **Class imbalance** (far more mdx/m2 doodads than wmo buildings in a typical client): training must use class weighting or balanced sampling; a raw majority-class baseline must be reported so "accuracy" cannot hide "predicts the majority class always."
- **Finer-than-asset-type families**: deriving a finer family taxonomy (e.g., "human building" vs "undead building") from the asset path is heuristic and may be noisy; the first cut uses the coarse wmo/mdx/m2 split, with finer families as an extensible, explicitly-labeled-as-heuristic layer.
- **Multi-variant assets** (same asset, multiple rotations/scales): the library currently captures one variant per asset (identity pose). If multi-variant captures are added later, the split must still isolate by asset, not by variant, to avoid leakage.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The classifier MUST accept a single object-library capture image (the `capture_rgb` for one asset) and predict the object's class, supervised by the library's own `asset_type`/family labels — never by minimap or terrain data.
- **FR-002**: The segmenter MUST accept a single object-library capture image and predict a per-pixel object-vs-background mask, supervised by the library's own `capture_mask`.
- **FR-003**: Both models MUST be **small and trained from scratch** (no DepthAnything-family or mandatory pretrained-backbone dependency), and each is an independently checkpointed specialist whose output feeds downstream stages by output, not shared weights.
- **FR-004**: The train/held-out split MUST isolate by **asset family/directory**, not by row, so near-duplicate assets (`castle01`/`castle02`, numbered variants) do not leak across the split. A leakage check (verifying no near-duplicate pair straddles train/held-out) MUST be run and reported.
- **FR-005**: Training MUST report a **majority-class baseline** alongside held-out accuracy/IoU, so a model that merely predicts the majority class cannot be reported as successful.
- **FR-006**: Blank or near-blank captures (mask coverage below a documented threshold) MUST be either filtered with a recorded threshold or assigned an explicit "empty" class — never silently trained on as if they were normal object captures.
- **FR-007**: Class imbalance MUST be handled (class weighting or balanced sampling), and per-class precision/recall MUST be reported, not only aggregate accuracy.
- **FR-008**: The quality lens (US3) MUST emit a report of captures whose predicted class disagrees with the labeled class, with confidence and disagreement severity, and a near-duplicate clustering by embedding proximity.
- **FR-009**: Per-asset embeddings emitted by the trained model MUST be **deterministic and reproducible** from a frozen checkpoint (identical library → identical embeddings).
- **FR-010**: All training, evaluation, and quality-lens steps MUST be **user-run**: the tooling prepares and prints exact commands (dry-run-first) and never launches heavy/billed training itself.
- **FR-011**: The feature MUST NOT mutate the source object-library zarr in place; any derived artifacts (checkpoints, embeddings, quality reports) are written to separate, provenance-bound outputs.
- **FR-012**: The retrieval integration (using embeddings to match minimap-detected object crops to library assets) is explicitly **out of scope** for this feature; this feature delivers the embedding + quality lens only.
- **FR-013**: Both models MUST run on a **loose captured image with no store or ground truth present** (inference on a single PNG), not only in batch mode against the zarr.

### Key Entities *(include if feature involves data)*

- **Object-library capture image**: the `capture_rgb` (H,W,3) for one asset — the model's input.
- **Object-library capture mask**: the `capture_mask` (H,W) for one asset — the segmenter's ground truth.
- **Asset class label**: derived from `asset_type` (wmo/mdx/m2) and, where the path taxonomy readily provides it, a finer family — the classifier's ground truth.
- **Asset family / directory**: the grouping key for the leakage-safe held-out split (e.g., assets sharing a parent directory or a name prefix are one family).
- **Per-asset embedding**: a fixed-length vector from the trained model's penultimate layer, used for near-duplicate clustering and (later) retrieval.
- **Quality report**: the US3 artifact listing candidate mislabels, near-duplicate clusters, and low-coverage captures.
- **Held-out split**: the family-isolated split reused to judge both the classifier (US1) and the segmenter (US2) honestly.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The classifier achieves held-out top-1 accuracy materially above the majority-class baseline (target: ≥15 percentage points above majority-class baseline on the coarse wmo/mdx/m2 split), with per-class precision/recall reported.
- **SC-002**: The segmenter achieves held-out per-pixel IoU materially above both the all-foreground and all-background trivial baselines (target: ≥0.20 IoU above the better trivial baseline), with failure cases stratified by mask-coverage bucket.
- **SC-003**: The leakage check confirms no near-duplicate asset pair straddles the train/held-out split (target: zero verified leakage violations), or, if leakage is unavoidable for a tiny family, those families are excluded from the held-out metric and reported separately.
- **SC-004**: The quality lens flags a non-empty set of candidate mislabels and near-duplicate clusters that, when manually inspected, are at least majority-genuine (target: ≥50% of top-flagged mislabels are genuinely mislabeled or genuinely confusable on manual review).
- **SC-005**: Both models are small and from-scratch (target: single-digit-millions of parameters, no pretrained-backbone requirement), consistent with the project's tiny-modular-specialist constraint.
- **SC-006**: The end-to-end path (library → family-isolated split → train classifier + segmenter → quality lens → embeddings) runs entirely from user-issued commands with no assistant-launched heavy jobs, and no source library zarr is mutated.

## Assumptions

- **Input source**: the object-library zarr produced by `build_object_library.py --from-harvest-stream` (the Spec 118 capture pipeline, now WMO-inclusive). No new harvest is required for this feature; it consumes the existing library.
- **Label source**: the coarse class label is `asset_type` from `assets.parquet` (wmo/mdx/m2). A finer family taxonomy is derived heuristically from the asset path's directory/name and is explicitly labeled as heuristic, not authoritative.
- **Mask quality**: the `capture_mask` is renderer-truth (flat white silhouette on black), already validated by the capture pipeline's smoke + contact-sheet proof. This feature does not re-validate mask correctness; it learns from the masks as given.
- **Single-variant captures**: the library currently captures one variant per asset (identity pose, orthographic top-down). Multi-variant (rotated/scaled) captures are a future extension; the split and models are designed to remain valid if multi-variant captures are added, by isolating on asset (not variant).
- **Backend**: training targets CUDA first but the model/training seams keep alternate runners (Vulkan/OpenCL/MLX) practical, per the dataset-builder guardrails. No CUDA-only assumption is hard-wired.
- **Loss vs input use**: this feature's models are trained on and evaluated against the library's own labels/masks. The downstream retrieval integration (matching minimap crops to library assets) is explicitly deferred and is not a success criterion here.
