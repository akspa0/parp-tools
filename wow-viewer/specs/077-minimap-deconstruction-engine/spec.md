# Feature Specification: Minimap Deconstruction Engine

**Feature Branch**: `077-minimap-deconstruction-engine`

**Created**: 2026-06-28

**Status**: Draft

**Input**: User description: "Use the existing harvester/object-mask/placement pipeline to build a per-object capture library, deconstruct baked minimap imagery into object and terrain layers, train tiny single-purpose models, and restore development-map terrain without needing ADT inputs at inference time."

## Problem Statement

The current minimap-to-terrain lanes have repeatedly failed when asked to learn directly from the fully baked minimap image. The minimap is not a pure terrain signal. It is a composite of terrain color, object roofs, object shadows, liquid, tinting, and client-era rendering artifacts.

The next route must stop treating the minimap as a monolithic truth image. Instead, it must deconstruct the image into the smallest useful parts:

1. object identity and coverage,
2. object-suppressed terrain-focused minimap priors,
3. height reconstruction from that prior,
4. optional later normal refinement,
5. optional later object restoration.

The current repo already contains the C# harvest contract for placement names, pose data, filtered/precise masks, and capture tooling. The missing piece is not a new parallel pipeline. The missing piece is a reusable per-object capture library that loads one asset at a time, records its precise mask and image, and stores that library for later teacher generation and minimap-only inference.

## User Scenarios & Testing

### User Story 1 - Build a Per-Object Capture Library (Priority: P1)

As a dataset operator, I want to load each used object asset one at a time and capture its top-down image, precise mask, and metadata into a reusable datastore, so later stages can reason about object identity without rediscovering assets from whole-world scenes.

**Why this priority**: This is the prerequisite for every later stage. Without a reusable object library, there is no stable teacher signal for object suppression, no asset-path provenance for restoration, and no consistent ADT-free fallback.

**Independent Test**: Run a bounded one-object-at-a-time capture job on a proof asset list and verify it writes a Zarr-backed object library with image arrays, mask arrays, and metadata rows tied to the original asset paths.

**Acceptance Scenarios**:

1. **Given** a bounded list of WMO and M2 assets already referenced by harvested placement tables, **When** the per-object capture job runs, **Then** it writes one or more reusable capture entries per asset into a dedicated object-library store.
2. **Given** an object capture entry, **When** the operator inspects it, **Then** the entry includes the original asset path, normalized lookup path, asset type, capture pose metadata, image data, and mask data.
3. **Given** repeated placements of the same asset across multiple tiles or builds, **When** the library is refreshed, **Then** the system preserves one stable canonical asset identity and records capture variants instead of duplicating the asset blindly.

---

### User Story 2 - Generate Teacher Deconstruction Priors from ADT-Backed Tiles (Priority: P1)

As a dataset operator, I want to use existing placement metadata and filtered precise masks to generate object-suppressed minimap priors for ADT-backed tiles, so the terrain model trains on terrain-focused imagery instead of baked object clutter.

**Why this priority**: The terrain model must be trained against a cleaner signal before any minimap-only inference stage can be trusted.

**Independent Test**: Run a bounded prior-generation job on a proof build/map and verify it emits raw minimap, teacher object mask, teacher object confidence, and processed minimap prior artifacts for tiles with and without objects.

**Acceptance Scenarios**:

1. **Given** a tile with harvested placement data and visible roofs, **When** teacher prior generation runs, **Then** it emits a processed minimap prior where object-heavy regions are suppressed or filled according to the documented teacher policy.
2. **Given** a tile with little or no object coverage, **When** teacher prior generation runs, **Then** the processed minimap prior remains close to the raw minimap and does not invent suppression noise.
3. **Given** the filtered precise mask excludes clutter that is not meaningfully represented in the minimap, **When** teacher prior generation runs, **Then** the prior preserves those terrain pixels instead of over-masking them.

---

### User Story 3 - Train a Height-Only Terrain Model on Deconstructed Priors (Priority: P1)

As a model developer, I want a small height-only model that consumes the processed minimap prior and predicts only `height_257`, so we can quickly validate whether deconstruction fixes the terrain convergence problem before adding any other terrain signals.

**Why this priority**: Height is the core terrain state. A height-only proof is cheaper, clearer, and easier to debug than a combined terrain-state model.

**Independent Test**: Run a bounded smoke training pass that consumes the new prior dataset and produces previews and metrics for `height_257` without any normals or liquid heads.

**Acceptance Scenarios**:

1. **Given** a processed-minimap-prior dataset, **When** the height-only trainer runs, **Then** it consumes only the documented minimal input channels and predicts only `height_257`.
2. **Given** a training run with object-heavy tiles, **When** loss is computed, **Then** the teacher deconstruction prior and filtered precise gate reduce object pollution without changing raw height ground truth.
3. **Given** a preview batch from the height-only model, **When** the operator reviews it, **Then** the model output can be compared directly against raw height truth and the processed prior that produced it.

---

### User Story 4 - Explain Objects Without ADT at Inference Time (Priority: P2)

As an inference operator, I want to explain object coverage and likely asset identity from a raw minimap tile without needing ADT-derived placement data, so development-map and PM4-only tiles can still be deconstructed before terrain inference.

**Why this priority**: This is the deployment case for development map and any other map where ADT supervision is incomplete or absent.

**Independent Test**: Run a bounded minimap-only inference job on a development-map proof set and verify it emits predicted object masks, asset candidates, and processed minimap priors without reading ADT placement arrays.

**Acceptance Scenarios**:

1. **Given** a raw minimap tile with no ADT placement metadata, **When** the object explanation stage runs, **Then** it predicts an object coverage mask from the minimap and object library alone.
2. **Given** a predicted object instance crop, **When** library matching runs, **Then** the system emits one or more asset candidates tied back to stored asset-path metadata.
3. **Given** the development map has PM4 and minimap coverage but incomplete ADT coverage, **When** the deconstruction engine runs, **Then** it still produces a processed minimap prior suitable for height inference.

---

### User Story 5 - Add Normals Only After Height Proof (Priority: P3)

As a model developer, I want normals handled as a separate follow-on lane after the height-only proof is validated, so we do not reintroduce a giant multi-task terrain model.

**Why this priority**: Normals are useful, but they are downstream. Height-first validation is the simpler proof owner and the safer execution order.

**Independent Test**: Produce analytic normals from predicted height as the baseline, then optionally run a bounded normal-refinement experiment as an isolated follow-on model.

**Acceptance Scenarios**:

1. **Given** a validated height-only terrain model, **When** normals are needed, **Then** the first baseline comes from deterministic height-to-normal derivation rather than a joint terrain model.
2. **Given** a later normal-refinement model is introduced, **When** it is trained, **Then** it is trained as a separate lane with its own dataset contract, trainer, checkpoint, and validation.

---

### Edge Cases

- Some placed assets do not visibly contribute to the minimap. The object library must record visibility or review state so these assets are not treated as mandatory mask targets.
- Some assets appear only partially on the tile edge. Teacher targets and inference review artifacts must preserve partial coverage instead of rejecting the sample.
- Some tiles contain overlapping roofs or stacked objects. Teacher labels and inference outputs must tolerate overlap without requiring perfect depth ordering in phase 1.
- Some development-map tiles have no ADT teacher but do have PM4 and minimap data. The runtime deconstruction path must not hard-fail on missing ADT.
- Some objects are filtered out of `object_filtered_mask` because they are clutter, trees, or otherwise not useful for terrain suppression. The teacher path must preserve that filtering policy.
- Some M2 or WMO assets may fail one-at-a-time capture due to loader/runtime issues. The library must mark those entries as failed or incomplete rather than silently dropping them.
- Z, pitch, and roll are weak or hidden from top-down imagery. Phase 1 inference should not require those pose terms for signoff.

## Requirements

### Functional Requirements

- **FR-001**: The feature MUST reuse the existing `wow-viewer` C# harvester, placement, and capture surfaces as the canonical owners of object metadata and mask generation.
- **FR-002**: The system MUST provide a one-object-at-a-time capture workflow that loads a single referenced asset and emits its image, mask, and metadata without depending on a full-scene batch render.
- **FR-003**: The per-object capture library MUST be stored under `wow-viewer/` as a dedicated Zarr-backed dataset with accompanying metadata rows.
- **FR-004**: Each object-library entry MUST preserve the original placement/listfile asset path used by the game data, plus a normalized lookup path for internal reuse.
- **FR-005**: Each object-library entry MUST include asset type (`m2`, `mdx`, or `wmo`), capture status, review state, and enough pose metadata to reproduce the capture.
- **FR-006**: Each object-library entry MUST include at least one image artifact and one precise binary or soft mask artifact.
- **FR-007**: The library SHOULD preserve multiple capture variants per asset when rotation, scale, or visibility materially changes the top-down appearance.
- **FR-008**: The teacher-prior generation path MUST consume the existing placement arrays, placement name tables, and filtered precise object signals when ADT-backed data is available.
- **FR-009**: The teacher-prior generation path MUST prefer the filtered precise/object-filtered terrain gate over coarse whole-object rectangles when both are available.
- **FR-010**: The teacher-prior generation path MUST emit a processed minimap prior that is explicitly reviewable alongside the raw minimap and the teacher object mask.
- **FR-011**: The processed-minimap-prior dataset MUST preserve raw height data as the authoritative terrain target and MUST NOT rewrite the target tensors themselves in phase 1.
- **FR-012**: The first terrain model lane under this spec MUST predict only `height_257`.
- **FR-013**: The first terrain model lane MUST NOT jointly predict normals, liquids, objects, or other terrain heads.
- **FR-014**: The first terrain model lane SHOULD start from the smallest viable V7-style or V18-style single-purpose minimap-to-height architecture without WDL priors.
- **FR-015**: The exact input channel contract for the height-only model MUST be documented and kept minimal.
- **FR-016**: The inference-time object explanation path MUST work when ADT placement metadata is absent.
- **FR-017**: The inference-time object explanation path MUST be decomposed into small independent surfaces such as object mask prediction, asset candidate matching, and lightweight pose estimation.
- **FR-018**: The first required pose outputs for minimap-only inference MUST be XY placement and yaw. Z SHOULD be derived from reconstructed terrain. Pitch and roll MAY be deferred.
- **FR-019**: The development-map runtime path MUST accept minimap and PM4 inputs without requiring ADT-derived signals.
- **FR-020**: Validation artifacts MUST show raw minimap, teacher or predicted object mask, processed minimap prior, terrain prediction preview, and recovered asset metadata side by side.
- **FR-021**: The implementation MUST remain inside `wow-viewer/` and MUST NOT introduce new dependencies on code outside the repo boundary.
- **FR-022**: All client-root validation and capture workflows MUST use staged clients under `output/tmp/wowarchive-clients/`.
- **FR-023**: The implementation MUST follow the one-model-one-signal rule. No phase may introduce a monolithic joint terrain-and-object model.
- **FR-024**: A normals lane MUST NOT begin until the height-only lane is validated on real data.
- **FR-025**: The object library MUST be reusable by both training-time teacher generation and minimap-only inference.
- **FR-026**: The spec MUST define explicit failure states for uncaptured, low-confidence, or minimap-invisible assets instead of silently treating them as good exemplars.

### Key Entities

- **ObjectLibraryEntry**: Canonical per-asset record keyed by original asset path, with normalized path, asset type, review state, visibility class, capture status, and pointers to image/mask tensors.
- **ObjectCaptureVariant**: A concrete captured view of one asset under a specific pose or capture setup, with image, mask, crop geometry, and capture metadata.
- **TeacherObjectMask**: Tile-level mask generated from ADT-backed placement data and filtered precise object signals, used only for supervised deconstruction and review.
- **ProcessedMinimapPrior**: Tile-level terrain-focused input derived from raw minimap plus object suppression, confidence, and optional fill channels.
- **HeightOnlyTrainingSample**: Training sample containing processed prior inputs, authoritative `height_257`, and review metadata.
- **InferenceObjectHypothesis**: Minimap-only prediction record containing object mask, asset candidates, confidence, XY, and yaw.
- **RecoveredObjectPlacement**: Downstream placement reconstructed from `InferenceObjectHypothesis` plus terrain-derived Z and library metadata.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A bounded proof run writes a reusable object-library Zarr store with at least 25 successfully captured assets, each with image, mask, and metadata rows.
- **SC-002**: A bounded teacher-prior run writes review artifacts for at least 100 ADT-backed tiles, including raw minimap, teacher object mask, and processed minimap prior.
- **SC-003**: A bounded height-only smoke run completes against the processed-prior dataset and emits training previews and metrics without requiring any normals head.
- **SC-004**: A bounded minimap-only inference proof runs on at least one development-map proof set and emits predicted object masks and processed priors without ADT placement input.
- **SC-005**: No phase in the implementation requires a shared-weight multi-task model to pass its independent validation gate.
- **SC-006**: The spec package clearly decomposes the work into independently validatable phases with no phase exceeding ten implementation steps.

## Assumptions

- The existing placement arrays, placement name tables, filtered masks, and precise masks are sufficient to seed the teacher path without inventing a new dataset contract.
- The current C# capture/runtime surfaces can be adapted to load and capture individual assets one at a time inside `wow-viewer`.
- A height-only proof is the right first terrain milestone; normals, liquids, and object restoration are follow-on work.
- The exact historical V7 code may or may not be directly reusable; the required contract is a tiny height-only terrain model without WDL priors, not blind attachment to a specific old file.
- Development-map inference can rely on minimap plus PM4 plus learned object explanation even when ADT-backed teacher data is absent.
- Proprietary client assets remain BYOD and are not redistributed by this workflow.

## Relationship to Other Specs

- **Supersedes in execution order**: `025-object-roof-mask-library-and-minimap-sieve` for current implementation planning. Spec 025 remains historical context only.
- **Informs**: `061-weak-signal-terrain-restoration` by defining a cleaner terrain-input contract before later weak-signal repair or scouting changes.
- **Informs**: `066-v19-height-regressor` by replacing raw baked minimap input with a deconstructed prior for future height-only runs.
- **References**: `075-scar-mask-segmentation` only as a deprecated example of why coarse full-tile segmentation without proper object/terrain decomposition is insufficient.
