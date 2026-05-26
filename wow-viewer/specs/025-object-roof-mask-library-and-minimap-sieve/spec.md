# Feature Specification: Object Roof Mask Library and Minimap Sieve

**Feature Branch**: `025-object-roof-mask-library-and-minimap-sieve`

**Created**: 2026-05-26

**Status**: Draft

**Input**: User request: build a curation library that harvests object top-view images tied to asset metadata, then uses that library to generate an on-the-fly object-roof mask for minimap inputs so training can safely ignore buildings and other object families even when placement metadata is absent at inference time.

## Problem Statement

The current terrain pipeline can already render object-visibility masks and use them as loss-side signals, but that is not enough for the intended input contract.

What is missing is a reusable object-roof corpus that tells the model what buildings, roofs, and other object families look like from above. Without that corpus, the trainer can only downweight object pixels after the fact. It cannot reliably identify and sieve object coverage out of the minimap input itself, especially when runtime inputs do not include `MODF`, `MDDF`, or full asset-path metadata.

The dataset already contains the necessary placement metadata. The task is to turn that into a curation layer and then into a learned object-roof segmentation signal that can run on every minimap input.

## Goal

Create a curation-and-sieve lane that:

1. harvests top-view object crops from the full corpus and ties them to stable asset metadata,
2. dedupes those crops into canonical object-family entries,
3. trains a separate object-identification model that learns pose-aware object identity and roof silhouettes from those crops,
4. generates an object-roof mask signal for minimap inputs,
5. feeds that signal into the main V18 model so it can learn to ignore object pixels safely while reconstructing terrain.

The first implementation slice should stay bounded: object roofs and building-family silhouettes first, not a full general-purpose semantic segmentation system.

## Scope

This feature owns the full object-roof guidance path inside `wow-viewer/data-harvester`:

1. Improve MdxViewer capture so known-used object assets can be rendered one at a time with explicit pose/transform metadata instead of only as full-scene batches.
2. Store those per-asset visual outputs in a separate Zarr datastore dedicated to object visual signals and roof exemplars.
3. Build an object-roof curation library from existing placement data, asset metadata, and the separate object-visual datastore.
4. Emit canonical roof images, family metadata, aliases, and asset-path provenance.
5. Produce training labels / masks for object pixels in minimap inputs.
6. Train a separate transformer-based object-identification / segmentation model in the Python `uv` environment that predicts object coverage and family class from minimap imagery and roof exemplars.
7. Use the predicted object-roof mask as an auxiliary on-the-fly signal during training and inference.
8. Feed the object-identification outputs into the main V18 model as auxiliary signals so the main model can reconstruct terrain while ignoring object pixels.
9. Prefer SAM2 as the first promptable mask-generation host, and allow SAM3 as a gated follow-on only when the Hugging Face token has approved access to the checkpoint.

Out of scope for the first slice:

- changing the terrain target tensors themselves
- redesigning the terrain model family
- replacing the existing renderer-truth object-mask pipeline
- shipping checkpoints or public datasets

## User Scenarios & Testing

### User Story 1 — Object roofs are curated as a library

A researcher can inspect a library of object top-view exemplars with stable metadata, so object families can be reviewed and reused instead of rediscovered for every run.

**Independent Test**: run the roof curator on a bounded corpus and verify it emits canonical roof crops, stable family IDs, and asset-path provenance for at least one building-heavy map.

**Acceptance Scenarios**:

1. **Given** placement metadata and source assets, **When** roof curation runs, **Then** it emits a top-view image for each canonical object family.
2. **Given** repeated placements of the same object family across builds, **When** curation completes, **Then** the family is deduped to one canonical entry plus variants.
3. **Given** a curated roof family, **When** the catalog is inspected, **Then** it shows asset-path metadata, family tags, and review state.

---

### User Story 2 — Minimap inputs can be sieved using learned object coverage

A trainer can identify building/object pixels in a minimap input even when direct placement metadata is missing, by applying a learned object-roof mask signal.

**Independent Test**: run a bounded mask-generation job on a tile set and verify the output mask is non-empty on object-rich tiles and sparse on terrain-only tiles.

**Acceptance Scenarios**:

1. **Given** a minimap tile with buildings, **When** the mask generator runs, **Then** it predicts object coverage over the building footprints.
2. **Given** a minimap tile with little or no object coverage, **When** the generator runs, **Then** it produces a low-object mask and preserves terrain pixels.
3. **Given** a tile with missing placement metadata, **When** the generator runs, **Then** it falls back to the learned object-roof recognizer instead of failing the sample.

---

### User Story 3 — Training can safely ignore object pixels without losing terrain truth

A normal-lane trainer can use the object-roof signal as an auxiliary mask so the model learns to ignore object pixels during terrain reconstruction rather than treating the object area as reliable terrain truth.

**Independent Test**: run a bounded training smoke pass and confirm the model consumes the object-roof signal, writes evidence, and still trains against the raw terrain targets.

**Acceptance Scenarios**:

1. **Given** a training batch with object-roof masks, **When** the loss is computed, **Then** object pixels are masked or downweighted in the image-sieving path.
2. **Given** a tile with a building roof over terrain, **When** the trainer runs, **Then** the auxiliary signal helps the model ignore the building while preserving terrain supervision.
3. **Given** the auxiliary signal is disabled, **When** training starts, **Then** the run falls back to the existing terrain-only contract.

### Edge Cases

- Some assets will have strong placement metadata and clean roof crops; others will only have partial footprints or noisy bounds. The curation layer must preserve both and mark confidence explicitly.
- Some object families are visually similar from above. The roof library must keep family-level lineage and not collapse distinct silhouettes into one class too early.
- Some runtime minimap inputs will not have placement metadata. The learned object-roof signal must cover those cases.
- Some non-building objects may be better treated as clutter than as architectural roofs. The first slice should be able to tag them separately or mark them as lower-confidence classes.

## Requirements

### Functional Requirements

- **FR-001**: The curation layer MUST emit top-view object exemplars from existing placement and asset metadata.
- **FR-002**: Each curated exemplar MUST carry stable asset-path provenance, object-family metadata, and review state.
- **FR-003**: The roof library MUST dedupe repeated objects into canonical families plus variants.
- **FR-004**: The library MUST preserve enough metadata to re-identify the original object source for each exemplar.
- **FR-005**: The system MUST generate an object-roof mask signal for minimap inputs.
- **FR-006**: The object-roof mask MUST work when placement metadata is available and MUST also support inference-time fallback when metadata is missing.
- **FR-007**: The first implementation slice MUST focus on buildings/roofs and related architectural object families before broadening to all object classes.
- **FR-008**: The mask signal MUST be consumable by training as an auxiliary sieve or ignore mask, not only as a loss-side downweighting term.
- **FR-009**: The trainer MUST continue to use raw terrain tensors as authoritative truth; the object-roof signal is auxiliary.
- **FR-010**: Validation outputs MUST expose the roof library exemplar, predicted object mask, and the underlying minimap tile together for review.
- **FR-011**: The object-roof pipeline MUST remain inside `wow-viewer/` and MUST not depend on external asset databases beyond the existing corpus and staged client data.
- **FR-012**: The pipeline SHOULD expose family-level confidence so low-confidence object classes can be reviewed or locked.
- **FR-013**: The object-roof signal SHOULD be reusable as an on-the-fly preprocessing step for all minimap inputs.
- **FR-014**: The separate object-identification model SHOULD be implemented in the Python `uv` workflow and SHOULD use the Hugging Face transformers stack as its first host unless a proven blocker requires a different backend.
- **FR-015**: The object-identification model outputs SHOULD include pose-aware class or mask signals that the main V18 model can consume as auxiliary inputs.

### Key Entities

- **Roof Exemplar**: a top-view image of a placed object or object family, tied to its source asset metadata.
- **Roof Family**: a canonical deduped family of similar roof exemplars with variants.
- **Asset Provenance**: the object path, build, map, tile, and placement metadata used to source a roof exemplar.
- **Object Visual Zarr Store**: a separate datastore of per-asset visual outputs, pose metadata, and object-family crops used to train and verify object recognition.
- **Asset Pose Capture**: a one-at-a-time capture job that records rotation, scale, transform, and source asset path for a known-used object.
- **Object Identification Model**: a separate transformer-based model that consumes minimap/object-visual inputs and predicts object class, roof mask, and pose-aware coverage signals.
- **Object-Roof Mask**: a pixel mask over a minimap input that marks object/building coverage.
- **Object Sieve**: the preprocessing step that uses the predicted mask to hide or downweight object pixels for terrain reconstruction.
- **Auxiliary Object Signal**: a mask, class, or confidence output from the object-identification model that is fed into the main V18 model.

## Success Criteria

- **SC-001**: A bounded roof-catalog run produces stable family IDs and top-view object exemplars for building-heavy tiles.
- **SC-002**: A bounded mask-generation run produces non-empty object-roof masks on tiles with visible buildings.
- **SC-003**: A separate object-visual Zarr datastore exists and contains per-asset visual outputs plus pose metadata for known-used object families.
- **SC-004**: A separate object-identification model exists and produces pose-aware object masks or class signals from roof pixels / minimap inputs.
- **SC-005**: The learned object-roof signal can be consumed in a training smoke run without breaking the terrain target contract.
- **SC-006**: The system still trains on mixed-coverage data where placement metadata is incomplete.
- **SC-007**: Review artifacts show the roof exemplar, minimap input, predicted mask, and object-family signal together for at least one proof tile.
- **SC-008**: The object-identification model is runnable via `uv` in the Python data-harvester workflow and emits pose-aware auxiliary signals that the main model can consume.

## Assumptions

- The corpus already contains enough placement metadata and asset paths to build a meaningful roof library.
- MdxViewer can be extended to capture known-used assets individually without requiring full-scene batch rendering for every object sample.
- Building roofs and other large architectural objects are the highest-value first class to model.
- A transformer-based recognizer is a reasonable first implementation host because the signal is visual, pose-sensitive, and benefits from pretrained object-context representations.
- SAM2 is the safest first host for promptable object masking because it is already documented, broad in utility, and aligned with the initial roof-sieve slice.
- SAM3 is a valid follow-on host if the Hugging Face token is authorized for the gated checkpoint and the model improves roof/object mask quality.
- A learned object-roof signal is preferable to hard-coding object ignorance into the terrain model.
- The existing object-mask loss path remains useful, but it is not the end state.
- The first useful version can be bounded to a small set of maps/builds before broadening.

## Relationship to Other Specs

- **Depends on**: `013-object-mask-rendering-fix` for renderer-truth visibility evidence.
- **Depends on**: `017-mdxviewer-port-headless-capture` for efficient per-asset capture and object-visual datastore generation.
- **Extends**: `024-v18-canvas-paste-refinement-layer` as a parallel curation lane for object content instead of terrain paste content.
- **Informs**: `021-cross-signal-curation-and-validation-rotation` by adding another cross-signal quality surface.
- **Enables**: future object-aware terrain training and inference that can sieve object pixels instead of only downweighting them.
