# Feature Specification: V18 Focused Two-Build Terrain Reconstruction System

**Feature Branch**: `047-v18-distill-corpus-open-source-loop`

**Created**: 2026-06-04

**Status**: Final Design Draft — 2026-06-05

## Intent

The V18 lane exists to turn curated minimap tiles into stitchable terrain
reconstruction for two real corpus anchors only:

- `0_5_3_3368`
- `3_3_5_12340`

The desired product is not a pretty single-tile preview. The desired product is
a pipeline that accepts a set of minimap tile images and emits terrain data
that can be quilted back into ADT terrain with believable shape and border
continuity.

## Problem Statement

The project already proved local terrain reconstruction in earlier model lines,
but the owner workflow drifted into too many secondary signals and too much
dataset width.

The actual problems to solve are:

1. use only the two builds that matter (`0_5_3_3368` and `3_3_5_12340`),
2. keep only tiles whose minimap/height/normal/liquid data is coherent enough
   to train against,
3. train two focused terrain models:
   - `minimap_rgb -> normalized height field`
   - `minimap_rgb -> normal field`
4. use the outputs of those models together in a downstream quilt/inference
   stage that reconstructs terrain, not just isolated tile previews.

The V18 dataset is considered functionally complete enough to support this lane.
The work remaining is to finalize the design, lock the focused operator
workflow, and build the curation/training surfaces that put the existing data
to work.

## Scope

### In Scope

- focused V18 corpus: `0_5_3_3368` + `3_3_5_12340` only
- V18 curation manifest generation on the focused V18 stores
- V18 height training from minimap input
- V18 normal training from minimap input
- liquid masks as curation and terrain-validity signals
- downstream design contract for quilt-level terrain stitching and ADT writeback

### Out of Scope

- all other builds
- renderer-truth capture as required training truth
- precise object masks and MDX/M2 omission masks in the active training lane
- new monolithic multitask terrain models
- synth/distill/open-source release work
- reopening alphaWDT writer work

## Design Principles

1. **Two builds only**
   More corpus width is not the active bottleneck.

2. **Curation over clever losses**
   Bad tiles are a larger problem than missing auxiliary masks.

3. **Independent models**
   Height and normals remain separate V18 models. They are used together during
   terrain reconstruction, but they are not trained as one shared-weight
   multitask checkpoint.

4. **Liquids matter**
   Liquid masks remain valid supervisory context because hidden underwater
   terrain is poorly explained by minimap imagery. They are used to filter or
   de-emphasize unusable tiles/regions, not to reintroduce a large speculative
   loss stack.

5. **WMO roof and basement occlusion both matter**
   Terrain-valid masking must exclude both underground/basement WMO regions and
   above-ground roof/top-geometry occlusion. The active lane may stay simple,
   but it cannot collapse those two visibility seams into one basement-only
   mask.

5. **Tile plausibility is not enough**
   The product must support quilt-level terrain reconstruction. The final
   pipeline includes a post-model stitching stage.

## User Scenarios & Testing

### User Story 1 — Focus the corpus to the two useful builds (Priority: P1)

A researcher can operate only on `0_5_3_3368` and `3_3_5_12340` and avoid the
other four builds entirely.

**Independent Test**: build, validate, and curate only the two focused V18
stores.

**Acceptance Scenarios**:

1. **Given** the staged client roots for the two focused builds, **When** the
   focused V18 workflow runs, **Then** only `0_5_3_3368.zarr` and
   `3_3_5_12340.zarr` are required inputs.
2. **Given** those two stores, **When** signal validation and curation run,
   **Then** they produce a focused manifest with no dependency on the other
   builds.

---

### User Story 2 — Curate only coherent terrain tiles (Priority: P1)

A researcher can build a focused curation manifest that rejects the tiles most
likely to poison terrain learning: blank tiles, liquid-dominated hidden-terrain
tiles, bad minimap/normal alignment, erased-normal leftovers, and similar
low-value rows.

**Independent Test**: run the focused V18 curation manifest builder and inspect
`summary.json`, `tiles.parquet`, and `kept_tiles.parquet`.

**Acceptance Scenarios**:

1. **Given** the focused V18 stores, **When** curation runs, **Then** the same
   curation surface is applied to both train and validation pools.
2. **Given** liquid masks and terrain-validity metrics, **When** tiles are
   scored, **Then** obviously unusable liquid/blank/what-plate rows can be
   excluded without requiring renderer truth, including rows with too little
   surviving trainable terrain.
3. **Given** the focused manifest, **When** a training run consumes it,
   **Then** both train and validation tile pools are drawn only from curated
   rows.

---

### User Story 3 — Train a V18 height model from minimap tiles (Priority: P1)

A researcher can train a focused V18 height model that predicts normalized
height structure from minimap imagery alone.

**Independent Test**: run a bounded V18 height training pass with the focused
manifest and verify evidence/checkpoints are produced.

**Acceptance Scenarios**:

1. **Given** the focused curated manifest, **When** the V18 height trainer
   runs, **Then** its input contract is `minimap_rgb -> normalized height`.
2. **Given** the active height lane, **When** loss is computed, **Then** it is
   plain height supervision over terrain-valid regions rather than a large
   auxiliary loss bundle.
3. **Given** the two-build focused lane, **When** epoch subsets are sampled,
   **Then** neither build may silently dominate because the focused operator
   defaults enforce near-equal per-build sampling when feasible.
3. **Given** the run finishes, **When** evidence is inspected, **Then** the run
   records command, seed, manifest, and checkpoint under `models/v18/height/`.

---

### User Story 4 — Train a V18 normal model from minimap tiles (Priority: P1)

A researcher can train a focused V18 normal model that predicts terrain normals
from minimap imagery alone.

**Independent Test**: run a bounded V18 normal training pass with the focused
manifest and verify evidence/checkpoints are produced.

**Acceptance Scenarios**:

1. **Given** the focused curated manifest, **When** the V18 normal trainer
   runs, **Then** its input contract is `minimap_rgb -> normal_xyz`.
2. **Given** the active normal lane, **When** loss is computed, **Then** it is
   plain masked cosine using terrain-valid normal regions rather than
   object/roof auxiliary weighting.
3. **Given** the run finishes, **When** evidence is inspected, **Then** the run
   records command, seed, manifest, and checkpoint under `models/v18/normal/`.

---

### User Story 5 — Reconstruct a stitched terrain quilt for downstream ADT work (Priority: P2)

A researcher can run height + normal inference over a set of minimap tiles and
feed those predictions into a quilt-level terrain assembly stage that prepares
the terrain for ADT writeback.

**Independent Test**: define and document the contract for a quilt inference job
that consumes predicted terrain tiles and emits a stitch-ready terrain quilt.

**Acceptance Scenarios**:

1. **Given** a set of focused minimap tiles, **When** inference runs for height
   and normals, **Then** per-tile predictions are emitted with enough metadata
   to place them back into quilt coordinates.
2. **Given** neighboring predicted height tiles, **When** the quilt assembly
   stage runs, **Then** it can solve border continuity rather than treating
   every tile as an isolated mesh.
3. **Given** a solved terrain quilt, **When** ADT writeback is reopened in a
   later implementation slice, **Then** the output contract already names the
   required terrain arrays and placement metadata.

## Functional Requirements

- **FR-001**: The focused V18 dataset contract MUST use only `0_5_3_3368` and
  `3_3_5_12340`.
- **FR-002**: The active V18 curation workflow MUST operate directly on
  `wow-viewer/output/datasets/v18/*.zarr`.
- **FR-003**: The active curation workflow MUST score and filter tiles using
  minimap, height, normal, terrain-validity, and liquid-derived signals.
- **FR-004**: The active curation workflow MUST produce `tiles.parquet`,
  `kept_tiles.parquet`, and `summary.json`.
- **FR-005**: The V18 height trainer MUST consume minimap RGB only.
- **FR-006**: The V18 normal trainer MUST consume minimap RGB only.
- **FR-007**: The V18 height and normal trainers MUST remain separate model
  runs and separate checkpoints.
- **FR-008**: Liquid masks MUST remain available to the curation and
  terrain-validity surfaces.
- **FR-008A**: The active height and normal losses MUST ignore terrain-hidden
  liquid/object regions via terrain-valid masking, without reintroducing a
  broad auxiliary liquid-weight stack.
- **FR-008B**: Terrain-valid masking and focused preview evidence MUST include
  both WMO basement/ground masks and WMO roof/top-geometry masks when those
  signals are present in the harvested store.
- **FR-009**: The active training lane MUST NOT require renderer-truth capture
  arrays.
- **FR-010**: The focused operator workflow MUST expose dedicated V18
  curation/training entrypoints instead of relying only on older V16-named
  scripts.
- **FR-010A**: The focused operator workflow MUST keep the two-build train/val
  sampling balanced by default, capping oversized balanced subsets when one
  build has fewer eligible rows.
- **FR-011**: Height training MAY remain normalized, but the final V18 design
  MUST treat quilt-level stitching as the owner of cross-tile continuity rather
  than isolated tile previews.
- **FR-012**: All active surfaces MUST remain under `wow-viewer/`.

## Key Entities

- **Focused V18 Corpus**: the two V18 stores for `0_5_3_3368` and
  `3_3_5_12340`.
- **Focused Curation Manifest**: the filtered list of trainable rows produced
  from the focused V18 corpus.
- **Height Model Run**: an independent V18 training run for minimap-to-height
  prediction.
- **Normal Model Run**: an independent V18 training run for minimap-to-normal
  prediction.
- **Liquid Validity Signal**: the liquid-aware terrain-validity context used to
  reject or de-emphasize poor training rows.
- **Terrain Quilt Job**: the future inference-stage job that combines per-tile
  predictions into a stitch-ready terrain quilt.

## Success Criteria

- **SC-001**: A dedicated focused V18 curation manifest is generated from the
  two V18 stores and is directly consumable by V18 training commands.
- **SC-002**: One bounded focused V18 height run completes and writes evidence
  under `models/v18/height/runs/`.
- **SC-003**: One bounded focused V18 normal run completes and writes evidence
  under `models/v18/normal/runs/`.
- **SC-004**: Focused operators no longer need to remember V16 dataset paths or
  six-build command sets to run the V18 lane.
- **SC-005**: The final design names a quilt-level terrain assembly contract so
  the model outputs are explicitly aimed at stitched ADT reconstruction rather
  than isolated tile previews only.

## Assumptions

- The focused V18 stores are complete enough to serve as the active training
  source.
- The existing curation logic is already strong enough to reject the worst
  misleading tiles once it is pointed at the focused V18 stores.
- The short-term objective is to stand up the focused training lane and prove
  it, not to solve final ADT writeback in the same slice.

## Relationship to Existing Specs

- **Builds on**: `001-v18-dataset-spec`
- **Supersedes for active execution**: older `047` wording that treated this as
  only a plain minimap-only smoke lane with no final terrain-system contract
- **Reuses**: the existing V16.1 / V18 trainer implementation surface, focused
  V18 corpus builder, and existing curation heuristics
