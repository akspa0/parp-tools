# Feature Specification: Minimap-Only Terrain Reconstruction (Spec 102 Reset)

**Feature Branch**: `102-v25-terrain-convergence`
**Reset Date**: 2026-07-12
**Status**: BLOCKED AT FEASIBILITY GATE
**Owner**: wow-viewer

## Problem Statement

The deployment input is one or more raw RGB minimap tiles. The required primary output is terrain height data aligned to the tile grid. No WDL, ADT height, normal, alpha, object mask, placement record, client coordinate, PM4 record, or target-derived signal exists at deployment unless a later user requirement explicitly adds it.

The previous Spec 102 mixed this minimap-only product with WDL-guided refinement and a large multi-head decompiler. That violated the program's modular-model rule and made failures impossible to isolate. All previous architecture and quality claims are invalidated until they pass the input-availability, single-output-model, and held-out-map gates below.

## Input Invariant

- Production model input is RGB minimap pixels only.
- Adjacent minimap tiles may be used only when they are supplied together as RGB pixels at deployment.
- Training labels may supervise outputs but MUST NOT enter the forward input, initialization, teacher-forcing route, feature cache, normalization anchor, or post-processing path.
- WDL may be derived from predicted terrain heights for evaluation or export. It MUST NOT be required to generate those heights.
- Historical V24/V25 checkpoints and metrics are evidence records, not accepted baselines or reusable architecture proof.

## Modular Pipeline Invariant

- Every learned model predicts exactly one signal.
- Every model trains independently with its own checkpoint, metrics, and stop gate.
- Models never share weights and never train jointly.
- A downstream model may consume frozen outputs from an upstream model.
- Every height model predicts a residual over an explicit simpler baseline.
- Failure of one stage blocks downstream training; it does not trigger more heads or a longer end-to-end run.

The initial height chain is:

1. **H0 Offset**: RGB minimap → one tile elevation-offset correction residual over the frozen RGB-flat baseline.
2. **H1 Coarse Relief**: RGB minimap + frozen H0 output → one low-frequency 33×33 relief residual.
3. **H2 Terrain Detail**: RGB minimap + frozen upsampled H1 output → one 257×257 detail residual.
4. **H3 Border Correction**: adjacent RGB tiles + frozen H2 borders → one shared-border correction residual.
5. **U1 Height Uncertainty**: RGB minimap + frozen height outputs → one 257×257 uncertainty signal.

H1 is not an externally supplied WDL prior. It is a learned coarse residual produced from deployment-available RGB and H0 output. A WDL-shaped export may be derived later from the predicted height chain.

## User Scenarios & Testing

### User Story 1 — Reconstruct height from minimap pixels (P1)

Given a raw RGB minimap tile, the user receives a 257×257 terrain height prediction assembled from independently validated residual stages. Uncertainty is produced by a separate model.

**Independent test**: Delete access to every training store after loading a PNG and run inference successfully.

### User Story 2 — Reconstruct adjacent tiles consistently (P1)

Given adjacent minimap tiles, the user receives height predictions whose shared borders do not form artificial cliffs.

**Independent test**: Evaluate held-out adjacent tiles and measure border disagreement before any stitching correction.

### User Story 3 — Reject unavailable-input leakage (P1)

Before GPU training, the operator receives a machine-readable audit of every model input and its deployment source. Training refuses to start if any input is unavailable from RGB minimap pixels at inference.

## Functional Requirements

- **FR-102-R001**: No model may have multiple prediction heads or optimize multiple output families. Each model has one output signal, one loss family, and one checkpoint.
- **FR-102-R002**: A deploy-input manifest MUST enumerate every tensor entering `forward`, its shape, and its RGB-only derivation. The trainer MUST fail closed when the manifest and model signature disagree.
- **FR-102-R003**: Dataset splitting MUST hold out complete maps and at least one build/era. Random tile splits from the same maps are not quality proof.
- **FR-102-R004**: The benchmark MUST include zero-height, train-global-mean, and an RGB-derived flat-height baseline evaluated on the identical held-out set. Per-tile target means are prohibited because they are unavailable at deployment.
- **FR-102-R005**: Training labels MAY include height, WDL, normals, objects, liquids, textures, and curation facts, but Phase 0 consumes only height as the optimization target and liquid/validity masks as loss masks.
- **FR-102-R006**: Absolute offset, relative relief, slope, low-frequency structure, border continuity, and uncertainty MUST be reported separately. A single aggregate loss is insufficient.
- **FR-102-R007**: Runs longer than three epochs are prohibited until a bounded smoke has finite gradients, stable validation, and beats the registered trivial baseline.
- **FR-102-R008**: Every GPU run MUST record command, code revision, dataset identity, split manifest, peak VRAM, energy-relevant duration, and per-epoch validation metrics.
- **FR-102-R009**: WDL export, object reconstruction, texture reconstruction, alpha reconstruction, and PM4 guidance are later independent phases. None may be used to claim Phase 0 height success.
- **FR-102-R010**: The current unified V25 trainer MUST remain fail-closed until a replacement RGB-only height trainer satisfies FR-102-R001 through FR-102-R008.
- **FR-102-R011**: H0 MUST pass its held-out offset gate before H1 training begins; H1 MUST pass its coarse-relief gate before H2 begins; H2 MUST pass before H3 or U1 begins.
- **FR-102-R012**: H0, H1, H2, H3, and U1 MUST use separate optimizers, training commands, checkpoints, and metric histories. Joint fine-tuning is prohibited.
- **FR-102-R013**: Object masks, cleaned terrain images, placements, tilesets, alpha maps, liquids, holes, normals, and shadows each require their own future single-output model and independent gate.

## Success Criteria

- **SC-102-R001**: Inference succeeds from RGB minimap files after all dataset and game-client paths are made unavailable.
- **SC-102-R002**: The deploy-input audit reports zero unavailable or target-derived inputs.
- **SC-102-R003**: On held-out maps, the Phase 0 model improves height L1 by at least 20% over the best registered deployable baseline on the identical evaluation set. Historical results count only if rerun on that frozen split.
- **SC-102-R004**: At least 95% of held-out shared borders remain within the registered border-error threshold before post-processing.
- **SC-102-R005**: Validation reports calibration: higher predicted uncertainty corresponds to higher observed height error.
- **SC-102-R006**: The bounded trainer remains below 7 GB peak VRAM and completes its three-epoch decision run without NaN, OOM, or silent CPU fallback.
- **SC-102-R007**: The registry shows exactly one output signal per checkpoint and no shared trainable weights between pipeline stages.

## Out of Scope Until Phase 0 Passes

- Externally supplied WDL-prior prediction or refinement as an internal prerequisite
- Object placements or PM4 snapping
- MTEX, MCLY, MCAL, liquid, hole, shadow, and normal generation
- ADT/WDL binary writing
- Claims of universal or production-ready reconstruction

## Assumptions

- RGB minimap tiles are the only guaranteed deployment artifact.
- Training-time terrain labels remain available for supervised evaluation.
- The initial goal is an honest staged feasibility result, including a documented stop if any residual stage fails its gate.
