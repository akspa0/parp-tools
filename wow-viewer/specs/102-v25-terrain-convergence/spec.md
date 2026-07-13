# Feature Specification: Minimap-Only Terrain Reconstruction (Spec 102 Reset)

**Feature Branch**: `102-v25-terrain-convergence`
**Reset Date**: 2026-07-12
**Status**: BLOCKED AT M0 SUPERVISION GATE
**Owner**: wow-viewer

> **2026-07-12 numeric-lattice correction:** `wdl_height_33 = height_257[::8, ::8]` is not a WDL MARE. The actual C# contract is `outer_17` (17x17 at `::16`) plus `inner_16` (16x16 at `8::16`). Treat the prior H1 target and every `wdl_height_33` consumer as invalid proof. The recovery decision is [`v25-v24-numeric-lattice-recovery-audit-2026-07-12.md`](../../docs/architecture/v25-v24-numeric-lattice-recovery-audit-2026-07-12.md).

## Problem Statement

The deployment input is one or more raw RGB minimap tiles. The required primary output is the terrain **mesh vertex field**: fixed ADT topology and world X/Y coordinates plus predicted numeric Z values. A preview heightmap, normal map, mesh render, or error heatmap is a visualization derived after inference, never the model's terrain representation or training input.

The previous Spec 102 mixed this minimap-only product with WDL-guided refinement and a large multi-head decompiler. That violated the program's modular-model rule and made failures impossible to isolate. All previous architecture and quality claims are invalidated until they pass the input-availability, single-output-model, and held-out-map gates below.

## Input Invariant

- Production model input is RGB minimap pixels only.
- Adjacent minimap tiles may be used only when they are supplied together as RGB pixels at deployment.
- Training labels may supervise outputs but MUST NOT enter the forward input, initialization, teacher-forcing route, feature cache, normalization anchor, or post-processing path.
- WDL may be derived from predicted terrain heights for evaluation or export. It MUST NOT be required to generate those heights.
- Raster minimap/shadow pixels are valid inputs because they are observed imagery. Terrain targets remain mesh-native numeric vertices; the model MUST NOT train image-to-image terrain reconstruction merely for storage or preview convenience.
- The existing dense `height_257` / `normal_xyz` arrays are materialized dataset views, not automatically the canonical mesh contract. Before any new terrain training, a real-data audit MUST prove their exact reversible mapping to raw MCVT vertices, including valid-node topology and coordinate order.
- Validation may render predictions to PNG/OBJ/interactive mesh views only after numeric mesh metrics have been calculated. Those artifacts MUST NOT re-enter dataset construction, normalization, or model `forward` inputs.
- A real-client **terrain-shadow capture** may be produced by the existing viewer capture automation with objects, liquids, diffuse textures, alpha-mask compositing, and vertex-colour tint disabled, using the fixed global-light settings. It is a training-time geometry-guidance/diagnostic target, not a deployment input and not an image-shaped terrain output.
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
2. **W1 WDL Lattice**: cleaned RGB + frozen H0 output → one paired 17x17 + 16x16 (545-sample) numeric WDL-lattice residual.
3. **H2 Terrain Vertices**: cleaned RGB + frozen W1 upsample → one mesh-native ADT vertex-Z residual; it predicts values at real vertices, not pixels in a 257x257 image.
4. **H3 Border Correction**: adjacent RGB tiles + frozen H2 borders → one shared-border correction residual.
5. **U1 Height Uncertainty**: RGB minimap + frozen height outputs → one 257×257 uncertainty signal.

W1 is not an externally supplied WDL prior. It is a learned numeric lattice residual produced from deployment-available cleaned RGB and frozen H0 output. A WDL export may be derived later from the predicted mesh-vertex chain.

## User Scenarios & Testing

### User Story 1 — Reconstruct height from minimap pixels (P1)

Given a raw RGB minimap tile, the user receives a terrain mesh-vertex-Z prediction assembled from independently validated residual stages. A 257×257 view may be rendered for inspection only. Uncertainty is produced by a separate model.

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
- **FR-102-R005**: Training labels MAY include raw MCVT vertex Z values, paired WDL lattice samples, normals, objects, liquids, textures, and curation facts. Native normals are numeric validation facts, never forward image inputs or a second output head.
- **FR-102-R005A**: The canonical terrain target MUST preserve raw vertex topology, valid-node mask, world-coordinate convention, and source row/chunk identity. A dense height array may be emitted as a verified projection for display, but cannot be the only stored or supervised terrain truth.
- **FR-102-R005B**: Each H2 prediction MUST be a numeric vertex-Z vector/lattice aligned to the canonical topology. The loss and all primary metrics operate only at real vertices; any raster interpolation cells are excluded.
- **FR-102-R005C**: Validation images are post-inference observability artifacts. They may display terrain-shadow input, rendered predicted mesh, rendered ground-truth mesh, and numeric vertex error, but cannot become training samples or model inputs.
- **FR-102-R005D**: A terrain-shadow capture contract MUST record staged client/build, map/tile, fixed camera, explicit renderer-light parameters, renderer revision, and every disabled render feature. It must use real client terrain and the canonical viewer renderer. Until client light tables are actually wired, the manifest MUST label the light as a fixed viewer contract and MUST NOT claim it was decoded from client light data.
- **FR-102-R005E**: The primary H2 loss remains numeric vertex-Z error. After its numeric baseline passes, a deterministic render of the predicted mesh under the recorded terrain-shadow contract MAY supply a separately reported guidance loss/metric against the real capture. The capture never enters `forward` and never replaces vertex metrics.
- **FR-102-R006**: Absolute offset, relative relief, slope, low-frequency structure, border continuity, and uncertainty MUST be reported separately. A single aggregate loss is insufficient.
- **FR-102-R007**: Runs longer than three epochs are prohibited until a bounded smoke has finite gradients, stable validation, and beats the registered trivial baseline.
- **FR-102-R008**: Every GPU run MUST record command, code revision, dataset identity, split manifest, peak VRAM, energy-relevant duration, and per-epoch validation metrics.
- **FR-102-R009**: WDL export, object reconstruction, texture reconstruction, alpha reconstruction, and PM4 guidance are later independent phases. None may be used to claim Phase 0 height success.
- **FR-102-R010**: The current unified V25 trainer MUST remain fail-closed until a replacement RGB-only height trainer satisfies FR-102-R001 through FR-102-R008.
- **FR-102-R011**: H0 and the frozen M0 cleaner MUST pass their gates before W1 begins; W1 MUST pass its numeric lattice gate before H2 begins; H2 MUST pass before H3 or U1 begins.
- **FR-102-R012**: M0, H0, W1, H2, H3, and U1 MUST use separate optimizers, training commands, checkpoints, and metric histories. Joint fine-tuning is prohibited.
- **FR-102-R013**: Object masks, cleaned terrain imagery, placements, tilesets, alpha maps, liquids, holes, normals, and shadows each require their own future single-output model or deterministic transform and independent gate. The first intended terrain visual **guidance** is real fixed-light terrain-shadow imagery; it is correlated to numeric mesh shape, not used as an excuse to rasterize the mesh target or add an unavailable deployment input.
- **FR-102-R014**: Every M0 validation PNG MUST be self-describing without external documentation. It embeds split, epoch, threshold, checkpoint, column meanings, tile identity, per-row IoU/Dice/pixel counts, and a colour legend for true positives, false positives, and false negatives. These panels are post-inference observability artifacts only.

## Success Criteria

- **SC-102-R001**: Inference succeeds from RGB minimap files after all dataset and game-client paths are made unavailable.
- **SC-102-R002**: The deploy-input audit reports zero unavailable or target-derived inputs.
- **SC-102-R003**: On held-out maps, the Phase 0 model improves vertex-Z L1 by at least 20% over the best registered deployable baseline on the identical real-vertex set. Historical results count only if rerun on that frozen split.
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
- Synthetic terrain targets, fabricated raster vertices, or preview images as terrain supervision
- Treating a terrain-shadow capture as deployment input unless a future product contract explicitly supplies it

## 2026-07-12 Invalidated Run Notice

The first M0 run violated the dataset contract by training from the reduced
`object_mask_256` array instead of the canonical `object_precise_mask_257`
signal. Its metrics and every W1 result derived from that checkpoint are
invalid and must not be used for feasibility decisions. M0 training now fails
closed unless the exact 257x257 precise array exists and derives its 256x256
loss target with the registered four-corner maximum projection. No coarse,
visibility, or fallback mask is permitted.

The corrected numeric-only store verifies build/map/tile identity against the
originating V18 indices and retains raw unrepaired height, numeric normals,
precise object masks, liquid coverage/height, MCNK flags, and liquid-source
provenance. A 2026-07-12 audit found that V18 extraction existed, but MH2O's
8x8 cells had been incorrectly emitted as sparse vertices on the 16x16
half-step coverage grid. The canonical builder now expands each present MH2O
cell to its 2x2 half-step coverage block and respects the exists bitmap. The
Spec 102 v3 store is an exact copy of the repaired V18 masks; the stale liquid
type raster is excluded. Curation rejects known mismatches, visually uniform
or water minimaps, missing liquid evidence, and tiles with at least 80% liquid
occlusion. The 2026-07-13 rerun used this exact v3/v6 contract. Its bounded
three-epoch run improved every epoch and authorized continuation; the resumed
12-epoch run reached held-out Northrend IoU `0.2764` but Alpha-era IoU only
`0.0743` (required `0.10`), with `2.15 GB` peak VRAM. Threshold calibration
selected `0.50` and did not improve the era result. M0 is therefore not frozen
and W1 remains blocked. W1 also has no eligible rows until real paired WDL
arrays exist.

## Assumptions

- RGB minimap tiles are the only guaranteed deployment artifact.
- Training-time terrain labels remain available for supervised evaluation.
- The initial goal is an honest staged feasibility result, including a documented stop if any residual stage fails its gate.
