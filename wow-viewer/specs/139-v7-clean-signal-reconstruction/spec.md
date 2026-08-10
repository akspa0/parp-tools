# Feature Specification: V7-Inspired Clean-Signal Terrain Reconstruction

**Feature Branch**: `139-v7-clean-signal-reconstruction`

**Created**: 2026-08-10

**Status**: Phase 5 implementation and minimap-observable raw-RGB diagnostic preparation complete; promotion and albedo-normalized transfer remain held pending user-run evidence

**Input**: User description: "Build the old v7 model idea with a modern architecture, guided by
clean synthetic data and a sane signal set. Remove the WDL-prior dependency so any minimap can be
processed after albedo normalization."

## Implementation checkpoint — 2026-08-10

The first bounded implementation slice is now present under `data-harvester/src/harvester/v60/`:

- `clean_signal_inputs.py` validates and packages exactly four deployment channels: luma, x/y
  finite-difference gradients, and albedo confidence. Rejected, quarantined, stale, and
  target-contaminated observations fail closed.
- `clean_signal_targets.py` preserves the existing per-tile range-floor semantics and emits a
  versioned edge-replicated box low-pass coarse field plus signed detail residual.
- `clean_signal_corpus.py` validates NPZ shapes, finite/range constraints, array hashes,
  recomposition, source-group leakage, split mode, and forbidden-signal provenance.
- `clean_signal_model.py` adapts `pyramid_cnn`, `segformer_b0`, and `unet_lite_v2` to one
  four-channel input and independent coarse/detail heads. Identities are JSON-serializable,
  hash-bound, random-initialized, and reconstructable without external weights.
- `v60_build_clean_signal_corpus.py` is a dry-run-first, atomic builder from the validated
  `control_manifest.json`; it preserves family/variant/split provenance and writes all seven
  named arrays plus hashes. It refuses existing output and leaves any failed `.partial` root
  non-validating for inspection.
- `v60_validate_clean_signal_corpus.py` is a fail-closed report CLI.
- `v60_visualize_clean_signal.py` renders family, variant, and complete cross-tile atlases while
  retaining validation metrics and row provenance in a JSON review report.
- `clean_signal_losses.py` defines versioned `parity` and `v7_structural_v1` profiles. Point,
  gradient, full-spectrum, Laplacian, Sobel edge, transition, border, and low/high-frequency
  components remain independently measurable and differentiable; adversarial and object/recovery
  terms are excluded from the first clean lane.
- `clean_signal_train.py` fixes deterministic within-family and complete-family split identities,
  lazy four-channel NPZ loading, independent final/coarse/detail evaluation, family and complexity
  bucket reports, and best/last checkpoints bound to model/loss/split identities.
- `v60_train_clean_signal.py` is dry-run by default, reports the shared split/model/loss matrix,
  refuses nonempty output roots, and requires `--confirm-run` before invoking the trainer.
- User-run evidence is now recorded: six CUDA cells completed on the within-family split. The best
  cell is `pyramid_cnn` + `v7_structural_v1` at final-height MAE `0.145868`; its same-architecture
  parity control is `0.150999`, and the tile-mean baseline is `0.181995`. This is a strong absolute
  baseline result, but the structural lift is below the 10% criterion, so it is not yet promoted.
- The user then completed the full-profile `pyramid_cnn` + `v7_structural_v1` complete-family run
  at `pyramid-full-structural-complete-v1`: best epoch 37, final-height MAE `0.173904` versus the
  `0.191047` tile-mean baseline (`8.97%` overall improvement), with 76 train and 32 held-out rows
  on CUDA. `cross_tile_burn` regressed `15.52%` and `cross_tile_lightning` regressed `229.79%`;
  the pathological bucket regressed `2.81%`. Under the explicit cross-tile acceptance scenario,
  the checkpoint remains diagnostic and is not promoted; real transfer stays blocked.
- CPU-focused contract proof passes: 36 new tests and 76 tests across `tests/v60`. Codex did not
  launch the user-owned CUDA run.
- The checkpoint is now consumable by the prediction-only diagnostic CLI. It reconstructs the model
  from checkpoint identity, selects the exact recorded held-out rows, writes per-row prediction and
  absolute-error NPZs, and renders full and cross-tile atlases. The user-run diagnostic is the next
  bounded gate; no model change or real transfer follows until its failure mode is understood.
- The diagnostic atlas identified a constant-field stability failure: `flat-v00` and
  `cross_tile_lightning-v01` have nearly identical four-channel inputs, while the zero-padding
  checkpoint emits the same non-flat ramp for both near-zero targets. The model contract now uses
  versioned `reflect-3x3-v1` padding for new checkpoints; legacy zero-padding identities remain
  reconstructable for comparison. One fresh full-profile confirmation run is required before any
  promotion decision changes.
- The reflect-padding confirmation completed at best epoch 80 with MAE `0.137891` versus the
  `0.191047` baseline (`27.82%` aggregate improvement). The flat-input ramp is fixed, but
  `cross_tile_lightning` remains `61.17%` below its baseline and `cross_tile_burn` remains `30.15%`
  below baseline. The next evidence is a full-profile within-family run with all 81 available
  training rows; real transfer remains blocked.
- Added a separate `real_terrain_synthetic` bridge corpus and image-only checkpoint evaluator.
  The first existing 16-row Alpha/Azeroth bridge scored MAE `0.323879` versus a `0.157124`
  tile-mean baseline (`-106.13%`) with the reflect-padding checkpoint. This is real-domain
  diagnostic evidence only: the bridge uses harvested terrain geometry and synthesized clean
  shading, while authored minimap RGB still requires a versioned albedo-normalization gate. The
  16 rows were an older diagnostic subset, not the intended corpus size.
- Added a Zarr-backed bridge builder for the complete v50.1 synthetic side. The verified dry run
  reports 1,330 rows (688 Kalimdor train, 642 Azeroth validation) with a complete-family,
  map-held-out split. Original source row indices and the source index hash are preserved in
  provenance; the source store is read-only and authored RGB remains excluded. The available
  v50.1 store is pre-Spec-133 and has raw `shadow_mask` rather than `terrain_shadow_256`. The
  builder therefore requires explicit `--input-signal shadow_mask` for a geometry-only raw-MCSH
  diagnostic and never silently aliases it to the deployment-clean signal.
- Added a minimap-observable raw-RGB diagnostic builder. It reads only `minimap_rgb`, derives
  luma/gradients, emits explicit absent confidence, and preserves the albedo gate as `not_run`.
  The actual v50.1 source contains 1,325 authored rows and 1,330 synthetic rows. This baseline is
  useful for measuring raw-pixel learnability, but it is not accepted albedo-normalized transfer.
- The user-run real-bridge training probe used 15 rows with one validation row. Its best epoch 4
  scored `0.313952` versus a `0.109902` validation baseline (`-185.66%`); all-16 evaluation of
  that checkpoint scored `0.293371` versus `0.157124` (`-86.71%`). Coarse error dominates detail
  error, so the next action is source-integrity auditing and multi-map expansion, not another run
  on the same 16 rows.

Model adapters and the synthetic corpus builder stay behind this contract gate.

## Problem Statement

The April v7 model was the closest previous attempt to useful terrain reconstruction, but its
contract was not deployable. It consumed a WDL height prior, height-derived min/max hints, normals,
liquid fields, and object masks. Several of those channels were unavailable at inference or were
derived from the answer being predicted. Its large model and old training corpus also mixed dirty
signals with useful supervision.

The current v60 control corpus is structurally valid, but the first one-channel architecture
bakeoff still failed the tile-mean baseline on held-out procedural families. That result does not
invalidate v7's multi-scale structural idea; it shows that architecture selection alone is not
enough. This feature tests the transferable v7 idea against a clean, reproducible observation and
exact synthetic height targets.

The deployment boundary is:

```text
arbitrary authored minimap
    -> versioned albedo normalization and textureless gate
    -> clean observation + image-derived confidence/gradient signals
    -> v7-inspired coarse structure + detail reconstruction
    -> relative height_257
```

No WDL prior, ground-truth height, ground-truth normals, liquid field, object mask, or other
target-derived signal may enter inference.

## Design Boundary

The first model input is a deployment-safe observation package:

- `clean_observation_luma_256`: one finite `[0,1]` channel representing the albedo-normalized
  terrain observation;
- `clean_observation_gradient_256`: two finite channels containing deterministic x/y gradients of
  that observation;
- `clean_observation_confidence_256`: one finite `[0,1]` channel emitted by the albedo operation,
  where low values identify pixels whose texture removal is uncertain.

The four channels are all computable from an arbitrary minimap before inference. The confidence
channel is allowed to be all zeros only when the operation explicitly records that confidence was
unavailable; it must never contain height or mask truth. Synthetic controls must produce the same
package and record the perturbation and confidence provenance.

The model predicts two training-visible components:

- `coarse_relief_257`: low-frequency relative relief;
- `detail_residual_257`: signed residual that completes the coarse field.

The exported product is `height_257 = coarse_relief_257 + detail_residual_257`, clamped only at the
published relative-height boundary. The two components are internal guidance surfaces, not
additional deployment inputs or separate terrain products.

## User Scenarios & Testing

### User Story 1 — Reproduce the v7 structural advantage without leakage (Priority: P1)

A researcher can train a v7-inspired terrain model whose input is limited to the clean observation
package and whose exact height target comes from project-owned synthetic terrain. The researcher can
compare a modern architecture against a small U-Net control using the same data, split, optimizer,
and final-height metric.

**Why this priority**: This isolates the part of v7 that appeared useful—multi-scale structure and
detail guidance—from the old WDL trestle and dirty signal contract.

**Independent Test**: Build a deterministic synthetic corpus, run the dry-run contract audit, and
verify that every model forward pass accepts only the four clean observation channels and emits the
two components plus a recomposed `height_257`.

**Acceptance Scenarios**:

1. **Given** a valid synthetic observation/height pair, **when** the model is initialized and run,
   **then** it produces finite coarse, detail, and recomposed 257×257 fields.
2. **Given** an inference input with WDL, height, normal, liquid, or object arrays attached,
   **when** the model is invoked, **then** those arrays are ignored or rejected and cannot alter the
   input tensor.
3. **Given** the same seed, corpus manifest, and architecture configuration, **when** the experiment
   is repeated, **then** the split, model identity, and synthesized observation hashes match.

### User Story 2 — Use synthetic terrain as guidance, not as a shortcut (Priority: P1)

A researcher can generate varied terrain observations with independent albedo-removal quality,
illumination, relief, cross-tile, fractal, island, flat, and sheer-dropoff controls. Exact height
targets and multi-scale structural targets are available for training, while observation-only input
features stay within the deployment contract.

**Why this priority**: The old v7 result may have come from its structural losses rather than its
unavailable auxiliary signals. Synthetic data lets that hypothesis be tested with exact evidence.

**Independent Test**: Generate the corpus twice from one configuration and compare row hashes,
observation hashes, target hashes, and provenance. Review visual panels for both observations and
coarse/detail targets.

**Acceptance Scenarios**:

1. **Given** a known height field and a synthesis configuration, **when** an observation is rendered,
   **then** the package carries the exact target, albedo/illumination parameters, confidence
   provenance, and a reproducible row hash.
2. **Given** a deliberately textured, partially normalized, or failed observation, **when** the
   gate is evaluated, **then** it is labeled rejected or quarantined rather than silently admitted.
3. **Given** a cross-tile pattern, **when** its four tiles are reviewed together, **then** the
   pattern remains continuous and is not restarted independently per tile.

### User Story 3 — Compare v7 guidance losses and modern architectures (Priority: P1)

A researcher can compare the current point/gradient loss against a v7-inspired structural loss
stack containing full-spectrum, Laplacian, edge, transition, tile-border, and multi-scale
low/high-frequency guidance. The same comparison can run with the pyramid CNN, SegFormer, and small
U-Net architecture candidates.

**Why this priority**: The architecture bakeoff showed that the pyramid CNN was only marginally
better than the U-Net and still worse than the trivial baseline. The next evidence must identify
whether v7's loss-side structural prior is the missing lever.

**Independent Test**: Run a dry-run ablation matrix and then a user-owned training run on a fixed
synthetic split. The report must expose final-height MAE, coarse MAE, detail MAE, frequency/edge/
curvature diagnostics, and per-family baseline-relative metrics for each cell.

**Acceptance Scenarios**:

1. **Given** identical model and split settings, **when** the structural stack is disabled or
   enabled, **then** the only changed training authority is the documented loss configuration.
2. **Given** a candidate architecture, **when** training completes, **then** the report identifies
   its parameter count, seed, input contract, loss weights, best epoch, and per-family metrics.
3. **Given** a model that improves aggregate MAE while degrading cross-tile or sheer-dropoff
   families, **when** the report is evaluated, **then** it is not promoted as a generalized winner.

### User Story 4 — Transfer to arbitrary albedo-normalized minimaps (Priority: P2)

A researcher can run the selected synthetic-trained model on a tiny accepted sample of real 0.x/1.x
minimaps after albedo normalization, without supplying WDL or any ground-truth terrain signal.

**Why this priority**: Synthetic success matters only if the input contract survives the transition to
real minimaps. This is the first deployability check for the v7-inspired lane.

**Independent Test**: The transfer command reads only accepted normalized observations and writes
height outputs, confidence/provenance, and visual validation artifacts. A separate audit proves
that target-side arrays were not read during inference.

**Acceptance Scenarios**:

1. **Given** an accepted normalized real minimap, **when** inference runs, **then** it produces a
   finite relative height field using only the four observation channels.
2. **Given** a rejected or quarantined albedo result, **when** transfer is attempted, **then** the
   row is refused and remains visible in the gate report.
3. **Given** a synthetic checkpoint that passes the control gate but fails the real transfer gate,
   **when** expansion is considered, **then** the report names albedo/domain shift as the next
   diagnosis rather than authorizing broad processing.

### Edge Cases

- Flat or near-flat terrain: the relative-height target keeps its range floor and reports ambiguity;
  the model must not invent high-relief structure merely to satisfy the structural losses.
- Sheer drop-offs and cross-tile motifs: edge and border metrics are reported separately; aggregate
  MAE cannot hide a family failure.
- Missing albedo confidence: the row is allowed only with an explicit absence flag and a zero-filled
  confidence channel; the model report must distinguish this from true high confidence.
- Non-finite, out-of-range, wrong-shape, or stale observation/target arrays: fail closed.
- Object-heavy real minimaps: object masks are not model inputs in this phase; the albedo gate or a
  later separately authorized object-sieve stage must determine whether the row is admissible.
- Any attempt to provide WDL, height, normal, liquid, or object arrays to inference: reject the
  invocation rather than silently using them.

## Requirements

### Functional Requirements

- **FR-001**: The feature MUST preserve v7's multi-scale coarse-plus-detail reconstruction idea
  without preserving its WDL trestle, 13-channel input order, or target-derived inputs.
- **FR-002**: Deployment inference MUST accept only the versioned four-channel clean observation
  package and MUST support any 256×256 minimap that passes the albedo/textureless gate.
- **FR-003**: The observation package MUST contain luma, x/y image gradients, and albedo-operation
  confidence, all finite and normalized under a versioned contract.
- **FR-004**: Synthetic generation MUST emit exact `height_257` supervision, deterministic
  coarse/detail target decomposition, independent observation perturbation parameters, and hashes.
- **FR-005**: Synthetic controls MUST include flat, smooth relief, mountainous, sheer-dropoff,
  fractal/ridged, lightning/burn, island/sea, chunk-grid, and cross-tile families, with whole-family
  and within-family split modes.
- **FR-006**: The model MUST emit coarse relief, signed detail residual, and recomposed relative
  `height_257`; every output MUST have independent validation metrics.
- **FR-007**: The loss system MUST provide a documented baseline and independently ablatable v7
  guidance terms for point error, gradient, full 2D frequency, Laplacian, Sobel edge, transition
  focus, tile border, and low/high-frequency bands.
- **FR-008**: The initial architecture registry MUST support `pyramid_cnn`, `segformer_b0`, and
  `unet_lite_v2` under one model/output contract. DPT is not required for the first lane because
  the prior control run was flat from epoch one.
- **FR-009**: Architecture and loss comparisons MUST use the same seeded corpus, split, training
  budget, and tile-mean baseline; reports MUST include per-family and per-complexity metrics.
- **FR-010**: The model MUST NOT read WDL, height, normal, liquid, object, alpha, or any other
  target-derived signal during inference, even if those arrays are present beside an input row.
- **FR-011**: Albedo normalization MUST write a versioned observation, confidence/absence status,
  residual quality metrics, and accepted/rejected/quarantined decision before inference admission.
- **FR-012**: The transfer evaluator MUST prove image-only inference and keep synthetic control,
  real transfer, and visual acceptance metrics separate.
- **FR-013**: Checkpoints and reports MUST bind architecture, loss configuration, input contract,
  corpus manifest, split, seed, and upstream albedo-operation identity.
- **FR-014**: Heavy training, synthetic corpus generation, real-client processing, and transfer runs
  MUST remain user-launched behind dry-run and fail-closed commands.

### Key Entities

- **CleanObservationRow**: One albedo-normalized observation package, confidence state, synthesis or
  real provenance, split membership, and content hashes.
- **StructuralTarget**: Exact height target plus deterministic coarse relief and signed detail
  residual derived from the target for training and validation only.
- **V7GuidanceConfig**: Versioned point, gradient, frequency, Laplacian, edge, transition, border,
  and low/high-frequency loss weights.
- **ArchitectureRun**: One model/loss/split run with parameter count, checkpoint identity, best epoch,
  final-height metrics, component metrics, and per-family results.
- **TransferDecision**: Synthetic gate result, accepted real-row count, domain comparison, visual
  artifacts, and hold/diagnose/expand decision.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Rebuilding the same synthetic configuration twice produces identical observation,
  target, manifest, and split hashes for 100% of rows.
- **SC-002**: A clean-observation model forward pass uses exactly four input channels and produces
  finite 257×257 coarse, detail, and recomposed fields; an inference audit finds zero reads of
  forbidden target-derived arrays.
- **SC-003**: On the within-family learnability split, the best structural-guidance run improves
  final-height MAE by at least 10% over the same architecture with only point and gradient losses.
- **SC-004**: On the held-out-family synthetic split, the promoted run beats the tile-mean baseline
  by at least 5% overall and does not regress any complexity bucket by more than 5% relative to that
  bucket's baseline.
- **SC-005**: The selected checkpoint processes 100% of accepted transfer rows without shape or
  finite-value failures and writes visual review artifacts for every row.
- **SC-006**: The transfer report records an explicit `hold`, `diagnose`, or `expand` decision and
  never treats synthetic success alone as authorization for broad real-data processing.

## Assumptions

- The existing C# terrain compositor remains the authority for synthetic terrain observations and
  exact heights; Python only validates, derives training-only targets, and orchestrates runs.
- Albedo normalization is a separate versioned operation owned by the v60 real-input lane. This
  feature consumes its accepted artifact and does not silently substitute a synthetic image.
- The first implementation uses the existing v60 architecture registry as the starting point and
  wraps the selected encoder with v7-inspired structural heads; it does not revive the 117M v7
  monolith.
- Real transfer initially remains limited to explicit 0.x/1.x rows. Later client eras are out of
  scope until this control and transfer gate passes.
- Object masks may later become loss-side evidence, but object identification, object prediction,
  and object-guided inference are out of scope for this first clean-signal lane.
