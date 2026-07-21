# Feature Specification: WDL-Lattice Coarse Prior for Terrain Geometry

**Feature Branch**: `117-wdl-lattice-prior`

**Created**: 2026-07-21

**Status**: Draft

**Input**: User description: "v50 WDL-lattice coarse prior for terrain geometry reconstruction — a per-tile coarse structural signal, one level coarser than the existing coarse geometry stage, predicted from minimap RGB alone and fed into the existing v50 coarse+detailer chain."

## Motivation

The v50 coarse+detailer geometry chain is the first approach in this project's history to beat the
trivial tile-mean baseline on relief-bearing regions under honest, spatially-isolated evaluation
(56.1% relative reduction in relief-region error, measured against a checkpoint neither stage was
tuned against). That chain currently sees two generated inputs: the minimap RGB itself, and — as of
this session — a generated per-chunk structure classification. This feature adds a third: a
coarser, per-tile height *lattice* one step above the chain's own "coarse" stage, on the theory that
a cheap skeleton of a tile's overall shape can steer the fine stages the way a low-resolution guide
image steers a detail pass.

Three things ground this in fact rather than hope:

1. **The lattice's exact sampling contract is already settled, not new design work.** Spec 108
   (`108-image-wdl-prior`) FR-001 already defines it precisely: outer samples at
   `height_257[::16,::16]` (17×17) and inner samples at `height_257[8::16,8::16]` (16×16), 545
   points total, never a stride-8 raster. A working C# reference implementation already exists
   (`TerrainWdlLattice.FromTerrainVertices`) and samples exactly this lattice from the same real
   MCVT vertex data `height_257` is itself derived from — it is wired into the tensor-pack pipeline
   (`TerrainTileTensorPack.WdlLattice`) but not yet exported as a v50 store signal.
2. **This is a per-tile signal, not a per-map one.** Every tile in the corpus already carries the
   real height data the lattice is sampled from, so the sample count matches the rest of the corpus
   (thousands of tiles) — not the handful of real maps this project's client build actually has.
3. **The one prior attempt at this idea failed on different, superseded infrastructure, which is
   evidence against that infrastructure, not against the idea.** Spec 108's own implementation
   (`harvester.v24.merged_wdl_prior`, a real+synthetic WDL merge through a separate legacy reader)
   produced `wdl_prior_strict_v1`: best epoch 1, 15 stale epochs after — the exact
   "structural-failure-epoch-1" pattern this project's own current trainers detect and flag as a
   non-success. It never used the v50 architecture, the current loss/training conventions, or a
   spatially-isolated held-out split. Nothing here re-attempts that pipeline.

A hard boundary carries over from the conversation that produced this spec: no GAN, no adversarial
loss, no generative-image technique of any kind. This project's established direction is exact
ground-truth supervision — the same reasoning that made `normal_guidance.py` "the non-adversarial
answer to PatchGAN for detail" applies here without exception.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Export the lattice as a real signal (Priority: P1)

Before any model exists, the 545-point WDL lattice becomes a first-class, readable signal in the
v50 store for every tile that has real height ground truth, derived deterministically from existing
data with no new client harvest.

**Why this priority**: Nothing else in this feature is buildable without the signal existing where
Python training code can read it. This is data plumbing, not modeling, and is fully separable from
whether the eventual model works.

**Independent Test**: Export the signal against the existing corrected v50 curriculum store and
confirm every tile with real height data produces exactly 545 finite lattice samples, with any tile
that cannot be excluded and counted rather than silently zero-filled.

**Acceptance Scenarios**:

1. **Given** a tile with real `height_257` ground truth, **When** the lattice is exported, **Then**
   its 17×17 outer and 16×16 inner samples match `height_257[::16,::16]` and
   `height_257[8::16,8::16]` exactly.
2. **Given** a tile whose real height data cannot support the full lattice, **When** export runs,
   **Then** that tile is excluded and counted, never fabricated or zero-filled.

---

### User Story 2 - Prove the lattice is learnable from a minimap alone before integrating anything (Priority: P1)

A standalone model predicts the 545-point lattice from minimap RGB alone, trained and evaluated
exclusively against the spatially-isolated held-out split, so the signal's learnability is known
before any integration work is attempted.

**Why this priority**: Spec 116 proved this pattern's value directly: cheap, no-integration-risk
measurements before committing to the expensive step. Betting on chain integration before knowing
whether the lattice is predictable at all would repeat the mistake this project has already paid
for once.

**Independent Test**: Train the standalone predictor, score lattice-point MAE against a trivial
per-tile-mean lattice baseline on the honest held-out split, and read a plain learnable/not-learnable
verdict before any chain-integration code is written.

**Acceptance Scenarios**:

1. **Given** only a minimap RGB tile, **When** the lattice is predicted, **Then** no client-derived
   ground truth is read at prediction time.
2. **Given** the held-out split, **When** the predictor is scored, **Then** lattice-point MAE is
   reported against the trivial per-tile-mean lattice baseline on that exact split.
3. **Given** the predictor underperforms the trivial baseline, **When** the result is reported,
   **Then** integration work (User Story 3) does not proceed until this is explicitly overridden.

---

### User Story 3 - Feed the generated lattice into the existing chain and measure relief-region error (Priority: P2)

The generated (never ground-truth) lattice prior is supplied to the existing coarse stage, the
existing detailer stage, or both, and relief-region error is compared against the already-real
established baseline — the current structure-augmented detailer result — on the identical held-out
split, settling empirically where the prior belongs rather than by design preference.

**Why this priority**: This is the payoff, but it is only cheap and safe to attempt once User
Stories 1 and 2 have proven the signal exists and is learnable. It depends on both.

**Independent Test**: Train paired coarse/detailer runs with and without the generated lattice prior
(feeding the coarse stage, the detailer stage, and both, as separate conditions of the same
experiment) on the identical held-out split, and compare relief-region MAE across all conditions
against the existing real baseline.

**Acceptance Scenarios**:

1. **Given** the generated lattice prior, **When** the coarse or detailer stage consumes it,
   **Then** it consumes the *predicted* lattice only, never the ground-truth lattice.
2. **Given** paired runs (with/without, per feed point), **When** they are compared, **Then**
   relief-region MAE is reported for every condition against the identical held-out split and the
   pre-existing structure-augmented baseline.
3. **Given** the comparison results, **When** the report is written, **Then** it states plainly
   which feed point (coarse, detailer, both, or neither) measurably helped, rather than presenting
   the intended design as the conclusion.

---

### Edge Cases

- A tile whose real height data has gaps exactly at a lattice sample coordinate — marked absent for
  that sample, never interpolated or fabricated to fill the point.
- A held-out tile with no exportable lattice at all — excluded from evaluation and counted, not
  scored as a zero-error or maximum-error case.
- The generated lattice store exists but does not cover every row a given coarse/detailer run
  selects — the run must refuse to proceed on the uncovered rows rather than silently skip or
  zero-fill them, matching how the existing feature-store contract already behaves.
- A result where the lattice prior helps one feed point but hurts the other — both are reported
  honestly; the feature does not require a uniformly positive result to be considered complete.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST export the WDL lattice as a first-class v50 store signal using the exact
  sampling contract already defined in Spec 108 FR-001 (17×17 outer at stride 16, 16×16 inner at
  stride 16 offset 8, 545 samples).
- **FR-002**: System MUST derive the lattice only from real MCVT-backed height data already present
  in the corpus. It MUST NOT read from or depend on the legacy V18/V24 WDL-file-reader
  infrastructure (`harvester.v24.merged_wdl_prior`/`wdl_reader`).
- **FR-003**: The standalone lattice predictor MUST predict from minimap RGB alone at inference
  time, with no client-derived ground truth read at prediction time.
- **FR-004**: The standalone lattice predictor MUST be trained and evaluated exclusively against a
  spatially-isolated held-out split (the existing Spec 116 split or an equivalent construction). It
  MUST refuse to run against a leaky or unspecified split.
- **FR-005**: System MUST NOT contain a GAN, adversarial loss, discriminator, or any other
  generative-image technique anywhere in this feature.
- **FR-006**: The chain-integration step MUST supply only the generated (never ground-truth)
  lattice prior to the coarse and/or detailer stage, as an additional input channel, mirroring the
  existing generated-feature-store input contract already used for the structure classifier.
- **FR-007**: Each stage (lattice predictor, coarse, detailer) MUST remain an independently trained,
  independently checkpointed, independently promotable model. No shared weights or multi-task heads
  across stages.
- **FR-008**: Every training run MUST validate and print its plan without training by default, and
  MUST require explicit confirmation before consuming compute.
- **FR-009**: System MUST report relief-region error with and without the lattice prior, per feed
  point, on the identical held-out split, so any change is attributable to the prior specifically.
- **FR-010**: Every reported result MUST record which held-out split and which upstream checkpoints
  (lattice predictor, coarse, detailer) produced it.
- **FR-011**: All training and any heavy rebuild MUST be executed by the user from documented
  commands with time/memory estimates. The assistant MUST NOT launch them.

### Key Entities

- **WDL Lattice**: The 545-sample (17×17 outer + 16×16 inner) coarse height sampling of one tile,
  deterministically derived from real MCVT vertex data — the same source `height_257` is derived
  from, at a much coarser sampling.
- **Lattice Predictor**: An independently trained, independently promotable model mapping minimap
  RGB alone to a generated WDL lattice, evaluated only against the spatially-isolated held-out
  split.
- **Generated Lattice Store**: The derived, checkpoint-bound store of predicted (never
  ground-truth) lattices, mirroring the existing Spec 115/116 generated-feature-store pattern —
  row-aligned, source stores immutable, bound to a checkpoint hash.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The WDL lattice is exported for every corpus tile carrying real height ground truth,
  with zero silent gaps — any tile that cannot produce a lattice is excluded and the exclusion is
  counted.
- **SC-002**: A standalone lattice predictor is trained on minimap RGB alone, evaluated on the
  spatially-isolated held-out split, and its lattice-point MAE is reported against the trivial
  per-tile-mean lattice baseline before any integration work proceeds.
- **SC-003**: When integrated, relief-region MAE with the lattice prior is compared against the
  already-established real baseline (the current structure-augmented detailer result) on the
  identical held-out split, and the report states plainly whether it helps, hurts, or is neutral.
- **SC-004**: The decision of where the prior feeds in (coarse input, detailer input, or both) is
  settled by measured relief-region error on held-out data, not by design preference stated in
  advance.
- **SC-005**: No GAN, adversarial loss, or generative-image component exists anywhere in the
  delivered code, verified by inspection.
- **SC-006**: Every reported result names the held-out split and upstream checkpoints that produced
  it, so no two incomparable numbers can be presented as a comparison.

## Assumptions

- The corrected v50 dual curriculum store (with the synthetic-lighting fix applied this session) is
  the input corpus; no new client harvest is required beyond exporting the new lattice signal from
  data already read during harvest.
- Real MCVT vertex data already backing `height_257` is sufficient to derive the lattice; no
  separate WDL client-file decode is needed or used.
- The existing coarse+detailer chain is the integration target. This feature does not propose a new
  top-level architecture; it proposes one more generated input to an already-validated chain.
- Model scale for the lattice predictor stays in the same small/lean capacity class as the existing
  v50 stages (on the order of ~1.5M parameters), consistent with this project's time-to-signal
  preference over large-model validation.
- This feature may show no integration benefit even if the standalone predictor (User Story 2)
  proves learnable. User Stories 1 and 2 are designed to be independently valuable and cheap to
  abandon after, exactly as Spec 116's early stories were.
- Vision-language, agentic image interpretation, and any generative/adversarial modeling technique
  are out of scope, per explicit user instruction and prior project precedent.
