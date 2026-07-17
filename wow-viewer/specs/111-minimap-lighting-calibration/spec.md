# Feature Specification: Minimap Lighting Calibration and Lighting-Aware Terrain Reconstruction

**Feature Branch**: `111-minimap-lighting-calibration`

**Created**: 2026-07-17

**Status**: Draft

**Input**: User description: "now that we have the ability to properly render the minimaps, we can synthesize higher resolution maps, lower resolution maps, and thus, have good control data that renders properly - and we KNOW how the sun's rays hit the terrain, it's 12:00 on their sun timer, we should be able to build a smart model that iterates over all the dataset minimaps and determines their time of day, then buckets them so we build a smarter model that can handle ANY minimap image input to build a 3d terrain output. use speckit. use what we built to drive home the model!"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Determine each authored minimap's lighting condition (Priority: P1)

A researcher preparing the terrain-reconstruction training set can find out, for every 0.5.3.3368
authored minimap that also has decoded ground-truth terrain, which time-of-day/sun-bearing best
explains its shading pattern -- not just its color tint, which the existing lighting-provenance
signal already covers, but the actual direction the terrain shadows fall. Every tile gets either a
confident bucket label or an explicit "not evaluated" / "low confidence" status; nothing is silently
guessed.

**Why this priority**: Everything else in this feature (rebalanced training data, a retrained model)
depends on first knowing the true lighting distribution of the real dataset. Without this, any
downstream rebalancing is built on an unverified assumption.

**Independent Test**: Run the shading-match inference over the existing 0.5.3.3368 dataset store(s)
and confirm every eligible tile receives a bucket label or an explicit non-evaluated status, with a
per-map/per-build distribution report a human can read and sanity-check against a few tiles by eye.

**Acceptance Scenarios**:

1. **Given** a 0.5.3.3368 dataset tile with both an authored minimap image and decoded ground-truth
   terrain (normals, texture layers) sufficient to render a synthesized minimap, **When** the
   shading-match inference runs, **Then** it renders synthesized candidates across a time-of-day
   sweep using the production `TerrainMinimapCompositor`/`TerrainSolarDirection` path (the same
   corrected, ground-truth-validated code the live viewer and exporter use, not a reimplementation),
   scores each candidate's shading pattern against the authored minimap, and records the best-fit
   time-of-day, a confidence score, and the evidence basis.
2. **Given** a tile whose terrain is too flat/low-relief to carry a usable shading signal, or whose
   candidates fit ambiguously (multiple times-of-day score near-identically), **When** the inference
   runs, **Then** it reports an explicit low-confidence or not-evaluated status rather than picking an
   arbitrary best match.
3. **Given** a tile already flagged by the existing tint-based provenance as likely carrying a baked
   MCSH shadow, **When** the shading-match inference runs, **Then** it accounts for that overlap
   rather than double-counting baked static shadow as ordinary directional hillshade.
4. **Given** a tile from any build other than 0.5.3.3368, **When** the dataset is iterated, **Then**
   the tile is left with its existing status untouched -- this inference is explicitly scoped to
   0.5.3.3368, the only build with a ground-truth-traced sun model.
5. **Given** the inference has run across a build's dataset, **When** a human requests the result,
   **Then** a per-map and overall lighting-bucket distribution is available in a form that shows how
   many tiles landed in each bucket and how many are not-evaluated/low-confidence.

---

### User Story 2 - Make synthetic training lighting match the real distribution (Priority: P2)

A researcher training the image-to-terrain reconstruction model wants the synthetic lighting variants
generated for training to reflect how 0.5.3.3368 minimaps are actually lit in the real dataset,
instead of an arbitrary or uniform sweep across time-of-day.

**Why this priority**: This is the payoff of User Story 1 -- it only has value once real tiles are
reliably bucketed. It directly improves training-data realism without touching model architecture.

**Independent Test**: Compare the resulting synthetic-variant time-of-day sampling distribution
against the real bucket distribution from User Story 1 and confirm they match within an agreed
tolerance, while every existing split-leak safeguard (variant tagging) still holds.

**Acceptance Scenarios**:

1. **Given** the real lighting-bucket distribution from User Story 1, **When** synthetic lighting
   variants are generated for training, **Then** their time-of-day sampling is reweighted to match
   that observed distribution within a defined tolerance, rather than sampling uniformly/arbitrarily.
2. **Given** a real bucket that has very few or zero real examples, **When** variants are
   reweighted, **Then** the system does not fabricate a real-example count it doesn't have -- it may
   still generate synthetic coverage there, but must not claim that coverage represents an observed
   real distribution.
3. **Given** the existing split-leak safeguard (`source_group_id`/`lighting_variant_id` tagging from
   the synthetic-lighting-variant system), **When** sampling is reweighted, **Then** the safeguard
   still prevents variants of the same source tile from crossing train/eval splits.
4. **Given** the rebalanced training data, **When** it is inspected for its input contract, **Then**
   ground-truth time-of-day or lighting direction is never present as a feature the model consumes --
   the lighting-bucket label is a sampling/curation signal only, matching the existing constraint that
   the deployed model must handle an input minimap of genuinely unknown lighting.

---

### User Story 3 - Retrain and evaluate the reconstruction model on rebalanced data (Priority: P3)

A researcher wants to know whether training on the lighting-rebalanced dataset actually improves the
terrain-reconstruction model's ability to handle an arbitrary real-world minimap, compared to the
currently deployed checkpoint.

**Why this priority**: This closes the loop and is the ultimate point of the feature, but it depends
entirely on User Stories 1 and 2 producing trustworthy inputs first, and it is the most
resource-intensive and highest-risk step (a real GPU training run).

**Independent Test**: Train a checkpoint on the rebalanced dataset, evaluate it against the current
deployed checkpoint on a held-out set, and produce an explicit go/no-go comparison a human reviews
before any checkpoint is promoted.

**Acceptance Scenarios**:

1. **Given** the rebalanced training dataset and config from User Story 2, **When** a training run is
   explicitly authorized and executed, **Then** it targets the existing reconstruction architecture
   (the current Spec 108 `WdlPriorNet` or whichever Spec 102 residual-chain stage is active) rather
   than introducing a new competing architecture, and produces a checkpoint plus training record.
2. **Given** a newly trained checkpoint, **When** it is evaluated, **Then** its held-out accuracy is
   compared against the current deployed checkpoint on the same evaluation set, and the comparison
   result (improved / regressed / inconclusive) is recorded before any promotion decision.
3. **Given** a regression in the comparison, **When** the go/no-go decision is made, **Then** the
   existing checkpoint remains deployed -- a regression must not be silently accepted.
4. **Given** any point where a GPU training run or cloud compute job (e.g. RunPod) would actually
   execute, **When** that point is reached, **Then** the system stops and requires an explicit,
   separate user go-ahead before launching it -- producing the plan, code, or rebalanced dataset does
   not itself authorize spending compute or money.

---

### Edge Cases

- A tile has an authored minimap but no decoded ground-truth terrain (or vice versa) -- excluded from
  shading-match inference with an explicit reason, not silently skipped.
- Terrain is nearly flat (desert, plains) and carries too little shading signal to distinguish
  candidate times-of-day -- must report low confidence, not a false-precise bucket.
- Two or more candidate times-of-day fit the real minimap's shading almost equally well (symmetric
  terrain, weak signal) -- must report the ambiguity, not arbitrarily pick one.
- A tile's authored minimap already carries baked MCSH shadow (per the existing tint+MCSH
  correlation) -- the shading-match signal must not treat that baked shadow as if it were the
  ordinary dynamic hillshade being matched.
- The real observed lighting distribution is heavily skewed (e.g., mostly midday captures) --
  rebalancing must not fabricate real-example density it doesn't have in sparsely covered buckets.
- A build other than 0.5.3.3368 appears in the same dataset store -- must remain out of scope and
  visibly untouched, not silently absorbed into the 0.5.3.3368 buckets.
- The retrained checkpoint regresses on the held-out set -- the go/no-go outcome must default to
  keeping the current checkpoint.
- A training run would require launching a RunPod (or other cloud) GPU job -- must halt for explicit
  user authorization at that exact point, consistent with prior guidance never to auto-launch billed
  compute.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST identify every 0.5.3.3368 dataset tile that has both an authored
  minimap image and decoded ground-truth terrain data sufficient to render a synthesized minimap for
  it.
- **FR-002**: For each identified tile, the system MUST render shading-match candidates across a
  time-of-day sweep using the existing, ground-truth-corrected `TerrainMinimapCompositor` /
  `TerrainSolarDirection` production code path -- not a separate reimplementation of the lighting
  model -- so the calibration signal cannot drift from what the viewer and exporter actually produce.
- **FR-003**: The shading-match score MUST be sensitive to the geometric/directional shading pattern
  (where terrain shadows fall) independent of the existing color-tint signal, so it provides new
  information rather than re-deriving the existing tint-based inference.
- **FR-004**: The system MUST record the shading-match result (best-fit time-of-day, confidence,
  evidence basis, and an explicit not-evaluated/low-confidence status when appropriate) as an
  additive extension of the existing `MinimapLightingProvenance` contract, preserving its "inference,
  not capture-proof" discipline.
- **FR-005**: The system MUST iterate this inference across the entire 0.5.3.3368 portion of the
  existing per-build dataset store(s) and produce a per-map and overall lighting-bucket distribution
  report, tagged with client build identity for traceability.
- **FR-006**: The system MUST NOT apply this shading-match inference to any build other than
  0.5.3.3368; tiles from other builds MUST remain explicitly untouched/not-evaluated by this feature.
- **FR-007**: The system MUST use the resulting real lighting-bucket distribution to reweight the
  existing synthetic-lighting-variant generation (from the image-only-reconstruction training
  pipeline) so its time-of-day sampling matches the observed real distribution within a defined
  tolerance, replacing the current arbitrary/uniform sweep.
- **FR-008**: Reweighting synthetic-variant sampling MUST preserve the existing split-leak safeguard
  (source/variant tagging) so that resampling cannot introduce train/eval leakage.
- **FR-009**: The rebalanced training data and any resulting model input contract MUST continue to
  exclude ground-truth time-of-day or lighting direction as a model input feature; the lighting-bucket
  label is a curation/sampling signal only.
- **FR-010**: Any new or modified model training MUST target the existing terrain-reconstruction
  architecture lineage (Spec 108 `WdlPriorNet` or the currently active Spec 102 residual-chain stage)
  rather than introducing a new competing architecture, and MUST NOT introduce any DepthAnything-family,
  multi-head, multi-task, or shared-weight model path.
- **FR-011**: A retrain-and-evaluate pass MUST compare the resulting checkpoint against the current
  deployed checkpoint on a held-out evaluation set and record an explicit improved/regressed/
  inconclusive outcome before any checkpoint promotion.
- **FR-012**: A regression in that comparison MUST result in keeping the currently deployed checkpoint
  -- no automatic promotion of a regressed checkpoint.
- **FR-013**: The system MUST require an explicit, separate user go-ahead immediately before launching
  any GPU training run or cloud compute job; preparing the spec, plan, code, or rebalanced dataset MUST
  NOT itself trigger that execution.

### Key Entities *(include if feature involves data)*

- **Minimap shading-match candidate**: A synthesized minimap rendered at one candidate time-of-day
  for a specific tile, used only to score against the real authored minimap's shading pattern.
- **Extended minimap lighting provenance**: The existing per-tile lighting-provenance record, gaining
  shading-matched time-of-day, confidence, and evidence fields alongside its existing tint-based
  fields.
- **Lighting-bucket distribution report**: A per-map and overall summary of how many 0.5.3.3368 tiles
  fall into each inferred lighting bucket, and how many are not-evaluated/low-confidence.
- **Rebalanced training sampling plan**: The time-of-day sampling weights applied to synthetic
  lighting-variant generation so it matches the real observed distribution.
- **Reconstruction checkpoint comparison**: A record comparing a newly trained checkpoint against the
  currently deployed one on a held-out set, with an explicit go/no-go outcome.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of 0.5.3.3368 dataset tiles that have both an authored minimap and ground-truth
  terrain receive either a shading-based lighting-bucket label or an explicit not-evaluated/
  low-confidence status -- zero tiles are silently left unlabeled.
- **SC-002**: The synthetic-lighting-variant training sample's time-of-day distribution matches the
  observed real 0.5.3.3368 distribution within an agreed tolerance, replacing the current arbitrary
  sweep.
- **SC-003**: A retrained checkpoint's held-out evaluation result is always compared against the
  current deployed checkpoint, with the comparison outcome recorded before any promotion decision;
  no checkpoint swap occurs without a passing comparison.
- **SC-004**: An explicit check confirms the deployed/evaluated model's input contract never includes
  ground-truth time-of-day or lighting-direction fields.
- **SC-005**: Zero DepthAnything-family or other forbidden architectures appear anywhere in the new
  training/evaluation code path.
- **SC-006**: No GPU training run or cloud compute job is launched without a separate, explicit user
  go-ahead recorded at the point of execution.

## Assumptions

- Scope is limited to 0.5.3.3368-sourced tiles, the only build with a ground-truth-traced sun model;
  other builds remain explicitly not-evaluated by this feature rather than silently included.
- The new shading-match fields extend the existing `MinimapLightingProvenance` record (additive
  fields) rather than becoming a second, separate parallel record, since both signals describe the
  same tile/minimap subject and existing consumers already expect one provenance record per tile.
- The retrain targets the existing Spec 108 `WdlPriorNet` (or whichever Spec 102 residual-chain stage
  is currently active) rather than a new architecture, per established project philosophy of
  preferring small/fast models and extending proven lineages before introducing something heavier.
- Canonical training-sample storage remains the existing per-build Zarr stores (Spec 109 convention);
  this feature adds fields and derived reports, not a new storage format.
- The existing synthetic-lighting-variant split-leak tagging (Spec 103) is preserved unchanged when
  its sampling is reweighted.

## Out of Scope

- Client builds other than 0.5.3.3368 (explicitly deferred; may be revisited once an equivalent
  ground-truth lighting proof exists for another build).
- Any new model architecture beyond the existing `WdlPriorNet`/residual-chain lineage, unless a later
  pass demonstrates the existing lineage is insufficient and justifies the change on its own.
- Automatic or unattended launch of the actual training run or any cloud GPU pod -- always a separate,
  explicitly authorized step, never a side effect of completing this feature's code or planning.
- Reproducing the native client's exact ghidra-traced lighting ray as literal minimap input data --
  this feature uses only the verified axis/azimuth proof already adopted into
  `TerrainSolarDirection`, matching the discipline already established in Spec 110's decision log
  (the raw traced vector itself remains separate diagnostic research, not a minimap/training input).
