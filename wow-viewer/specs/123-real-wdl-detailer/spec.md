# Feature Specification: Ground-Up v50 Terrain Height Model — Real WDL Prior + Residual Detailer

**Feature Branch**: `123-real-wdl-detailer`

**Created**: 2026-07-30

**Status**: Draft

**Input**: User description: "build a new model entirely from the ground up using our clean v50
dataset. Don't repeat the mistakes of yesterday, and move on, upwards and forwards. We need to
think about this problem with a fresh set of eyes. The dataset is phase 1, and that's now
completed." Elaborated in session: the model must use the real, deployment-available per-map WDL
client file as its coarse height prior (not a synthetic signal derived from the same height data
it is trying to predict), train exactly one residual detailer on top of it, and have its training
data selection driven by the new Spec 122 curation manifest.

## Problem Statement

This project has attempted a minimap-to-terrain-height model many times (Spec 094 Stage A, Spec
117, Spec 121 — three distinct architectures), and every attempt that tried to *predict* a coarse
WDL-scale height lattice from minimap RGB alone either failed outright or transferred so poorly
across map regions that it was not usable. The most recent attempt (Spec 121) even reached a
tentative positive verdict on a narrowed "within-map completion" framing, only for the
detailer built on top of it to show just a 5% improvement — read at the time as "the prior was
already at the noise floor," i.e. it carried no real information beyond what the detailer could
already infer from the minimap directly.

This session's investigation found the likely root cause, verified against the current source, not
assumed: every one of those attempts trained against `wdl_outer_17`/`wdl_inner_16`, the v50 signal
computed by `TerrainWdlLattice.FromTerrainVertices` — a downsample of the *same per-tile ADT height
array* the model is ultimately trying to predict. That signal cannot exist at deployment time for a
genuinely unseen tile, because producing it requires the exact ground-truth height data the model
does not have. Trying to predict a synthetic, self-referential, training-only signal from an RGB
image that does not strongly encode elevation was set up to fail from the start — independent of
which architecture, backbone, or training reframe was tried.

A different, real signal already exists in this project and was designed for exactly this purpose,
but was never wired into the v50 pipeline: the per-map `.wdl` client file. It is a low-resolution
terrain asset the game client ships for the *entire map*, independent of which specific tiles have
been separately authored or harvested — meaning it is genuinely available at deployment time for
any real, already-mapped WoW region, the same way an authored minimap is. A working reader for it
already exists in this codebase and is exercised by a dedicated tool and a dormant data-harvester
shim; it has simply never been connected to the v50 signal catalog or any v50 model.

**A real WDL file is not available for every input this project ultimately cares about.** This
project's stated deployment target has never been limited to "re-derive height for a region we
already fully have real data for" (a task with no practical value, since the real height would
already be readable directly) — it is decompiling/reconstructing terrain from a minimap image
alone, including for content that has no surviving `.wdl`: hand-painted or concept minimaps, and
recovering cut/removed alpha-era content where a minimap may be the only asset left. For that class
of input there is no real WDL to read, and there never will be — a minimap image alone does not
contain enough information to derive one. This is not the same claim Specs 117/121 tested and
found false (that a *dedicated model stage* can predict a WDL-scale lattice from RGB well enough to
transfer across regions); it is the more basic fact that no amount of image analysis reconstructs a
missing low-resolution height *file*. The two prior specs' negative result is why this design does
not add a second "predict WDL" stage to compensate — instead, the single detailer model is trained
to degrade gracefully when the real prior is absent, exactly the mechanism this project's own `V7`
lineage already validated (`terrain_refiner_train.py`'s `--wdl-prior-dropout`, detailed below) —
so "we cannot build a WDL from a minimap tile" is answered by not needing to, not by trying harder.

**Prior-absent mode still needs to not be confused by objects, and this project's only working tool
for that is coarse semantic classification, not object detection.** A minimap's raw pixel color
mixes true terrain color with roofs, roads, and other object color the terrain does not actually
have — Spec 115 measured this directly (roads misread as slopes, a color-as-depth-proxy confound)
and fixed it by feeding a *generated semantic feature map* (broad classes — road, natural terrain,
etc. — from a proven, real, from-scratch classifier) as an extra input channel, cutting road-region
height error 21%. That is a fundamentally different, easier task than Specs 119/120's object
*identification* (which blob is which specific building) — it does not ask "what is this," only
"what broad family of surface is this," and it is already proven, not aspirational. This spec's
prior-absent mode reuses that same proven classifier as an input, not the dead instance-retrieval
line; see FR-018. Object-heavy prior-absent tiles are still expected to score worse than open
terrain — this spec does not promise to close that gap, only to avoid pretending it is not there.

Separately, the newly-completed Spec 122 curation pass gives this model something no prior attempt
had: a durable, per-tile record of which training data is trustworthy and how, including a
newly-quantified finding that a large share of this project's synthetic (non-authored) minimap
renders have a real shading mismatch against the authored client art. Building a new model without
using that record to shape training data selection would waste the entire preceding phase of work.

## Governing Principle

The coarse height prior fed to this model is **real, deployment-available data read directly from
the game client whenever it exists** — never a value predicted from an image, and never a value
derived from the exact ground truth the model is trying to reconstruct. When no real prior exists
for a given input (including genuinely novel, non-client minimaps with no corresponding `.wdl` at
all), the same single model still produces a height prediction by falling back to whatever signal
the minimap alone provides — a graceful-degradation *mode* of one model, trained via prior dropout,
never a second model whose job is reconstructing the missing prior. Exactly one model is trained: a
residual detailer that refines the prior (or its absence) into full-resolution height, guided by
the minimap image. Training data selection is driven by the Spec 122 curation manifest, not by ad
hoc, one-off filtering logic invented again for this spec.

## Relationship To Existing Specs

- **Supersedes / closes**: Spec 117 (RGB→WDL-lattice-from-scratch, plateaued above tile-mean) and
  Spec 121 (three RGB→WDL architectures; within-map completion reached a tentative positive
  verdict but the resulting detailer showed only marginal improvement). Both are recorded, decisive
  negative evidence for *predicting* a coarse prior from RGB alone at this signal's resolution —
  this spec does not retry that task and any future session should read those specs' research.md
  before proposing to.
- **Completes**: Spec 094's Stage 0 design (real WDL read + deterministic real/synthetic merge +
  quincunx upsample), which was correctly designed in 2026-07-06 but never connected to the v50
  pipeline. This spec is the connection.
- **Depends on**: Spec 122 (canonical curation manifest — bucket/finding selection for training
  data) and Spec 116 (spatially-isolated held-out split — the only split machinery proven to avoid
  the measured 99.6% train/val adjacency leak).
- **Reuses as precedent, re-justified not inherited blindly**: Spec 114's residual-detailer
  architecture family (SegFormer-B0 trunk, zero-init residual head, ~3.7M params, 9-11% MAE
  improvement over coarse-only — the one piece of this project's terrain-model history with a
  clear, real, positive, externally-corroborated result) for the refinement mechanics, and the
  `v50-model-stage-run-v1` run record contract. **Also reuses `terrain_refiner_train.py`'s
  `V7TileDataset`/`--wdl-prior-dropout` mechanism** (already implemented, part of the still
  load-bearing Spec103 package per this session's dependency audit) — the only existing machinery
  in this codebase that trains a single model to work both with and without its coarse prior
  present, and evaluates both conditions separately (`val_ds` vs. `val_noprior_ds`). This is the
  direct answer to "a minimap alone cannot yield a WDL": the model does not need one, it needs
  demonstrated robustness to the prior's absence, which this mechanism already provides a working,
  proven pattern for.
- **Externally corroborated by**: a background research pass this session found that "coarse
  elevation prior + RGB-guided residual refinement" is an independently-converged-upon pattern in
  outside remote-sensing literature (guided DEM/DSM super-resolution — Real-GDSR, Prompt2DEM, and
  related work), reported in `specs/122-dataset-curation/research.md` Part B. This is outside
  validation that the shape of this design is sound, not just an internally-preferred guess.

## Out Of Scope (Explicit)

- Any model or stage that predicts a coarse height lattice (WDL-scale or otherwise) from minimap
  RGB alone. This exact task has been attempted three times and is a closed, negative question —
  see Spec 117 and Spec 121 research.md for the full evidence trail.
- Object-*instance* detection, identification, or retrieval on minimaps — i.e. determining which
  specific object a given minimap blob is (Specs 119/120, closed decisive negative: real instances
  measured at p50=10px, indistinguishable from unrelated blobs at ~0.99 cosine similarity at that
  scale). Not used, not revived, including as a preprocessing/cleaning step before this model's
  inference. This is a different, harder task than the semantic *family* classification this spec
  does reuse — see FR-018.
- Training or validating Spec 118 User Story 3's building/doodad semantic segmenter as part of this
  spec. It is code-complete but not yet validated against real data; this spec may use it later as
  an optional ablation once that validation exists elsewhere, but does not depend on it and does
  not perform that validation itself.
- Any DepthAnything-family model or backbone (project-wide blacklist).
- Texture/alpha reconstruction, liquid reconstruction, minimap super-resolution — separate lanes,
  untouched by this spec.
- RunPod or other cloud training infrastructure, and the parallel legacy-Python-detangling/cleanup
  effort the user has also requested this session — tracked separately, not this spec's concern.
- Changing the v50 Zarr store's existing signal-writing contract beyond additively wiring in the
  new real-WDL-prior signal; no existing signal is renamed, removed, or reinterpreted.
- Multi-task models, shared weights between stages, or any model whose output is not a single
  well-defined residual signal (project constitution, Residual Model Chain principle).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Real WDL Prior Becomes a First-Class, Deployment-Available v50 Signal (Priority: P1)

A model-training operator needs a coarse height prior for a tile that is genuinely available at
deployment time — not one that requires the tile's own ground-truth height data to compute. They
get a per-tile real WDL reading (from the map's `.wdl` client file) merged with a clearly-labeled
synthetic fallback only where real WDL genuinely has no coverage, with the source and confidence of
every merged point recorded — never silently blended into a single number that hides which parts
are real.

**Why this priority**: Every other part of this spec depends on this signal actually being real and
deployment-available. If this story is wrong, everything built on top of it inherits the same flaw
that closed Specs 117 and 121.

**Independent Test**: For a held-out tile whose ground-truth height the pipeline never touches
while producing the prior, confirm the prior is still fully produced — proving it does not depend
on the value it would otherwise be used to predict.

**Acceptance Scenarios**:

1. **Given** a map with a real `.wdl` client file and a specific tile, **When** the prior is built
   for that tile, **Then** the real WDL reading is used wherever it covers that tile's grid points,
   and each such point is marked with its real source and full confidence.
2. **Given** a tile or grid point the real WDL file does not cover, **When** the prior is built,
   **Then** a synthetic fallback is used only for that point, is marked with a distinct source and
   lower confidence than a real point, and the tile is never silently treated as if it were fully
   real-covered.
3. **Given** the same tile's prior requested twice, **When** built once from only the client `.wdl`
   file and once including the tile's own harvested height data (where available, for comparison
   only), **Then** the two are recorded and compared, and the comparison confirms the real-WDL-only
   prior does not require the tile's own height ground truth to exist.
4. **Given** the existing synthetic `wdl_outer_17`/`wdl_inner_16` v50 signal, **When** this new real
   prior is added, **Then** the existing signal is left unchanged and is never presented as
   equivalent to the new real prior in any downstream training or evaluation code.

---

### User Story 2 - One Residual Detailer, Real Prior In, Full Height Out (Priority: P1)

A model-training operator trains exactly one small model that takes the authored minimap image and
the upsampled real WDL prior as input and predicts the full-resolution height field as the prior
plus a learned residual. No stage in this pipeline ever predicts a coarse prior from the image —
the prior is always the real signal from User Story 1, when it exists.

**Why this priority**: This is the actual deliverable — the model. It is P1-after-US1 because it is
meaningless without a trustworthy, real prior to refine.

**Independent Test**: On the frozen Spec 116 spatially-isolated held-out split, the trained
detailer's height error is compared against two trivial baselines — tile-mean, and the upsampled
real prior with zero learned refinement — and beats both by a recorded margin.

**Acceptance Scenarios**:

1. **Given** the real WDL prior and the authored minimap for a tile, **When** the detailer runs,
   **Then** its output is the upsampled prior plus a learned residual, never a value produced
   independently of the prior.
2. **Given** a frozen held-out split with no spatial leakage, **When** the detailer is evaluated,
   **Then** the run record contains the detailer's error, the tile-mean baseline error, and the
   upsampled-prior-only (no learning) baseline error, so the model's real contribution is visible
   on its own, separate from the prior's.
3. **Given** the model checkpoint, **When** its size is recorded, **Then** it is a single-digit to
   low-tens-of-millions-parameter model and trains within a single consumer GPU's 12GB VRAM budget.
4. **Given** the trained model, **When** compared against the families of models this design draws
   on (the Spec 114 detailer refinement precedent and the `terrain_refiner_train.py` prior-dropout
   precedent), **Then** the comparison is explicit in the run record — this design is justified as
   still appropriate, not silently assumed.

---

### User Story 3 - The Model Degrades Gracefully When No Real Prior Exists (Priority: P1)

A model-training operator trains the same single detailer to also handle the case where a tile or
region has no real WDL prior at all — not the rare partial-coverage gap User Story 1's synthetic
fallback handles, but a genuinely novel input (a hand-painted minimap, or reconstructed alpha-era
content with no surviving `.wdl`) where no prior can be produced by any means. The model is trained
with the prior randomly withheld a fraction of the time (reusing `terrain_refiner_train.py`'s
`V7TileDataset`/`--wdl-prior-dropout` pattern), and is evaluated separately on prior-present and
prior-absent held-out samples, so both conditions are honestly reported rather than only the easier
one.

**Why this priority**: This is the direct answer to the concern that a minimap alone cannot yield a
WDL. Without this story, the model would hard-fail or silently degrade in an unmeasured way on
exactly the class of input — reconstruction from a minimap with no surviving real data — that gives
this whole model lineage its actual point.

**Independent Test**: The same held-out split is evaluated twice: once with each tile's real prior
supplied normally, and once with every tile's prior forcibly withheld. Both results are recorded
and compared; the prior-absent condition is expected to be worse than the prior-present condition
(this is not a contradiction — it is the model correctly using the prior when available) but is
still required to run without crashing and to beat a trivial no-elevation-information baseline.

**Acceptance Scenarios**:

1. **Given** training data, **When** the model trains, **Then** the real prior is withheld for a
   configurable fraction of training samples (matching the existing `--wdl-prior-dropout` pattern),
   and the model receives an explicit "prior absent" signal for those samples rather than a
   zero-filled value indistinguishable from a real low-elevation reading.
2. **Given** the frozen held-out split, **When** evaluation runs, **Then** it reports both a
   prior-present error and a prior-absent error as two distinct, separately-labeled numbers in the
   run record — never averaged together into one figure that hides the difference.
3. **Given** a genuinely novel minimap with no corresponding `.wdl` file anywhere (the true
   deployment case this story exists for), **When** inference runs, **Then** it completes without
   crashing and produces a height field, explicitly flagged in its output as prior-absent, so a
   consumer of the result knows it is the degraded-confidence case.
4. **Given** the prior-absent evaluation result, **When** compared against a trivial baseline that
   uses no elevation information at all (e.g. a global constant), **Then** the model still beats it
   — the degraded mode must be genuinely better than nothing, even though it is not held to the
   same bar as the prior-present mode.
5. **Given** the prior-absent evaluation set, **When** results are reported, **Then** they are
   stratified by object/road surface coverage (reusing Spec 115's proven semantic classifier, not
   the dead object-instance-retrieval line), so an object-dense tile's worse accuracy is visible on
   its own rather than averaged into a single number that hides it.

---

### User Story 4 - Training Data Selection Is Driven By The Curation Manifest, Not Reinvented (Priority: P1)

A model-training operator selects which tiles this model trains on by reading the Spec 122
curation manifest — not by writing new, one-off filtering logic. They can see, per tile, which
quality buckets and mismatch findings applied, and the training run itself records exactly which
selection rule was used and how many tiles it included or excluded.

**Why this priority**: This is the direct payoff of treating "the dataset" as a completed prior
phase — if this model's training data selection does not visibly depend on the curation manifest,
Spec 122 has not actually changed anything about how models get trained.

**Independent Test**: Two training runs use different curation-bucket selection rules (e.g.
clean-only vs. clean-plus-low-severity-mismatch) on the identical split, and both runs' records
show which rule was used and the resulting tile counts, with results comparable side by side.

**Acceptance Scenarios**:

1. **Given** the curation manifest for the training store, **When** a training run starts, **Then**
   it reads bucket and finding data from the manifest to decide which tiles to include, and the
   run record states the exact selection rule and resulting tile count.
2. **Given** a tile flagged with a synthetic-minimap-fidelity-gap finding, **When** that tile would
   otherwise be included via a synthesized (not authored) minimap, **Then** this spec's design
   makes an explicit, recorded decision about how that finding affects the tile's use (exclude it,
   down-weight it in the loss, or knowingly include it) — the decision must be visible in the spec
   and the run record, never silently defaulted.
3. **Given** a tile flagged `pathological` or `blank` by the curation manifest's difficulty/coverage
   buckets, **When** the default training selection runs, **Then** that tile is excluded by
   default, with the exclusion count recorded — and an operator can still deliberately include it
   via an explicit override, consistent with Spec 122's partition-not-filter principle (the data
   remains available, the default choice is just conservative).
4. **Given** a height-normal-mismatch finding on a tile, **When** the default training selection
   runs, **Then** that tile is excluded by default from this specific model's training (poisoned
   height supervision would directly corrupt this model's target), with the same override
   availability as Scenario 3.

---

### Edge Cases

- A map or tile region with no real WDL coverage at all (only synthetic fallback available): the
  tile is trainable but its prior confidence is recorded as lower; the training run reports the
  real-vs-synthetic-fallback split of its training set so a synthetic-fallback-heavy run is visible,
  not hidden.
- A tile whose real WDL reading and synthetic fallback disagree substantially (a data-quality
  question in its own right, distinct from but related to Spec 122's mismatch findings): the
  disagreement is recorded, not silently resolved by picking one value.
- A held-out map used only for evaluation, never training: the prior-building mechanism must work
  on it identically to a training-set map, since it never depends on that map's own ground-truth
  height data.
- A tile where the curation manifest's checks were not_evaluable (e.g. missing normals): the
  training-selection rule must treat "not evaluated" distinctly from "evaluated and clean" per
  Spec 122's own contract — it must not silently count as either a pass or a fail.
- Model inference on an out-of-distribution minimap that still corresponds to a real, mapped WoW
  region with a real `.wdl` file: the real WDL prior is obtainable (it depends only on the map, not
  on any harvested tile data) and is used normally; the deployment chain must be provably able to
  run on such an input with no ground-truth signal read at any point.
- Model inference on a genuinely novel minimap with **no** corresponding `.wdl` anywhere (a
  hand-painted image, or reconstructed content from a client snapshot that never shipped a WDL for
  that state) — the case User Story 3 exists for: the model MUST run in prior-absent mode and
  produce a result, never crash or silently substitute a fabricated prior it presents as real.
- A tile that has real WDL coverage during training but is evaluated in forced prior-absent mode
  (User Story 3's dropout validation): the ground-truth comparison still uses the tile's real
  height, so the prior-absent error is measured honestly against the same target the prior-present
  run is measured against, not a different or easier one.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST read the real per-map WDL client file and produce a per-tile coarse
  height prior from it, independent of any harvested per-tile ground-truth height data for that
  tile.
- **FR-002**: Where the real WDL file does not cover a given tile or grid point, the system MUST
  produce a synthetic fallback value, and MUST record, per point, whether its source was real or
  synthetic fallback, plus a confidence value that is strictly higher for real points.
- **FR-003**: The real WDL prior MUST be exposed as a new, distinct v50 signal, separate from and
  never conflated with the existing synthetic `wdl_outer_17`/`wdl_inner_16` signal.
- **FR-004**: The system MUST provide a deterministic, exact method for placing the real WDL prior's
  native grid points onto the full-resolution height grid (upsampling), such that every native
  sample point maps to its true position with no approximation error at those exact points.
- **FR-005**: The system MUST train exactly one model — a residual detailer — whose output is the
  upsampled real WDL prior (or its explicit absence, per FR-016) plus a learned residual; no stage
  in this design may predict a coarse prior from an image.
- **FR-006**: The detailer's total parameter count MUST be recorded and MUST fall within a
  single-digit to low-tens-of-millions band, and the model MUST train within a single consumer
  GPU's 12GB VRAM budget.
- **FR-007**: Every training and evaluation run MUST report the model's prior-present error
  alongside both a tile-mean baseline and an upsampled-prior-with-no-learning baseline, so the
  learned model's actual contribution over the raw prior is always visible.
- **FR-008**: Training data selection MUST be driven by the Spec 122 curation manifest: the default
  selection MUST exclude tiles flagged `pathological`/`blank` (difficulty/coverage buckets) and
  tiles with a height-normal-mismatch finding, with an explicit, documented override path for an
  operator who wants to deliberately include them.
- **FR-009**: The design MUST make an explicit, recorded decision for how tiles with a
  synthetic-minimap-fidelity-gap finding are handled when a synthesized (not authored) minimap
  would otherwise be used as this model's input — the decision (exclude, down-weight, or
  knowingly include) MUST be stated in this spec and reflected in the training run record; it MUST
  NOT be left to an unstated default in the implementation.
- **FR-010**: Every training run MUST use the Spec 116 spatially-isolated held-out split (no
  fallback to an unspecified or adjacency-leaky split).
- **FR-011**: Every training run MUST record which curation-manifest selection rule was applied and
  the resulting included/excluded tile counts, in the run record.
- **FR-012**: The model MUST be independently trained and independently checkpointed, with no
  shared weights with any other stage or model (constitution Residual Model Chain principle); it
  MUST NOT be a multi-task model.
- **FR-013**: All CLIs in this spec MUST be dry-run-first; all training launches remain user-run
  (project-wide rule); the assistant prepares and hands off exact commands.
- **FR-014**: The deployment/inference chain MUST run on an authored minimap alone when no real WDL
  prior is available, and on an authored minimap plus the map's real WDL prior when one is
  available; in neither case does it read ground-truth height, a mismatch finding, or any other
  training-only signal at inference time.
- **FR-015**: This spec MUST NOT modify the v50 Zarr store's existing signal-writing contract beyond
  additively adding the new real-WDL-prior signal(s); no existing signal is renamed, removed, or
  reinterpreted.
- **FR-016**: The detailer MUST be trained with the real prior withheld for a configurable fraction
  of samples (reusing the `--wdl-prior-dropout` pattern), with an explicit "prior absent" input
  signal distinct from any real low-elevation value, so the same model produces a usable result
  when no real prior exists for a given input.
- **FR-017**: Every evaluation MUST report prior-present and prior-absent error as two distinct
  numbers, never blended into one; the prior-absent condition MUST be compared against a trivial
  no-elevation-information baseline (not the tile-mean or upsampled-prior baselines, which assume
  information the prior-absent case does not have) and MUST beat it.
- **FR-018**: The model MUST reuse Spec 115's proven semantic terrain-feature classifier (broad
  surface-family classification — road, natural terrain, etc. — not per-object identification) as
  a generated input channel, via the same `--feature-store` mechanism already validated across
  Specs 115-118, to reduce the color-as-elevation confound objects and roads otherwise create —
  especially in prior-absent mode, where the minimap is the model's only signal. This spec MUST
  NOT use, train, or depend on Specs 119/120's object-instance detection/retrieval (confirmed dead
  end) for this or any other purpose.
- **FR-019**: Prior-absent evaluation MUST be reported stratified by object/road coverage (reusing
  the object-coverage or terrain-feature-coverage stratification pattern already used elsewhere in
  this project), not only as one aggregate number — so a reviewer can see whether accuracy holds up
  on open terrain even where it degrades on object-dense tiles, rather than one number hiding the
  other.

### Key Entities

- **Real WDL Prior**: per-tile coarse height data assembled from the map's real `.wdl` client file
  plus a clearly-labeled synthetic fallback for uncovered points, with per-point source and
  confidence — the model's coarse-height input whenever it exists for a given map/tile; not every
  possible input has one.
- **Residual Detailer**: the single trained model in this spec; predicts a residual over the
  upsampled real WDL prior when present, or over an explicit prior-absent baseline when not,
  guided by the authored minimap image, to produce full-resolution height in both cases.
- **Prior-Dropout Training**: the mechanism (reused from `terrain_refiner_train.py`) that randomly
  withholds the real prior during training, with a distinct "absent" signal, so the detailer learns
  to degrade gracefully rather than hard-depend on the prior always being present.
- **Semantic Terrain-Feature Map**: Spec 115's proven, from-scratch, broad-family (road/natural/
  etc.) classifier output, reused here as a generated input channel to reduce the color-as-
  elevation confound — explicitly not an object-instance detector, and not Specs 119/120's dead
  retrieval line.
- **Curation-Driven Training Selection**: the rule, read from the Spec 122 curation manifest, that
  decides which tiles this model trains on by default, with an explicit override path and a
  recorded decision for the synthetic-fidelity-gap case.
- **Held-Out Split**: the Spec 116 spatially-isolated split, reused unmodified.
- **Model Stage Run Record**: the existing `v50-model-stage-run-v1` (or successor) record, extended
  in practice (not schema) to always carry the curation-selection rule and tile counts, and both
  baseline comparisons.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The real WDL prior is producible for a held-out tile using zero information from that
  tile's own harvested ground-truth height data — verified by construction (the prior-building code
  path never reads that tile's height array) and by direct test.
- **SC-002**: The trained detailer's held-out height error is at least 9% below the
  upsampled-real-prior-only baseline (matching or exceeding the Spec 114 precedent's cleared bar),
  on the frozen Spec 116 split.
- **SC-003**: The trained detailer's held-out height error is below the tile-mean baseline by a
  recorded margin (the bar every prior attempt in this exact chain has been measured against).
- **SC-004**: A training run using the default curation-driven selection excludes a measurably
  different (and reported) tile set than a run with no curation filtering at all, proving the
  selection rule has real effect, not just theoretical existence.
- **SC-005**: The full deployment chain runs on at least two genuinely out-of-distribution inputs —
  one for a real, mapped region (real prior available, used normally) and one with no corresponding
  `.wdl` anywhere (prior-absent mode, e.g. a hand-painted minimap) — both with no crash and no
  ground-truth signal read, and both produce a result the user issues an explicit visual verdict on.
- **SC-006**: The checkpoint is within the 12GB VRAM / single-digit-to-low-tens-of-millions-param
  band and this is recorded, not merely assumed.
- **SC-007**: The prior-absent held-out error beats a trivial no-elevation-information baseline by a
  recorded margin, proving the degraded mode is genuinely usable and not merely non-crashing.
- **SC-008**: The prior-absent stratified report (FR-019) shows the model beating the trivial
  baseline on low-object-coverage tiles specifically, not only in an aggregate that could be
  carried entirely by easy, open-terrain tiles.

## Assumptions

- The real per-map `.wdl` client file is available and readable for the primary corpus build
  (`0_5_3_3368`) via the existing `WowViewer.Core.IO.Maps.WdlSummaryReader`; per Spec 094's own
  verified finding, alpha-era WDLs are packaged as a loose per-map mini-MPQ and LK-era WDLs are
  inside the main archives — both are readable by the existing reader.
  MAHO (hole) data is not exposed by the existing reader; this spec does not depend on it.
- The exact real/synthetic merge policy and the exact quincunx upsample-to-257×257 method are
  taken as Spec 094's already-verified design (Implementation Amendments A1/A5/A6 in
  `specs/094-wdl-prior-v24/spec.md`) rather than re-derived from scratch — re-deriving a
  already-solved deterministic geometry problem would not be "fresh eyes," it would be redundant
  risk.
- The primary training/evaluation corpus is Kalimdor and Azeroth (never PVPZone02/Kalidar for
  gauging anything, per standing project guidance); the authored minimap remains the deployment
  input of record (project decision, Spec 112).
- The prior-dropout mechanism and its default rate are taken as `terrain_refiner_train.py`'s
  already-implemented `V7TileDataset`/`--wdl-prior-dropout` design (default 0.25) rather than
  re-derived; this is the one existing piece of this codebase that already solves "one model,
  prior-present and prior-absent," and planning should evaluate reusing or adapting it directly
  before considering a new implementation.
- A minimap with no corresponding `.wdl` file is expected to produce a measurably less accurate
  height prediction than one with a real prior — this is accepted as inherent to the problem (per
  Specs 117/121's finding that RGB alone carries limited elevation signal), not a defect to
  engineer away. The bar for that mode (SC-007) is "genuinely useful," not "as good as with a real
  prior."
- Object/road contamination of minimap color (buildings, roofs, roads misread as elevation cues) is
  a real, separate constraint on the prior-absent mode's accuracy, distinct from the general
  RGB-elevation-signal weakness above. This spec addresses it only with Spec 115's proven semantic
  terrain-feature classifier (broad surface family, not per-object identity) as a generated input
  channel — the one tool in this project's history that has actually demonstrated a positive result
  on this exact confound. It explicitly does not attempt, revive, or wait on object-instance
  detection (Specs 119/120, dead) or the unvalidated Spec 118 US3 segmenter to solve this further;
  if the resulting accuracy is still insufficient, that is a valid, reportable outcome for planning
  to weigh, not something this spec pre-solves by assuming a better classifier will appear.
- The synthetic-minimap-fidelity-gap handling decision (FR-009) is expected, based on this
  session's real Kalimdor measurement (592/731 evaluated tiles flagged high-severity gap), to lean
  toward authored-only training data by default — but this spec requires the decision be made and
  recorded explicitly during planning, not assumed here, since it is a genuine open design choice
  the user has flagged interest in exploring (using the gap as a loss signal) rather than a settled
  question.
- "Ground up" means a new model and a new, correct prior signal — it does not mean discarding
  proven infrastructure (Spec 116's split, Spec 114's detailer architecture family, the run-record
  contract) for its own sake; re-justifying reuse is the "fresh eyes" requirement, not reinventing
  working machinery.
