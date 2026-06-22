# Implementation Plan: V16.1.1 Curated Normal Acceleration

**Spec**: `007-v16-1-1-curated-normal-acceleration/spec.md`
**Created**: 2026-05-21

## Phase 1: Contract And Routing

**Goal**: Make `V16.1.1` the explicit next normal-lane execution target instead
of leaving future chats to infer the upgrade path from scattered discussion.

### Step 1.1 — Create the V16.1.1 spec pack
Write `spec.md`, `plan.md`, and `tasks.md` for the new lane.
**Validation**: the new spec folder exists and describes the curation-first,
normal-first upgrade direction.

### Step 1.2 — Update continuity routing
Record in continuity docs that `V16.1` is the landed base and `V16.1.1` is the
next implementation slice.
**Validation**: `activeContext.md` and `progress.md` point fresh chats at the
V16.1.1 normal-acceleration lane.

## Phase 2: Difficulty-Aware Curation

**Goal**: Upgrade the curation layer from simple keep/reject filtering into a
usefulness-scoring service for the normal lane.

### Step 2.1 — Add per-tile usefulness scoring
Score tiles using deformation richness, normal coverage, terrain-only validity,
painted-alpha/MCLY presence, and minimap-vs-target usefulness metrics.
**Validation**: a manifest build writes per-tile score fields and score
breakdowns.

### Step 2.2 — Add difficulty buckets
Assign `easy`, `medium`, `hard`, and `pathological` buckets from the scoring
surface.
**Validation**: curation output contains bucket counts and sample rows for each
bucket that survived.

### Step 2.3 — Keep blank genesis rejection hard
Preserve hard rejection for `blank_what_plate_tile` (blank whiteplate) and other proven garbage
cases before bucket assignment.
**Validation**: rejected outputs still classify blank genesis tiles separately.

### Step 2.4 — Add small scouting-manifest command guidance
Write the bounded manifest recipe for short normal-lane scouting runs.
**Validation**: operator docs or plan notes include a `400`-tile mixed scouting
contract.

## Phase 3: Bucket-Aware Epoch Sampling

**Goal**: Make the trainer spend more epochs on useful terrain without throwing
away easier tiles completely.

### Step 3.1 — Extend manifest ingestion for bucket metadata
Teach the dataset/trainer seam to read difficulty buckets and usefulness scores.
**Validation**: a smoke run can print the available bucket counts at startup.

### Step 3.2 — Add bucket-biased epoch sampling
Allow epoch subsets to over-sample `hard` tiles while preserving some `easy`
and `medium` tiles for stability.
**Validation**: per-epoch evidence records bucket usage and selected tile mix.

### Step 3.3 — Emit evidence for sampler behavior
Write run artifacts that show bucket ratios and selected tiles by epoch.
**Validation**: one focused smoke run writes sampler evidence alongside the
existing epoch-order logs.

## Phase 4: Hard-Region Normal Loss

**Goal**: Emphasize deformation-rich terrain regions inside each tile, not just
harder tiles across the corpus.

### Step 4.1 — Refine the detail-weight map
Extend the current normal-detail steering into a first-class hard-region weight
surface using height gradients, local normal variation, painted transitions, and
terrain-valid guidance.
**Validation**: the trainer logs new hard-region weighting metrics.

### Step 4.2 — Keep terrain-only masking authoritative
Ensure object, liquid, and invalid-terrain guidance still caps what can gain
hard-region emphasis.
**Validation**: validation previews show the effective hard-region surface never
re-enables object-polluted areas as strong terrain truth.

### Step 4.3 — Compare against the current detail-boost baseline
Run a focused smoke comparison between plain `--normal-detail-boost` and the
new hard-region weighting.
**Validation**: both run roots exist and the evidence makes the weighting
difference reviewable.

## Phase 5: Optional Uncertainty-Guided Normal Training

**Goal**: Let the model represent ambiguity explicitly if the simpler hard-region
lane still wastes too much loss on uncertain pixels.

### Step 5.1 — Add an optional uncertainty head
Extend the normal trainer with an optional uncertainty prediction path that can
be enabled by CLI/config without changing the dataset format.
**Validation**: model creation and CLI help expose the option cleanly.

### Step 5.2 — Add uncertainty-aware loss weighting
Use predicted uncertainty to attenuate or redistribute pressure on ambiguous
pixels.
**Validation**: a smoke run completes and logs uncertainty metrics.

### Step 5.3 — Add uncertainty review artifacts
Write validation evidence that lets the operator inspect uncertainty separately
from the RGB normal panel.
**Validation**: validation output contains uncertainty artifacts.

## Phase 6: Geometry-Consistency Supervision

**Goal**: Encourage the model to learn coherent local terrain shape rather than
only isolated per-pixel vector agreement.

### Step 6.1 — Add local-relative normal consistency
Introduce a bounded local geometry consistency term over neighboring normals.
**Validation**: the new term can be enabled in a smoke run without destabilizing
the trainer surface.

### Step 6.2 — Record the operator comparison surface
Document which runs compare baseline, hard-region, and uncertainty/consistency
variants.
**Validation**: the run naming scheme and comparison intent are written down in
the operator-facing notes.

## Phase 7: Operator Commands And Fresh-Chat Handoff

**Goal**: Leave the next chat with exact bounded commands and clear proof gates.

### Step 7.0 — Repair the validation review surface
Keep V16.1.1 model validation at least as legible as dataset validation by
restoring best-gated previews with visible labels and multiple samples per
artifact.
**Validation**: new-best preview outputs show panel labels plus build/map/tile
headers across multiple validation rows instead of a single unlabeled tile.

### Step 7.0B — Add startup VRAM autotune
Use the dormant `target_vram_gb` seam to probe a batch-size ladder against the
current card, record the chosen batch size, and optionally preserve the
steps-per-epoch budget by rescaling `train_epoch_tiles`.
**Validation**: a focused smoke run writes autotune evidence, prints the chosen
batch size before training, and logs per-epoch peak CUDA memory plus guidance.

### Step 7.1 — Publish the new curation commands
Write the V16.1.1 manifest-build commands for full-corpus and `400`-tile
scouting modes.
**Validation**: the commands are documented in README or implementation notes.

### Step 7.2 — Publish the new normal-training commands
Write hand-runnable training commands for bucket-aware scouting and longer runs.
**Validation**: the commands are documented and correspond to real CLI flags.

### Step 7.3 — Sync continuity and stop
Update memory-bank truth surfaces after the planning slice so the fresh chat
starts from V16.1.1 instead of reopening V16.1 discussion.
**Validation**: continuity docs mention the next bounded implementation slice.
