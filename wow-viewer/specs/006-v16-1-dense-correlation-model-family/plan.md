# Implementation Plan: V16.1 Dense Correlation Model Family

**Spec**: `006-v16-1-dense-correlation-model-family/spec.md`
**Created**: 2026-05-21

## Phase 1: V16.1 Contract And Naming

**Goal**: Land the durable V16.1 direction so future chats do not keep spending
time on the V16 shared-head architecture.

### Step 1.1 — Freeze the architecture boundary
Create the V16.1 spec pack and document that V16.1 means V16-derived dense
correlation families linked together at the output assembly boundary.
**Validation**: `spec.md`, `plan.md`, and `tasks.md` exist and explicitly route
future work into the V16.1 dense-correlation lane.

### Step 1.2 — Record continuity truth
Update the memory-bank continuity files so the live ML lane now treats V16 as a
baseline and V16.1 as the next architecture slice.
**Validation**: `activeContext.md` and `progress.md` mention V16.1 and the
split-and-link dense-correlation framing.

## Phase 2: Height First

**Goal**: Prove the first dense-correlation family with the most important terrain
signal before touching the rest of the stack.

### Step 2.1 — Extract shared trainer utilities
Move only non-model, non-loss helper logic out of `train_v16.py` into a shared
V16.1-safe utility surface.
**Validation**: helpers are importable without dragging in a shared multitask
loss/model contract.

### Step 2.2 — Create `train_v16_1_height.py`
Implement a dedicated height-only trainer, run layout, checkpoint naming, and
validation artifact contract.
**Validation**: 1-epoch CPU smoke run against `3_3_5_12340` completes.

### Step 2.3 — Compare against V16 baseline
Run a bounded validation comparison between V16 height output and V16.1 height
output on the same curated tiles.
**Validation**: one comparison summary exists under the V16.1 run root.

## Phase 3: Normals Second

**Goal**: Directly test whether the current normals failure is architectural or
fundamental.

### Step 3.1 — Create `train_v16_1_normal.py`
Implement a dedicated normal-only trainer with normal-mask-aware loss and
focused validation panels.
**Validation**: 1-epoch CPU smoke run completes and writes normal-only review
artifacts.

### Step 3.2 — Height-vs-normal independence proof
Show that changing the normal trainer does not require touching the height
trainer code or checkpoint contract.
**Validation**: two disjoint run roots and checkpoint names exist.

## Phase 4: Liquid Family

**Goal**: Treat liquids as a first-class terrain understanding problem rather
than a single soft mask.

### Step 4.1 — Create `train_v16_1_liquid.py`
Implement a dedicated liquid trainer that predicts footprint/placement plus
liquid type evidence.
**Validation**: 1-epoch CPU smoke run completes and writes both footprint and
type-aware review artifacts.

### Step 4.2 — Define liquid-type supervision contract
Document the initial type label surface and how it is derived from the existing
V16 dataset signals.
**Validation**: type classes and degradation rules are written into the trainer
contract.

## Phase 5: Texture Decomposition/Recomposition

**Goal**: Move alpha prediction into a dedicated MCLY/MCAL decomposition family
with recomposition proof.

### Step 5.1 — Audit and map the existing D1 work
Read the existing `train_d1.py`, `D1UNet`, and `D1Dataset` surfaces and define
what is kept, what is discarded, and what moves to Zarr-backed V16.1 data.
**Validation**: one short migration note exists in the V16.1 implementation
artifacts or spec references.

### Step 5.2 — Create `train_v16_1_texcomp.py`
Implement a dedicated decomposition trainer for `mcly_texture_ids` +
`alpha_256` / `mcly_layer_mask`, reusing the useful parts of D1 rather than
restarting the design.
**Validation**: 1-epoch CPU smoke run completes.

### Step 5.3 — Add recomposition proof
Write validation artifacts that show predicted decomposition outputs plus a
recomposed terrain-only image.
**Validation**: one recomposed review panel is emitted for each validation run.

### Step 5.4 — Migrate to V16-quality loss signals
Ensure the V16.1 texture decomposition trainer uses the same object-mask-derived
loss gating quality level available in the V16 Zarr dataset.
**Validation**: training config or validation notes show Zarr-backed signals and
object-mask gating are active.

## Phase 6: Remaining Single-Target Trainers

**Goal**: Extend the same pattern to the rest of the terrain targets without
reintroducing coupling.

### Step 6.1 — Holes trainer
Create `train_v16_1_holes.py`.
**Validation**: CPU smoke run.

## Phase 7: Shared Object Loss Gating

**Goal**: Keep object masks as reusable loss signals across all appropriate V16.1
trainers.

### Step 7.1 — Define shared loss-weight contract
Extract or document a shared object-mask weighting contract that V16.1 trainers
can reuse.
**Validation**: one utility or contract note names the allowed mask signals and
their intended use.

### Step 7.2 — Apply gating in the first V16.1 trainers
Use shared object-mask loss weighting in height, normals, liquids, and texture
decomposition wherever object pixels should be downweighted.
**Validation**: trainer configs or validation notes show that gating is active.

## Phase 8: Stitched Inference

**Goal**: Replace shared-head inference with an explicit per-target checkpoint
assembly surface.

### Step 8.1 — Checkpoint manifest contract
Define the checkpoint manifest and CLI arguments for stitched V16.1 inference.
**Validation**: spec + command help text exist.

### Step 8.2 — Initial stitched inference implementation
Combine the per-target outputs into one terrain prediction bundle.
**Validation**: one end-to-end smoke run writes a combined output directory with
per-target provenance.
