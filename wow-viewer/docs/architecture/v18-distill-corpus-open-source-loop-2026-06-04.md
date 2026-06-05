# V18 Focused Two-Build Minimap-to-Terrain Loop — 2026-06-04

## Purpose

This document is the architecture summary for the active spec-047 lane. It
compresses the current decisions so future sessions do not keep reopening the
same failed branches.

The active spec pack is:

- `wow-viewer/specs/047-v18-distill-corpus-open-source-loop/spec.md`
- `wow-viewer/specs/047-v18-distill-corpus-open-source-loop/plan.md`
- `wow-viewer/specs/047-v18-distill-corpus-open-source-loop/tasks.md`

If this document and the spec disagree, the spec wins.

## Current Truth State

The focused staged client roots are:

- `I:\parp\parp-tools\output\tmp\wowarchive-clients\0_5_3_3368\World of Warcraft`
- `I:\parp\parp-tools\output\tmp\wowarchive-clients\3_3_5_12340\World of Warcraft`

The active corpus remains bounded to:

- `wow-viewer/output/datasets/v18/0_5_3_3368.zarr`
- `wow-viewer/output/datasets/v18/3_3_5_12340.zarr`

Renderer-truth capture is not the proof owner for this lane. Both focused
stores were previously cleared back to honest renderer-truth coverage:

- `has_object_visibility_mask = 0`
- `has_no_object_minimap = 0`

That reset matters only as a guardrail: stale capture artifacts must not be
mistaken for training truth again.

## First Bounded Proofs

Two bounded smoke runs were completed against the focused V18 stores on
2026-06-04 using the simplified minimap-only contract:

- height run:
  - run dir: `wow-viewer/models/v18/height/runs/v18_height_focus_minimap_smoke_20260604_r2/`
  - command surface: `scripts/train_v18.py height`
  - build mix: 16 train tiles + 4 val tiles from each focused build
  - bounded budget: 1 epoch, batch size 4, 32 train tiles, 8 val tiles
  - result: `val_loss = 0.6626`
- normal run:
  - run dir: `wow-viewer/models/v18/normal/runs/v18_normal_focus_minimap_smoke_20260604_r2/`
  - command surface: `scripts/train_v18.py normal`
  - build mix: 16 train tiles + 4 val tiles from each focused build
  - bounded budget: 1 epoch, batch size 4, 32 train tiles, 8 val tiles
  - result: `val_loss = 0.2251`
  - active contract log line: `Normal contract: input=minimap_rgb -> output=normals_xyz | simplified_loss=true`

The first normal attempt exposed one real leftover seam: `_preview_normal(...)`
still expected old weighted-loss tensors such as `terrain_valid_mask`. That was
patched so the simplified lane now previews from `base_mask`, `train_mask`, and
`invalid_mask` instead.

## Scope Reset

The active lane is now deliberately basic:

1. train on `0_5_3_3368` and `3_3_5_12340` only,
2. use `minimap_rgb` as the only model input,
3. train one height model and one normal model,
4. keep the losses plain and local to the target signal.

That means:

- no renderer-truth capture dependency,
- no object-mask loss gating,
- no roof-mask loss gating,
- no liquid weighting,
- no refiner-driven active signoff,
- no synth/distill/open-source loop in this iteration.

## Why This Reset Happened

The prior lane kept drifting into weak or misleading supervision surfaces:

- blank or stale renderer artifacts,
- capture-path ownership confusion,
- object/roof weighting that was not required for the core task,
- too much time spent debugging secondary signals instead of proving the main
  terrain correlation.

The repo does not need another speculative signal stack right now. It needs a
working `minimap -> terrain` loop.

## Active Model Contracts

### Height

- input: `minimap_rgb`
- target: `height_257`
- loss: plain `L1(pred_height, target_height)`

### Normals

- input: `minimap_rgb`
- target: `normal_xyz`
- validity mask: `normal_mask`
- loss: masked cosine loss only

These contracts are implemented in the existing V16.1 / V18 trainer surface.
The active default normal route is the minimap-only variant, not the old
height/refiner/object-roof branches.

## Pipeline Summary

```text
staged 0.5.3 + 3.3.5 client roots
        │
        ▼
build_focused_two_build_corpus.py
        │
        ▼
focused V18 Zarr stores
        │
        ├──► train_v16_1_height.py
        │     └──► models/v18/height/runs/<run-name>/
        │
        └──► train_v16_1_normal.py
              └──► models/v18/normal/runs/<run-name>/
```

## Core Decisions

### Decision 1: Two builds are enough for the active proof

The active lane is not trying to maximize corpus width. It is trying to stand
up a trustworthy baseline. `0_5_3_3368` and `3_3_5_12340` are enough for that.

### Decision 2: Minimap-only input is the proof owner

The user asked to go back to basics. The active proof owner is the minimap
image itself, not auxiliary channels.

### Decision 3: Plain losses beat speculative weighting here

Height uses plain L1. Normals use masked cosine. If that basic lane fails, the
answer is not to immediately add more loss terms. The answer is to understand
the corpus or the model.

### Decision 4: Renderer truth is explicitly out of scope

Capture output may still be useful later, but it is not allowed to silently
re-enter signoff for this lane.

## Risks and Guardrails

- **Spec drift**: do not let spec 047 quietly regain the distill/open-source
  scope while the active repo work is just minimap-only training.
- **Signal drift**: do not treat roof/object mismatches as blockers unless a
  specific active trainer path consumes them again.
- **Validation drift**: do not call old previews or old checkpoints current
  proof after changing losses or input contracts.

## Open Follow-Up

- Validate both focused stores specifically for the minimap/height/normal
  contract.
- Scale the bounded height proof into a real multi-epoch run.
- Scale the bounded normal proof into a real multi-epoch run.
- Use those outputs before reopening any fancier lane.
