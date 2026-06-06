# V18 Focused Two-Build Terrain Reconstruction System — 2026-06-04

## Purpose

This document is the architecture summary for the active spec-047 lane. It
compresses the final owner design so future sessions do not keep reopening old
branches that already failed.

The active spec pack is:

- `wow-viewer/specs/047-v18-distill-corpus-open-source-loop/spec.md`
- `wow-viewer/specs/047-v18-distill-corpus-open-source-loop/plan.md`
- `wow-viewer/specs/047-v18-distill-corpus-open-source-loop/tasks.md`

If this document and the spec disagree, the spec wins.

## Active Boundary

The focused staged client roots are:

- `I:\parp\parp-tools\output\tmp\wowarchive-clients\0_5_3_3368\World of Warcraft`
- `I:\parp\parp-tools\output\tmp\wowarchive-clients\3_3_5_12340\World of Warcraft`

The active corpus remains bounded to:

- `wow-viewer/output/datasets/v18/0_5_3_3368.zarr`
- `wow-viewer/output/datasets/v18/3_3_5_12340.zarr`

These two stores are the only active corpus owner for V18. The current job is
to curate and train against them, not to reopen wider build scope.

Renderer-truth capture is not the proof owner for this lane. Both focused
stores were previously cleared back to honest renderer-truth coverage:

- `has_object_visibility_mask = 0`
- `has_no_object_minimap = 0`

That reset matters only as a guardrail: stale capture artifacts must not be
mistaken for training truth again.

## What The System Is For

The desired output is not a pretty single-tile preview. The desired output is
a pipeline that accepts a set of minimap tiles and emits terrain predictions
that can be quilted back into a believable ADT terrain surface.

The active terrain system is:

1. focused V18 corpus for `0_5_3_3368` and `3_3_5_12340`,
2. focused curation over minimap, height, normal, terrain-validity, and liquid
   signals,
3. one `minimap_rgb -> normalized height` model,
4. one `minimap_rgb -> normal_xyz` model,
5. later quilt-level stitching and ADT writeback follow-through.

## First Bounded Proofs

The first full focused curation run was completed on 2026-06-05:

- command surface: `scripts/build_v18_curation_manifest.py`
- manifest root: `wow-viewer/output/datasets/v18/curation/v18_focus_terrain_v1/`
- corpus coverage: `6763` audited rows across the two focused builds
- kept rows: `4096` (`keep_ratio = 0.6056`)
- dominant reject causes:
  - `blank_minimap_blank_normals = 2396`
  - `blank_what_plate_tile = 221`
  - `normal_minimap_edge_mismatch = 36`
  - `wmo_loss_wipeout_tile = 14`
- kept difficulty mix:
  - `easy = 8`
  - `medium = 30`
  - `hard = 3070`
  - `pathological = 988`

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

## Active Training Contract

The active lane is deliberately basic:

1. train on `0_5_3_3368` and `3_3_5_12340` only,
2. use `minimap_rgb` as the only model input,
3. train one height model and one normal model,
4. keep the losses plain and local to the target signal,
5. use liquids as curation and terrain-validity context rather than as a large
   auxiliary loss stack.

That means:

- no renderer-truth capture dependency,
- no precise object-mask loss gating,
- no roof-mask loss gating,
- no liquid-derived weighted-loss stack,
- no refiner-driven active signoff,
- no synth/distill/open-source loop in this iteration.

The active height and normal runs still mask loss to terrain-valid regions
derived from the harvested validity tensors. That keeps liquid-hidden and
object-hidden terrain from poisoning optimization without turning liquids into a
separate weighted auxiliary objective.

That offline supervised-eval path must not be confused with deployment proof:

- training `val_loss` and preview panels can use hidden truth/mask tensors for
  scoring and evidence
- the runtime proof surface for spec `047` is minimap-only inference through
  `scripts/infer_v18_focus.py`
- if a question is “can the current checkpoint run with only minimap input?”,
  the answer must come from that inference surface, not from trainer val logs

Those terrain-valid tensors now explicitly include both the basement/ground
object masks and the WMO roof/top-geometry mask when present. Preview evidence
must show the actual combined training weight rather than a basement-only
object-weight panel.

The focused operator surface also now defaults to strict near-equal per-build
sampling. When one build has fewer eligible rows, oversized pool/epoch requests
are automatically capped to the largest feasible balanced subset instead of
quietly running a skewed epoch.

Focused V18 training now also supports restrained rotating bucket coverage:

- `train_v18_focus.py` defaults to `--train-bucket-rotation-fraction 0.10`
- each epoch trains on roughly ten percent of every available build/bucket
  stratum in the focused train pool
- later epochs rotate through the remaining rows instead of replaying the same
  full curated pool every time
- very small strata can complete their coverage cycle sooner than larger strata;
  that is expected and preferred to dead epochs

## Why This Design Holds

The prior lane kept drifting into weak or misleading supervision surfaces:

- blank or stale renderer artifacts,
- capture-path ownership confusion,
- object/roof weighting that was not required for the core task,
- too much time spent debugging secondary signals instead of proving the main
  terrain correlation.

The repo does not need another speculative signal stack right now. It needs a
working minimap-to-terrain loop with a focused operator surface.

## Active Model Contracts

### Height

- input: `minimap_rgb`
- target: `normalized height_257`
- loss: plain `L1(pred_height, target_height)`

### Normals

- input: `minimap_rgb`
- target: `normal_xyz`
- validity mask: `normal_mask`
- loss: masked cosine loss only

Height and normals remain separate runs and separate checkpoints. They are used
together downstream, but they are not trained as one shared-weight multitask
model.

## Pipeline Summary

```text
staged 0.5.3 + 3.3.5 client roots
        │
        ▼
focused V18 Zarr stores
        │
        ▼
build_v18_curation_manifest.py
        │
        ├──► kept_tiles.parquet / summary.json
        │
        ├──► train_v18_focus.py height
        │     └──► models/v18/height/runs/<run-name>/
        │
        └──► train_v18_focus.py normal
              └──► models/v18/normal/runs/<run-name>/
```

## Core Decisions

### Decision 1: Two builds are enough for the active proof

The active lane is not trying to maximize corpus width. It is trying to stand
up a trustworthy baseline. `0_5_3_3368` and `3_3_5_12340` are enough for that.

### Decision 2: Minimap-only input is the proof owner

The active proof owner is the minimap image itself, not auxiliary channels.

### Decision 3: Plain losses beat speculative weighting here

Height uses plain L1. Normals use masked cosine. If that basic lane fails, the
answer is not to immediately add more loss terms. The answer is to understand
the corpus or the model.

### Decision 4: Liquids stay in curation

Liquid masks remain useful because water-only and hidden-terrain tiles are a
real curation problem. They stay in the validity/filtering surface without
reopening the old large loss stack.

### Decision 5: Renderer truth is explicitly out of scope

Capture output may still be useful later, but it is not allowed to silently
re-enter signoff for this lane.

## Risks and Guardrails

- **Spec drift**: do not let spec 047 quietly regain the distill/open-source
  scope or a giant auxiliary-loss stack.
- **Signal drift**: do not reintroduce object/roof precision masks as active
  blockers unless a specific trainer path consumes them again.
- **Validation drift**: do not call old previews or old checkpoints current
  proof after changing losses, curation, or input contracts.
- **Liquid-hidden overtraining**: if minimap-visible water dominates a tile,
  reject low-trainable rows in curation and keep terrain-valid masking live in
  the height/normal losses.

## Open Follow-Up

- Scale the bounded height proof into a real multi-epoch run through
  `train_v18_focus.py height`.
- Scale the bounded normal proof into a real multi-epoch run through
  `train_v18_focus.py normal`.
- Tune focused runs for the observed 8 GB lane via startup autotune and the new
  rotating bucket-coverage epochs instead of the earlier smoke-budget settings.
- Keep the focused curation manifest stable unless a specific reject-pattern
  review justifies threshold changes.
- Keep quilt-level stitching and ADT writeback as the next downstream design
  owner instead of smuggling it into trainer changes.
