---
description: "Plan or implement the v7.6 paired training alignment slice. Use when the goal is to bring the paired reconstruction branch onto the structured datasets workflow, remove hardcoded legacy cache roots, and make train, inference, and stitch behavior follow the documented output contract."
name: "wow-viewer Training 03 v7.6 Paired Training Alignment"
argument-hint: "Optional v7.6 blocker, cache-root issue, inference contract, or proof target to prioritize"
agent: "codex"
---

Plan or implement the `v7.6` alignment slice: make the paired reconstruction branch consume the same structured dataset truth discipline instead of living off legacy cache assumptions.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/docs/VLM_Training_Guide.md`
4. `gillijimproject_refactor/docs/v76-model-architecture-guide.md`
5. `gillijimproject_refactor/docs/v76-output-dataset-spec.md`
6. `gillijimproject_refactor/src/WoWMapConverter/scripts/cache_v7_6_data.py`
7. `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7_6.py`
8. `gillijimproject_refactor/src/WoWMapConverter/scripts/inference_v7_6.py`
9. `gillijimproject_refactor/src/WoWMapConverter/scripts/stitch_full_map.py`
10. `.codex/prompts/wow-viewer-training/01-shared-dataset-truth-gates.md`
11. `AGENTS.md`

## Goal

Realign `v7.6` with the structured dataset workflow so its cache building, training, inference, and stitched outputs all agree on one explicit input and output contract.

## Current Concrete Problem

- `v7.6` is a real paired-output branch, but it is not yet cleanly aligned with the structured `datasets/` workflow.
- the cache builder still hardcodes a legacy dataset root under `test_data/vlm-datasets/053_Azeroth_v30`.
- inference and stitched outputs still lean on loose-file assumptions instead of the documented output dataset spec.
- until those contracts are explicit, `v7.6` runs are hard to compare, reproduce, or trust.

## Non-Negotiable Constraints

- Keep `v7.6` as a separate paired branch; do not force it into `v7.5.1` supervision semantics.
- Remove hardcoded legacy dataset roots and make the real dataset contract explicit.
- Align cache inputs, train inputs, inference outputs, and stitched outputs to `v76-output-dataset-spec.md`.
- Prefer manifest-driven or config-driven paths over one-off local edits.
- Do not claim `v7.6` recovery from a cache-only patch if the inference and output contract are still loose.
- Keep the scope on data-flow and reproducibility; do not turn this into a new architecture branch unless the current guide is clearly wrong.

## What The Work Must Produce

1. the exact root and manifest contract `v7.6` should consume
2. the exact script or config changes needed to remove legacy-root drift
3. the exact output contract for train, inference, and stitch steps
4. the minimum reproducible command sequence for cache, train, inference, and stitch proof
5. the pass or fail gate for saying `v7.6` is aligned enough to iterate seriously
6. the follow-on backlog after the first aligned run is proven

## Deliverables

Return all items:

1. current drift map
2. required script or config alignment work
3. first reproducible runbook
4. proof expectations for outputs
5. residual risks after the first slice
6. what should stay out of scope for this slice

## First Output

Start with:

1. the exact reason `v7.6` is still off the structured workflow
2. the single highest-leverage alignment slice to do first
3. the first concrete proof sequence after that slice lands
4. what you are explicitly not claiming yet
