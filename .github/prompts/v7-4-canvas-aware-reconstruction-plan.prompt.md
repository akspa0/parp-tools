---
description: "Plan the next V7.4 canvas-aware terrain reconstruction slice, including dataset curation, dedupe, alpha-brush harvesting, liquid-mask sanity, shared-backbone model tiering, and wow-viewer ownership direction."
name: "V7.4 Canvas-Aware Reconstruction Plan"
argument-hint: "Optional focus area such as liquid supervision, alpha-brush harvesting, concept clustering, refiner-model tiering, or manifest/corpus curation"
agent: "agent"
---

Plan the next narrow V7.4 reconstruction slice without collapsing back into generic training advice or isolated hyperparameter churn.

If the ask is only about the older provenance-first dataset contract seam and not the broader V7.4 canvas-aware direction, use [`.github/prompts/vlm-dataset-reconstruction-plan.prompt.md`](.github/prompts/vlm-dataset-reconstruction-plan.prompt.md) instead.

## Read First

1. [`gillijimproject_refactor/plans/v7_4_canvas_aware_reconstruction_plan_2026-04-12.md`](gillijimproject_refactor/plans/v7_4_canvas_aware_reconstruction_plan_2026-04-12.md)
2. [`gillijimproject_refactor/plans/vlm_dataset_reconstruction_plan_2026-03-31.md`](gillijimproject_refactor/plans/vlm_dataset_reconstruction_plan_2026-03-31.md)
3. [`gillijimproject_refactor/plans/wow_viewer_ml_tool_suite_cutover_plan_2026-04-10.md`](gillijimproject_refactor/plans/wow_viewer_ml_tool_suite_cutover_plan_2026-04-10.md)
4. [`gillijimproject_refactor/plans/mdxviewer_ui_panel_and_prefab_library_plan_2026-04-05.md`](gillijimproject_refactor/plans/mdxviewer_ui_panel_and_prefab_library_plan_2026-04-05.md)
5. [`gillijimproject_refactor/docs/VLM_DATASET_EXPORTER.md`](gillijimproject_refactor/docs/VLM_DATASET_EXPORTER.md)
6. [`gillijimproject_refactor/docs/VLM_Training_Guide.md`](gillijimproject_refactor/docs/VLM_Training_Guide.md)
7. [`gillijimproject_refactor/memory-bank/activeContext.md`](gillijimproject_refactor/memory-bank/activeContext.md)
8. [`gillijimproject_refactor/memory-bank/progress.md`](gillijimproject_refactor/memory-bank/progress.md)
9. [`gillijimproject_refactor/memory-bank/data-paths.md`](gillijimproject_refactor/memory-bank/data-paths.md)
10. [`gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py`](gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py)

## Goal

Make one concrete planning step toward a V7.4 system that understands WoW terrain as a hierarchical authored canvas, with trustworthy supervision and reusable authored-pattern priors.

## Non-Negotiable Constraints

- treat the world as a hierarchical authored canvas, not only a bag of disconnected tiles
- preserve explicit provenance and completeness metadata in the dataset contract
- do not describe all liquid masks as valid surface supervision when below-terrain liquid cases still exist
- do not respond with arbitrary-augmentation suggestions as the main fix
- do not recommend one giant monolithic model as the default first implementation step
- keep the long-term ownership direction aligned with [`wow_viewer_ml_tool_suite_cutover_plan_2026-04-10.md`](gillijimproject_refactor/plans/wow_viewer_ml_tool_suite_cutover_plan_2026-04-10.md)

## Current Working Assumptions

- the next gain comes from dataset truth ownership, duplicate-concept control, alpha-brush harvesting, and liquid/object supervision sanity
- a shared encoder with staged heads or follow-on consumers is preferred over many disconnected tiny models and also preferred over one immediate everything-model
- alpha-mask layers should be modeled as authored reusable patterns with paired 2D and 3D structure
- a detail/refiner tier may be justified, but only after the base world-structure model is trained on trustworthy data

## What The Work Must Produce

1. the exact next V7.4 seam to plan
2. the files or plans that should own it
3. the dataset truths or assumptions it depends on
4. the real-data validation needed
5. what remains out of scope for that slice
6. what continuity and prompt surfaces must be updated afterward

## Deliverables

Return all items:

1. the current canvas-aware boundary you are assuming
2. the single next slice you would land
3. why that slice is the correct next step
4. exact files or contracts to change later in code mode
5. exact validation to run
6. what you are explicitly not claiming yet

## First Output

Start with:

1. the single biggest current dataset or supervision flaw still blocking a trustworthy V7.4 base model
2. whether that flaw belongs to dataset contract, curation, brush harvesting, liquid sanity, or model tiering
3. the narrowest next slice to close it
4. what proof would make that slice real

