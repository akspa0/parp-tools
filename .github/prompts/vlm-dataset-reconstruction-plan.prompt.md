---
description: "Plan or implement the next provenance-first VLM dataset slice for real-map terrain reconstruction in gillijimproject_refactor. Use when the task is vlm-export, dataset schema, real-map curation, per-tile provenance, missing-layer reconstruction, v7-like model prep, shadow/object association, height/normal/MCCV export, or train/val/test split planning."
name: "VLM Dataset Reconstruction Plan"
argument-hint: "Optional map family, exporter seam, dataset schema gap, reconstruction target, or curation slice to prioritize"
agent: "agent"
---

Plan or implement the next narrow VLM dataset reconstruction slice without drifting into generic model-training advice.

If the ask is actually about repairing the `development` map outputs themselves rather than building the dataset/training contract, use `development-repair-implementation-plan.prompt.md` instead.

## Read First

1. `gillijimproject_refactor/plans/vlm_dataset_reconstruction_plan_2026-03-31.md`
2. `gillijimproject_refactor/docs/VLM_DATASET_EXPORTER.md`
3. `gillijimproject_refactor/docs/VLM_Training_Guide.md`
4. `gillijimproject_refactor/memory-bank/data-paths.md`
5. `gillijimproject_refactor/memory-bank/activeContext.md`
6. `gillijimproject_refactor/memory-bank/progress.md`
7. `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDatasetExporter.cs`
8. `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDataModels.cs`
9. `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM/HeightmapBakeService.cs`
10. `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmShadowAssociationService.cs`

## Goal

Make one concrete step toward a real-map, provenance-rich dataset contract that can support a v7-like model for reconstructing missing terrain/map layers.

## Non-Negotiable Constraints

- use real map exports from real client data as the supervision corpus
- treat `test_data/development/World/Maps/development` as the reconstruction target/evaluation corpus, not the only teacher corpus
- do not describe old v6/v30 docs as the current schema if the exporter now emits more channels
- do not collapse observed channels and generated fallback channels into one unlabeled dataset contract
- do not jump to generic finetuning work before the schema, manifest, and curation rules are explicit

## Current Working Assumptions

- `VlmDatasetExporter` already emits more than the current docs advertise, including chunk heights, local/global heightmaps, normals, MCCV, raw shadow bits, shadow analysis, alpha masks, liquids, objects, WDL, and binary tile output
- the likely highest-value next slice is dataset manifest/provenance/categorization work, not another one-off image export tweak
- future training should be profile-aware and map-aware, with train/val/test splits built by map family rather than random tile shuffling

## What The Work Must Produce

1. the exact dataset or exporter seam to change next
2. the real files that should own that seam
3. the real-data validation needed for the seam
4. the map/profile provenance rules that must remain explicit
5. what should stay out of scope for this slice
6. which continuity/docs/prompt files must be updated afterward

## Deliverables

Return all items:

1. the current VLM dataset boundary you are assuming
2. the single next slice you would land
3. why that slice is the right next step
4. exact files to change
5. exact validation to run
6. what you are explicitly not claiming yet

## First Output

Start with:

1. whether the current exporter already has the channels needed for the target slice
2. the single biggest provenance/schema gap still blocking trustworthy training data
3. the narrowest next slice to close that gap
4. what real-map validation would prove the slice is real