---
description: "Plan or implement the shared dataset-truth recovery slice for wow-viewer training. Use when the real blocker is stale export roots, uncertain dataset authority, missing manifests, weak brush or control coverage, or a need to prove that v7.5.1 and v7.6 are training against the right corpus before spending more GPU time."
name: "wow-viewer Training 01 Shared Dataset Truth Gates"
argument-hint: "Optional dataset root, manifest problem, export wrapper, or audit symptom to prioritize"
agent: "codex"
---

Plan or implement the dataset-truth gate that both model lines depend on: make the structured `datasets/` surface trustworthy before retraining anything.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/docs/VLM_Training_Guide.md`
4. `gillijimproject_refactor/docs/ML_DATASET_GROUNDING.md`
5. `gillijimproject_refactor/plans/wow_viewer_dataset_builder_tool_plan_2026-04-14.md`
6. `gillijimproject_refactor/scripts/export_ml_corpus.ps1`
7. `gillijimproject_refactor/scripts/build_minimal_ml_manifest.py`
8. `wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs`
9. `.codex/prompts/wow-viewer-dataset-builder-plan.md`
10. `AGENTS.md`

## Goal

Define or land the minimum corpus-truth gates that make a new `v7.5.1` or `v7.6` training run worth trusting.

## Current Concrete Problem

- dataset ownership is supposed to live under structured `datasets/`, but the practical workflow still risks drifting across wrappers, ad hoc exports, and stale local roots.
- `v7.5.1` depends on grounded supervision such as brush and `terrain_only_minimap` coverage that can silently go missing or stale.
- `v7.6` still has known legacy-root drift in its cache path, so it cannot be trusted until the shared corpus contract is explicit.
- GPU time is easy to waste if the manifest, metadata rows, brush harvest, and control-signal coverage are not validated first.

## Non-Negotiable Constraints

- Do not treat a training launch as proof if the corpus root, manifest contract, and audit results are still ambiguous.
- `datasets/` is the authoritative export surface; loose legacy dataset roots are reference inputs only until explicitly migrated.
- Prefer `wow-viewer` converter commands for audit and export truth when they already exist.
- `export_ml_corpus.ps1` and related scripts are wrappers or orchestration helpers, not the canonical owners of dataset semantics.
- Keep this slice focused on corpus truth, manifests, and audit signals; do not turn it into a model-architecture rewrite.
- Respect BYOD constraints. Plan around reproducible local commands, manifests, and validation notes, not redistributed corpora or trained outputs.

## What The Work Must Produce

1. the exact authoritative dataset roots and naming contract
2. the exact manifest and metadata files that must exist before training
3. the exact audit commands and checks for signal, brush, and control coverage
4. the refresh or repair sequence for stale or partial exports
5. the pass or fail gate for saying a corpus is ready for `v7.5.1`
6. the pass or fail gate for saying a corpus is ready for `v7.6`
7. the follow-on handoff to the model-line-specific prompt

## Deliverables

Return all items:

1. authoritative dataset contract
2. current drift or ambiguity map
3. audit and repair command sequence
4. missing-truth risk list
5. first trustworthy proof target
6. what should stay out of scope for this slice

## First Output

Start with:

1. the exact dataset-truth problem you think is blocking real training
2. the single most important corpus gate that must be made explicit first
3. the fastest audit or repair sequence to prove or disprove readiness
4. what you are explicitly not claiming yet
