---
description: "Route wow-viewer dataset and training recovery work to the right Codex prompt. Use when the task is getting v7.5.1 grounded terrain training trustworthy again, aligning v7.6 paired reconstruction to the structured datasets workflow, auditing corpus truth and manifests, or deciding the fastest sequence to a first real retrain."
name: "wow-viewer Training Plan Set"
argument-hint: "Describe the model line, corpus problem, training blocker, or proof target you want to attack next"
agent: "codex"
---

Choose the right detailed prompt for the staged `wow-viewer` training and dataset-truth recovery path.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/memory-bank/data-paths.md`
4. `gillijimproject_refactor/docs/VLM_Training_Guide.md`
5. `gillijimproject_refactor/docs/ML_DATASET_GROUNDING.md`
6. `gillijimproject_refactor/docs/v75-model-architecture-guide.md`
7. `gillijimproject_refactor/docs/v76-model-architecture-guide.md`
8. `gillijimproject_refactor/docs/v76-output-dataset-spec.md`
9. `gillijimproject_refactor/plans/wow_viewer_dataset_builder_tool_plan_2026-04-14.md`
10. `wow-viewer/README.md`
11. `.codex/prompts/wow-viewer-dataset-builder-plan.md`
12. `AGENTS.md`

## Goal

Route the current request to the correct ordered prompt in `.codex/prompts/wow-viewer-training/` so `v7.5.1` and `v7.6` stop sharing one vague "training cleanup" bucket and instead move through concrete corpus-truth and model-line-specific recovery slices.

## Current Working Assumptions

- `datasets/` is the authoritative corpus surface.
- `v7.5.1` is the active grounded terrain-regression line and should be treated as the first trustworthy retrain target.
- `v7.6` is a separate paired reconstruction branch and should not silently inherit `v7.5.1` supervision assumptions.
- the currently shipped `wow-viewer` converter ML commands (`ml-corpus`, `ml-audit-signals`, `ml-harvest-brushes`, `ml-generate-controls`, and `ml-repair-normalmaps`) are the active compatibility bridge for corpus export and audit proof today, not the final long-range owner of shared dataset semantics.
- the long-range ownership target is still shared `wow-viewer` core plus a dedicated dataset-builder surface, with training scripts and cache builders treated as downstream consumers that must be realigned to that contract.
- training or inference workflow planning must respect the Bring Your Own Data policy: prefer reproducible local commands, manifests, and validation notes over any assumption that corpora, trained weights, or predicted outputs ship with the toolchain.

## Ordered Prompts

- `wow-viewer-training/01-shared-dataset-truth-gates.md`
- `wow-viewer-training/02-v75-grounded-training-recovery.md`
- `wow-viewer-training/03-v76-paired-training-alignment.md`

## Routing Rules

- Use `01-shared-dataset-truth-gates.md` when the real blocker is stale export roots, missing manifests or metadata rows, missing brush or control coverage, uncertainty about `datasets/` authority, ambiguity about fixed dataset or client roots, or a need to prove the corpus is trustworthy before any training run means anything.
- Use `02-v75-grounded-training-recovery.md` when the real blocker is getting the active grounded `v7.5.1` line back to a full, repeatable retrain over structured data with brush-aware sampling and `terrain_only_minimap` truth intact.
- Use `03-v76-paired-training-alignment.md` when the real blocker is the paired `v7.6` branch still leaning on hardcoded legacy cache roots, loose-file inference assumptions, or a training path that has not been brought under the structured dataset workflow.
- Start with `01-shared-dataset-truth-gates.md` unless the user already has a fresh, audited corpus with current manifest plus metadata plus signal proof and is asking about a clearly model-line-specific issue.

## Deliverables

Return all items:

1. the best next prompt to run
2. why it is the correct slice now
3. which ordered prompt should follow after it
4. what concrete repo, script, and dataset scope the next slice should include
5. what should stay out of scope for the next slice
6. what proof level is realistic for that slice

## First Output

Start with:

1. the exact training problem you think the user is trying to solve
2. the single best next prompt from the ordered set
3. the narrow proof that would make that slice real
4. what you are explicitly not claiming yet
