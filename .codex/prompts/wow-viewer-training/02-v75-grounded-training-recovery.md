---
description: "Plan or implement the v7.5.1 grounded training recovery slice. Use when the goal is to get the active terrain-regression line back to a repeatable, trustworthy retrain over structured datasets with brush-aware sampling and terrain_only_minimap truth intact."
name: "wow-viewer Training 02 v7.5.1 Grounded Training Recovery"
argument-hint: "Optional v7.5.1 blocker, trainer script, sampler problem, or proof target to prioritize"
agent: "codex"
---

Plan or implement the `v7.5.1` recovery slice: get the active grounded model line back to a full retrain path that we can trust.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/docs/VLM_Training_Guide.md`
4. `gillijimproject_refactor/docs/ML_DATASET_GROUNDING.md`
5. `gillijimproject_refactor/docs/v75-model-architecture-guide.md`
6. `gillijimproject_refactor/scripts/export_ml_corpus.ps1`
7. `wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs`
8. `.codex/prompts/wow-viewer-training/01-shared-dataset-truth-gates.md`
9. `AGENTS.md`

## Goal

Recover the active `v7.5.1` grounded terrain-regression line so a new training run is driven by fresh structured data, the intended supervision channels, and a documented acceptance gate instead of tribal knowledge.

## Current Concrete Baseline

- `v7.5.1` is the active grounded terrain-regression line.
- the current architecture expects 13 input channels and 2 outputs.
- `terrain_only_minimap` is the preferred active image truth surface.
- brush-aware sampling is part of the intended supervision path.
- early corpus anchors such as `0.5.3`, `0.5.5`, and `0.6.0` matter for continuity, but they do not replace a fresh rerun proof.
- the exporter and trainer path have partial proof already, but the line still needs a trustworthy end-to-end retrain.

## Non-Negotiable Constraints

- Keep `v7.5.1` separate from `v7.6`; do not solve paired reconstruction drift by smearing the lines together.
- `terrain_only_minimap` and brush imprint coverage are active supervision, not optional nice-to-haves.
- prefab channels remain deferred or experimental unless the user explicitly asks to promote them.
- Do not hide stale-corpus problems behind trainer-side hacks or hardcoded fallbacks.
- Prefer fixing dataset readiness and trainer configuration clarity over speculative architecture churn.
- A successful slice must end with an explicit runbook and acceptance gate, not only code cleanup.

## What The Work Must Produce

1. the exact `v7.5.1` blockers that still prevent a trustworthy rerun
2. the exact corpus assumptions the trainer and sampler should consume
3. the exact scripts, configs, or loaders that need cleanup or realignment
4. the minimum command sequence for a fresh export, audit, and train launch
5. the run-level acceptance gate for saying `v7.5.1` is back in a trustworthy state
6. the next backlog items after the first real retrain succeeds

## Deliverables

Return all items:

1. current blocker map
2. fastest trustworthy retrain sequence
3. repo or script scope to change
4. acceptance criteria for the first real run
5. evidence to capture from the run
6. what should stay out of scope for this slice

## First Output

Start with:

1. the exact reason `v7.5.1` is not trustworthy enough today
2. the single highest-leverage recovery slice to do first
3. the first concrete commands or scripts that should run after the slice lands
4. what you are explicitly not claiming yet
