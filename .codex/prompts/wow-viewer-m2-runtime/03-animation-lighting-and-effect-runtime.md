---
description: "Implement or plan the M2 runtime slice for remaining animated-state ownership after the first parser/evaluator pass is real. Use when animated bone/skinning solve, sequence-driven material/light application, model-local diffuse/emissive semantics, or residual effect runtime behavior still live only in notes, inspect output, or compatibility code."
name: "wow-viewer M2 Runtime 03 Animation Lighting And Effect Runtime"
argument-hint: "Optional sequence family, lighting seam, effect recipe, or real asset to prioritize"
agent: "codex"
---

Implement or plan the wow-viewer M2 runtime slice that finishes animated runtime ownership after the first parser/evaluator baseline.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
4. `gillijimproject_refactor/plans/wow_viewer_m2_runtime_plan_2026-03-31.md`
5. `gillijimproject_refactor/src/MdxViewer/Rendering/MdxAnimator.cs`
6. `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs`
7. `wow-viewer/README.md`
8. `AGENTS.md`

## Goal

Complete the remaining animation/runtime state that the native client treats as first-class inside wow-viewer-owned contracts:

- animated bone pose and skinned-vertex application
- applying already-evaluated scalar/texture/material/light state into a real render consumer
- model-local diffuse/emissive evaluation in the real runtime path
- residual effect/combiner runtime semantics beyond simple recipe classification

## Current Validated Baseline

- external `%04d-%02d.anim` choose/load and alias ready-state ownership are already real
- effect-recipe classification is already real
- first-party animated block parsing/evaluation for colors, texture weights, texture transforms, and lights is already real
- `WowViewer.Tool.Inspect m2 inspect --time-ms` can already print evaluated animated runtime state on real assets

## Non-Negotiable Constraints

- Do not hide lighting/effect state inside global renderer fields.
- Do not claim particle/ribbon scene submission closure from this slice.
- Keep alias chains and readiness state explicit.
- Keep native effect/combiner names labeled as behavior-recovery evidence, not raw format terms.
- Do not widen this slice into a world-runtime refactor.
- Do not spend this slice re-planning already-landed external animation loading or inspect-only evaluator ownership as if they are missing.

## What The Work Must Produce

1. the exact wow-viewer contracts for animated bone/state application and runtime consumption
2. the exact files that should own render-consumer application of effect recipes and lighting evaluation
3. the narrowest real proof that animated state is no longer inspect-only or viewer-local
4. the explicit boundary that still remains for slice 04 scene submission

## Deliverables

Return all items:

1. the exact animation/lighting/effect seam to implement
2. why it is the right next step
3. exact files to change
4. exact validation to run
5. what should stay out of scope
6. which continuity files must be updated afterward

## First Output

Start with:

1. the animation/lighting/effect boundary you are assuming now
2. the single first runtime seam you would land
3. the narrowest proof that would make that seam real
4. what you are explicitly not claiming yet