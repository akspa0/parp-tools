---
description: "Implement or plan the M2 runtime slice for external animation ownership, model-local lighting, and effect state. Use when `%04d-%02d.anim`, alias chains, diffuse/emissive evaluation, or combiner/effect routing still live only in native notes or compatibility code."
name: "wow-viewer M2 Runtime 03 Animation Lighting And Effect Runtime"
argument-hint: "Optional sequence family, lighting seam, effect recipe, or real asset to prioritize"
agent: "codex"
---

Implement or plan the wow-viewer M2 runtime slice that owns animation-file state, model-local lighting, and effect/combiner runtime decisions.

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

Move animation/runtime state that the native client treats as first-class into wow-viewer-owned contracts:

- external `%04d-%02d.anim` files and alias chains
- sequence readiness state
- animated scalar/texture/material state
- model-local diffuse/emissive evaluation
- explicit effect/combiner recipe state instead of ad hoc renderer-local choices

## Non-Negotiable Constraints

- Do not hide lighting/effect state inside global renderer fields.
- Do not claim particle/ribbon scene submission closure from this slice.
- Keep alias chains and readiness state explicit.
- Keep native effect/combiner names labeled as behavior-recovery evidence, not raw format terms.
- Do not widen this slice into a world-runtime refactor.

## What The Work Must Produce

1. the exact wow-viewer contracts for animation file ownership and runtime state
2. the exact files that should own effect recipes and lighting evaluation
3. the narrowest real proof that animated/effect state is library-owned instead of viewer-local
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