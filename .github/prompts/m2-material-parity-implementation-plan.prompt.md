---
description: "Implement the M2 material-parity slice in MdxViewer for transparent, cutout, and reflective surfaces so PM4 object-variant matching is visually trustworthy instead of guessed from flattened shader behavior."
name: "M2 Material Parity Implementation Plan"
argument-hint: "Optional symptom, model path, material type, blend mode, or reflective-surface clue to prioritize"
agent: "agent"
---

Implement the M2 material-parity slice in `gillijimproject_refactor/src/MdxViewer` with PM4 object-variant matching as the runtime goal.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/memory-bank/data-paths.md`
4. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
5. `.github/prompts/pre-release-m2-rendering-recovery.prompt.md`
6. `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs`
7. `gillijimproject_refactor/src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`
8. `gillijimproject_refactor/src/MdxViewer/Terrain/WorldAssetManager.cs`

## Goal

Make transparent, alpha-cutout, additive, and reflective M2-family surfaces render credibly enough that PM4-to-object variant matching is based on real material appearance instead of renderer guesswork.

## Why This Slice Exists

- PM4 matching quality is currently limited by scene trustworthiness, not only geometry placement.
- The active renderer already has partial support for:
  - blend-mode routing
  - alpha-cutout heuristics
  - premultiplied alpha
  - sphere environment mapping
  - basic specular
- Those pieces are still too flattened and heuristic-driven to trust for object variant decisions.

## Scope

- In scope:
  - `ModelRenderer` material-state extraction and pass routing
  - layer-level blend/depth/cull handling for M2-family assets
  - alpha-test versus blended-path correctness
  - reflective/env-mapped material handling when the source metadata supports it
  - texture binding and fallback behavior that directly affects visible material output
- Out of scope unless direct evidence forces it:
  - terrain shading
  - WMO-specific parity work beyond keeping shared behavior consistent
  - broad renderer architecture rewrites with no immediate effect on material correctness

## Non-Negotiable Constraints

- Do not treat build success as renderer parity.
- Do not collapse all non-opaque material behavior into one generic transparent path.
- Do not hide missing metadata behind new broad heuristics if the data can be extracted explicitly.
- Keep fixes aligned with the current viewer architecture; do not start a speculative engine rewrite.
- Update memory-bank files in the same change set when behavior or known risks materially change.

## Required Implementation Order

1. Inventory the current material decision path from extracted layer metadata into `ModelRenderer` uniforms and GPU state.
2. Separate at least these visible families before changing shader code:
   - opaque
   - alpha-key / cutout
   - blended transparency
   - additive
   - reflective / env-mapped
3. Confirm where the active path is flattening or overriding source material intent.
4. Land the smallest renderer/material extraction slice that improves one confirmed failure family without regressing the others.
5. Keep logging specific enough to show which pass and material mode a batch used.
6. Update memory-bank files with the exact parity slice landed and the remaining gaps.

## Investigation Checklist

- Verify how `ModelRenderer` derives:
  - effective blend mode
  - alpha-cutout usage
  - premultiply behavior
  - depth test and depth write state
  - cull mode
  - env-map usage
- Verify what `WarcraftNetM2Adapter` and related model paths expose versus what the renderer actually consumes.
- Compare one mostly opaque asset and one transparency/reflection-heavy asset before claiming the next slice is correctly chosen.
- Prefer evidence that names the exact missing behavior rather than generic "materials still wrong" summaries.

## Validation Rules

- Build the changed viewer solution.
- If you do not run runtime validation on real data, say so explicitly.
- If automated tests are not added or run, say so explicitly.
- Do not claim PM4 matching benefit unless the rendering change is actually visible on real assets.

## Deliverables

Return all items:

1. exact material/rendering behavior implemented
2. files changed and why
3. what material families were fixed versus still flattened
4. build status
5. automated-test status
6. runtime-validation status
7. memory-bank updates made

## First Output

Start with:

1. the current M2 material-parity gap in the active renderer
2. which material family is the best first slice
3. where the current pipeline is flattening source intent
4. what files will be changed first