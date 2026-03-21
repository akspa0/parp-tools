---
description: "Improve skybox and environment parity in MdxViewer so backdrop, horizon, and sky context stop misleading PM4 object-variant matching after material and lighting fixes are in place."
name: "Sky Environment Parity Implementation Plan"
argument-hint: "Optional skybox symptom, backdrop issue, horizon mismatch, or map/environment clue to prioritize"
agent: "agent"
---

Implement the skybox and environment parity slice in `gillijimproject_refactor/src/MdxViewer` after material and lighting work have established a more trustworthy base scene.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/memory-bank/data-paths.md`
4. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
5. `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`
6. `gillijimproject_refactor/src/MdxViewer/Rendering/SkyDomeRenderer.cs`
7. `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs`
8. `gillijimproject_refactor/src/MdxViewer/Terrain/LightService.cs`

## Goal

Make skybox, sky backdrop, and environment context believable enough that PM4 object-variant matching is not biased by the current procedural or heuristic sky behavior.

## Why This Slice Exists

- The active viewer already has a minimal skybox-backdrop path and a procedural sky dome.
- Those systems are useful recovery steps, but they are still too heuristic-driven for trustworthy final matching.
- Sky/environment parity should be implemented after material and lighting improvements, otherwise visual changes are hard to attribute correctly.

## Scope

- In scope:
  - skybox instance detection and selection rules
  - backdrop render ordering and camera anchoring behavior
  - interaction between skybox assets, procedural sky, and lighting-driven sky colors
  - environment context that visibly changes how objects are perceived at distance
- Out of scope unless direct evidence requires it:
  - unrelated model-material fixes
  - speculative weather systems
  - terrain shader rewrites with no clear sky/environment dependency

## Non-Negotiable Constraints

- Do not remove the current minimal sky path unless the replacement is ready in the same change set.
- Do not hardcode map-specific sky hacks and call that parity.
- Keep sky/environment behavior tied to decoded runtime data whenever possible.
- Update memory-bank files in the same change set when the sky/environment contract changes.

## Required Implementation Order

1. Inventory the current sky pipeline:
   - skybox model detection
   - skybox instance selection
   - `RenderSkyboxBackdrop(...)`
   - procedural `SkyDomeRenderer`
   - lighting-driven sky color overrides
2. Identify where the current path is heuristic or contradictory.
3. Decide which source of truth should win in each case:
   - asset skybox
   - lighting-derived sky colors
   - procedural fallback
4. Implement the smallest coherent parity slice that reduces visible contradiction without destabilizing the rest of the scene.
5. Record what still remains heuristic after the change.

## Investigation Checklist

- Confirm how skybox-like models are currently classified.
- Confirm whether multiple sky systems are active at once and which one visually dominates.
- Confirm how lighting data currently influences the sky dome and backdrop.
- Distinguish between:
  - missing sky data
  - present data but wrong precedence
  - correct data with wrong render ordering/state

## Validation Rules

- Build the changed viewer solution.
- If you do not run runtime validation on real data, say so explicitly.
- If automated tests are not added or run, say so explicitly.
- Do not call sky/environment parity complete if the scene still relies on unresolved heuristic conflicts.

## Deliverables

Return all items:

1. exact sky/environment behavior implemented
2. files changed and why
3. which sky/context conflicts were removed
4. what remains heuristic or incomplete
5. build status
6. automated-test status
7. runtime-validation status
8. memory-bank updates made

## First Output

Start with:

1. the current sky/environment path in the viewer
2. the biggest contradiction or heuristic currently affecting scene trustworthiness
3. the smallest coherent slice to land first
4. what files will be changed first