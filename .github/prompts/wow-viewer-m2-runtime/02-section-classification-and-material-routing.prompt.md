---
description: "Implement or plan the second wow-viewer M2 runtime slice. Use when active section ownership, bone-palette remap, unresolved native flags such as `0x20` and `0x40`, or material/effect routing still collapse into generic geoset assumptions."
name: "wow-viewer M2 Runtime 02 Section Classification And Material Routing"
argument-hint: "Optional real asset, section flag, material family, or adapter seam to prioritize"
agent: "agent"
---

Implement or plan the M2 runtime slice that turns active skin-profile state into a real section/material ownership seam in `wow-viewer`.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
4. `gillijimproject_refactor/plans/wow_viewer_m2_runtime_plan_2026-03-31.md`
5. `gillijimproject_refactor/src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`
6. `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs`
7. `wow-viewer/README.md`
8. `.github/copilot-instructions.md`

## Goal

Create a wow-viewer-owned active section contract that preserves the native structural seams instead of flattening everything into generic geosets or loose material layers.

## Current Concrete Problem

- native `.skin` initialization copies/remaps section records into active runtime state
- unresolved native section flags such as `0x20` and propagated `0x40` still matter even though their final semantics are not closed
- the current compatibility path still tends to blend together section ownership, material routing, and renderer behavior too early

## Non-Negotiable Constraints

- Do not erase unresolved native flags just because their names are not closed yet.
- Do not collapse every section/batch into one generic renderable geoset model.
- Keep raw/native terminology distinct from local aliases or guesses.
- Do not claim final shader/effect parity from section-contract work alone.
- Keep this slice narrower than full animation or scene-submission ownership.

## What The Work Must Produce

1. the exact active-section and material-routing contracts to add in wow-viewer
2. the exact files that should own section remap, flag preservation, and material/effect recipe metadata
3. the real-asset proof that the section contract is not synthetic
4. the explicit unresolved items that must stay labeled research
5. the follow-on hook for slice 03 animation/lighting/effect state

## Deliverables

Return all items:

1. the exact section/material seam to implement
2. why it is the correct second slice
3. exact files to change
4. exact validation to run
5. what should stay out of scope
6. which continuity files must be updated afterward
7. which names are still research-only

## First Output

Start with:

1. the active section/material boundary you are assuming now
2. the single first section-routing seam you would land
3. the narrowest proof that would make that seam real
4. what you are explicitly not claiming yet