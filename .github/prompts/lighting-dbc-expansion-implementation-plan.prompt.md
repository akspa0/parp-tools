---
description: "Expand MdxViewer lighting data coverage beyond the current Light and LightData subset so world lighting, fog, and environment context are close enough to support PM4 object-variant matching."
name: "Lighting DBC Expansion Implementation Plan"
argument-hint: "Optional map, time-of-day symptom, light table, fog issue, or environment mismatch to prioritize"
agent: "agent"
---

Expand the lighting-data pipeline in `gillijimproject_refactor/src/MdxViewer` so scene lighting is materially closer to the client behavior that PM4 object-variant matching depends on.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/memory-bank/data-paths.md`
4. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
5. `gillijimproject_refactor/src/MdxViewer/Terrain/LightService.cs`
6. `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`
7. `gillijimproject_refactor/src/MdxViewer/Rendering/SkyDomeRenderer.cs`

## Goal

Broaden the viewer lighting model beyond the current `Light` + `LightData` subset so ambient, direct, sky, fog, and related environment context are no longer the main reason object variants look wrong during PM4 matching.

## Why This Slice Exists

- Current `LightService` already interpolates useful core fields, but it is still a partial environment model.
- PM4 variant matching depends on more than raw geometry; incorrect lighting can make the wrong asset look plausible.
- Lighting expansion must happen after material parity work so newly extracted material behavior has better scene context.

## Scope

- In scope:
  - identifying which DBC/DB2 tables or record families the active viewer still ignores
  - expanding `LightService` and adjacent consumers to use the next verified lighting/environment fields
  - keeping time-of-day interpolation and local/global blending coherent as coverage expands
  - fog and sky-color data when they are part of the same lighting contract
- Out of scope unless direct evidence requires it:
  - broad terrain renderer changes
  - speculative atmospheric post-processing
  - skybox asset parity that belongs in the next dedicated slice

## Non-Negotiable Constraints

- Do not invent cinematic lighting behavior that is not backed by client data.
- Do not claim parity from reading tables alone; note what is decoded, consumed, and actually used by rendering.
- Keep lighting data flow traceable from table decode to runtime scene state.
- Update memory-bank files in the same change set when decoded coverage materially changes.

## Required Implementation Order

1. Inventory what `LightService` already loads, interpolates, and exposes.
2. Identify the next missing lighting/environment record families that materially affect visible world output.
3. Confirm how those records are represented in the available client data for the active build families.
4. Add the smallest credible expansion to decode and propagate those values.
5. Wire the new values into the runtime scene only where the consumer behavior is clear.
6. Record the remaining uncovered lighting chain explicitly in memory-bank files.

## Investigation Checklist

- Name the exact tables/records already consumed.
- Name the exact tables/records still ignored but likely relevant.
- Confirm whether the missing data affects:
  - direct light
  - ambient light
  - sky top / horizon colors
  - fog color / density / distance
  - time-of-day transition behavior
  - local light zone blending
- Distinguish between "decoded but unused" and "not decoded at all."

## Validation Rules

- Build the changed viewer solution.
- If you do not run runtime validation on real data, say so explicitly.
- If automated tests are not added or run, say so explicitly.
- Do not call lighting parity complete unless the remaining uncovered data path is explicitly small and justified.

## Deliverables

Return all items:

1. exact lighting/environment data added
2. files changed and why
3. which tables or record families are now decoded and consumed
4. what still remains uncovered
5. build status
6. automated-test status
7. runtime-validation status
8. memory-bank updates made

## First Output

Start with:

1. what `LightService` currently covers
2. the biggest missing lighting/environment slice still affecting visible output
3. the next table or record family to implement
4. what files will be changed first