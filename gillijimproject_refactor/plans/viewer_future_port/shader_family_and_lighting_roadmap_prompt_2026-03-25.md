# Shader Family And Lighting Roadmap Prompt

Use this prompt in a fresh planning chat when the goal is to plan the work after the first enhanced terrain slice: lighting-model expansion, shader-family reconstruction, and domain rollout beyond terrain.

## Prompt

Design a roadmap for taking `MdxViewer` from an initial enhanced terrain slice to a broader enhanced renderer with stronger shader-family reconstruction and materially better lighting behavior.

The roadmap must be realistic about what is known from reverse engineering versus what still requires more investigation.

## The Roadmap Must Cover

1. The next `LightService` expansion stages after the first terrain slice.
2. How terrain shader-family work should evolve beyond a first enhanced shader.
3. When and how to branch into WMO/map-object shader families.
4. When and how to branch into Model2 or later `2.x` material/shader families.
5. What liquid and particle work should wait until better light/material state exists.
6. Which parts are blocked on more reverse engineering.

## Required Assumptions

- reverse-engineering evidence already suggests terrain and object rendering used specialized shader families rather than one generic material path
- current runtime light/material state is still simplified
- parity claims should lag behind implementation until real-data validation exists
- terrain should remain the first enhanced domain, not the final one

## Required Constraints

- Do not collapse terrain, WMO, Model2, liquid, and particles into one milestone.
- Separate “safe now” work from “high risk” work and from “blocked on more RE” work.
- Do not treat shader translation as sufficient if the required runtime inputs are still missing.
- Keep the roadmap incremental enough that the active viewer can keep shipping intermediate wins.

## Expected Deliverables

1. Milestone sequence
2. Dependency map between lighting and shader-family work
3. Safe-now / high-risk / blocked sections
4. Runtime validation expectations per milestone
5. Explicit statements of what should not be claimed yet

## Current Seams And Domains

- terrain: `src/MdxViewer/Terrain/TerrainRenderer.cs`
- lighting: `src/MdxViewer/Terrain/LightService.cs`
- scene plumbing: `src/MdxViewer/Terrain/WorldScene.cs`
- model rendering: `src/MdxViewer/Rendering/ModelRenderer.cs`
- WMO rendering: `src/MdxViewer/Rendering/WmoRenderer.cs`
- liquid: `src/MdxViewer/Terrain/LiquidRenderer.cs`

## Validation Rules

- Require build validation for milestone landings.
- Require runtime validation for terrain-facing and lighting-facing claims.
- Do not describe a later milestone as client-faithful unless the required material inputs, light inputs, and scene behavior are all validated together.