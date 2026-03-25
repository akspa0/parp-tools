# Enhanced Terrain Shader And Lighting Prompt

Use this prompt in a fresh planning chat when working on the enhanced-quality terrain renderer path, shader-family translation strategy, and lighting-model expansion for `MdxViewer`.

## Companion Prompts

Use these companion prompt files when the work should be narrower than the full umbrella brief:

- `plans/enhanced_renderer_plan_set_2026-03-25.md`
- `plans/enhanced_renderer_architecture_prompt_2026-03-25.md`
- `plans/enhanced_terrain_first_slice_prompt_2026-03-25.md`
- `plans/shader_family_and_lighting_roadmap_prompt_2026-03-25.md`

## Prompt

Design a concrete implementation plan for an enhanced-quality rendering path in `gillijimproject_refactor/src/MdxViewer` that improves terrain visual quality and prepares the renderer for client-family shader translation, while preserving the current historical fallback path.

The plan must assume:

- the current viewer already has a `Render Quality` window, but it is intentionally narrow and currently covers sampler-state quality plus runtime multisample toggling
- the current terrain renderer is still a generic base-plus-overlay GLSL path with decoded alpha maps and shadow maps, not a client-faithful terrain shader-family implementation
- the active viewer already has reverse-engineering evidence that older and later WoW clients used stronger shader specialization, including:
	- terrain families such as `psTerrain` / `psSpecTerrain`
	- `Model2.bls` for later `2.x`
	- multiple map-object pixel-shader families such as `MapObjTransDiffuse.bls` and `MapObjTransSpecular.bls`
- the active viewer does **not** currently have the runtime light/material state needed to claim client-faithful parity just by swapping shaders
- `LightService` remains a simplified nearest-zone DBC interpolator rather than a full engine-equivalent world-light system
- this work is for an **enhanced-quality option**, not for replacing the historical renderer or over-claiming exact parity immediately

The plan must treat shader translation as reconstruction of shader families and runtime inputs for the active renderer, not as a naive attempt to execute original Blizzard shader payloads directly.

## What The Plan Must Produce

1. A target architecture for `Historical` versus `Enhanced` render modes.
2. A first practical implementation slice for terrain only.
3. A render-quality UI/settings expansion plan.
4. A terrain-lighting expansion plan for `LightService` and scene plumbing.
5. A shader-family translation strategy for later terrain, WMO/map-object, and Model2 work.
6. A risk register that explicitly separates:
	- terrain decode risk
	- terrain shading risk
	- lighting-model risk
	- later shader-family parity risk
7. A real-data validation strategy that does not over-claim.

## Required Constraints

- Do not propose a vague “replace the whole renderer” rewrite.
- Keep `Historical` as the default path.
- Keep the current terrain decode pipeline unchanged in the first implementation slice.
- Do not change MCAL/MCLY decode rules as part of the first enhanced rendering slice.
- Preserve the current split between terrain decode/storage and terrain shading.
- Do not claim that shader-family reconstruction alone will make lighting “correct” without corresponding runtime light-state work.
- Do not treat archived code, library tests, or synthetic fixtures as proof for the active viewer.
- Favor a vertical slice that can actually land without destabilizing the active viewer.

## Terrain Guardrails

The plan must respect the active terrain risk guidance:

- treat commit `343dadfa27df08d384614737b6c5921efe6409c8` as the baseline when terrain rendering behavior materially changes
- keep terrain alpha decode and terrain shading as separate seams
- preserve current layer indexing and split-ADT assumptions
- do not add relaxed alpha heuristics or reinterpret `MCLY` flags as part of enhanced shading work

The plan must explicitly keep the first enhanced slice away from changes in:

- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
- `src/MdxViewer/Terrain/TerrainChunkData.cs`
- `src/WoWMapConverter/WoWMapConverter.Core/Formats/LichKing/Mcal.cs`

unless the planner can justify that with a separate decode-risk section.

## Current Code Seams To Use

Use the active code structure and current practical seams, especially:

- `src/MdxViewer/ViewerApp_RenderQuality.cs`
- `src/MdxViewer/Rendering/RenderQualitySettings.cs`
- `src/MdxViewer/Terrain/TerrainRenderer.cs`
- `src/MdxViewer/Terrain/LightService.cs`
- `src/MdxViewer/Terrain/WorldScene.cs`
- `src/MdxViewer/Rendering/ModelRenderer.cs`
- `src/MdxViewer/Rendering/WmoRenderer.cs`
- `src/MdxViewer/Terrain/LiquidRenderer.cs`

The plan should assume the current terrain renderer already has a usable seam for:

- texture filtering upgrades
- texture upload hooks
- terrain shader replacement or dual-path selection
- alpha/shadow texture reuse

but does **not** yet have a robust material/shader-family translation layer.

## High-Priority Questions The Plan Must Answer

1. How should `Historical` and `Enhanced` terrain rendering coexist without making regressions harder to diagnose?
2. What should the first enhanced terrain shader do, and just as importantly, what should it explicitly *not* do yet?
3. How should anisotropic filtering, mip bias, and terrain-specific sampling quality fit into the existing `Render Quality` panel?
4. What minimum `LightService` expansion is required before the enhanced terrain path produces meaningfully better lighting?
5. When should later shader-family work branch into:
	- terrain
	- WMO / map-object
	- Model2 / later `2.x`
	- liquid
	- particles
6. What parts of client shader/material behavior are blocked on more reverse engineering instead of immediate implementation?

## Suggested Deliverable Structure

1. Current-state inventory
2. Historical vs Enhanced target architecture
3. First implementation slice
4. Terrain lighting expansion plan
5. Shader-family translation strategy
6. Risk register
7. Real-data validation plan
8. Later milestones and blocked seams

## Validation Rules

- Be explicit about what is build-validated only versus runtime-validated.
- If no automated tests are proposed, say that explicitly and give the real-data validation step instead.
- Do not describe the enhanced path as client-faithful parity unless the corresponding runtime light/material state is also addressed and validated.
- For terrain-facing slices, require real-data validation against the fixed development terrain paths before claiming safety.

## Fixed Data Reminder

Use the fixed paths already documented in the repo memory bank, especially:

- `test_data/development/World/Maps/development`
- `test_data/WoWMuseum/335-dev/World/Maps/development`

Do not ask for alternate paths unless those fixed paths are genuinely missing.