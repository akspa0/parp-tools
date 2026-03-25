# Enhanced Renderer Architecture Prompt

Use this prompt in a fresh planning chat when the goal is to define the runtime architecture for `Historical` versus `Enhanced` rendering in `gillijimproject_refactor/src/MdxViewer`.

## Prompt

Design a concrete architecture plan for adding an `Enhanced` render path to `MdxViewer` while preserving the current `Historical` path as the default and most stable renderer.

The plan must focus on runtime boundaries and ownership, not on speculative visual goals alone.

## The Architecture Plan Must Cover

1. Where render mode selection should live.
2. How render mode settings should be stored, surfaced, and persisted.
3. Which systems should remain shared between modes.
4. Which systems should branch by mode.
5. How terrain, WMO/map-object, Model2, liquid, and particles should be staged into the new mode boundary over time.
6. How regressions should be isolated so that terrain decode bugs are not confused with enhanced shading bugs.

## Required Assumptions

- `Render Quality` currently exists and is the practical UI seam for user-facing render-mode controls.
- `TerrainRenderer` is the first meaningful enhanced-render candidate.
- `LightService` is not yet rich enough to support strong parity claims.
- the current renderer still uses broad generic GLSL paths instead of domain-specific shader families.
- the enhanced path should be additive, not a replacement rewrite.

## Required Constraints

- Keep `Historical` as the default mode.
- Avoid proposals that duplicate the whole scene graph or world-loading pipeline without need.
- Preserve terrain decode/storage contracts in the first architecture stage.
- Keep feature ownership readable: sampler quality, shader selection, lighting inputs, and material policy should not be mixed into one opaque settings object.
- Favor seams that let the user switch modes and compare behavior without changing loaded data.

## Expected Deliverables

1. A mode-boundary diagram in prose.
2. A list of shared versus mode-specific systems.
3. A settings/persistence plan.
4. A recommended order for branching renderer domains.
5. A risk register for architecture mistakes that would make debugging harder later.

## Current Seams To Use

- `src/MdxViewer/ViewerApp_RenderQuality.cs`
- `src/MdxViewer/Rendering/RenderQualitySettings.cs`
- `src/MdxViewer/Terrain/TerrainRenderer.cs`
- `src/MdxViewer/Terrain/WorldScene.cs`
- `src/MdxViewer/Rendering/ModelRenderer.cs`
- `src/MdxViewer/Rendering/WmoRenderer.cs`
- `src/MdxViewer/Terrain/LiquidRenderer.cs`

## Validation Rules

- Call out what can be validated by build only.
- Call out what requires runtime comparison in the fixed development terrain paths.
- Do not treat an architecture document as proof that later parity work is low risk.