# Enhanced Terrain First Slice Prompt

Use this prompt in a fresh planning chat when the goal is to produce a landable first implementation plan for enhanced terrain rendering only.

## Prompt

Design a concrete first implementation slice for enhanced terrain rendering in `gillijimproject_refactor/src/MdxViewer`.

This first slice must improve terrain presentation in a meaningful way while staying narrow enough to land without destabilizing the active historical renderer or changing terrain decode semantics.

## The First Slice Must Produce

1. A precise scope for the first landable change.
2. A file-by-file change map.
3. A statement of what the slice explicitly will not do yet.
4. A validation checklist using real data.
5. A rollback strategy if the enhanced path misbehaves.

## Required Assumptions

- the first slice is terrain-only
- the first slice may improve texture sampling and terrain shading behavior
- the first slice must not modify MCAL decode rules, MCLY interpretation, or terrain layer indexing
- `TerrainRenderer` and `RenderQualitySettings` are the main code seams
- `LightService` may be expanded only to the minimum degree needed for visible improvement, not full parity

## Required Constraints

- Keep `Historical` terrain rendering intact and selectable.
- Do not change:
	- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
	- `src/MdxViewer/Terrain/TerrainChunkData.cs`
	- `src/WoWMapConverter/WoWMapConverter.Core/Formats/LichKing/Mcal.cs`
  as part of the first slice unless there is a separate decode-risk justification.
- Do not promise specular parity, full material parity, or full day/night parity in the first slice.
- Favor one vertical slice over several half-finished rendering ideas.

## Questions The Plan Must Answer

1. Should the first slice include anisotropic filtering, mip bias, terrain-specific sampler controls, or only a shader split?
2. What is the minimum enhanced terrain shader worth landing first?
3. What new render-quality UI controls are justified immediately versus later?
4. What minimum light inputs are required so the enhanced terrain does not look like a cosmetic post-process?
5. What real-data screenshots or runtime checks are required before calling the slice successful?

## Expected Deliverables

1. Scope statement
2. File-by-file implementation map
3. Acceptance criteria
4. Risk register
5. Real-data validation checklist

## Validation Rules

- Build success alone is not enough.
- Require runtime validation against the fixed `development` terrain paths.
- Separate shader bugs from decode bugs in the validation notes.
- If no automated tests are added, say that explicitly.