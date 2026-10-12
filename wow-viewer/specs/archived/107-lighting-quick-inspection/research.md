# Research

**Decision**: Use the evaluated fog range as the scene culling envelope. `WorldScene` already converts classic fixed-unit fog through `TerrainLightingMath.ComputeClientFogRange`; the visible defect is only `ViewerApp.GetSceneFarPlane`, which forces at least 6000 units despite a lower active FogEnd.

**Hover decision**: A ray intersection selects one nearest placement and supports exact-path text. Screen brush candidates can overlap and carry an `AdditionalHitCount`; they are useful for click selection but not reliable enough for an “exact” hover card.
