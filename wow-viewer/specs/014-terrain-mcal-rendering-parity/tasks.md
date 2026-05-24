# Tasks: Terrain MCAL Rendering Parity (Spec 014)

## Phase 1: Thread Shadow Map Through Runtime Chunk Data

- [x] T1.1: Add `byte[]? ShadowMap` property and constructor parameter to `WorldTerrainChunkData` (WowViewer.Core.Runtime)
- [x] T1.2: Pass `textureChunk?.ShadowMap` in `WorldTerrainTileBuilder.Read` line 87 (LK path)
- [x] T1.3: Alpha path defaults to null (no MCSH in alpha ADTs) — constructor default handles it
- [x] T1.4: `dotnet build` + `dotnet test` — no regressions (477 pass, 15 pre-existing failures)

## Phase 2: Fill Alpha Channel with Shadow Map

- [x] T2.1: Add shadow map write to `FillAlphaShadowSlice` channel 3 after the layer loop, matching MdxViewer `TerrainTileMeshBuilder.cs:289-300`
- [x] T2.2: Use edge-clamped index (clamp x/y to size-2) for shadow map source, matching MdxViewer `EdgeFixedIndex`
- [x] T2.3: Shadow darkening in terrain shader: `float shadow = alphaShadow.a; result *= mix(1.0, 0.4, shadow)`
- [x] T2.4: `dotnet build` + `dotnet test` — no regressions

## Phase 3: Verify Terrain Shader UV

- [x] T3.1: Read terrain shader UV code in `WorldGpuPreviewRenderer.cs` around line 1549-1598
- [x] T3.2: Compare against MdxViewer terrain shader UV computation — identical: `vec2(-vWorldPosition.y, -vWorldPosition.x) * texScale`
- [x] T3.3: No fix needed — UV computation is correct
- [x] T3.4: Final `dotnet build` + `dotnet test` — pass

## Validation

- [x] GPU capture on 3_3_5_12340 / development_0_0 (loose ADTs): 292 KB renders, terrain + objects visible
- [x] GPU capture on 0_5_3_3368 / Azeroth_30_48: 54 KB renders (was 2 KB blank before spec 013 fixes)
- [x] 3_3_5_12340 / Azeroth_30_48 via MPQ: terrain loads but alpha-WDT format tiles produce blank captures (pre-existing, not a regression)
