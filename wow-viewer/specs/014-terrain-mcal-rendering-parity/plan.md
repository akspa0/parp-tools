# Implementation Plan: Terrain MCAL Rendering Parity (Spec 014)

**Status**: Active
**Created**: 2026-05-23

## Phase 1: Thread Shadow Map Through Runtime Chunk Data

**Goal**: Make shadow map data available to the GPU renderer by adding it to `WorldTerrainChunkData`.

**Steps**:
1. Add `byte[]? ShadowMap` property to `WorldTerrainChunkData`
2. Pass `textureChunk?.ShadowMap` in `WorldTerrainTileBuilder.Read` (LK path)
3. Pass shadow map in `AlphaEmbeddedAdtReader.TryReadAlphaTerrainChunk` (Alpha path)
4. Build and test

**Validation**: `dotnet build` passes. `dotnet test` passes. ShadowMap is non-null for chunks that have MCSH data.

## Phase 2: Fill Alpha Channel with Shadow Map in GPU Renderer

**Goal**: Upload shadow map into channel 3 of the alpha-shadow texture array.

**Steps**:
1. In `FillAlphaShadowSlice`, after writing channels 0-2 for texture layers, write chunk.ShadowMap into channel 3
2. Use edge-clamped indexing matching MdxViewer reference (clamp x/y to size-2)
3. Build and test

**Validation**: `dotnet build` passes. GPU capture on `3_3_5_12340 / Azeroth_30_48` shows shadow data in alpha channel.

## Phase 3: Verify Terrain Shader UV and Final Validation

**Goal**: Confirm terrain shader UV computation is correct for the fixed coordinate system.

**Steps**:
1. Review terrain shader diffuse UV code (`vWorldPosition` usage) in `WorldGpuPreviewRenderer`
2. Compare against MdxViewer's UV computation
3. Fix any remaining UV inversion
4. Run `dotnet test` for regression

**Validation**: Terrain texture tiling matches MdxViewer reference on both `0_5_3_3368` and `3_3_5_12340`.
