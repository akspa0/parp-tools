# WoW 4.0.0.11927 Terrain Blend Recovery Guide

## Purpose

This guide translates the Ghidra work on `wow.exe` build `4.0.0.11927` into concrete implementation targets for the active `MdxViewer` terrain pipeline.

The important conclusion is simple:

- the ADT file format is still pre-split and mostly 3.x-shaped
- the **runtime terrain texturing path is not just a local MCAL decode**

If the viewer only decodes MCAL bytes chunk-by-chunk, it will miss two client behaviors that matter on the fixed `development` dataset:

1. 8-bit residual alpha synthesis for layers with no direct alpha payload
2. texture-id-based edge stitching across linked neighbor chunks

## Runtime Call Chain

Current renamed functions:

- `CMapChunk_RefreshBlendTextures` at `0x00675e30`
- `CMapChunk_BuildSingleLayerBlendTexture` at `0x00675b70`
- `CMapChunk_BuildChunkBlendTextureSet` at `0x00675c30`
- `CMapChunk_UnpackOneAlphaLayer` at `0x00675290`
- `CMapChunk_UnpackChunkAlphaSet` at `0x00675330`
- `CMapChunk_UnpackAlphaBits` at `0x00674b70`
- `CMapChunk_UnpackAlphaShadowBits` at `0x00674560`

The blend refresh path creates a `TerrainBlend` texture resource through `FUN_006730f0`. That resource is built from decoded alpha data, not directly from raw MCAL bytes.

## What The Client Actually Does

### 1. Choose 4-bit vs 8-bit

`CMapChunk_UnpackOneAlphaLayer` selects:

- `genformat = 3` for 4-bit alpha
- `genformat = 2` for 8-bit alpha

### 2. Use Direct Alpha When Present

If the layer descriptor has `MCLY.use_alpha_map (0x100)` set, the client reads the direct alpha payload.

For 8-bit layers:
- `MCLY 0x200` means compressed and routes through `RLE_Decompress`
- otherwise raw bytes are consumed directly

### 3. Synthesize Missing 8-bit Alpha

If an 8-bit layer has no direct alpha payload, helpers at `0x006748d0` and `0x006749c0` derive it as the residual of the other layers.

Operationally, the missing layer behaves like:

```text
255 - alpha_of_other_layer_0 - alpha_of_other_layer_1 - alpha_of_other_layer_2
```

This is the first recovery step the active viewer must implement.

### 4. Build A Neighbor-Aware Alpha Set

`CMapChunk_UnpackChunkAlphaSet` builds a stitched alpha set across the current chunk and three linked neighbor chunks.

Important details:

- neighbor layers are matched by **texture id**
- matching is not based only on local overlay slot number
- the stitched output feeds the final runtime `TerrainBlend` resource

This is the second recovery step the viewer must implement in an approximate form.

## Active Viewer Mapping

### Current viewer reality

`StandardTerrainAdapter` outputs per-chunk `64x64` alpha maps in `TerrainChunkData.AlphaMaps`, and `TerrainRenderer` binds those as overlay alpha textures.

That means we are not reproducing Blizzard's internal `TerrainBlend` resource directly. The practical porting target is therefore:

1. keep local decode working
2. mutate the per-chunk alpha textures so they better match the runtime blend result

### Minimal implementation model

#### Step 1: Residual synthesis

For `Cataclysm400` chunks in 8-bit mode:

- inspect each overlay layer `1..3`
- if the layer lacks a direct alpha payload and no decoded alpha map exists yet
- synthesize its alpha from the sibling layer maps

This should replace the current viewer fallback where the renderer treats such a layer as implicitly full alpha.

#### Step 2: Neighbor edge stitching

After local decode and residual synthesis:

- find same-tile neighbors by chunk coordinate
- match neighbor layers by texture id
- copy edge texels from matching neighbors onto the current chunk edge

The low-risk approximation is:

- right edge from the neighbor chunk to the east/right
- bottom edge from the neighbor chunk to the south/down
- bottom-right corner from the diagonal chunk when available

This does not reproduce the entire client blend resource, but it is the smallest viewer-side step that addresses the RE findings directly.

## Known Open Questions

1. The exact public meaning of every runtime fixed/unfixed selector bit is not fully closed yet.
2. The client sometimes folds shadow modulation into alpha decode helpers, but the viewer already handles shadow separately.
3. The runtime `TerrainBlend` resource dimensions and callback flags may vary by chunk state and are not fully ported.

These are follow-up RE targets, not blockers for the first active viewer recovery slice.

## Recommended Future Prompt Inputs

Any future 4.0 terrain session should read:

1. `docs/archive/WoW_400_ADT_Analysis.md`
2. this guide
3. `docs/ADT_WDT_Format_Specification.md`
4. `memory-bank/activeContext.md`
5. `memory-bank/progress.md`
6. `src/MdxViewer/memory-bank/terrain_editing_plan_2026-02-14.md`

Then it should answer three questions before changing code:

1. Is the problem local decode, residual synthesis, or edge stitching?
2. Is the bug in adapter output or in renderer use of those outputs?
3. What real-data tile from `test_data/development/World/Maps/development` is being used as the check?