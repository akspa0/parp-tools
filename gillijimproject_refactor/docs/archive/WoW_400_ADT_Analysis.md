# WoW 4.0.0.11927 ADT Format Analysis

**Analysis Date**: Mar 21 2026  
**Binary**: `wow.exe` (4.0.0.11927 Cataclysm Beta)  
**Tool**: Ghidra

This document supersedes the earlier simplified note that treated 4.0.0 terrain alpha as "just 3.3.5 plus RLE". The on-disk ADT format is still pre-split, but the runtime terrain texturing path is more sophisticated than the active viewer previously modeled.

## 1. Renamed Runtime Functions

### 1.1 Alpha Decode Helpers
| Function | Address | Description |
|----------|---------|-------------|
| `CMapChunk_UnpackAlphaBits` | `0x00674b70` | Main per-layer alpha dispatcher |
| `CMapChunk_UnpackAlphaShadowBits` | `0x00674560` | Alpha decode with optional shadow modulation |
| `CMapChunk_UnpackOneAlphaLayer` | `0x00675290` | Builds one layer alpha texture for a chunk |
| `CMapChunk_UnpackChunkAlphaSet` | `0x00675330` | Builds a stitched alpha set using linked neighbor chunks |
| `RLE_Decompress` | `0x00673230` | RLE decompression routine |

### 1.2 Blend Texture Assembly
| Function | Address | Description |
|----------|---------|-------------|
| `CMapChunk_BuildSingleLayerBlendTexture` | `0x00675b70` | Builds one chunk-level blend texture |
| `CMapChunk_BuildChunkBlendTextureSet` | `0x00675c30` | Builds a stitched multi-chunk blend texture set |
| `CMapChunk_RefreshBlendTextures` | `0x00675e30` | Top-level terrain blend refresh for a chunk |
| `TerrainBlend_SingleLayerCallback` | `0x00675540` | Callback used by `TerrainBlend` resource creation |
| `TerrainBlend_ChunkSetCallback` | `0x006755a0` | Callback used for stitched alpha-set generation |

### 1.3 Relevant String References
- `"CMapChunk::UnpackAlphaBits(): Bad genformat."` at `0x00a2402c`
- `"CMapChunk::UnpackAlphaShadowBits(): Bad genformat."` at `0x00a23ff8`
- `"terrainAlphaBitDepth"` at `0x00a24bfc`
- `"Alpha map bit depth set to %dbit on restart."` at `0x00a24430`
- `"TerrainBlend"` resource creation string inside `FUN_006730f0`

## 2. What Stayed The Same

- 4.0.0.11927 still uses **pre-split ADTs**.
- `WDT.MPHD` `0x4 | 0x80` still matters for 8-bit alpha support.
- `MCLY 0x200` is still the per-layer compressed-alpha flag.
- RLE still decompresses into a 64x64 output buffer.

These facts are true but incomplete. The missing piece was the runtime blend-texture assembly around those decoded alpha values.

## 3. Alpha Dispatch Model

`CMapChunk_UnpackOneAlphaLayer` selects `genformat` from chunk state:

```c
local_8 = 3;
if ((*(byte *)(param_1 + 8) & 4) != 0) {
    local_8 = 2;
}
```

- `genformat = 3`: 4-bit alpha path
- `genformat = 2`: 8-bit alpha path

It then builds a per-layer descriptor from the chunk's layer table:

- flags from `MCLY`
- optional alpha pointer if `MCLY.use_alpha_map (0x100)` is set
- optional shadow-mask pointer when shadow blending is active
- chunk/header `0x8000` state passed separately into the unpackers

## 4. RLE Decompression Is Still Standard

Decompiled from `FUN_00673230`:

```c
int RLE_Decompress(byte* src, byte* dest, int maxSize) {
    int iRead = 0;
    int iWrite = 0;

    while (iWrite < maxSize) {
        byte ctrl = src[iRead++];

        if (ctrl & 0x80) {
            byte value = src[iRead++];
            int count = ctrl & 0x7F;
            memset(&dest[iWrite], value, count);
            iWrite += count;
        }
        else {
            int count = ctrl;
            for (int i = 0; i < count; i++)
                dest[iWrite++] = src[iRead++];
        }
    }

    return iRead;
}
```

Key points:
- control byte bit 7 selects fill vs copy
- bits 0-6 contain the run count
- output target is the runtime alpha buffer for the current decode width/height

## 5. 8-bit Alpha Has A Residual Synthesis Path

This was the major missing behavior in the viewer.

When `genformat == 2` and a layer has **no direct alpha payload pointer**, `CMapChunk_UnpackAlphaBits` does not leave the layer blank. Instead it routes to helpers that synthesize the layer as the residual of the other layer alphas.

Relevant helpers:

| Address | Meaning |
|---------|---------|
| `0x006748d0` | 8-bit residual synthesis helper A |
| `0x006749c0` | 8-bit residual synthesis helper B |

The core computation is structurally:

```c
alpha = 255 - alpha_other_0 - alpha_other_1 - alpha_other_2;
```

with optional shadow modulation applied afterward.

Practical implication:
- a local MCAL-only decoder is insufficient for 4.0.0
- missing direct alpha on an overlay layer is not automatically "full alpha" or "zero alpha"
- the client can derive it from sibling layers at runtime

## 6. Runtime Blend Textures Are Neighbor-Aware

`CMapChunk_RefreshBlendTextures` does not only decode the current chunk. It refreshes `TerrainBlend` textures through one of two paths:

1. `CMapChunk_BuildSingleLayerBlendTexture`
2. `CMapChunk_BuildChunkBlendTextureSet`

The second path calls `CMapChunk_UnpackChunkAlphaSet`, which does all of the following:

- reads the current chunk's layer table
- walks three linked neighbor chunk pointers from `chunk + 0x18`
- matches neighbor layers against the current chunk by **texture id**, not by overlay slot index alone
- builds a stitched alpha set before creating the final `TerrainBlend` resource

This means the runtime blend model is seam-aware:
- edge texels can come from neighboring chunks with the same texture id
- different chunks may use different overlay slot indices for the same texture
- a pure per-chunk local alpha decode can still be wrong even if MCAL bytes are decoded correctly

## 7. Shadow Modulation Is Folded Into Some Runtime Helpers

Multiple 8-bit helpers optionally modulate alpha by shadow bits using:

```c
alpha = (alpha * 0xB2) >> 8;
```

This is approximately `178 / 256 ~= 0.695`.

In the active viewer, shadow is already carried separately through `MCSH` and the terrain renderer. That means we should be careful not to double-apply this modulation when porting 4.0 behavior.

## 8. Important Correction To The Earlier Jan Note

The earlier archive note oversimplified the fixed/unfixed selector as a direct `MCNK` bit-`0x8` rule.

What is actually visible now:
- `genformat` comes from chunk state (`chunk + 8` bit `0x4`)
- helper family selection also depends on runtime blend-builder state around `TerrainBlend`
- chunk/header `0x8000` is passed separately into the alpha/shadow unpackers

The exact public-facing name/polarity of every involved runtime bit is still not fully closed. Do not collapse this into a one-line "MCNK bit X means Y" rule without another RE pass.

## 9. Implementation Guidance For The Active Viewer

The minimum behavior needed to get closer to 4.0.0 terrain texturing is:

1. preserve the existing 4-bit / 8-bit / compressed local decode
2. synthesize missing 8-bit layer alpha as residual coverage from sibling layers
3. stitch chunk-edge alpha by matching neighbor layers via texture id
4. keep shadow as a separate viewer path unless we deliberately port the runtime combined-alpha helpers end-to-end

What is still out of scope for a minimal viewer recovery pass:

- recreating the full internal `TerrainBlend` resource system
- proving the exact fixed/unfixed runtime bit source beyond the current evidence
- matching every blend-texture dimension/LOD case in `CMapChunk_BuildChunkBlendTextureSet`

## 10. Version Notes

4.0.0.11927 still uses the **pre-split ADT** file layout. Split ADTs (`_tex0.adt`, `_obj0.adt`) were introduced later.

That does **not** mean terrain texturing behavior is identical to 3.3.5. The key 4.0 delta uncovered here is the runtime blend-texture assembly built on top of the same monolithic ADT inputs.
