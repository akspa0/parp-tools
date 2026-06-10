# WMO Render Pass Architecture (Build 3368 — Ghidra-Confirmed)

**Status**: active  
**Date**: 2026-05-30  
**Source**: Ghidra RE of wowclient.exe build 3368 (0.5.3.3368)  
**Owner**: wow-viewer engine runtime

## 1. Overview

The native client renders WMO groups through a dispatch system that selects render passes based on group flags, batch material properties, and interior/exterior mode. This document captures every render pass, the dispatch logic, batch structure, material flags, liquid rendering, lightmap handling, and portal-walk visibility as confirmed by Ghidra decompilation.

## 2. Top-Level Dispatch

### `CMapObj::RenderGroup` (0x0069bd50)

```
for each visible frustum in frustumList:
    if (group.flags & 0x48) == 0:
        (*DAT_00ec1b98)()              // Interior render callback
    else:
        (*DAT_00ec1ca0)(group, frustumIdx)  // Exterior render callback

// Post-pass rendering:
if (group.flags & 0x1000):
    RenderLiquid(group)               // Group has liquid data
if (CWorld::enables & 0x40000000):
    RenderGroupNormals(group)         // Debug: vertex normals
if (CWorld::enables & 0x1000):
    RenderPortals(group)              // Debug: portal wireframe
```

### Interior vs Exterior Mode

The function pointers `DAT_00ec1b98` and `DAT_00ec1ca0` are set during `CWorldScene::PrepareRender` based on the current camera position and day/night state. They select between interior and exterior render passes for each group.

- **Interior** (`flags & 0x48 == 0`): Camera inside WMO. Uses MOCV vertex colors, no dynamic lighting, interior fog possible.
- **Exterior** (`flags & 0x48 != 0`): Camera outside or group flagged as exterior. Uses dynamic lighting, no interior fog, sun direction applies.

## 3. Render Pass Functions

### Pass Table

| Function | Address | Group Flags | Texture | MOCV | Lighting | Lightmap | Notes |
|----------|---------|-------------|---------|------|----------|----------|-------|
| `RenderGroup_Int` | 0x0069db70 | `& 0x48 == 0` | No | Yes | Off | No | Interior base: MOCV vertex color only |
| `RenderGroup_Ext` | 0x0069da70 | `& 0x48 != 0` | No | No | On | No | Exterior base: vertex-lit, no vertex color |
| `RenderGroupColorTex_Int` | 0x0069d210 | `& 0x48 == 0` | Yes | Yes | Off | No | Interior textured + vertex color |
| `RenderGroupColorTex_Ext` | 0x0069d450 | `& 0x48 != 0` | Yes | Yes | On | No | Exterior textured + vertex color |
| `RenderGroupColorTex` | 0x0069d6f0 | dispatches | Yes | Yes | varies | No | Dispatches Int/Ext |
| `RenderGroupLightTex` | 0x0069ca10 | any | Yes | No | per-batch | No | Per-batch material flags |
| `RenderGroupLightmap` | 0x0069d770 | any | Yes | No | No | Yes (UV) | Lightmap UV channel + texture |
| `RenderGroupLightmapTex_Int` | 0x0069cc90 | `& 0x48 == 0` | Yes | No | Off | Yes (tex1) | Interior: lighting off, lightmap on tex1 |
| `RenderGroupLightmapTex_Ext` | 0x0069cf00 | `& 0x48 != 0` | Yes | No | On | No (tex1) | Exterior: lighting on, no lightmap on tex1 |
| `RenderGroupLightmapTex` | 0x0069d190 | dispatches | Yes | No | varies | varies | Dispatches Int/Ext |
| `RenderGroupTex` | 0x0069d8c0 | any | Yes | No | No | No | Texture only, white vertex color |
| `RenderGroupBsp` | 0x0069df60 | debug | No | No | No | No | BSP polygon debug renderer |

### Pass Selection Logic

The render mode is selected per-group based on:
1. **Group flags** (`flags & 0x48`): Determines interior vs exterior path
2. **Batch presence**: Each group has `intBatch[4]` and `extBatch[4]` sub-ranges
3. **Material flags per batch**: Each batch's MOMT flags determine which pass handles it

## 4. Batch Structure

### Sub-Batches

Each WMO group maintains two arrays of 4 sub-batches:

```
intBatch[4]  — interior rendering batches
extBatch[4]  — exterior rendering batches

struct SMOBatch {
    uint vertStart;         // Start vertex index
    void* gxbuf;           // GPU buffer handle
    uint batchStartIndex;   // Start index in index buffer
    uint batchCount;        // Number of indices
    // Also: bounding box (tx, ty, tz, bx, by, bz) for CullBatch
};
```

### CullBatch (0x0069c630)

Each batch has a bounding box. `CullBatch` performs frustum culling against the batch's AABB before rendering:

```
bool CullBatch(SMOBatch* batch):
    localBox = AABB(batch.tx, batch.ty, batch.tz, batch.bx, batch.by, batch.bz)
    return FrustumCull(localBox) != 0
```

## 5. Per-Batch Material Flags (MOMT)

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | Lighting | Dynamic lighting enabled for this batch |
| 1 | Fog | Fog enabled for this batch |
| 2 | Culling | Backface culling enabled |
| 3 | TexAddr | Texture address mode |
| 4 | WrapClamp | Wrap vs clamp texture mode |
| 0x10 | Emissive | Self-illumination (unlit) |
| 0x20 | WindowLit | Exterior sun override for interior windows |

**WindowLit** (0x20): A special flag for window polygons in interior groups. When the group is rendered in interior mode, windows with this flag receive exterior sun lighting instead of interior MOCV color, simulating sunlight coming through windows.

## 6. Liquid Rendering

### `RenderLiquid` (0x0069e4b0)

```
1. Scan first liquid tile for type: liquid & 0xF
2. Type dispatch:
   - 0, 4, 8: Water
     - flags & 0x48 == 0: RenderInteriorWater(group, 4)
     - flags & 0x48 != 0: RenderExteriorWater(group, 4)
   - 2, 3, 6, 7: Magma → RenderMagma(group, type)
3. Setup: GxVS_PassThru, disable culling, get liquid texture
4. Water: tex blend mode 3 (modulate)
5. Magma: separate render path
```

### Interior Water (`RenderInteriorWater_0` at 0x0069e5d0)

- Vertex format: Position + Color + TC0 (no normals)
- Color from material diffColor (`materialList[liquidMtlId].diffColor`)
- Interior fog applied when `DayNightGetInfo()->intFog != 0 && this == camMapObj`
- Fog start/end/color from `DayNightGetInfo()->intFogInfo`

### Exterior Water (`RenderExteriorWater_0` at 0x0069e7a0)

- Vertex format: Position + Normal + Color + TC0
- Color from `DayNightGetInfo()->light.WaterArray[3]` (day/night-dependent)
- Normal = (0, 0, 1) for flat water surface
- No interior fog

### Magma (`RenderMagma` at 0x0069e930)

- Separate render path for lava/magma liquid types
- Type value (2/3/6/7) passed as parameter for texture selection

## 7. Lightmap System

### `CreateLightmaps` (0x006adba0)

- Each group can have `lightmapTexCount` lightmap textures
- Lightmaps are 256×256, format: DXT if GPU supports it, else RGB565
- Created on first demand with label "Lightmap"
- On-demand streaming via `UpdateLightmapTex` callback
- Flush timeout: 30 seconds (`lightmapTexFlushTime = 30.0`)
- Each lightmap stored in `lightmapTexList` array with stride 0x8004

### Lightmap Pass Differences

| Pass | Interior | Exterior |
|------|----------|----------|
| `RenderGroupLightmapTex` | Lighting OFF, lightmap on tex1 | Lighting ON, NO lightmap on tex1 |
| `RenderGroupLightmap` | Lightmap UV channel only, no separate color | Same |

The key difference: In interior mode, the lightmap provides the "lighting" (baked illumination replaces dynamic lighting). In exterior mode, dynamic lighting is used and the lightmap is not applied to tex1.

## 8. Portal Walk Visibility

### `RRenderThruPortals` (0x0069bf60)

Recursive portal traversal for determining which groups are visible:

```
RRenderThruPortals(groupIdx, fromGroupIdx, parentRect, depth):
    group = GetGroup(groupIdx)
    if depth > maxRLevel: return
    if group.flags & 0x10000: skip  // Always-render groups handled separately
    
    // Liquid first-frame toggle
    if group.flags & 0x1000 && !rDrawSharedLiquidFirst:
        rDrawSharedLiquidToggle = depth & 1
        rDrawSharedLiquidFirst = true
    
    // Render callback for this group
    if gRenderCallback: gRenderCallback(depth & 1 == rDrawSharedLiquidToggle, ...)
    
    // Walk portals to adjacent groups
    for each portalRef in group.portalRefs:
        if portalRef.groupIndex == fromGroupIdx: continue  // Don't go back
        if portal already visited this frame: continue
        
        RTransformPortal(portal, portalExt, cpIgnore)
        mark portal visited (gRenderCount)
        
        if portal not facing camera: continue
        if portal screen rect doesn't intersect parent rect: continue
        
        clip portal rect to parent rect
        if clipped rect is too small: continue
        
        // Interior: add exterior-visible groups to extViewList
        if bIntRender && adjacent group has flags & 8:
            add to extViewList with screen rect
        
        // Recurse into adjacent group
        CWorldScene::FrustumPush()
        FrustumSet(camFrustumCorners, clippedRect)
        RRenderThruPortals(adjacentGroupIdx, groupIdx, clippedRect, depth + 1)
        CWorldScene::FrustumPop()
```

### `IntRender` (0x0069b870)

1. Set `bIntRender = 1`
2. For each group in the player's visibility list:
   - Skip groups with `flags & 0x10000` (handled by `RenderAlways`)
   - Skip groups without `flags & 8` (not interior-visible)
   - Frustum cull, then `RRenderThruPortals(groupIdx, 0xFFFF, screenRect, 0)`
3. Process `_extViewList` (exterior-visible groups discovered during portal walk):
   - For each exterior group, frustum cull, then `RRenderThruPortals`
4. Process `RenderAlways` groups (flags & 0x10000):
   - Frustum cull, then `RenderAlways(groupIdx)`

### `ExtRender` (0x0069bb80)

1. Set `bIntRender = 0`
2. For each group:
   - If `flags & 0x10000`: frustum cull, `RenderAlways(groupIdx)`
   - If `flags & 8` and not `0x10000`: frustum cull, `RRenderThruPortals(groupIdx, 0xFFFF, rect, 0)`

### `RenderAlways` (0x0069bf00)

Used for groups with `flags & 0x10000` — always visible regardless of portal walk. These are typically skybox or exterior shell groups.

## 9. Group Flags Summary

| Flag | Meaning | Effect on Rendering |
|------|---------|---------------------|
| `0x08` | Has exterior visibility | Group can be seen from outside (via portal walk or direct) |
| `0x40` | Exterior render path | Group rendered with exterior lighting/sun |
| `0x08 | 0x40` = `0x48` | Full exterior group | Dispatches to exterior render callback |
| `0x80` | No render | Group should not be rendered |
| `0x88` | No render + no collide | Skip entirely, no minimap |
| `0x1000` | Has liquid | `RenderLiquid` called after group passes |
| `0x10000` | Always visible | `RenderAlways` path, not portal-walk |

## 10. Interior Fog

Applied when ALL conditions are met:
1. `DayNightGetInfo()->intFog != 0` (interior fog is enabled for current time of day)
2. `this == CWorldScene::camMapObj` (the WMO is the one the camera is inside)

Fog parameters:
- Start: `DayNightGetInfo()->intFogInfo.start`
- End: `DayNightGetInfo()->intFogInfo.end`
- Color: `DayNightGetInfo()->intFogInfo.color`

Interior fog applies to both the WMO group surfaces and the water surface (set in `RenderInteriorWater_0`).

## 11. Key Data Structures

### DNInfo (Day/Night Info)

Global singleton at `DAT_010b23b0` (returned by `DayNightGetInfo` at 0x006bd8b0).

Contains:
- `intFog` (uchar): Whether interior fog is active
- `intFogInfo` (struct): Interior fog start, end, color
- `light` (struct): Day/night lighting state, including `WaterArray[3]` for water color
- `fogInfo` (struct): Exterior fog color

### CMapObjGroup Key Fields

- `flags` (uint): Group flags (see section 9)
- `portalCount` / `portalStart`: Portal references
- `lightmapTexCount`: Number of lightmap textures
- `lightmapTexList`: Array of lightmap texture data
- `liquidMtlId`: Material index for liquid surface
- `liquidVerts` (C2iVector): Liquid grid dimensions (x, y)
- `liquidCorner` (C3Vector): World position of liquid grid corner
- `liquidVertexList`: Array of liquid vertex data (height + flags)
- `liquidTileList`: Array of liquid tile data

## 12. Implications for wow-viewer

### Current MdxViewer Gaps

1. **No interior/exterior split**: All groups rendered the same way
2. **No per-batch material flag testing**: All batches rendered with same state
3. **No interior fog**: Missing entirely
4. **Lightmap pass conflated**: Single lightmap path instead of Int/Ext split
5. **Liquid type dispatch simplified**: No magma path, no interior/exterior water difference
6. **Group flag checks missing**: `0x88` groups may be rendered, `0x10000` groups not handled

### Required Fixes for WowViewer.Core.Runtime

1. Implement interior/exterior render mode selection based on group flags
2. Add per-batch MOMT flag testing (lighting, fog, culling, emissive, window-lit)
3. Add interior fog system driven by DayNightGetInfo
4. Split lightmap pass into LightmapTex_Int and LightmapTex_Ext
5. Add liquid type dispatch (water interior/exterior, magma)
6. Add group flag filtering (skip 0x88, always-render 0x10000)
7. Implement portal-walk visibility for interior rendering
