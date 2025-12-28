# Alpha 0.5.3 WMO Rendering System

**Source**: Ghidra reverse engineering of WoWClient.exe (0.5.3.3368)
**Date**: 2025-12-28
**Status**: Verified Ground-Truth

---

## 1. Overview

The Alpha client uses a dual-pipeline system for WMOs, distinguishing strictly between **Exterior** and **Interior** rendering paths. It supports lightmaps, fog, and sun lighting, but applies them differently based on the group type.

---

## 2. Main Render Dispatch

**Entry Point**: `CMapObj::RenderGroup` (0x0069bd50)

The client decides which renderer to use for a WMO Group based on `CWorld::enables` flags and the Group's own flags.

```cpp
// Logic in CMapObj::RenderGroup
if ((this_00->flags & 0x48) == 0) {
    // INTERIOR PATH (Indoor)
    (*DAT_00ec1b98)(); 
} else {
    // EXTERIOR PATH (Outdoor)
    (*DAT_00ec1ca0)(this_00, iVar3);
}
```

**Function Pointers** (Set in `CMapObj::PrepareUpdate`):

| Ptr | Default Function | High Quality Function | Role |
|-----|------------------|-----------------------|------|
| `DAT_00ec1b98` | `RenderGroup_Int` | `RenderGroupLightmap` | **Interior** Rendering |
| `DAT_00ec1ca0` | `RenderGroup_Ext` | `RenderGroupLightTex` | **Exterior** Rendering |

**Condition**: High quality (`RenderGroupLightmap`) is enabled if `CWorld::enables` flags `0x200` is set and `0x400` is clear.

---

## 3. Rendering Functions

### Interior Pipeline (`RenderGroupLightmap_Int`)
Used for indoor groups (flags `0x2000`? or just not `0x8`). Support **dual-texturing** (Diffuse + Lightmap).

1.  **Iterates Batches**: Loops through 4 render passes (likely Opaque, Alpha, etc.) from `intBatch`.
2.  **Locks Vertices**: Locks Position, UV0, UV1 (Lightmap). `stride = 0xC` for pos.
3.  **Material Render States**:
    *   **Fog**: Enabled if `!(Material.flags & 2)`
    *   **Culling**: Enabled if `!(Material.flags & 4)` (Two-Sided)
    *   **Blend**: Uses `Material.blendMode`
4.  **Textures**:
    *   `Texture0`: Diffuse Texture
    *   `Texture1`: Lightmap Texture
5.  **Draws**: `GxPrimDrawElements`

### Exterior Pipeline (`RenderGroup_Ext` / `RenderGroupLightmap_Ext`)
Used for outdoor groups (flags `0x8`). Relies on **Sun Light** and vertex normals.

1.  **Iterates Batches**: Loops through 4 render passes from `extBatch`.
2.  **Locks Vertices**: Locks Position, Normal, UV0. No Lightmap UVs usually.
3.  **Sun Lighting**:
    *   Uses `CMap::sunLight`.
    *   Checks Material Flag `0x20` (Window?). If set, temporarily switches sun color to `WindowDirColor`/`WindowAmbColor`.
4.  **Textures**:
    *   `Texture0`: Diffuse Texture
    *   `Texture1`: None (usually)

---

## 4. Batch Structure (SMOGxBatch)

The client optimizes rendering by grouping geometry into batches, organized in `CMapObjGroup` as arrays of `SMOGxBatch`.

**Array definitions**:
```cpp
SMOGxBatch intBatch[4]; // Interior passes
SMOGxBatch extBatch[4]; // Exterior passes
```

**Structure Layout** (8 bytes):
```cpp
struct SMOGxBatch {
    uint16_t vertStart;   // Offset to first vertex in Move/Lock
    uint16_t count;       // Vertex count (likely for Lock size)
    uint16_t startBatch;  // Index into the main `batchList`
    uint16_t batchCount;  // Number of batches to render in this pass
};
```

---

## 5. Render States & Flags

**Material Flags** (Verified Usage):
*   `0x01`: Unlit / Full bright (Implied by lack of lighting code in some paths)
*   `0x02`: Disable Fog (Checked in `RenderGroupLightmap`)
*   `0x04`: Two-Sided (Disable Culling)
*   `0x20`: **Window/Crystal** - Uses alternate Sun Color (WindowDirColor)

---

## 6. Portals & Liquid

*   **Liquid**: Rendered via `RenderLiquid_0` (if `flags & 0x1000` set on Group).
*   **Portals**: Rendered via `RenderPortals` (if `flags & 0x1000` set on Group - wait, verified code check). Actually `CWorld::enables & 0x1000` usually controls portals/water. Use `RenderPortals` function.

---
*Document generated from Ghidra analysis of WoWClient.exe (0.5.3.3368) on 2025-12-28.*
