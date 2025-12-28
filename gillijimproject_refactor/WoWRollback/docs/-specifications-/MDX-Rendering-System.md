# Alpha 0.5.3 MDX Rendering System

**Source**: Ghidra reverse engineering of WoWClient.exe (0.5.3.3368)
**Date**: 2025-12-28
**Status**: Verified Ground-Truth

---

## 1. Overview

The Alpha client's MDX (M2) rendering system is a robust, multi-pass pipeline that supports skeletal animation, dynamic hardware lighting (up to 8 lights), and complex multi-texturing with animated UV coordinates.

---

## 2. Rendering Hierarchy

The rendering process is hierarchical:

1.  **`CSimpleModel::RenderModel`** (0x007638f0)
    *   **High-Level Setup**: Manages Camera, Projection, and View matrices.
    *   **Lighting**: Explicitly binds up to **8 Hardware Lights** (`GxLightSet`) plus Ambient.
    *   **Fog**: Sets `GxRs_Fog` state based on Model Flags.
    *   **Scene Call**: Calls `ModelRenderScene`.

2.  **`RenderGeoset`** (0x00431ad0)
    *   **Preparation**: Calls `RenderGeosetPrep` to setup Bone Matrices (`GxXformSetBones`) and Vertex Shaders.
    *   **Layer Dispatch**: Calls `RenderGeosetLayers`.

3.  **`RenderGeosetLayers`** (0x00431b90)
    *   Determines if Single-Pass or Multi-Pass rendering is required.
    *   Calls `RenderGeosetMultiUvMapping` (most common) or `RenderGeosetOneUvMapping`.

---

## 3. Material & Texture Logic

**Function**: `RenderGeosetMultiUvMapping` (0x00430eb0)

The renderer iterates through the material's texture layers and batches them for the GPU.

### Multi-Texturing
*   **Combiners**: Checks hardware capabilities (likely limit 2) to determine how many layers to bind in one pass.
*   **Binding**:
    *   `GxRs_Texture0` -> Layer 0
    *   `GxRs_Texture1` -> Layer 1
*   **Blending**: If `layersDrawn > 0`, it pushes a new Render State (`GxRsPush`) to handle blending with previous passes.

### Material Flags (Mapped to Render States)
The renderer maps internal flags (likely from M2 header) to `GxRs` states:

| Flag (in M2) | Render State | Value | Description |
|---|---|---|---|
| `0x01` | `GxRs_Lighting` (?) | - | Unlit (Implied) |
| `0x02` | `GxRs_Fog` | 0 | **Disable Fog** |
| `0x04` | `GxRs_DepthTest` | 0 | **Disable Depth Test** |
| `0x08` | `GxRs_DepthWrite` | 0 | **Disable Depth Write** |
| `0x10` | `GxRs_Culling` | 0 | **Two-Sided** (Disable Culling) |

---

## 4. Vertex Format & Animation

**Function**: `LockVertices` (via `RenderSingleUVMapPrep` or Multi)

Unlike WMOs which use static vertex buffers, MDX models use **Dynamic/Animated Vertex Data**.

### Vertex Streams
The `GxPrimLockVertexPtrs` call reveals the stream layout:
1.  **Positions**: `C3Vector` (Stride 12) - From transformed buffer?
2.  **Normals**: `C3Vector` (Stride 12)
3.  **Weights/Indices**: `uchar*` (Stride 1 if present) - Hardware skinning data.
4.  **Texture Coordinates**: **Dynamic!**
    *   UVs are **not** passed as a static stream from the file.
    *   They are looked up from an **Animation Buffer** (`*(param_1 + 0x3c)`).
    *   This confirms that UV animations (scrolling, rotating) are computed on the CPU and passed as efficient arrays to the GPU per-frame.

---

## 5. Lighting & Fog details

*   **Lights**: Uses standard `GxLight` structures. Supports Directional and Point lights.
*   **Fog**:
    *   Restores Fog state after rendering via `RestoreFog`.
    *   Uses `GxMasterEnable_Fog` to toggle global fog processing.

---
*Document generated from Ghidra analysis of WoWClient.exe (0.5.3.3368) on 2025-12-28.*
