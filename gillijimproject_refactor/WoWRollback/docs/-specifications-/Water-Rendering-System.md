# Alpha 0.5.3 Water Rendering System

**Source**: Ghidra reverse engineering of WoWClient.exe (0.5.3.3368)
**Date**: 2025-12-28
**Status**: Verified Ground-Truth

---

## 1. Overview

The Alpha client contains a fully functional **procedural water system** that was largely disabled or simplified in later versions. It features:
- **Radial Wave Simulation**: Dynamic ripples from footsteps and objects
- **Procedural Textures**: Real-time generation of water textures based on Day/Night cycle
- **Pixel Shader Water**: Uses `Ocean0.bls` for advanced rendering
- **Magma & Slime**: Distinct rendering paths for different liquid types

---

## 2. Console Variables (CVars)

These variables control the water system. They can be set via console or config.

| CVar | Address | Description | Default/Usage |
|------|---------|-------------|---------------|
| `waterParticulates` | 0x0089d904 | Toggles floating particles | - |
| `waterRipples` | 0x0089d918 | **Master switch for ripples** | Set to 1 to enable footstep ripples |
| `waterSpecular` | 0x0089d928 | Usage of `CMap::psOcean0` shader | 1 = use pixel shader |
| `waterWaves` | 0x0089d938 | Procedural wave animation | - |
| `waterMaxLOD` | 0x0089d944 | Max geometry detail | 0-1 range |
| `showWater` | 0x0089d950 | Global water rendering toggle | 1 = on |

**Hidden Flag**:
- `CWorld::enables & 0x1000000`: Must be set for `RenderWater` to execute (checked in `RenderWater @ 0x0066e560`).

---

## 3. Ripple Simulation (WaterRadWave)

The client simulates radial waves (ripples) using a pre-allocated pool of 48 nodes.

### Structure (WaterRadWave)
**Size**: 64 bytes (0x40)

```cpp
struct WaterRadWave {
    TSLinkBase link;          // Linked list node (next/prev)
    C3Vector pos;             // Center position (x,y,z)
    float length;             // Wave radius length
    float amplitude;          // Wave height
    float timeLength;         // Duration in seconds
    float frequency;          // Oscillation frequency
    float velocity;           // Expansion speed
    float curTime;            // Current elapsed time
    float ra;                 // Inner radius (state)
    float rb;                 // Outer radius (state)
    float ooLength;           // 1.0 / length (optimization)
    float ooTimeLength;       // 1.0 / timeLength (optimization)
    float decay;              // Linear decay factor (1.0 -> 0.0)
};
```

### Simulation Logic

**Update Function** (`WaterRadWave::Update @ 0x00672ac0`):
Called every tick for active ripples.

```cpp
int WaterRadWave::Update(float dt) {
    this->curTime += dt;
    
    // Check if wave has expired
    if (this->curTime > this->timeLength) {
        return 0; // Remove wave
    }
    
    // Expand wave ring
    this->rb = this->curTime * this->velocity;  // Outer radius
    this->ra = this->rb - this->length;         // Inner radius
    
    // Calculate decay
    this->decay = 1.0f - (this->curTime * this->ooTimeLength);
    
    return 1; // Keep wave
}
```

### Triggering Ripples (`OnMoveUpdate @ 0x005f36fc`)

When a unit moves through water:
1. Checks `QueryObjectLiquid`
2. If in water and depth is shallow enough:
   - Calls `CWorld::WaterRipple`
   - **Parameters**: 
     - `length`: 0.73333335
     - `amplitude`: 1.0
     - `timeLength`: 0.16666667 (very short?)
     - `velocity`: 6.6666665

---

## 4. Procedural Texture Generation

The client generates water textures **dynamically** every frame based on the Day/Night cycle.

**Callback**: `CMap::WaterDiffTexCallback @ 0x00673490`

### Logic
1. Reads colors from `DayNightGetInfo()->light.WaterArray` (pairs of colors)
2. Interpolates colors based on Day/Night time
3. Generates a **64-pixel gradient** texture
4. Applies **HSV color space adjustment** at edges (using `RGBtoHSV` / `HSVtoRGB`)
5. Uploads to `riverDiffTexid` and `oceanDiffTexid` handles

**Generated Handles**:
- `skyTexid`
- `riverDiffTexid`
- `oceanDiffTexid`

**Texture Format**: `GxTex_Argb8888` (32-bit color)

---

## 5. Rendering Pipeline

**Main Render Function**: `RenderLiquid_0 @ 0x0069e4b0`

### Liquid Types

| ID | Name | Render Path |
|----|------|-------------|
| 0 | Water (Standard) | `RenderExteriorWater_0` or `RenderInteriorWater_0` |
| 1 | Ocean (Fatigue) | `RenderOcean_0` (implied) |
| 2 | Magma | `RenderMagma` |
| 3 | Slime | `RenderMagma` (shared path) |
| 4 | River | Same as Water |
| 6 | Magma (Alt) | Same as Magma |

### Exterior Water Render (`RenderExteriorWater_0 @ 0x0069e7a0`)

1. Allocates vertex buffer (`GxAllocVertexMem`)
2. Retrieves shallow color from `DayNightGetInfo()->light.WaterArray[3]`
3. Iterates liquid vertices:
   - Pos: `(x, y, height)`
   - Color: `shallowClr` (modulated by depth)
   - UVs: Generated based on world pos
4. Draws TriangleStrip

### Shader Support

- **Pixel Shader**: `Shaders/Pixel/Ocean0.bls`
- **Vertex Shader**: `GxVS_PassThru` (usually)

### Resources

- **Ocean Texture**: `XTextures\ocean\ocean_h.%d.blp` (animated sequence)
- **Sound DBC**: `DBFilesClient\SoundWaterType.dbc`

---

## 6. How to Enable Effects

To "wake up" the water system in the client:

1. **Enable Ripples**:
   Set `waterRipples` CVar to `1`.
   ```cpp
   // Console
   waterRipples 1
   ```

2. **Enable Specular**:
   Set `waterSpecular` CVar to `1` (requires pixel shader support).

3. **Verify Global Enable**:
   Ensure `showWater` is `1`.

---

*Document generated from Ghidra analysis of WoWClient.exe (0.5.3.3368) on 2025-12-28.*
