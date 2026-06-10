# Liquid Rendering System (Alpha 0.5.3)

## Overview

This document covers the liquid rendering system in World of Warcraft Alpha 0.5.3, based on reverse engineering analysis using Ghidra. The system handles rendering of terrain liquids (rivers, oceans, lava, slime) and WMO (World Map Object) liquids.

## Liquid Data Structures

### CChunkLiquid (Terrain Liquid)

The `CChunkLiquid` structure represents liquid data for an ADT map chunk (MCNK).

**Size**: 0x338 bytes (824 bytes)

```c
struct CChunkLiquid {
    CInterval height;              // 0x00: Min/max height (l=min, h=max)
    SWFlowv flowvs[8];            // 0x08: Flow vertices for animation
    TSLink<CChunkLiquid> sceneLink;    // 0x108: Render scene linked list
    TSLink<CChunkLiquid> lameAssLink;  // 0x118: Free list linked list
    CImVector verts[9][9];        // 0x128: Height grid (9x9 vertices)
    // ... additional fields
};
```

**Key Fields**:
- `height.l`: Minimum liquid height
- `height.h`: Maximum liquid height  
- `flowvs[8]`: Array of 8 flow vertices for animated flow effects
- `verts[9][9]`: 9×9 height grid stored as CImVector (16-bit per component)

### SWFlowv (Flow Vertex)

The `SWFlowv` structure stores flow animation data for liquids.

**Size**: 28 bytes (7 floats)

```c
struct SWFlowv {
    C3Vector sphere;              // 0x00: Sphere center and radius
    float sphere.r;               // 0x0C: Sphere radius
    C3Vector dir;                // 0x10: Flow direction vector
};
```

**Usage**: Used for animating water flow direction and intensity.

### SMOLVert (WMO Liquid Vertex)

The `SMOLVert` structure represents a vertex in WMO (World Map Object) liquid.

**Structure**:
```c
struct SMOLVert {
    union {
        struct {
            float height;         // Vertex height
        } waterVert;
        
        struct {
            ushort s;             // S (U) coordinate
            ushort t;             // T (V) coordinate
        } magmaVert;
    };
};
```

**Usage**: WMO liquids use a different vertex format than terrain liquids.

### SMOLTile (WMO Liquid Tile)

The `SMOLTile` structure represents a liquid tile in WMO groups.

**Size**: 2 bytes

```c
struct SMOLTile {
    uint8_t liquid;              // Liquid type (4 bits used)
    uint8_t flags;               // Tile flags
};
```

## Liquid Types

The Alpha 0.5.3 client supports multiple liquid types with different rendering behaviors.

### Liquid Type Enumeration

Based on Ghidra analysis, the following liquid types are used:

| Type | Value | Rendering | Notes |
|------|-------|-----------|-------|
| Water (General) | 0 | Standard water | Exterior water rendering |
| Ocean | 4 | Standard water | Exterior ocean rendering |
| Magma | 2 | Lava rendering | Heat glow effect |
| Slime | 3 | Lava rendering | Green tinted lava |
| River | 8 | Standard water | River water |
| Lava (Alternate) | 6 | Lava rendering | Alternate lava type |
| Slime (Alternate) | 7 | Lava rendering | Alternate slime type |

### Liquid Type Detection

From `PrepareRenderLiquid()` at 0x0066a590:

```c
switch(newLiquid & 3) {
case 0:
case 1:
    // Water types
    // InitParticles with scale 0.027777778
    break;
case 2:
    // Magma/lava types  
    // InitParticles with scale 0.11111111
    break;
case 3:
    // Disable particles
    break;
}
```

## Rendering Pipeline

### Terrain Liquid Rendering

#### CMapObj::RenderExteriorWater_0

**Address**: 0x0069e7a0

This function renders exterior water (rivers, oceans) from terrain chunks.

**Vertex Format**: CGxVertexPNCT0 (0x24 bytes per vertex)

```
Layout:
- Position (3 floats): X, Y, Z
- Normal (3 floats): NX, NY, NZ  
- Color/Diffuse (CImVector): ARGB color
- Texture Coords (2 floats): U, V
```

**Rendering Steps**:

1. **Allocate Vertex Buffer**:
```c
uint numVerts = liquidVerts.x * liquidVerts.y;
GxAllocVertexMem(numVerts * 0x24);  // CGxVertexPNCT0 format
```

2. **Build Vertex Data**:
```c
for each vertex:
    Position = liquidCorner + grid offset
    Normal = (0, 0, 1) // Upward facing
    Color = DayNightGetInfo()->WaterArray[3]
    UV = Calculated from position
```

3. **Generate Indices**:
```c
RenderWaterIndices_0(this, group, idxBase, 0, &idxSub);
// Uses TriangleStrip with tile-based culling
```

4. **Draw**:
```c
GxPrimDrawElements(GxPrim_TriangleStrip, idxSub, idxBase);
```

#### CMapObj::RenderMagma

**Address**: 0x0069e930

Renders lava/magma with animated UV coordinates.

**Vertex Format**: CGxVertexPCT0 (0x18 bytes per vertex)

```
Layout:
- Position (3 floats): X, Y, Z
- Color/Diffuse (CImVector): ARGB
- Texture Coords (2 floats): U, V
```

**UV Animation**:
```c
// Texture matrix setup for lava animation
local_74.d1 = time * 0.1f;  // Scroll V coordinate

GxXformPush(GxXform_Tex0, &local_74);  // Apply texture transform
// Draw...
GxXformPop(GxXform_Tex0);               // Restore
```

### WMO Liquid Rendering

#### CMapObj::RenderLiquid_0

**Address**: 0x0069e4b0

Main WMO liquid rendering entry point.

**Flow**:
1. Query liquid type from tile list
2. Set render state (culling, blending)
3. Get appropriate liquid texture
4. Dispatch to specific renderer:
   - `RenderExteriorWater_0()` for water types (0, 4, 8)
   - `RenderMagma()` for lava types (2, 3, 6, 7)

**Render State Setup**:
```c
GxRsSet(GxRs_Culling, 0);          // Disable culling
GxRsSet(GxRs_Texture0, texture);   // Bind liquid texture

switch(liquidType) {
case 0:
case 4:
case 8:
    GxRsSet(GxRs_TexBlend0, 3);   // Alpha blending for water
    RenderExteriorWater_0(...);
    break;
case 2:
case 3:
case 6:
case 7:
    RenderMagma(...);             // Lava rendering
    break;
}
```

### Index Generation

#### CMapObj::RenderWaterIndices_0

**Address**: 0x0069e370

Generates TriangleStrip indices for liquid mesh with tile-based culling.

**Algorithm**:
- Iterates through liquid tiles (SMOLTile array)
- Skips tiles with `liquid & 0xf == 0xf` (no liquid)
- Respects `rDrawSharedLiquidToggle` culling flag
- Generates TriangleStrip vertices with proper restart handling

```c
// For each active tile:
ushort v0 = currentVertex;
ushort v1 = currentVertex + 1;
ushort v2 = currentVertex + width;
ushort v3 = currentVertex + width + 1;

// Output: v0, v1, v2, v3 (quad as two triangles)
// Or: v0, v1, v2 (TriangleStrip continuation)
```

## Liquid Texture System

### Texture Loading

#### CMap::GetLiquidTexture

**Address**: 0x006736b0

Loads animated liquid textures with frame cycling.

**Key Globals**:
```c
liquidTexBaseName[9];     // Texture path format (e.g., "XTextures\\River\\%02d.blp")
liquidTexLoopTime[9];     // Loop time in seconds per animation frame
liquidTex[9][30];        // Texture handles (9 types, 30 frames max)
liquidTexLoaded[9];       // Loading state flags
```

**Texture Path Format**:
```
Base: "XTextures\\%s\\%02d.blp"
Example: "XTextures\\River\\01.blp"
```

**Frame Selection**:
```c
float loopPosition = (currentTime / loopTime) * 30.0f;
int frame = ROUND(loopPosition) % 30;
return liquidTex[type][frame];
```

### Texture Update System

#### Global State Tracking

```c
CMap::riverDiffTexUpdated = false;   // River texture animation
CMap::oceanDiffTexUpdated = false;   // Ocean texture animation
```

**Note**: The `riverDiffTexUpdated` and `oceanDiffTexUpdated` flags track whether liquid textures need texture lookups refresh. These are reset in `PrepareRenderLiquid()`.

### Liquid Type Constants

Based on `GetLiquidTexture()` analysis:

```c
LIQUID_COUNT = 8          // Maximum liquid types
MAX_FRAMES = 30           // Animation frames per texture

// Liquid types
#define LIQUID_WATER     0
#define LIQUID_OCEAN     4  
#define LIQUID_MAGMA     2
#define LIQUID_SLIME     3
#define LIQUID_RIVER     8
```

## Culling and Visibility

### Chunk Liquid Culling

#### CWorldScene::CullChunkLiquid

**Address**: 0x0066d290

Frustum culling for terrain liquid chunks.

**Process**:
1. Get bounding box from chunk position and liquid height
2. Check frustum culling
3. Check clip buffer culling
4. Move visible liquids to `visLiquidList`
5. Skip culled liquids

**Bounding Box Calculation**:
```c
void CChunkLiquid::GetAaBox(CChunkLiquid* this, CAaBox* box) {
    // X, Y bounds from chunk position (chunk->field_0x3c)
    box->b.x = chunkPos.x;
    box->b.y = chunkPos.y;
    box->t.x = chunkPos.x + CHUNK_SIZE;
    box->t.y = chunkPos.y + CHUNK_SIZE;
    
    // Z bounds from liquid height
    box->b.z = height.l;  // Min height
    box->t.z = height.h;  // Max height
}
```

### Distance-Based Sorting

Liquid chunks are sorted by distance for proper rendering order:

```c
int distance = ROUND(0.1f * chunkY - BASE_OFFSET);
if (distance < 0) distance = 0;
if (distance > 25) distance = 25;  // Max sorting distance

// Add to appropriate distance bucket
sortTable.table[distance].liquidList[liquidType]->Add(chunk);
```

## Liquid Query System

### Liquid Status Query

#### CWorld::QueryLiquidStatus

**Address**: 0x00664e70 (terrain), 0x00688240 (WMO)

Queries liquid type at a specific position.

```c
void CWorld::QueryLiquidStatus(
    C3Vector* position,      // World position
    uint* liquidType,        // Output: Liquid type
    float* height,           // Output: Liquid height
    C3Vector* flowDir        // Output: Flow direction
);
```

### Fishability Query

#### CWorld::QueryLiquidFishable

**Address**: 0x00688060

Determines if a position is in fishable water.

```c
bool CWorld::QueryLiquidFishable(C3Vector* position);
```

### Liquid Sounds

#### CWorld::QueryLiquidSounds

**Address**: 0x00664e90

Gets appropriate sound for liquid type.

```c
void CWorld::QueryLiquidSounds(uint liquidType, SoundEntry* sounds);
```

## API Reference

### Core Functions

| Function | Address | Purpose |
|----------|---------|---------|
| `CWorldScene::AddChunkLiquid` | 0x0066b120 | Add liquid to render list |
| `CWorldScene::CullChunkLiquid` | 0x0066d290 | Frustum culling |
| `CWorldScene::PrepareRenderLiquid` | 0x0066a590 | Pre-render setup |
| `CMap::GetLiquidTexture` | 0x006736b0 | Load animated texture |
| `CMapObj::RenderLiquid_0` | 0x0069e4b0 | WMO liquid rendering |
| `CMapObj::RenderExteriorWater_0` | 0x0069e7a0 | Terrain water rendering |
| `CMapObj::RenderMagma` | 0x0069e930 | Lava/magma rendering |
| `CMapObj::RenderWaterIndices_0` | 0x0069e370 | Index generation |
| `CMap::AllocChunkLiquid` | 0x00691860 | Allocate terrain liquid |
| `CMap::FreeChunkLiquid` | 0x00691960 | Free terrain liquid |
| `CChunkLiquid::GetAaBox` | 0x006766f0 | Get bounding box |

### Data Structures

```c
// Liquid vertex formats
CGxVertexPNCT0:  0x24 bytes (Terrain water)
CGxVertexPCT0:   0x18 bytes (Magma/lava)

// Flow animation
SWFlowv:         28 bytes (7 floats)

// WMO liquid
SMOLVert:        4 bytes (union: height OR st coords)
SMOLTile:        2 bytes (liquid type + flags)

// Terrain liquid
CChunkLiquid:    0x328 bytes (height grid + flow)
```

## Rendering Considerations

### Alpha Blending

Water rendering uses alpha blending for transparency:

```c
GxRsSet(GxRs_TexBlend0, 3);  // Enable alpha blending
```

### Texture Filtering

Liquid textures use appropriate filtering based on settings:

```c
if (CWorld::enables < 0) {
    filter = GxTex_Anisotropic;
} else if (CWorld::enables & 0x800000) {
    filter = GxTex_LinearMipLinear;
} else {
    filter = GxTex_LinearMipNearest;
}
```

### Performance

- Liquid chunks are sorted by distance (25 distance buckets)
- Tile-based culling skips empty tiles
- Animated textures use pre-loaded frames (up to 30)
- Flow vertices are reused for animation

## Historical Context

The liquid rendering system in Alpha 0.5.3 represents an early implementation that evolved significantly through later releases:

- **Alpha 0.5.3**: Basic liquid with simple height grids and animated textures
- **Vanilla 1.12.x**: Enhanced with better shader effects
- **TBC/WotLK**: Added underwater fog, reflections, and improved particles

---

**Note**: This documentation is based on reverse engineering analysis of the leaked World of Warcraft Alpha 0.5.3 client binary using Ghidra.
