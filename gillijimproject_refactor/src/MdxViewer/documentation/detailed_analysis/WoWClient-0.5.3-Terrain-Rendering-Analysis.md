# WoW Alpha 0.5.3 (Build 3368) Terrain Rendering Analysis

## Overview

This document provides a deep analysis of the terrain rendering system in WoW Alpha 0.5.3 (Build 3368, Dec 11 2003), based on Ghidra reverse engineering of WoWClient.exe. It covers texture layering, clamping, blending, and the complete rendering pipeline.

## Related Functions

| Function | Address | Purpose |
|----------|---------|---------|
| [`LoadWdt`](LoadWdt) | 0x0067fde0 | WDT file loading |
| [`MdxReadTextures`](MdxReadTextures) | 0x0044e310 | Texture loading for models |
| [`LoadTexture`](LoadTexture) | 0x006afc40 | Individual texture loading |
| [`ProcessTextures`](ProcessTextures) | Various | Texture processing |
| [`ComputeFogBlend`](ComputeFogBlend) | 0x00689b40 | Fog blending calculations |
| [`GetFog`](GetFog) | 0x0068b1c0 | Fog retrieval |
| [`LoadLightsAndFog`](LoadLightsAndFog) | 0x006c4110 | Light and fog data loading |

---

## WDT File Format (World Map Definition)

### Chunk Structure

```
WDT File:
├── MVER (Version)
│   └── uint32_t version
├── MPHD (Map Header)
│   ├── uint32_t flags
│   ├── uint32_t something
│   └── ... more fields
├── MAIN (Area Information)
│   ├── uint32_t areaCount
│   └── AREA[areaCount] entries
├── MODF (Map Object Definitions)
│   └── Map object definitions
└── doodadNames (Null-terminated strings)
```

### MPHD Chunk Structure

```c
struct SMPHeader {
    uint32_t flags;           // Map flags
    uint32_t something;       // Unknown field
    uint32_t something2;      // Unknown field
    // ... more fields
};
```

### MAIN Chunk Structure

```c
struct SMAArea {
    uint32_t areaId;         // Area ID
    uint32_t flags;          // Area flags
    uint32_t nameOffset;    // Offset into name table
    uint32_t waterLevel;    // Water level for this area
    uint32_t waterType;     // Water type
    // ... more fields
};
```

---

## Texture Layer System

### Model Texture Loading (MdxReadTextures)

The texture system in models uses a chunked loading approach:

```c
void MdxReadTextures(uchar* data, uint size, uint flags,
                     CModelComplex* model,
                     CStatus* status) {
    // Seek to TEXS chunk
    uint* texData = (uint*)MDLFileBinarySeek(data, size, 'TEXS');
    
    if (texData != NULL) {
        uint texCount = *texData / 0x10c;
        
        if (*texData != texCount * 0x10c) {
            // Error: Invalid texture section size
        }
        
        // Ensure texture array capacity
        if (model->m_textures.count < texCount) {
            TSFixedArray::Reserve(model->m_textures, 
                                 texCount - model->m_textures.count, 1);
        }
        
        // Process each texture
        ProcessTextures(texData + 1, texCount, flags, status,
                       model->m_textures.data);
    }
}
```

### Texture Structure

```c
struct CModelTexture {
    uint32_t flags;          // Texture flags
    uint32_t type;           // Texture type (replaceable ID)
    char filename[260];      // Texture filename (MAX_PATH)
    uint32_tunk000;         // Unknown field
};
```

### Texture Flags

| Flag | Value | Description |
|------|-------|-------------|
| TEXFLAG_WRAP_S | 0x1 | Wrap S coordinate |
| TEXFLAG_WRAP_T | 0x2 | Wrap T coordinate |
| TEXFLAG_CUBE_MAP | 0x4 | Cubemap texture |
| TEXFLAG_COMPRESSED | 0x8 | Compressed format |
| TEXFLAG_SRGB | 0x10 | sRGB colorspace |

---

## Texture Clamping and Grid Line Prevention

### UV Coordinate System

The engine uses standard UV mapping with the following characteristics:

```c
struct C2Vector {
    float u;  // +0x00: U coordinate (horizontal)
    float v;  // +0x04: V coordinate (vertical)
};
```

### Clamping Implementation

From WMO material analysis, clamping flags are stored in the material:

```c
struct SMOMaterial {
    uint32_t flags;
    
    // Clamping flags
    // F_CLAMP_S (0x40): Clamp S coordinate
    // F_CLAMP_T (0x80): Clamp T coordinate
    
    uint32_t blendMode;
    uint32_t textureId1;
    uint32_t textureId2;
    uint32_t textureId3;
    // ... colors
};
```

### Preventing Grid Lines

**Key Techniques:**

1. **Texture Wrap Modes**
   - `TEXFLAG_WRAP_S` / `TEXFLAG_WRAP_T`: Standard wrapping
   - `F_CLAMP_S` / `F_CLAMP_T`: Clamping for terrain edges
   - Must be set on BOTH the texture AND material

2. **Mipmap Bias**
   - The engine uses mipmapping for LOD
   - `Script_GetTerrainMip` / `Script_SetTerrainMip` control mip levels
   - Proper mip filtering prevents aliasing at edges

3. **Texture Filtering**
   - Linear filtering by default
   - Anisotropic filtering for slopes
   - Minification filter: `GX_TEXF_LINEAR_MIPLINEAR`

4. **Seamless Texture Tiling**
   - Textures should have matching edge pixels
   - Normal maps must be seamless
   - Height maps require matching gradients

### Grid Line Causes

1. **UV Discontinuity**
   - Vertices not shared between terrain chunks
   - UV seams at tile boundaries
   - Solution: Ensure vertex sharing

2. **Precision Issues**
   - Float precision at tile edges
   - Solution: Use double precision for boundary calculations

3. **Z-Fighting**
   - Depth buffer precision issues
   - Solution: Offset geometry or increase depth precision

4. **Mipmap Artifacts**
   - Blurring at high-frequency transitions
   - Solution: Use trilinear filtering, proper mip selection

---

## Blend Mode System

### Material Blend Modes

| Value | Name | Description | Equation |
|-------|------|-------------|----------|
| 0 | Opaque | No blending | `src` |
| 1 | Transparent | Alpha blending | `srcAlpha, 1-srcAlpha` |
| 2 | Blend | Standard blend | `srcAlpha, 1-srcAlpha` |
| 3 | Add | Additive | `srcAlpha, 1` |
| 4 | AddAlpha | Additive alpha | `srcAlpha, 1-srcAlpha` |
| 5 | Modulate | Multiply | `dstColor, 0` |

### Alpha Test

For terrain with transparency (foliage, etc.):

```c
// Alpha test pseudocode
if (textureAlpha < alphaTestRef) {
    discard;
}
```

**Alpha Test Reference**: Typically 0.125 (1/8) for foliage

---

## Fog System

### Fog Types

| Type | Description | Implementation |
|------|-------------|----------------|
| Linear | Distance-based | `mix(color, fogColor, (dist - start) / (end - start))` |
| Exponential | Density-based | `color * pow(2, -density * dist)` |
| Exponential2 | Squared density | `color * pow(2, -density² * dist²)` |

### Fog Functions

```c
// Get fog for a specific distance
void GetFog(float distance, float* fogFactor);

// Compute fog blend factor
void ComputeFogBlend(float distance, float* blendFactor);

// Query camera fog settings
void QueryCameraFog(Camera* cam, FogSettings* settings);

// Query map object fog
void QueryMapObjFog(MapObj* obj, FogSettings* settings);
```

### Fog Structure

```c
struct FogData {
    float color[4];          // RGBA fog color
    float start;             // Fog start distance
    float end;               // Fog end distance
    float density;           // Fog density (for exp/exp2)
    uint32_t type;           // Fog type
};
```

### Fog Integration

```c
// Shader pseudocode
float fogFactor;

if (fog.type == LINEAR) {
    fogFactor = clamp((fog.end - dist) / (fog.end - fog.start), 0, 1);
} else if (fog.type == EXP) {
    fogFactor = exp(-fog.density * dist);
} else if (fog.type == EXP2) {
    fogFactor = exp(-fog.density * fog.density * dist * dist);
}

fogFactor = clamp(fogFactor, 0, 1);
finalColor = mix(fog.color, sceneColor, fogFactor);
```

---

## Terrain Rendering Pipeline

### Level of Detail (LOD)

The engine uses multiple LOD levels:

1. **Distance-based LOD**
   - Close terrain: Full detail mesh
   - Medium: Reduced detail
   - Far: Simplified mesh

2. **Texture LOD**
   - Mipmapping for textures
   - Anisotropy for slopes

3. **Shader LOD**
   - Simplified shaders for distance
   - Reduced pixel operations

### Chunk Loading

```c
// Terrain chunk structure
struct TerrainChunk {
    uint32_t flags;          // Chunk flags
    uint32_t headerSize;     // Header size
    float position[3];      // Chunk position
    uint32_t size;          // Chunk data size
    
    // Sub-chunks
    MCVT (height data)
    MCLY (layer data)
    MCSH (shadow data)
    MCAL (alpha map)
    MCNR (normal data)
    // ... more
};
```

### Sub-Chunk Details

#### MCVT - Height Vertices

```
MCVT Chunk:
├── uint32_t magic     // 'MCVT' (0x5443564d)
├── uint32_t size      // Byte count
└── float heights[145] // 17x17 grid + 9 inner points
                      // Total: 145 height values
```

#### MCLY - Texture Layers

```
MCLY Chunk:
├── uint32_t magic     // 'MCLY' (0x594c434d)
├── uint32_t size      // Byte count
└── MCLYEntry entries[]
```

**MCLYEntry Structure**:

```c
struct MCLYEntry {
    uint32_t textureId;      // Texture ID (from MTEX)
    uint32_t flags;         // Layer flags
    uint32_t alphaMapId;    // Alpha map reference
    uint32_t unknown;        // Unknown field
};
```

**Layer Flags**:

| Flag | Value | Description |
|------|-------|-------------|
| MCLY_FLAG_USE_ALPHA | 0x1 | Use alpha map |
| MCLY_FLAG_SPECULAR | 0x2 | Specular enabled |
| MCLY_FLAG_SHADOW | 0x4 | Cast shadows |

#### MCAL - Alpha Maps

```
MCAL Chunk:
├── uint32_t magic     // 'MCAL' (0x4c41434d)
├── uint32_t size      // Byte count
└── uint8_t alphaData[] // Alpha values (0-255)
```

**Alpha Map Dimensions**: 64x64 or 256x256 per layer

#### MCSH - Shadow Maps

```
MCSH Chunk:
├── uint32_t magic     // 'MCSH' (0x4853434d)
├── uint32_t size      // Byte count
└── uint8_t shadowData[] // Shadow values
```

---

## Complete Rendering Pipeline

### Pre-Render

1. **Load terrain data**
   - Parse WDT for active tiles
   - Load ADT files for active tiles
   - Parse sub-chunks (MCVT, MCLY, MCAL, etc.)

2. **Build render data**
   - Generate mesh from MCVT heights
   - Build alpha map from MCAL data
   - Create shadow maps from MCSH

3. **Texture preparation**
   - Load textures from MCLY entries
   - Generate mipmaps
   - Apply clamping/wrapping flags

### Render

1. **Culling**
   - Frustum culling per chunk
   - Occlusion culling (if enabled)
   - Distance culling

2. **LOD Selection**
   - Select mesh LOD based on distance
   - Select texture mip level
   - Enable/disable features

3. **Render Passes**

   **Pass 1: Opaque Geometry**
   - Render terrain base layer
   - Apply first texture layer
   - Write depth
   
   **Pass 2: Alpha-Blended Layers**
   - For each additional layer:
     - Sample alpha map
     - Blend with current
     - Render if alpha > 0
   
   **Pass 3: Shadows**
   - Render shadow map overlays
   - Blend with terrain
    
   **Pass 4: Fog**
   - Apply fog based on distance
   - Mix fog color with terrain

### Post-Render

1. **Memory cleanup**
   - Unload distant tiles
   - Release unused textures
   - Clear render lists

---

## DBC Integration

### Map DBC

The Map.dbc file defines world maps:

```c
struct MapDBC {
    uint32_t id;           // Map ID
    char name[64];         // Map name
    uint32_t areaId;       // Starting area
    float fogStart;        // Fog start
    float fogEnd;          // Fog end
    float fogDensity;      // Fog density
    uint32_t mapType;      // Map type (battleground, etc.)
    uint32_t unk4;         // Unknown
};
```

### AreaTable DBC

The AreaTable.dbc defines zones:

```c
struct AreaTableDBC {
    uint32_t id;           // Area ID
    uint32_t mapId;        // Parent map ID
    char name[64];         // Area name
    uint32_t flags;        // Area flags
    float fogStart;        // Zone-specific fog
    float fogEnd;
    float fogDensity;
    uint32_t unk;          // Unknown
};
```

---

## Edge Case Handling

### Grid Line Prevention Checklist

1. [ ] **Vertex Sharing**
   - Ensure adjacent chunks share vertices
   - UV coordinates match at edges
   - Normals averaged at edges

2. [ ] **Texture Coordinates**
   - Wrap mode matches on adjacent tiles
   - Clamp mode for tile edges (if needed)
   - UV scaling consistent

3. [ ] **Alpha Map Seams**
   - Alpha values match at edges
   - Blend between layers smooth

4. [ ] **Shadow Map Continuity**
   - Shadow values consistent
   - No shadow gaps at edges

5. [ ] **Mipmap Boundary**
   - Mipmaps generated properly
   - No mipmap seams visible
   - Anisotropy handles slopes

6. [ ] **Z-Buffer Precision**
   - Near/far plane set properly
   - Polygon offset applied
   - Depth test enabled

---

## Optimization Techniques

### Texture Caching

```c
// Texture cache structure
struct TextureCache {
    CTextureHash textures[256];
    uint32_t cacheHits;
    uint32_t cacheMisses;
    
    // LRU eviction
    TSList<CTexture*> lruList;
};
```

### Async Loading

```
Loading Pipeline:
1. Main thread: Queue texture requests
2. IO thread: Read texture from disk
3. Decoding thread: Decode BLP format
4. GPU upload: Transfer to graphics memory
5. Main thread: Use texture
```

### Vertex Buffer Management

- Static terrain: Single static vertex buffer
- Dynamic terrain: Update changed chunks only
- Compression: Store heights as 16-bit where possible

---

## Debugging Tools

### Console Commands

| Command | Description |
|---------|-------------|
| `ShowTerrain` | Toggle terrain rendering |
| `DetailDoodadTest` | Test detail doodads |
| `SetTerrainMip` | Set terrain mip level |
| `ToggleFog` | Toggle fog rendering |

### Visual Debugging

1. **Wireframe Mode**: Show mesh edges
2. **Texture Coordinates**: Visualize UVs
3. **Alpha Map**: Show blending
4. **Shadow Map**: Show shadows
5. **Normals**: Show vertex normals

---

## References

- **WDT Format**: [`LoadWdt`](LoadWdt) at 0x0067fde0
- **Texture Loading**: [`MdxReadTextures`](MdxReadTextures) at 0x0044e310
- **Fog System**: [`ComputeFogBlend`](ComputeFogBlend) at 0x00689b40
- **Lighting**: [`LoadLightsAndFog`](LoadLightsAndFog) at 0x006c4110
