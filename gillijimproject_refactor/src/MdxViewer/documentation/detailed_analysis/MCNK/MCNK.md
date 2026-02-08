# WoW Alpha 0.5.3 (Build 3368) MCNK Chunk - Complete Analysis

## Overview

This document provides a comprehensive analysis of the MCNK (Map CHuNK) chunk in World of Warcraft Alpha 0.5.3 (Build 3368, Dec 11 2003), based on Ghidra reverse engineering of `WoWClient.exe`. The MCNK chunk represents a single terrain tile within an ADT (Area Data) file and contains all terrain geometry, texture layers, shadows, liquids, and object references for that tile.

## Related Ghidra Functions

| Function | Address | Purpose |
|----------|---------|---------|
| [`CMapChunk::CMapChunk()`](#cmapchunk-constructor) | 0x00698510 | MCNK class constructor |
| [`CWorldScene::AddMapChunk()`](#addmapchunk) | 0x0066b050 | Adds chunk to render queue |
| [`CWorldScene::RenderChunks()`](#renderchunks) | 0x0066de50 | Main chunk rendering loop |
| [`CMap::AllocChunkLiquid()`](#allocchunkliquid) | 0x00691860 | Allocates liquid vertex data |
| [`AddChunkLiquid()`](#addchunkliquid) | 0x0066b120 | Adds liquid to chunk |
| [`CChunkTex::CChunkTex()`](#cchunktex) | 0x00697820 | Texture chunk constructor |
| [`CChunkLayer::CChunkLayer()`](#cchunklayer) | 0x00697870 | Layer chunk constructor |
| [`CMapChunk::CreateDetailDoodads()`](#createdetaildoodads) | - | Creates detail doodads |

---

## MCNK Header Structure (Alpha 0.5.3)

The MCNK header in Alpha 0.5.3 is **128 bytes** and contains offset pointers to all sub-chunks. Unlike later versions, offsets are relative to the **end of the MCNK header** (offset 0x80).

### Header Fields

```c
struct SMChunkHeader_Alpha {
    uint32_t flags;              // 0x00: Chunk flags (see below)
    uint32_t indexX;              // 0x04: X position in chunk grid (0-15)
    uint32_t indexY;             // 0x08: Y position in chunk grid (0-15)
    float radius;                // 0x0C: Chunk bounding radius
    uint32_t nLayers;             // 0x10: Number of texture layers
    uint32_t nDoodadRefs;         // 0x14: Number of doodad references
    uint32_t offsHeight;         // 0x18: Offset to MCVT (height vertices)
    uint32_t offsNormal;         // 0x1C: Offset to MCNR (normals)
    uint32_t offsLayer;          // 0x20: Offset to MCLY (layers)
    uint32_t offsRefs;           // 0x24: Offset to MCRF (references)
    uint32_t offsAlpha;          // 0x28: Offset to MCAL (alpha maps)
    uint32_t sizeAlpha;          // 0x2C: Size of alpha map data
    uint32_t offsShadow;         // 0x30: Offset to MCSH (shadow map)
    uint32_t sizeShadow;         // 0x34: Size of shadow map data
    uint32_t areaId;             // 0x38: Area ID (used for zone ID)
    uint32_t nMapObjRefs;        // 0x3C: Number of WMO references
    uint16_t holes;              // 0x40: Hole mask (bitmask for holes)
    uint16_t pad0;               // 0x42: Padding
    uint16_t predTex[8];         // 0x44: Predicted texture levels (LOD)
    uint8_t noEffectDoodad[8];   // 0x54: Doodads without effects
    uint32_t offsSndEmitters;    // 0x5C: Offset to MCSE (sound emitters)
    uint32_t nSndEmitters;       // 0x60: Number of sound emitters
    uint32_t offsLiquid;         // 0x64: Offset to MCLQ (liquid data)
    uint8_t pad1[24];            // 0x68: Unused/padding
    uint32_t McnkChunksSize;     // 0x80: Total size of MCNK sub-chunks
    uint8_t pad2[16];            // 0x84: Unused/padding
    uint32_t MclqOffset;         // 0x94: MCLQ offset (additional)
    uint8_t pad3[36];            // 0x98: Unused/padding
};
```

### SMChunkFlags Enumeration

```c
enum SMChunkFlags {
    FLAG_SHADOW      = 0x01,  // Chunk has shadow data
    FLAG_IMPASS      = 0x02,  // Chunk is impassable
    FLAG_LQ_RIVER    = 0x04,  // Contains river liquid
    FLAG_LQ_OCEAN    = 0x08,  // Contains ocean liquid
    FLAG_LQ_MAGMA    = 0x10,  // Contains magma/lava liquid
};
```

---

## CMapChunk Class Analysis (Ghidra)

The `CMapChunk` class is the runtime representation of an MCNK chunk. The constructor reveals the internal structure:

### Constructor Analysis

```cpp
CMapChunk * __thiscall CMapChunk::CMapChunk(CMapChunk *this)
{
    // Base class initialization
    CMapBaseObj::CMapBaseObj((CMapBaseObj *)this);
    
    // Initialize link lists for scene graph
    this->sceneLink = {0x0, 0x0};           // Scene graph link
    this->doodadDefLinkList = {0, 0, 0};   // Doodad definitions
    this->mapObjDefLinkList = {0, 0, 0};    // Map object definitions
    this->entityLinkList = {0, 0, 0};       // Entity links
    this->lightLinkList = {0, 0, 0};        // Light links
    
    // Sound emitter list
    TSList<class_CMapSoundEmitter>::InitializeTerminator();
    this->soundEmitterList.field_0x4 = 0x4c;
    
    // Grid position
    this->aIndex = {0, 0};                  // ADT grid indices
    this->sOffset = {0, 0};                 // Sub-chunk offsets
    this->cOffset = {0, 0};                 // Chunk offsets
    
    // Random seed for procedural effects
    NTempest::CRndSeed::SetSeed(&this->rSeed, 0);
    
    // ========================================
    // TERRAIN DATA BUFFERS
    // ========================================
    
    // 145 vertices (9x9 outer + 8x8 inner)
    // This matches MCVT format: 81 outer + 64 inner = 145
    memset(this->vertexList, 0, 145 * sizeof(C3Vector));
    
    // 256 planes for normal computation
    // This corresponds to 16x16 quad subdivision
    for (int i = 0; i < 256; i++) {
        this->planeList[i].n = {0.0f, 0.0f, 1.0f};
        this->planeList[i].d = 0.0f;
    }
    
    // Texture/shadow resources
    this->shadowTexture = (CChunkTex *)0x0;
    this->shadowGxTexture = (CGxTex *)0x0;
    this->shaderTexture = (CChunkTex *)0x0;
    this->shaderGxTexture = (CGxTex *)0x0;
    
    // Detail doodads
    this->detailDoodadInst = (CDetailDoodadInst *)0x0;
    
    // 4 liquid layers (rivers, oceans, magma, slime)
    this->liquids[0] = (CChunkLiquid *)0x0;
    this->liquids[1] = (CChunkLiquid *)0x0;
    this->liquids[2] = (CChunkLiquid *)0x0;
    this->liquids[3] = (CChunkLiquid *)0x0;
    
    // Layer count
    this->nLayers = 0;
    
    // Graphics buffer
    this->gxBuf = (CGxBuf *)0x0;
    
    // Async loading
    this->asyncObject = (CAsyncObject *)0x0;
    
    // Virtual function table
    *(undefined ***)this = &_vftable_;
    
    return this;
}
```

### Key CMapChunk Data Members

```cpp
class CMapChunk : public CMapBaseObj {
    // Position/grid info
    C3Vector aIndex;           // ADT grid position (x, y)
    C3Vector sOffset;          // Sub-chunk offsets
    C3Vector cOffset;          // Chunk offsets
    
    // Terrain geometry
    C3Vector vertexList[145];   // 9x9 + 8x8 = 145 vertices (MCVT data)
    C4Plane planeList[256];     // 16x16 = 256 planes (for normals)
    
    // Graphics resources
    CChunkTex *shadowTexture;   // Shadow map texture
    CGxTex *shadowGxTexture;   // GPU shadow texture
    CChunkTex *shaderTexture;  // Shader texture
    CGxTex *shaderGxTexture;   // GPU shader texture
    
    // Detail doodads
    CDetailDoodadInst *detailDoodadInst;  // Instance data
    
    // Liquid layers
    CChunkLiquid *liquids[4];   // 4 possible liquid types
    
    // Layer system
    uint32_t nLayers;           // Number of texture layers
    CChunkLayer *layers;        // Array of texture layers
    
    // Graphics buffer
    CGxBuf *gxBuf;              // GPU buffer
    
    // Scene graph links
    TSLink<CMapChunk> sceneLink;            // Scene graph
    TSExplicitList<CMapBaseObjLink> doodadDefLinkList;   // Doodads
    TSExplicitList<CMapBaseObjLink> mapObjDefLinkList;  // WMOs
    TSExplicitList<CMapBaseObjLink> entityLinkList;     // Entities
    TSExplicitList<CMapBaseObjLink> lightLinkList;      // Lights
    TSList<class_CMapSoundEmitter> soundEmitterList;     // Sound emitters
    
    // Procedural
    TSHashObjectChunk_CLightList rSeed;      // Random seed
};
```

---

## Sub-Chunks

### MCVT - Height Vertices

**Size**: 580 bytes (145 floats × 4 bytes)

**Format**: Unlike later versions, Alpha MCVT stores all outer vertices first, then all inner vertices:

```c
struct SMChunkVertex_Alpha {
    float outer[9*9];    // 81 outer vertices (corners of quads)
    float inner[8*8];    // 64 inner vertices (center of quads)
                        // Total: 145 floats
};
```

**Vertex Grid Layout** (9×9 outer with 8×8 inner):
```
Outer vertices: (x,y) where x∈[0,8], y∈[0,8] = 81 vertices
Inner vertices: (x,y) where x∈[0,7], y∈[0,7] = 64 vertices
```

**Height Values**: Unlike later versions, Alpha MCVT contains **absolute height values**, not values relative to the chunk header.

---

### MCNR - Normals

**Size**: 448 bytes (145 normals × 3 bytes + 13 bytes padding)

**Format**: Same order as MCVT (outer first, then inner):

```c
struct SMNormal {
    uint8_t n[145][3];    // Compressed normals (X, Y, Z)
    uint8_t pad[13];      // Padding to align to 4 bytes
};
```

**Normal Encoding**: Each component is an 8-bit signed value where:
- 127 = +1.0
- -127 = -1.0
- 0 = 0.0

---

### MCLY - Texture Layers

**Size**: Variable (depends on number of layers)

**Structure**:

```c
struct SMLayer {
    uint32_t textureId;    // Index into MTEX chunk
    uint32_t props;        // Layer properties (only use_alpha_map implemented)
    uint32_t offsAlpha;    // Offset to alpha map within MCAL
    uint16_t effectId;     // Effect ID
    uint8_t pad[2];       // Padding
};
```

**Layer Properties Flags**:
- `0x1`: USE_ALPHA_MAP - Use alpha blending for this layer

**Rendering Order**: Layers are rendered in order (first to last), so layer 0 is typically the ground layer and subsequent layers are detail/texture overlays.

---

### MCAL - Alpha Maps

**Size**: Variable (stored in `sizeAlpha` field)

**Format**: Raw 8-bit alpha values (0-255). Alpha map dimensions are typically 64×64 per layer, though this can vary.

**Storage**: Each layer's alpha data is stored sequentially. The offset for each layer's alpha data is stored in the corresponding MCLY entry.

---

### MCRF - Object References

**Size**: Variable (depends on number of references)

**Format**: Since Alpha ADTs don't have MMDX/MWMO chunks, MCRF entries directly index into MDNM and MONM chunks:

```c
struct SMChunkRef {
    uint32_t doodadIndices[];   // Indices into MDNM chunk (M2 models)
    uint32_t wmoIndices[];     // Indices into MONM chunk (WMOs)
};
```

**Note**: The split between doodad and WMO indices is determined by the `nDoodadRefs` field in the header.

---

### MCSH - Shadow Map

**Size**: Variable (stored in `sizeShadow` field)

**Format**: 8-bit shadow intensity values (0-255). Typical size is 64×64 (4096 bytes) for the shadow map.

**Purpose**: Self-shadowing for the terrain chunk, computed at import time.

---

### MCLQ - Liquid Data

**Size**: Variable

**Structure** (from Ghidra `AllocChunkLiquid`):

```c
struct CChunkLiquid {
    float height[2];           // Liquid height range (min, max)
    SWFlowv flowvs[2];         // 2 flow vertices
    TSLink<CChunkLiquid> sceneLink;      // Scene graph link
    TSLink<CChunkLiquid> lameAssLink;    // Internal link
};

struct SWFlowv {
    // Flow vertex data (direction, speed, etc.)
    float x, y, z;            // Position
    float dirX, dirY;         // Flow direction
    float amplitude;          // Wave amplitude
    float frequency;          // Wave frequency
};
```

**Liquid Types** (determined by flags):
- FLAG_LQ_RIVER (0x04): River
- FLAG_LQ_OCEAN (0x08): Ocean
- FLAG_LQ_MAGMA (0x10): Magma/Lava

---

### MCSE - Sound Emitters

**Size**: Variable

**Purpose**: Positional sound emitters within the chunk.

**Structure** (inferred from usage):
```c
struct SMChunkSoundEmitter {
    uint32_t soundId;         // Sound ID to play
    float position[3];        // Emitter position
    float radius;             // Sound radius
};
```

---

## Rendering Pipeline (Ghidra Analysis)

### CWorldScene::RenderChunks

The main chunk rendering loop:

```cpp
void __fastcall CWorldScene::RenderChunks(void)
{
    // Initialize world matrix (identity)
    C44Matrix cMat = identity;
    
    // Get first visible chunk
    CMapChunk *chunk = sortTable.visChunkList.head;
    
    while (chunk != NULL) {
        // Check if terrain rendering is enabled
        if ((CWorld::enables & 2) != 0) {
            // Calculate chunk position relative to camera
            C3Vector offset = chunk->position - camPos;
            
            // Set world transform
            NTempest::C44Matrix::Translate(&cMat, &offset);
            GxXformSet(GxXform_World, &cMat);
            
            // Select lighting for chunk
            CMap::SelectLight((CMapBaseObj *)chunk);
            
            // Render the chunk
            CMapChunk::Render(chunk);
        }
        
        // Handle detail doodads
        if (chunk->detailDoodadInst == NULL) {
            // Create detail doodads if needed
            CMapChunk::CreateDetailDoodads(chunk);
        }
        
        chunk = chunk->next;
    }
}
```

### Chunk Sorting

The `CWorldScene::AddMapChunk` function adds chunks to a distance-based sorting table:

```cpp
void __fastcall CWorldScene::AddMapChunk(CMapChunk *chunk, float camDist)
{
    if (chunk == NULL) return;
    
    // Calculate sort bucket (0-25)
    int bucket = (int)(WORLD_CONSTANT * camDist - OFFSET);
    bucket = clamp(bucket, 0, 25);
    
    // Insert into bucket's chunk list
    // Chunks are rendered from back to front for proper blending
}
```

The sort table has 26 buckets (0-25) based on distance from camera, allowing for efficient LOD and culling decisions.

---

## Data Buffer Sizes (from Constructor)

| Buffer | Size | Purpose |
|--------|------|---------|
| `vertexList` | 145 × sizeof(C3Vector) | MCVT height data |
| `planeList` | 256 × sizeof(C4Plane) | Pre-computed plane equations |
| `liquids` | 4 × sizeof(CChunkLiquid) | Up to 4 liquid layers |
| CChunkLiquid | 0x338 (824) bytes | Liquid vertex + flow data |

---

## Alpha vs. Later Version Differences

| Aspect | Alpha 0.5.3 | Later Versions (1.x+) |
|--------|-------------|----------------------|
| Header Size | 128 bytes | 128 bytes (same) |
| MCVT Order | Outer first, then inner | Interleaved 9-8-9-8 pattern |
| MCVT Values | Absolute heights | Relative to header base |
| MCNR Order | Outer first, then inner | Interleaved pattern |
| MCNR Size | 448 bytes | Same (145 × 3 + 13 pad) |
| MCRF | Direct indices | Points to MMDX/MWMO indices |
| MTEX | Not present in ADT | Separate MTEX chunk |
| MDNM/MONM | In MHDR area | Separate chunks |
| Position | In header (float[3]) | In MHDR offset |

---

## Implementation Notes

### Reading MCNK in Alpha Format

```c
void ReadMcnkChunk(FileStream *adtFile, int mcnkOffset, int headerSize) {
    int dataStart = mcnkOffset + 8;  // Skip 'MCNK' + size
    
    // Read 128-byte header
    SMChunkHeader_Alpha header;
    ReadBytes(adtFile, dataStart, &header, sizeof(header));
    
    // Read sub-chunks using offsets
    // Offsets are relative to dataStart (after header)
    
    // MCVT - Heights
    int mcvtOffset = dataStart + header.offsHeight;
    float heights[145];
    ReadFloats(adtFile, mcvtOffset, 145, heights);
    
    // MCNR - Normals
    int mcnrOffset = dataStart + header.offsNormal;
    SMNormal normals;
    ReadBytes(adtFile, mcnrOffset, &normals, sizeof(normals));
    
    // MCLY - Layers
    int mclyOffset = dataStart + header.offsLayer;
    int mclySize = header.offsRefs - header.offsLayer;
    ReadLayers(adtFile, mclyOffset, mclySize);
    
    // MCRF - References
    int mcrfOffset = dataStart + header.offsRefs;
    ReadReferences(adtFile, mcrfOffset, header.nDoodadRefs, header.nMapObjRefs);
    
    // MCAL - Alpha maps (if present)
    if (header.offsAlpha > 0 && header.sizeAlpha > 0) {
        int mcalOffset = dataStart + header.offsAlpha;
        ReadAlphaMaps(adtFile, mcalOffset, header.sizeAlpha);
    }
    
    // MCSH - Shadow map (if present)
    if (header.offsShadow > 0 && header.sizeShadow > 0) {
        int mcshOffset = dataStart + header.offsShadow;
        ReadShadowMap(adtFile, mcshOffset, header.sizeShadow);
    }
    
    // MCLQ - Liquid (if present)
    if (header.offsLiquid > 0) {
        int mclqOffset = dataStart + header.offsLiquid;
        ReadLiquidData(adtFile, mclqOffset);
    }
    
    // MCSE - Sound emitters (if present)
    if (header.offsSndEmitters > 0 && header.nSndEmitters > 0) {
        int mcseOffset = dataStart + header.offsSndEmitters;
        ReadSoundEmitters(adtFile, mcseOffset, header.nSndEmitters);
    }
}
```

### Converting Alpha to LK Format

When converting Alpha MCNK to Legion/Classic format:

1. **Reorder MCVT vertices** from [81 outer, 64 inner] to interleaved [9-8-9-8...]
2. **Convert MCNR normals** to same interleaved order
3. **Rebuild offsets** - LK uses absolute offsets, not relative
4. **Add chunk flags** for LOD system
5. **Copy MCLY/MCAL through directly** (same format)
6. **Recalculate MCRF** to point to MMDX/MWMO indices

---

## References

- **wowdev.wiki/Alpha**: https://wowdev.wiki/Alpha
- **wowdev.wiki/ADT/v18**: ADT format for 1.x clients
- **Ghidra Analysis**: WoWClient.exe (0.5.3.3368)
- **Source Code**: `src/gillijimproject-csharp/WowFiles/Alpha/McnkAlpha.cs`

---

## Change Log

| Date | Description |
|------|-------------|
| 2024-XX-XX | Initial analysis from Ghidra reverse engineering |
| 2024-XX-XX | Added CMapChunk class structure |
| 2024-XX-XX | Added rendering pipeline documentation |
| 2024-XX-XX | Added sub-chunk detailed analysis |
