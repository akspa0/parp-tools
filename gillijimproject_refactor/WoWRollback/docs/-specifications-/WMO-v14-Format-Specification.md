# WMO v14 Format Specification

**Version**: WMO v14 (Alpha 0.5.3.3368)  
**Sources**: Ghidra reverse engineering of WoWClient.exe + existing parser code  
**Date**: 2025-12-28

---

## Table of Contents

1. [Overview](#1-overview)
2. [File Structure](#2-file-structure)
3. [Root Chunks (Inside MOMO)](#3-root-chunks-inside-momo)
4. [Group Chunks (Inside MOGP)](#4-group-chunks-inside-mogp)
5. [Portal System](#5-portal-system)
6. [Doodad System](#6-doodad-system)
7. [Rendering Pipeline](#7-rendering-pipeline)
8. [v14 vs v17 Differences](#8-v14-vs-v17-differences)
9. [Conversion Guide](#9-conversion-guide)

---

## 1. Overview

WMO v14 is a **monolithic single-file format** used in WoW Alpha 0.5.3. Unlike later versions (v17+), all groups are embedded in a single file rather than split into separate `_XXX.wmo` files.

### Key Characteristics

- **Version**: 14 (0x0E)
- **Format**: Single monolithic file
- **Container**: Uses MOMO wrapper chunk
- **Materials**: 44 bytes (not 64 like v17)
- **Origin**: Similar to Quake 3 BSP architecture

### Ghidra Confirmation

```c
// From CMapObj::WaitLoad @ 0x00694970
if (this->fileHeader.version == 0xE) {  // Version 14!
    // Valid WMO v14
}
```

---

## 2. File Structure

### Top-Level Layout

```
MVER (4 bytes) - Version = 14
MOMO (container) - Contains all other chunks
  ├── MOHD - Header
  ├── MOTX - Texture names
  ├── MOMT - Materials (44 bytes each!)
  ├── MOGN - Group names
  ├── MOGI - Group info (32 bytes each)
  ├── MOSB - Skybox name
  ├── MOPV - Portal vertices
  ├── MOPT - Portal info
  ├── MOPR - Portal references
  ├── MOVV - Visible block vertices
  ├── MOVB - Visible block list
  ├── MOLT - Lights
  ├── MODS - Doodad sets
  ├── MODN - Doodad names
  ├── MODD - Doodad definitions
  └── MOGP[n] - Group data (one per group)
        ├── [Header 64 bytes]
        ├── MOVT - Vertices
        ├── MOVI - Indices
        ├── MONR - Normals
        ├── MOTV - UVs
        ├── MOPY - Face info
        ├── MOBA - Render batches
        ├── MOBN - BSP nodes
        └── MOBR - BSP face refs
```

### Chunk Header

```c
struct SIffChunk {
    uint32 token;   // FourCC (reversed on disk)
    uint32 size;    // Data size (excludes 8-byte header)
};
```

---

## 3. Root Chunks (Inside MOMO)

### MVER - Version

```c
struct MVER {
    uint32 version;  // Always 14 for Alpha
};
```

### MOHD - Header

```c
struct MOHD_v14 {
    /*0x00*/ uint32 nTextures;       // Number of textures
    /*0x04*/ uint32 nGroups;         // Number of groups
    /*0x08*/ uint32 nPortals;        // Number of portals
    /*0x0C*/ uint32 nLights;         // Number of lights
    /*0x10*/ uint32 nDoodadNames;    // Number of M2 model names
    /*0x14*/ uint32 nDoodadDefs;     // Number of doodad instances
    /*0x18*/ uint32 nDoodadSets;     // Number of doodad sets
    /*0x1C*/ CArgb  ambColor;        // Ambient color (BGRA)
    /*0x20*/ uint32 wmoID;           // WMO ID (for DBC lookup)
    /*0x24*/ CAaBox boundingBox;     // 24 bytes (min/max C3Vector)
    /*0x3C*/ uint32 flags;           // WMO flags
};
// Total: 64 bytes
```

### MOTX - Texture Names

Null-terminated texture path strings:

```
"Dungeons\Textures\brick.blp\0"
"Dungeons\Textures\floor.blp\0"
...
```

### MOMT - Materials (44 bytes each in v14!)

**Critical**: v14 uses 44-byte materials, NOT 64-byte like v17!

```c
struct SMOMaterial_v14 {
    /*0x00*/ uint32 version;         // V14 only - material version!
    /*0x04*/ uint32 flags;           // Material flags
    /*0x08*/ uint32 blendMode;       // Blend mode (0=opaque, 1=alpha key, etc)
    /*0x0C*/ uint32 texture1Offset;  // Offset into MOTX for diffuse texture
    /*0x10*/ uint32 sidnColor;       // Emissive/self-illumination color
    /*0x14*/ uint32 frameSidnColor;  // Runtime emissive (frame)
    /*0x18*/ uint32 texture2Offset;  // Offset into MOTX for env map
    /*0x1C*/ uint32 diffColor;       // Diffuse color tint
    /*0x20*/ uint32 groundType;      // Ground type for footsteps
    /*0x24*/ uint8  padding[8];      // Padding to 44 bytes
};
// Total: 44 bytes (NOT 64!)
```

#### Material Flags

```c
enum SMOMaterial_Flags {
    F_UNLIT      = 0x01,  // Disable lighting
    F_UNFOGGED   = 0x02,  // Disable fog
    F_TWOSIDED   = 0x04,  // Two-sided rendering
    F_EXT_LIGHT  = 0x08,  // Use exterior lighting
    F_NIGHT_ONLY = 0x10,  // Only visible at night
    F_WINDOW     = 0x20,  // Window (transparent interior)
};
```

### MOGN - Group Names

Null-terminated group name strings:

```
"entrance\0"
"main_hall\0"
...
```

### MOGI - Group Info (32 bytes each)

```c
struct SMOGroupInfo_v14 {
    /*0x00*/ uint32 flags;           // Group flags
    /*0x04*/ CAaBox boundingBox;     // 24 bytes (min/max)
    /*0x1C*/ int32  nameOffset;      // Offset into MOGN (-1 = no name)
};
// Total: 32 bytes
```

#### Group Flags (from Ghidra)

```c
enum GroupFlags {
    FLAG_BSP           = 0x0001,  // Has BSP tree
    FLAG_LIGHT_MAP     = 0x0002,  // Has lightmap
    FLAG_COLOR_VERTEX  = 0x0004,  // Has vertex colors
    FLAG_EXTERIOR      = 0x0008,  // Outdoor group
    FLAG_EXTERIOR_LIT  = 0x0040,  // Exterior lit
    FLAG_UNREACHABLE   = 0x0080,  // Unreachable
    FLAG_SHOW_SKY      = 0x0100,  // Render skybox
    FLAG_OCEAN         = 0x0200,  // Is ocean
    FLAG_LIQUID        = 0x1000,  // Has liquid (from Ghidra: flags & 0x1000)
    FLAG_INTERIOR      = 0x2000,  // Interior group
    FLAG_SLIME         = 0x4000,  // Slime liquid
    FLAG_MAGMA         = 0x8000,  // Magma liquid
};
```

### MOPV - Portal Vertices

```c
C3Vector portalVertices[];  // 12 bytes each (x, y, z floats)
```

### MOPT - Portal Info

```c
struct SMOPortal {
    uint16 startVertex;    // Index into MOPV
    uint16 count;          // Number of vertices
    C4Plane plane;         // Portal plane (normal + distance)
};
```

### MOPR - Portal References

```c
struct SMOPortalRef {
    uint16 portalIndex;    // Index into portal list
    uint16 groupIndex;     // Target group index
    int16  side;           // Which side of portal (-1/1)
    uint16 padding;
};
// 8 bytes each
```

### MOLT - Lights

```c
struct SMOLight {
    uint8  type;           // Light type (0=omni, 1=spot, 2=direct, 3=ambient)
    uint8  useAtten;       // Use attenuation
    uint8  padding[2];
    CArgb  color;          // Light color
    C3Vector position;     // Light position
    float  intensity;      // Light intensity
    float  attenStart;     // Attenuation start
    float  attenEnd;       // Attenuation end
};
```

### MODS - Doodad Sets

```c
struct SMODoodadSet {
    char   name[20];       // Set name (null-terminated)
    uint32 firstIndex;     // First doodad in MODD
    uint32 nDoodads;       // Number of doodads
    uint32 padding;
};
// 32 bytes each
```

### MODN - Doodad Names

Null-terminated MDX model path strings:

```
"World\...\torch.mdx\0"
"World\...\chair.mdx\0"
...
```

### MODD - Doodad Definitions

```c
struct SMODoodadDef {
    uint32 nameOffset;     // Offset into MODN
    uint32 flags;          // Doodad flags
    C3Vector position;     // Position (relative to WMO)
    C4Quaternion rotation; // Rotation quaternion (x,y,z,w)
    float  scale;          // Uniform scale
    CArgb  color;          // Color tint
};
// 40 bytes each
```

---

## 4. Group Chunks (Inside MOGP)

Each MOGP chunk contains a 64-byte header followed by sub-chunks.

### MOGP Header (64 bytes)

```c
struct SMOGroupHeader_v14 {
    /*0x00*/ uint32 nameOffset;       // Offset into MOGN
    /*0x04*/ uint32 descNameOffset;   // Descriptive name offset
    /*0x08*/ uint32 flags;            // Group flags
    /*0x0C*/ CAaBox boundingBox;      // 24 bytes (min/max)
    /*0x24*/ uint16 portalStart;      // First portal in MOPR
    /*0x26*/ uint16 portalCount;      // Number of portals
    /*0x28*/ uint16 transBatchCount;  // Transparent batch count
    /*0x2A*/ uint16 intBatchCount;    // Interior batch count
    /*0x2C*/ uint16 extBatchCount;    // Exterior batch count
    /*0x2E*/ uint16 padding;
    /*0x30*/ uint8  fogIndices[4];    // Fog indices
    /*0x34*/ uint32 liquidType;       // Liquid type
    /*0x38*/ uint32 groupID;          // WMO area table ID
    /*0x3C*/ uint32 unused[2];
};
// Total: 64 bytes (0x40)
```

### MOVT - Vertices

```c
C3Vector vertices[];  // 12 bytes each (x, y, z floats)
```

### MOVI - Indices

```c
uint16 indices[];  // Triangle indices (3 per face)
```

### MONR - Normals

```c
C3Vector normals[];  // 12 bytes each (x, y, z floats)
```

### MOTV - Texture Coordinates

```c
C2Vector uvs[];  // 8 bytes each (u, v floats)
```

### MOPY - Face Info (2 bytes each)

```c
struct SMOPoly {
    uint8 flags;       // Face flags
    uint8 materialId;  // Material index
};
```

#### Face Flags

```c
enum SMOPoly_Flags {
    F_UNK_0x01       = 0x01,
    F_NOCAMCOLLIDE   = 0x02,  // No camera collision
    F_DETAIL         = 0x04,  // Detail geometry
    F_COLLISION      = 0x08,  // Has collision
    F_HINT           = 0x10,  // Hint
    F_RENDER         = 0x20,  // Renderable
    F_COLLIDE_HIT    = 0x80,  // Collidable for projectiles
};
```

### MOBA - Render Batches (24 bytes each in v14)

**Ghidra-verified structure (2025-12-29):**

```c
struct SMOBatch_v14 {
    /*0x00*/ uint8  lightMap;        // Lightmap index
    /*0x01*/ uint8  materialId;      // Material index (CRITICAL: at 0x01, not 0x17!)
    /*0x02*/ uint8  reserved[12];    // Bounding box (unused in v14)
    /*0x0E*/ uint16 startIndex;      // First index in MOVI (CRITICAL: uint16, not uint32!)
    /*0x10*/ uint16 indexCount;      // Index count (faces * 3)
    /*0x12*/ uint16 minVertex;       // Min vertex index for batch
    /*0x14*/ uint16 maxVertex;       // Max vertex index for batch
    /*0x16*/ uint8  flags;           // Batch flags
    /*0x17*/ uint8  padding;         // Alignment
};
// Total: 24 bytes
```

> [!IMPORTANT]
> In v14, `materialId` is at offset **0x01** (not 0x17) and `startIndex` is at offset **0x0E** as **uint16** (not uint32).
> This was verified by Ghidra analysis of RenderGroupTex @ 0x0069d8c0.

### MOBN - BSP Nodes

```c
struct SMOBSPNode {
    int16  planetype;     // Plane type (0=YZ, 1=XZ, 2=XY, 3=leaf)
    int16  children[2];   // Child node indices (-1 = none)
    uint16 numFaces;      // Number of faces (leaf only)
    uint16 firstFace;     // First face index (leaf only)
    float  planeDist;     // Plane distance
};
// 16 bytes each
```

### MOBR - BSP Face References

```c
uint16 faceIndices[];  // Indices into face list for BSP leaves
```

### MOLR - Light References (optional)

```c
uint16 lightIndices[];  // Indices into MOLT
```

### MODR - Doodad References (optional)

```c
uint16 doodadIndices[];  // Indices into MODD
```

### MOCV - Vertex Colors (optional)

```c
CArgb vertexColors[];  // 4 bytes each (BGRA)
```

### MLIQ - Liquid Data (optional)

```c
struct SMOLiquidHeader {
    C2iVector vertexDims;   // Vertex grid dimensions
    C2iVector tileDims;     // Tile grid dimensions
    C3Vector  corner;       // Base corner position
    uint16    materialId;   // Liquid material
};
// Followed by vertex heights and tile flags
```

---

## 5. Portal System

Portals define visibility between WMO groups, enabling efficient indoor rendering.

### Portal Rendering Flow (from Ghidra @ 0x0069bf60)

```c
void RRenderThruPortals(CMapObj *this, uint groupIndex, ..., int depth) {
    CMapObjGroup *group = GetGroup(this, groupIndex, 0);
    
    for (int i = 0; i < group->portalCount; i++) {
        SMOPortalRef *ref = &this->portalRefList[group->portalStart + i];
        SMOPortal *portal = &this->portalList[ref->portalIndex];
        
        // Skip if same group
        if (ref->groupIndex == groupIndex) continue;
        
        // Transform portal to screen space
        RTransformPortal(this, portal, &portalExt, cpIgnore);
        
        // Check portal visibility with view frustum
        float dot = camPos.x * portal->plane.n.x + 
                    camPos.y * portal->plane.n.y + 
                    camPos.z * portal->plane.n.z + portal->plane.d;
        
        if (ref->side < 0) dot = -dot;
        if (dot < 0.0f) continue;  // Behind portal
        
        // Clip portal rect to current view rect
        CRect newRect = ClipRect(portalRect, viewRect);
        
        // Recursive render through portal
        CWorldScene::FrustumPush();
        CWorldScene::FrustumSet(&camCorners, &newRect);
        RRenderThruPortals(this, ref->groupIndex, depth + 1);
        CWorldScene::FrustumPop();
    }
}
```

### Portal Intersection (from Ghidra @ 0x00693d90)

```c
bool VectorIntersectPortal(CMapObj *this, C3Vector *rayStart, C3Vector *rayEnd, 
                           uint groupIndex, uint *hitGroup) {
    CMapObjGroup *group = GetGroup(this, groupIndex, 0);
    
    for (int i = 0; i < group->portalCount; i++) {
        SMOPortalRef *ref = &this->portalRefList[group->portalStart + i];
        SMOPortal *portal = &this->portalList[ref->portalIndex];
        
        // Test ray against portal polygon triangles
        for (int v = 1; v < portal->count - 1; v++) {
            C3Vector *v0 = &this->portalVertexList[portal->startVertex];
            C3Vector *v1 = &this->portalVertexList[portal->startVertex + v];
            C3Vector *v2 = &this->portalVertexList[portal->startVertex + v + 1];
            
            if (RayIntersectTri(rayStart, rayDir, v0, v1, v2, &t)) {
                if (t >= 0.0f && t <= 1.0f) {
                    *hitGroup = ref->groupIndex;
                    return true;
                }
            }
        }
    }
    return false;
}
```

---

## 6. Doodad System

Doodads are MDX models placed within WMO groups.

### Doodad Loading (from Ghidra @ 0x006a2640)

```c
void AddDoodad(CMapObjGroup *this, uint doodadIndex, C3Vector *position, uint flags) {
    // Load MDX model
    CModel *model = LoadDoodadModel(doodadNameTable[doodadIndex]);
    
    // Transform vertices by doodad position
    for (int i = 0; i < model->vertexCount; i++) {
        C3Vector vertex = model->vertices[i];
        vertex.x += position->x;
        vertex.y += position->y;
        vertex.z += position->z;
        
        this->geom->vertexList[baseVertex + i] = vertex;
    }
    
    // Copy normals, UVs, and indices
    // ...
}
```

### Doodad Set Selection

```c
void GetDoodadSet(CMapObj *this, uint setIndex) {
    SMODoodadSet *set = &this->doodadSets[setIndex];
    
    for (uint i = set->firstIndex; i < set->firstIndex + set->nDoodads; i++) {
        SMODoodadDef *def = &this->doodadDefs[i];
        // Instantiate doodad at def->position with def->rotation and def->scale
    }
}
```

---

## 7. Rendering Pipeline

### Group Rendering (from Ghidra @ 0x0069bd50)

```c
void CMapObj::RenderGroup(uint groupIndex, int param2, C44Matrix *viewMatrix, ...) {
    CMapObjGroup *group = GetGroup(this, groupIndex, 0);
    
    // Create lightmaps if needed
    CMapObjGroup::CreateLightmaps(group);
    
    // Render based on group flags
    if ((group->flags & 0x48) == 0) {
        // Exterior rendering
        RenderGroup_Ext(group);
    } else {
        // Interior rendering with lightmaps
        RenderGroup_Int(group);
    }
    
    // Render liquid if present
    if (group->flags & 0x1000) {
        RenderLiquid_0(this, group);
    }
    
    // Debug: render normals
    if (CWorld::enables & 0x40000000) {
        RenderGroupNormals(this, group);
    }
    
    // Debug: render portals
    if (CWorld::enables & 0x1000) {
        RenderPortals(this, group);
    }
}
```

### Render Modes

| Function | Address | Purpose |
|----------|---------|---------|
| `RenderGroup` | 0x0069bd50 | Main group render |
| `RenderGroup_Int` | 0x0069db70 | Interior with lightmap |
| `RenderGroup_Ext` | 0x0069da70 | Exterior rendering |
| `RenderGroupTex` | 0x0069d8c0 | Textured rendering |
| `RenderGroupLightmap` | 0x0069d770 | Lightmap only |
| `RenderGroupColorTex` | 0x0069d6f0 | Vertex color + texture |
| `RenderGroupBsp` | 0x0069df60 | BSP debug render |
| `RenderPortals` | 0x0069dc90 | Portal debug render |

---

## 8. v14 vs v17 Differences

### Structural Differences

| Aspect | v14 | v17+ |
|--------|-----|------|
| File layout | Monolithic | Root + group files |
| Material size | 44 bytes | 64 bytes |
| Material texture | `texture` at +0x01 in MOBA | `materialId` in MOMT |
| MOMO container | Yes | No |
| Shader field | None | In MOMT |
| Group files | Embedded | Separate `_XXX.wmo` |

### Material Structure Comparison

```c
// v14 MOMT - 44 bytes
struct SMOMaterial_v14 {
    uint32 version;          // +0x00 - V14 ONLY
    uint32 flags;            // +0x04
    uint32 blendMode;        // +0x08
    uint32 texture1Offset;   // +0x0C
    uint32 sidnColor;        // +0x10
    uint32 frameSidnColor;   // +0x14
    uint32 texture2Offset;   // +0x18
    uint32 diffColor;        // +0x1C
    uint32 groundType;       // +0x20
    uint8  padding[8];       // +0x24
};  // = 44 bytes

// v17 MOMT - 64 bytes
struct SMOMaterial_v17 {
    uint32 flags;            // +0x00
    uint32 shader;           // +0x04 - NEW
    uint32 blendMode;        // +0x08
    uint32 texture1Offset;   // +0x0C
    uint32 sidnColor;        // +0x10
    uint32 frameSidnColor;   // +0x14
    uint32 texture2Offset;   // +0x18
    uint32 diffColor;        // +0x1C
    uint32 groundType;       // +0x20
    uint32 texture3Offset;   // +0x24 - NEW
    uint32 color2;           // +0x28 - NEW
    uint32 flags2;           // +0x2C - NEW
    uint32 runTimeData[4];   // +0x30 - NEW
};  // = 64 bytes
```

### MOBA Batch Comparison

Both v14 and v17 use 24-byte MOBA entries, but field meanings differ:

```c
// v14 MOBA - texture field is material index
struct SMOBatch_v14 {
    uint8  lightMap;         // +0x00
    uint8  texture;          // +0x01 - MATERIAL INDEX
    // ...
};

// v17 MOBA - uses material_id_large
struct SMOBatch_v17 {
    // Different layout
    uint8  material_id_large; // Material for large triangles
};
```

---

## 9. Conversion Guide

### v14 → v17 Conversion

1. **Write MVER** with version = 17
2. **Write MOHD** (expand to v17 format)
3. **Write MOTX** (unchanged)
4. **Write MOMT** (expand 44 → 64 bytes):
   - Remove `version` field
   - Add `shader` field (default 0)
   - Add `texture3Offset`, `color2`, `flags2`, `runTimeData`
5. **Write MOGN** (unchanged)
6. **Write MOGI** (unchanged)
7. **Write MODS** (unchanged)
8. **Split groups** into separate `_XXX.wmo` files:
   - Each file: MVER + MOGP
   - MOGP contains: header + MOPY + MOVI + MOVT + MONR + MOTV + MOBA

### v17 → v14 Conversion

1. **Write MVER** with version = 14
2. **Write MOMO** container:
   - **MOHD** (copy, no changes needed)
   - **MOTX** (unchanged)
   - **MOMT** (shrink 64 → 44 bytes):
     - Add `version` field at start
     - Remove `shader`, `texture3Offset`, `color2`, `flags2`, `runTimeData`
   - **MOGN** (unchanged)
   - **MOGI** (unchanged)
   - **MODS, MODN, MODD** (unchanged)
   - **Merge group files** into MOGP chunks

### Code Reference

See `WmoV14ToV17Converter.cs`:

```csharp
private void WriteMaterialV17(BinaryWriter writer, WmoMaterial material) {
    writer.Write(material.Flags);           // flags
    writer.Write((uint)0);                  // shader (NEW)
    writer.Write(material.BlendMode);       // blendMode
    writer.Write(material.Texture1Offset);  // texture1
    writer.Write(material.EmissiveColor);   // sidnColor
    writer.Write((uint)0);                  // frameSidnColor
    writer.Write(material.Texture2Offset);  // texture2
    writer.Write(material.DiffuseColor);    // diffColor
    writer.Write(material.GroundType);      // groundType
    writer.Write((uint)0);                  // texture3 (NEW)
    writer.Write((uint)0);                  // color2 (NEW)
    writer.Write((uint)0);                  // flags2 (NEW)
    writer.Write(new byte[16]);             // runTimeData (NEW)
}
```

---

## Quick Reference Tables

### Chunk Sizes

| Chunk | v14 Size | v17 Size | Notes |
|-------|----------|----------|-------|
| MOMT entry | 44 bytes | 64 bytes | +20 bytes |
| MOGI entry | 32 bytes | 32 bytes | Same |
| MOBA entry | 24 bytes | 24 bytes | Same but different fields |
| MOGP header | 64 bytes | 68 bytes | +4 bytes |

### Key Addresses (Ghidra)

| Function | Address | Purpose |
|----------|---------|---------|
| `CMapObj::GetGroup` | 0x006947f0 | Get group by index |
| `CMapObj::GetGroupInfo` | 0x00694b30 | Get SMOGroupInfo |
| `CMapObj::WaitLoad` | 0x00694970 | Wait for async load |
| `CMapObj::ReadGroup` | 0x006aeea0 | Read group from file |
| `CMapObj::RenderGroup` | 0x0069bd50 | Render a group |
| `RRenderThruPortals` | 0x0069bf60 | Portal rendering |
| `VectorIntersectPortal` | 0x00693d90 | Portal ray intersection |
| `AddDoodad` | 0x006a2640 | Add doodad to group |

---

*Document generated from Ghidra analysis of WoWClient.exe (0.5.3.3368) with Wowae.pdb symbols and existing parser code in WmoV14Parser.cs and WmoV14ToV17Converter.cs.*
