# WMO v14 Format Specification (Ghidra-Verified)

**Source**: Fresh Ghidra decompilation of WoWClient.exe (0.5.3.3368) with Wowae.pdb  
**Date**: 2025-12-28  
**Trust Level**: HIGH - All data from actual binary analysis

---

## Critical Corrections from Old Documentation

| Item | Old Docs Said | **Ghidra Says** | Function |
|------|---------------|-----------------|----------|
| MOGI size | 32 bytes | **40 bytes** (`/0x28`) | CreateDataPointers |
| MOPY size | 2 bytes | **4 bytes** (`>>2`) | CMapObjGroup::CreateDataPointers |
| Index chunk | MOVI | **MOIN** | CMapObjGroup::CreateDataPointers |
| MODD size | Various | **40 bytes** (`/0x28`) | CreateDataPointers |
| MFOG chunk | Missing | **48 bytes** (`/0x30`) | CreateDataPointers |
| MOLV chunk | Missing | **Present** (lightmap UVs) | CMapObjGroup::CreateDataPointers |

---

## 1. File Loading (AsyncPostloadCallbackHeader @ 0x006ae0f0)

```c
// Version check - MUST be 14 (0x0E)
if (*(int *)((int)param_1 + 0x1b4) != 0xe) {
    SysMsgPrintf(SYSMSG_FATAL, 2, "MAPWRONGVERSION: %s", ...);
    return;
}

// MOMO container check - MUST be 'MOMO' (0x4D4F4D4F)
if (*(int *)((int)param_1 + 0x1b8) != 0x4d4f4d4f) {
    SErrDisplayError("mapObj->fileHeader.iffChunkHeader...");
}
```

**Key Requirements:**
- Version at offset 0x1b4 must equal **14 (0x0E)**
- MOMO magic at offset 0x1b8 must equal **0x4D4F4D4F**

---

## 2. Root Chunk Parsing (CreateDataPointers @ 0x006aeaa0)

All sizes and offsets are **Ghidra-verified**:

### Chunk Order (Sequential)

| # | Chunk | FourCC | Entry Size | Calculation |
|---|-------|--------|------------|-------------|
| 1 | MOHD | 0x4D4F4844 | Header | Single struct |
| 2 | MOTX | 0x4D4F5458 | Variable | String table |
| 3 | MOMT | 0x4D4F4D54 | **44 bytes** | `size / 0x2c` |
| 4 | MOGN | 0x4D4F474E | Variable | String table |
| 5 | MOGI | 0x4D4F4749 | **40 bytes** | `size / 0x28` |
| 6 | MOPV | 0x4D4F5056 | 12 bytes | `size / 0xc` |
| 7 | MOPT | 0x4D4F5054 | **20 bytes** | `size / 0x14` |
| 8 | MOPR | 0x4D4F5052 | 8 bytes | `size >> 3` |
| 9 | MOLT | 0x4D4F4C54 | **32 bytes** | `size >> 5` |
| 10 | MODS | 0x4D4F4453 | **32 bytes** | `size >> 5` |
| 11 | MODN | 0x4D4F444E | Variable | String table |
| 12 | MODD | 0x4D4F4444 | **40 bytes** | `size / 0x28` |
| 13 | MFOG | 0x4D464F47 | **48 bytes** | `size / 0x30` |
| 14 | MCVP | 0x4D435650 | 16 bytes | `size >> 4` (optional) |

### Ghidra Source Code

```c
void __thiscall CMapObj::CreateDataPointers(CMapObj *this) {
    pData = this->data + 0x14;
    
    // MOHD - Header
    pSVar1 = ReadChunkHeader(this, &pData, 0x4d4f4844);
    this->header = (SMOHeader *)pData;
    pData = pData + pSVar1->size;
    
    // MOTX - Texture names
    pSVar1 = ReadChunkHeader(this, &pData, 0x4d4f5458);
    this->textureNameList = (char *)pData;
    this->textureNameCount = pSVar1->size;
    pData = pData + pSVar1->size;
    
    // MOMT - Materials (44 bytes each!)
    pSVar1 = ReadChunkHeader(this, &pData, 0x4d4f4d54);
    this->materialList = (SMOMaterial *)pData;
    this->materialCount = pSVar1->size / 0x2c;  // ← 44 bytes!
    pData = pData + pSVar1->size;
    
    // MOGN - Group names
    pSVar1 = ReadChunkHeader(this, &pData, 0x4d4f474e);
    this->groupNameList = (char *)pData;
    this->groupNameCount = pSVar1->size;
    pData = pData + pSVar1->size;
    
    // MOGI - Group info (40 bytes each!)
    pSVar1 = ReadChunkHeader(this, &pData, 0x4d4f4749);
    this->groupInfoList = (SMOGroupInfo *)pData;
    this->groupCount = pSVar1->size / 0x28;  // ← 40 bytes!
    pData = pData + pSVar1->size;
    
    // MOPV - Portal vertices (12 bytes each)
    pSVar1 = ReadChunkHeader(this, &pData, 0x4d4f5056);
    this->portalVertexList = (C3Vector *)pData;
    this->portalVertexCount = pSVar1->size / 0xc;
    pData = pData + pSVar1->size;
    
    // MOPT - Portals (20 bytes each)
    pSVar1 = ReadChunkHeader(this, &pData, 0x4d4f5054);
    this->portalList = (SMOPortal *)pData;
    this->portalCount = pSVar1->size / 0x14;  // ← 20 bytes!
    pData = pData + pSVar1->size;
    
    // MOPR - Portal refs (8 bytes each)
    pSVar1 = ReadChunkHeader(this, &pData, 0x4d4f5052);
    this->portalRefList = (SMOPortalRef *)pData;
    this->portalRefCount = pSVar1->size >> 3;
    pData = pData + pSVar1->size;
    
    // MOLT - Lights (32 bytes each)
    pSVar1 = ReadChunkHeader(this, &pData, 0x4d4f4c54);
    this->lightList = (SMOLight *)pData;
    this->lightCount = pSVar1->size >> 5;
    pData = pData + pSVar1->size;
    
    // MODS - Doodad sets (32 bytes each)
    pSVar1 = ReadChunkHeader(this, &pData, 0x4d4f4453);
    this->doodadSetList = (SMODoodadSet *)pData;
    this->doodadSetCount = pSVar1->size >> 5;
    pData = pData + pSVar1->size;
    
    // MODN - Doodad names
    pSVar1 = ReadChunkHeader(this, &pData, 0x4d4f444e);
    this->doodadNameList = (char *)pData;
    this->doodadNameCount = pSVar1->size;
    pData = pData + pSVar1->size;
    
    // MODD - Doodad defs (40 bytes each!)
    pSVar1 = ReadChunkHeader(this, &pData, 0x4d4f4444);
    this->doodadDefList = (SMODoodadDef *)pData;
    this->doodadDefCount = pSVar1->size / 0x28;  // ← 40 bytes!
    pData = pData + pSVar1->size;
    
    // MFOG - Fog (48 bytes each)
    pSVar1 = ReadChunkHeader(this, &pData, 0x4d464f47);
    this->fogList = (SMOFog *)pData;
    this->fogCount = pSVar1->size / 0x30;  // ← 48 bytes!
    pData = pData + pSVar1->size;
    
    // MCVP - Convex volume planes (optional, 16 bytes each)
    pSVar1 = ReadOptionalChunkHeader(this, &pData, 0x4d435650);
    if (pSVar1 != NULL) {
        this->convexVolumePlanes = (C4Plane *)pData;
        this->volumePlaneCount = pSVar1->size >> 4;
    }
}
```

---

## 3. Group Chunk Parsing (CMapObjGroup::CreateDataPointers @ 0x006af2d0)

### Group Chunk Order (Sequential, ALL REQUIRED)

| # | Chunk | FourCC | Entry Size | Calculation |
|---|-------|--------|------------|-------------|
| 1 | MOPY | 0x4D4F5059 | **4 bytes** | `size >> 2` |
| 2 | MOVT | 0x4D4F5654 | 12 bytes | `size / 0xc` |
| 3 | MONR | 0x4D4F4E52 | 12 bytes | `size / 0xc` |
| 4 | MOTV | 0x4D4F5456 | 8 bytes | `size >> 3` |
| 5 | **MOLV** | 0x4D4F4C56 | 8 bytes | `size >> 3` |
| 6 | **MOIN** | 0x4D4F494E | 2 bytes | `size >> 1` |
| 7 | MOBA | 0x4D4F4241 | **24 bytes** | `size / 0x18` |

**CRITICAL**: Index chunk is **MOIN** (0x4D4F494E), NOT MOVI!

### Ghidra Source Code

```c
void __thiscall CMapObjGroup::CreateDataPointers(CMapObjGroup *this, uchar *param_1) {
    // MOPY - Polygon info (4 bytes each!)
    if (*(int *)param_1 != 0x4d4f5059) {
        SErrDisplayError("pIffChunk->token == 'MOPY'");
    }
    this->polyList = (SMOPoly *)(param_1 + 8);
    this->polyCount = *(uint *)(param_1 + 4) >> 2;  // ← 4 bytes per poly!
    
    // MOVT - Vertices (12 bytes each)
    if (*(int *)(next) != 0x4d4f5654) {
        SErrDisplayError("pIffChunk->token == 'MOVT'");
    }
    this->vertexList = (C3Vector *)(next + 8);
    this->vertexCount = *(uint *)(next + 4) / 0xc;
    
    // MONR - Normals (12 bytes each)
    if (*(int *)(next) != 0x4d4f4e52) {
        SErrDisplayError("pIffChunk->token == 'MONR'");
    }
    this->normalList = (C3Vector *)(next + 8);
    this->normalCount = *(uint *)(next + 4) / 0xc;
    
    // MOTV - Texture UVs (8 bytes each)
    if (*(int *)(next) != 0x4d4f5456) {
        SErrDisplayError("pIffChunk->token == 'MOTV'");
    }
    this->textureVertexList = (C2Vector *)(next + 8);
    this->textureVertexCount = *(uint *)(next + 4) >> 3;
    
    // MOLV - Lightmap UVs (8 bytes each) - MISSING FROM OLD DOCS!
    if (*(int *)(next) != 0x4d4f4c56) {
        SErrDisplayError("pIffChunk->token == 'MOLV'");
    }
    this->lightmapVertexList = (C2Vector *)(next + 8);
    this->lightmapVertexCount = *(uint *)(next + 4) >> 3;
    
    // MOIN - Indices (2 bytes each) - NOT MOVI!
    if (*(int *)(next) != 0x4d4f494e) {
        SErrDisplayError("pIffChunk->token == 'MOIN'");
    }
    this->indexList = (ushort *)(next + 8);
    this->indexCount = *(uint *)(next + 4) >> 1;
    
    // MOBA - Batches (24 bytes each)
    if (*(int *)(next) != 0x4d4f4241) {
        SErrDisplayError("pIffChunk->token == 'MOBA'");
    }
    this->batchList = (SMOBatch *)(next + 8);
    this->batchCount = *(uint *)(next + 4) / 0x18;
    
    CreateOptionalDataPointers(this, next);
}
```

---

## 4. Optional Group Chunks (CreateOptionalDataPointers @ 0x006af4d0)

### Flag-Controlled Optional Chunks

| Flag | Chunk | FourCC | Entry Size |
|------|-------|--------|------------|
| 0x200 | MOLR | 0x4D4F4C52 | 2 bytes (`>>1`) |
| 0x800 | MODR | 0x4D4F4452 | 2 bytes (`>>1`) |
| 0x001 | MOBN | 0x4D4F424E | **16 bytes** (`>>4`) |
| 0x001 | MOBR | 0x4D4F4252 | 2 bytes (`>>1`) |
| 0x400 | MPBV | 0x4D504256 | Variable |
| 0x400 | MPBP | 0x4D504250 | Variable |
| 0x400 | MPBI | 0x4D504249 | Variable |
| 0x400 | MPBG | 0x4D504247 | Variable |
| 0x004 | MOCV | 0x4D4F4356 | **4 bytes** (`>>2`) |
| 0x002 | Lightmaps | - | Via CreateLightmapPointers |
| 0x1000 | MLIQ | 0x4D4C4951 | **38-byte header** |

### MLIQ Liquid Structure (Ghidra-verified)

```c
// From CreateOptionalDataPointers
if ((this->flags & 0x1000) != 0) {
    if (*(int *)param_1 != 0x4d4c4951) {  // 'MLIQ'
        SErrDisplayError("pIffChunk->token == ' MLIQ'");
    }
    
    // Liquid header - 38 bytes (0x26)
    (this->liquidVerts).x = *(long *)(param_1 + 0x08);  // verts X
    (this->liquidVerts).y = *(long *)(param_1 + 0x0c);  // verts Y
    (this->liquidTiles).x = *(long *)(param_1 + 0x10);  // tiles X
    (this->liquidTiles).y = *(long *)(param_1 + 0x14);  // tiles Y
    (this->liquidCorner).x = *(float *)(param_1 + 0x18); // corner X
    (this->liquidCorner).y = *(float *)(param_1 + 0x1c); // corner Y
    (this->liquidCorner).z = *(float *)(param_1 + 0x20); // corner Z
    this->liquidMtlId = *(ushort *)(param_1 + 0x24);     // material ID
    this->liquidVertexList = (SMOLVert *)(param_1 + 0x26); // vertex data starts
    this->liquidTileList = liquidVertexList + (vertsX * vertsY);
}
```

### MLIQ Header Structure (38 bytes)

```c
struct MLIQHeader {
    /*0x00*/ uint32 chunkId;      // 'MLIQ' (in chunk header)
    /*0x04*/ uint32 chunkSize;    // (in chunk header)
    /*0x08*/ int32  vertsX;       // Vertex grid X dimension
    /*0x0C*/ int32  vertsY;       // Vertex grid Y dimension
    /*0x10*/ int32  tilesX;       // Tile grid X dimension
    /*0x14*/ int32  tilesY;       // Tile grid Y dimension
    /*0x18*/ float  cornerX;      // Base corner X
    /*0x1C*/ float  cornerY;      // Base corner Y
    /*0x20*/ float  cornerZ;      // Base corner Z
    /*0x24*/ uint16 materialId;   // Liquid material
    /*0x26*/ // Vertex data starts here
};
```

---

## 5. Structure Sizes Summary

### Ghidra-Verified Sizes

| Structure | Size | Source |
|-----------|------|--------|
| SMOMaterial | **44 bytes** | `size / 0x2c` |
| SMOGroupInfo | **40 bytes** | `size / 0x28` |
| SMOPortal | **20 bytes** | `size / 0x14` |
| SMOPortalRef | 8 bytes | `size >> 3` |
| SMOLight | **32 bytes** | `size >> 5` |
| SMODoodadSet | **32 bytes** | `size >> 5` |
| SMODoodadDef | **40 bytes** | `size / 0x28` |
| SMOFog | **48 bytes** | `size / 0x30` |
| SMOPoly | **4 bytes** | `size >> 2` |
| SMOBatch | **24 bytes** | `size / 0x18` |
| CAaBspNode | **16 bytes** | `size >> 4` |
| CImVector (color) | 4 bytes | `size >> 2` |
| C3Vector | 12 bytes | `size / 0xc` |
| C2Vector | 8 bytes | `size >> 3` |
| C4Plane | 16 bytes | `size >> 4` |
| Index (MOIN) | 2 bytes | `size >> 1` |

---

## 6. Key Ghidra Addresses

| Function | Address | Purpose |
|----------|---------|---------|
| AsyncPostloadCallbackHeader | 0x006ae0f0 | Version & MOMO check |
| AsyncPostloadCallback | 0x006ae270 | Triggers CreateData |
| CreateData | 0x006ae6e0 | Main WMO setup |
| CreateDataPointers | 0x006aeaa0 | Root chunk parsing |
| CreateAllGroups | 0x006ae8d0 | Group iteration |
| CMapObjGroup::CreateDataPointers | 0x006af2d0 | Group chunk parsing |
| CreateOptionalDataPointers | 0x006af4d0 | Optional chunk parsing |
| CreateMaterials | 0x006aed30 | Material setup |
| ReadChunkHeader | 0x006ae990 | Chunk header reading |

---

## 7. Corrections to Our Parser Code

Based on Ghidra analysis, these corrections are needed:

### WmoV14Parser.cs Issues

1. **MOPY size**: Parser assumes 2 bytes, Ghidra shows **4 bytes**
2. **Index chunk**: Parser looks for MOVI, client expects **MOIN**
3. **MOGI size**: May be using 32 bytes, should be **40 bytes**
4. **Missing MOLV**: Lightmap UVs chunk not handled
5. **MODD size**: Check against **40 bytes**

### Recommended Fixes

```csharp
// WRONG (old assumption):
this->polyCount = chunk.size / 2;   // 2 bytes per MOPY

// CORRECT (Ghidra-verified):
this->polyCount = chunk.size / 4;   // 4 bytes per MOPY!

// WRONG (old assumption):
if (chunkId == "MOVI") ...

// CORRECT (Ghidra-verified):
if (chunkId == "MOIN") ...  // Different chunk name!
```

---

*This document contains ONLY Ghidra-verified information from decompilation of WoWClient.exe (0.5.3.3368). No old community documentation was used.*
