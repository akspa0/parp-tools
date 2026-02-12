# Task 1: MCNK Header Structure Analysis

## Overview
Analysis of WoW Alpha 0.5.3 (build 3368) MCNK chunk header structure using Ghidra decompilation.

## Key Findings

### MCNK Fourcc Verification
- **Constant**: `0x4D434E4B` ("MCNK" in forward byte order)
- **Location**: Verified in [`CMapChunk::SyncLoad`](SyncLoad:698d90) @ 0x00698d90
- **Source**: String reference @ 0x008a126c: `"iffChunk->token=='MCNK'"`

### Functions Analyzed

#### 1. CMapChunk::CMapChunk (Constructor)
**Address**: 0x00698510

**Key Class Members Identified**:
```c
class CMapChunk : public CMapBaseObj {
    TSLink<CMapChunk> sceneLink;
    TSExplicitList<CMapBaseObjLink,8> doodadDefLinkList;
    TSExplicitList<CMapBaseObjLink,8> mapObjDefLinkList;
    TSExplicitList<CMapBaseObjLink,8> entityLinkList;
    TSExplicitList<CMapBaseObjLink,8> lightLinkList;
    TSList<CMapSoundEmitter> soundEmitterList;
    
    C2iVector aIndex;              // Chunk array index
    C2iVector sOffset;             // Some offset
    C2iVector cOffset;             // Another offset
    NTempest::CRndSeed rSeed;     // Random seed
    
    C3Vector normalList[0x91];     // 145 normals
    C3Vector vertexList[0x91];     // 145 vertices
    C4Plane planeList[0x100];      // 256 planes
    
    CChunkTex* shadowTexture;
    CGxTex* shadowGxTexture;
    CChunkTex* shaderTexture;
    CGxTex* shaderGxTexture;
    CDetailDoodadInst* detailDoodadInst;
    
    CChunkLiquid* liquids[4];      // Up to 4 liquid layers
    uint32 nLayers;                // Number of texture layers
    
    CGxBuf* gxBuf;
    CAsyncObject* asyncObject;
    
    uint32 fileOffset;             // Offset in WDT file
    uint32 fileSize;               // Size of MCNK chunk
    // ... more fields
};
```

#### 2. CMapChunk::SyncLoad
**Address**: 0x00698d90

**Decompiled Code**:
```c
void __thiscall CMapChunk::SyncLoad(
    CMapChunk *this,
    SMChunk **param_1,
    SMLayer **param_2,
    uchar **param_3,
    uchar **param_4)
{
  uchar *puVar1;
  
  if (CMap::wdtFile == NULL) {
    _SErrDisplayError_24(0x85100000, sourceFile, 0x2b3, "CMap::wdtFile", NULL, 1);
  }
  
  // Seek to chunk position and read
  SFile::SetFilePointer(CMap::wdtFile, this->fileOffset, NULL, 0);
  SFile::Read(CMap::wdtFile, &DAT_00e6e5e4, this->fileSize, NULL, NULL, NULL);
  
  // Verify MCNK magic
  if (DAT_00e6e5e4 != 0x4d434e4b) {  // "MCNK"
    _SErrDisplayError_24(0x85100000, sourceFile, 700, "iffChunk->token=='MCNK'", NULL, 1);
  }
  
  // Set SMChunk pointer (header starts at offset +8)
  *param_1 = (SMChunk *)&DAT_00e6e5ec;
  
  // Verify MCLY magic
  if (DAT_00e6ea70 != 0x4d434c59) {  // "MCLY"
    _SErrDisplayError_24(0x85100000, sourceFile, 0x2cb, "iffChunk->token=='MCLY'", NULL, 1);
  }
  
  // Set layer pointer
  *param_2 = (SMLayer *)&DAT_00e6ea78;
  
  // Calculate shadow and alpha texture pointers
  puVar1 = (uchar *)(DAT_00e6ea74 + 0xe6ea80 + *(int *)(&DAT_00e6ea7c + DAT_00e6ea74));
  *param_3 = puVar1;
  *param_4 = puVar1 + (*param_1)->sizeShadow;
}
```

**Key Insights**:
1. MCNK chunk is read directly from WDT file (monolithic format confirmed)
2. Magic verification uses forward byte order: `0x4D434E4B` not reversed
3. SMChunk header starts at offset +8 from chunk start (after 4-byte magic + 4-byte size)
4. MCLY (layer) subchunk follows MCNK header
5. Shadow and alpha textures are calculated using `sizeShadow` field from SMChunk

### SMChunk Structure (Partial)

Based on the decompilation, the SMChunk structure includes:

```c
struct SMChunk {
    // Header fields accessed in code
    uint32 sizeShadow;     // Size of shadow texture data
    // ... more fields to be determined
};
```

### Memory Offsets Observed

From the decompilation addresses:
- `0x00e6e5e4`: MCNK magic (start of chunk)
- `0x00e6e5ec`: SMChunk header start (+8 bytes)
- `0x00e6ea70`: MCLY magic (appears to be +1164 bytes from chunk start)
- `0x00e6ea78`: MCLY data start (+8 bytes from MCLY magic)

This suggests the MCNK header is approximately **1164 bytes** before MCLY starts, but this needs verification by examining the actual header struct.

## Next Steps

To complete the MCNK header documentation, we need to:
1. Find the SMChunk structure definition or examine more field accesses
2. Trace all field offsets used when parsing MCNK data
3. Document which offsets are absolute vs relative
4. Identify flag fields and their meanings
5. Determine exact header size

## Cross-References

Functions that reference MCNK:
- `CMapChunk::SyncLoad` @ 0x00698d90 (primary parser)
- `CMapChunk::Create` @ 0x00698e99 (uses same string check)

## Confidence Level

**Medium** - We have confirmed:
- ✅ MCNK magic number (0x4D434E4B) in forward byte order
- ✅ Basic chunk reading flow
- ✅ Relationship to MCLY subchunk
- ✅ Class structure with 145 vertices/normals (matches 81 outer + 64 inner = 145 total)

Still investigating:
- ⏳ Complete SMChunk field layout
- ⏳ Exact header size
- ⏳ Flag field meanings
- ⏳ Absolute vs relative offset fields
