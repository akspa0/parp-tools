# Task 3: MCLQ Liquid Data Structure Analysis

## Overview
Analysis of MCLQ liquid chunk structure in WoW Alpha 0.5.3 (build 3368).

## Key Findings

### CChunkLiquid Class Structure
**Size**: 0x338 bytes (824 bytes)

### Functions Analyzed

#### 1. CMap::AllocChunkLiquid
**Address**: 0x00691860

**Key Class Members Identified**:
```c
class CChunkLiquid {
    CFloatRange height;      // Min/max height range (.l = low, .h = high)
    SWFlowv flowvs[2];       // Flow vectors (2 entries)
    
    TSLink<CChunkLiquid> sceneLink;
    TSLink<CChunkLiquid> lameAssLink;
    
    CMapChunk* chunk;        // Parent chunk pointer
    
    // Vertex data (size calculated from 0x338 total - other fields)
    // verts array at end of structure
    
    // Total structure size: 0x338 bytes
};
```

#### 2. CWorldScene::AddChunkLiquid
**Address**: 0x0066b120

This function shows liquid types are categorized:
```c
void __fastcall CWorldScene::AddChunkLiquid(CChunkLiquid *param_1, uint param_2)
{
  if (param_2 > 3) {
    _SErrDisplayError("type < LQ_LAST");
  }
  // param_2 is liquid type (0-3)
  // Adds to sortTable.table[distance].liquidList[type]
}
```

### Liquid Types (LQ_* enum)

Based on the check `type < LQ_LAST` where LQ_LAST = 4:
```c
enum LiquidType {
    LQ_TYPE_0 = 0,  // Likely Water
    LQ_TYPE_1 = 1,  // Likely Ocean
    LQ_TYPE_2 = 2,  // Likely Magma/Lava
    LQ_TYPE_3 = 3,  // Likely Slime
    LQ_LAST = 4     // Count
};
```

### String References

Found strings:
- `"liquid"` @ 0x0089e4ec
- `"liquidTexBaseName[liquid]"` @ 0x0089f1d4
- `"liquid < LIQUID_COUNT"` @ 0x0089f1f0
- `"liquids[i]==0"` @ 0x008a11a0 (used in ~CMapChunk @ 0x006987e9)
- `".?AVCChunkLiquid@@"` @ 0x0089f708 (type info)

### MCLQ vs MLIQ

**Important**: Found reference to `'MLIQ'` @ 0x008a2930:
- `"pIffChunk->token == 'MLIQ'"`

This suggests:
- **MCLQ**: Terrain liquid chunks (in ADT/WDT)
- **MLIQ**: WMO liquid chunks (in WMO files)

### Structure Members

From the allocation and usage:

```c
struct CChunkLiquid {
    // Offset 0x00: Links for list management
    TSLink<CChunkLiquid> sceneLink;      // ~0x08 bytes
    TSLink<CChunkLiquid> lameAssLink;    // ~0x08 bytes
    
    // Offset ~0x10: Height range
    CFloatRange height;                   // 0x08 bytes (float min, float max)
    
    // Offset ~0x18: Flow vectors
    SWFlowv flowvs[2];                   // Unknown size per entry
    
    // Offset unknown: Parent chunk
    CMapChunk* chunk;                     // 0x04 bytes (at field_0x70 based on usage)
    
    // Vertex grid
    // Size suggests 9x9 vertex grid (81 vertices * some bytes per vertex)
    // OR 8x8 + flags = various arrangements
    // verts[] array
    
    // Total: 0x338 bytes
};
```

### Height Grid Dimensions

Based on structure size analysis:
- Total size: 0x338 = 824 bytes
- If 9×9 grid: 81 vertices
- If 8×8 grid: 64 vertices

With overhead (links, height range, flowvs, pointers) ~64-80 bytes, remaining space:
- ~744-760 bytes for vertex data
- 744 / 81 ≈ 9.2 bytes per vertex (if 9×9)
- 760 / 64 ≈ 11.9 bytes per vertex (if 8×8)

**Likely**: 9×9 grid with ~9 bytes per vertex (float height + byte flags)

### Per-Vertex Data Format

**Estimated structure**:
```c
struct LiquidVertex {
    float height;    // 4 bytes - water surface height
    uint8 flags;     // 1 byte - vertex flags (fishable, etc.)
    // Possibly more bytes (normal? flow?)
};
```

### Liquid Type Determination

From string `"liquidTexBaseName[liquid]"`, liquid types are indexed to select textures.

Likely determined by:
1. Type field in MCLQ chunk header
2. Indexed into texture array (4 types)

### Related Functions

Found liquid-related functions:
- `AllocChunkLiquid` @ 0x00691860 (allocation)
- `FreeChunkLiquid` @ 0x00691960 (deallocation)
- `AddChunkLiquid` @ 0x0066b120 (scene management)
- `RenderLiquid_0` @ 0x0069e4b0 (rendering)
- `PrepareRenderLiquid` @ 0x0066a590 (render prep)
- `QueryLiquidStatus` @ 0x00664e70+ (multiple, collision/gameplay queries)
- `QueryLiquidFishable` @ 0x00688060+ (fishing mechanics)
- `GetLiquidTexture` @ 0x006736b0 (texture selection)
- `UpdateLiquidTextures` @ 0x006738c0 (texture updates)

## Partial MCLQ Structure (Reconstructed)

Based on findings:

```c
struct MCLQ_Header {
    float minHeight;        // 0x00: Minimum liquid height
    float maxHeight;        // 0x04: Maximum liquid height
    uint32 type;            // 0x08: Liquid type (0-3)
    uint32 flags;           // 0x0C: Flags (?)
    // More fields...
};

struct MCLQ_Vertex {
    float height;           // Vertex height
    uint8 flags;            // Vertex flags
    // Possibly: flow vectors, normals
};

struct MCLQ_Chunk {
    MCLQ_Header header;
    MCLQ_Vertex vertices[9][9];  // Or [8][8], needs verification
};
```

## Next Steps

To complete the MCLQ documentation, we need to:
1. Find the actual MCLQ chunk parser (search for 0x4D434C51 constant)
2. Trace the exact header layout
3. Confirm vertex grid dimensions (8×8 or 9×9)
4. Document per-vertex data format completely
5. Identify all flag meanings

## Cross-References

Functions related to liquid data:
- `CMap::AllocChunkLiquid` @ 0x00691860
- `CWorldScene::AddChunkLiquid` @ 0x0066b120
- `~CMapChunk` @ 0x006987e9 (deallocates liquids[4])

## Confidence Level

**Medium** - We have determined:
- ✅ CChunkLiquid class size (0x338 bytes)
- ✅ Liquid types are 0-3 (4 total types)
- ✅ Height range stored as min/max floats
- ✅ Flow vectors present (2 entries)
- ✅ Chunks can have up to 4 liquid instances
- ✅ Distinction between MCLQ (terrain) and MLIQ (WMO)

Still investigating:
- ⏳ Exact MCLQ header layout
- ⏳ Vertex grid dimensions (8×8 vs 9×9)
- ⏳ Per-vertex data format details
- ⏳ Complete flag meanings
- ⏳ Flow vector format

## Differences from Later WoW Versions

Alpha 0.5.3 uses inline MCLQ chunks within the monolithic WDT file, whereas later versions use MH2O in separate ADT files. The liquid type system appears simpler (4 types vs. later expanded sets).
