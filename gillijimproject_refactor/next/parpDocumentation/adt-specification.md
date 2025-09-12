# ADT File Format Specification

ADT (Area Definition Table) files represent WoW map tiles (533.33x533.33 units), containing terrain (MCNK chunks x64 for heights/verts/normals/liquids), textures (MTEX), models (MMDX), WMO placements (MODF), M2 doodads (MDDF), wiredata (MWMO). Versions via MVER (18/22/23); MHDR header with flags/offsets/counts. Special files (tex0/obj0: small, non-monolithic, e.g., only MTEX or MDDF).

Spec from AdtParser.cs (1955 lines): Manual parsing handles standard/special/reversed chunks (REVM for MVER); extracts filename coords (_XX_YY.adt → tx/ty), parses MTEX (null-terminated textures), MMDX (models), MCNK (terrain: heights/shadows/alpha), MODF (WMO 64 bytes: name_id/unique_id/pos_xyz/rot_xyz/bbox_min/max/flags with FileDataID bit), MDDF (M2 36 bytes: name_id/unique_id/pos_xyz/rot_xyz/scale/flags with FileDataID). Why: Tile-based world with layered content (terrain + objects); how: Chunk scan (readChunkId + size, skip unrecognized), null-strings for names, placement arrays. AdtFlatPlateBuilder generates flat 129x129 vert plates (128x128 squares, fan tris) for prefab base.

## Overall Structure
Standard ADT: MVER, MHDR (flags/chunk offsets x12 for MCNK/MTEX/etc.), MCNK x64 (terrain), MTEX (textures), MMDX (models), MODF (WMO), MDDF (M2), MWMO (wiredata), ... (optional: lights, fog). Size ~1MB/tile.

Special: Small files (e.g., tex0: only MTEX/MTEX), parsed with simplified chunk scan.

### Chunk Header
```c
struct ChunkHeader {
    char signature[4];  // e.g., "MHDR" (or reversed "REVM" for legacy)
    uint32_t size;      // Payload
};
```

## Key Chunks

### MVER: Version (4 bytes)
**Purpose**: ADT version (18/22/23).

**Plain English**: Identifies format variant (e.g., 18=classic, 23=modern with liquids). Code: br.ReadInt32(); logs for debugging.

**C Struct**:
```c
struct MVER {
    int32_t version;  // 18, 22, or 23
};
```

### MHDR: Header (104 bytes)
**Purpose**: Tile metadata: flags (bit0=has objects, bit1=has doodads, etc.), offsets/counts to chunks (e.g., MCNK offset/count=64).

**Plain English**: Defines tile content (e.g., nMCNK=64 always); offsets for seeking. Code: br.ReadUInt32() for flags, then 12 uint offsets/counts. Why: Efficient random access to chunks.

**C Struct**:
```c
struct MHDR {
    uint32_t flags;         // Content flags (bit0=obj0, bit1=doodads, bit2=occlusion, etc.)
    uint32_t adt_id;        // Tile ID (tx*64 + ty)
    uint32_t unknown_0x08;  // TBD
    // 12 pairs: offset, count for chunks (MCNK, MTEX, MMDX, MODF, MDDF, MWMO, etc.)
    uint32_t mcnk_offset, mcnk_count;  // Terrain chunks (64)
    uint32_t mtex_offset, mtex_count;  // Textures
    uint32_t mmdx_offset, mmdx_count;  // Models
    uint32_t modf_offset, modf_count;  // WMO placements
    uint32_t mddf_offset, mddf_count;  // M2 placements
    uint32_t mwmo_offset, mwmo_count;  // Wiredata
    // ... (6 more pairs for optional chunks)
};  // 104 bytes
```

### MCNK: Terrain Chunk (variable/entry x64, ~1KB/chunk)
**Purpose**: Per-chunk terrain: heights (MCVT 145x145 shorts), normals (MCNR 129x129 shorts), liquids (MCLQ), shadows (MCNR?).

**Plain English**: 64 MCNK subchunks define tile terrain; each has header (flags/offset/counts), MCVT heights (145x145 for 128x128 quads), MCNR normals/shadows. Code: Skips in AdtParser (logs size), but AdtFlatPlateBuilder generates flat (z=0) plates from tile coords. Why: Heightmap for terrain; how: Parsed in manual loop for special files.

**C Struct** (Entry, ~1KB):
```c
struct MCNK {
    uint32_t flags;         // Chunk flags (water/terrain type)
    uint32_t unknown_0x04;  // TBD
    float bbox_min_xyz[3];  // Chunk bbox min
    float bbox_max_xyz[3];  // Chunk bbox max
    uint32_t sound_emitter_count;  // Sounds
    uint32_t liquid_level_count;   // Liquids
    uint32_t unknown_0x28;  // TBD
    uint32_t offset_mcv t;  // MCVT offset (145*145*2 bytes)
    uint32_t offset_mc nr;  // MCNR offset (129*129*2 bytes)
    uint32_t offset_mclq;   // MCLQ offset (liquid)
    uint32_t unknown_0x38;  // TBD
    // Subchunks follow (MCVT heights, MCNR normals, etc.)
    int16_t heights[145][145];  // Heightmap
    int16_t normals[129][129];  // Normals/shadows
    // MCLQ (liquid: type/heightmap)
};  // Variable ~1KB/chunk
```

### MODF: WMO Placement (64 bytes/placement, variable count)
**Purpose**: Array of WMO instances: name_id (index into MWMO or FileDataID), unique_id, pos_xyz, rot_xyz, bbox min/max, flags.

**Plain English**: Places WMO buildings in tile; name_id to model name/FileDataID, unique_id for uniqueness, pos/rot for transform, bbox for culling, flags for state (visible/destructible). Code: br.ReadUInt32() for name_id/unique_id, ReadSingle for pos/rot, skip bbox (6 floats), read flags. Why: Dynamic buildings; how: Array of fixed-size structs.

**C Struct** (Placement, 64 bytes):
```c
struct MODFPlacement {
    uint32_t name_id;       // WMO name index or FileDataID
    uint32_t unique_id;     // Unique instance ID
    float pos_x, pos_y, pos_z;  // Position
    float rot_x, rot_y, rot_z;  // Rotation (quaternion?)
    float bbox_min_x/y/z;   // Bbox min (skipped in code)
    float bbox_max_x/y/z;   // Bbox max (skipped)
    uint32_t flags;         // State flags (visible, etc.)
};  // 64 bytes/placement
```

Chunk: "MODF", size = count * 64.

### MDDF: M2 Placement (36 bytes/placement, variable count)
**Purpose**: Array of M2 doodad instances: name_id, unique_id, pos_xyz, rot_xyz, scale, flags.

**Plain English**: Places M2 models (trees, props); similar to MODF but with scale. Code: br.ReadUInt32() for name_id/unique_id, ReadSingle for pos/rot/scale, skip flags (1 uint). Why: Scattered details; how: Fixed structs.

**C Struct** (Placement, 36 bytes):
```c
struct MDDFPlacement {
    uint32_t name_id;       // M2 name index or FileDataID
    uint32_t unique_id;     // Unique instance ID
    float pos_x, pos_y, pos_z;  // Position
    float rot_x, rot_y, rot_z;  // Rotation
    float scale;            // Scale factor
    uint32_t flags;         // State flags
};  // 36 bytes/placement
```

Chunk: "MDDF", size = count * 36.

### MTEX: Textures (variable, null-terminated strings)
**Purpose**: List of texture names (blp files).

**Plain English**: Referenced by MCNK/MOPY. Code: ReadNullTerminatedString in loop until endPos.

**C Struct**:
```c
struct MTEX {
    char texture_names[variable][];  // Null-terminated .blp paths
};  // Size = sum(strlen+1)
```

### MMDX: Models (variable, null-terminated strings)
**Purpose**: List of M2 model names.

**Plain English**: Referenced by MDDF. Code: Similar to MTEX.

**C Struct**:
```c
struct MMDX {
    char model_names[variable][];  // Null-terminated .m2 paths
};
```

## Special Files
- **tex0/obj0**: Small ADTs (e.g., only MTEX or MDDF); parsed with simplified chunk scan (skip unrecognized, read null-strings for names).

## Data Arrangement and Usage
ADT: MHDR offsets → chunks (MCNK x64 for terrain grid, MODF/MDDF for objects). Special: Auxiliary (tex0= textures only).

- **ADTPreFabTool**: Loads ADT via WoWFormatLib (not code-found, inferred), exports GLB with minimap overlays (BLP decode to PNG, grid draw), prefab scan (hamming similarity on heights for chunk selection).
- **AdtParser**: Manual parse for analysis (AdtAnalysisResult with TextureNames/ModelNames/WmoPlacements; placements to WmoPlacementInfo with FileDataID). Why: Handles legacy/special/reversed; how: Chunk loop with try-catch for EOS.

Validate with AdtParser.ParseAsync (returns AdtAnalysisResult with errors if invalid).