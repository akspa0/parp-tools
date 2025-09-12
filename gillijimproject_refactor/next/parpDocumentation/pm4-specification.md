# PM4 File Format Specification

This document provides a comprehensive specification of the PM4 file format, reverse-engineered from the parpToolbox codebase. PM4 files are binary assets used in World of Warcraft for phasing, meshing, and scene data, structured as IFF-style chunked files (similar to RIFF/WAV). The format consists of a series of chunks (e.g., MVER, MSHD, MSLK, MSUR, MSCN, MSVI, MSPV, MSPI, MPRL, MPRR, MDBH, MDOS, MDSF), each with a 4-byte signature, 4-byte size, and payload. Chunks are optional but follow a typical order for valid files.

The specification is presented in plain English with corresponding C struct definitions (little-endian, aligned). Field meanings are decoded from code analysis, comments (e.g., "DECODED: Object Type Flags"), and export logic (e.g., how PM4FacesTool groups MSUR surfaces via shared MSVI indices using DSU, or Pm4BatchObjExporter extracts objects by MSLK ParentIndex). Priority tools (PM4FacesTool, PM4NextExporter) reveal data arrangement: MSLK links define object hierarchy (type/group ID, path geometry via MSPI/MSPV), MSUR surfaces reference MSVI indices into MSVT verts, MSCN provides exterior boundaries, and cross-tile linkage via MSLK LinkId (0xFFFFYYXX for tile YX).

## Overall File Structure
A PM4 file starts with chunks in this observed order (not all required):
1. MVER (version, 4 bytes)
2. MSHD (header, 32 bytes)
3. MSLK (linkage entries, variable)
4. MSPI (path indices, variable)
5. MSPV (path vertices, variable)
6. MSVT (mesh vertices, variable; often absent, use MSPV)
7. MSVI (mesh indices, variable)
8. MSUR (surface definitions, variable)
9. MSCN (exterior vertices, variable)
10. MSRN (referenced normals, variable; optional)
11. MPRL (position data, variable)
12. MPRR (reference data, variable)
13. MDBH (destructible header, optional)
14. MDOS (destructible objects, optional)
15. MDSF (destructible structure, optional)

Total size: Sum of chunk sizes + headers. Files are ~100KB-10MB depending on complexity. Validation: Chunks like MSVI indices must bound to MSVT verts (via MSVI.ValidateIndices(vertexCount)).

### C Struct for Chunk Header (Common to All)
```c
// IFF-style chunk header (8 bytes)
struct ChunkHeader {
    char signature[4];  // e.g., "MSHD"
    uint32_t size;      // Payload size (excluding header)
};
```

## Chunk Specifications

### MVER: Version Chunk (4 bytes)
**Purpose**: Stores the PM4 file version (typically 1). Essential for format identification; exporters check for compatibility.

**Plain English**: A simple version tag ensuring the parser knows the expected structure. All analyzed files have version 1; higher values may indicate extensions (e.g., new fields in MSLK).

**C Struct**:
```c
struct MVER {
    uint32_t version;  // Typically 1
};
```

**Usage in Code**: `MVER.Version` in PM4File; ignored in exports but logged in diagnostics.

### MSHD: Header Chunk (32 bytes)
**Purpose**: PM4 file header with 8 unknown uint32 fields (0x00-0x1C). Likely metadata like bounds, flags, or offsets; all observed as constants or tile-specific.

**Plain English**: The header provides global file info, but fields remain TBD (e.g., Unknown_0x00 may be magic/tile ID). Code reads all 8 uints sequentially; size fixed at 32 bytes. Exporters ignore but validate size.

**C Struct**:
```c
struct MSHD {
    uint32_t unk00;  // TBD (possibly magic or flags)
    uint32_t unk04;  // TBD
    uint32_t unk08;  // TBD
    uint32_t unk0C;  // TBD
    uint32_t unk10;  // TBD
    uint32_t unk14;  // TBD
    uint32_t unk18;  // TBD
    uint32_t unk1C;  // TBD
};  // Total: 32 bytes
```

**Usage in Code**: `MSHDChunk.Load` reads 8 uints; `ToString` hex-dumps for debugging. PM4FacesTool skips; Pm4BatchObjExporter uses for validation.

### MSLK: Linkage Chunk (20 bytes/entry, variable count)
**Purpose**: Defines object linkages and metadata. Each entry classifies objects (type/subtype flags), groups them (GroupObjectId), references path geometry (MSPI first/count), encodes tile crossing (LinkId=0xFFFFYYXX for tile Y,X), and cross-references (RefIndex). Used for hierarchy and cross-tile navigation.

**Plain English**: MSLK is the "object catalog": entries represent nodes in a scene graph. TypeFlags (1-18 values) classify (e.g., walkable vs. prop), Subtype (0-7) variants. GroupObjectId organizes; MspiFirstIndex/MspiIndexCount points to path verts in MSPV (for non-geometry, -1). LinkId signals tile boundaries (0xFFFFYYXX = tile YX). RefIndex links to other structures; SystemFlag=0x8000 constant. Exports group by ParentIndex (inferred from code); PM4FacesTool uses for CK24 extraction ((key & 0xFFFFFF00) >> 8), Pm4BatchObjExporter filters HasGeometry=true entries by ParentIndex for object extraction.

**C Struct** (Entry, 20 bytes):
```c
struct MSLKEntry {
    uint8_t type_flags;     // Object type classification (1-18 values)
    uint8_t subtype;        // Object subtype variant (0-7 values)
    uint16_t reserved_padding;  // Always 0x0000
    uint32_t group_object_id;   // Organizational grouping ID
    int24_t mspi_first_index;   // Index into MSPI/MSPV (path geometry; -1 = non-geometry)
    uint8_t mspi_index_count;   // Count of contiguous MSPI entries
    uint32_t link_id;       // Tile crossing: 0xFFFFYYXX (YY=Y tile, XX=X tile)
    uint16_t ref_index;     // Cross-reference to other structures (high/low bytes)
    uint16_t system_flag;   // Constant 0x8000
};  // Total: 20 bytes/entry
```

**Chunk Header**: Signature "MSLK", size = count * 20.

**Usage in Code**: `MSLK.Entries` list; `ValidateIndices` not direct but used in MSPI. PM4FacesTool: CK24 from CompositeKey (inferred MSLK-derived); Pm4BatchObjExporter: Groups by ParentIndex (Unknown_0x04?), filters HasGeometry (type_flags?).

### MSUR: Surface Chunk (32 bytes/entry, variable count)
**Purpose**: Defines mesh surfaces: group key (type bucket, 0=M2 props), index count into MSVI, attribute mask (bit7=liquid?), normal XYZ (floats), height (float), MSVI first index, MDOS index (destructible), packed params (two uint16).

**Plain English**: MSUR catalogs polygonal surfaces: SurfaceGroupKey batches (0=non-walkable M2, non-zero=walkable). IndexCount + MsviFirstIndex reference contiguous MSVI indices into MSVT verts. Normal (Nx/Ny/Nz) and Height define plane for snapping (e.g., --snap-to-plane in PM4FacesTool scales Height). AttributeMask flags (e.g., liquid). MdosIndex links to destructibles; PackedParams (hi/lo words) TBD (flags/IDs). Exports triangulate via fan (EmitTri fan from first vert); PM4FacesTool DSU connects via shared verts in MSVI ranges; Pm4BatchObjExporter extracts per MSUR in objects.

**C Struct** (Entry, 32 bytes):
```c
struct MSUREntry {
    uint8_t group_key;      // Surface type bucket (0=M2 props, non-zero=walkable)
    uint8_t index_count;    // Number of indices in MSVI for this surface
    uint8_t attribute_mask; // Surface flags (bit7=liquid candidate)
    uint8_t padding;        // Alignment
    float nx;               // Surface normal X
    float ny;               // Surface normal Y
    float nz;               // Surface normal Z
    float height;           // Surface plane height (Y-world)
    uint32_t msvi_first;    // Starting index in MSVI
    uint32_t mdos_index;    // Index into MDOS (destructible states)
    uint32_t packed_params; // Two uint16: high/low words (TBD flags/IDs)
};  // Total: 32 bytes/entry
```

**Chunk Header**: "MSUR", size = count * 32.

**Usage in Code**: `MSURChunk.Entries` (alias Surfaces); `ValidateIndices(msviCount)` checks ranges. PM4FacesTool: Groups by GroupKey/CompositeKey (inferred from MSLK?), assembles tris from MsviFirst+IndexCount; Pm4BatchObjExporter: ExtractSurfaceGeometry reads StartIndex/IndexCount for fan tris.

### MSCN: Exterior Vertices Chunk (12 bytes/vertex, variable count)
**Purpose**: List of Vector3 (float XYZ) for object exterior boundaries (not normals, as previously thought).

**Plain English**: MSCN defines outline points for collision/pathing; each is a world-space Vector3. Used for sidecar exports (--mscn-sidecar in PM4FacesTool, vertices-only OBJ per tile). Code applies canonical transforms (ToCanonicalWorldCoordinates: Y, -X, Z for WoW orientation). Pm4BatchObjExporter includes in objects if present.

**C Struct** (Vertex, 12 bytes):
```c
struct MSCNVertex {
    float x;  // File X (transformed to -X world)
    float y;  // File Y (transformed to Y world)
    float z;  // File Z (preserved)
};  // Total: 12 bytes/vertex
```

**Chunk Header**: "MSCN", size = count * 12.

**Usage in Code**: `MSCNChunk.ExteriorVertices`; `ToCanonicalWorldCoordinates(v)` for export. PM4FacesTool: ExportMscnSidecar groups by tile, applies pre-transforms (basis/rotZ/flip).

### MSVI: Mesh Indices Chunk (4 bytes/index, variable count)
**Purpose**: Array of uint indices referencing MSVT vertices for mesh triangles.

**Plain English**: MSVI holds the polygon connectivity: each uint is a vertex index into MSVT. Surfaces (MSUR) reference contiguous ranges (MsviFirst + IndexCount). Code validates all < MSVT.Count; exports triangulate fans/N-gons. PM4FacesTool: EmitTriMapped uses for dedup/shared vert DSU; degenerates skipped.

**C Struct** (Index, 4 bytes):
```c
struct MSVIIndex {
    uint32_t index;  // Index into MSVT vertices (0-based)
};  // Total: 4 bytes/index
```

**Chunk Header**: "MSVI", size = count * 4.

**Usage in Code**: `MSVIChunk.Indices`; `ValidateIndices(vertexCount)` bounds check; `GetIndicesForSurface(first, count)` range extract. PM4FacesTool: Assembles from ranges in AssembleAndWrite.

### MSPV: Path Vertices Chunk (12 bytes/vertex, variable count)
**Purpose**: Array of C3Vector (float XYZ) for pathfinding vertices, referenced by MSPI.

**Plain English**: MSPV stores path points (e.g., navigation meshes); each is a float vector. Linked via MSLK MspiFirstIndex/MspiIndexCount for non-geometry objects (-1 if no path). Code reads as floats (not int); used in exports if MSLK IsGeometryNode=false.

**C Struct** (Vertex, 12 bytes):
```c
struct MSPVVertex {
    float x, y, z;  // Path vertex coordinates
};  // Total: 12 bytes/vertex
```

**Chunk Header**: "MSPV", size = count * 12.

**Usage in Code**: `MSPVChunk.Vertices` (C3Vector); no direct validation. Pm4BatchObjExporter: Includes if linked via MSLK.

### MSPI: Path Indices Chunk (4 bytes/index, variable count)
**Purpose**: uint indices into MSPV for path segments.

**Plain English**: MSPI defines path connectivity: each uint indexes an MSPV vertex. Referenced by MSLK for object paths. Code validates < MSPV.Count; used for non-mesh geometry.

**C Struct** (Index, 4 bytes):
```c
struct MSPIIndex {
    uint32_t index;  // Index into MSPV
};  // Total: 4 bytes/index
```

**Chunk Header**: "MSPI", size = count * 4.

**Usage in Code**: `MSPIChunk.Indices`; `ValidateIndices(vertexCount)`. Linked via MSLK MspiFirst/MspiIndexCount.

## Other Chunks (Brief)
- **MSVT**: Mesh vertices (similar to MSPV but for main geometry; often absent, inferred from code as float vectors).
- **MSRN**: Referenced normals (float vectors; optional).
- **MPRL/MPRR**: Position/reference data (TBD; uint arrays for offsets/IDs).
- **MDBH/MDOS/MDSF**: Destructible buildings (header/objects/structure; optional, linked via MSUR MdosIndex).

## Data Arrangement and Usage in Priority Tools
PM4 data forms a graph: MSLK entries (nodes) link MSUR surfaces (meshes via MSVI→MSVT) and paths (MSPI→MSPV); cross-tile via LinkId. MSCN bounds objects.

- **PM4FacesTool**: Loads Pm4Scene (global Vertices/Indices/Surfaces from chunks); groups MSUR by strategy (e.g., CompositeInstances: DSU unions surfaces sharing MSVI verts within CK24 from CompositeKey (MSLK-derived?)); assembles local verts/tris (TryMap dedups, EmitTri fan triangulates IndexCount); exports OBJ with transforms. Reveals: Surfaces contiguous in MSVI; CK24 = (CompositeKey & 0xFFFFFF00)>>8 for grouping; tile bucketing via DominantTileIdFor (MSVI ranges in TileIndexOffsetByTileId).

- **PM4NextExporter/Pm4BatchObjExporter**: UnifiedMapScene merges all PM4s (offsets for verts/indices/links); extracts objects by MSLK HasGeometry=true + ParentIndex (Unknown_0x04?) grouping; ExtractSurfaceGeometry fans MSUR IndexCount from MSVI; global merged OBJ. Reveals: ParentIndex hierarchies objects; cross-tile preserved via offsets; HasGeometry filters mesh nodes.

This spec enables custom parsers/exporters; validate with PM4FacesTool's coverage CSVs (e.g., index ranges match MSVI).