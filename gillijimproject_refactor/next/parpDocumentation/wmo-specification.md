# WMO File Format Specification (V14 and V17)

WMO (World Model Object) files define 3D buildings/structures in WoW, with versions V14 (legacy) and V17 (modern). They are IFF-chunked binaries containing root header (MOHD), textures (MOTX), groups (MOGP + subchunks like MOVT verts, MOVI faces, MOPY flags), portals (MPOR), liquids (MLIQ), etc. V14 and V17 share layout but V17 has extended fields (e.g., larger headers).

Spec derived from V14WmoFile.cs/V17WmoFile.cs: Loads MOHD/MOTX/MOGP, parses MOVT (Vector3 verts), MOVI (ushort tri indices), MOPY (byte face flags). Why: Buildings as hierarchical groups (root + portals + groups); how: Chunk scanning, naive subchunk association post-MOGP. Exporters like WmoMeshExporter write OBJ from groups; PM4NextExporter matches PM4 buildings to WMO via WmoMatcher.

## Overall Structure
Root file with chunks:
- MVER (version: 14 or 17)
- MOHD (header: nGroups, nPortals, nLights, etc.)
- MOTX (null-terminated texture names)
- MODT (doodad set names)
- MODS (doodad sets)
- MODD (doodad definitions)
- MOGP (group headers, repeated nGroups times; each followed by subchunks: MOVI/MOVT/MOPY/MOVB/MOTE/MOTV/MOBA)
- PORT (portal definitions)
- MLIQ (liquid data)
- ... (other optional: lights, physics)

Size: 1MB-100MB. Validation: MOHD counts match chunk instances (e.g., nGroups = MOGP count).

### Chunk Header
```c
struct ChunkHeader {
    char signature[4];  // e.g., "MOHD"
    uint32_t size;      // Payload
};
```

## Key Chunks

### MOHD: Header (64 bytes V17)
**Purpose**: File metadata: group/portal/light counts, bounding box, flags.

**Plain English**: Defines WMO scope (nGroups for sub-buildings, nPortals for interiors). BBox for culling; flags for render/physics.

**C Struct (V17 - 3.3.5)**:
```c
struct MOHD_V17 {
    uint32_t nTextures;     // 0x00: Number of textures
    uint32_t nGroups;       // 0x04: Number of MOGP groups
    uint32_t nPortals;      // 0x08: Number of portals
    uint32_t nLights;       // 0x0C: Number of lights
    uint32_t nDoodadNames;  // 0x10: Number of doodad names
    uint32_t nDoodadDefs;   // 0x14: Number of doodad definitions
    uint32_t nDoodadSets;   // 0x18: Number of doodad sets
    uint32_t ambColor;      // 0x1C: Ambient color (CArgb)
    uint32_t wmoID;         // 0x20: WMOAreaTable ID
    float bbox[6];          // 0x24: Bounding box (min xyz, max xyz)
    uint16_t flags;         // 0x3C: Flags
    uint16_t numLod;        // 0x3E: LOD count (Legion+)
};  // 64 bytes
```

**C Struct (V14 - Alpha 0.5.3)**:
```c
struct MOHD_V14 {
    uint32_t nTextures;     // 0x00
    uint32_t nGroups;       // 0x04
    uint32_t nPortals;      // 0x08
    uint32_t nLights;       // 0x0C
    uint32_t nDoodadNames;  // 0x10
    uint32_t nDoodadDefs;   // 0x14
    uint32_t nDoodadSets;   // 0x18
    uint32_t ambColor;      // 0x1C
    uint32_t wmoID;         // 0x20
    uint8_t padding[0x1C];  // 0x24: No bounding box in v14
};
```
**Note**: V14 (Alpha) has no bounding box - it was computed at load time.

**Usage**: Header = FromSpan(mohd.Data); Groups.Count == nGroups.

### MOTX: Textures (variable, null-terminated strings)
**Purpose**: List of texture file names used by groups.

**Plain English**: Array of zero-terminated strings (e.g., "building_texture.blp"). Code: MOTXParser.Parse(data) reads until end. Why: Reference for MOPY texture indices.

**C Struct**:
```c
struct MOTX {
    char texture_names[variable][];  // Null-terminated strings concatenated
};  // Size = sum(strlen(name)+1)
```

**Usage**: TextureNames = MOTXParser.Parse(motx.Data); Groups[i].Materials reference indices.

### MOGP: Group Header (36 bytes/entry, nGroups times)
**Purpose**: Per-group metadata: name offset, flags, bbox, doodad/sound counts.

**Plain English**: Each group is a sub-mesh; MOGP defines its name (MODN offset), flags (visible/occluded), bbox. Subchunks follow: MOVT (verts), MOVI (faces), MOPY (flags/textures). Code: MOGPGroupHeader.FromSpan(data); associates next MOVT/MOVI/MOPY.

**C Struct**:
```c
struct MOGP {
    uint32_t name_offset;   // Index into MODN strings
    uint32_t description;   // Unknown (TBD)
    float bbox_min_x/y/z, bbox_max_x/y/z;  // Group bbox
    uint32_t flags;         // Visibility/occlusion
    uint32_t nDoodads;      // Doodads in group
    uint32_t nSounds;       // Sounds in group
    uint32_t nUnknown;      // TBD
};  // 36 bytes
```

**Usage**: Header = FromSpan(mogp.Data); then parse subchunks (MOVT: list Vector3, MOVI: list (ushort a,b,c), MOPY: list byte flags).

### Other Chunks (Brief)
- **MOVT**: Verts (Vector3 array, 12 bytes each).
- **MOVI**: Faces ((ushort a,b,c) tris, 6 bytes each).
- **MOPY**: Per-face flags (byte: texture index + render flags, 1 byte each).
- **PORT**: Portals (connects groups; float pos/rot/dir, uint flags).
- **MLIQ**: Liquids (water/lava; height maps, types).
- **MODF/MODD**: Doodads (placements/definitions; M2 models).

## Data Arrangement and Usage in Priority Tools
WMO: Root (MOHD) → Groups (MOGP + MOVT/MOVI/MOPY for mesh) → Portals/Liquids. V14/V17 identical layout; V17 for newer content.

- **PM4FacesTool/NextExporter**: WmoMatcher matches PM4 buildings (MSUR surfaces) to WMO groups via bbox/vert similarity; exports WMO as OBJ via WmoMeshExporter (from groups).
- **Pm4BatchObjExporter**: Loads WMO for unified scenes; extracts doodads (MODF) linked to PM4 MSLK.

For ADT/WDT/M2 integration, see respective specs. Validate with V14WmoFile/V17WmoFile.Load (counts match MOHD).