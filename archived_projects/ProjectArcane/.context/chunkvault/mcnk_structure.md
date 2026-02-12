# MCNK Chunk Structure

## Overview

The MCNK (Map Chunk) is the most complex chunk in the ADT file format. Each ADT file contains 256 MCNK chunks (in a 16x16 grid), and each MCNK represents a small section of terrain. What makes MCNK unique is that it contains its own subchunks following the same chunk format.

## MCNK Header Structure

```
struct SMChunk {
    /*0x00*/ uint32_t flags;
    /*0x04*/ uint32_t IndexX;
    /*0x08*/ uint32_t IndexY;
    /*0x0C*/ float heightmap_layers;    // number of height points - 8
    /*0x10*/ float heightmap_pos_y;
    /*0x14*/ float heightmap_pos_z;
    /*0x18*/ uint32_t skip1;
    /*0x1C*/ uint32_t skip2;
    /*0x20*/ uint32_t layercount;       // number of used layers, maximum is 4
    /*0x24*/ uint32_t doodadRefs;       // number of doodad references
    /*0x28*/ uint32_t mcvt;             // offset to MCVT sub-chunk (height map)
    /*0x2C*/ uint32_t mcnr;             // offset to MCNR sub-chunk (normal vectors)
    /*0x30*/ uint32_t mcly;             // offset to MCLY sub-chunk (texture layers)
    /*0x34*/ uint32_t mclq;             // offset to MCLQ sub-chunk (liquid data), is 0 if not present
    /*0x38*/ uint32_t mcrf;             // offset to MCRF sub-chunk (doodad references)
    /*0x3C*/ uint8_t  holes[8];         // up to version 9: 4, version 10+: 8. Terrain holes as bit vector (see below)
    /*0x44*/ uint16_t padding;
    /*0x46*/ uint16_t area_id;          // this looks like a map area id (zone)
    /*0x48*/ uint16_t map_layer_id;     // overlay layer (Cata+)
    /*0x4A*/ uint16_t predTex;          // number of textures referenced in LQ files
    /*0x4C*/ uint16_t num_effects_doodad;
    /*0x4E*/ uint16_t padding2;
    /*0x50*/ uint32_t mcse;             // offset to MCSE sub-chunk (sound emitters)
    /*0x54*/ uint32_t num_sound_emitters;
    /*0x58*/ uint32_t mcbb;             // offset to MCBB (Cata+) <Cata: skip padding during parse depending on version.
    /*0x5C*/ uint32_t mccv;             // offset to MCCV sub-chunk (vertex shading info)
    /*0x60*/ uint32_t mclv;             // offset to MCLV sub-chunk (Cata+)
    /*0x64*/ uint32_t unused_one;       // appears to always be 1
    /*0x68*/ float radius;
    /*0x6C*/ char padding3[0x10];
};
```

## MCNK Subchunks

MCNK contains multiple subchunks, each with their own format:

### MCVT (Height Map)
Contains 145 height values (a 9x9 grid + 8x8 midpoints) defining the terrain shape.

### MCNR (Normal Vectors)
Contains 145 normal vectors for the height map vertices, used for lighting.

### MCLY (Texture Layers)
Contains information about the texture layers on this chunk. Each entry references a texture in the MTEX/MDID chunks.

### MCCV (Vertex Colors)
Optional vertex coloring information (added in WotLK).

### MCLV (Light Values)
Optional vertex lighting information (added in Cata).

### MCLQ/MH2O (Liquid Data)
Water/liquid information for this chunk.

### MCRF (Doodad References)
References to doodads (MMID/MMDX) that appear in this chunk.

### MCSE (Sound Emitters)
Sound emitter definitions for this chunk.

### MCBB (Bounding Box)
Bounding box for the chunk (added in MoP).

### MCAL (Alpha Maps)
Alpha maps for blending between different texture layers.

### MCSH (Shadows)
Shadow map information.

## Relationships Within MCNK

```
MCNK Header
   |
   +---> MCVT (Heights) 
   |      |
   |      +---> Defines terrain shape
   |
   +---> MCNR (Normals)
   |      |
   |      +---> Used for lighting the terrain
   |
   +---> MCLY (Layers) -------> References textures in MTEX/MDID
   |      |
   |      +---> MCAL (Alpha map)
   |             |
   |             +---> Defines how texture layers blend
   |
   +---> MCRF (Doodad References) -------> References entries in MMID/MMDX
   |
   +---> MCLQ/MH2O (Liquid) -------> Defines water/liquid areas
```

## Implementation Challenges

1. **Complex Indexing**: Many subchunks reference each other and external chunks
2. **Version Differences**: Some subchunks were added in later expansions
3. **Variable Formats**: Some subchunks have multiple possible formats (e.g., MCAL)
4. **Holes and Flags**: Terrain holes and flags must be properly interpreted
5. **File Splitting**: In Cataclysm+, MCNK chunks are distributed across files

## Parsing Strategy

1. Parse the MCNK header
2. Use the offsets in the header to locate and parse each subchunk
3. Process version-specific features based on the MVER chunk
4. Validate all references to external chunks (MTEX/MMID/etc.)
5. Build the terrain data structure combining all subchunks

## Typical Implementation Approach

```csharp
public class MCNK
{
    public MCNKHeader Header { get; set; }
    public MCVT HeightMap { get; set; }
    public MCNR Normals { get; set; }
    public MCLY[] TextureLayers { get; set; }
    public MCAL AlphaMap { get; set; }
    public MCRF DoodadReferences { get; set; }
    public MCLQ Liquid { get; set; }
    public MCSE SoundEmitters { get; set; }
    // Additional properties for newer subchunks
}
```

This implementation allows for modular parsing of each subchunk while maintaining the relationships between them. 