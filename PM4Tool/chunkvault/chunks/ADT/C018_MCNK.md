# C018: MCNK

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
Map Chunk - contains terrain information for a 33.33 × 33.33 yard section of the map. Each ADT file contains 256 MCNK chunks in a 16×16 grid.

## Structure
```csharp
struct SMChunk 
{
    /*0x00*/ uint32_t flags;
    /*0x04*/ uint32_t IndexX;
    /*0x08*/ uint32_t IndexY;
    /*0x0C*/ float heightmap_layers;    // number of height points - 8
    /*0x10*/ float heightmap_pos_y;
    /*0x14*/ float heightmap_pos_z;
    /*0x18*/ uint32_t skip1;            // unused by client
    /*0x1C*/ uint32_t skip2;            // unused by client
    /*0x20*/ uint32_t layercount;       // number of used texture layers, maximum is 4
    /*0x24*/ uint32_t doodadRefs;       // number of doodad references
    /*0x28*/ uint32_t mcvt;             // offset to MCVT sub-chunk (height map)
    /*0x2C*/ uint32_t mcnr;             // offset to MCNR sub-chunk (normal vectors)
    /*0x30*/ uint32_t mcly;             // offset to MCLY sub-chunk (texture layers)
    /*0x34*/ uint32_t mclq;             // offset to MCLQ sub-chunk (liquid data), is 0 if not present
    /*0x38*/ uint32_t mcrf;             // offset to MCRF sub-chunk (doodad references)
    /*0x3C*/ uint8_t  holes[8];         // up to version 9: 4, version 10+: 8. Terrain holes as bit vector
    /*0x44*/ uint16_t padding;
    /*0x46*/ uint16_t area_id;          // map area ID (zone)
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
    /*0x68*/ float radius;              // bounding sphere radius
    /*0x6C*/ char padding3[0x10];
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| flags | uint32 | Various flags for the map chunk |
| IndexX | uint32 | X index in the map chunk grid (0-15) |
| IndexY | uint32 | Y index in the map chunk grid (0-15) |
| heightmap_layers | float | Number of height points - 8 |
| heightmap_pos_y | float | Y position of the heightmap |
| heightmap_pos_z | float | Z position of the heightmap |
| skip1 | uint32 | Unused by client |
| skip2 | uint32 | Unused by client |
| layercount | uint32 | Number of texture layers used (1-4) |
| doodadRefs | uint32 | Number of doodad references |
| mcvt | uint32 | Offset to MCVT subchunk (heights) |
| mcnr | uint32 | Offset to MCNR subchunk (normals) |
| mcly | uint32 | Offset to MCLY subchunk (texture layers) |
| mclq | uint32 | Offset to MCLQ subchunk (liquid), 0 if not present |
| mcrf | uint32 | Offset to MCRF subchunk (doodad refs) |
| holes | uint8[8] | Terrain holes bitfield |
| padding | uint16 | Padding |
| area_id | uint16 | Map area ID (zone) |
| map_layer_id | uint16 | Overlay layer (Cata+) |
| predTex | uint16 | Number of textures referenced in LQ files |
| num_effects_doodad | uint16 | Number of effect doodads |
| padding2 | uint16 | Padding |
| mcse | uint32 | Offset to MCSE subchunk (sound emitters) |
| num_sound_emitters | uint32 | Number of sound emitters |
| mcbb | uint32 | Offset to MCBB subchunk (Cata+) |
| mccv | uint32 | Offset to MCCV subchunk (vertex colors) |
| mclv | uint32 | Offset to MCLV subchunk (Cata+) |
| unused_one | uint32 | Appears to always be 1 |
| radius | float | Bounding sphere radius |
| padding3 | char[16] | Padding |

## Flag Values
| Value | Name | Description |
|-------|------|-------------|
| 0x1 | HasMCSH | Contains shadow map |
| 0x2 | Impassable | Impassable terrain |
| 0x4 | LiquidRiver | Contains a river |
| 0x8 | LiquidOcean | Contains ocean water |
| 0x10 | LiquidMagma | Contains magma/lava |
| 0x20 | LiquidSlime | Contains slime (ooze) |
| 0x40 | HasMCCV | Has MCCV chunk with vertex colors |
| 0x8000 | DoNotFixAlphaMap | Do not fix alpha map for common locations |
| 0x10000 | HighResHoles | High resolution holes |

## Dependencies
- MCIN (C003) - Contains offsets to MCNK chunks (<Cata)
- MTEX (C004) - Textures referenced by MCLY subchunk
- MDID (C013) - Texture file IDs referenced by MCLY subchunk (8.1.0+)
- MMID (C006) - Model indices referenced by MCRF subchunk

## Subchunks
MCNK contains multiple subchunks, each following the standard chunk format:
- MCVT: Height map data (145 height values)
- MCNR: Normal vectors for terrain
- MCLY: Texture layer information
- MCRF: Doodad references
- MCSH: Shadow map (optional)
- MCAL: Alpha maps for texture blending
- MCLQ: Legacy liquid data
- MCSE: Sound emitters
- MCCV: Vertex colors (WotLK+)
- MCLV: Light values (Cata+)
- MCBB: Bounding box (MoP+)

## Implementation Notes
- The MCNK chunk is the most complex in the ADT format
- Offsets in the header are relative to the start of the MCNK data
- Each MCNK represents a 33.33 × 33.33 yard square of terrain
- Each MCNK can have up to 4 texture layers
- Terrain holes are represented by bit flags
- Version-specific subchunks should be checked based on MVER
- In Cataclysm+, different subchunks may be in different files

## Implementation Example
```csharp
[Flags]
public enum MCNKFlags : uint
{
    None = 0,
    HasMCSH = 0x1,
    Impassable = 0x2,
    LiquidRiver = 0x4,
    LiquidOcean = 0x8,
    LiquidMagma = 0x10,
    LiquidSlime = 0x20,
    HasMCCV = 0x40,
    DoNotFixAlphaMap = 0x8000,
    HighResHoles = 0x10000
}

public class MCNKHeader
{
    public MCNKFlags Flags { get; set; }
    public uint IndexX { get; set; }
    public uint IndexY { get; set; }
    public float HeightmapLayers { get; set; }
    public float HeightmapPosY { get; set; }
    public float HeightmapPosZ { get; set; }
    public uint Skip1 { get; set; }
    public uint Skip2 { get; set; }
    public uint LayerCount { get; set; }
    public uint DoodadRefs { get; set; }
    public uint MCVTOffset { get; set; }
    public uint MCNROffset { get; set; }
    public uint MCLYOffset { get; set; }
    public uint MCLQOffset { get; set; }
    public uint MCRFOffset { get; set; }
    public byte[] Holes { get; set; }  // 8 bytes
    public ushort Padding { get; set; }
    public ushort AreaId { get; set; }
    public ushort MapLayerId { get; set; }
    public ushort PredTex { get; set; }
    public ushort NumEffectsDoodad { get; set; }
    public ushort Padding2 { get; set; }
    public uint MCSEOffset { get; set; }
    public uint NumSoundEmitters { get; set; }
    public uint MCBBOffset { get; set; }
    public uint MCCVOffset { get; set; }
    public uint MCLVOffset { get; set; }
    public uint UnusedOne { get; set; }
    public float Radius { get; set; }
}

public class MCNK
{
    public MCNKHeader Header { get; set; }
    public Dictionary<string, IChunk> Subchunks { get; set; } = new Dictionary<string, IChunk>();
    
    // Helper methods to access specific subchunks
    public MCVT GetHeightMap() => Subchunks.ContainsKey("MCVT") ? (MCVT)Subchunks["MCVT"] : null;
    public MCNR GetNormals() => Subchunks.ContainsKey("MCNR") ? (MCNR)Subchunks["MCNR"] : null;
    // etc.
}
```

## Terrain Holes
The MCNK chunk's "holes" field defines terrain holes. Each bit represents a hole in a specific sub-region of the chunk.

- In versions before 10, holes uses 4 bytes (32 bits)
- In version 10+, holes uses 8 bytes (64 bits) for higher resolution holes

For standard resolution holes (flags & 0x10000 == 0), each bit represents a 4x4 sub-region.
For high resolution holes (flags & 0x10000 != 0), each bit represents a 2x2 sub-region.

## Usage Context
MCNK chunks form the foundation of the terrain in World of Warcraft. Each MCNK represents a square section of the map and contains all the data needed to render that section: height values, normals, textures, etc. The MCNK chunks are arranged in a 16x16 grid to form a complete map tile. MCNK references textures from MTEX/MDID and models from MMID/MMDX to create a complete visual representation of the world. 