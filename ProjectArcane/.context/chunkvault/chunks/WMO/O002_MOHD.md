# O002: MOHD

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MOHD (Map Object Header) chunk contains global information about the WMO, including counts of various elements, the bounding box of the entire model, and flags that define the behavior and rendering of the model. This chunk is only found in root WMO files and serves as the main header for the entire WMO structure.

## Structure
```csharp
struct MOHD
{
    /*0x00*/ uint32_t nTextures;       // Number of textures in MOTX
    /*0x04*/ uint32_t nGroups;         // Number of groups (MOGP chunks in the group files)
    /*0x08*/ uint32_t nPortals;        // Number of portals in the WMO
    /*0x0C*/ uint32_t nLights;         // Number of lights in the WMO
    /*0x10*/ uint32_t nDoodadNames;    // Number of filenames in MODN
    /*0x14*/ uint32_t nDoodadDefs;     // Number of doodad definitions in MODD
    /*0x18*/ uint32_t nDoodadSets;     // Number of doodad sets in MODS
    /*0x1C*/ uint32_t ambientColor;    // Base ambient color for the model (used as ambientColor unless flags bit 1 is set)
    /*0x20*/ uint32_t wmoId;           // WMO ID in the WMOAreaTable.dbc
    /*0x24*/ CAaBox boundingBox;       // Bounding box for the entire model (24 bytes: min and max XYZ)
    /*0x3C*/ uint16_t flags;           // Global WMO flags
    /*0x3E*/ uint16_t numLod;          // Number of LOD levels (added in later versions)
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| nTextures | uint32_t | Number of textures referenced in the MOTX chunk |
| nGroups | uint32_t | Number of groups/group files this WMO is split into (limit: 512) |
| nPortals | uint32_t | Number of portals in the MOPT chunk |
| nLights | uint32_t | Number of lights in the MOLT chunk |
| nDoodadNames | uint32_t | Number of M2 model filenames in the MODN chunk |
| nDoodadDefs | uint32_t | Number of doodad instances in the MODD chunk |
| nDoodadSets | uint32_t | Number of doodad sets in the MODS chunk |
| ambientColor | uint32_t | BGRA color value for the global ambient lighting |
| wmoId | uint32_t | Foreign key to the WMOAreaTable.dbc table |
| boundingBox | CAaBox | Bounding box for the entire WMO (min and max points) |
| flags | uint16_t | Global WMO flags (see below) |
| numLod | uint16_t | Number of LOD levels (added in later versions) |

## WMO Flags
| Flag Value | Name | Description |
|------------|------|-------------|
| 0x0001 | DO_NOT_ATTENUATE_VERTICES_BASED_ON_DISTANCE_TO_PORTAL | Disables vertex color attenuation |
| 0x0002 | USE_UNIFIED_RENDER_PATH | Uses a unified rendering approach where ambient color is added at runtime (WotLK+) |
| 0x0004 | USE_LIQUID_TYPE_DBC_ID | Uses proper LiquidType.dbc IDs instead of internal types |
| 0x0008 | DO_NOT_FIX_VERTEX_COLOR_ALPHA | Alters the behavior of CMapObjGroup::FixColorVertexAlpha |
| 0x0010 | LOD | WMO has LOD levels (Warlords of Draenor+) |
| 0x0020 | DEFAULT_MAX_LOD | If LOD flag is set, this means use numLod entries for LOD levels |

## Dependencies
- MOTX: The nTextures field should match the number of texture paths in MOTX
- MODN: The nDoodadNames field should match the number of model paths in MODN
- MODS: The nDoodadSets field should match the number of doodad sets in MODS
- MODD: The nDoodadDefs field should match the number of doodad instances in MODD
- MOPT/MOPV: The nPortals field should match the number of portals defined
- MOLT: The nLights field should match the number of lights defined

## Implementation Notes
- This chunk appears only in the root WMO file
- The ambientColor is a BGRA value (Blue in the lowest byte, Alpha in the highest)
- The boundingBox defines the overall extents of the model for culling and LOD selection
- When flag 0x0002 is set, the game uses a unified rendering approach
- When flag 0x0004 is set, liquids use DBC-defined liquid types
- The numLod field was added in Warlords of Draenor and may not be present in earlier versions
- There is a hard limit of 512 groups per WMO

## Implementation Example
```csharp
public class MOHD : IChunk
{
    public uint TextureCount { get; set; }
    public uint GroupCount { get; set; }
    public uint PortalCount { get; set; }
    public uint LightCount { get; set; }
    public uint DoodadNameCount { get; set; }
    public uint DoodadDefinitionCount { get; set; }
    public uint DoodadSetCount { get; set; }
    public Color AmbientColor { get; set; }
    public uint WmoId { get; set; }
    public BoundingBox BoundingBox { get; set; }
    public WmoFlags Flags { get; set; }
    public ushort LodLevels { get; set; }
    
    public void Parse(BinaryReader reader, long size)
    {
        TextureCount = reader.ReadUInt32();
        GroupCount = reader.ReadUInt32();
        PortalCount = reader.ReadUInt32();
        LightCount = reader.ReadUInt32();
        DoodadNameCount = reader.ReadUInt32();
        DoodadDefinitionCount = reader.ReadUInt32();
        DoodadSetCount = reader.ReadUInt32();
        
        uint ambientColorRaw = reader.ReadUInt32();
        AmbientColor = new Color(
            (byte)((ambientColorRaw >> 16) & 0xFF),      // R
            (byte)((ambientColorRaw >> 8) & 0xFF),       // G
            (byte)(ambientColorRaw & 0xFF),              // B
            (byte)((ambientColorRaw >> 24) & 0xFF)       // A
        );
        
        WmoId = reader.ReadUInt32();
        
        BoundingBox = new BoundingBox();
        BoundingBox.Min = new Vector3(
            reader.ReadSingle(),    // X min
            reader.ReadSingle(),    // Y min
            reader.ReadSingle()     // Z min
        );
        BoundingBox.Max = new Vector3(
            reader.ReadSingle(),    // X max
            reader.ReadSingle(),    // Y max
            reader.ReadSingle()     // Z max
        );
        
        Flags = (WmoFlags)reader.ReadUInt16();
        
        // Check if we have enough data for the LOD field (added in later versions)
        if (size >= 0x40)
        {
            LodLevels = reader.ReadUInt16();
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        writer.Write(TextureCount);
        writer.Write(GroupCount);
        writer.Write(PortalCount);
        writer.Write(LightCount);
        writer.Write(DoodadNameCount);
        writer.Write(DoodadDefinitionCount);
        writer.Write(DoodadSetCount);
        
        uint ambientColorRaw = 
            (uint)AmbientColor.B | 
            ((uint)AmbientColor.G << 8) | 
            ((uint)AmbientColor.R << 16) | 
            ((uint)AmbientColor.A << 24);
        writer.Write(ambientColorRaw);
        
        writer.Write(WmoId);
        
        writer.Write(BoundingBox.Min.X);
        writer.Write(BoundingBox.Min.Y);
        writer.Write(BoundingBox.Min.Z);
        writer.Write(BoundingBox.Max.X);
        writer.Write(BoundingBox.Max.Y);
        writer.Write(BoundingBox.Max.Z);
        
        writer.Write((ushort)Flags);
        writer.Write(LodLevels);
    }
}

[Flags]
public enum WmoFlags : ushort
{
    DoNotAttenuateVerticesBasedOnDistanceToPortal = 0x0001,
    UseUnifiedRenderPath = 0x0002,
    UseLiquidTypeDBC = 0x0004,
    DoNotFixVertexColorAlpha = 0x0008,
    HasLod = 0x0010,
    DefaultMaxLod = 0x0020
}
```

## CAaBox Structure
The CAaBox structure represents an axis-aligned bounding box used for culling and visibility testing:

```csharp
struct CAaBox
{
    /*0x00*/ float min_x;
    /*0x04*/ float min_y;
    /*0x08*/ float min_z;
    /*0x0C*/ float max_x;
    /*0x10*/ float max_y;
    /*0x14*/ float max_z;
};
```

## Validation Requirements
- The nGroups field must be > 0 and <= 512
- The boundingBox max values should be greater than their corresponding min values
- The counts (nTextures, nPortals, etc.) should match the actual number of entries in their respective chunks

## Usage Context
The MOHD chunk serves as the main header for the entire WMO structure. It provides essential information for rendering and managing the model, including:

1. Counts of key elements like groups, portals, and doodads
2. Overall model boundaries for culling
3. Global ambient color for lighting calculations
4. Flags controlling rendering behavior
5. WMO ID for area identification

This information is used throughout the rendering and interaction systems to properly handle the WMO in the game world. 