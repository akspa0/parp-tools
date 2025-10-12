# W005: MODF

## Type
WDT Chunk

## Source
WDT.md

## Description
The MODF (Map Object Definition) chunk contains placement information for global WMO (World Map Object) models. These are typically large structures that span multiple ADT tiles, such as major buildings, bridges, or other significant structures. This chunk works with the MWMO and MWID chunks to define global WMO placements in the world.

## Structure
```csharp
struct SMODoodadDef // sizeof(0x40)
{
    /*0x00*/ uint32_t name_id;        // Index into MWID list for the filename
    /*0x04*/ uint32_t unique_id;      // Unique identifier for this instance
    /*0x08*/ Vector3 position;        // Position (X, Y, Z) coordinates
    /*0x14*/ Vector3 rotation;        // Rotation (X, Y, Z) in radians
    /*0x20*/ Vector3 scale;           // Scale factor (usually 1.0f for all axes)
    /*0x2C*/ Color ambient_color;     // RGBA ambient color
    /*0x30*/ uint32_t flags;          // Various flags for this WMO
    /*0x34*/ uint16_t doodad_set;     // Doodad set index for this WMO
    /*0x36*/ uint16_t name_set;       // Name set index for this WMO
    /*0x38*/ uint16_t unknown1;       // Padding or unused
    /*0x3A*/ uint16_t unknown2;       // Padding or unused
    /*0x3C*/ uint32_t unknown3;       // Possibly file data ID in later versions
};

struct MODF
{
    /*0x00*/ SMODoodadDef wmo_instances[];  // Array of WMO instance definitions
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| wmo_instances | SMODoodadDef[] | Array of WMO instance placement data |

### SMODoodadDef Properties
| Name | Type | Description |
|------|------|-------------|
| name_id | uint32_t | Index into the MWID list for the filename |
| unique_id | uint32_t | Unique identifier for this instance |
| position | Vector3 | Position coordinates (X, Y, Z) in world space |
| rotation | Vector3 | Rotation angles (X, Y, Z) in radians |
| scale | Vector3 | Scale factors for each axis (typically 1.0) |
| ambient_color | Color | RGBA ambient color for this instance |
| flags | uint32_t | Various flags affecting appearance and behavior |
| doodad_set | uint16_t | Doodad set index for this WMO |
| name_set | uint16_t | Name set index for this WMO |
| unknown1 | uint16_t | Unknown/padding |
| unknown2 | uint16_t | Unknown/padding |
| unknown3 | uint32_t | Possible FileDataID in later versions |

## Flag Values
| Value | Name | Description |
|-------|------|-------------|
| 0x01 | WMO_DESTRUCTIBLE | WMO can be destroyed |
| 0x02 | WMO_UNKNOWN1 | Unknown usage |
| 0x04 | WMO_UNKNOWN2 | Unknown usage |
| 0x08 | WMO_HAS_LIGHT | WMO has internal lighting |
| 0x10 | WMO_UNKNOWN3 | Unknown usage |
| 0x20 | WMO_UNKNOWN4 | Unknown usage |
| 0x40 | WMO_UNKNOWN5 | Unknown usage |
| 0x80 | WMO_UNKNOWN6 | Unknown usage |
| 0x100 | WMO_UNKNOWN7 | Unknown usage |
| 0x200 | WMO_UNKNOWN8 | Unknown usage |
| 0x400 | WMO_UNKNOWN9 | Unknown usage |
| 0x800 | WMO_HAS_DOODADS | WMO has doodads |

## Dependencies
- MWMO (W004) - Contains the WMO filenames
- MWID (W006) - Contains offsets into the MWMO chunk for filenames

## Implementation Notes
- The MODF chunk is only present if the map contains global WMO objects
- Each entry in the `wmo_instances` array represents one WMO placement in the world
- The `name_id` field references an index in the MWID array, which in turn provides an offset into the MWMO string table
- Rotation values are in radians (not degrees)
- The chunk size should be a multiple of 0x40 (64 bytes, the size of SMODoodadDef)
- The number of entries can be calculated by dividing the chunk size by the structure size
- The `unique_id` field should be globally unique for each WMO instance
- In versions 22+, the filename may be referenced by FileDataID instead of through MWID/MWMO

## Implementation Example
```csharp
public class SMODoodadDef
{
    public uint NameId { get; set; }           // Index into MWID
    public uint UniqueId { get; set; }          // Unique identifier
    public Vector3 Position { get; set; }      // X, Y, Z position
    public Vector3 Rotation { get; set; }      // X, Y, Z rotation in radians
    public Vector3 Scale { get; set; }         // Scale factor (usually 1.0)
    public Color AmbientColor { get; set; }    // RGBA ambient color
    public uint Flags { get; set; }            // Various flags
    public ushort DoodadSet { get; set; }      // Doodad set index
    public ushort NameSet { get; set; }        // Name set index
    public ushort Unknown1 { get; set; }       // Padding/unused
    public ushort Unknown2 { get; set; }       // Padding/unused
    public uint Unknown3 { get; set; }         // Possible FileDataID
    
    // Helper methods for flags
    public bool IsDestructible => (Flags & 0x01) != 0;
    public bool HasLight => (Flags & 0x08) != 0;
    public bool HasDoodads => (Flags & 0x800) != 0;
    
    // Helper method to get the WMO filename using MWID and MWMO
    public string GetFilename(MWID mwid, MWMO mwmo)
    {
        if (NameId >= mwid.Offsets.Count)
            throw new IndexOutOfRangeException($"NameId {NameId} is out of range");
            
        uint offset = mwid.GetOffset((int)NameId);
        return mwmo.GetFilenameByOffset((int)offset);
    }
}

public class MODF : IChunk
{
    public List<SMODoodadDef> WmoInstances { get; private set; } = new List<SMODoodadDef>();
    
    public void Parse(BinaryReader reader, long size)
    {
        const int ENTRY_SIZE = 0x40; // 64 bytes per entry
        int count = (int)(size / ENTRY_SIZE);
        WmoInstances.Clear();
        
        for (int i = 0; i < count; i++)
        {
            var instance = new SMODoodadDef
            {
                NameId = reader.ReadUInt32(),
                UniqueId = reader.ReadUInt32(),
                Position = new Vector3
                {
                    X = reader.ReadSingle(),
                    Y = reader.ReadSingle(),
                    Z = reader.ReadSingle()
                },
                Rotation = new Vector3
                {
                    X = reader.ReadSingle(),
                    Y = reader.ReadSingle(),
                    Z = reader.ReadSingle()
                },
                Scale = new Vector3
                {
                    X = reader.ReadSingle(),
                    Y = reader.ReadSingle(),
                    Z = reader.ReadSingle()
                },
                AmbientColor = new Color
                {
                    R = reader.ReadByte(),
                    G = reader.ReadByte(),
                    B = reader.ReadByte(),
                    A = reader.ReadByte()
                },
                Flags = reader.ReadUInt32(),
                DoodadSet = reader.ReadUInt16(),
                NameSet = reader.ReadUInt16(),
                Unknown1 = reader.ReadUInt16(),
                Unknown2 = reader.ReadUInt16(),
                Unknown3 = reader.ReadUInt32()
            };
            
            WmoInstances.Add(instance);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var instance in WmoInstances)
        {
            writer.Write(instance.NameId);
            writer.Write(instance.UniqueId);
            
            writer.Write(instance.Position.X);
            writer.Write(instance.Position.Y);
            writer.Write(instance.Position.Z);
            
            writer.Write(instance.Rotation.X);
            writer.Write(instance.Rotation.Y);
            writer.Write(instance.Rotation.Z);
            
            writer.Write(instance.Scale.X);
            writer.Write(instance.Scale.Y);
            writer.Write(instance.Scale.Z);
            
            writer.Write(instance.AmbientColor.R);
            writer.Write(instance.AmbientColor.G);
            writer.Write(instance.AmbientColor.B);
            writer.Write(instance.AmbientColor.A);
            
            writer.Write(instance.Flags);
            writer.Write(instance.DoodadSet);
            writer.Write(instance.NameSet);
            writer.Write(instance.Unknown1);
            writer.Write(instance.Unknown2);
            writer.Write(instance.Unknown3);
        }
    }
    
    // Helper to add a new WMO instance
    public void AddWmoInstance(SMODoodadDef instance)
    {
        WmoInstances.Add(instance);
    }
    
    // Helper to get all instances that reference a specific nameId
    public List<SMODoodadDef> GetInstancesByNameId(uint nameId)
    {
        return WmoInstances.Where(i => i.NameId == nameId).ToList();
    }
}
```

## WMO Positioning System
The coordinate system for WMO placement works as follows:

1. Position (X, Y, Z): World space coordinates for the origin of the WMO
   - X: East-West axis (positive eastward)
   - Y: North-South axis (positive northward)
   - Z: Elevation (positive upward)

2. Rotation (X, Y, Z): Rotation angles in radians
   - X: Pitch (rotation around X-axis)
   - Y: Yaw (rotation around Y-axis)
   - Z: Roll (rotation around Z-axis)
   - Rotation is applied in the order of Y, Z, X (yaw, roll, pitch)

3. Scale (X, Y, Z): Scale factors for each axis
   - Typically all set to 1.0
   - Non-uniform scaling is possible but uncommon

## Ambient Color
The ambient color field affects the overall lighting of the WMO:
- It modifies the ambient light component of the WMO
- Values are in RGBA format (0-255 for each component)
- A common value is (255, 255, 255, 255) for unmodified lighting

## Version Information
- Present in all versions of WDT files if global WMO objects are included
- In version 18, all WMO names will always have their M2 (doodad set) files loaded
- In version 22+, only referenced M2s from a given WMO will be loaded
- For files created post-Cataclysm, the Unknown3 field may contain a FileDataID reference

## Usage Context
The MODF chunk provides placement data for global WMO objects in the world:

- Defines how major structures are positioned in the world
- Controls scale, rotation, and positioning of WMO models
- Sets ambient lighting and other visual properties
- Determines which doodad sets are active within the WMO
- Flags control various behaviors like destructibility

Global WMOs differ from ADT-specific WMOs in several important ways:
1. They are loaded regardless of which ADT tile the player is in
2. They typically span multiple ADT tiles
3. They are defined at the world level rather than the tile level
4. They are often used for major architectural features that need to be visible from a distance

The MODF chunk works together with the MWMO and MWID chunks to define the complete WMO placement system:
1. MWMO stores the unique WMO filenames
2. MWID provides efficient access to these filenames
3. MODF contains the actual placement data for each WMO instance 