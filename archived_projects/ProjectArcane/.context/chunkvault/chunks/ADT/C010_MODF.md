# C010: MODF

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
Placement information for WMO (World Model Object) models.

## Structure
```csharp
struct SMMapObjDef
{
    uint32_t nameId;           // references an entry in the MWID chunk, which is an index into MWMO
    uint32_t uniqueId;         // this ID should be unique for all ADTs currently loaded. Best, it's unique for the whole map.
    C3Vector position;         // position
    C3Vector rotation;         // rotation, degrees
    CAaBox extents;            // position plus the transformed wmo bounding box. used for defining if a wmo instance is rendered as well as collision.
    uint16_t flags;            // values from enum MODFFlags
    uint16_t doodadSet;        // which WMO doodad set is used
    uint16_t nameSet;          // which WMO name set is used. Used for renaming goldshire inn to northshire inn while using the same model.
    uint16_t scale;            // Legion+ only. fixed point value. divide by 1024 to get real value.
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| nameId | uint32 | Reference to an entry in the MWID chunk (index into MWMO) |
| uniqueId | uint32 | Unique identifier for this WMO |
| position | C3Vector | Position of the WMO in the world |
| rotation | C3Vector | Rotation of the WMO in degrees |
| extents | CAaBox | Position plus the transformed WMO bounding box |
| flags | uint16 | Various flags affecting the WMO |
| doodadSet | uint16 | Which WMO doodad set is used |
| nameSet | uint16 | Which WMO name set is used |
| scale | uint16 | (Legion+) Scale factor, fixed point (divide by 1024) |

## Flag Values (MODFFlags enum)
| Value | Description |
|-------|-------------|
| 0x01 | Destroyable (sets bit 1 in wmo root flags to make it destroyable) |
| 0x02 | Use LOD from parent WMO |
| 0x04 | Has custom water level |
| 0x08 | Unknown |
| 0x10 | Unknown |
| 0x20 | Unused |
| 0x40 | Has vertex colors |
| 0x80 | Outdoor |
| 0x100 | Clift to map height |
| 0x200 | Unknown |
| 0x400 | Unused |
| 0x800 | Unused |
| 0x1000 | Unused |
| 0x2000 | Is File ID instead of name |
| 0x4000 | unk4000 |
| 0x8000 | Has lights |

## Dependencies
- MHDR (C002) - Contains the offset to this chunk
- MWID (C008) - Referenced by nameId to identify which model to use
- MWMO (C007) - Contains the actual WMO filenames

## Additional Structures
```csharp
struct CAaBox
{
    C3Vector min;  // minimum corner of the box
    C3Vector max;  // maximum corner of the box
}
```

## Implementation Notes
- Split files: appears in obj file
- Each entry describes the placement of a WMO model in the world
- Position is in world coordinates (see note on coordinate system)
- Rotation is in degrees
- Extents define the bounding box of the WMO for rendering and collision
- Scale is only used in Legion+ and is a fixed-point value (divide by 1024)

## Implementation Example
```csharp
public class CAaBox
{
    public C3Vector Min { get; set; } = new C3Vector();
    public C3Vector Max { get; set; } = new C3Vector();
}

[Flags]
public enum MODFFlags : ushort
{
    None = 0x0,
    Destroyable = 0x01,
    UseLodFromParent = 0x02,
    HasCustomWaterLevel = 0x04,
    Unknown8 = 0x08,
    Unknown16 = 0x10,
    Unused32 = 0x20,
    HasVertexColors = 0x40,
    Outdoor = 0x80,
    CliftToMapHeight = 0x100,
    Unknown512 = 0x200,
    Unused1024 = 0x400,
    Unused2048 = 0x800,
    Unused4096 = 0x1000,
    IsFileId = 0x2000,
    Unknown16384 = 0x4000,
    HasLights = 0x8000
}

public class MODF
{
    public List<MODFEntry> Entries { get; set; } = new List<MODFEntry>();

    public class MODFEntry
    {
        public uint NameId { get; set; }
        public uint UniqueId { get; set; }
        public C3Vector Position { get; set; } = new C3Vector();
        public C3Vector Rotation { get; set; } = new C3Vector();
        public CAaBox Extents { get; set; } = new CAaBox();
        public MODFFlags Flags { get; set; }
        public ushort DoodadSet { get; set; }
        public ushort NameSet { get; set; }
        public ushort Scale { get; set; }
    }
}
```

## Parsing Example
```csharp
public MODF ParseMODF(byte[] data)
{
    var modf = new MODF();
    using (var ms = new MemoryStream(data))
    using (var reader = new BinaryReader(ms))
    {
        while (ms.Position < ms.Length)
        {
            var entry = new MODF.MODFEntry
            {
                NameId = reader.ReadUInt32(),
                UniqueId = reader.ReadUInt32(),
                Position = new C3Vector
                {
                    X = reader.ReadSingle(),
                    Y = reader.ReadSingle(),
                    Z = reader.ReadSingle()
                },
                Rotation = new C3Vector
                {
                    X = reader.ReadSingle(),
                    Y = reader.ReadSingle(),
                    Z = reader.ReadSingle()
                },
                Extents = new CAaBox
                {
                    Min = new C3Vector
                    {
                        X = reader.ReadSingle(),
                        Y = reader.ReadSingle(),
                        Z = reader.ReadSingle()
                    },
                    Max = new C3Vector
                    {
                        X = reader.ReadSingle(),
                        Y = reader.ReadSingle(),
                        Z = reader.ReadSingle()
                    }
                },
                Flags = (MODFFlags)reader.ReadUInt16(),
                DoodadSet = reader.ReadUInt16(),
                NameSet = reader.ReadUInt16(),
                Scale = reader.ReadUInt16()
            };
            
            modf.Entries.Add(entry);
        }
    }
    return modf;
}
```

## Usage Context
The MODF chunk provides placement information for WMO (World Model Object) models in the world. These are typically larger structures like buildings, bridges, and caves. Each entry specifies which model to use (via nameId), where to place it (position), how to orient it (rotation), its bounding box (extents), and various other properties. The models themselves are referenced through the MWID and MWMO chunks. 