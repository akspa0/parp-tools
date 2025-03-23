# C009: MDDF

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
Placement information for doodads (M2 models).

## Structure
```csharp
struct SMDoodadDef
{
    uint32_t nameId;         // references an entry in the MMID chunk, which is an index into MMDX
    uint32_t uniqueId;       // this ID should be unique for all ADTs currently loaded. Best, it's unique for the whole map.
    C3Vector position;       // position
    C3Vector rotation;       // rotation, degrees
    uint16_t scale;          // scale * 1024
    uint16_t flags;          // 0x1: biodome/invisible 
                            // 0x8: was: Use MDX / internal matrix for transformations
                            // 0x40: was: indoor, now: legacy data, added prior to WoD.
                            // 0x80: is fileID not filename
                            // 0x1000: WoD+. Entry models that have been modified since WMO was placed will have this flag set. Otherwise, read from WMO. Always set for new entries.
                            // 0x8000: WoD+. See above.
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| nameId | uint32 | Reference to an entry in the MMID chunk (index into MMDX) |
| uniqueId | uint32 | Unique identifier for this doodad |
| position | C3Vector | Position of the doodad in the world |
| rotation | C3Vector | Rotation of the doodad in degrees |
| scale | uint16 | Scale factor * 1024 |
| flags | uint16 | Various flags affecting the doodad |

## Flag Values
| Value | Description |
|-------|-------------|
| 0x1 | Biodome/invisible |
| 0x8 | Was: Use MDX / internal matrix for transformations |
| 0x40 | Was: indoor, now: legacy data, added prior to WoD |
| 0x80 | Is fileID not filename |
| 0x1000 | WoD+. Entry models modified since WMO placement. Always set for new entries |
| 0x8000 | WoD+. Related to model modifications |

## Dependencies
- MHDR (C002) - Contains the offset to this chunk
- MMID (C006) - Referenced by nameId to identify which model to use
- MMDX (C005) - Contains the actual model filenames

## Implementation Notes
- Split files: appears in obj file
- Each entry describes the placement of a doodad (M2 model) in the world
- Position is in world coordinates (see note on coordinate system)
- Rotation is in degrees
- Scale is multiplied by 1024 (fixed point)

## Coordinate System Translation
An important note about the coordinate system used: WoW's main coordinate system is right-handed:
- The positive X-axis points north, the positive Y-axis points west
- The Z-axis is vertical height, with 0 being sea level
- The origin of the coordinate system is in the center of the map

## Implementation Example
```csharp
public class C3Vector
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Z { get; set; }
}

[Flags]
public enum DoodadFlags : ushort
{
    None = 0x0,
    Invisible = 0x1,
    MatrixTransform = 0x8,
    LegacyData = 0x40,
    FileId = 0x80,
    ModifiedSinceWmoPlacement = 0x1000,
    ModificationRelated = 0x8000
}

public class MDDF
{
    public List<MDDFEntry> Entries { get; set; } = new List<MDDFEntry>();

    public class MDDFEntry
    {
        public uint NameId { get; set; }
        public uint UniqueId { get; set; }
        public C3Vector Position { get; set; } = new C3Vector();
        public C3Vector Rotation { get; set; } = new C3Vector();
        public ushort Scale { get; set; }
        public DoodadFlags Flags { get; set; }
    }
}
```

## Parsing Example
```csharp
public MDDF ParseMDDF(byte[] data)
{
    var mddf = new MDDF();
    using (var ms = new MemoryStream(data))
    using (var reader = new BinaryReader(ms))
    {
        while (ms.Position < ms.Length)
        {
            var entry = new MDDF.MDDFEntry
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
                Scale = reader.ReadUInt16(),
                Flags = (DoodadFlags)reader.ReadUInt16()
            };
            
            mddf.Entries.Add(entry);
        }
    }
    return mddf;
}
```

## Usage Context
The MDDF chunk provides placement information for doodads (M2 models) in the world. Each entry specifies which model to use (via nameId), where to place it (position), how to orient it (rotation), and how to scale it (scale). The models themselves are referenced through the MMID and MMDX chunks. 