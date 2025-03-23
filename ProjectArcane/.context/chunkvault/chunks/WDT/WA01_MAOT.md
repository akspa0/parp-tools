# WA01: MAOT

## Type
Alpha WDT Chunk

## Source
Alpha.md

## Description
The MAOT (Map Object Table) chunk is a key component of the Alpha WDT format, containing size and offset information for map objects. Unlike the modern WDT format which references separate ADT files, the Alpha format embeds terrain data directly within the WDT file, and MAOT serves as an index to this embedded data.

## Structure
```csharp
struct SMMapObjDef // sizeof(0x28)
{
    /*0x00*/ uint32_t size;       // Size of the map object data
    /*0x04*/ uint32_t offset;     // Offset to the map object data (MAOI chunk)
    /*0x08*/ float height1;       // Minimum height value
    /*0x0C*/ float height2;       // Maximum height value
    /*0x10*/ C3Vector position;   // Position in world space
    /*0x1C*/ uint32_t flags;      // Object flags
    /*0x20*/ uint32_t unk;        // Unknown value
    /*0x24*/ uint32_t ground;     // Ground type or information
};

struct MAOT
{
    /*0x00*/ uint32_t num_entries;           // Number of entries in the table
    /*0x04*/ SMMapObjDef map_objects[];      // Array of map object definitions
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| num_entries | uint32_t | Number of map object entries in the table |
| map_objects | SMMapObjDef[] | Array of map object definitions |

### SMMapObjDef Properties
| Name | Type | Description |
|------|------|-------------|
| size | uint32_t | Size in bytes of the map object data in the MAOI chunk |
| offset | uint32_t | Offset in bytes to the map object data in the MAOI chunk |
| height1 | float | Minimum height value for this map object |
| height2 | float | Maximum height value for this map object |
| position | C3Vector | Position of the map object in world coordinates |
| flags | uint32_t | Various flags for this map object |
| unk | uint32_t | Unknown value/purpose |
| ground | uint32_t | Ground type information |

## Dependencies
- MAOI (WA02) - Contains the actual map object data referenced by this table
- MAOH (WA03) - Contains header information for the map objects

## Implementation Notes
- The MAOT chunk serves as an index or table of contents for the terrain data embedded in the Alpha WDT file
- Each entry points to a section of data in the MAOI chunk using size and offset values
- The position field indicates the world location of this map object (similar to ADT tile positions in modern format)
- Height values (height1 and height2) appear to define the vertical bounds of the terrain
- Unlike modern WDT which uses a fixed 64×64 grid, the Alpha format may use a more flexible object-based approach
- The flags field likely contains information about the object type and properties
- The ground field may indicate terrain type or other ground-related information

## Implementation Example
```csharp
public class SMMapObjDef
{
    public uint Size { get; set; }           // Size of object data
    public uint Offset { get; set; }         // Offset to object data
    public float MinHeight { get; set; }     // Minimum height
    public float MaxHeight { get; set; }     // Maximum height
    public Vector3 Position { get; set; }    // Position
    public uint Flags { get; set; }          // Flags
    public uint Unknown { get; set; }        // Unknown
    public uint Ground { get; set; }         // Ground type
    
    // Helper method to calculate height range
    public float HeightRange => MaxHeight - MinHeight;
    
    // Helper method to check if a world position is within this object's bounds
    // (This is a simplified example - actual implementation would be more complex)
    public bool ContainsPosition(float x, float y)
    {
        // Assuming each object covers a square area of side length 533.33333 yards (ADT tile size)
        const float TILE_SIZE = 533.33333f;
        float halfSize = TILE_SIZE / 2;
        
        return (x >= Position.X - halfSize && x <= Position.X + halfSize &&
                y >= Position.Y - halfSize && y <= Position.Y + halfSize);
    }
}

public class MAOT : IChunk
{
    public uint NumEntries { get; private set; }
    public List<SMMapObjDef> MapObjects { get; private set; } = new List<SMMapObjDef>();
    
    public void Parse(BinaryReader reader, long size)
    {
        NumEntries = reader.ReadUInt32();
        MapObjects.Clear();
        
        for (int i = 0; i < NumEntries; i++)
        {
            var mapObj = new SMMapObjDef
            {
                Size = reader.ReadUInt32(),
                Offset = reader.ReadUInt32(),
                MinHeight = reader.ReadSingle(),
                MaxHeight = reader.ReadSingle(),
                Position = new Vector3
                {
                    X = reader.ReadSingle(),
                    Y = reader.ReadSingle(),
                    Z = reader.ReadSingle()
                },
                Flags = reader.ReadUInt32(),
                Unknown = reader.ReadUInt32(),
                Ground = reader.ReadUInt32()
            };
            
            MapObjects.Add(mapObj);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        writer.Write(NumEntries);
        
        foreach (var mapObj in MapObjects)
        {
            writer.Write(mapObj.Size);
            writer.Write(mapObj.Offset);
            writer.Write(mapObj.MinHeight);
            writer.Write(mapObj.MaxHeight);
            
            writer.Write(mapObj.Position.X);
            writer.Write(mapObj.Position.Y);
            writer.Write(mapObj.Position.Z);
            
            writer.Write(mapObj.Flags);
            writer.Write(mapObj.Unknown);
            writer.Write(mapObj.Ground);
        }
    }
    
    // Helper method to find map object containing a world position
    public SMMapObjDef FindMapObjectAt(float x, float y)
    {
        return MapObjects.FirstOrDefault(obj => obj.ContainsPosition(x, y));
    }
    
    // Helper method to get map object data from MAOI chunk
    public byte[] GetMapObjectData(SMMapObjDef mapObj, MAOI maoi)
    {
        return maoi.GetObjectData((int)mapObj.Offset, (int)mapObj.Size);
    }
}
```

## Terrain Organization
In the Alpha WDT format, terrain appears to be organized into discrete objects defined by the MAOT entries:

1. Each map object has a position in world space
2. Map objects likely correspond to what would later become ADT tiles
3. The height range (height1 to height2) defines the vertical bounds of the terrain
4. The size and offset fields point to the associated terrain data in the MAOI chunk
5. This approach allows for a more dynamic terrain organization than the fixed grid used in later versions

## Version Information
- Present only in the Alpha version of the WDT format
- Replaced in later versions by the MAIN chunk and separate ADT files
- Represents an earlier, more self-contained approach to world data storage

## Architectural Significance
The MAOT chunk highlights the fundamental architectural difference between Alpha and modern WDT formats:

- **Alpha (Self-Contained)**: MAOT indexes terrain data embedded directly within the WDT file
- **Modern (Reference-Based)**: MAIN indexes separate ADT files that contain terrain data

This evolution from a monolithic to a modular approach likely occurred to improve:
1. Memory efficiency (only loading needed terrain)
2. Development workflow (allowing parallel work on different map areas)
3. Data management (easier to update individual map sections) 