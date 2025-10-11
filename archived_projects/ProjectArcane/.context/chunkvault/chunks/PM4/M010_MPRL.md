# M010: MPRL (Position Data)

## Type
PM4 Data Chunk

## Source
PM4 Format Documentation

## Description
The MPRL chunk contains position and reference data that is used by the MPRR chunk. It consists of an array of structures containing positional information and reference data for in-game objects or locations. This chunk is specific to the PM4 format and is not present in PD4 files.

## Structure
The MPRL chunk has the following structure:

```csharp
struct MPRL
{
    /*0x00*/ mprl_entry[] mprl;
}

struct mprl_entry
{
    /*0x00*/ C3Vector position;
    /*0x0C*/ float _0x0c;
    /*0x10*/ float _0x10;
    /*0x14*/ float _0x14;
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| position | C3Vector | 3D position in the world (x, y, z) |
| _0x0c | float | Unknown purpose - possibly rotation or scale |
| _0x10 | float | Unknown purpose - possibly rotation or scale |
| _0x14 | float | Unknown purpose - possibly rotation or scale |

## Dependencies
None directly, but this chunk is referenced by:
- MPRR (Reference Data) - Contains references to entries in this chunk

## Implementation Notes
- Each entry is 24 bytes in size (12 for C3Vector + 3 floats × 4 bytes), so the number of entries is the chunk size divided by 24
- The position vector uses the standard XYZ ordering in float format
- The three additional float values may represent rotation, scale, or other transformation data
- There is no documented special coordinate transformation for these positions, unlike MSVT vertices

## C# Implementation Example

```csharp
public class MprlChunk : IChunk
{
    public const string Signature = "MPRL";
    public List<MprlEntry> Entries { get; private set; }

    public MprlChunk()
    {
        Entries = new List<MprlEntry>();
    }

    public void Read(BinaryReader reader, uint size)
    {
        // Calculate number of entries
        int entryCount = (int)(size / 24); // Each entry is 24 bytes
        Entries.Clear();

        for (int i = 0; i < entryCount; i++)
        {
            var position = new C3Vector
            {
                X = reader.ReadSingle(),
                Y = reader.ReadSingle(),
                Z = reader.ReadSingle()
            };

            var entry = new MprlEntry
            {
                Position = position,
                Float1 = reader.ReadSingle(),
                Float2 = reader.ReadSingle(),
                Float3 = reader.ReadSingle()
            };
            
            Entries.Add(entry);
        }
    }

    public void Write(BinaryWriter writer)
    {
        foreach (var entry in Entries)
        {
            writer.Write(entry.Position.X);
            writer.Write(entry.Position.Y);
            writer.Write(entry.Position.Z);
            writer.Write(entry.Float1);
            writer.Write(entry.Float2);
            writer.Write(entry.Float3);
        }
    }

    // Get entry by index with bounds checking
    public MprlEntry GetEntry(int index)
    {
        if (index < 0 || index >= Entries.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(index), "Index out of range for MPRL entries");
        }
        
        return Entries[index];
    }
}

public class MprlEntry
{
    public C3Vector Position { get; set; }
    public float Float1 { get; set; }   // _0x0c
    public float Float2 { get; set; }   // _0x10
    public float Float3 { get; set; }   // _0x14

    public override string ToString()
    {
        return $"Position: {Position}, Float1: {Float1}, Float2: {Float2}, Float3: {Float3}";
    }
}

public struct C3Vector
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Z { get; set; }

    public override string ToString() => $"({X}, {Y}, {Z})";
}
```

## Related Information
- This chunk is specific to the PM4 format and not present in PD4 files
- The MPRR chunk contains references to entries in this chunk
- The exact purpose of the additional float values is not documented
- Unlike MSVT, no special coordinate transformation is documented for these positions
- The positions may represent object placements, waypoints, or other location-based data in the game world 