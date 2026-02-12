# M014: MDSF (Structure Data)

## Type
PM4 Data Chunk

## Source
PM4 Format Documentation

## Description
The MDSF chunk contains data related to structures or static objects within the map. This chunk is specific to the PM4 format and is not present in PD4 files. It consists of an array of structures containing data about in-game structures, though the exact nature of this data is not well documented.

## Structure
The MDSF chunk has the following structure:

```csharp
struct MDSF
{
    /*0x00*/ mdsf_entry[] mdsf;
}

struct mdsf_entry
{
    /*0x00*/ uint32_t _0x00;
    /*0x04*/ uint32_t _0x04;
    /*0x08*/ uint32_t _0x08;
    /*0x0C*/ uint32_t _0x0c;
    /*0x10*/ uint32_t _0x10;
    /*0x14*/ uint32_t _0x14;
    /*0x18*/ float _0x18;
    /*0x1C*/ float _0x1c;
    /*0x20*/ float _0x20;
    /*0x24*/ float _0x24;
    /*0x28*/ float _0x28;
    /*0x2C*/ float _0x2c;
    /*0x30*/ float _0x30;
    /*0x34*/ float _0x34;
    /*0x38*/ uint32_t _0x38;
    /*0x3C*/ uint32_t _0x3c;
    /*0x40*/ uint32_t _0x40;
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| _0x00-_0x14 | uint32_t | Six uint32_t values with undefined purposes |
| _0x18-_0x34 | float | Seven float values with undefined purposes (possibly position, rotation, scale) |
| _0x38-_0x40 | uint32_t | Three uint32_t values with undefined purposes |

## Dependencies
None directly documented.

## Implementation Notes
- Each entry is 68 bytes in size (16 fields with varying types), so the number of entries is the chunk size divided by 68
- The purpose of these fields is not well documented
- Based on the field types, it appears that:
  - The first six fields (_0x00-_0x14) are integer values, possibly flags, indices, or identifiers
  - The next seven fields (_0x18-_0x34) are floating-point values, suggesting position, rotation, scale, or other geometric properties
  - The last three fields (_0x38-_0x40) are integer values, possibly additional indices or flags
- All fields should be preserved during reading/writing to maintain file integrity

## C# Implementation Example

```csharp
public class MdsfChunk : IChunk
{
    public const string Signature = "MDSF";
    public List<MdsfEntry> Entries { get; private set; }

    public MdsfChunk()
    {
        Entries = new List<MdsfEntry>();
    }

    public void Read(BinaryReader reader, uint size)
    {
        // Calculate number of entries
        int entryCount = (int)(size / 68); // Each entry is 68 bytes
        Entries.Clear();

        for (int i = 0; i < entryCount; i++)
        {
            var entry = new MdsfEntry
            {
                // Read integer values
                Value00 = reader.ReadUInt32(),
                Value04 = reader.ReadUInt32(),
                Value08 = reader.ReadUInt32(),
                Value0C = reader.ReadUInt32(),
                Value10 = reader.ReadUInt32(),
                Value14 = reader.ReadUInt32(),
                
                // Read float values
                Float18 = reader.ReadSingle(),
                Float1C = reader.ReadSingle(),
                Float20 = reader.ReadSingle(),
                Float24 = reader.ReadSingle(),
                Float28 = reader.ReadSingle(),
                Float2C = reader.ReadSingle(),
                Float30 = reader.ReadSingle(),
                Float34 = reader.ReadSingle(),
                
                // Read remaining integer values
                Value38 = reader.ReadUInt32(),
                Value3C = reader.ReadUInt32(),
                Value40 = reader.ReadUInt32()
            };
            
            Entries.Add(entry);
        }
    }

    public void Write(BinaryWriter writer)
    {
        foreach (var entry in Entries)
        {
            // Write integer values
            writer.Write(entry.Value00);
            writer.Write(entry.Value04);
            writer.Write(entry.Value08);
            writer.Write(entry.Value0C);
            writer.Write(entry.Value10);
            writer.Write(entry.Value14);
            
            // Write float values
            writer.Write(entry.Float18);
            writer.Write(entry.Float1C);
            writer.Write(entry.Float20);
            writer.Write(entry.Float24);
            writer.Write(entry.Float28);
            writer.Write(entry.Float2C);
            writer.Write(entry.Float30);
            writer.Write(entry.Float34);
            
            // Write remaining integer values
            writer.Write(entry.Value38);
            writer.Write(entry.Value3C);
            writer.Write(entry.Value40);
        }
    }

    // Helper method to verify expected chunk size
    public bool VerifySize(uint chunkSize)
    {
        // Size should be a multiple of 68 bytes
        return chunkSize % 68 == 0;
    }
}

public class MdsfEntry
{
    // Integer values
    public uint Value00 { get; set; }
    public uint Value04 { get; set; }
    public uint Value08 { get; set; }
    public uint Value0C { get; set; }
    public uint Value10 { get; set; }
    public uint Value14 { get; set; }
    
    // Float values (possibly position, rotation, scale)
    public float Float18 { get; set; }
    public float Float1C { get; set; }
    public float Float20 { get; set; }
    public float Float24 { get; set; }
    public float Float28 { get; set; }
    public float Float2C { get; set; }
    public float Float30 { get; set; }
    public float Float34 { get; set; }
    
    // Additional integer values
    public uint Value38 { get; set; }
    public uint Value3C { get; set; }
    public uint Value40 { get; set; }

    public override string ToString()
    {
        return $"MDSF Entry: [0x00]={Value00:X8}, Floats: ({Float18}, {Float1C}, {Float20})...";
    }
    
    // Helper method to get potential position vector (assuming these fields represent position)
    public Vector3 GetPotentialPosition()
    {
        return new Vector3(Float18, Float1C, Float20);
    }
    
    // Helper method to get potential rotation vector (assuming these fields represent rotation)
    public Vector3 GetPotentialRotation()
    {
        return new Vector3(Float24, Float28, Float2C);
    }
    
    // Helper method to get potential scale (assuming this field represents scale)
    public float GetPotentialScale()
    {
        return Float30;
    }
}

public struct Vector3
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Z { get; set; }

    public Vector3(float x, float y, float z)
    {
        X = x;
        Y = y;
        Z = z;
    }

    public override string ToString() => $"({X}, {Y}, {Z})";
}
```

## Related Information
- This chunk is specific to the PM4 format and not present in PD4 files
- The exact purpose of the fields is not well documented
- The mix of integer and floating-point values suggests this chunk contains both reference data and geometric data
- The floating-point values (_0x18-_0x34) may represent position, rotation, scale, or other geometric properties
- The integer values may represent structure types, flags, or references to other data
- The chunk name suggests it's related to structures in the game world, which might include buildings, walls, or other static objects
- Without further documentation, all fields should be preserved during reading/writing to maintain file integrity 