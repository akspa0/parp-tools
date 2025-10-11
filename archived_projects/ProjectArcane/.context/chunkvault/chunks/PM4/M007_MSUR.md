# M007: MSUR (Surface Definitions)

## Type
PM4 Geometry Chunk

## Source
PM4 Format Documentation

## Description
The MSUR chunk defines surfaces or polygons that make up the model geometry. Each entry in this chunk references a range of indices in the MSVI chunk, which in turn reference vertices in the MSVT chunk. This structure allows for efficient reuse of vertex and index data across multiple surfaces. The MSUR chunk consists of an array of structures containing information about index ranges and associated surface properties.

## Structure
The MSUR chunk has the following structure:

```csharp
struct MSUR
{
    /*0x00*/ msur_entry[] msur;
}

struct msur_entry
{
    /*0x00*/ uint8_t  _0x00;          // Earlier documentation has this as bitmask32 flags
    /*0x01*/ uint8_t  _0x01;          // Count of indices in #MSVI
    /*0x02*/ uint8_t  _0x02;
    /*0x03*/ uint8_t  _0x03;          // Always 0 in version_48, likely padding
    /*0x04*/ float    _0x04;
    /*0x08*/ float    _0x08;
    /*0x0C*/ float    _0x0c;
    /*0x10*/ float    _0x10;
    /*0x14*/ uint32_t MSVI_first_index;
    /*0x18*/ uint32_t _0x18;
    /*0x1C*/ uint32_t _0x1c;
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| _0x00 | uint8_t | Earlier documentation mentions this as bitmask32 flags |
| _0x01 | uint8_t | Count of indices to use from MSVI |
| _0x02 | uint8_t | Unknown purpose |
| _0x03 | uint8_t | Always 0 in version 48, likely padding |
| _0x04 | float | Unknown float value, possibly related to surface properties |
| _0x08 | float | Unknown float value, possibly related to surface properties |
| _0x0c | float | Unknown float value, possibly related to surface properties |
| _0x10 | float | Unknown float value, possibly related to surface properties |
| MSVI_first_index | uint32_t | First index to use from MSVI chunk |
| _0x18 | uint32_t | Unknown purpose |
| _0x1c | uint32_t | Unknown purpose |

## Dependencies
- **MSVI** - Contains the indices referenced by MSVI_first_index and _0x01 (count)

## Implementation Notes
- Each entry is 32 bytes in size, so the number of entries is the chunk size divided by 32
- The _0x01 field specifies how many consecutive indices from MSVI to use for this surface
- The indices likely define quads or n-gons rather than triangles
- The four float values (_0x04, _0x08, _0x0c, _0x10) may represent surface properties such as normals, texture coordinates, or material properties
- Range validation should ensure MSVI_first_index + _0x01 doesn't exceed MSVI array length

## C# Implementation Example

```csharp
public class MsurChunk : IChunk
{
    public const string Signature = "MSUR";
    public List<MsurEntry> Entries { get; private set; }

    public MsurChunk()
    {
        Entries = new List<MsurEntry>();
    }

    public void Read(BinaryReader reader, uint size)
    {
        // Calculate number of entries
        int entryCount = (int)(size / 32); // Each entry is 32 bytes
        Entries.Clear();

        for (int i = 0; i < entryCount; i++)
        {
            var entry = new MsurEntry
            {
                Flag = reader.ReadByte(),
                IndexCount = reader.ReadByte(),
                Unknown1 = reader.ReadByte(),
                Padding = reader.ReadByte(),
                Float1 = reader.ReadSingle(),
                Float2 = reader.ReadSingle(),
                Float3 = reader.ReadSingle(),
                Float4 = reader.ReadSingle(),
                MsviFirstIndex = reader.ReadUInt32(),
                Unknown2 = reader.ReadUInt32(),
                Unknown3 = reader.ReadUInt32()
            };
            
            Entries.Add(entry);
        }
    }

    public void Write(BinaryWriter writer)
    {
        foreach (var entry in Entries)
        {
            writer.Write(entry.Flag);
            writer.Write(entry.IndexCount);
            writer.Write(entry.Unknown1);
            writer.Write(entry.Padding);
            writer.Write(entry.Float1);
            writer.Write(entry.Float2);
            writer.Write(entry.Float3);
            writer.Write(entry.Float4);
            writer.Write(entry.MsviFirstIndex);
            writer.Write(entry.Unknown2);
            writer.Write(entry.Unknown3);
        }
    }

    public bool ValidateIndices(int msviArrayLength)
    {
        foreach (var entry in Entries)
        {
            // Ensure the range is valid
            if (entry.MsviFirstIndex >= msviArrayLength ||
                entry.MsviFirstIndex + entry.IndexCount > msviArrayLength)
            {
                return false;
            }
        }
        return true;
    }
    
    // Get all indices for a specific surface
    public uint[] GetSurfaceIndices(MsurEntry entry, MsviChunk msviChunk)
    {
        if (entry.MsviFirstIndex >= msviChunk.Indices.Count)
            return new uint[0];
            
        int count = Math.Min(entry.IndexCount, msviChunk.Indices.Count - (int)entry.MsviFirstIndex);
        uint[] indices = new uint[count];
        
        for (int i = 0; i < count; i++)
        {
            indices[i] = msviChunk.Indices[(int)entry.MsviFirstIndex + i];
        }
        
        return indices;
    }
}

public class MsurEntry
{
    public byte Flag { get; set; }                // _0x00
    public byte IndexCount { get; set; }          // _0x01
    public byte Unknown1 { get; set; }            // _0x02
    public byte Padding { get; set; }             // _0x03
    public float Float1 { get; set; }             // _0x04
    public float Float2 { get; set; }             // _0x08
    public float Float3 { get; set; }             // _0x0C
    public float Float4 { get; set; }             // _0x10
    public uint MsviFirstIndex { get; set; }      // _0x14
    public uint Unknown2 { get; set; }            // _0x18
    public uint Unknown3 { get; set; }            // _0x1C
}
```

## Related Information
- Each entry defines a surface using a range of indices from MSVI that reference vertices in MSVT
- The IndexCount field (_0x01) specifies how many consecutive indices to use from MSVI starting at MsviFirstIndex
- The indices likely define quads or n-gons rather than triangles
- The four float values might represent surface properties like normals, material indices, or texture parameters
- This chunk is present in both PM4 and PD4 formats with identical structure
- When implementing rendering, special care must be taken to properly transform MSVT vertices using the coordinate transformation formulas 