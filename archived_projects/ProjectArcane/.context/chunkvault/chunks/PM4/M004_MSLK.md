# M004: MSLK (Links)

## Type
PM4 Geometry Chunk

## Source
PM4 Format Documentation

## Description
The MSLK chunk defines links or connections between vertices using references to indices in the MSPI chunk. Each entry in this chunk specifies a range of indices in the MSPI array, allowing for efficient representation of geometric structures. The MSLK chunk consists of an array of structures containing information about index ranges and associated properties.

## Structure
The MSLK chunk has the following structure:

```csharp
struct MSLK
{
    /*0x00*/ mslk_entry[] mslk;
}

struct mslk_entry
{
    /*0x00*/ uint8_t  _0x00;          // Earlier documentation has this as bitmask32 flags
    /*0x01*/ uint8_t  _0x01;
    /*0x02*/ uint16_t _0x02;          // Always 0 in version_48, likely padding
    /*0x04*/ uint32_t _0x04;          // An index somewhere
    /*0x08*/ int24_t  MSPI_first_index; // -1 if _0x0b is 0
    /*0x0B*/ uint8_t  MSPI_index_count;
    /*0x0C*/ uint32_t _0x0c;          // Always 0xffffffff in version_48
    /*0x10*/ uint16_t _0x10;
    /*0x12*/ uint16_t _0x12;          // Always 0x8000 in version_48
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| _0x00 | uint8_t | Earlier documentation mentions this as bitmask32 flags |
| _0x01 | uint8_t | Unknown purpose |
| _0x02 | uint16_t | Always 0 in version 48, likely padding |
| _0x04 | uint32_t | An index reference to unknown data |
| MSPI_first_index | int24_t | First index to use from MSPI chunk (-1 if MSPI_index_count is 0) |
| MSPI_index_count | uint8_t | Number of consecutive indices to use from MSPI |
| _0x0c | uint32_t | Always 0xffffffff in version 48 |
| _0x10 | uint16_t | Unknown purpose |
| _0x12 | uint16_t | Always 0x8000 in version 48 |

## Dependencies
- **MSPI** - Contains the indices referenced by MSPI_first_index and MSPI_index_count

## Implementation Notes
- Each entry is 20 bytes in size, so the number of entries is the chunk size divided by 20
- MSPI_first_index is a signed 24-bit integer, packed with MSPI_index_count in a 32-bit field
- The value -1 for MSPI_first_index indicates no indices are used (when MSPI_index_count is 0)
- Care should be taken when parsing the 24-bit + 8-bit packed values
- Range validation should ensure MSPI_first_index + MSPI_index_count doesn't exceed MSPI array length

## C# Implementation Example

```csharp
public class MslkChunk : IChunk
{
    public const string Signature = "MSLK";
    public List<MslkEntry> Entries { get; private set; }

    public MslkChunk()
    {
        Entries = new List<MslkEntry>();
    }

    public void Read(BinaryReader reader, uint size)
    {
        // Calculate number of entries
        int entryCount = (int)(size / 20); // Each entry is 20 bytes
        Entries.Clear();

        for (int i = 0; i < entryCount; i++)
        {
            var entry = new MslkEntry
            {
                Flag0 = reader.ReadByte(),
                Flag1 = reader.ReadByte(),
                Padding = reader.ReadUInt16(),
                IndexReference = reader.ReadUInt32()
            };

            // Read the packed first_index (24 bits) and index_count (8 bits)
            uint packedValue = reader.ReadUInt32();
            int firstIndex = (int)(packedValue & 0x00FFFFFF);
            
            // Handle sign extension for 24-bit signed value
            if ((firstIndex & 0x00800000) != 0)
            {
                firstIndex |= unchecked((int)0xFF000000);
            }
            
            entry.MspiFirstIndex = firstIndex;
            entry.MspiIndexCount = (byte)((packedValue >> 24) & 0xFF);
            
            entry.UnknownValue0 = reader.ReadUInt32();
            entry.UnknownValue1 = reader.ReadUInt16();
            entry.UnknownValue2 = reader.ReadUInt16();
            
            Entries.Add(entry);
        }
    }

    public void Write(BinaryWriter writer)
    {
        foreach (var entry in Entries)
        {
            writer.Write(entry.Flag0);
            writer.Write(entry.Flag1);
            writer.Write(entry.Padding);
            writer.Write(entry.IndexReference);
            
            // Pack 24-bit first_index and 8-bit index_count
            uint packedValue = (uint)(entry.MspiFirstIndex & 0x00FFFFFF);
            packedValue |= (uint)(entry.MspiIndexCount << 24);
            writer.Write(packedValue);
            
            writer.Write(entry.UnknownValue0);
            writer.Write(entry.UnknownValue1);
            writer.Write(entry.UnknownValue2);
        }
    }

    public bool ValidateIndices(int mspiArrayLength)
    {
        foreach (var entry in Entries)
        {
            // Skip validation for entries with no indices
            if (entry.MspiIndexCount == 0 || entry.MspiFirstIndex == -1)
                continue;
                
            // Ensure the range is valid
            if (entry.MspiFirstIndex < 0 || 
                entry.MspiFirstIndex >= mspiArrayLength ||
                entry.MspiFirstIndex + entry.MspiIndexCount > mspiArrayLength)
            {
                return false;
            }
        }
        return true;
    }
}

public class MslkEntry
{
    public byte Flag0 { get; set; }              // _0x00
    public byte Flag1 { get; set; }              // _0x01
    public ushort Padding { get; set; }          // _0x02
    public uint IndexReference { get; set; }     // _0x04
    public int MspiFirstIndex { get; set; }      // _0x08 (24 bits)
    public byte MspiIndexCount { get; set; }     // _0x0B (8 bits)
    public uint UnknownValue0 { get; set; }      // _0x0C
    public ushort UnknownValue1 { get; set; }    // _0x10
    public ushort UnknownValue2 { get; set; }    // _0x12
}
```

## Related Information
- Each entry defines a group of indices in MSPI that form a connected structure
- When MSPI_index_count is 0, MSPI_first_index is set to -1
- The _0x0c field is consistently 0xFFFFFFFF in version 48
- The _0x12 field is consistently 0x8000 in version 48
- This chunk is present in both PM4 and PD4 formats with identical structure 