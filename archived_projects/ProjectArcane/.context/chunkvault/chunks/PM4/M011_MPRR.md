# M011: MPRR (Reference Data)

## Type
PM4 Data Chunk

## Source
PM4 Format Documentation

## Description
The MPRR chunk contains reference data that relates to the MPRL chunk. It consists of an array of structures that reference MPRL entries, creating relationships between different position data. This chunk is specific to the PM4 format and is not present in PD4 files.

## Structure
The MPRR chunk has the following structure:

```csharp
struct MPRR
{
    /*0x00*/ mprr_entry[] mprr;
}

struct mprr_entry
{
    /*0x00*/ uint32_t _0x00;
    /*0x04*/ uint32_t _0x04;
    /*0x08*/ uint32_t mprl_index; // index into #MPRL
    /*0x0C*/ uint32_t _0x0c;
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| _0x00 | uint32_t | Unknown purpose - possibly a type or flag |
| _0x04 | uint32_t | Unknown purpose - possibly a reference or identifier |
| mprl_index | uint32_t | Index into the MPRL array |
| _0x0c | uint32_t | Unknown purpose - possibly additional reference data |

## Dependencies
- **MPRL** - Contains position data referenced by mprl_index

## Implementation Notes
- Each entry is 16 bytes in size, so the number of entries is the chunk size divided by 16
- The mprl_index field references an entry in the MPRL chunk by its index
- The index should be validated to ensure it's within the bounds of the MPRL array
- The purpose of the other fields is not well documented, but they likely provide additional reference context

## C# Implementation Example

```csharp
public class MprrChunk : IChunk
{
    public const string Signature = "MPRR";
    public List<MprrEntry> Entries { get; private set; }

    public MprrChunk()
    {
        Entries = new List<MprrEntry>();
    }

    public void Read(BinaryReader reader, uint size)
    {
        // Calculate number of entries
        int entryCount = (int)(size / 16); // Each entry is 16 bytes
        Entries.Clear();

        for (int i = 0; i < entryCount; i++)
        {
            var entry = new MprrEntry
            {
                Value1 = reader.ReadUInt32(),
                Value2 = reader.ReadUInt32(),
                MprlIndex = reader.ReadUInt32(),
                Value3 = reader.ReadUInt32()
            };
            
            Entries.Add(entry);
        }
    }

    public void Write(BinaryWriter writer)
    {
        foreach (var entry in Entries)
        {
            writer.Write(entry.Value1);
            writer.Write(entry.Value2);
            writer.Write(entry.MprlIndex);
            writer.Write(entry.Value3);
        }
    }

    // Validate indices against MPRL array length
    public bool ValidateIndices(int mprlArrayLength)
    {
        foreach (var entry in Entries)
        {
            if (entry.MprlIndex >= mprlArrayLength)
            {
                return false;
            }
        }
        return true;
    }

    // Get a reference to the corresponding MPRL entry
    public MprlEntry GetReferencedEntry(MprlChunk mprlChunk, int indexInMprr)
    {
        if (indexInMprr < 0 || indexInMprr >= Entries.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(indexInMprr), "Index out of range for MPRR entries");
        }
        
        uint mprlIndex = Entries[indexInMprr].MprlIndex;
        
        if (mprlIndex >= mprlChunk.Entries.Count)
        {
            throw new InvalidOperationException($"Invalid MPRL index {mprlIndex} referenced by MPRR entry {indexInMprr}");
        }
        
        return mprlChunk.Entries[(int)mprlIndex];
    }
}

public class MprrEntry
{
    public uint Value1 { get; set; }      // _0x00
    public uint Value2 { get; set; }      // _0x04
    public uint MprlIndex { get; set; }   // mprl_index
    public uint Value3 { get; set; }      // _0x0c

    public override string ToString()
    {
        return $"Value1: 0x{Value1:X8}, Value2: 0x{Value2:X8}, MprlIndex: {MprlIndex}, Value3: 0x{Value3:X8}";
    }
}
```

## Related Information
- This chunk is specific to the PM4 format and not present in PD4 files
- The mprl_index field references entries in the MPRL chunk
- The exact purpose of the other fields is not documented
- The relationship between MPRL and MPRR might represent object placement data
- This reference system allows multiple MPRR entries to reference the same position data in MPRL
- This pattern of using index references is similar to how MSVI references MSVT and MSPI references MSPV 