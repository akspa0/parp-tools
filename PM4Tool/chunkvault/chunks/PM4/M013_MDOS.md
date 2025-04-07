# M013: MDOS (Object Data)

## Type
PM4 Data Chunk

## Source
PM4 Format Documentation

## Description
The MDOS chunk contains data related to object placement or properties within the map. This chunk is specific to the PM4 format and is not present in PD4 files. It consists of an array of structures containing object-related data, though the exact nature of this data is not well documented.

## Structure
The MDOS chunk has the following structure:

```csharp
struct MDOS
{
    /*0x00*/ mdos_entry[] mdos;
}

struct mdos_entry
{
    /*0x00*/ uint32_t _0x00;
    /*0x04*/ uint32_t _0x04;
    /*0x08*/ uint32_t _0x08;
    /*0x0C*/ uint32_t _0x0c;
    /*0x10*/ uint32_t _0x10;
    /*0x14*/ uint32_t _0x14;
    /*0x18*/ uint32_t _0x18;
    /*0x1C*/ uint32_t _0x1c;
    /*0x20*/ uint32_t _0x20;
    /*0x24*/ uint32_t _0x24;
    /*0x28*/ uint32_t _0x28;
    /*0x2C*/ uint32_t _0x2c;
    /*0x30*/ uint32_t _0x30;
    /*0x34*/ uint32_t _0x34;
    /*0x38*/ uint32_t _0x38;
    /*0x3C*/ uint32_t _0x3c;
    /*0x40*/ uint32_t _0x40;
    /*0x44*/ uint32_t _0x44;
    /*0x48*/ uint32_t _0x48;
    /*0x4C*/ uint32_t _0x4c;
    /*0x50*/ uint32_t _0x50;
    /*0x54*/ uint32_t _0x54;
    /*0x58*/ uint32_t _0x58;
    /*0x5C*/ uint32_t _0x5c;
    /*0x60*/ uint32_t _0x60;
    /*0x64*/ uint32_t _0x64;
    /*0x68*/ uint32_t _0x68;
    /*0x6C*/ uint32_t _0x6c;
    /*0x70*/ uint32_t _0x70;
    /*0x74*/ uint32_t _0x74;
    /*0x78*/ uint32_t _0x78;
    /*0x7C*/ uint32_t _0x7c;
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| _0x00-_0x7c | uint32_t | A series of 32 uint32_t values with undefined purposes |

## Dependencies
None directly documented.

## Implementation Notes
- Each entry is 128 bytes in size (32 uint32_t fields × 4 bytes), so the number of entries is the chunk size divided by 128
- The purpose of these fields is not well documented
- Some fields may represent position, rotation, scale, object type, or other properties
- Some fields may be flags, indices, or references to other data
- All fields should be preserved during reading/writing to maintain file integrity

## C# Implementation Example

```csharp
public class MdosChunk : IChunk
{
    public const string Signature = "MDOS";
    public List<MdosEntry> Entries { get; private set; }

    public MdosChunk()
    {
        Entries = new List<MdosEntry>();
    }

    public void Read(BinaryReader reader, uint size)
    {
        // Calculate number of entries
        int entryCount = (int)(size / 128); // Each entry is 128 bytes (32 uint32_t values)
        Entries.Clear();

        for (int i = 0; i < entryCount; i++)
        {
            var entry = new MdosEntry();
            
            // Read all 32 uint32_t values
            for (int j = 0; j < 32; j++)
            {
                entry.Values[j] = reader.ReadUInt32();
            }
            
            Entries.Add(entry);
        }
    }

    public void Write(BinaryWriter writer)
    {
        foreach (var entry in Entries)
        {
            // Write all 32 uint32_t values
            for (int j = 0; j < 32; j++)
            {
                writer.Write(entry.Values[j]);
            }
        }
    }

    // Helper method to verify expected chunk size
    public bool VerifySize(uint chunkSize)
    {
        // Size should be a multiple of 128 bytes
        return chunkSize % 128 == 0;
    }
}

public class MdosEntry
{
    public uint[] Values { get; private set; }

    public MdosEntry()
    {
        Values = new uint[32];
    }

    // Common property naming pattern for accessing array elements
    public uint Field00 { get => Values[0]; set => Values[0] = value; }
    public uint Field04 { get => Values[1]; set => Values[1] = value; }
    public uint Field08 { get => Values[2]; set => Values[2] = value; }
    public uint Field0C { get => Values[3]; set => Values[3] = value; }
    public uint Field10 { get => Values[4]; set => Values[4] = value; }
    public uint Field14 { get => Values[5]; set => Values[5] = value; }
    public uint Field18 { get => Values[6]; set => Values[6] = value; }
    public uint Field1C { get => Values[7]; set => Values[7] = value; }
    public uint Field20 { get => Values[8]; set => Values[8] = value; }
    public uint Field24 { get => Values[9]; set => Values[9] = value; }
    public uint Field28 { get => Values[10]; set => Values[10] = value; }
    public uint Field2C { get => Values[11]; set => Values[11] = value; }
    public uint Field30 { get => Values[12]; set => Values[12] = value; }
    public uint Field34 { get => Values[13]; set => Values[13] = value; }
    public uint Field38 { get => Values[14]; set => Values[14] = value; }
    public uint Field3C { get => Values[15]; set => Values[15] = value; }
    public uint Field40 { get => Values[16]; set => Values[16] = value; }
    public uint Field44 { get => Values[17]; set => Values[17] = value; }
    public uint Field48 { get => Values[18]; set => Values[18] = value; }
    public uint Field4C { get => Values[19]; set => Values[19] = value; }
    public uint Field50 { get => Values[20]; set => Values[20] = value; }
    public uint Field54 { get => Values[21]; set => Values[21] = value; }
    public uint Field58 { get => Values[22]; set => Values[22] = value; }
    public uint Field5C { get => Values[23]; set => Values[23] = value; }
    public uint Field60 { get => Values[24]; set => Values[24] = value; }
    public uint Field64 { get => Values[25]; set => Values[25] = value; }
    public uint Field68 { get => Values[26]; set => Values[26] = value; }
    public uint Field6C { get => Values[27]; set => Values[27] = value; }
    public uint Field70 { get => Values[28]; set => Values[28] = value; }
    public uint Field74 { get => Values[29]; set => Values[29] = value; }
    public uint Field78 { get => Values[30]; set => Values[30] = value; }
    public uint Field7C { get => Values[31]; set => Values[31] = value; }

    public override string ToString()
    {
        // Format the first few fields for debug output
        return $"MDOS Entry: [0x00]={Field00:X8}, [0x04]={Field04:X8}, [0x08]={Field08:X8}, [0x0C]={Field0C:X8}, ...";
    }
}
```

## Related Information
- This chunk is specific to the PM4 format and not present in PD4 files
- The exact purpose of the fields is not well documented
- The large number of fields suggests this may contain detailed object placement or property data
- Some fields might be positional (x, y, z coordinates), rotational (pitch, yaw, roll), or scale values
- Other fields might be flags, type identifiers, or references to other data in the file
- Without further documentation, all fields should be preserved during reading/writing to maintain file integrity 