# M009: MSHD (Header)

## Type
PM4 Control Chunk

## Source
PM4 Format Documentation

## Description
The MSHD chunk contains header information for the PM4 file. It provides metadata about the file and likely contains information that helps in the overall interpretation of the data. This chunk typically follows the MVER chunk and precedes the actual data chunks. It functions as a container for file-level information rather than geometric data.

## Structure
The MSHD chunk has the following structure:

```csharp
struct MSHD
{
    /*0x00*/ uint32_t _0x00;
    /*0x04*/ uint32_t _0x04;
    /*0x08*/ uint32_t _0x08;
    /*0x0C*/ uint32_t _0x0c;
    /*0x10*/ uint32_t _0x10;
    /*0x14*/ uint32_t _0x14;
    /*0x18*/ uint32_t _0x18;
    /*0x1C*/ uint32_t _0x1c;
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| _0x00 | uint32_t | Unknown purpose - possible version-specific field |
| _0x04 | uint32_t | Unknown purpose - possible metadata flag |
| _0x08 | uint32_t | Unknown purpose - possible metadata field |
| _0x0c | uint32_t | Unknown purpose - possible metadata field |
| _0x10 | uint32_t | Unknown purpose - possible metadata field |
| _0x14 | uint32_t | Unknown purpose - possible metadata field |
| _0x18 | uint32_t | Unknown purpose - possible metadata field |
| _0x1c | uint32_t | Unknown purpose - possible metadata field |

## Dependencies
- **MVER** - Should be parsed before MSHD to ensure version compatibility

## Implementation Notes
- MSHD is typically the second chunk in a PM4 file, following MVER
- The exact meaning of the fields is not well documented
- All fields are 32-bit unsigned integers
- The total size of this chunk should be 32 bytes (8 fields × 4 bytes)
- Care should be taken to preserve all fields when modifying files, even if their purpose is unknown

## C# Implementation Example

```csharp
public class MshdChunk : IChunk
{
    public const string Signature = "MSHD";
    
    public uint Field00 { get; set; }
    public uint Field04 { get; set; }
    public uint Field08 { get; set; }
    public uint Field0C { get; set; }
    public uint Field10 { get; set; }
    public uint Field14 { get; set; }
    public uint Field18 { get; set; }
    public uint Field1C { get; set; }

    public MshdChunk()
    {
        // Initialize with default values (zeros)
        Field00 = 0;
        Field04 = 0;
        Field08 = 0;
        Field0C = 0;
        Field10 = 0;
        Field14 = 0;
        Field18 = 0;
        Field1C = 0;
    }

    public void Read(BinaryReader reader)
    {
        Field00 = reader.ReadUInt32();
        Field04 = reader.ReadUInt32();
        Field08 = reader.ReadUInt32();
        Field0C = reader.ReadUInt32();
        Field10 = reader.ReadUInt32();
        Field14 = reader.ReadUInt32();
        Field18 = reader.ReadUInt32();
        Field1C = reader.ReadUInt32();
    }

    public void Write(BinaryWriter writer)
    {
        writer.Write(Field00);
        writer.Write(Field04);
        writer.Write(Field08);
        writer.Write(Field0C);
        writer.Write(Field10);
        writer.Write(Field14);
        writer.Write(Field18);
        writer.Write(Field1C);
    }

    // Helper method to verify expected chunk size
    public bool VerifySize(uint chunkSize)
    {
        // MSHD should be exactly 32 bytes
        return chunkSize == 32;
    }

    public override string ToString()
    {
        return $"MSHD: [0x00]={Field00:X8}, [0x04]={Field04:X8}, [0x08]={Field08:X8}, [0x0C]={Field0C:X8}, " +
               $"[0x10]={Field10:X8}, [0x14]={Field14:X8}, [0x18]={Field18:X8}, [0x1C]={Field1C:X8}";
    }
}
```

## Related Information
- MSHD is typically positioned right after the MVER chunk
- The exact purpose of each field is not well documented, but they likely provide metadata for the file
- This structure is present in both PM4 and PD4 formats with identical structure
- MSHD does not contain direct references to other chunks, but other chunks may use information from this header
- Although the meaning of the fields is unknown, they should be preserved during reading/writing to maintain file integrity 