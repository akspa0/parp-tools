# D001: DBC Header

## Type
DBC Structure Component

## Source
DBC Format Documentation

## Description
The DBC header is a fixed-size structure at the beginning of every DBC file that defines the overall file structure, record count, and data organization. It provides essential information needed to parse the file correctly, including the number of records, fields per record, and size of the string block. The header is always the first structure in a DBC file and is consistent across all DBC files regardless of content type.

## Structure
```csharp
struct DBCHeader
{
    /*0x00*/ uint32_t magic;          // 'WDBC' signature (0x43424457)
    /*0x04*/ uint32_t recordCount;    // Number of records in the file
    /*0x08*/ uint32_t fieldCount;     // Number of fields per record
    /*0x0C*/ uint32_t recordSize;     // Size of each record in bytes
    /*0x10*/ uint32_t stringBlockSize;// Size of the string block at the end of the file
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| magic | uint32_t | Magic number identifying the file format, always 'WDBC' (0x43424457) |
| recordCount | uint32_t | Number of records (rows) in the DBC file |
| fieldCount | uint32_t | Number of fields (columns) in each record |
| recordSize | uint32_t | Size of each record in bytes |
| stringBlockSize | uint32_t | Size of the string block in bytes |

## Header Size
The DBC header is always 20 bytes (5 uint32_t values) in size.

## Implementation Notes
- The magic value must be 'WDBC' (0x43424457) for a valid DBC file
- The `recordSize` should equal `fieldCount * 4` as all DBC fields are 4 bytes in size
- String fields in records contain offsets into the string block, not the actual strings
- If `stringBlockSize` is 0, the file contains no string data
- The total file size should equal: `20 + (recordCount * recordSize) + stringBlockSize`
- The header is consistent across all versions of DBC files (Classic through Wrath of the Lich King)

## Implementation Example
```csharp
public class DBCHeader
{
    public const uint MAGIC = 0x43424457; // 'WDBC'
    
    public uint Magic { get; private set; }
    public uint RecordCount { get; private set; }
    public uint FieldCount { get; private set; }
    public uint RecordSize { get; private set; }
    public uint StringBlockSize { get; private set; }
    
    public DBCHeader()
    {
        Magic = MAGIC;
        RecordCount = 0;
        FieldCount = 0;
        RecordSize = 0;
        StringBlockSize = 0;
    }
    
    public void Parse(BinaryReader reader)
    {
        Magic = reader.ReadUInt32();
        
        if (Magic != MAGIC)
            throw new InvalidDataException($"Invalid DBC file: expected magic {MAGIC:X}, got {Magic:X}");
        
        RecordCount = reader.ReadUInt32();
        FieldCount = reader.ReadUInt32();
        RecordSize = reader.ReadUInt32();
        StringBlockSize = reader.ReadUInt32();
        
        // Validate that record size matches field count
        if (RecordSize != FieldCount * 4)
            throw new InvalidDataException($"Invalid DBC header: record size {RecordSize} doesn't match field count {FieldCount} * 4");
    }
    
    public void Write(BinaryWriter writer)
    {
        writer.Write(Magic);
        writer.Write(RecordCount);
        writer.Write(FieldCount);
        writer.Write(RecordSize);
        writer.Write(StringBlockSize);
    }
    
    // Helper method to validate file size
    public bool ValidateFileSize(long fileSize)
    {
        long expectedSize = 20 + (RecordCount * RecordSize) + StringBlockSize;
        return fileSize == expectedSize;
    }
    
    // Helper method to get the offset to the string block
    public uint GetStringBlockOffset()
    {
        return 20 + (RecordCount * RecordSize);
    }
    
    // Helper method to get the offset to the record data
    public uint GetRecordDataOffset()
    {
        return 20;
    }
}
```

## Usage Example
```csharp
// Reading a DBC file
using (FileStream fileStream = new FileStream("Spell.dbc", FileMode.Open, FileAccess.Read))
using (BinaryReader reader = new BinaryReader(fileStream))
{
    // Parse the header
    var header = new DBCHeader();
    header.Parse(reader);
    
    // Validate file size
    if (!header.ValidateFileSize(fileStream.Length))
        throw new InvalidDataException("File size doesn't match expected size from header");
    
    // Now we can parse records and access the string block
    // ...
}
```

## Relationship to Record Format
The header defines key properties needed to parse records:
- `recordCount`: Number of records to read
- `fieldCount`: Number of fields in each record
- `recordSize`: Size of each record in bytes

Records immediately follow the header in the file, starting at offset 20.

## Relationship to String Block
The header provides information about the string block:
- `stringBlockSize`: Size of the string block in bytes
- The string block starts at offset `20 + (recordCount * recordSize)`

String fields in records contain offsets relative to the start of the string block.

## Validation Requirements
A valid DBC file must have:
- Magic signature set to 'WDBC' (0x43424457)
- RecordSize equal to FieldCount * 4
- Total file size matching: 20 + (RecordCount * RecordSize) + StringBlockSize 