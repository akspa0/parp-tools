# D004: DB2 Header

## Type
DB2 Structure Component

## Source
DB2 Format Documentation

## Description
The DB2 header is an extended version of the DBC header, introduced in Cataclysm (WoW 4.0+) to add more features to the database container format. It provides additional metadata about the file's structure, contents, and localization. The DB2 format went through several revisions (WDB2, WDB5, WDB6), each adding more capabilities while maintaining backwards compatibility with the core concepts.

## Structure
The DB2 header has evolved through several versions, with the base structure (WDB2) as follows:

```csharp
struct DB2Header_V1 // WDB2
{
    /*0x00*/ uint32_t magic;           // 'WDB2' signature (0x32424457)
    /*0x04*/ uint32_t recordCount;     // Number of records
    /*0x08*/ uint32_t fieldCount;      // Number of fields per record
    /*0x0C*/ uint32_t recordSize;      // Size of a record in bytes
    /*0x10*/ uint32_t stringBlockSize; // Size of the string block
    /*0x14*/ uint32_t tableHash;       // Hash of the table name
    /*0x18*/ uint32_t layoutHash;      // Hash of the data layout
    /*0x1C*/ uint32_t minId;           // Minimum ID in the table
    /*0x20*/ uint32_t maxId;           // Maximum ID in the table
    /*0x24*/ uint32_t locale;          // Locale (language) of the file
    /*0x28*/ uint32_t copyTableSize;   // Size of copy table in bytes
};
```

Later versions (WDB5, WDB6) added more fields:

```csharp
struct DB2Header_V2 // WDB5/WDB6
{
    /*0x00*/ uint32_t magic;           // 'WDB5' or 'WDB6' signature
    /*0x04*/ uint32_t recordCount;     // Number of records
    /*0x08*/ uint32_t fieldCount;      // Number of fields per record
    /*0x0C*/ uint32_t recordSize;      // Size of a record in bytes
    /*0x10*/ uint32_t stringBlockSize; // Size of the string block
    /*0x14*/ uint32_t tableHash;       // Hash of the table name
    /*0x18*/ uint32_t layoutHash;      // Hash of the data layout
    /*0x1C*/ uint32_t minId;           // Minimum ID in the table
    /*0x20*/ uint32_t maxId;           // Maximum ID in the table
    /*0x24*/ uint32_t locale;          // Locale (language) of the file
    /*0x28*/ uint32_t copyTableSize;   // Size of copy table in bytes
    /*0x2C*/ uint16_t flags;           // Flags for file format options
    /*0x2E*/ uint16_t idIndex;         // Index of the ID field (WDB5 only)
    /*0x2E*/ uint16_t totalFieldCount; // Total field count including dynamic fields (WDB6 only)
    /*0x30*/ uint32_t bitpackedDataOffset; // Offset to bitpacked data
    /*0x34*/ uint32_t lookupColumnCount;   // Number of lookup columns
    /*0x38*/ uint32_t fieldStorageInfoSize; // Size of field storage info
    /*0x3C*/ uint32_t commonDataSize;      // Size of common data
    /*0x40*/ uint32_t palletDataSize;      // Size of pallet data
    /*0x44*/ uint32_t sectionCount;        // Number of sections (WDB6 only)
};
```

## Properties
### Base Properties (WDB2)
| Name | Type | Description |
|------|------|-------------|
| magic | uint32_t | Magic number: 'WDB2', 'WDB5', or 'WDB6' (depending on version) |
| recordCount | uint32_t | Number of records in the file |
| fieldCount | uint32_t | Number of fields per record |
| recordSize | uint32_t | Size of each record in bytes |
| stringBlockSize | uint32_t | Size of the string block in bytes |
| tableHash | uint32_t | Hash of the table name (CRC-32) |
| layoutHash | uint32_t | Hash of the field layout (CRC-32) |
| minId | uint32_t | Minimum ID in the table |
| maxId | uint32_t | Maximum ID in the table |
| locale | uint32_t | Locale ID for the file (e.g., 0 for English) |
| copyTableSize | uint32_t | Size of the copy table in bytes (for record reuse) |

### Additional Properties (WDB5/WDB6)
| Name | Type | Description |
|------|------|-------------|
| flags | uint16_t | Flags for format options |
| idIndex/totalFieldCount | uint16_t | Index of ID field (WDB5) or total fields (WDB6) |
| bitpackedDataOffset | uint32_t | Offset to bitpacked data section |
| lookupColumnCount | uint32_t | Number of lookup columns |
| fieldStorageInfoSize | uint32_t | Size of field storage information |
| commonDataSize | uint32_t | Size of common data block |
| palletDataSize | uint32_t | Size of pallet data block |
| sectionCount | uint32_t | Number of sections (WDB6 only) |

## Flag Values
The `flags` field in WDB5/WDB6 may contain the following bits:

| Value | Flag Name | Description |
|-------|-----------|-------------|
| 0x01 | HasNonInlineIds | IDs are stored separately, not in the record |
| 0x02 | HasReferenceData | File contains reference data for lookups |
| 0x04 | HasEmbeddedStrings | Strings are embedded in records rather than in string block |
| 0x08 | HasLocalizedStrings | File contains localized strings |
| 0x10 | HasColumnMetadata | File includes metadata about columns |
| 0x20 | HasSparseTable | Table is sparse (not all IDs exist) |
| 0x40 | HasIndexTable | Contains an index table for lookup |
| 0x80 | HasRelationshipData | Contains relationship mapping data |

## Header Size
- WDB2 header: 44 bytes (0x2C)
- WDB5 header: 48 bytes (0x30) + variable sections
- WDB6 header: 68 bytes (0x44) + variable sections

## Implementation Notes
- The DB2 format introduced support for sparse tables where not all IDs between minId and maxId exist
- The tableHash and layoutHash are used to identify specific tables and their structures
- The locale field enables multi-language support within a single file format
- The copyTable enables reuse of record data to save space
- Later versions (WDB5+) introduced advanced compression techniques like bit-packing and palletized data
- WDB6 added support for multiple sections within the same file

## Implementation Example
```csharp
public class DB2Header
{
    // Magic values
    public const uint MAGIC_WDB2 = 0x32424457; // 'WDB2'
    public const uint MAGIC_WDB5 = 0x35424457; // 'WDB5'
    public const uint MAGIC_WDB6 = 0x36424457; // 'WDB6'
    
    // Common fields
    public uint Magic { get; set; }
    public uint RecordCount { get; set; }
    public uint FieldCount { get; set; }
    public uint RecordSize { get; set; }
    public uint StringBlockSize { get; set; }
    public uint TableHash { get; set; }
    public uint LayoutHash { get; set; }
    public uint MinId { get; set; }
    public uint MaxId { get; set; }
    public uint Locale { get; set; }
    public uint CopyTableSize { get; set; }
    
    // WDB5/WDB6 fields
    public ushort Flags { get; set; }
    public ushort IdIndex { get; set; } // WDB5 only
    public ushort TotalFieldCount { get; set; } // WDB6 only
    public uint BitpackedDataOffset { get; set; }
    public uint LookupColumnCount { get; set; }
    public uint FieldStorageInfoSize { get; set; }
    public uint CommonDataSize { get; set; }
    public uint PalletDataSize { get; set; }
    public uint SectionCount { get; set; } // WDB6 only
    
    // Calculated properties
    public bool IsWDB2 => Magic == MAGIC_WDB2;
    public bool IsWDB5 => Magic == MAGIC_WDB5;
    public bool IsWDB6 => Magic == MAGIC_WDB6;
    
    // Flag properties
    public bool HasNonInlineIds => (Flags & 0x01) != 0;
    public bool HasReferenceData => (Flags & 0x02) != 0;
    public bool HasEmbeddedStrings => (Flags & 0x04) != 0;
    public bool HasLocalizedStrings => (Flags & 0x08) != 0;
    public bool HasColumnMetadata => (Flags & 0x10) != 0;
    public bool HasSparseTable => (Flags & 0x20) != 0;
    public bool HasIndexTable => (Flags & 0x40) != 0;
    public bool HasRelationshipData => (Flags & 0x80) != 0;
    
    public DB2Header()
    {
        Magic = MAGIC_WDB2; // Default to WDB2
    }
    
    public void Parse(BinaryReader reader)
    {
        // Read common header fields
        Magic = reader.ReadUInt32();
        
        if (Magic != MAGIC_WDB2 && Magic != MAGIC_WDB5 && Magic != MAGIC_WDB6)
            throw new InvalidDataException($"Invalid DB2 file: expected WDB2/WDB5/WDB6 magic, got {Magic:X}");
        
        RecordCount = reader.ReadUInt32();
        FieldCount = reader.ReadUInt32();
        RecordSize = reader.ReadUInt32();
        StringBlockSize = reader.ReadUInt32();
        TableHash = reader.ReadUInt32();
        LayoutHash = reader.ReadUInt32();
        MinId = reader.ReadUInt32();
        MaxId = reader.ReadUInt32();
        Locale = reader.ReadUInt32();
        CopyTableSize = reader.ReadUInt32();
        
        // Read additional fields for WDB5/WDB6
        if (IsWDB5 || IsWDB6)
        {
            Flags = reader.ReadUInt16();
            
            if (IsWDB5)
                IdIndex = reader.ReadUInt16();
            else // WDB6
                TotalFieldCount = reader.ReadUInt16();
                
            BitpackedDataOffset = reader.ReadUInt32();
            LookupColumnCount = reader.ReadUInt32();
            FieldStorageInfoSize = reader.ReadUInt32();
            CommonDataSize = reader.ReadUInt32();
            PalletDataSize = reader.ReadUInt32();
            
            if (IsWDB6)
                SectionCount = reader.ReadUInt32();
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        // Write common header fields
        writer.Write(Magic);
        writer.Write(RecordCount);
        writer.Write(FieldCount);
        writer.Write(RecordSize);
        writer.Write(StringBlockSize);
        writer.Write(TableHash);
        writer.Write(LayoutHash);
        writer.Write(MinId);
        writer.Write(MaxId);
        writer.Write(Locale);
        writer.Write(CopyTableSize);
        
        // Write additional fields for WDB5/WDB6
        if (IsWDB5 || IsWDB6)
        {
            writer.Write(Flags);
            
            if (IsWDB5)
                writer.Write(IdIndex);
            else // WDB6
                writer.Write(TotalFieldCount);
                
            writer.Write(BitpackedDataOffset);
            writer.Write(LookupColumnCount);
            writer.Write(FieldStorageInfoSize);
            writer.Write(CommonDataSize);
            writer.Write(PalletDataSize);
            
            if (IsWDB6)
                writer.Write(SectionCount);
        }
    }
    
    // Helper method to get the header size
    public int GetHeaderSize()
    {
        if (IsWDB2)
            return 44; // 0x2C
        else if (IsWDB5)
            return 48; // 0x30
        else if (IsWDB6)
            return 68; // 0x44
        else
            return 44; // Default to WDB2 size
    }
    
    // Helper method to get the data offset
    public uint GetDataOffset()
    {
        uint offset = (uint)GetHeaderSize();
        
        // Add sizes of field information blocks in WDB5/WDB6
        if (IsWDB5 || IsWDB6)
        {
            offset += FieldStorageInfoSize;
        }
        
        return offset;
    }
}
```

## Usage Example
```csharp
// Reading a DB2 file
using (FileStream fileStream = new FileStream("Item.db2", FileMode.Open, FileAccess.Read))
using (BinaryReader reader = new BinaryReader(fileStream))
{
    // Parse the header
    var header = new DB2Header();
    header.Parse(reader);
    
    // Determine if this is a sparse table
    if (header.HasSparseTable)
    {
        // Handle sparse table format
        uint sparseCount = header.MaxId - header.MinId + 1;
        Console.WriteLine($"Sparse table with {header.RecordCount} records out of {sparseCount} possible IDs");
    }
    
    // Get the offset to the record data
    uint dataOffset = header.GetDataOffset();
    
    // Handle different format versions
    if (header.IsWDB2)
    {
        Console.WriteLine("Processing WDB2 format");
        // Handle WDB2 specifics...
    }
    else if (header.IsWDB5)
    {
        Console.WriteLine("Processing WDB5 format");
        // Handle WDB5 specifics...
    }
    else if (header.IsWDB6)
    {
        Console.WriteLine("Processing WDB6 format");
        // Handle WDB6 specifics...
    }
}
```

## Evolution from DBC to DB2
The DB2 format represents an evolution of the DBC format with several key improvements:

1. **Sparse Tables**: Support for tables where not all IDs between minimum and maximum exist, saving space.
2. **Metadata**: Additional information about the table structure and layout.
3. **Copy Tables**: Ability to reuse data across multiple records to save space.
4. **Localization**: Improved support for multiple languages within a single file.
5. **Compression**: Advanced techniques like bit-packing and palletized data to reduce file size.
6. **Column Information**: Metadata about column types and storage.
7. **Multiple Sections**: Ability to store multiple related datasets within a single file (WDB6).

## Relationship to Other Components
- **DBC Header**: Base format that DB2 extends with additional fields
- **Record Data**: Follows the header but may use more complex storage formats
- **String Block**: Similar to DBC but may handle multiple languages

## Localization IDs
The `locale` field uses these common values:

| Locale ID | Language |
|-----------|----------|
| 0 | enUS (English) |
| 1 | koKR (Korean) |
| 2 | frFR (French) |
| 3 | deDE (German) |
| 4 | zhCN (Chinese Simplified) |
| 5 | zhTW (Chinese Traditional) |
| 6 | esES (Spanish, Spain) |
| 7 | esMX (Spanish, Mexico) |
| 8 | ruRU (Russian) |

## Version Differences
- **WDB2**: Basic extension of DBC with sparse tables and copy tables
- **WDB5**: Added bit-packing, palletized data, and improved metadata
- **WDB6**: Added support for multiple sections and total field count 