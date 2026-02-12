# D006: Copy Table

## Type
DB2 Structure Component

## Source
DB2 Format Documentation

## Description
The Copy Table is a component introduced in the WDB2 format (and continued in later versions) that enables record data reuse. It provides a mechanism for records to reference the data of another record instead of duplicating it, significantly reducing file size when many records share identical or highly similar data. This is particularly useful for items, spells, and other game entities that often have many variants with only minor differences.

## Structure
The Copy Table appears in the file after the record data section and contains entries mapping a record ID to another "parent" record ID:

```csharp
struct CopyTableEntry
{
    /*0x00*/ uint32_t newRecordId; // ID of the record that references another record's data
    /*0x04*/ uint32_t sourceRecordId; // ID of the source record whose data is reused
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| newRecordId | uint32_t | The ID of the record that reuses another record's data |
| sourceRecordId | uint32_t | The ID of the source record whose data is being reused |

## Copy Table Size
The total size of the Copy Table section is given by the `copyTableSize` field in the DB2 header. The number of entries in the copy table can be calculated as:
```
copyTableEntryCount = copyTableSize / sizeof(CopyTableEntry)
```

Where each `CopyTableEntry` is 8 bytes (0x08).

## Implementation Notes
- The Copy Table allows for significant space savings when many records share similar data
- Records referenced by the Copy Table may not have actual data in the file's record section
- When loading a record found in the Copy Table, the client should load the source record's data
- Typically, specific field values in the copied record get overridden after copying
- The source record must appear before any records that copy from it
- Multiple records can reference the same source record

## Record Override Mechanism
When a record copies data from another record via the Copy Table, specific fields can still be overridden:

1. The base record data is copied from the source record
2. Field-specific overrides are applied based on the game's internal logic
3. Some fields (like the record ID) are always overridden
4. Other fields may be overridden based on a field mask or special rules

This mechanism is particularly useful for item variants, spell ranks, and NPC variants that share most properties but differ in a few specific values.

## Implementation Example
```csharp
public class CopyTableEntry
{
    public uint NewRecordId { get; set; }
    public uint SourceRecordId { get; set; }
    
    public void Parse(BinaryReader reader)
    {
        NewRecordId = reader.ReadUInt32();
        SourceRecordId = reader.ReadUInt32();
    }
    
    public void Write(BinaryWriter writer)
    {
        writer.Write(NewRecordId);
        writer.Write(SourceRecordId);
    }
}

public class CopyTable
{
    private Dictionary<uint, uint> _copyMap = new Dictionary<uint, uint>();
    
    public int EntryCount => _copyMap.Count;
    
    // Check if a record ID is in the copy table
    public bool HasCopy(uint recordId)
    {
        return _copyMap.ContainsKey(recordId);
    }
    
    // Get the source record ID for a given record
    public uint GetSourceRecordId(uint recordId)
    {
        return _copyMap.TryGetValue(recordId, out uint sourceId) ? sourceId : recordId;
    }
    
    // Parse the copy table from the file
    public void Parse(BinaryReader reader, uint copyTableSize)
    {
        int entryCount = (int)(copyTableSize / 8); // 8 bytes per entry
        
        for (int i = 0; i < entryCount; i++)
        {
            var entry = new CopyTableEntry();
            entry.Parse(reader);
            _copyMap[entry.NewRecordId] = entry.SourceRecordId;
        }
    }
    
    // Write the copy table to the file
    public void Write(BinaryWriter writer)
    {
        foreach (var pair in _copyMap)
        {
            writer.Write(pair.Key);   // newRecordId
            writer.Write(pair.Value); // sourceRecordId
        }
    }
    
    // Add a copy entry
    public void AddCopyEntry(uint newRecordId, uint sourceRecordId)
    {
        _copyMap[newRecordId] = sourceRecordId;
    }
    
    // Get all entries as a list
    public List<CopyTableEntry> GetEntries()
    {
        return _copyMap.Select(pair => new CopyTableEntry 
        { 
            NewRecordId = pair.Key, 
            SourceRecordId = pair.Value 
        }).ToList();
    }
}
```

## Usage Example
```csharp
// Reading a DB2 file with copy table
public void LoadDB2WithCopyTable(string filePath)
{
    using (FileStream fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
    using (BinaryReader reader = new BinaryReader(fileStream))
    {
        // Parse the header
        var header = new DB2Header();
        header.Parse(reader);
        
        // Calculate positions
        long recordDataPos = header.GetHeaderSize();
        if (header.IsWDB5 || header.IsWDB6)
            recordDataPos += header.FieldStorageInfoSize;
            
        long stringBlockPos = recordDataPos + (header.RecordSize * header.RecordCount);
        long copyTablePos = stringBlockPos + header.StringBlockSize;
        
        // Create dictionary to store records by ID
        Dictionary<uint, byte[]> recordsById = new Dictionary<uint, byte[]>();
        
        // Read the record data
        reader.BaseStream.Position = recordDataPos;
        for (uint i = 0; i < header.RecordCount; i++)
        {
            // Read the record data
            byte[] recordData = reader.ReadBytes((int)header.RecordSize);
            
            // Extract the record ID (assuming first 4 bytes are ID)
            uint recordId = BitConverter.ToUInt32(recordData, 0);
            
            // Store the record data
            recordsById[recordId] = recordData;
        }
        
        // Read the copy table if present
        CopyTable copyTable = new CopyTable();
        if (header.CopyTableSize > 0)
        {
            reader.BaseStream.Position = copyTablePos;
            copyTable.Parse(reader, header.CopyTableSize);
            
            Console.WriteLine($"Read copy table with {copyTable.EntryCount} entries");
            
            // Process the copy table by creating "virtual" records
            foreach (var entry in copyTable.GetEntries())
            {
                if (recordsById.TryGetValue(entry.SourceRecordId, out byte[] sourceData))
                {
                    // Copy the source record data
                    byte[] copyData = new byte[sourceData.Length];
                    Array.Copy(sourceData, copyData, sourceData.Length);
                    
                    // Override the record ID (first 4 bytes)
                    BitConverter.GetBytes(entry.NewRecordId).CopyTo(copyData, 0);
                    
                    // Store the copied record
                    recordsById[entry.NewRecordId] = copyData;
                }
                else
                {
                    Console.WriteLine($"Warning: Source record {entry.SourceRecordId} " +
                                      $"for copy {entry.NewRecordId} not found");
                }
            }
        }
        
        Console.WriteLine($"Total records after copy table processing: {recordsById.Count}");
    }
}
```

## Common Use Cases
The Copy Table is particularly useful for these game data scenarios:

1. **Item Variants**: Items that share most properties but differ in quality, level, or visual appearance
2. **Spell Ranks**: Different ranks of the same spell that share most properties but differ in power
3. **NPC Variants**: Different versions of NPCs that share most properties but differ in level or abilities
4. **Quest Chains**: Related quests that share objectives, rewards, or other properties
5. **Achievement Criteria**: Different criteria for similar achievements

## Space Savings Example
Consider a database with 10,000 items where:
- 7,000 items are unique
- 3,000 items are variants of existing items

Without a copy table:
- All 10,000 items require full record storage
- At 100 bytes per record = 1,000,000 bytes

With a copy table:
- 7,000 unique items require full record storage: 700,000 bytes
- 3,000 variant items require only copy table entries: 24,000 bytes (8 bytes each)
- Total: 724,000 bytes (28% space saving)

## Relationship to Other Components
- **DB2 Header**: Defines the copyTableSize and indicates its presence
- **Record Data**: The Copy Table references records in this section
- **String Block**: Copied records may reference strings in the string block
- **Field Storage Info**: Defines how to interpret record data, including copied records

## Evolution in Later Formats
The concept of the Copy Table has evolved slightly in later formats:

- **WDB2**: Basic record copying with simple ID mapping
- **WDB5/WDB6**: Enhanced with more sophisticated field-specific overrides
- **WDC1/2/3**: Further refined with section-aware copying and additional optimizations

## Validation Requirements
- Copy Table size must match the `copyTableSize` value in the DB2 header
- Source record IDs must refer to valid records that exist in the file
- The copy table should be located after the string block
- Circular references should not exist (a record cannot directly or indirectly reference itself) 