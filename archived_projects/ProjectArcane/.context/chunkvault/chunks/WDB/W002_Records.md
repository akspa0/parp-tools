# W002: WDB Records

## Type
WDB Structure Component

## Source
WDB Format Documentation

## Description
The Records component of WDB files contains the actual cached data retrieved from the game server. Unlike DBC/DB2 records which follow a fixed structure dictated by the file header, WDB records are more variable in nature, with each record specifying its own length. This flexibility allows the cache to store diverse types of server data that may change in structure over time.

## Structure
Each record in a WDB file follows this general structure:

```csharp
struct WDBRecord
{
    /*0x00*/ uint32_t id;           // Record identifier (typically matches a DBC/DB2 entry ID)
    /*0x04*/ uint32_t length;       // Length of the record data in bytes (excluding id and length fields)
    /*0x08*/ byte[]   data;         // Variable-length record data (size determined by length field)
};
```

Records are stored sequentially in the file, with each record immediately following the previous one. The end of the record section is typically indicated by the file's EOF marker (8 null bytes).

## Properties
| Name | Type | Description |
|------|------|-------------|
| id | uint32_t | Identifier for the record, typically corresponding to an entry ID in a DBC/DB2 file |
| length | uint32_t | Length of the record data in bytes, excluding the id and length fields themselves |
| data | byte[] | Variable-length byte array containing the actual record data |

## Record ID
The record ID serves as the primary key for the cached data and typically corresponds to:
- Item IDs for Item-cache.wdb
- Creature IDs for CreatureCache.wdb
- GameObject IDs for GameObjectCache.wdb
- Quest IDs for QuestCache.wdb
- Achievement IDs for AchievementCache.wdb

This allows the client to quickly find the cached data corresponding to a specific game entity.

## Record Organization
Records in WDB files are not sorted by ID and don't maintain a specific order. New records are typically appended to the end of the file, and the client must scan the entire file to find a specific record. Some implementations maintain an in-memory index of IDs to record positions for faster lookup.

## Data Format
The data portion of each record varies based on the specific cache type:

1. **Structure-based data**: Most cache records contain a structured set of fields specific to the entity type, mirroring the corresponding server database structure.

2. **String data**: Some fields within records contain strings, which are embedded directly in the record data rather than using a string block like DBC/DB2.

3. **Binary data**: Some records may contain binary data like textures, model information, or other complex data types.

## Implementation Notes
- The client validates record lengths when loading cache data
- Records with incorrect lengths are typically discarded as corrupted
- The client must know the specific structure of each cache type to interpret the data
- Unlike DBC/DB2, there's no string block - strings are embedded in the records
- Records are often merged with data from DBC/DB2 files to create the complete game object
- Later cached versions of a record typically override earlier versions
- The cache may contain records that don't exist in the client's DBC/DB2 files

## Implementation Example
```csharp
public class WDBRecord
{
    public uint Id { get; set; }
    public uint Length { get; set; }
    public byte[] Data { get; set; }
    
    public void Parse(BinaryReader reader)
    {
        Id = reader.ReadUInt32();
        Length = reader.ReadUInt32();
        
        if (Length > 0)
            Data = reader.ReadBytes((int)Length);
        else
            Data = new byte[0];
    }
    
    public void Write(BinaryWriter writer)
    {
        writer.Write(Id);
        writer.Write(Length);
        
        if (Data != null && Data.Length > 0)
            writer.Write(Data);
    }
    
    // Helper method to check if record appears valid
    public bool IsValid()
    {
        return Length == (Data?.Length ?? 0);
    }
}

public class WDBFile
{
    public WDBHeader Header { get; set; } = new WDBHeader();
    public Dictionary<uint, WDBRecord> Records { get; set; } = new Dictionary<uint, WDBRecord>();
    
    public bool LoadFromFile(string filePath)
    {
        try
        {
            using (FileStream fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            using (BinaryReader reader = new BinaryReader(fileStream))
            {
                // Parse header
                Header.Parse(reader);
                
                // Parse records until EOF
                while (reader.BaseStream.Position < reader.BaseStream.Length - 8) // 8 bytes for EOF marker
                {
                    var record = new WDBRecord();
                    record.Parse(reader);
                    
                    if (record.IsValid())
                    {
                        // If the record ID already exists, newer records override older ones
                        Records[record.Id] = record;
                    }
                    else
                    {
                        Console.WriteLine($"Invalid record with ID {record.Id}: " +
                                          $"Expected length {record.Length}, got {record.Data?.Length ?? 0}");
                        // Corrupted record; break parsing or try to recover
                        break;
                    }
                }
                
                return true;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading WDB file: {ex.Message}");
            return false;
        }
    }
    
    public bool SaveToFile(string filePath)
    {
        try
        {
            using (FileStream fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write))
            using (BinaryWriter writer = new BinaryWriter(fileStream))
            {
                // Write header
                Header.Write(writer);
                
                // Write records
                foreach (var record in Records.Values)
                {
                    record.Write(writer);
                }
                
                // Write EOF marker (8 null bytes)
                for (int i = 0; i < 8; i++)
                    writer.Write((byte)0);
                
                return true;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error saving WDB file: {ex.Message}");
            return false;
        }
    }
    
    // Helper method to get a record by ID
    public WDBRecord GetRecord(uint id)
    {
        if (Records.TryGetValue(id, out WDBRecord record))
            return record;
        return null;
    }
    
    // Helper method to add or update a record
    public void AddOrUpdateRecord(uint id, byte[] data)
    {
        var record = new WDBRecord
        {
            Id = id,
            Length = (uint)(data?.Length ?? 0),
            Data = data
        };
        
        Records[id] = record;
    }
}
```

## Usage Example
```csharp
// Example of parsing an item cache file and extracting item information
public class ItemCache
{
    private WDBFile _cacheFile = new WDBFile();
    
    public bool LoadItemCache(string filePath)
    {
        return _cacheFile.LoadFromFile(filePath);
    }
    
    // Get a specific item from the cache
    public ItemInfo GetItem(uint itemId)
    {
        var record = _cacheFile.GetRecord(itemId);
        if (record == null)
            return null;
        
        // Parse the item data from the record bytes
        return ParseItemInfo(record.Data);
    }
    
    // Parse item information from record bytes
    private ItemInfo ParseItemInfo(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            var item = new ItemInfo();
            
            // Example parsing - actual format depends on client version
            item.Quality = reader.ReadInt32();
            item.ItemLevel = reader.ReadInt32();
            
            // Read strings - these are typically null-terminated
            item.Name = ReadCString(reader);
            item.Description = ReadCString(reader);
            
            // Read remaining fields...
            
            return item;
        }
    }
    
    // Helper method to read null-terminated strings
    private string ReadCString(BinaryReader reader)
    {
        List<byte> bytes = new List<byte>();
        byte b;
        while ((b = reader.ReadByte()) != 0)
            bytes.Add(b);
        
        return Encoding.UTF8.GetString(bytes.ToArray());
    }
    
    // Example item info class
    public class ItemInfo
    {
        public int Quality { get; set; }
        public int ItemLevel { get; set; }
        public string Name { get; set; }
        public string Description { get; set; }
        // Other item properties...
    }
}
```

## Common Data Types in WDB Records
The specific structure of the data portion varies by cache type, but commonly includes:

1. **Primitive Types**:
   - Integer values (8, 16, 32, and 64-bit)
   - Floating-point values
   - Boolean values

2. **String Types**:
   - Null-terminated strings (CString)
   - Length-prefixed strings

3. **Complex Types**:
   - Lists of primitive values
   - Nested structures
   - Coordinate data (2D and 3D vectors)
   - Color values (RGB/RGBA)

## Record Data Interpretation
Unlike DBC/DB2 files, which have a fixed schema, the interpretation of WDB record data depends on:

1. The specific cache file type (determined by filename)
2. The client version that created the cache
3. The type of entity being cached

The client has internal knowledge of how to interpret each type of cache data based on its structure.

## Relationship to Other Components
- **WDB Header**: Precedes the record section and provides version information
- **EOF Marker**: Follows the last record to indicate the end of the file

## Validation Requirements
- Each record's data length must match its length field
- Records with invalid lengths should be treated as corrupted
- The record section must be followed by the EOF marker (8 null bytes)
- Record IDs should be valid identifiers for the appropriate entity type 