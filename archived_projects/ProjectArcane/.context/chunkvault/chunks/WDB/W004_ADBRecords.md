# W004: ADB Records

## Type
ADB Structure Component

## Source
ADB Format Documentation

## Description
The Records component of ADB files stores the actual cached data retrieved from the game server. Like WDB records, ADB records store variable-length data with their own size specification, but they use an enhanced structure with additional metadata. ADB records also have stronger validation mechanisms and can include more complex data structures compared to their WDB counterparts.

## Structure
Each record in an ADB file follows this general structure:

```csharp
struct ADBRecord
{
    /*0x00*/ uint32_t id;           // Record identifier (typically matches a DB2 entry ID)
    /*0x04*/ uint32_t length;       // Length of the record data in bytes (excluding id and length fields)
    /*0x08*/ uint32_t timestamp;    // Timestamp when this record was cached
    /*0x0C*/ byte[]   data;         // Variable-length record data (size determined by length field)
};
```

Records are stored sequentially in the file, with each record immediately following the previous one, similar to WDB files. The end of the record section is typically indicated by the file's EOF marker (8 null bytes).

## Properties
| Name | Type | Description |
|------|------|-------------|
| id | uint32_t | Identifier for the record, typically corresponding to an entry ID in a DB2 file |
| length | uint32_t | Length of the record data in bytes, excluding the id, length, and timestamp fields |
| timestamp | uint32_t | UNIX timestamp indicating when this specific record was cached |
| data | byte[] | Variable-length byte array containing the actual record data |

## Record ID
As with WDB, the record ID serves as the primary key for the cached data and typically corresponds to:
- Item IDs for Item-cache.adb
- Creature IDs for CreatureCache.adb
- GameObject IDs for GameObjectCache.adb
- Quest IDs for QuestCache.adb
- Achievement IDs for AchievementCache.adb

## Record Timestamp
The per-record timestamp is a key enhancement over WDB, allowing:
1. Individual record invalidation based on age
2. Tracking of when specific game entities were last updated
3. Detection of stale data at the record level (rather than file level)
4. More granular cache management

## Record Organization
ADB files may organize records more efficiently than WDB:
1. Records are still appended sequentially but may be more likely to be sorted by ID
2. The header's minId/maxId fields help with memory preallocation and lookup optimization
3. The recordCount field in the header helps validate file integrity

## Data Format
The data portion of each record follows similar conventions to WDB:

1. **Structure-based data**: Most cache records contain a structured set of fields specific to the entity type, mirroring the corresponding server database structure.

2. **String data**: Strings are embedded directly in the record data rather than using a string block.

3. **Binary data**: Some records may contain binary data like textures, model information, or other complex data types.

4. **Compressed data**: Some ADB implementations may use compression for the data portion to save space.

## Implementation Notes
- ADB records provide more validation points than WDB records (timestamps, consistent header metadata)
- The client validates record lengths and timestamps when loading cache data
- The ADB format is generally more robust against corruption than WDB
- Record access is more efficient due to header metadata about ID ranges and record counts
- Like WDB, ADB records are often merged with data from DB2 files to create the complete game object
- Individual records can be invalidated based on their timestamps without affecting the entire cache

## Implementation Example
```csharp
public class ADBRecord
{
    public uint Id { get; set; }
    public uint Length { get; set; }
    public uint Timestamp { get; set; }
    public byte[] Data { get; set; }
    
    public void Parse(BinaryReader reader)
    {
        Id = reader.ReadUInt32();
        Length = reader.ReadUInt32();
        Timestamp = reader.ReadUInt32();
        
        if (Length > 0)
            Data = reader.ReadBytes((int)Length);
        else
            Data = new byte[0];
    }
    
    public void Write(BinaryWriter writer)
    {
        writer.Write(Id);
        writer.Write(Length);
        writer.Write(Timestamp);
        
        if (Data != null && Data.Length > 0)
            writer.Write(Data);
    }
    
    // Helper method to check if record appears valid
    public bool IsValid()
    {
        return Length == (Data?.Length ?? 0);
    }
    
    // Helper method to check if record is older than specified time
    public bool IsOlderThan(TimeSpan maxAge)
    {
        var recordTime = DateTimeOffset.FromUnixTimeSeconds(Timestamp);
        var currentTime = DateTimeOffset.UtcNow;
        
        return (currentTime - recordTime) > maxAge;
    }
    
    // Helper method to update timestamp to current time
    public void UpdateTimestamp()
    {
        Timestamp = (uint)DateTimeOffset.UtcNow.ToUnixTimeSeconds();
    }
}

public class ADBFile
{
    public ADBHeader Header { get; set; } = new ADBHeader();
    public Dictionary<uint, ADBRecord> Records { get; set; } = new Dictionary<uint, ADBRecord>();
    
    public bool LoadFromFile(string filePath)
    {
        try
        {
            using (FileStream fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            using (BinaryReader reader = new BinaryReader(fileStream))
            {
                // Parse header
                Header.Parse(reader);
                
                // Pre-allocate dictionary based on header info (optimization)
                Records = new Dictionary<uint, ADBRecord>((int)Header.RecordCount);
                
                // Parse records until EOF
                uint recordsRead = 0;
                while (reader.BaseStream.Position < reader.BaseStream.Length - 8) // 8 bytes for EOF marker
                {
                    var record = new ADBRecord();
                    record.Parse(reader);
                    
                    if (record.IsValid())
                    {
                        Records[record.Id] = record;
                        recordsRead++;
                    }
                    else
                    {
                        Console.WriteLine($"Invalid record with ID {record.Id}: " +
                                          $"Expected length {record.Length}, got {record.Data?.Length ?? 0}");
                        // Corrupted record; break parsing or try to recover
                        break;
                    }
                }
                
                // Verify record count matches header
                if (recordsRead != Header.RecordCount)
                {
                    Console.WriteLine($"Warning: Header indicates {Header.RecordCount} records, " +
                                     $"but found {recordsRead} records");
                    // May still proceed, but note the discrepancy
                }
                
                return true;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading ADB file: {ex.Message}");
            return false;
        }
    }
    
    public bool SaveToFile(string filePath)
    {
        try
        {
            // Update header information to match current records
            Header.RecordCount = (uint)Records.Count;
            Header.Timestamp = (uint)DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            
            if (Records.Count > 0)
            {
                Header.MinId = Records.Keys.Min();
                Header.MaxId = Records.Keys.Max();
            }
            else
            {
                Header.MinId = 0;
                Header.MaxId = 0;
            }
            
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
            Console.WriteLine($"Error saving ADB file: {ex.Message}");
            return false;
        }
    }
    
    // Helper method to get a record by ID
    public ADBRecord GetRecord(uint id)
    {
        if (Records.TryGetValue(id, out ADBRecord record))
            return record;
        return null;
    }
    
    // Helper method to add or update a record
    public void AddOrUpdateRecord(uint id, byte[] data)
    {
        var record = new ADBRecord
        {
            Id = id,
            Length = (uint)(data?.Length ?? 0),
            Timestamp = (uint)DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            Data = data
        };
        
        Records[id] = record;
        
        // Update header min/max ID range
        if (Records.Count == 1)
        {
            Header.MinId = id;
            Header.MaxId = id;
        }
        else
        {
            Header.MinId = Math.Min(Header.MinId, id);
            Header.MaxId = Math.Max(Header.MaxId, id);
        }
        
        Header.RecordCount = (uint)Records.Count;
    }
    
    // Helper method to remove stale records
    public int RemoveStaleRecords(TimeSpan maxAge)
    {
        var staleIds = Records.Values
            .Where(r => r.IsOlderThan(maxAge))
            .Select(r => r.Id)
            .ToList();
        
        foreach (var id in staleIds)
        {
            Records.Remove(id);
        }
        
        // Update header if records were removed
        if (staleIds.Count > 0)
        {
            Header.RecordCount = (uint)Records.Count;
            
            if (Records.Count > 0)
            {
                Header.MinId = Records.Keys.Min();
                Header.MaxId = Records.Keys.Max();
            }
            else
            {
                Header.MinId = 0;
                Header.MaxId = 0;
            }
        }
        
        return staleIds.Count;
    }
}
```

## Usage Example
```csharp
// Example of parsing a creature cache file and extracting creature information
public class CreatureCache
{
    private ADBFile _cacheFile = new ADBFile();
    
    public bool LoadCreatureCache(string filePath, uint currentBuild, uint currentLocale)
    {
        bool success = _cacheFile.LoadFromFile(filePath);
        
        if (success)
        {
            // Validate cache against current client
            if (!_cacheFile.Header.MatchesClientBuild(currentBuild) ||
                !_cacheFile.Header.MatchesClientLocale(currentLocale))
            {
                Console.WriteLine("Cache file does not match current client build/locale");
                return false;
            }
            
            // Automatically remove records older than 7 days
            int removedCount = _cacheFile.RemoveStaleRecords(TimeSpan.FromDays(7));
            if (removedCount > 0)
            {
                Console.WriteLine($"Removed {removedCount} stale creature records");
                // Optionally save the pruned cache back to disk
                _cacheFile.SaveToFile(filePath);
            }
        }
        
        return success;
    }
    
    // Get a specific creature from the cache
    public CreatureInfo GetCreature(uint creatureId)
    {
        var record = _cacheFile.GetRecord(creatureId);
        if (record == null)
            return null;
        
        // Check if record is too old (more than 1 day)
        if (record.IsOlderThan(TimeSpan.FromDays(1)))
        {
            Console.WriteLine($"Warning: Creature {creatureId} data is more than 1 day old");
            // Might still use it, but note that it could be stale
        }
        
        // Parse the creature data from the record bytes
        return ParseCreatureInfo(record.Data);
    }
    
    // Parse creature information from record bytes
    private CreatureInfo ParseCreatureInfo(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            var creature = new CreatureInfo();
            
            // Example parsing - actual format depends on client version
            creature.Level = reader.ReadInt32();
            creature.Health = reader.ReadInt32();
            creature.Faction = reader.ReadInt32();
            
            // Read strings - these are typically null-terminated
            creature.Name = ReadCString(reader);
            creature.Subname = ReadCString(reader);
            
            // Read remaining fields...
            
            return creature;
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
    
    // Example creature info class
    public class CreatureInfo
    {
        public int Level { get; set; }
        public int Health { get; set; }
        public int Faction { get; set; }
        public string Name { get; set; }
        public string Subname { get; set; }
        // Other creature properties...
    }
}
```

## Differences from WDB Records
ADB records have several improvements over WDB records:

1. **Timestamp Tracking**:
   - WDB: No per-record timestamp
   - ADB: Each record has its own timestamp for fine-grained invalidation

2. **Validation**:
   - WDB: Limited validation through record length
   - ADB: Multiple validation points including header metadata and timestamps

3. **Access Optimization**:
   - WDB: No metadata to assist with record lookup
   - ADB: Header contains recordCount, minId, maxId to optimize memory allocation and lookup

4. **Record Management**:
   - WDB: Typically all-or-nothing cache invalidation
   - ADB: Supports selective record invalidation based on timestamps

## Common Data Patterns
ADB records often follow specific patterns for different entity types:

1. **Items**: Include stats, quality, name, description, requirements, and icon references
2. **Creatures**: Include level, health, faction, name, subname, and display info
3. **GameObjects**: Include display ID, type, name, and interaction flags
4. **Quests**: Include title, description, objectives, rewards, and requirements
5. **Achievements**: Include name, description, criteria, points, and rewards

## Relationship to Other Components
- **ADB Header**: Precedes the record section and provides context for the records
- **EOF Marker**: Follows the last record to indicate the end of the file (8 null bytes)

## Validation Requirements
- Each record's data length must match its length field
- Record timestamps should be valid UNIX timestamps
- The record count should match the header's recordCount field
- All record IDs should be within the header's minId and maxId range
- Records with invalid data should be discarded during loading
- The record section must be followed by the EOF marker 