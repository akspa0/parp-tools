# W003: ADB Header

## Type
ADB Structure Component

## Source
ADB Format Documentation

## Description
The ADB Header is the enhanced header structure used in ADB (Advanced DataBase) cache files, introduced in the Cataclysm expansion. This header extends the functionality of the WDB header by adding additional metadata fields including table identification, timestamps, and record counts. These improvements allow for more robust cache management and validation compared to the simpler WDB format.

## Structure
The ADB header has an extended structure compared to WDB:

```csharp
struct ADBHeader
{
    /*0x00*/ uint32_t signature;    // 'ADBC' signature (0x43424441)
    /*0x04*/ uint32_t build;        // Client build number
    /*0x08*/ uint32_t locale;       // Client locale
    /*0x0C*/ uint32_t tableHash;    // Hash of the table name
    /*0x10*/ uint32_t timestamp;    // Last update timestamp
    /*0x14*/ uint32_t recordCount;  // Number of records in the file
    /*0x18*/ uint32_t maxId;        // Highest ID in the cache
    /*0x1C*/ uint32_t minId;        // Lowest ID in the cache
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| signature | uint32_t | Magic number identifying the file as an ADB file, always 'ADBC' (0x43424441) |
| build | uint32_t | The client build number when the cache was created |
| locale | uint32_t | Locale ID identifying the language version of the client |
| tableHash | uint32_t | Hash value identifying the specific data table this cache represents |
| timestamp | uint32_t | UNIX timestamp indicating when the cache was last updated |
| recordCount | uint32_t | The number of records stored in this cache file |
| maxId | uint32_t | The highest ID value found among the cached records |
| minId | uint32_t | The lowest ID value found among the cached records |

## Signature Value
The signature field contains the value 'ADBC' stored as a little-endian 32-bit integer (0x43424441), distinguishing it from the 'WDBC' signature used in WDB files.

## Header Size
The ADB header is exactly 32 bytes (0x20) in size, twice the size of the WDB header.

## Table Hash Calculation
The tableHash field is typically a CRC32 or similar hash of the table name (e.g., "Item", "Creature", "GameObject"). This allows the client to verify that the cache file contains the expected type of data without relying solely on the filename.

## Implementation Notes
- The ADB header provides more robust cache validation than WDB
- The timestamp field enables time-based invalidation policies
- The recordCount field allows for pre-allocation of memory when loading
- The min/max ID fields help with sparse collections and memory optimization
- ADB caches are typically found in the same location as WDB caches in the client's Cache directory
- Different ADB files may have additional extension fields beyond the basic header shown above

## Implementation Example
```csharp
public class ADBHeader
{
    public const uint SIGNATURE = 0x43424441; // 'ADBC' in ASCII (little-endian)
    
    public uint Signature { get; set; }
    public uint Build { get; set; }
    public uint Locale { get; set; }
    public uint TableHash { get; set; }
    public uint Timestamp { get; set; }
    public uint RecordCount { get; set; }
    public uint MaxId { get; set; }
    public uint MinId { get; set; }
    
    public ADBHeader()
    {
        Signature = SIGNATURE;
        Timestamp = (uint)DateTimeOffset.UtcNow.ToUnixTimeSeconds();
    }
    
    public void Parse(BinaryReader reader)
    {
        Signature = reader.ReadUInt32();
        
        if (Signature != SIGNATURE)
            throw new InvalidDataException($"Invalid ADB file: expected ADBC signature, got {Signature:X}");
        
        Build = reader.ReadUInt32();
        Locale = reader.ReadUInt32();
        TableHash = reader.ReadUInt32();
        Timestamp = reader.ReadUInt32();
        RecordCount = reader.ReadUInt32();
        MaxId = reader.ReadUInt32();
        MinId = reader.ReadUInt32();
    }
    
    public void Write(BinaryWriter writer)
    {
        writer.Write(Signature);
        writer.Write(Build);
        writer.Write(Locale);
        writer.Write(TableHash);
        writer.Write(Timestamp);
        writer.Write(RecordCount);
        writer.Write(MaxId);
        writer.Write(MinId);
    }
    
    // Helper method to check if cache matches current client build
    public bool MatchesClientBuild(uint clientBuild)
    {
        return Build == clientBuild;
    }
    
    // Helper method to check if cache matches current client locale
    public bool MatchesClientLocale(uint clientLocale)
    {
        return Locale == clientLocale;
    }
    
    // Helper method to check if cache is for the expected table
    public bool MatchesTableHash(uint expectedTableHash)
    {
        return TableHash == expectedTableHash;
    }
    
    // Helper method to check if cache is older than specified time
    public bool IsOlderThan(TimeSpan maxAge)
    {
        var cacheTime = DateTimeOffset.FromUnixTimeSeconds(Timestamp);
        var currentTime = DateTimeOffset.UtcNow;
        
        return (currentTime - cacheTime) > maxAge;
    }
    
    // Helper method to calculate a table hash (simple example)
    public static uint CalculateTableHash(string tableName)
    {
        // Using a simple FNV-1a hash for example purposes
        const uint FNV_PRIME = 16777619;
        const uint FNV_OFFSET_BASIS = 2166136261;
        
        uint hash = FNV_OFFSET_BASIS;
        foreach (byte b in Encoding.ASCII.GetBytes(tableName))
        {
            hash ^= b;
            hash *= FNV_PRIME;
        }
        
        return hash;
    }
}
```

## Usage Example
```csharp
// Reading an ADB cache file with validation
public bool LoadADBCache(string filePath, uint currentBuild, uint currentLocale, string tableName)
{
    try
    {
        using (FileStream fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
        using (BinaryReader reader = new BinaryReader(fileStream))
        {
            // Parse the header
            var header = new ADBHeader();
            header.Parse(reader);
            
            // Validate the cache against current client
            if (!header.MatchesClientBuild(currentBuild))
            {
                Console.WriteLine($"Cache build {header.Build} doesn't match client build {currentBuild}");
                return false;
            }
            
            if (!header.MatchesClientLocale(currentLocale))
            {
                Console.WriteLine($"Cache locale {header.Locale} doesn't match client locale {currentLocale}");
                return false;
            }
            
            // Calculate expected table hash
            uint expectedTableHash = ADBHeader.CalculateTableHash(tableName);
            if (!header.MatchesTableHash(expectedTableHash))
            {
                Console.WriteLine($"Cache table hash {header.TableHash:X} doesn't match expected hash {expectedTableHash:X}");
                return false;
            }
            
            // Check if cache is too old (more than 1 day old)
            if (header.IsOlderThan(TimeSpan.FromDays(1)))
            {
                Console.WriteLine("Cache is older than 1 day, consider refreshing");
                // May still proceed but note that data might be stale
            }
            
            // Continue parsing records...
            Console.WriteLine($"Found {header.RecordCount} records in cache (ID range: {header.MinId}-{header.MaxId})");
            return true;
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error loading ADB cache: {ex.Message}");
        return false;
    }
}
```

## Timestamp Usage
The timestamp field serves several important purposes:

1. **Cache invalidation**: The client can determine if a cache is too old and should be refreshed
2. **Conflict resolution**: When multiple cache files exist, the most recent can be preferred
3. **Cache analytics**: The client can track how frequently specific caches are updated
4. **Debugging**: Helps developers identify when cached data was last updated

The timestamp is typically stored as a UNIX timestamp (seconds since January 1, 1970, UTC).

## Differences from WDB Header
The ADB header provides several advantages over the WDB header:

1. **Table Identification**:
   - WDB: Relies solely on filename to identify data type
   - ADB: Uses tableHash for explicit type identification

2. **Record Metadata**:
   - WDB: No information about record count or ID range
   - ADB: Provides recordCount, minId, maxId for better memory management

3. **Time Awareness**:
   - WDB: No timestamp information
   - ADB: Includes timestamp for time-based invalidation

4. **Signature**:
   - WDB: Uses 'WDBC' signature
   - ADB: Uses 'ADBC' signature

## Relationship to Other Components
- **ADB Records**: The header precedes the record data and provides information about the records
- **EOF Marker**: The file typically ends with the same 8-byte null EOF marker as WDB files

## Validation Requirements
- The signature field must be 'ADBC' (0x43424441)
- The build and locale fields should match the current client
- The tableHash should match the expected value for the specific cache type
- The timestamp should be a valid UNIX timestamp
- The recordCount should match the actual number of records in the file
- The minId and maxId should accurately reflect the ID range of the records 