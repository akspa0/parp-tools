# W001: WDB Header

## Type
WDB Structure Component

## Source
WDB Format Documentation

## Description
The WDB Header is the fixed-size structure at the beginning of WDB (Warcraft DataBase) cache files that defines the format, version, and locale of the cached data. It provides essential metadata used by the client to validate and interpret the cached information. The header is relatively simple compared to DBC/DB2 headers, reflecting the more flexible nature of cache data.

## Structure
The WDB header has a simple structure with four 32-bit fields:

```csharp
struct WDBHeader
{
    /*0x00*/ uint32_t signature;    // 'WDBC' signature (0x43424457)
    /*0x04*/ uint32_t build;        // Client build number
    /*0x08*/ uint32_t locale;       // Client locale
    /*0x0C*/ uint32_t unknownA;     // Unknown value, possibly version or flags
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| signature | uint32_t | Magic number identifying the file as a WDB file, always 'WDBC' (0x43424457) |
| build | uint32_t | The client build number when the cache was created |
| locale | uint32_t | Locale ID identifying the language version of the client |
| unknownA | uint32_t | Unknown value, possibly a version identifier or flags field |

## Signature Value
The signature field contains the value 'WDBC' stored as a little-endian 32-bit integer (0x43424457). This is the same signature used in DBC files but serves a different purpose in WDB files.

## Header Size
The WDB header is exactly 16 bytes (0x10) in size.

## Locale Values
The locale field uses the same values as in DBC files:

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

## Implementation Notes
- The WDB header is used to validate that the cache file was created by the same client version/locale
- When the client version changes, WDB caches are typically invalidated and rebuilt
- The header alone doesn't indicate what type of data is stored in the cache (item data, creature data, etc.) 
- File type is typically determined by the filename (e.g., Item-cache.wdb)
- The unknownA field's purpose is not fully documented but may relate to cache versioning or flags

## Implementation Example
```csharp
public class WDBHeader
{
    public const uint SIGNATURE = 0x43424457; // 'WDBC' in ASCII (little-endian)
    
    public uint Signature { get; set; }
    public uint Build { get; set; }
    public uint Locale { get; set; }
    public uint UnknownA { get; set; }
    
    public WDBHeader()
    {
        Signature = SIGNATURE;
    }
    
    public void Parse(BinaryReader reader)
    {
        Signature = reader.ReadUInt32();
        
        if (Signature != SIGNATURE)
            throw new InvalidDataException($"Invalid WDB file: expected WDBC signature, got {Signature:X}");
        
        Build = reader.ReadUInt32();
        Locale = reader.ReadUInt32();
        UnknownA = reader.ReadUInt32();
    }
    
    public void Write(BinaryWriter writer)
    {
        writer.Write(Signature);
        writer.Write(Build);
        writer.Write(Locale);
        writer.Write(UnknownA);
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
}
```

## Usage Example
```csharp
// Reading a WDB cache file
public bool LoadWDBCache(string filePath, uint currentBuild, uint currentLocale)
{
    try
    {
        using (FileStream fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
        using (BinaryReader reader = new BinaryReader(fileStream))
        {
            // Parse the header
            var header = new WDBHeader();
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
            
            // Continue parsing records...
            return true;
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error loading WDB cache: {ex.Message}");
        return false;
    }
}
```

## Relationship to Data Records
- The header is immediately followed by the cached data records
- The header doesn't specify the number or size of records (unlike DBC/DB2)
- Records must be parsed sequentially, as there's no table of contents

## Differences from DBC Header
Although the WDB header shares the same signature as the DBC header, there are key differences:

1. **Purpose**: 
   - DBC header: Defines a fixed database table structure
   - WDB header: Identifies a cache file with variable content

2. **Fields**:
   - DBC header: Contains record count, field count, record size, and string block size
   - WDB header: Contains build number, locale, and an unknown field

3. **Validation**:
   - DBC header: Used to parse a known table structure
   - WDB header: Used primarily to validate cache against client version

## Validation Requirements
- The signature field must be 'WDBC' (0x43424457)
- The build field should match the current client build number for the cache to be valid
- The locale field should match the current client locale for the cache to be valid
- The entire header must be exactly 16 bytes in length

## Cache Invalidation
The client typically invalidates and rebuilds cache files in these scenarios:

1. Client build number changes (patches, expansions)
2. Client locale changes
3. Cache corruption is detected
4. Manual deletion of cache files by the user 