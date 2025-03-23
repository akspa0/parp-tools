# WDB and ADB Cache Formats Documentation

## Overview
WDB (World Database) and ADB (Advanced Database) are client-side cache formats used by the World of Warcraft client to store dynamic data retrieved from game servers. Unlike DBC/DB2 files which contain static game data shipped with the client, WDB/ADB files store data that is dynamically generated or served during gameplay, such as item stats, creature information, and quest details.

These formats allow the client to:
1. Reduce server requests by caching frequently accessed data
2. Improve performance by storing previously retrieved information locally
3. Provide offline access to previously encountered game entities
4. Reduce bandwidth usage by only requesting data unknown to the client

## File Structure
WDB and ADB files follow this general structure:

1. **Header**: Contains format identification, version information, and metadata
   - WDB headers are simpler and include a signature, build number, and locale
   - ADB headers add table hash, timestamp, and record count information

2. **Data Records**: Variable-length records containing cached entity data
   - Each record includes an ID, length field, and variable data
   - ADB adds per-record timestamps for more granular cache invalidation

3. **EOF Marker**: 8 null bytes indicating the end of the file
   - Identical in both WDB and ADB formats
   - Used to validate file completeness and detect truncation

## Format Versions
| Format | Expansion | Description |
|--------|-----------|-------------|
| WDB    | Classic - Wrath | Original cache format with basic header and variable-length records |
| ADB    | Cataclysm - Current | Enhanced cache format with additional metadata and timestamps |

## WDB Header Structure
The WDB header is a simple fixed-size structure at the beginning of every WDB file:

```csharp
struct WDBHeader
{
    /*0x00*/ uint32_t signature;    // 'WDBC' magic identifier
    /*0x04*/ uint32_t build;        // Client build number
    /*0x08*/ uint32_t locale;       // Client locale ID
    /*0x0C*/ uint32_t unknownA;     // Unknown, possibly flags or version
};
```

## ADB Header Structure
The ADB header extends the WDB header with additional metadata fields:

```csharp
struct ADBHeader
{
    /*0x00*/ uint32_t signature;    // 'ADBC' magic identifier
    /*0x04*/ uint32_t build;        // Client build number
    /*0x08*/ uint32_t locale;       // Client locale ID
    /*0x0C*/ uint32_t tableHash;    // CRC32 hash of the table name
    /*0x10*/ uint32_t timestamp;    // UNIX timestamp when the cache was created/updated
    /*0x14*/ uint32_t recordCount;  // Number of records in the file
    /*0x18*/ uint32_t maxId;        // Highest record ID in the file
    /*0x1C*/ uint32_t minId;        // Lowest record ID in the file
};
```

## Comparison with DBC/DB2 Formats
| Aspect | WDB/ADB | DBC/DB2 |
|--------|---------|---------|
| **Purpose** | Client-side cache for dynamic data | Static game data shipped with client |
| **Source** | Game server during gameplay | Client installation files |
| **Lifespan** | Temporary, subject to invalidation | Persistent, versioned with client |
| **Structure** | Variable-length records with IDs | Fixed-length records in tabular format |
| **String Storage** | Inline within record data | Separate string block |
| **Changes** | May be modified during gameplay | Read-only, replaced during patches |
| **Completeness** | May contain only subset of data | Contains all defined entries |
| **Localization** | Typically in client's locale only | Contains data for all supported locales |

## Common Cache Types
| Filename | Content | Description |
|----------|---------|-------------|
| Item-cache.wdb/adb | Item data | Stats, requirements, and display information for items the player has seen |
| CreatureCache.wdb/adb | NPC data | Information about creatures the player has encountered |
| GameObjectCache.wdb/adb | Object data | Details about game objects like chests, doors, and interactive items |
| QuestCache.wdb/adb | Quest data | Quest text, objectives, and rewards for quests the player has seen |
| NameCache.wdb/adb | Name data | Player, NPC, and object names the client has encountered |
| PageTextCache.wdb/adb | Text data | Book content, sign text, and other readable text in the game |
| AchievementCache.wdb/adb | Achievement data | Achievement criteria, rewards, and descriptions |

## Implementation Status
| Component | Status | File |
|-----------|--------|------|
| WDB Header | Completed | [W001_Header.md](chunks/WDB/W001_Header.md) |
| WDB Records | Completed | [W002_Records.md](chunks/WDB/W002_Records.md) |
| ADB Header | Completed | [W003_ADBHeader.md](chunks/WDB/W003_ADBHeader.md) |
| ADB Records | Completed | [W004_ADBRecords.md](chunks/WDB/W004_ADBRecords.md) |
| EOF Marker | Completed | [W005_EOFMarker.md](chunks/WDB/W005_EOFMarker.md) |

## Next Steps
1. ~~Document the WDB header structure~~ ✅
2. ~~Document the WDB record structure~~ ✅
3. ~~Document the ADB header improvements~~ ✅
4. ~~Document the ADB record structure enhancements~~ ✅
5. ~~Document the EOF marker structure~~ ✅
6. Implement parsing code for WDB and ADB formats
7. Create utility classes for managing cache files
8. Develop cache validation and repair tools

## Special Implementation Considerations
1. **Cache Integrity**: WDB/ADB files may be corrupted or truncated during crashes or improper client shutdowns
2. **Cache Invalidation**: 
   - WDB files are typically invalidated by client build or locale changes
   - ADB files can be invalidated by timestamps, client builds, or server signals
3. **Partial Data**: Cache files may only contain a subset of the total game data
4. **Cache Rebuilding**: Clients may need to rebuild caches after patches or database updates
5. **Versioning**: Different expansions may use different cache formats or structures 