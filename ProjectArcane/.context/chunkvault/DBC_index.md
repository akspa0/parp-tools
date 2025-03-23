# Client Database (DBC/DB2) Format Documentation

## Overview
The DBC (DataBase Container) format is used by World of Warcraft to store structured game data in a tabular format, similar to a database. These files contain essential game information including items, spells, creatures, quests, achievements, and more. The client uses this data to function without requiring constant server communication for basic game data.

In later expansions (starting with Cataclysm), the DB2 format was introduced as an extension of DBC with additional features and optimizations. Both formats follow similar principles but with different internal structures.

## File Structure
Unlike the chunk-based formats (ADT, WDT, WDL), DBC/DB2 files use a fixed header followed by record-based data. Each DBC file represents a database table with:

1. A header containing format information and record counts
2. A string block containing all string data referenced by the records
3. Record data containing structured fields in tabular format

The file format is designed for efficient random access to records and fast loading at game startup.

## File Versions
| Format | Expansion | Description |
|--------|-----------|-------------|
| DBC | Classic - Wrath | Original database container format |
| DB2 (WDB2) | Cataclysm | Enhanced format with copy tables and sparse tables |
| DB2 (WDB5) | Warlords of Draenor | Added bit-packing, field storage info, and optimization |
| DB2 (WDB6) | Legion | Further optimized with additional storage types and sections |
| WDC1/2/3 | BfA - Dragonflight | Latest formats with more advanced compression techniques |

## DBC Format Structure
The DBC format has the following structure:

### Header
```csharp
struct DBCHeader
{
    /*0x00*/ uint32_t magic;          // 'WDBC' signature
    /*0x04*/ uint32_t recordCount;    // Number of records in the file
    /*0x08*/ uint32_t fieldCount;     // Number of fields per record
    /*0x0C*/ uint32_t recordSize;     // Size of each record in bytes
    /*0x10*/ uint32_t stringBlockSize;// Size of the string block at the end of the file
};
```

### Records
Following the header is a block of records, each having the same structure defined by the DBC's schema. Each record is `recordSize` bytes long and contains `fieldCount` fields.

### String Block
The string block follows all records and contains all strings referenced by any string field in the records. String fields in records contain offsets into this string block, where the actual string data is stored. Each string is null-terminated.

## DB2 Format Structure
The DB2 format expands on DBC with additional features:

### Header
```csharp
struct DB2Header
{
    /*0x00*/ uint32_t magic;          // 'WDB2' signature for v1, 'WDB5'/'WDB6' for v2
    /*0x04*/ uint32_t recordCount;    // Number of records in the file
    /*0x08*/ uint32_t fieldCount;     // Number of fields per record
    /*0x0C*/ uint32_t recordSize;     // Size of each record in bytes
    /*0x10*/ uint32_t stringBlockSize;// Size of the string block
    /*0x14*/ uint32_t tableHash;      // Hash of the table name
    /*0x18*/ uint32_t layoutHash;     // Hash of the layout definition
    /*0x1C*/ uint32_t minId;          // Minimum ID of records in the file
    /*0x20*/ uint32_t maxId;          // Maximum ID of records in the file
    /*0x24*/ uint32_t locale;         // Locale of the file (if localized)
    /*0x28*/ uint32_t copyTableSize;  // Secondary index size
    // Additional fields in later formats
};
```

DB2 adds support for:
- Sparse tables (not all IDs between minId and maxId exist)
- File-based lookups via ID ranges
- Locale-specific data
- Advanced field types and encodings
- Copy data for reusing entries
- Bit-packing and palletized data compression (WDB5+)
- Field storage information for optimized storage (WDB5+)

## Common Database Tables
The most frequently used DBC/DB2 tables include:

| Table Name | Description | Used For |
|------------|-------------|----------|
| Spell.dbc | Spell data | All spell information |
| Item.dbc | Item data | Basic item properties |
| Creature.dbc | Creature data | NPC information |
| Map.dbc | Map data | Information about world maps and instances |
| Quest.dbc | Quest data | Quest templates and requirements |
| Talent.dbc | Talent data | Character talent information |
| Achievement.dbc | Achievement data | Achievement criteria and rewards |

## Relationship to Other Formats
DBC/DB2 files provide the data referenced by other World of Warcraft file formats:

- ADT files reference terrain textures defined in DBC/DB2 tables
- WMO/M2 models use material properties defined in DBC/DB2 tables
- Game mechanics reference spells, items, and other entities defined in DBC/DB2 tables

## Implementation Status
| Component | Status | Documentation |
|-----------|--------|--------------|
| DBC Header | ✅ Documented | [D001_Header.md](chunks/DBC/D001_Header.md) |
| DBC Records | ✅ Documented | [D002_Records.md](chunks/DBC/D002_Records.md) |
| DBC String Block | ✅ Documented | [D003_StringBlock.md](chunks/DBC/D003_StringBlock.md) |
| DB2 Header | ✅ Documented | [D004_DB2Header.md](chunks/DBC/D004_DB2Header.md) |
| Field Storage Info | ✅ Documented | [D005_FieldStorageInfo.md](chunks/DBC/D005_FieldStorageInfo.md) |
| Copy Table | ✅ Documented | [D006_CopyTable.md](chunks/DBC/D006_CopyTable.md) |

**Overall Status:** 6/6 components documented (100% complete)

## Documented Components

### DBC Format Components
1. **DBC Header** - The fixed-size structure at the beginning of DBC files containing metadata about the file's contents.
2. **DBC Records** - The tabular data stored as fixed-size records with a consistent structure.
3. **DBC String Block** - A collection of null-terminated strings referenced by string fields in records.

### DB2 Format Components
4. **DB2 Header** - An extended header structure with additional metadata for optimization.
5. **Field Storage Info** - Metadata describing how fields are stored, compressed, and accessed.
6. **Copy Table** - A mechanism for record data reuse to optimize file size.

## Advanced Feature Documentation
The documentation covers several advanced DBC/DB2 features:

- **String Handling**: How strings are stored and referenced in both formats
- **Bit-Packing**: How fields can use exact bit sizes to optimize storage
- **Palletized Data**: How fields with limited unique values use shared value tables
- **Copy Tables**: How data can be reused between similar records
- **Sparse Tables**: How non-contiguous ID ranges are handled efficiently
- **Localization**: How different language versions are managed in the files

## Special Implementation Considerations
- DBC/DB2 files may be localized, requiring special handling for different game clients
- The schema for each table must be defined externally as it's not stored in the file
- Modern clients may use encrypted or compressed DB2 files
- Some fields use special encoding techniques to save space
- String lookups require efficient indexing into the string block
- Copy table processing requires carefully handling record inheritance

## Next Steps
With the core DBC/DB2 format documentation complete, next steps include:

1. Document specific table schemas for commonly used tables (Spell, Item, Creature)
2. Create a format validator for DBC/DB2 files
3. Develop a comprehensive parser implementation combining all documented components
4. Design a visualization tool for inspecting DBC/DB2 content
5. Document relationships between different tables in the DB format 