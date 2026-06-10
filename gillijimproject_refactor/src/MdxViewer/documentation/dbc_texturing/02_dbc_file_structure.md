# DBC File Structure

## Overview

DBC (Database Client) files are binary database files used by World of Warcraft to store structured game data. This document details the file format and structure as it relates to texture references.

## Binary Format Specification

### File Header

```c
struct DBCHeader {
    uint32_t magic;          // 'WDBC' (0x43424457)
    uint32_t recordCount;    // Number of records
    uint32_t fieldCount;     // Number of fields per record
    uint32_t recordSize;     // Size of each record in bytes
    uint32_t stringBlockSize; // Size of string block in bytes
};
```

**Total Header Size**: 20 bytes

### Data Layout

```
┌─────────────────────────────┐
│  DBC Header (20 bytes)      │
├─────────────────────────────┤
│  Record 0                   │
│  Record 1                   │
│  ...                        │
│  Record N-1                 │
├─────────────────────────────┤
│  String Block               │
│  (Null-terminated strings)  │
└─────────────────────────────┘
```

## String References

DBC files use offset-based string references:
- Strings stored in a separate block at the end of the file
- Fields contain byte offsets into the string block
- Offset 0 always points to an empty string
- All strings are null-terminated ASCII/UTF-8

### String Resolution

```c
// Given a string offset from a record field
char* GetString(uint32_t offset) {
    return (char*)(stringBlock + offset);
}
```

## Texture-Related DBC Files

### CreatureDisplayInfo.dbc

Stores creature appearance data including model and texture references.

**Structure** (simplified):
```c
struct CreatureDisplayInfoRec {
    uint32_t ID;                    // Record ID
    uint32_t ModelID;               // Reference to CreatureModelData
    uint32_t SoundID;               // Sound reference
    uint32_t ExtraDisplayInfoID;    // Reference to CreatureDisplayInfoExtra
    float    Scale;                 // Model scale
    uint32_t Opacity;               // Alpha value (0-255)
    uint32_t Skin1;                 // String offset: texture path 1
    uint32_t Skin2;                 // String offset: texture path 2
    uint32_t Skin3;                 // String offset: texture path 3
    uint32_t PortraitTextureName;   // String offset: portrait texture
    uint32_t BloodID;               // Blood type
    uint32_t NPCSoundID;            // NPC sound set
    uint32_t ParticleColorID;       // Particle color reference
    uint32_t CreatureGeosetData;    // Geoset visibility flags
    uint32_t ObjectEffectPackageID; // Effect package reference
};
```

**Key Fields for Texturing**:
- `Skin1`, `Skin2`, `Skin3`: Replacement textures for model texture slots
- Paths are relative to the game data directory
- Empty strings (offset 0) mean use default model texture

**Example Texture Paths**:
```
Creature\Murloc\Murloc.blp
Creature\Murloc\MurlocOrange.blp
Creature\Murloc\MurlocGreen.blp
```

### CreatureModelData.dbc

Links creature display info to actual model files.

```c
struct CreatureModelDataRec {
    uint32_t ID;                    // Record ID
    uint32_t Flags;                 // Model flags
    uint32_t ModelPath;             // String offset: path to MDX/M2 file
    uint32_t SizeClass;             // Size category
    float    ModelScale;            // Base scale
    uint32_t BloodID;               // Blood type
    uint32_t FootprintTextureID;    // Footprint texture reference
    float    FootprintTextureLength;
    float    FootprintTextureWidth;
    float    FootprintParticleScale;
    uint32_t FoleyMaterialID;       // Sound material
    uint32_t FootstepShakeSize;
    uint32_t DeathThudShakeSize;
    uint32_t CollisionWidth;
    float    CollisionHeight;
    float    MountHeight;
    float    GeoBoxMin[3];          // Bounding box
    float    GeoBoxMax[3];
    float    WorldEffectScale;
    float    AttachedEffectScale;
};
```

### CharSections.dbc

Character customization textures (skin, face, hair, etc.).

```c
struct CharSectionsRec {
    uint32_t ID;                    // Record ID
    uint32_t RaceID;                // Character race
    uint32_t SexID;                 // 0 = Male, 1 = Female
    uint32_t BaseSection;           // Section type (skin, face, etc.)
    uint32_t VariationIndex;        // Which variation
    uint32_t ColorIndex;            // Color variation
    uint32_t TextureName1;          // String offset: primary texture
    uint32_t TextureName2;          // String offset: secondary texture  
    uint32_t TextureName3;          // String offset: tertiary texture
    uint32_t Flags;                 // Flags for this section
};
```

**Section Types**:
- 0: Base skin
- 1: Face
- 2: Facial hair (lower)
- 3: Facial hair (upper)
- 4: Hair
- 5: Underwear

**Example Paths**:
```
Character\Human\Male\HumanMaleSkin00_00.blp
Character\Human\Male\HumanMaleFaceUpper00_00.blp
Character\Human\Male\HumanMaleFacialHair01_00.blp
```

### ItemDisplayInfo.dbc

Item appearance and texture data.

```c
struct ItemDisplayInfoRec {
    uint32_t ID;                    // Record ID
    uint32_t ModelName1;            // String offset: left-hand model
    uint32_t ModelName2;            // String offset: right-hand model
    uint32_t ModelTexture1;         // String offset: left texture
    uint32_t ModelTexture2;         // String offset: right texture
    uint32_t InventoryIcon1;        // String offset: icon texture
    uint32_t InventoryIcon2;        // String offset: icon texture 2
    uint32_t GeosetGroup1;          // Geoset visibility
    uint32_t GeosetGroup2;
    uint32_t GeosetGroup3;
    uint32_t Flags;                 // Display flags
    uint32_t SpellVisualID;         // Visual effect
    uint32_t GroupSoundIndex;       // Sound group
    uint32_t HelmetGeosetVis1;      // Helmet geoset visibility
    uint32_t HelmetGeosetVis2;
    uint32_t TextureType;           // Texture variation type
    uint32_t ItemVisual;            // Item visual effect
    uint32_t ParticleColorID;       // Particle color
};
```

## DBC Cache System

### Hash-Based Caching

The client uses template hash tables for caching DBC records:

```c
template<typename T, typename KeyType, typename HashKey>
struct DBCache {
    struct DBCACHEHASH {
        KeyType key;
        T* record;
        DBCACHEHASH* next;  // Collision chain
    };
    
    TSHashTable<DBCACHEHASH, HashKey> hashTable;
    TSExplicitList<DBCACHEHASH> recordList;
};
```

**Cached DBC Types**:
- `DBCache<CreatureStats_C, int, HASHKEY_INT>`
- `DBCache<ItemStats_C, int, HASHKEY_INT>`
- `DBCache<QuestCache, int, HASHKEY_INT>`
- `DBCache<NPCText, int, HASHKEY_INT>`

### Cache Lifecycle

```
┌──────────────┐
│ LoadDBCaches │
└──────┬───────┘
       │
       ▼
┌─────────────────────┐
│ DBCache_Initialize  │  ← Allocates hash tables
└──────┬──────────────┘
       │
       ▼
┌─────────────────────────┐
│ DBCache_RegisterHandlers │  ← Sets up callbacks
└──────┬──────────────────┘
       │
       ▼
    [Runtime]
       │
       ▼
┌──────────────────┐
│ DBCache_Destroy  │  ← Cleanup on exit
└──────────────────┘
```

## Reading DBC Records

### Sequential Access

```c
// Iterate through all records
for (uint32_t i = 0; i < header.recordCount; i++) {
    void* record = dataStart + (i * header.recordSize);
    // Process record
}
```

### Index-Based Access

```c
// Access specific record by index
void* GetRecord(uint32_t index) {
    if (index >= header.recordCount) return NULL;
    return dataStart + (index * header.recordSize);
}
```

### Cache Lookup

```c
// Lookup by ID (uses hash table)
Record* FindRecord(int id) {
    uint32_t hash = ComputeHash(id);
    DBCACHEHASH* entry = hashTable[hash];
    
    while (entry) {
        if (entry->key == id) {
            return entry->record;
        }
        entry = entry->next;
    }
    
    return NULL;
}
```

## Texture Path Construction

### Path Components

DBC texture paths are stored in several formats:

1. **Full Relative Path**:
   ```
   Creature\Murloc\Murloc.blp
   ```

2. **Base + Suffix Pattern**:
   ```
   Base: Character\Human\Male\HumanMaleSkin
   Suffix: 00_00.blp
   Result: Character\Human\Male\HumanMaleSkin00_00.blp
   ```

3. **Variation Pattern**:
   ```
   Base: Item\Armor\Leather\Chest_01
   Variants: _RED.blp, _BLUE.blp, _GREEN.blp
   ```

### Path Normalization

The client normalizes all paths:
- Converts forward slashes to backslashes (on Windows)
- Converts to lowercase for case-insensitive comparison
- Removes redundant path separators

```c
void NormalizePath(char* path) {
    // Convert to lowercase
    _strlwr(path);
    
    // Replace / with \
    for (char* p = path; *p; p++) {
        if (*p == '/') *p = '\\';
    }
}
```

## Data Validation

### Integrity Checks

Common validation when loading DBC files:

```c
bool ValidateDBC(DBCHeader* header) {
    // Check magic number
    if (header->magic != 0x43424457) return false;
    
    // Validate record size
    if (header->recordSize < header->fieldCount * 4) return false;
    
    // Check reasonable bounds
    if (header->recordCount == 0 || header->recordCount > 1000000) return false;
    
    return true;
}
```

### String Block Validation

```c
bool ValidateStringOffset(uint32_t offset, uint32_t stringBlockSize) {
    return offset < stringBlockSize;
}
```

## Performance Considerations

### Memory Layout
- Records are fixed-size for fast indexing
- String block allows variable-length data
- Cache-friendly sequential access

### Loading Strategy
- DBC files memory-mapped for fast loading
- String offsets enable lazy string access
- Hash tables provide O(1) lookups after initialization

## Endianness

DBC files use little-endian byte order (Intel x86 native). On big-endian systems, byte swapping is required:

```c
uint32_t SwapEndian(uint32_t value) {
    return ((value & 0xFF000000) >> 24) |
           ((value & 0x00FF0000) >> 8)  |
           ((value & 0x0000FF00) << 8)  |
           ((value & 0x000000FF) << 24);
}
```

---

**Next**: [Texture Loading Workflow](03_texture_loading_workflow.md)
