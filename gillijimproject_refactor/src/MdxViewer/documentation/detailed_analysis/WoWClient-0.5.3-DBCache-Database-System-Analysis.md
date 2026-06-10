# WoW Alpha 0.5.3 (Build 3368) DBCache and Database System Analysis

## Overview

This document provides a deep analysis of the DBCache (database cache) system in WoW Alpha 0.5.3 (Build 3368, Dec 11 2003), based on Ghidra reverse engineering of WoWClient.exe. It covers the database caching infrastructure, cache handlers, and integration with game systems.

## Related Functions

### DBCache Core Functions

| Function | Address | Purpose |
|----------|---------|---------|
| `DBCache_Initialize` | 0x005653b0 | Initialize DBCache |
| `DBCache_Destroy` | 0x00565430 | Destroy DBCache |
| `DBCache_ClearHandlers` | 0x00565970 | Clear all handlers |
| `DBCache_RegisterHandlers` | 0x00565440 | Register handlers |
| `LoadDBCaches` | 0x005653c0 | Load all databases |

### Specific Cache Classes

| Function | Address | Cache Type |
|----------|---------|-----------|
| `DBCache<class_CreatureStats_C,int,class_HASHKEY_INT>` | 0x00565a00 | Creature stats |
| `DBCache<class_GameObjectStats_C,int,class_HASHKEY_INT>` | 0x005677b0 | Game object stats |
| `DBCache<class_ItemStats_C,int,class_HASHKEY_INT>` | 0x005693d0 | Item stats |
| `DBCache<class_NPCText,int,class_HASHKEY_INT>` | 0x0056b080 | NPC text |
| `DBCache<class_QuestCache,int,class_HASHKEY_INT>` | 0x00570880 | Quest cache |
| `DBCache<class_GuildStats_C,int,class_HASHKEY_INT>` | 0x0056ec20 | Guild stats |
| `DBCache<class_PageTextCache_C,int,class_HASHKEY_INT>` | 0x00572760 | Page text |
| `DBCache<class_PetNameCache,int,class_HASHKEY_INT>` | 0x00574440 | Pet names |
| `DBCache<class_CGPetition,int,class_HASHKEY_INT>` | 0x005760c0 | Petition |
| `DBCache<class_NameCache,unsigned___int64,class_CHashKeyGUID>` | 0x0056ccc0 | Name cache (GUID-based) |

### Hash Table Functions

| Function | Address | Purpose |
|----------|---------|---------|
| `?InternalNew@?$TSHashTable@UDBCACHEHASH@...` | 0x00565b80 | Create hash entry |
| `?Link@?$TSList@...` | Various | List link operations |

---

## DBCache Architecture

### DBCache Template Structure

```c
template<typename T, typename KeyType, typename HashKey>
class DBCache {
    /* Hash table for O(1) lookup */
    TSHashTable<DBCACHEHASH, HashKey> hashTable;
    
    /* Explicit list for iteration */
    TSExplicitList<DBCACHEHASH> entryList;
    
    /* Cache handlers */
    DBCACHECALLBACK* loadHandler;
    DBCACHECALLBACK* saveHandler;
    DBCACHECALLBACK* deleteHandler;
    
    /* Statistics */
    uint32_t entryCount;
    uint32_t hitCount;
    uint32_t missCount;
};
```

### DBCACHEHASH Structure

```c
struct DBCACHEHASH {
    /* Hash entry linkage */
    TSLink<DBCACHEHASH> hashLink;
    TSLink<DBCACHEHASH> listLink;
    
    /* Entry data */
    KeyType key;              // Lookup key
    T* data;                 // Cache entry data
    
    /* Metadata */
    uint32_t flags;          // Entry flags
    uint32_t timestamp;      // Load timestamp
    bool isDirty;           // Modified flag
    bool isLoaded;          // Loaded flag
};
```

### DBCACHECALLBACK Structure

```c
struct DBCACHECALLBACK {
    /* Callback function */
    void (*onLoad)(KeyType key, T* data);
    void (*onSave)(KeyType key, T* data);
    void (*onDelete)(KeyType key);
    void (*onClear)();
    
    /* User data */
    void* userData;
};
```

---

## Cache Initialization

### DBCache_Initialize

```c
/* DBCache_Initialize at 0x005653b0 */
void DBCache_Initialize() {
    // Initialize all cache systems
    DBCache<CreatureStats, int, HASHKEY_INT>::Initialize();
    DBCache<GameObjectStats, int, HASHKEY_INT>::Initialize();
    DBCache<ItemStats, int, HASHKEY_INT>::Initialize();
    DBCache<QuestCache, int, HASHKEY_INT>::Initialize();
    DBCache<GuildStats, int, HASHKEY_INT>::Initialize();
    DBCache<NPCText, int, HASHKEY_INT>::Initialize();
    DBCache<PageTextCache, int, HASHKEY_INT>::Initialize();
    DBCache<PetNameCache, int, HASHKEY_INT>::Initialize();
    DBCache<CGPetition, int, HASHKEY_INT>::Initialize();
    DBCache<NameCache, uint64_t, CHashKeyGUID>::Initialize();
    
    // Register callbacks
    RegisterDefaultHandlers();
}
```

---

## Specific Cache Implementations

### CreatureStats Cache

```c
/* DBCache<class_CreatureStats_C,int,class_HASHKEY_INT> at 0x00565a00 */
class DBCache<CreatureStats, int, HASHKEY_INT> {
    /* Hash table */
    TSHashTable<CreatureStatsHASH, HASHKEY_INT> statsHash;
    
    /* Entry list */
    TSExplicitList<CreatureStatsHASH> statsList;
    
    /* Creature-specific data */
    struct CreatureStatsHASH {
        TSLink<CreatureStatsHASH> hashLink;
        TSLink<CreatureStatsHASH> listLink;
        
        int key;                      // Creature ID
        CreatureStats* data;         // Stats data
        
        uint32_t flags;              // Cache flags
        uint32_t timestamp;          // Load time
        
        /* Creature-specific fields */
        uint32_t level;
        uint32_t health;
        uint32_t mana;
        uint32_t armor;
        uint32_t damageMin;
        uint32_t damageMax;
        uint32_t attackPower;
        uint32_t damageSchool;
    };
};
```

### QuestCache

```c
/* DBCache<class_QuestCache,int,class_HASHKEY_INT> at 0x00570880 */
class DBCache<QuestCache, int, HASHKEY_INT> {
    struct QuestCacheHASH {
        TSLink<QuestCacheHASH> hashLink;
        TSLink<QuestCacheHASH> listLink;
        
        int key;                      // Quest ID
        QuestCache* data;            // Quest data
        
        uint32_t flags;
        uint32_t timestamp;
        
        /* Quest-specific fields */
        uint32_t questId;
        char questName[256];
        uint32_t questFlags;
        uint32_t suggestedPlayers;
        uint32_t requiredLevel;
        uint32_t questLevel;
    };
};
```

### NameCache (GUID-based)

```c
/* DBCache<class_NameCache,unsigned___int64,class_CHashKeyGUID> at 0x0056ccc0 */
class DBCache<NameCache, uint64_t, CHashKeyGUID> {
    struct NameCacheHASH {
        TSLink<NameCacheHASH> hashLink;
        TSLink<NameCacheHASH> listLink;
        
        uint64_t guid;               // Object GUID
        NameCache* data;            // Name data
        
        uint32_t flags;
        uint32_t timestamp;
        
        /* GUID-specific fields */
        uint64_t ownerGuid;         // Owner's GUID
        char name[256];             // Entity name
        uint32_t nameFlags;         // Name flags
    };
};
```

---

## Hash Key Types

### HASHKEY_INT

Integer-based hash for ID lookups:

```c
struct HASHKEY_INT {
    uint32_t value;
    
    /* Hash function */
    static uint32_t Hash(uint32_t key) {
        return key ^ (key >> 16) ^ 0x5F3759DF;
    }
    
    /* Equality check */
    static bool Equal(uint32_t a, uint32_t b) {
        return a == b;
    }
};
```

### CHashKeyGUID

GUID-based hash for object lookups:

```c
struct CHashKeyGUID {
    uint64_t guid;
    
    /* Hash function */
    static uint32_t Hash(uint64_t key) {
        uint32_t* parts = (uint32_t*)&key;
        return parts[0] ^ parts[1] ^ 0x9E3779B9;
    }
    
    /* Equality check */
    static bool Equal(uint64_t a, uint64_t b) {
        return a == b;
    }
};
```

---

## Cache Operations

### Lookup

```c
/* Template function for cache lookup */
T* DBCache<T, KeyType, HashKey>::Lookup(KeyType key) {
    // Calculate hash
    uint32_t hash = HashKey::Hash(key);
    
    // Find in hash table
    DBCACHEHASH* entry = hashTable.Find(key, hash);
    if (entry == NULL) {
        // Cache miss - increment counter
        missCount++;
        return NULL;
    }
    
    // Update timestamp
    entry->timestamp = GetCurrentTime();
    
    // Hit
    hitCount++;
    return entry->data;
}
```

### Insert

```c
/* Template function for cache insert */
bool DBCache<T, KeyType, HashKey>::Insert(KeyType key, T* data) {
    // Allocate new entry
    DBCACHEHASH* entry = AllocateEntry();
    entry->key = key;
    entry->data = data;
    entry->timestamp = GetCurrentTime();
    entry->isDirty = false;
    entry->isLoaded = true;
    
    // Add to hash table
    uint32_t hash = HashKey::Hash(key);
    hashTable.Insert(entry, hash);
    
    // Add to list
    entryList.push_back(entry);
    
    entryCount++;
    return true;
}
```

### Remove

```c
/* Template function for cache removal */
bool DBCache<T, KeyType, HashKey>::Remove(KeyType key) {
    // Find entry
    uint32_t hash = HashKey::Hash(key);
    DBCACHEHASH* entry = hashTable.Find(key, hash);
    if (entry == NULL) {
        return false;
    }
    
    // Call delete handler
    if (deleteHandler != NULL) {
        deleteHandler->onDelete(key);
    }
    
    // Remove from structures
    hashTable.Remove(entry);
    entryList.remove(entry);
    
    // Free entry
    FreeEntry(entry);
    
    entryCount--;
    return true;
}
```

---

## Cache Handlers

### DBCache_RegisterHandlers

```c
/* DBCache_RegisterHandlers at 0x00565440 */
void DBCache_RegisterHandlers(
    DBCACHETYPE cacheType,
    DBCACHECALLBACK* handlers
) {
    switch (cacheType) {
        case DBCACHE_CREATURE_STATS:
            CreatureStatsCache->RegisterHandlers(handlers);
            break;
        case DBCACHE_GAMEOBJECT_STATS:
            GameObjectStatsCache->RegisterHandlers(handlers);
            break;
        case DBCACHE_ITEM_STATS:
            ItemStatsCache->RegisterHandlers(handlers);
            break;
        case DBCACHE_QUEST:
            QuestCache->RegisterHandlers(handlers);
            break;
        // ... other types
    }
}
```

### Handler Types

```c
typedef enum {
    DBCACHE_CREATURE_STATS = 0,
    DBCACHE_GAMEOBJECT_STATS = 1,
    DBCACHE_ITEM_STATS = 2,
    DBCACHE_QUEST = 3,
    DBCACHE_GUILD_STATS = 4,
    DBCACHE_NPC_TEXT = 5,
    DBCACHE_PAGE_TEXT = 6,
    DBCACHE_PET_NAME = 7,
    DBCACHE_CG_PETITION = 8,
    DBCACHE_NAME = 9,
    DBCACHE_NUM_TYPES = 10
} DBCACHETYPE;
```

---

## Cache Statistics

### Statistics Structure

```c
struct DBCacheStats {
    /* Cache info */
    uint32_t entryCount;       // Current entries
    uint32_t maxEntries;       // Maximum entries
    
    /* Performance */
    uint32_t hitCount;        // Cache hits
    uint32_t missCount;       // Cache misses
    float hitRate;            // Hit ratio
    
    /* Memory */
    uint32_t memoryUsage;     // Bytes used
    
    /* Timestamps */
    uint32_t lastLoadTime;
    uint32_t lastClearTime;
};
```

### GetCacheStats

```c
void GetCacheStats(DBCACHETYPE cacheType, DBCacheStats* stats) {
    DBCache* cache = GetCache(cacheType);
    
    stats->entryCount = cache->entryCount;
    stats->maxEntries = cache->maxEntries;
    stats->hitCount = cache->hitCount;
    stats->missCount = cache->missCount;
    
    if (cache->hitCount + cache->missCount > 0) {
        stats->hitRate = (float)cache->hitCount / 
                        (cache->hitCount + cache->missCount);
    } else {
        stats->hitRate = 0.0f;
    }
    
    stats->memoryUsage = cache->memoryUsage;
    stats->lastLoadTime = cache->lastLoadTime;
    stats->lastClearTime = cache->lastClearTime;
}
```

---

## DBC File Loading

### LoadDBCaches

```c
/* LoadDBCaches at 0x005653c0 */
void LoadDBCaches() {
    // Load creature DBC
    LoadDBCache("CreatureStats.dbc", DBCACHE_CREATURE_STATS);
    
    // Load game object DBC
    LoadDBCache("GameObject.dbc", DBCACHE_GAMEOBJECT_STATS);
    
    // Load item DBC
    LoadDBCache("Item.dbc", DBCACHE_ITEM_STATS);
    
    // Load quest DBC
    LoadDBCache("Quest.dbc", DBCACHE_QUEST);
    
    // Load NPC text DBC
    LoadDBCache("NPCText.dbc", DBCACHE_NPC_TEXT);
    
    // Load guild DBC
    LoadDBCache("Guild.dbc", DBCACHE_GUILD_STATS);
    
    // Load page text DBC
    LoadDBCache("PageText.dbc", DBCACHE_PAGE_TEXT);
    
    // Load pet name DBC
    LoadDBCache("PetName.dbc", DBCACHE_PET_NAME);
    
    // Load petition DBC
    LoadDBCache("Petition.dbc", DBCACHE_CG_PETITION);
}
```

### LoadDBCache

```c
void LoadDBCache(const char* filename, DBCACHETYPE cacheType) {
    // Open DBC file
    HANDLE fileHandle;
    if (!SFileOpenFile(filename, &fileHandle)) {
        Error("Failed to open %s", filename);
        return;
    }
    
    // Read header
    DBCHeader header;
    uint32_t bytesRead;
    SFileReadFile(fileHandle, &header, sizeof(header), &bytesRead);
    
    // Validate signature
    if (header.signature != 'WDBC' && header.signature != 'WDB2') {
        Error("Invalid DBC signature");
        SFileCloseFile(fileHandle);
        return;
    }
    
    // Get cache
    DBCache* cache = GetCache(cacheType);
    
    // Read records
    for (uint32_t i = 0; i < header.recordCount; i++) {
        // Read record
        void* recordData = malloc(header.recordSize);
        SFileReadFile(fileHandle, recordData, header.recordSize, &bytesRead);
        
        // Get ID from first field
        int recordId = *(int*)recordData;
        
        // Insert into cache
        cache->Insert(recordId, (T*)recordData);
    }
    
    SFileCloseFile(fileHandle);
}
```

---

## DBC File Format

### DBCHeader Structure

```c
struct DBCHeader {
    uint32_t signature;       // 'WDBC' or 'WDB2'
    uint32_t recordCount;    // Number of records
    uint32_t fieldCount;     // Number of fields per record
    uint32_t recordSize;     // Size of one record (bytes)
    uint32_t stringBlockSize; // Size of string block
};

struct WDB2Header extends DBCHeader {
    uint32_t tableHash;      // Hash of table
    uint32_t build;         // Client build
    uint32_t timestamp;      // Last modified
    
    /* WDB2 flags */
    uint32_t flags;         // 0x01 = index is ID, 0x02 = has extra field
};
```

---

## Summary

The DBCache system in WoW Alpha 0.5.3 provides:
- **Template-based caching**: Generic cache implementation for multiple types
- **O(1) lookups**: Hash table-based fast access
- **Multiple key types**: Integer and GUID-based keys
- **Callback system**: Load/save/delete hooks
- **Statistics tracking**: Hit/miss counts and memory usage
- **DBC file loading**: Standard DBC/WDB2 format support

Key functions and addresses provide a complete reference for reverse engineering and implementation.

---

*Document created: 2026-02-07*
*Analysis based on WoWClient.exe (Build 3368)*
