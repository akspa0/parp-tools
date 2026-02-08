# Optimization Techniques

## Overview

This document covers memory management, caching strategies, and optimization techniques for the DBC texturing system based on analysis of the WoW client implementation.

## Memory Management

### Texture Cache Architecture

The client uses a two-tier caching system:

```c
// Primary cache - hash table by file path
TSHashTable<UCTextureHash, HASHKEY_TEXTUREFILE> g_TextureCache;

// Solid color cache - hash table by color value
TSHashTable<UCSolidTextureHash, HASHKEY_NONE> g_SolidTextureCache;
```

### Reference Counting

Automatic memory management through reference counting:

```c
struct TextureHandle {
    uint32_t refCount;      // Number of references
    void* gpuResource;      // GPU texture object
    uint32_t width;
    uint32_t height;
    uint32_t format;
    char path[260];         // Original path
};

void IncrementRefCount(HTEXTURE__* handle) {
    if (handle) {
        handle->refCount++;
    }
}

void DecrementRefCount(HTEXTURE__* handle) {
    if (handle) {
        handle->refCount--;
        
        if (handle->refCount == 0) {
            // Free GPU resources
            ReleaseGPUTexture(handle->gpuResource);
            
            // Remove from cache
            RemoveFromCache(handle->path);
            
            // Free handle
            free(handle);
        }
    }
}
```

### Memory Pools

Efficient allocation for small objects:

```c
struct MemoryPool {
    void* blocks[256];
    uint32_t blockSize;
    uint32_t numBlocks;
    uint32_t nextFree;
};

MemoryPool g_textureHandlePool;
MemoryPool g_cacheEntryPool;

void InitializeMemoryPools() {
    // Texture handles pool
    InitPool(&g_textureHandlePool, sizeof(TextureHandle), 1024);
    
    // Cache entries pool
    InitPool(&g_cacheEntryPool, sizeof(UCTextureHash), 2048);
}

void* AllocFromPool(MemoryPool* pool) {
    if (pool->nextFree >= pool->numBlocks) {
        // Expand pool
        ExpandPool(pool);
    }
    
    void* ptr = pool->blocks[pool->nextFree++];
    return ptr;
}
```

## Caching Strategies

### Hash Table Implementation

```c
struct TSHashTable {
    UCTextureHash** buckets;
    uint32_t numBuckets;
    uint32_t numEntries;
    uint32_t collisions;
};

uint32_t ComputeTextureHash(const char* path) {
    uint32_t hash = 0x811C9DC5;  // FNV-1a initial value
    
    for (const char* p = path; *p; p++) {
        hash ^= (uint8_t)tolower(*p);
        hash *= 0x01000193;  // FNV prime
    }
    
    return hash;
}

UCTextureHash* LookupInCache(const char* path) {
    uint32_t hash = ComputeTextureHash(path);
    uint32_t bucket = hash % g_TextureCache.numBuckets;
    
    UCTextureHash* entry = g_TextureCache.buckets[bucket];
    
    while (entry) {
        if (_stricmp(entry->path, path) == 0) {
            return entry;  // Cache hit
        }
        entry = entry->next;
    }
    
    return NULL;  // Cache miss
}
```

### Cache Eviction Policy

LRU (Least Recently Used) eviction:

```c
struct CacheEntry {
    HTEXTURE__* handle;
    char path[260];
    uint32_t lastAccessFrame;
    uint32_t accessCount;
    CacheEntry* next;
};

void UpdateAccessTime(CacheEntry* entry) {
    entry->lastAccessFrame = g_currentFrame;
    entry->accessCount++;
}

void EvictLRUTextures(uint32_t targetMemory) {
    uint32_t currentMemory = GetTextureMemoryUsage();
    
    if (currentMemory <= targetMemory) {
        return;  // Already under budget
    }
    
    // Build LRU list
    CacheEntry** entries = SortByAccessTime();
    uint32_t numEntries = GetCacheSize();
    
    // Evict oldest first
    for (uint32_t i = 0; i < numEntries; i++) {
        if (entries[i]->handle->refCount == 1) {  // Only cache holds it
            uint32_t size = GetTextureSize(entries[i]->handle);
            
            HandleClose(entries[i]->handle);
            currentMemory -= size;
            
            if (currentMemory <= targetMemory) {
                break;
            }
        }
    }
    
    free(entries);
}
```

## Texture Compression

### DXT Compression

World of Warcraft uses DXT (S3TC) compression extensively:

```c
/**
 * DXT1: 6:1 compression ratio, no alpha or 1-bit alpha
 * DXT3: 4:1 compression ratio, explicit alpha
 * DXT5: 4:1 compression ratio, interpolated alpha
 */
uint32_t CalculateDXTSize(uint32_t width, uint32_t height, 
                         DXTFormat format) {
    uint32_t numBlocks = ((width + 3) / 4) * ((height + 3) / 4);
    
    switch (format) {
        case DXT1:
            return numBlocks * 8;   // 8 bytes per 4x4 block
        case DXT3:
        case DXT5:
            return numBlocks * 16;  // 16 bytes per 4x4 block
        default:
            return 0;
    }
}

void CompressTextureDXT1(const uint8_t* rgba, uint8_t* dxt,
                        uint32_t width, uint32_t height) {
    for (uint32_t y = 0; y < height; y += 4) {
        for (uint32_t x = 0; x < width; x += 4) {
            // Extract 4x4 block
            uint32_t block[16];
            Extract4x4Block(rgba, width, x, y, block);
            
            // Find color endpoints
            uint16_t color0, color1;
            FindColorEndpoints(block, &color0, &color1);
            
            // Generate lookup table
            uint32_t lookupBits = GenerateLookupTable(block, 
                                                     color0, color1);
            
            // Write compressed block
            WriteDXT1Block(dxt, color0, color1, lookupBits);
            dxt += 8;
        }
    }
}
```

### Runtime Compression

For dynamic textures:

```c
HTEXTURE__* CreateCompressedTexture(const uint8_t* rgba,
                                   uint32_t width, uint32_t height) {
    uint32_t compressedSize = CalculateDXTSize(width, height, DXT1);
    uint8_t* compressed = malloc(compressedSize);
    
    // Compress using hardware if available
    if (HasHardwareDXTCompression()) {
        CompressTextureDXT1_HW(rgba, compressed, width, height);
    } else {
        CompressTextureDXT1(rgba, compressed, width, height);
    }
    
    // Upload to GPU
    HTEXTURE__* handle = CreateGPUTexture(compressed, width, height,
                                         FORMAT_DXT1);
    
    free(compressed);
    return handle;
}
```

## Mipmap Optimization

### Generating Mipmaps

```c
void GenerateMipmaps(TextureData* texture) {
    uint32_t width = texture->width;
    uint32_t height = texture->height;
    uint32_t mipLevel = 1;
    
    uint8_t* prevMip = texture->pixels;
    
    while (width > 1 || height > 1) {
        uint32_t newWidth = max(1, width / 2);
        uint32_t newHeight = max(1, height / 2);
        
        uint8_t* mipData = malloc(newWidth * newHeight * 4);
        
        // Box filter downsampling
        for (uint32_t y = 0; y < newHeight; y++) {
            for (uint32_t x = 0; x < newWidth; x++) {
                uint32_t r = 0, g = 0, b = 0, a = 0;
                
                // Sample 2x2 region from previous mip
                for (uint32_t dy = 0; dy < 2 && (y*2+dy) < height; dy++) {
                    for (uint32_t dx = 0; dx < 2 && (x*2+dx) < width; dx++) {
                        uint32_t srcIdx = ((y*2+dy) * width + (x*2+dx)) * 4;
                        r += prevMip[srcIdx + 0];
                        g += prevMip[srcIdx + 1];
                        b += prevMip[srcIdx + 2];
                        a += prevMip[srcIdx + 3];
                    }
                }
                
                // Average and store
                uint32_t dstIdx = (y * newWidth + x) * 4;
                mipData[dstIdx + 0] = r / 4;
                mipData[dstIdx + 1] = g / 4;
                mipData[dstIdx + 2] = b / 4;
                mipData[dstIdx + 3] = a / 4;
            }
        }
        
        // Upload mip level to GPU
        UploadMipLevel(texture->gpuHandle, mipLevel, mipData,
                      newWidth, newHeight);
        
        // Next iteration
        if (mipLevel > 1) {
            free(prevMip);
        }
        prevMip = mipData;
        width = newWidth;
        height = newHeight;
        mipLevel++;
    }
}
```

### Mipmap Streaming

Load mipmaps on demand based on usage:

```c
void StreamMipmaps(HTEXTURE__* texture, float distance) {
    uint32_t requiredMips = CalculateRequiredMips(distance);
    uint32_t currentMips = GetLoadedMips(texture);
    
    if (currentMips < requiredMips) {
        // Stream in higher resolution mips
        for (uint32_t mip = currentMips; mip < requiredMips; mip++) {
            QueueMipLoad(texture, mip, PRIORITY_HIGH);
        }
    } else if (currentMips > requiredMips + 2) {
        // Can safely unload lowest quality mips
        for (uint32_t mip = requiredMips + 2; mip < currentMips; mip++) {
            UnloadMipLevel(texture, mip);
        }
    }
}
```

## Batching

### Batch Texture Loading

```c
void BatchLoadTextures(const char** paths, uint32_t count,
                      HTEXTURE__** outHandles) {
    // Sort by MPQ location for sequential reads
    SortPathsByMPQOffset(paths, count);
    
    // Pre-warm file system cache
    PrefetchFiles(paths, count);
    
    // Load all textures
    for (uint32_t i = 0; i < count; i++) {
        outHandles[i] = LoadTexture(paths[i]);
    }
    
    // Batch GPU uploads
    BatchGPUUpload(outHandles, count);
}

void SortPathsByMPQOffset(const char** paths, uint32_t count) {
    struct PathOffset {
        const char* path;
        uint64_t offset;
    };
    
    PathOffset* offsets = malloc(sizeof(PathOffset) * count);
    
    // Get file offsets
    for (uint32_t i = 0; i < count; i++) {
        offsets[i].path = paths[i];
        offsets[i].offset = GetMPQFileOffset(paths[i]);
    }
    
    // Sort by offset
    qsort(offsets, count, sizeof(PathOffset), CompareOffsets);
    
    // Reorder paths array
    for (uint32_t i = 0; i < count; i++) {
        paths[i] = offsets[i].path;
    }
    
    free(offsets);
}
```

## Lazy Loading

### Deferred Texture Creation

```c
struct LazyTexture {
    char path[260];
    HTEXTURE__* handle;
    bool loaded;
    uint32_t priority;
};

HTEXTURE__* GetOrLoadTexture(LazyTexture* lazy) {
    if (!lazy->loaded) {
        // Load on first access
        lazy->handle = LoadTexture(lazy->path);
        lazy->loaded = true;
    }
    
    return lazy->handle;
}

// Use placeholder until real texture loads
HTEXTURE__* GetTexturePlaceholder(LazyTexture* lazy) {
    if (lazy->loaded) {
        return lazy->handle;
    } else {
        // Return low-res placeholder
        return GetPlaceholderTexture();
    }
}
```

## Texture Atlasing

### Atlas Generation

```c
struct TextureAtlas {
    HTEXTURE__* gpuTexture;
    uint32_t width;
    uint32_t height;
    uint32_t numTextures;
    
    struct Entry {
        uint16_t x, y;
        uint16_t width, height;
        char name[64];
    } *entries;
};

TextureAtlas* CreateAtlas(const char** texturePaths, uint32_t count) {
    // Load all textures
    TextureData** textures = malloc(sizeof(TextureData*) * count);
    for (uint32_t i = 0; i < count; i++) {
        textures[i] = LoadTextureData(texturePaths[i]);
    }
    
    // Pack textures into atlas
    uint32_t atlasWidth = 2048;
    uint32_t atlasHeight = 2048;
    uint8_t* atlasData = calloc(atlasWidth * atlasHeight, 4);
    
    TextureAtlas* atlas = malloc(sizeof(TextureAtlas));
    atlas->width = atlasWidth;
    atlas->height = atlasHeight;
    atlas->numTextures = count;
    atlas->entries = malloc(sizeof(TextureAtlas::Entry) * count);
    
    // Simple left-to-right, top-to-bottom packing
    uint32_t x = 0, y = 0, rowHeight = 0;
    
    for (uint32_t i = 0; i < count; i++) {
        TextureData* tex = textures[i];
        
        // Check if we need to go to next row
        if (x + tex->width > atlasWidth) {
            x = 0;
            y += rowHeight;
            rowHeight = 0;
        }
        
        // Copy texture into atlas
        CopyTextureRegion(atlasData, atlasWidth, atlasHeight,
                         x, y, tex->pixels, tex->width, tex->height);
        
        // Store entry
        atlas->entries[i].x = x;
        atlas->entries[i].y = y;
        atlas->entries[i].width = tex->width;
        atlas->entries[i].height = tex->height;
        strncpy(atlas->entries[i].name, texturePaths[i], 63);
        
        // Advance position
        x += tex->width;
        rowHeight = max(rowHeight, tex->height);
        
        FreeTextureData(textures[i]);
    }
    
    // Create GPU texture
    atlas->gpuTexture = CreateGPUTextureFromData(atlasData, 
                                                atlasWidth, 
                                                atlasHeight);
    
    free(atlasData);
    free(textures);
    
    return atlas;
}
```

## Performance Monitoring

### Instrumentation

```c
struct TexturePerfStats {
    uint64_t totalLoadTime;
    uint64_t totalUploadTime;
    uint32_t numLoads;
    uint32_t numUploads;
    uint32_t cacheHits;
    uint32_t cacheMisses;
    uint32_t bytesLoaded;
    uint32_t bytesUploaded;
};

TexturePerfStats g_texPerfStats;

void ResetPerfStats() {
    memset(&g_texPerfStats, 0, sizeof(g_texPerfStats));
}

void PrintPerfStats() {
    double avgLoadTime = (double)g_texPerfStats.totalLoadTime / 
                        max(1, g_texPerfStats.numLoads);
    double cacheHitRate = 100.0 * g_texPerfStats.cacheHits /
                         max(1, g_texPerfStats.cacheHits + 
                             g_texPerfStats.cacheMisses);
    
    LogInfo("=== Texture Performance Stats ===");
    LogInfo("Loads: %d (avg %.2f ms)", 
            g_texPerfStats.numLoads, avgLoadTime / 1000.0);
    LogInfo("Cache hit rate: %.1f%%", cacheHitRate);
    LogInfo("Data loaded: %.2f MB",
            g_texPerfStats.bytesLoaded / (1024.0 * 1024.0));
    LogInfo("Data uploaded: %.2f MB",
            g_texPerfStats.bytesUploaded / (1024.0 * 1024.0));
}
```

---

**Next**: [Error Handling](08_error_handling.md)
