# Troubleshooting Guide

## Common Issues and Solutions

This guide covers common problems encountered when working with the DBC texturing system and their solutions.

## Issue 1: Magenta/Pink Textures

### Symptoms
- Models rendered with bright magenta (0xFFFF00FF) textures
- Missing visual details on characters or creatures

### Cause
The magenta texture is a fallback placeholder created when:
1. Texture file not found in game data
2. Invalid or corrupt texture file
3. Incorrect texture path in DBC or model file
4. MPQ archive not loaded properly

### Solutions

**Check texture path existence:**
```c
bool VerifyTexturePath(const char* path) {
    // Try to open file
    FILE* f = fopen(path, "rb");
    if (f) {
        fclose(f);
        return true;
    }
    
    // Also check in MPQ archives
    if (MPQFileExists(path)) {
        return true;
    }
    
    LogError("Texture not found: %s", path);
    return false;
}
```

**Enable texture loading debug output:**
```c
// In ProcessTextures function, add logging:
if (useDefault || !texInfo->filename[0]) {
    LogDebug("Using fallback texture for slot %d (reason: %s)",
            i, useDefault ? "forced default" : "empty path");
    textureArray[i].handle = TextureCreateSolid(0xFFFF00FF);
} else {
    LogDebug("Loading texture: %s", texInfo->filename);
    // ... load texture
}
```

**Common path issues:**
- Missing leading backslash
- Wrong case (should be lowercase on Linux)
- Forward slashes instead of backslashes
- Extra spaces in path

**Fix:**
```c
void FixTexturePath(char* path) {
    // Trim whitespace
    TrimWhitespace(path);
    
    // Convert to lowercase
    _strlwr(path);
    
    // Replace forward slashes
    for (char* p = path; *p; p++) {
        if (*p == '/') *p = '\\';
    }
    
    // Ensure no double backslashes
    RemoveDoubleBackslashes(path);
}
```

---

## Issue 2: Texture Load Failures

### Symptoms
- NULL texture handles returned
- Error messages in log about failed loads
- Black textures instead of correct appearance

### Cause
1. BLP file corruption
2. Unsupported BLP format version
3. Out of memory
4. GPU/driver issues

### Diagnostic Code

```c
HTEXTURE__* LoadTextureWithDiagnostics(const char* path) {
    LogInfo("Attempting to load: %s", path);
    
    // Check file exists
    if (!FileExists(path)) {
        LogError("File not found: %s", path);
        return CreateFallbackTexture();
    }
    
    // Check file size
    uint32_t fileSize = GetFileSize(path);
    LogInfo("File size: %d bytes", fileSize);
    
    if (fileSize < 148) {  // Minimum BLP header size
        LogError("File too small to be valid BLP: %s", path);
        return CreateFallbackTexture();
    }
    
    // Try to load
    CGxTexFlags flags;
    flags.filter = GxTex_LinearMipLinear;
    
    CStatus status;
    HTEXTURE__* texture = TextureCreate(path, flags, &status, 0);
    
    if (!texture) {
        LogError("TextureCreate failed with status: 0x%X", 
                status.errorCode);
        
        // Try alternate loading method
        texture = LoadTextureRaw(path);
        
        if (!texture) {
            LogError("All loading methods failed for: %s", path);
            return CreateFallbackTexture();
        }
    }
    
    LogInfo("Successfully loaded: %s (handle: %p)", path, texture);
    return texture;
}
```

### Solutions

**Check BLP format:**
```c
bool ValidateBLPFormat(const void* data, uint32_t size) {
    if (size < 4) return false;
    
    const uint32_t* magic = (const uint32_t*)data;
    
    // Check for BLP magic numbers
    if (*magic == '0PLB') {  // BLP0
        LogInfo("BLP0 format detected");
        return true;
    }
    if (*magic == '1PLB') {  // BLP1
        LogInfo("BLP1 format detected");
        return true;
    }
    if (*magic == '2PLB') {  // BLP2
        LogInfo("BLP2 format detected");
        return true;
    }
    
    LogError("Invalid BLP magic: 0x%X", *magic);
    return false;
}
```

**Memory debugging:**
```c
void CheckTextureMemory() {
    MEMORYSTATUS memStatus;
    GlobalMemoryStatus(&memStatus);
    
    LogInfo("Physical memory: %d%% used", 
            memStatus.dwMemoryLoad);
    LogInfo("Available: %d MB", 
            memStatus.dwAvailPhys / (1024 * 1024));
    
    // Check texture cache size
    uint32_t numTextures, totalMemory;
    GetCacheStats(&numTextures, &totalMemory);
    
    LogInfo("Texture cache: %d textures, %d MB",
            numTextures, totalMemory / (1024 * 1024));
    
    if (totalMemory > 512 * 1024 * 1024) {  // >512 MB
        LogWarning("Texture memory usage high, consider purging");
        PurgeUnusedTextures();
    }
}
```

---

## Issue 3: Wrong Textures Applied

### Symptoms
- Creature/character has incorrect appearance
- Texture doesn't match expected result
- Item displays wrong texture

### Cause
1. Incorrect DBC record ID used
2. DBC cache not loaded/initialized
3. Texture slot mismatch
4. Override priority incorrect

### Debugging

```c
void DebugCreatureTextures(uint32_t displayInfoID) {
    LogInfo("=== Debugging Display Info %d ===", displayInfoID);
    
    CreatureDisplayInfoRec* info = 
        g_CreatureDisplayInfoCache.Lookup(displayInfoID);
    
    if (!info) {
        LogError("DisplayInfo not found!");
        return;
    }
    
    LogInfo("Model ID: %d", info->ModelID);
    LogInfo("Scale: %.2f", info->Scale);
    
    const char* skin1 = GetDBCString(info->Skin1);
    const char* skin2 = GetDBCString(info->Skin2);
    const char* skin3 = GetDBCString(info->Skin3);
    
    LogInfo("Skin1: %s", skin1 ? skin1 : "(none)");
    LogInfo("Skin2: %s", skin2 ? skin2 : "(none)");
    LogInfo("Skin3: %s", skin3 ? skin3 : "(none)");
    
    // Check if model exists
    CreatureModelDataRec* modelData = 
        g_CreatureModelDataCache.Lookup(info->ModelID);
    
    if (modelData) {
        const char* modelPath = GetDBCString(modelData->ModelPath);
        LogInfo("Model path: %s", modelPath ? modelPath : "(none)");
    } else {
        LogError("Model data %d not found!", info->ModelID);
    }
}
```

### Solutions

**Verify DBC lookup:**
```c
void VerifyDBCLookup(uint32_t id) {
    // Check if DBC caches initialized
    if (!g_DBCInitialized) {
        LogError("DBC caches not initialized!");
        LoadDBCaches();
    }
    
    // Try lookup
    void* record = g_CreatureDisplayInfoCache.Lookup(id);
    
    if (!record) {
        LogError("Record %d not found in cache", id);
        
        // Try linear search in raw DBC
        record = LinearSearchDBC("CreatureDisplayInfo.dbc", id);
        
        if (record) {
            LogWarning("Record found in DBC but not in cache!");
            LogWarning("Cache may be incorrectly initialized");
        }
    }
}
```

**Validate texture slot mapping:**
```c
void ValidateTextureSlots(CModelComplex* model) {
    LogInfo("Model has %d texture slots", model->m_textures.count);
    
    for (uint32_t i = 0; i < model->m_textures.count; i++) {
        CModelTexture* tex = &model->m_textures.data[i];
        
        LogInfo("Slot %d:", i);
        LogInfo("  Type: %d", tex->type);
        LogInfo("  Flags: 0x%X", tex->flags);
        LogInfo("  Handle: %p", tex->handle);
        
        if (tex->handle) {
            LogInfo("  Valid texture bound");
        } else {
            LogWarning("  No texture bound!");
        }
    }
}
```

---

## Issue 4: Performance Problems

### Symptoms
- Slow texture loading
- FPS drops when new models appear
- High memory usage
- Long loading screens

### Causes
1. Synchronous loading blocks rendering
2. No texture caching
3. Loading same texture multiple times
4. Large uncompressed textures
5. No mipmaps causing GPU thrashing

### Performance Profiling

```c
struct TextureLoadStats {
    uint32_t totalLoads;
    uint32_t cacheHits;
    uint32_t cacheMisses;
    uint32_t failedLoads;
    uint64_t totalLoadTime;  // Microseconds
    uint64_t totalFileSize;  // Bytes
};

TextureLoadStats g_texStats = {0};

HTEXTURE__* LoadTextureProfiled(const char* path) {
    uint64_t startTime = GetMicroseconds();
    
    // Check cache
    HTEXTURE__* cached = FindInCache(path);
    if (cached) {
        g_texStats.cacheHits++;
        g_texStats.totalLoads++;
        return cached;
    }
    
    g_texStats.cacheMisses++;
    
    // Load from file
    HTEXTURE__* texture = TextureCreate(path, /* ... */);
    
    uint64_t loadTime = GetMicroseconds() - startTime;
    g_texStats.totalLoadTime += loadTime;
    g_texStats.totalLoads++;
    
    if (texture) {
        uint64_t fileSize = GetTextureFileSize(path);
        g_texStats.totalFileSize += fileSize;
        
        LogDebug("Loaded %s in %lld μs (%lld bytes)", 
                path, loadTime, fileSize);
    } else {
        g_texStats.failedLoads++;
    }
    
    return texture;
}

void PrintTextureStats() {
    LogInfo("=== Texture Loading Statistics ===");
    LogInfo("Total loads: %d", g_texStats.totalLoads);
    LogInfo("Cache hits: %d (%.1f%%)", 
            g_texStats.cacheHits,
            100.0f * g_texStats.cacheHits / g_texStats.totalLoads);
    LogInfo("Cache misses: %d", g_texStats.cacheMisses);
    LogInfo("Failed loads: %d", g_texStats.failedLoads);
    LogInfo("Average load time: %lld μs",
            g_texStats.totalLoadTime / max(1, g_texStats.cacheMisses));
    LogInfo("Total data loaded: %.2f MB",
            g_texStats.totalFileSize / (1024.0 * 1024.0));
}
```

### Solutions

**Use asynchronous loading:**
```c
void LoadModelAsync(const char* modelPath, 
                   LoadCompleteCallback callback) {
    // Load model structure immediately (small)
    CModelComplex* model = AllocateModel();
    
    // Queue textures for async load
    for (uint32_t i = 0; i < model->numTextures; i++) {
        const char* texPath = model->textureInfo[i].path;
        
        AsyncLoadRequest* request = CreateAsyncRequest();
        request->path = texPath;
        request->priority = PRIORITY_NORMAL;
        request->callback = OnTextureLoadComplete;
        request->userData = model;
        
        QueueAsyncLoad(request);
    }
    
    // Return model immediately (textures will load in background)
    callback(model, true);
}
```

**Implement texture streaming:**
```c
void UpdateTextureStreaming(Camera* camera) {
    // Get visible models
    CModelComplex** visibleModels = GetVisibleModels(camera);
    uint32_t numVisible = GetNumVisibleModels();
    
    for (uint32_t i = 0; i < numVisible; i++) {
        CModelComplex* model = visibleModels[i];
        float distance = DistanceToCamera(model, camera);
        
        // Determine appropriate LOD level
        uint32_t lodLevel = CalculateLOD(distance);
        
        // Stream in textures for this LOD
        for (uint32_t t = 0; t < model->numTextures; t++) {
            StreamTextureLOD(model, t, lodLevel);
        }
    }
}
```

**Optimize cache:**
```c
void OptimizeTextureCache() {
    // Remove duplicates
    DeduplicateCache();
    
    // Compress unused textures
    CompressUnusedTextures();
    
    // Generate mipmaps if missing
    GenerateMissingMipmaps();
    
    // Defragment GPU memory
    DefragmentGPUMemory();
}
```

---

## Issue 5: Crash on Texture Load

### Symptoms
- Application crash when loading specific textures
- Access violation errors
- Heap corruption detected

### Diagnostic Steps

```c
// Enable guard pages around texture allocations
#ifdef DEBUG_TEXTURES
void* AllocateTextureMemory(uint32_t size) {
    // Allocate extra for guard pages
    uint32_t pageSize = 4096;
    uint32_t totalSize = size + (pageSize * 2);
    
    void* memory = VirtualAlloc(NULL, totalSize,
                               MEM_COMMIT | MEM_RESERVE,
                               PAGE_READWRITE);
    
    if (!memory) return NULL;
    
    // Protect guard pages
    VirtualProtect(memory, pageSize, PAGE_NOACCESS, NULL);
    VirtualProtect((char*)memory + pageSize + size, 
                  pageSize, PAGE_NOACCESS, NULL);
    
    // Return usable memory
    return (char*)memory + pageSize;
}
#endif

// Validate all texture data before GPU upload
bool ValidateTextureData(TextureData* data) {
    if (!data) {
        LogError("NULL texture data");
        return false;
    }
    
    if (!data->pixels) {
        LogError("NULL pixel data");
        return false;
    }
    
    if (data->width == 0 || data->height == 0) {
        LogError("Invalid dimensions: %dx%d", 
                data->width, data->height);
        return false;
    }
    
    if (data->width > 4096 || data->height > 4096) {
        LogError("Dimensions too large: %dx%d",
                data->width, data->height);
        return false;
    }
    
    uint32_t expectedSize = data->width * data->height * 
                           GetBytesPerPixel(data->format);
    
    if (data->dataSize < expectedSize) {
        LogError("Data size mismatch: have %d, need %d",
                data->dataSize, expectedSize);
        return false;
    }
    
    return true;
}
```

### Solutions

**Safe texture creation:**
```c
HTEXTURE__* SafeTextureCreate(const char* path) {
    __try {
        return TextureCreate(path, /* ... */);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        LogError("Exception loading texture: %s", path);
        LogError("Exception code: 0x%X", GetExceptionCode());
        return CreateFallbackTexture();
    }
}
```

---

## Issue 6: Character Textures Not Updated

### Symptoms
- Character customization changes not visible
- Texture swap doesn't take effect
- Old texture still showing

### Cause
1. Texture handle not updated in model
2. Cache returning stale texture
3. GPU texture not re-uploaded
4. Rendering using wrong texture reference

### Solution

```c
void ForceUpdateCharacterTexture(CModelComplex* model, 
                                uint32_t slot,
                                const char* newPath) {
    // Remove from cache first
    RemoveFromCache(newPath);
    
    // Close old texture
    if (model->m_textures.data[slot].handle) {
        HandleClose(model->m_textures.data[slot].handle);
        model->m_textures.data[slot].handle = NULL;
    }
    
    // Force reload
    CGxTexFlags flags = GetDefaultFlags();
    CStatus status;
    HTEXTURE__* newTex = TextureCreate(newPath, flags, &status, 0);
    
    if (newTex) {
        model->m_textures.data[slot].handle = newTex;
        
        // Force re-bind on next render
        model->textureDirtyFlags |= (1 << slot);
    }
}
```

---

## Debug Commands

Useful debug commands for troubleshooting:

```c
// Dump all loaded textures
void DumpLoadedTextures() {
    FILE* f = fopen("texture_dump.txt", "w");
    
    for (uint32_t i = 0; i < g_TextureCache.numBuckets; i++) {
        UCTextureHash* entry = g_TextureCache.buckets[i];
        
        while (entry) {
            fprintf(f, "Path: %s\n", entry->path);
            fprintf(f, "  Handle: %p\n", entry->handle);
            fprintf(f, "  RefCount: %d\n", entry->refCount);
            fprintf(f, "  Size: %d bytes\n", 
                   GetTextureSize(entry->handle));
            fprintf(f, "\n");
            
            entry = entry->next;
        }
    }
    
    fclose(f);
}

// Reload all textures
void ReloadAllTextures() {
    LogInfo("Reloading all textures...");
    
    // Backup paths
    uint32_t count = 0;
    char** paths = CollectAllTexturePaths(&count);
    
    // Clear cache
    ClearTextureCache();
    
    // Reload all
    for (uint32_t i = 0; i < count; i++) {
        LoadTexture(paths[i]);
        free(paths[i]);
    }
    
    free(paths);
    LogInfo("Reloaded %d textures", count);
}
```

---

**Next**: [Performance Benchmarks](14_performance.md)
