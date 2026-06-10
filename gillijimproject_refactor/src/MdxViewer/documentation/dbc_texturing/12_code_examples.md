# Code Examples

## Overview

This document provides practical code examples demonstrating how to use the DBC texturing system. All examples are based on reverse engineering analysis of the WoW client binary.

## Example 1: Loading a Creature Model with Textures

Complete example of loading a creature model and applying textures from DBC.

```c
#include "DBCache.h"
#include "Model.h"
#include "Texture.h"

/**
 * Load a creature model with textures from DBC
 * @param displayInfoID - CreatureDisplayInfo.dbc record ID
 * @return Loaded model or NULL on failure
 */
CModelComplex* LoadCreatureModel(uint32_t displayInfoID) {
    // Step 1: Lookup display info in DBC
    CreatureDisplayInfoRec* displayInfo = 
        g_CreatureDisplayInfoCache.Lookup(displayInfoID);
    
    if (!displayInfo) {
        LogError("CreatureDisplayInfo %d not found", displayInfoID);
        return NULL;
    }
    
    // Step 2: Get model data reference
    CreatureModelDataRec* modelData = 
        g_CreatureModelDataCache.Lookup(displayInfo->ModelID);
    
    if (!modelData) {
        LogError("CreatureModelData %d not found", displayInfo->ModelID);
        return NULL;
    }
    
    // Step 3: Load MDX model file
    const char* modelPath = GetDBCString(modelData->ModelPath);
    CModelComplex* model = LoadMDXFile(modelPath);
    
    if (!model) {
        LogError("Failed to load model: %s", modelPath);
        return NULL;
    }
    
    // Step 4: Apply texture overrides from DBC
    const char* skin1 = GetDBCString(displayInfo->Skin1);
    const char* skin2 = GetDBCString(displayInfo->Skin2);
    const char* skin3 = GetDBCString(displayInfo->Skin3);
    
    // Apply skin overrides if specified
    if (skin1 && skin1[0]) {
        OverrideModelTexture(model, 0, skin1);
    }
    if (skin2 && skin2[0]) {
        OverrideModelTexture(model, 1, skin2);
    }
    if (skin3 && skin3[0]) {
        OverrideModelTexture(model, 2, skin3);
    }
    
    // Step 5: Apply scale from DBC
    model->scale = displayInfo->Scale;
    
    return model;
}

/**
 * Override a specific texture slot in a model
 */
void OverrideModelTexture(CModelComplex* model, uint32_t slot, 
                         const char* texturePath) {
    if (slot >= model->m_textures.count) {
        LogWarning("Texture slot %d out of range", slot);
        return;
    }
    
    // Close existing texture if present
    if (model->m_textures.data[slot].handle) {
        HandleClose(model->m_textures.data[slot].handle);
    }
    
    // Load new texture
    CGxTexFlags flags;
    flags.filter = GxTex_LinearMipLinear;
    flags.wrapU = GxTex_Repeat;
    flags.wrapV = GxTex_Repeat;
    
    CStatus status;
    model->m_textures.data[slot].handle = 
        TextureCreate(texturePath, flags, &status, 0);
    
    if (!model->m_textures.data[slot].handle) {
        LogError("Failed to load texture: %s", texturePath);
        // Use fallback
        CImVector magenta = 0xFFFF00FF;
        model->m_textures.data[slot].handle = 
            TextureCreateSolid(&magenta, &status);
    }
}
```

## Example 2: Character Customization

Apply character skin, face, and hair textures from CharSections.dbc.

```c
/**
 * Apply character customization textures
 */
void ApplyCharacterCustomization(CModelComplex* characterModel,
                                 uint32_t race, uint32_t gender,
                                 uint32_t skinColor, uint32_t face,
                                 uint32_t hairStyle, uint32_t hairColor,
                                 uint32_t facialHair) {
    // Load base skin texture
    CharSectionsRec* skinSection = FindCharSection(
        race, gender, 
        SECTION_SKIN, skinColor, 0
    );
    
    if (skinSection) {
        const char* skinTex = GetDBCString(skinSection->TextureName1);
        ApplyCharacterTexture(characterModel, TEXSLOT_BODY, skinTex);
    }
    
    // Load face texture
    CharSectionsRec* faceSection = FindCharSection(
        race, gender,
        SECTION_FACE, face, skinColor
    );
    
    if (faceSection) {
        const char* faceTex1 = GetDBCString(faceSection->TextureName1);
        const char* faceTex2 = GetDBCString(faceSection->TextureName2);
        
        ApplyCharacterTexture(characterModel, TEXSLOT_FACE_LOWER, faceTex1);
        ApplyCharacterTexture(characterModel, TEXSLOT_FACE_UPPER, faceTex2);
    }
    
    // Load hair texture
    CharSectionsRec* hairSection = FindCharSection(
        race, gender,
        SECTION_HAIR, hairStyle, hairColor
    );
    
    if (hairSection) {
        const char* hairTex = GetDBCString(hairSection->TextureName1);
        ApplyCharacterTexture(characterModel, TEXSLOT_HAIR, hairTex);
    }
    
    // Load facial hair texture (if male and selected)
    if (gender == GENDER_MALE && facialHair > 0) {
        CharSectionsRec* facialHairSection = FindCharSection(
            race, gender,
            SECTION_FACIALHAIR_LOWER, facialHair, hairColor
        );
        
        if (facialHairSection) {
            const char* facialTex = GetDBCString(
                facialHairSection->TextureName1
            );
            ApplyCharacterTexture(characterModel, 
                                TEXSLOT_FACIAL_HAIR, facialTex);
        }
    }
}

/**
 * Find a CharSections.dbc record matching criteria
 */
CharSectionsRec* FindCharSection(uint32_t race, uint32_t gender,
                                uint32_t section, uint32_t variation,
                                uint32_t color) {
    // Iterate through CharSections.dbc records
    for (uint32_t i = 0; i < g_CharSectionsCount; i++) {
        CharSectionsRec* rec = &g_CharSections[i];
        
        if (rec->RaceID == race &&
            rec->SexID == gender &&
            rec->BaseSection == section &&
            rec->VariationIndex == variation &&
            rec->ColorIndex == color) {
            return rec;
        }
    }
    
    return NULL;
}

/**
 * Apply texture to specific character slot
 */
void ApplyCharacterTexture(CModelComplex* model, uint32_t slot,
                           const char* texturePath) {
    if (!texturePath || !texturePath[0]) return;
    
    CGxTexFlags flags;
    flags.filter = GxTex_LinearMipLinear;
    flags.wrapU = GxTex_Clamp;  // Character textures use clamp
    flags.wrapV = GxTex_Clamp;
    
    CStatus status;
    HTEXTURE__* texture = TextureCreate(texturePath, flags, &status, 0);
    
    if (texture) {
        // Close old texture
        if (model->m_textures.data[slot].handle) {
            HandleClose(model->m_textures.data[slot].handle);
        }
        
        model->m_textures.data[slot].handle = texture;
    } else {
        LogError("Failed to load character texture: %s", texturePath);
    }
}
```

## Example 3: Item Display on Character

Apply equipped item models and textures to character.

```c
/**
 * Equip an item on a character model
 */
void EquipItem(CModelComplex* characterModel, uint32_t itemID,
              uint32_t inventorySlot) {
    // Lookup item in database
    ItemRec* item = g_ItemCache.Lookup(itemID);
    if (!item) return;
    
    // Get display info
    ItemDisplayInfoRec* displayInfo = 
        g_ItemDisplayInfoCache.Lookup(item->DisplayInfoID);
    if (!displayInfo) return;
    
    // Determine which model to use based on slot
    const char* modelPath = NULL;
    const char* texturePath = NULL;
    
    switch (inventorySlot) {
        case INVSLOT_HEAD:
        case INVSLOT_CHEST:
        case INVSLOT_LEGS:
        case INVSLOT_FEET:
        case INVSLOT_HANDS:
        case INVSLOT_WRIST:
        case INVSLOT_SHOULDER:
            // For armor, texture is baked onto character model
            texturePath = GetDBCString(displayInfo->ModelTexture1);
            ApplyArmorTexture(characterModel, inventorySlot, texturePath);
            break;
            
        case INVSLOT_MAINHAND:
        case INVSLOT_OFFHAND:
            // For weapons, load separate model
            modelPath = GetDBCString(displayInfo->ModelName1);
            texturePath = GetDBCString(displayInfo->ModelTexture1);
            AttachWeaponModel(characterModel, inventorySlot, 
                            modelPath, texturePath);
            break;
    }
}

/**
 * Apply armor texture to character model
 */
void ApplyArmorTexture(CModelComplex* model, uint32_t slot,
                      const char* texturePath) {
    if (!texturePath || !texturePath[0]) return;
    
    // Determine which texture slot to use based on armor slot
    uint32_t texSlot;
    switch (slot) {
        case INVSLOT_CHEST:  texSlot = TEXSLOT_ARMOR_CHEST; break;
        case INVSLOT_LEGS:   texSlot = TEXSLOT_ARMOR_LEGS; break;
        case INVSLOT_HANDS:  texSlot = TEXSLOT_ARMOR_HANDS; break;
        case INVSLOT_FEET:   texSlot = TEXSLOT_ARMOR_FEET; break;
        default: return;
    }
    
    // Load and apply texture
    CGxTexFlags flags;
    flags.filter = GxTex_LinearMipLinear;
    flags.wrapU = GxTex_Clamp;
    flags.wrapV = GxTex_Clamp;
    
    CStatus status;
    HTEXTURE__* texture = TextureCreate(texturePath, flags, &status, 0);
    
    if (texture) {
        if (model->m_textures.data[texSlot].handle) {
            HandleClose(model->m_textures.data[texSlot].handle);
        }
        model->m_textures.data[texSlot].handle = texture;
    }
}

/**
 * Attach weapon model to character
 */
void AttachWeaponModel(CModelComplex* characterModel, uint32_t slot,
                      const char* modelPath, const char* texturePath) {
    // Load weapon model
    CModelComplex* weaponModel = LoadMDXFile(modelPath);
    if (!weaponModel) return;
    
    // Override texture if specified
    if (texturePath && texturePath[0]) {
        OverrideModelTexture(weaponModel, 0, texturePath);
    }
    
    // Attach to appropriate bone
    const char* attachBone = (slot == INVSLOT_MAINHAND) ? 
                            "Bone_Weapon_R" : "Bone_Weapon_L";
    
    AttachModelToBone(characterModel, weaponModel, attachBone);
}
```

## Example 4: Asynchronous Texture Loading

Load textures in background without blocking main thread.

```c
/**
 * Queue multiple textures for async loading
 */
void PreloadTextures(const char** texturePaths, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        AsyncTextureLoadRequest request;
        request.path = texturePaths[i];
        request.priority = PRIORITY_LOW;
        request.callback = OnTextureLoaded;
        request.userData = NULL;
        
        QueueAsyncTextureLoad(&request);
    }
}

/**
 * Callback when async texture finishes loading
 */
void OnTextureLoaded(HTEXTURE__* texture, void* userData, 
                    bool success) {
    if (success) {
        LogInfo("Texture loaded: %p", texture);
        
        // Add to cache if not already present
        AddToCacheIfNeeded(texture);
    } else {
        LogError("Failed to load texture");
    }
}

/**
 * Wait for specific textures to finish loading
 */
bool WaitForTextures(HTEXTURE__** textures, uint32_t count,
                    uint32_t timeoutMs) {
    uint32_t startTime = GetTickCount();
    
    while (GetTickCount() - startTime < timeoutMs) {
        bool allLoaded = true;
        
        for (uint32_t i = 0; i < count; i++) {
            if (!IsTextureLoaded(textures[i])) {
                allLoaded = false;
                break;
            }
        }
        
        if (allLoaded) return true;
        
        // Process async callbacks
        AsyncTextureHandler();
        
        // Small sleep to avoid busy waiting
        Sleep(1);
    }
    
    return false;  // Timeout
}
```

## Example 5: Texture Cache Management

Efficient texture caching to avoid redundant loads.

```c
/**
 * Enhanced texture loading with caching
 */
HTEXTURE__* LoadTextureWithCache(const char* path) {
    // Normalize path
    char normalizedPath[MAX_PATH];
    NormalizePath(path, normalizedPath);
    
    // Check cache first
    HTEXTURE__* cached = FindInCache(normalizedPath);
    if (cached) {
        IncrementRefCount(cached);
        return cached;
    }
    
    // Load from file
    CGxTexFlags flags;
    flags.filter = GxTex_LinearMipLinear;
    flags.wrapU = GxTex_Repeat;
    flags.wrapV = GxTex_Repeat;
    
    CStatus status;
    HTEXTURE__* texture = TextureCreate(normalizedPath, flags, 
                                       &status, 0);
    
    if (texture) {
        AddToCache(normalizedPath, texture);
    }
    
    return texture;
}

/**
 * Clear unused textures from cache
 */
void PurgeUnusedTextures() {
    TSHashTable<UCTextureHash, HASHKEY_TEXTUREFILE>* cache = 
        &g_TextureCache;
    
    // Iterate through all cache entries
    for (uint32_t bucket = 0; bucket < cache->numBuckets; bucket++) {
        UCTextureHash* entry = cache->buckets[bucket];
        UCTextureHash* prev = NULL;
        
        while (entry) {
            UCTextureHash* next = entry->next;
            
            // Remove if reference count is 1 (only cache holds it)
            if (entry->refCount == 1) {
                LogInfo("Purging unused texture: %s", entry->path);
                
                // Close handle
                HandleClose(entry->handle);
                
                // Remove from linked list
                if (prev) {
                    prev->next = next;
                } else {
                    cache->buckets[bucket] = next;
                }
                
                // Free entry
                free(entry);
            } else {
                prev = entry;
            }
            
            entry = next;
        }
    }
}

/**
 * Get cache statistics
 */
void GetCacheStats(uint32_t* outNumTextures, uint32_t* outTotalMemory) {
    uint32_t count = 0;
    uint32_t memory = 0;
    
    TSHashTable<UCTextureHash, HASHKEY_TEXTUREFILE>* cache = 
        &g_TextureCache;
    
    for (uint32_t bucket = 0; bucket < cache->numBuckets; bucket++) {
        UCTextureHash* entry = cache->buckets[bucket];
        
        while (entry) {
            count++;
            memory += GetTextureMemorySize(entry->handle);
            entry = entry->next;
        }
    }
    
    *outNumTextures = count;
    *outTotalMemory = memory;
}
```

## Example 6: Handling Missing Textures

Graceful fallback when textures are missing or fail to load.

```c
/**
 * Load texture with fallback options
 */
HTEXTURE__* LoadTextureWithFallback(const char* path) {
    CGxTexFlags flags;
    flags.filter = GxTex_LinearMipLinear;
    flags.wrapU = GxTex_Repeat;
    flags.wrapV = GxTex_Repeat;
    
    CStatus status;
    
    // Try primary path
    HTEXTURE__* texture = TextureCreate(path, flags, &status, 0);
    if (texture) return texture;
    
    // Try with lowercase
    char lowerPath[MAX_PATH];
    strcpy(lowerPath, path);
    _strlwr(lowerPath);
    
    texture = TextureCreate(lowerPath, flags, &status, 0);
    if (texture) return texture;
    
    // Try replacing extension
    char* ext = strrchr(lowerPath, '.');
    if (ext) {
        strcpy(ext, ".tga");  // Try TGA instead of BLP
        texture = TextureCreate(lowerPath, flags, &status, 0);
        if (texture) return texture;
    }
    
    // Give up - use fallback
    LogWarning("Texture not found, using fallback: %s", path);
    return CreateFallbackTexture();
}

/**
 * Create a recognizable fallback texture
 */
HTEXTURE__* CreateFallbackTexture() {
    // Create 2x2 checkerboard pattern (magenta and black)
    static HTEXTURE__* cachedFallback = NULL;
    
    if (!cachedFallback) {
        uint32_t pixels[4] = {
            0xFFFF00FF,  // Magenta
            0xFF000000,  // Black
            0xFF000000,  // Black
            0xFFFF00FF   // Magenta
        };
        
        cachedFallback = CreateTextureFromPixels(2, 2, pixels);
    }
    
    IncrementRefCount(cachedFallback);
    return cachedFallback;
}
```

## Example 7: Dynamic Texture Swapping

Change textures at runtime for visual effects.

```c
/**
 * Swap texture for visual effect (e.g., seasonal changes)
 */
void SwapModelTexture(CModelComplex* model, uint32_t slot,
                     const char* newTexturePath) {
    if (slot >= model->m_textures.count) return;
    
    // Store old texture
    HTEXTURE__* oldTexture = model->m_textures.data[slot].handle;
    
    // Load new texture
    CGxTexFlags flags;
    flags.filter = GxTex_LinearMipLinear;
    flags.wrapU = GxTex_Repeat;
    flags.wrapV = GxTex_Repeat;
    
    CStatus status;
    HTEXTURE__* newTexture = TextureCreate(newTexturePath, flags, 
                                          &status, 0);
    
    if (newTexture) {
        // Apply new texture
        model->m_textures.data[slot].handle = newTexture;
        
        // Release old texture
        if (oldTexture) {
            HandleClose(oldTexture);
        }
    } else {
        LogError("Failed to swap texture: %s", newTexturePath);
    }
}

/**
 * Animate texture by cycling through variations
 */
void AnimateTexture(CModelComplex* model, uint32_t slot,
                   const char* baseTexture, uint32_t frameCount,
                   float fps) {
    static float timer = 0.0f;
    static uint32_t currentFrame = 0;
    
    float deltaTime = GetDeltaTime();
    timer += deltaTime;
    
    float frameTime = 1.0f / fps;
    if (timer >= frameTime) {
        timer -= frameTime;
        currentFrame = (currentFrame + 1) % frameCount;
        
        // Build frame texture path
        char framePath[MAX_PATH];
        sprintf(framePath, "%s_%02d.blp", baseTexture, currentFrame);
        
        // Swap to frame texture
        SwapModelTexture(model, slot, framePath);
    }
}
```

## Example 8: Batch Texture PreLoading

Efficiently pre-load multiple textures for smooth gameplay.

```c
/**
 * Preload all textures for a zone
 */
void PreloadZoneTextures(uint32_t zoneID) {
    // Get list of creatures in zone
    uint32_t* creatureIDs = GetCreaturesInZone(zoneID);
    uint32_t creatureCount = GetCreatureCount(zoneID);
    
    // Collect all unique texture paths
    char** texturePaths = malloc(sizeof(char*) * 1000);
    uint32_t textureCount = 0;
    
    for (uint32_t i = 0; i < creatureCount; i++) {
        CreatureDisplayInfoRec* displayInfo = 
            LookupCreatureDisplayInfo(creatureIDs[i]);
        
        if (displayInfo) {
            AddUniqueTexture(texturePaths, &textureCount,
                           GetDBCString(displayInfo->Skin1));
            AddUniqueTexture(texturePaths, &textureCount,
                           GetDBCString(displayInfo->Skin2));
            AddUniqueTexture(texturePaths, &textureCount,
                           GetDBCString(displayInfo->Skin3));
        }
    }
    
    // Preload all textures asynchronously
    PreloadTextures((const char**)texturePaths, textureCount);
    
    // Cleanup
    for (uint32_t i = 0; i < textureCount; i++) {
        free(texturePaths[i]);
    }
    free(texturePaths);
    free(creatureIDs);
}

/**
 * Add texture path to list if not already present
 */
void AddUniqueTexture(char** list, uint32_t* count, const char* path) {
    if (!path || !path[0]) return;
    
    // Check if already in list
    for (uint32_t i = 0; i < *count; i++) {
        if (strcmp(list[i], path) == 0) {
            return;  // Already present
        }
    }
    
    // Add to list
    list[*count] = _strdup(path);
    (*count)++;
}
```

---

**Next**: [Troubleshooting Guide](13_troubleshooting.md)
