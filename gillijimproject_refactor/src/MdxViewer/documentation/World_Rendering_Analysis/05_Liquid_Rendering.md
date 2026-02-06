# Liquid Rendering System

## Preparation Function

**Function:** `CWorldScene::PrepareRenderLiquid` @ 0x0066a590

### Purpose

Prepare liquid rendering (water, lava, slime) for the current scene.

### Key Operations

1. Query liquid status at camera position.
2. Set up particles for water/magma effects.
3. Update liquid textures.

### Algorithm

```c
void PrepareRenderLiquid() {
    // Get camera position
    C3Vector camQueryPos = camPos;
    
    // Find minimum Z level (from a static table of heights)
    for (int i = 0; i < 0x25; i++) {
        if (DAT_00e40b40[i] <= camQueryPos.z) {
            camQueryPos.z = DAT_00e40b40[i];
        }
    }
    
    // Query liquid status
    uint newLiquid = 0xf;
    C3Vector lqDir;
    float liquidLevel;
    
    // Check for MapObject-based liquids first (e.g., inside WMOs)
    if (camMapObjDef == NULL || camMapObj == NULL || camMapObjGroup == NULL) {
        CWorld::QueryLiquidStatus(&camQueryPos, &newLiquid, &liquidLevel, &lqDir);
    } else {
        C3Vector localPos = camQueryPos * camMapObjDef->invMat;
        CMapObjGroup::QueryLiquidStatus(camMapObjGroup, &localPos, &newLiquid, &liquidLevel, &lqDir);
    }
    
    // Check if liquid type changed
    bool liquidChanged = false;
    if (newLiquid == 0xf) {
        if (camLiquid != 0xf) {
            liquidChanged = true;
        }
    } else {
        // Set up particles based on liquid type
        switch (newLiquid & 3) {
            case LIQUID_WATER:
            case LIQUID_OCEAN:
                if (newLiquid != camLiquid && (CWorld::enables & 0x2000000) != 0) {
                    Particulate::InitParticles(CWorld::particulate, newLiquid);
                    Particulate::SetScale(CWorld::particulate, 0.027777778);
                    CWorld::particulate->show = true;
                }
                break;
            case LIQUID_MAGMA:
                if (newLiquid != camLiquid && (CWorld::enables & 0x2000000) != 0) {
                    Particulate::InitParticles(CWorld::particulate, newLiquid);
                    Particulate::SetScale(CWorld::particulate, 0.11111111);
                    CWorld::particulate->show = true;
                }
                break;
            case LIQUID_SLIME:
                CWorld::particulate->show = false;
                break;
        }
    }
    
    // Update current liquid
    camLiquid = newLiquid;
    
    // Force day/night update if liquid changed
    if (liquidChanged) {
        DayNightForceFullUpdate();
    }
    
    // Reset texture update flags
    CMap::riverDiffTexUpdated = false;
    CMap::oceanDiffTexUpdated = false;
}
```

## Liquid Status Query

**Function:** `CMap::QueryLiquidStatus` @ 0x00664e70

### Purpose

Query liquid status at a given world position.

### Parameters

```c
void QueryLiquidStatus(
    C3Vector* position,    // Position to query
    uint* liquidType,       // Output liquid type (0=Water, 1=Ocean, 2=Magma, 3=Slime, 15=None)
    float* liquidLevel,     // Output liquid surface Z level
    C3Vector* liquidDir     // Output liquid flow direction
);
```

### Liquid Properties

```c
struct LiquidProperties {
    uint type;              // Liquid type
    float level;            // Liquid level (Z position)
    C3Vector direction;     // Liquid flow direction
    float speed;            // Liquid flow speed
    float opacity;          // Liquid opacity
    C3Color color;          // Liquid color
};
```

## Particle Effects

### Water Particles
- **Scale:** 0.027777778 (1/36)
- **Purpose:** Small water droplets/spray.
- **Trigger:** When player enters water.

### Magma Particles
- **Scale:** 0.11111111 (1/9)
- **Purpose:** Magma bubbles and heat distortion.
- **Trigger:** When player enters magma.

### Slime Particles
- **Scale:** None
- **Purpose:** No visual particles.

## Liquid Textures

### Texture Types

```c
enum LiquidTextureType {
    LIQUID_TEXTURE_DIFFUSE,     // Diffuse texture
    LIQUID_TEXTURE_NORMAL,      // Normal map
    LIQUID_TEXTURE_SPECULAR,    // Specular map
    LIQUID_TEXTURE_ENVIRONMENT, // Environment map
};
```

## Implementation Guidelines

### C# Liquid Rendering

```csharp
public class LiquidRenderer
{
    private enum LiquidType : uint
    {
        Water = 0x0,
        Ocean = 0x1,
        Magma = 0x2,
        Slime = 0x3,
        None = 0xf
    }
    
    private LiquidType currentLiquid = LiquidType.None;
    
    public void Update(C3Vector cameraPosition)
    {
        LiquidType newLiquid;
        float level;
        C3Vector dir;
        
        // CMap.QueryLiquidStatus logic
        QueryLiquidStatus(cameraPosition, out newLiquid, out level, out dir);
        
        if (newLiquid != currentLiquid)
        {
            // Update particles and lighting
            UpdateLiquidEffects(newLiquid);
            currentLiquid = newLiquid;
        }
    }
}
```

## References

- `CWorldScene::PrepareRenderLiquid` (0x0066a590)
- `CMap::QueryLiquidStatus` (0x00664e70)
- `CMap::QueryLiquidSounds` (0x00664e90)
- `CMap::QueryLiquidFishable` (0x00688060)