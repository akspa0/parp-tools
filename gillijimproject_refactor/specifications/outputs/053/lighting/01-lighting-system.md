# WoW Alpha 0.5.3 Lighting System Analysis

## Overview

Analysis of WoW Alpha 0.5.3 (Build 3368) binary via Ghidra reveals the lighting system uses **`.lit`** files for area lighting and fog, with per-object lights handled through the rendering pipeline.

## Key Findings

### 1. `.lit` File Format - Area Lighting

#### `LoadLightsAndFog` Function (0x006c4110)

**File Loading:**
```cpp
bool __fastcall LoadLightsAndFog(char *filename, LightGroup *lightGroup)
{
    // Open .lit file
    SFile::Open(filename, &lightdata);
    
    // Read version magic
    SFile::Read(lightdata, &versionNumber, 4, ...);
    // Version: 0x80000004 (decimal: -2147483644)
    // This matches the spec: "Lights version 0x4"
    
    // Read light count
    SFile::Read(lightdata, &lightCount, 4, ...);
    
    // Allocate LightData array
    // Each LightData is 0x560 bytes
    TSFixedArray<LightData>::ReallocData(lightGroup, lightCount);
    
    // Read main light data (0x40 bytes per light)
    SFile::Read(lightdata, lightData, 0x40, ...);
    
    // Read 4 light groups per light at different offsets
    ReadSingleLightGroup(lightdata, lightData + 0x40);   // Group 1
    ReadSingleLightGroup(lightdata, lightData + 0x188);  // Group 2
    ReadSingleLightGroup(lightdata, lightData + 0x2d0);  // Group 3
    ReadSingleLightGroup(lightdata, lightData + 0x418);  // Group 4
}
```

**LightData Structure (0x560 bytes):**
```
Offset  Size  Field
0x00    4     Type/flags
0x04    4     Light ID
0x08    12    Position (C3Vector: X, Y, Z)
0x14    12    Color (C3Vector: R, G, B)
0x20    4     Intensity/Falloff
0x24    12    Group 1 data (light settings)
0x188   12    Group 2 data
0x2d0   12    Group 3 data
0x418   12    Group 4 data
...     ?     Padding/unknown
```

### 2. Per-Object Lights (MDX/WMO)

#### `CMapLight` Class
```cpp
class CMapLight {
    CMapBaseObj base;           // Base object
    CGxLight gxLight;           // Direct3D light
    float attenDenom;           // Attenuation denominator
    float attenEnd;             // Attenuation end distance
    float attenStart;           // Attenuation start distance
    uint flags;                 // 0x80 = enabled?
};
```

#### `CGxLight` Structure
```cpp
class CGxLight {
    // D3D8 light parameters:
    uint type;                  // D3DLIGHT_POINT, D3DLIGHT_SPOT, etc.
    C3Vector position;           // Light position
    C3Vector direction;         // Light direction (for spotlights)
    float range;                // Light range
    float falloff;              // Falloff factor
    float attenuation0;          // Constant attenuation
    float attenuation1;         // Linear attenuation
    float attenuation2;         // Quadratic attenuation
    float theta;                // Spotlight inner angle
    float phi;                  // Spotlight outer angle
    CArgb diffuse;              // Diffuse color
    CArgb specular;             // Specular color
    CArgb ambient;              // Ambient color
};
```

### 3. Fog Implementation

#### Fog Query Functions

**`CWorld::QueryMapObjFog`** (0x00663950 / 0x006896d0):
```cpp
int __fastcall CWorld::QueryMapObjFog(
    ulong objectId,     // 0 = camera/world fog
    Fogs &fog,          // Output fog parameters
    float &distance     // Output fog distance
) {
    if (objectId == 0) {
        // Camera/world level fog
        return CMapEntity::QueryCameraFog(fog, distance);
    }
    
    // Check if object has custom fog
    if (object->flags & 0x01) {
        return CMapEntity::QueryMapObjFog(object, fog, distance);
    }
    
    return 0;  // No custom fog
}
```

#### Fog Parameters Structure
```cpp
struct Fog {
    uint flags;           // Fog enable flags
    float start;          // Fog start distance
    float end;            // Fog end distance
    float density;        // Fog density
    CArgb color;         // Fog color (RGBA)
    uint fogType;         // 0 = none, 1 = linear, 2 = exponential, etc.
};
```

### 4. Light Selection Callback

**`SelectLight` Callback:**
```cpp
void __fastcall SelectLight(
    HMODEL__ *model,      // The model
    CMapDoodadDef *doodad, // Doodad placement
    int callbackType      // Callback type
) {
    // Called when model needs to select which lights affect it
    // Determines which lights from LightGroup affect this object
    // Uses distance from light to object center
}
```

## Fog Rendering Pipeline

### 1. Camera Fog Setup

**`CMapEntity::QueryCameraFog`**:
- Gets world/camera fog settings
- Applies based on current camera position
- Sets D3D render states

### 2. Per-Object Fog

**`CSimpleModel_SetFogColor`** (0x00774f80):
- Sets per-model fog color
- Used for special effects or indoor areas

**`CSimpleModel_SetFogNear/Far`** (0x00775210 / 0x00775160):
- Sets per-model fog start/end distances
- Overrides world fog for this model

### 3. Fog Blending

**`ComputeFogBlend`** (0x00689b40):
```cpp
void __fastcall ComputeFogBlend(
    float fragmentDepth,   // Depth of current fragment
    Fog &fog,              // Fog parameters
    float &alpha,          // Output blend alpha
    CArgb &color           // Output blended color
) {
    // Calculate fog factor based on distance
    // Linear fog: (end - z) / (end - start)
    // Exponential fog: e^(-density * z)
    // Apply to fragment color
}
```

## Light Types

The client supports D3D8 light types:
- `D3DLIGHT_POINT` - Point light (omnidirectional)
- `D3DLIGHT_SPOT` - Spotlight (directional with cone)
- `D3DLIGHT_DIRECTIONAL` - Directional light (sun/moon)

## Related Functions

| Function | Address | Purpose |
|----------|---------|---------|
| `LoadLightsAndFog` | 0x006c4110 | Load .lit file |
| `ReadSingleLightGroup` | - | Read light group data |
| `CMapLight::CMapLight` | 0x00686570 | Initialize light object |
| `QueryMapObjFog` | 0x00663950 | Get object fog |
| `QueryCameraFog` | - | Get camera fog |
| `ComputeFogBlend` | 0x00689b40 | Calculate fog blend |
| `SetFogColors` | 0x006bc780 | Set fog colors |
| `SelectLight` | - | Light selection callback |
| `CalcLightColors` | 0x006c4da0 | Calculate light colors |

## File References

- **`.lit`** - Area lighting/fog data (loaded per zone/area)
- **`.mdx`** - Model with embedded material lighting
- **`.wmo`** - World Map Object with lighting groups
- **MCSE/MCLQ** - Terrain chunk with ground effect lighting

## Observations

1. **No `light.dbc` found** - Alpha 0.5.3 doesn't use database files for lights
2. **Area-based lighting** - .lit files define lighting per zone/area
3. **Multiple light groups** - Each light has 4 groups (possibly for time-of-day)
4. **Fog integration** - Fog is part of .lit file, not separate
5. **Per-model fog** - Models can override world fog settings

## Next Steps

- [ ] Analyze `ReadSingleLightGroup` for light group format
- [ ] Document LightData structure completely
- [ ] Analyze fog rendering in shader pipeline
- [ ] Compare with modern WoW lighting system
