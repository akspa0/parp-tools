# WoW Alpha 0.5.3 (Build 3368) SkyBox and Sky Rendering Analysis

## Overview

This document provides a deep analysis of the sky rendering system in WoW Alpha 0.5.3 (Build 3368, Dec 11 2003), based on Ghidra reverse engineering of WoWClient.exe. It covers skyboxes, sky dome rendering, weather effects, DBC integration, and atmospheric effects.

## Related Functions

| Function | Address | Purpose |
|----------|---------|---------|
| [`LoadLightsAndFog`](LoadLightsAndFog) | 0x006c4110 | Load light and fog data |
| [`SetFogColors`](SetFogColors) | 0x006bc780 | Set fog colors |
| [`GetFog`](GetFog) | 0x0068b1c0 | Get fog parameters |
| [`ComputeFogBlend`](ComputeFogBlend) | 0x00689b40 | Compute fog blend factor |
| [`QueryCameraFog`](QueryCameraFog) | 0x00689bf0 | Query camera fog |
| [`QueryMapObjFog`](QueryMapObjFog) | 0x006896d0 | Query map object fog |
| [`CSimpleModel_SetFogColor`](CSimpleModel_SetFogColor) | 0x00774f80 | Set model fog color |
| [`CSimpleModel_SetFogFar`](CSimpleModel_SetFogFar) | 0x00775210 | Set fog far distance |
| [`CSimpleModel_SetFogNear`](CSimpleModel_SetFogNear) | 0x00775160 | Set fog near distance |

---

## Sky Rendering Architecture

### Sky Components

The sky system consists of multiple layers:

```
Sky Rendering Layers:
├── Skybox (Cube map or single texture)
├── Sky dome (Geometric dome)
├── Sun/Moon (Positional light sources)
├── Fog (Atmospheric scattering)
├── Clouds (Cloud layers)
└── Weather (Rain, snow effects)
```

### Sky DBC Integration

Sky settings are controlled by DBC files:

| DBC File | Purpose |
|----------|---------|
| Skybox.dbc | Skybox textures and settings |
| GroundEffectTexture.dbc | Ground effects |
| Weather.dbc | Weather effects |
| Light.dbc | Light source definitions |

---

## Fog System (Sky Atmosphere)

### Fog Data Structure

```c
/* From LoadLightsAndFog at 0x006c4110 */
struct LightDataFog {
    float color[4];          // RGBA fog color
    float start;             // Fog start distance
    float end;               // Fog end distance
    float density;           // Fog density (exp/exp2)
    uint32_t type;          // Fog type
};

struct LightData {
    uint32_t numLights;      // Number of lights
    uint32_t numFogs;        // Number of fog entries
    LightDataLight lights[];
    LightDataFog fogs[];
};
```

### Fog Types

| Type | Value | Formula |
|------|-------|---------|
| None | 0 | No fog |
| Linear | 1 | `(end - dist) / (end - start)` |
| Exponential | 2 | `e^(-density * dist)` |
| Exponential2 | 3 | `e^(-density² * dist²)` |

### Fog Implementation

```c
/* GetFog at 0x0068b1c0 */
void __fastcall GetFog(float distance, FogData* fog) {
    uint32_t fogIndex = GetCurrentFogIndex();
    FogData* currentFog = &fogData[fogIndex];
    
    // Copy fog parameters
    fog->color[0] = currentFog->color[0];
    fog->color[1] = currentFog->color[1];
    fog->color[2] = currentFog->color[2];
    fog->color[3] = currentFog->color[3];
    fog->start = currentFog->start;
    fog->end = currentFog->end;
    fog->density = currentFog->density;
    fog->type = currentFog->type;
}

/* ComputeFogBlend at 0x00689b40 */
float __fastcall ComputeFogBlend(float distance, FogData* fog) {
    float blendFactor;
    
    switch (fog->type) {
        case FOG_LINEAR:
            blendFactor = (fog->end - distance) / (fog->end - fog->start);
            break;
            
        case FOG_EXP:
            blendFactor = exp(-fog->density * distance);
            break;
            
        case FOG_EXP2:
            blendFactor = exp(-fog->density * fog->density * distance * distance);
            break;
            
        default:
            blendFactor = 1.0f;
            break;
    }
    
    // Clamp to [0, 1]
    return max(0.0f, min(1.0f, blendFactor));
}
```

### Fog Color Application

```c
/* SetFogColors at 0x006bc780 */
void __fastcall SetFogColors(float r, float g, float b) {
    FogData* currentFog = GetActiveFog();
    
    currentFog->color[0] = r;
    currentFog->color[1] = g;
    currentFog->color[2] = b;
    currentFog->color[3] = 1.0f;  // Alpha
    
    // Update all shader uniforms
    UpdateFogUniforms(currentFog);
}
```

---

## Skybox Rendering

### Skybox Structure

```c
struct Skybox {
    CTexture* textures[6];    // Cube map faces
    // OR
    CTexture* texture;       // Single equirectangular texture
    
    float rotation;          // Skybox rotation
    float scale;             // Skybox scale
    bool isRotated;         // Rotation flag
    bool isActive;          // Active flag
};
```

### Skybox Rendering Pipeline

```
Skybox Render Order:
1. Clear depth buffer
2. Render skybox (farthest objects)
3. Enable depth write
4. Render terrain
5. Render objects
6. Apply fog overlay
```

### Skybox Shader

```glsl
// Skybox vertex shader
uniform mat4 viewProjection;
in vec3 position;

void main() {
    vec4 worldPos = modelMatrix * vec4(position, 0.0);
    gl_Position = viewProjection * worldPos;
    texCoord = normalize(position);
}

// Skybox fragment shader
uniform samplerCube skybox;
uniform vec3 fogColor;

void main() {
    vec4 skyColor = texture(skybox, texCoord);
    
    // Apply fog based on distance
    float fogFactor = computeFogFactor(gl_FragCoord.z);
    vec3 finalColor = mix(fogColor, skyColor.rgb, fogFactor);
    
    gl_FragColor = vec4(finalColor, 1.0);
}
```

---

## Sky Dome System

### Dome Geometry

```c
struct SkyDome {
    uint32_t vertexCount;    // Number of vertices
    uint32_t indexCount;      // Number of indices
    C3Vector* vertices;       // Vertex positions
    uint16_t* indices;       // Triangle indices
    C2Vector* uvs;          // Texture coordinates
    C3Vector* normals;      // Vertex normals
};
```

### Dome Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Radius | 1000.0 | Dome radius |
| Segments | 32 | Horizontal segments |
| Rings | 16 | Vertical rings |
| Rotation | 0.0 | Y-axis rotation |

---

## Sun and Moon Rendering

### Celestial Bodies

```c
struct CelestialBody {
    uint32_t id;             // Body ID
    CTexture* texture;       // Body texture
    float position[3];       // Position (X, Y, Z)
    float rotation[3];       // Rotation (pitch, yaw, roll)
    float scale;            // Texture scale
    float intensity;        // Light intensity
    bool isEnabled;         // Visible flag
};

struct Sun : CelestialBody {
    // Sun-specific
    float coronaScale;       // Corona size
    float haloScale;         // Halo size
};

struct Moon : CelestialBody {
    // Moon-specific
    uint32_t phase;          // Moon phase (0-7)
    uint32_t textureId;     // Moon texture ID
};
```

### Celestial Positioning

```c
void UpdateCelestialPositions(float timeOfDay) {
    // Time: 0.0 = midnight, 0.5 = noon, 1.0 = midnight
    
    float sunAngle = timeOfDay * 2.0 * PI;
    float moonAngle = sunAngle + PI;
    
    // Sun position
    sun.position[0] = cos(sunAngle) * SUN_DISTANCE;
    sun.position[1] = sin(sunAngle) * SUN_DISTANCE;
    sun.position[2] = 0.0;
    
    // Moon position (opposite to sun)
    moon.position[0] = cos(moonAngle) * MOON_DISTANCE;
    moon.position[1] = sin(moonAngle) * MOON_DISTANCE;
    moon.position[2] = 0.0;
    
    // Update light directions
    UpdateSunLightDirection();
    UpdateMoonLightDirection();
}
```

---

## Weather System

### Weather DBC Integration

```c
/* Weather entry from Weather.dbc */
struct WeatherRec {
    uint32_t id;             // Weather ID
    uint32_t areaId;         // Area ID (0 = all areas)
    uint32_t weatherType;    // Type: Clear, Rain, Snow, Sandstorm
    uint32_t unk1;           // Unknown
    float intensity;         // Effect intensity (0-1)
    float unk2;              // Unknown
    uint32_t soundId;       // Associated sound
    uint32_t particleId;    // Particle system ID
    uint32_t windowId;      // Effect window
    uint32_t fogId;         // Fog override ID
};
```

### Weather Types

| Type | Value | Visual |
|------|-------|--------|
| Clear | 0 | No precipitation |
| Rain | 1 | Rain droplets |
| Snow | 2 | Snowflakes |
| Storm | 3 | Lightning + rain |
| Sandstorm | 4 | Sand particles |

### Weather Effects

```c
struct WeatherSystem {
    uint32_t currentWeather;     // Current weather ID
    uint32_t targetWeather;      // Target weather ID
    float transitionProgress;    // 0.0 to 1.0
    float intensity;             // Current intensity
    
    // Particle system
    ParticleSystem* particles;
    
    // Audio
    uint32_t soundId;
    bool soundPlaying;
};

void UpdateWeather(float deltaTime) {
    // Smooth transition
    if (currentWeather != targetWeather) {
        transitionProgress += deltaTime / TRANSITION_TIME;
        
        if (transitionProgress >= 1.0) {
            currentWeather = targetWeather;
            transitionProgress = 1.0;
        }
        
        // Interpolate intensity
        intensity = lerp(currentIntensity, targetIntensity, transitionProgress);
    }
    
    // Update particles
    if (weatherSystem.particles != NULL) {
        weatherSystem.particles->SetIntensity(intensity);
        weatherSystem.particles->Update(deltaTime);
    }
}
```

---

## Sky Color System

### Sky Colors by Time of Day

```c
struct SkyColorCurve {
    float time[8];          // Time points (0.0 = midnight)
    Color color[8];         // Corresponding sky colors
    
    // Sample sky color at specific time
    Color Sample(float timeOfDay) {
        // Find segment
        int segment = 0;
        for (int i = 0; i < 7; i++) {
            if (timeOfDay >= time[i] && timeOfDay < time[i+1]) {
                segment = i;
                break;
            }
        }
        
        // Interpolate
        float t = (timeOfDay - time[segment]) / (time[segment+1] - time[segment]);
        return lerp(color[segment], color[segment+1], t);
    }
};
```

### Default Sky Colors

| Time | Sky Color | Sun Color | Ambient |
|------|-----------|------------|---------|
| 0:00 | Black | - | Dark blue |
| 4:00 | Dawn pink | Orange | Dim orange |
| 6:00 | Orange | Yellow | Light orange |
| 8:00 | Light blue | White | Gray |
| 12:00 | Sky blue | White | Full |
| 16:00 | Light blue | Yellow | Dim yellow |
| 18:00 | Orange | Orange | Orange |
| 20:00 | Purple | - | Dark purple |
| 24:00 | Black | - | Dark blue |

---

## Lighting Integration

### Light Data Loading

```c
/* LoadLightsAndFog at 0x006c4110 */
void __fastcall LoadLightsAndFog(char* filename) {
    // Open and parse light/fog data file
    IFFChunk* chunk = OpenIFFFile(filename);
    
    // Load fog entries
    IFFChunk* fogChunk = FindChunk(chunk, 'FOG');
    while (fogChunk != NULL) {
        uint32_t fogId = fogChunk->id;
        
        // Read fog data
        ReadFogData(fogChunk, &lightData.fogs[fogId]);
        
        fogChunk = FindNextChunk(fogChunk, 'FOG');
    }
    
    // Load light entries
    IFFChunk* lightChunk = FindChunk(chunk, 'LGHT');
    while (lightChunk != NULL) {
        uint32_t lightId = lightChunk->id;
        
        // Read light data
        ReadLightData(lightChunk, &lightData.lights[lightId]);
        
        lightChunk = FindNextChunk(lightChunk, 'LGHT');
    }
}
```

### Global Illumination

```c
struct GlobalIllumination {
    CArgb ambientColor;      // Ambient light color
    CArgb diffuseColor;      // Diffuse light color
    CArgb specularColor;     // Specular light color
    float sunDirection[3];   // Sun light direction
    float sunIntensity;      // Sun intensity
    float fogColor[4];      // Fog color (for sky blend)
};

void UpdateGlobalIllumination(float timeOfDay) {
    // Sample sky color for ambient
    Color skyColor = skyColorCurve.Sample(timeOfDay);
    
    // Set ambient from sky
    globalIllumination.ambientColor = skyColor * 0.3f;
    
    // Set fog color from sky
    globalIllumination.fogColor[0] = skyColor.r;
    globalIllumination.fogColor[1] = skyColor.g;
    globalIllumination.fogColor[2] = skyColor.b;
    globalIllumination.fogColor[3] = 1.0f;
    
    // Update sun position
    float sunAngle = timeOfDay * 2.0 * PI - PI/2;
    globalIllumination.sunDirection[0] = cos(sunAngle);
    globalIllumination.sunDirection[1] = sin(sunAngle);
    globalIllumination.sunDirection[2] = 0.0f;
    
    // Intensity based on sun height
    globalIllumination.sunIntensity = max(0.0f, sin(sunAngle));
}
```

---

## Camera Fog Query

### Camera Fog Settings

```c
/* QueryCameraFog at 0x00689bf0 */
void __fastcall QueryCameraFog(Camera* camera, FogSettings* settings) {
    // Get current fog
    FogData* currentFog = GetActiveFog();
    
    // Adjust for camera height (fog gets thinner at altitude)
    float heightFactor = 1.0f - (camera->position.y / MAX_HEIGHT);
    heightFactor = max(0.0f, heightFactor);
    
    settings->start = currentFog->start * heightFactor;
    settings->end = currentFog->end * heightFactor;
    settings->density = currentFog->density * heightFactor;
    
    settings->color[0] = currentFog->color[0];
    settings->color[1] = currentFog->color[1];
    settings->color[2] = currentFog->color[2];
    settings->type = currentFog->type;
}
```

### Map Object Fog

```c
/* QueryMapObjFog at 0x006896d0 */
void __fastcall QueryMapObjFog(MapObj* obj, FogSettings* settings) {
    // Check for area-specific fog
    uint32_t areaId = obj->GetAreaId();
    FogData* areaFog = GetAreaFog(areaId);
    
    if (areaFog != NULL) {
        // Use area fog
        *settings = *areaFog;
    } else {
        // Use global fog
        *settings = *GetActiveFog();
    }
    
    // Adjust for object height
    float heightFactor = 1.0f - (obj->position.y / MAX_HEIGHT);
    settings->start *= heightFactor;
    settings->end *= heightFactor;
    settings->density *= heightFactor;
}
```

---

## Model Fog Control

### Setting Fog on Models

```c
/* CSimpleModel_SetFogColor at 0x00774f80 */
void __fastcall CSimpleModel::SetFogColor(float r, float g, float b) {
    m_fogColor.r = r;
    m_fogColor.g = g;
    m_fogColor.b = b;
    m_fogColor.a = 1.0f;
    
    // Mark for shader update
    m_dirtyFlags |= DIRTY_FOG;
}

/* CSimpleModel_SetFogFar at 0x00775210 */
void __fastcall CSimpleModel::SetFogFar(float far) {
    m_fogFar = far;
    m_dirtyFlags |= DIRTY_FOG;
}

/* CSimpleModel_SetFogNear at 0x00775160 */
void __fastcall CSimpleModel::SetFogNear(float near) {
    m_fogNear = near;
    m_dirtyFlags |= DIRTY_FOG;
}
```

---

## Complete Sky Rendering Pipeline

### Pre-Render

1. **Update time of day**
   ```c
   float timeOfDay = GetCurrentTime() / 86400.0f;
   ```

2. **Update celestial positions**
   ```c
   UpdateCelestialPositions(timeOfDay);
   ```

3. **Sample sky colors**
   ```c
   Color skyColor = skyColorCurve.Sample(timeOfDay);
   ```

4. **Update fog settings**
   ```c
   SetFogColors(skyColor.r, skyColor.g, skyColor.b);
   UpdateFogStartEnd(timeOfDay);
   ```

5. **Update lighting**
   ```c
   UpdateGlobalIllumination(timeOfDay);
   ```

### Render

1. **Clear buffers**
   ```c
   Clear(CLEAR_COLOR | CLEAR_DEPTH, skyColor, 1.0f, 0);
   ```

2. **Render skybox**
   ```c
   RenderSkybox(skybox);
   ```

3. **Render celestial bodies**
   ```c
   RenderSun();
   RenderMoon();
   ```

4. **Render sky dome** (if using dome)
   ```c
   RenderSkyDome();
   ```

5. **Render terrain** (with fog)
   ```c
   RenderTerrain();
   ```

6. **Render objects** (with fog)
   ```c
   RenderObjects();
   ```

### Post-Render

1. **Weather effects**
   ```c
   if (weatherSystem.active) {
       RenderWeatherParticles();
   }
   ```

2. **Screen effects**
   ```c
   if (screenEffects.enabled) {
       RenderScreenEffects();
   }
   ```

---

## Optimization Techniques

### Level of Detail

| Distance | Detail Level |
|----------|-------------|
| 0 - 100m | Full skybox |
| 100m - 500m | Simplified sky |
| 500m - 1000m | Fog-only |
| 1000m+ | Full fog |

### Culling

1. **Frustum culling**
   - Skybox always visible (draw if in view)
   
2. **Distance culling**
   - Farther objects = less detail

### Shader Optimization

1. **Vertex shader**
   - Simple position transform
   - No lighting
   
2. **Fragment shader**
   - Single texture lookup
   - Simple fog calculation

---

## Debugging Tools

### Console Commands

| Command | Description |
|---------|-------------|
| `ToggleFog` | Toggle fog rendering |
| `ShowTerrain` | Toggle terrain |
| `DetailDoodadTest` | Test detail doodads |
| `Weather <type>` | Force weather type |

### Visual Debugging

1. **Wireframe sky**
   - Show sky dome wireframe
   
2. **Fog debug**
   - Show fog start/end distances
   
3. **Light debug**
   - Show light directions

---

## References

### Related Functions

| Function | Address |
|----------|---------|
| [`LoadLightsAndFog`](LoadLightsAndFog) | 0x006c4110 |
| [`SetFogColors`](SetFogColors) | 0x006bc780 |
| [`GetFog`](GetFog) | 0x0068b1c0 |
| [`ComputeFogBlend`](ComputeFogBlend) | 0x00689b40 |
| [`QueryCameraFog`](QueryCameraFog) | 0x00689bf0 |
| [`QueryMapObjFog`](QueryMapObjFog) | 0x006896d0 |
| [`CSimpleModel_SetFogColor`](CSimpleModel_SetFogColor) | 0x00774f80 |
| [`CSimpleModel_SetFogFar`](CSimpleModel_SetFogFar) | 0x00775210 |
| [`CSimpleModel_SetFogNear`](CSimpleModel_SetFogNear) | 0x00775160 |

### DBC Files

| DBC File | Purpose |
|----------|---------|
| Skybox.dbc | Skybox definitions |
| Weather.dbc | Weather effects |
| Light.dbc | Light sources |
| GroundEffectTexture.dbc | Ground effects |
