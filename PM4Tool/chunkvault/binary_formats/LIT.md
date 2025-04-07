# LIT (Lighting) Format

## Overview
LIT files store legacy lighting information for World of Warcraft, primarily used for sky colors and lighting conditions. This format has been superseded by DBC files in modern versions but remains important for legacy content and private servers.

## File Structure

### Header
```cpp
struct LitHeader {
    uint32_t version;    // 0x80000003 (v8.3), 0x80000004 (v8.4), 0x80000005 (v8.5)
    int32_t lightCount;  // Number of lights, -1 for single partial entry
};
```

### Light List Entry
```cpp
struct LightListData {
    C2iVector chunk;           // (-1,-1) for default, (0,0) otherwise
    int32_t chunkRadius;       // -1 for default, 0 otherwise
    C3Vector lightLocation;    // Light position (X,Y,Z)
    float lightRadius;         // Light radius (typically scaled by 36)
    float lightDropoff;        // Light attenuation
    char lightName[32];        // Light name (0 or 0xFFFD padded in v8.3)
};
```

### Light Data Block
```cpp
struct DiskLightDataItem {
    int32_t highlightCount[numHighLights];  // numHighLights = 14 (v8.3) or 18 (v8.4+)
    LightMarker highlightMarker[numHighLights][32];
    float fogEnd[32];
    float fogStartScaler[32];
    int32_t highlightSky;
    float skyData[4][32];
    int32_t cloudMask;
    float paramData[4][10];  // v8.5+ only
};

struct LightMarker {
    int32_t time;           // Half-minutes from midnight (0-2880)
    CImVector color;        // BGRX color
};
```

## Color Track Definitions
```cpp
enum LightColorTrack {
    GlobalDiffuse = 0,     // Global diffuse lighting
    GlobalAmbient = 1,     // Global ambient lighting
    SkyTop = 2,           // Sky color (top)
    SkyMiddle = 3,        // Sky color (middle)
    SkyHorizonUpper = 4,  // Sky color (middle to horizon)
    SkyAboveHorizon = 5,  // Sky color (above horizon)
    SkyHorizon = 6,       // Sky color (horizon)
    FogColor = 7,         // Fog and background mountains
    Reserved8 = 8,        // Unknown
    SunColor = 9,         // Sun and sun halo
    SunHaloLarge = 10,    // Larger sun halo
    Reserved11 = 11,      // Unknown
    CloudColor = 12,      // Cloud coloring
    Reserved13 = 13,      // Unknown
    Reserved14 = 14,      // Unknown
    GroundShadow = 15,    // Ground shadow color
    WaterLight = 16,      // Water color (light)
    WaterDark = 17       // Water color (dark)
};
```

## Implementation Notes

### Version Differences
1. **Version 8.3 (0x80000003)**
   - 14 highlight entries
   - 0xFFFD padding in names
   - No paramData array

2. **Version 8.4 (0x80000004)**
   - 18 highlight entries
   - Zero padding in names
   - No paramData array

3. **Version 8.5 (0x80000005)**
   - 18 highlight entries
   - Zero padding in names
   - Added paramData array

### Light Data Organization
1. **Four Data Blocks**
   - Block 0: Default appearance
   - Block 1: Usually black (special state)
   - Block 2: Ghost view lighting
   - Block 3: Usually black (special state)

### Time-Based Colors
1. **Color Interpolation**
   - Time values in half-minutes (0-2880)
   - Linear interpolation between points
   - BGRX color format
   - 24-hour day cycle

### Best Practices
1. **Loading Strategy**
   - Validate version first
   - Handle negative light counts
   - Support all three versions
   - Implement proper interpolation

2. **Error Handling**
   - Check version compatibility
   - Validate light count
   - Verify data block sizes
   - Handle missing data gracefully

3. **Memory Management**
   - Efficient time-based lookup
   - Cache interpolated values
   - Handle large light counts
   - Optimize color transitions

### Modern Replacements
1. **DBC Files**
   - Light.dbc
   - LightParams.dbc
   - LightFloatBand.dbc
   - LightIntBand.dbc
   - LightSkybox.dbc

### Usage Context
1. **World Lighting**
   - Sky color gradients
   - Time-based transitions
   - Atmospheric effects
   - Shadow calculations

2. **Special Effects**
   - Ghost realm lighting
   - Water reflections
   - Cloud coloring
   - Sun effects 