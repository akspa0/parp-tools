# WoWClient.exe (Build 3368) - Lua Scripting & Lighting System Analysis

## Document Information
- **Binary**: WoWClient.exe (December 11, 2003)
- **Build**: 3368
- **Analysis Date**: 2026-02-07
- **Ghidra Version**: 11.x with MCP bridge

---

## Table of Contents
1. [Lua Scripting System (FrameScript)](#1-lua-scripting-system-framescript)
2. [Lighting System](#2-lighting-system)
3. [Timing & Performance](#3-timing--performance)
4. [Function Reference](#4-function-reference)
5. [Data Structures](#5-data-structures)

---

## 1. Lua Scripting System (FrameScript)

### Overview

WoW Alpha 0.5.3 uses **FrameScript**, a custom wrapper around **Lua 5.0** for UI scripting and game logic. The system provides a complete Lua environment integrated into the game engine with custom extensions for WoW-specific functionality.

### Key Components

#### 1.1 Lua State Initialization

**Function**: [`FrameScript_Initialize()`](0x006e61d0)

The Lua state is initialized with the following steps:

```c
// Opens Lua state
DAT_010b3b68 = lua_open();

// Disables garbage collection for performance
lua_disablegc((int)DAT_010b3b68);

// Sets up FrameScript metatable
lua_pushstring((int)DAT_010b3b68, "__framescript_meta");
lua_newtable((int)DAT_010b3b68);
lua_pushstring((int)DAT_010b3b68, "__index");
lua_pushcclosure((int)DAT_010b3b68, FrameScript_Object::LookupScriptMethod, 0);
lua_settable((int)DAT_010b3b68, -3);
lua_settable((int)DAT_010b3b68, LUA_REGISTRYINDEX);

// Hooks custom getglobal and next
lua_pushstring((int)DAT_010b3b68, "__getglobal");
lua_pushcclosure((int)DAT_010b3b68, &LAB_006e63d0, 0);
lua_settable((int)DAT_010b3b68, LUA_REGISTRYINDEX);

// Loads standard Lua libraries
luaopen_string((int)DAT_010b3b68);
luaopen_table((int)DAT_010b3b68);
luaopen_math((int)DAT_010b3b68);

// Loads and executes compatibility script
SFile::LoadFile("Interface/FrameXML/compat.lua", ...);
luaL_loadbuffer((int)DAT_010b3b68, buffer, size, "compat.lua");
lua_pcall((int)DAT_010b3b68, 0, 0, 0);
```

**Loaded Lua Libraries**:
- `string` - String manipulation
- `table` - Table operations  
- `math` - Mathematical functions

#### 1.2 Script Execution

**Function**: [`FrameScript_Execute()`](0x006e7150)

Simple script execution wrapper:

```c
void FrameScript_Execute(const char* script, const char* chunkName) {
    ulong length = SStrLen(script);
    FrameScript_ExecuteBuffer(script, length, chunkName);
}
```

**Function**: [`FrameScript_ExecuteBuffer()`](0x006e7050)

Core execution with error handling:

```c
int FrameScript_ExecuteBuffer(void* script, ulong length, char* chunkName) {
    lua_State* L = FrameScript_GetContext();
    
    // Load the script
    int status = luaL_loadbuffer(L, script, length, chunkName);
    if (status != 0) {
        // Error loading - pcall to get error message
        lua_pcall(L, 1, 0, errorHandler);
        return 0;
    }
    
    // Execute the script
    status = lua_pcall(L, 0, 0, errorHandler);
    if (status != 0) {
        lua_settop(L, -3);
        return 0;
    }
    
    lua_settop(L, -2);
    return 1;  // Success
}
```

#### 1.3 Parameterized Execution (Format String Arguments)

**Function**: [`FrameScript_ExecuteV()`](0x006e7360)

This is the key function for executing scripts with parameters using format strings:

```c
void FrameScript_ExecuteV(
    int functionId,           // Function ID to call
    FrameScript_Object* obj,   // Optional object context
    char* format,             // Format string (%d, %f, %s, %u)
    char* args                // Variadic arguments
)
```

**Supported Format Specifiers**:

| Specifier | Type | Description |
|-----------|------|-------------|
| `%d` | int | Signed 32-bit integer |
| `%f` | double/float | Floating point number |
| `%s` | char* | Null-terminated string |
| `%u` | unsigned | Unsigned 32-bit integer |

**Execution Flow**:

```c
// Parse format string and push arguments to Lua stack
while (*format != '\0') {
    if (*format == '%') {
        format++;
        switch (*format) {
            case 'd':  // Integer
                lua_pushnumber(L, *(int*)args);
                args += 4;
                break;
            case 'f':  // Float/Double
                lua_pushnumber(L, *(double*)args);
                args += 8;
                break;
            case 's':  // String
                lua_pushstring(L, *(char**)args);
                args += 4;
                break;
            case 'u':  // Unsigned
                lua_pushnumber(L, *(unsigned*)args);
                args += 4;
                break;
        }
        // Store in arg table for script access
        lua_pushstring(L, "arg1");
        lua_insert(L, -2);
        lua_settable(L, LUA_GLOBALSINDEX);
    }
    format++;
}

// Get function by ID and call it
lua_rawgeti(L, LUA_REGISTRYINDEX, functionId);
lua_pcall(L, 0, 0, errorHandler);

// Cleanup
lua_pushnil(L);
lua_settable(L, LUA_GLOBALSINDEX);
```

**Usage Example**:
```c
// C code
int playerId = 1234;
const char* spellName = "Fireball";
FrameScript_ExecuteV(42, NULL, "%d%s", playerId, spellName);

// Lua code can access as global variables or arg table
-- playerId would be available as _G["arg1"]
-- spellName would be available as _G["arg2"]
```

#### 1.4 Function Registration

**Function**: [`FrameScript_RegisterFunction()`](0x006e6d00)

Register C functions for Lua execution:

```c
void FrameScript_RegisterFunction(
    const char* name,                    // Lua function name
    int (__fastcall* func)(lua_State*)    // C function pointer
) {
    lua_State* L = FrameScript_GetContext();
    lua_pushcclosure(L, func, 0);        // Create closure
    lua_pushstring(L, name);             // Push name
    lua_insert(L, -2);                   // Swap
    lua_settable(L, LUA_GLOBALSINDEX);  // Register in globals
}
```

**Function Signature for Registered Functions**:

```c
int __fastcall LuaFunctionName(lua_State* L) {
    // Get arguments from Lua stack
    int argc = lua_gettop(L);
    
    // Access arguments
    if (argc >= 1 && lua_isnumber(L, 1)) {
        int value = (int)lua_tonumber(L, 1);
    }
    
    // Return values
    lua_pushnumber(L, result);
    return 1;  // Number of return values
}
```

#### 1.5 FrameScript Functions List

| Function | Address | Purpose |
|----------|---------|---------|
| `FrameScript_Execute` | 0x006e7150 | Execute Lua script |
| `FrameScript_Execute` | 0x006e7170 | Execute Lua script (variant) |
| `FrameScript_Execute` | 0x006e71c0 | Execute Lua script (variant) |
| `FrameScript_Execute` | 0x006e7340 | Execute Lua script (variant) |
| `FrameScript_ExecuteV` | 0x006e7360 | Execute with format args |
| `FrameScript_ExecuteBuffer` | 0x006e7050 | Execute buffer |
| `FrameScript_ExecuteFile` | 0x006e7000 | Execute file |
| `FrameScript_Initialize` | 0x006e61d0 | Initialize Lua state |
| `FrameScript_RegisterFunction` | 0x006e6d00 | Register C function |
| `FrameScript_UnregisterFunction` | 0x006e6d40 | Unregister function |
| `FrameScript_GetVariable` | 0x006e6dd0 | Get variable value |
| `FrameScript_SetVariable` | 0x006e6d80 | Set variable value |
| `FrameScript_GetContext` | 0x006e6c90 | Get Lua state |
| `FrameScript_DisplayError` | 0x006e6cc0 | Display error |
| `FrameScript_CompileFunction` | 0x006e70f0 | Compile function |
| `FrameScript_Destroy` | 0x006e68b0 | Cleanup |

#### 1.6 Compatibility Layer

**File**: `Interface/FrameXML/compat.lua`

This Lua file is loaded during initialization and provides WoW-specific Lua functions that wrap or extend basic Lua functionality. It likely contains:

- UI widget methods
- Game API bindings
- Compatibility shims for newer Lua features

---

## 2. Lighting System

### Overview

The lighting system in WoW Alpha 0.5.3 uses a sophisticated time-based interpolation system with multiple light types, fog control, and weather integration.

### Key Components

#### 2.1 Light Data Loading

**Function**: [`LoadLightsAndFog()`](0x006c4110)

Loads lighting data from `.lit` format files:

```c
bool LoadLightsAndFog(const char* filename, LightGroup* lightGroup) {
    SFile* file;
    
    // Open light data file
    if (!SFile::Open(filename, &file))
        return false;
    
    // Read version magic
    int version;
    SFile::Read(file, &version, 4);
    if (version != 0x80000004)  // -0x7ffffffc in two's complement
        return false;
    
    // Read light count
    int lightCount;
    SFile::Read(file, &lightCount, 4);
    
    // Read each light (0x560 = 1376 bytes each)
    for (int i = 0; i < lightCount; i++) {
        // Read 0x40 byte header
        SFile::Read(file, &lightData[i], 0x40);
        
        // Read light groups at offsets:
        // 0x40:  Primary light data
        // 0x188: Secondary light data (328 bytes)
        // 0x2d0: Tertiary light data (328 bytes)
        // 0x418: Quaternary light data (328 bytes)
        ReadSingleLightGroup(file, &lightData[i].group1);
        ReadSingleLightGroup(file, &lightData[i].group2);
        ReadSingleLightGroup(file, &lightData[i].group3);
        ReadSingleLightGroup(file, &lightData[i].group4);
    }
    
    SFile::Close(file);
    return true;
}
```

#### 2.2 Single Light Group Reading

**Function**: [`ReadSingleLightGroup()`](0x006c43c0)

Reads a light group from disk format to memory format:

```c
undefined4 ReadSingleLightGroup(SFile* file, int dest) {
    DiskLightDataItem diskData;
    
    // Read 0x1550 bytes from file
    SFile::Read(file, &diskData, 0x1550);
    
    // Process 18 highlight marker groups (0x12)
    for (int i = 0; i < 0x12; i++) {
        int count = diskData.m_highlightCount[i];
        
        // Allocate marker array if needed
        if (count != currentMarkers[i].count) {
            if (count == 0) {
                FreeMarkers(&currentMarkers[i]);
            } else {
                ReallocMarkers(&currentMarkers[i], count);
            }
        }
        
        // Copy markers from disk to memory
        for (int j = 0; j < count; j++) {
            markers[i][j].time = diskData.m_highlightMarker[i][j].time;
            markers[i][j].color = diskData.m_highlightMarker[i][j].color;
        }
        
        // Special handling for sky data (index 9)
        if (i == 9) {
            // Sky data: 4 floats per marker (16 bytes stride)
            CopySkyData(&diskData.m_skyData, &skyMarkers[i], count);
        }
        
        // Special handling for fog data (index 7)
        if (i == 7) {
            // Fog data: 2 floats (fogEnd, fogStartScaler)
            CopyFogData(&diskData.m_fogEnd, &fogMarkers[i], count);
        }
    }
    
    // Copy global values
    m_highlightSky = diskData.m_highlightSky;
    m_cloudMask = diskData.m_cloudMask;
    
    return 1;
}
```

#### 2.3 Light Color Calculation

**Function**: [`CalcLightColors()`](0x006c4da0)

Calculates interpolated light colors based on time of day:

```c
void CalcLightColors(
    int time,                    // Current game time (0-2880 minutes)
    CurrentLight* out,           // Output structure
    LightDataItem* dayData,      // Day lighting data
    LightDataItem* stormData,     // Storm/weather lighting data
    int stormIntensity           // Storm intensity (0-100)
) {
    // Calculate all light colors from day data
    CalcIndividualLightColor(time, 0, dayData, &out->DirectColor, NULL);      // Direct
    CalcIndividualLightColor(time, 1, dayData, &out->AmbientColor, NULL);     // Ambient
    CalcIndividualLightColor(time, 2, dayData, &out->SkyArray[0], NULL);      // Sky 1
    CalcIndividualLightColor(time, 3, dayData, &out->SkyArray[1], NULL);      // Sky 2
    CalcIndividualLightColor(time, 4, dayData, &out->SkyArray[2], NULL);      // Sky 3
    CalcIndividualLightColor(time, 5, dayData, &out->SkyArray[3], NULL);      // Sky 4
    CalcIndividualLightColor(time, 6, dayData, &out->SkyArray[4], NULL);      // Sky 5
    CalcIndividualLightColor(time, 7, dayData, &out->SkyArray[5], NULL);      // Sky 6
    CalcIndividualLightColor(time, 8, dayData, &out->ShadowOpacity, NULL);    // Shadow
    CalcIndividualLightColor(time, 9, dayData, &out->CloudArray[0], NULL);   // Cloud 1
    CalcIndividualLightColor(time, 10, dayData, &out->CloudArray[1], NULL);  // Cloud 2
    CalcIndividualLightColor(time, 11, dayData, &out->CloudArray[2], NULL);  // Cloud 3
    CalcIndividualLightColor(time, 12, dayData, &out->CloudArray[3], NULL);  // Cloud 4
    CalcIndividualLightColor(time, 13, dayData, &out->CloudArray[4], NULL);  // Cloud 5
    CalcIndividualLightColor(time, 14, dayData, &out->WaterArray[0], NULL); // Water 1
    CalcIndividualLightColor(time, 15, dayData, &out->WaterArray[1], NULL); // Water 2
    CalcIndividualLightColor(time, 16, dayData, &out->WaterArray[2], NULL); // Water 3
    CalcIndividualLightColor(time, 17, dayData, &out->WaterArray[3], NULL); // Water 4
    
    // Fog parameters
    CalcIndividualLightColor(time, 18, dayData, NULL, &out->FogEnd);
    CalcIndividualLightColor(time, 19, dayData, NULL, &out->FogStartScalar);
    
    // Cloud data
    CalcIndividualLightColor(time, 21, dayData, NULL, &out->CloudData[1]);
    
    // Apply storm/weather blending if needed
    if (stormData != NULL && stormIntensity > 0) {
        // Calculate storm colors
        CurrentLight stormColors;
        memset(&stormColors, 0, sizeof(stormColors));
        
        // Calculate all storm light colors
        CalcStormColors(time, stormData, &stormColors);
        
        // Blend day and storm colors (weighted average)
        int dayWeight = 100 - stormIntensity;
        int stormWeight = stormIntensity;
        
        out->DirectColor = BlendColor(out->DirectColor, stormColors.DirectColor, stormWeight);
        out->AmbientColor = BlendColor(out->AmbientColor, stormColors.AmbientColor, stormWeight);
        out->FogEnd = out->FogEnd * dayWeight * 0.01f + stormColors.FogEnd * stormWeight * 0.01f;
        out->FogStartScalar = out->FogStartScalar * dayWeight * 0.01f + 
                              stormColors.FogStartScalar * stormWeight * 0.01f;
        out->ShadowOpacity = BlendColor(out->ShadowOpacity, stormColors.ShadowOpacity, stormWeight);
        out->CloudData[1] = out->CloudData[1] * dayWeight * 0.01f + 
                           stormColors.CloudData[1] * stormWeight * 0.01f;
    }
}
```

#### 2.4 Individual Light Color Interpolation

**Function**: [`CalcIndividualLightColor()`](0x006c4680)

Core time-based color interpolation:

```c
void CalcIndividualLightColor(
    int time,           // Current time (0-2880 minutes = 48 hours)
    int lightType,      // Light type (0-17)
    LightDataItem* src, // Source data
    CImVector* out,     // Output color
    float* outFloat     // Output float (for fog/sky)
) {
    // Clamp light type to valid range
    int effectiveType = lightType;
    if (effectiveType > 0x11) effectiveType = 9;   // Clamp fog types
    if (effectiveType > 0x13) effectiveType = 9;   // Clamp float types
    
    // Get marker array for this light type
    LightMarker* markers = src->m_highlightMarker[effectiveType];
    int markerCount = markers->count;
    
    if (markerCount <= 0) {
        out->r = out->g = out->b = 0;
        out->a = 0xFF;
        return;
    }
    
    // Handle time wrapping (0xb40 = 2880 = 48 hours in minutes)
    int wrappedTime = time;
    while (wrappedTime >= 0xb40) wrappedTime -= 0xb40;
    while (wrappedTime < 0) wrappedTime += 0xb40;
    
    // Find surrounding markers and interpolate
    int i = 0;
    do {
        int currentTime = markers[i].time;
        int nextTime = markers[i + 1].time;
        
        // Handle wrap-around
        if (currentTime > nextTime) {
            nextTime += 0xb40;
            if (wrappedTime < currentTime) {
                wrappedTime += 0xb40;
            }
        }
        
        if (currentTime <= wrappedTime && wrappedTime <= nextTime) {
            // Found the interval - interpolate
            int range = nextTime - currentTime;
            int position = wrappedTime - currentTime;
            float t = (float)position / (float)range;
            
            if (lightType < 0x12) {
                // Color interpolation (RGB)
                out->r = markers[i].color.r + 
                        (markers[i + 1].color.r - markers[i].color.r) * t;
                out->g = markers[i].color.g + 
                        (markers[i + 1].color.g - markers[i].color.g) * t;
                out->b = markers[i].color.b + 
                        (markers[i + 1].color.b - markers[i].color.b) * t;
                out->a = 0xFF;
            } else if (outFloat != NULL) {
                // Float interpolation (fog/sky values)
                float currentVal = markers[i].floatVal;
                float nextVal = markers[i + 1].floatVal;
                *outFloat = currentVal + (nextVal - currentVal) * t;
            }
            return;
        }
        
        i++;
    } while (i < markerCount);
}
```

#### 2.5 Light Marker Structure

```c
// Time marker with color value
struct LightMarker {
    int time;           // Game time (minutes, 0-2880)
    CImVector color;    // RGB color (4 bytes: r, g, b, a)
};

// Alternative marker for float values
struct LightDataSky {
    float values[4];    // Sky-specific float values
};

struct LightDataFog {
    float fogEnd;       // Fog end distance
    float fogStart;    // Fog start scalar
};
```

#### 2.6 Current Light Output Structure

```c
struct CurrentLight {
    CImVector DirectColor;      // Direct light color (4 bytes)
    CImVector AmbientColor;     // Ambient light color (4 bytes)
    CImVector SkyArray[6];      // Sky gradient colors (24 bytes)
    CImVector CloudArray[5];    // Cloud colors (20 bytes)
    CImVector WaterArray[4];    // Water reflection colors (16 bytes)
    CImVector ShadowOpacity;    // Shadow darkness (4 bytes)
    float FogEnd;               // Fog end distance
    float FogStartScalar;       // Fog start relative to end
    float CloudData[2];         // Cloud-specific data
    float Darkness;             // Overall darkness level
    
    // Total size: ~104 bytes
};
```

#### 2.7 Light Types

| Type ID | Name | Purpose |
|---------|------|---------|
| 0 | Direct | Main directional light (sun/moon) |
| 1 | Ambient | Ambient/hemisphere light |
| 2-7 | Sky1-Sky6 | Sky gradient colors (horizon to zenith) |
| 8 | Shadow | Shadow opacity/color |
| 9-13 | Cloud1-Cloud5 | Cloud lighting colors |
| 14-17 | Water1-Water4 | Water reflection colors |
| 18 | FogEnd | Distance fog ends |
| 19 | FogStartScalar | Relative fog start |
| 21 | CloudData | Additional cloud parameters |

#### 2.8 GPU Light Creation

**Function**: [`CreateGxLight()`](0x0044a240)

Creates a GPU light resource:

```c
int CreateGxLight(MLDXLightData* data) {
    // Create GPU light object
    int lightHandle = GxuLightCreate();
    if (lightHandle == 0) return 0;
    
    // Lock light for writing
    uint* lightData = GxuLightLock();
    if (lightData == NULL) return 0;
    
    // Convert and set color intensities
    float r = data->color.r * 255.0f;  // 0x437f0000 multiplier
    float g = data->color.g * 255.0f;
    float b = data->color.b * 255.0f;
    
    // Clamp and convert to fixed-point (0x44000000 offset)
    lightData[5] = FixedPointConvert(r);  // Red intensity
    lightData[4] = FixedPointConvert(g);  // Green intensity
    lightData[6] = FixedPointConvert(b);  // Blue intensity
    
    // Set falloff parameters
    float falloffStart = data->falloffStart * 255.0f;
    float falloffEnd = data->falloffEnd * 255.0f;
    lightData[0x12] = FixedPointConvert(falloffStart);
    lightData[0x11] = FixedPointConvert(falloffEnd);
    lightData[4] = FixedPointConvert(data->intensity);
    
    // Set flags
    lightData[7] = data->flags;
    lightData[8] = data->type;
    
    // Unlock and return
    GxuLightUnlock();
    return lightHandle;
}
```

#### 2.9 Light Management

**Function**: [`AllocLight()`](0x00690f80)

Allocates a `CMapLight` object from the pool:

```c
CMapLight* CMap::AllocLight() {
    CMapLight* light;
    
    // Check freelist first
    if (DAT_00e6e47c != NULL) {
        light = DAT_00e6e47c;
        goto reuse_light;
    }
    
    // Allocate new light (0xc4 = 196 bytes)
    light = SMemAlloc(0xc4, "CMapLight", 0xfffffffe, 8);
    if (light == NULL) return NULL;
    
    // Initialize
    CMapLight::CMapLight(light);

reuse_light:
    // Unlink from freelist
    UnlinkFromFreelist(light);
    
    // Add to active list
    AddToActiveList(light);
    
    return light;
}
```

---

## 3. Timing & Performance

### Overview

The timing system uses multiple sources: standard C time functions, Windows API, and potentially CPU timing.

### Time Functions

#### 3.1 Standard C Time

**Function**: [`OsGetTime()`](0x0045bc20)

```c
long OsGetTime(time_t* result) {
    time_t t = time(NULL);  // Seconds since epoch
    if (result != NULL) *result = t;
    return (long)t;
}
```

#### 3.2 Formatted Timestamp

**Function**: [`OsGetTimeStamp()`](0x0045bbb0)

```c
void OsGetTimeStamp(char* buffer, ulong size) {
    time_t rawTime;
    time(&rawTime);
    
    struct tm* tmInfo = localtime(&rawTime);
    
    // Format: "m/d/y H:M:S"
    strftime(buffer, size, "%m/%d/%y %H:%M:%S", tmInfo);
}
```

#### 3.3 Windows API Imports

| Function | Address | Purpose |
|----------|---------|---------|
| `_GetTickCount@0` | 0x006f0878 | Milliseconds since boot |
| `_QueryPerformanceCounter@4` | 0x006f0860 | High-resolution counter |
| `_QueryPerformanceFrequency@4` | 0x006f0854 | Counter frequency |

**Usage Pattern**:

```c
// High-resolution timing
LARGE_INTEGER counter;
LARGE_INTEGER frequency;

QueryPerformanceFrequency(&frequency);
QueryPerformanceCounter(&counter);

double seconds = (double)counter.QuadPart / (double)frequency.QuadPart;

// Or with GetTickCount
DWORD start = GetTickCount();
// ... perform work ...
DWORD elapsed = GetTickCount() - start;
```

### Game Time System

The game time is tracked in **minutes** with a range of 0-2880 (48 hours):

```
0x0000 = Midnight (0:00)
0x0300 = 3:00 AM
0x0600 = 6:00 AM (dawn)
0x0C00 = Noon (12:00)
0x1200 = 6:00 PM (dusk)
0x1800 = 9:00 PM
0x2400 = Midnight (24:00 = 0x0)
0xB40 = 2880 minutes = 48 hours
```

### Time Wrapping

Light markers and other time-based data wrap at `0x0B40` (2880 minutes = 48 hours):

```c
int WrapGameTime(int time) {
    while (time >= 0xB40) time -= 0xB40;
    while (time < 0) time += 0xB40;
    return time;
}
```

---

## 4. Function Reference

### Lua Scripting Functions

| Address | Function Name | Description |
|---------|---------------|-------------|
| 0x006e7150 | FrameScript_Execute | Execute Lua script |
| 0x006e7170 | FrameScript_Execute | Execute Lua script (var) |
| 0x006e71c0 | FrameScript_Execute | Execute Lua script (var) |
| 0x006e7340 | FrameScript_Execute | Execute Lua script (var) |
| 0x006e7360 | FrameScript_ExecuteV | Execute with format args |
| 0x006e7050 | FrameScript_ExecuteBuffer | Execute buffer |
| 0x006e7000 | FrameScript_ExecuteFile | Execute file |
| 0x006e61d0 | FrameScript_Initialize | Initialize Lua |
| 0x006e6d00 | FrameScript_RegisterFunction | Register C function |
| 0x006e6d40 | FrameScript_UnregisterFunction | Unregister function |
| 0x006e6dd0 | FrameScript_GetVariable | Get variable |
| 0x006e6e90 | FrameScript_GetVariable | Get variable (var) |
| 0x006e6f50 | FrameScript_GetVariable | Get variable (var) |
| 0x006e6d80 | FrameScript_SetVariable | Set variable |
| 0x006e6e40 | FrameScript_SetVariable | Set variable (var) |
| 0x006e6f00 | FrameScript_SetVariable | Set variable (var) |
| 0x006e6c90 | FrameScript_GetContext | Get Lua state |
| 0x006e6cc0 | FrameScript_DisplayError | Display error |
| 0x006e70f0 | FrameScript_CompileFunction | Compile function |
| 0x006e7130 | FrameScript_ReleaseFunction | Release function |
| 0x006e6890 | FrameScript_Flush | Flush scripts |
| 0x006e68b0 | FrameScript_Destroy | Destroy state |
| 0x006e6ac0 | FrameScript_CreateEvents | Create events |
| 0x006e6c80 | FrameScript_DestroyEvents | Destroy events |
| 0x006e6b30 | FrameScript_SignalEvent | Signal event |
| 0x006e6bd0 | FrameScript_SignalEvent | Signal event (var) |
| 0x006e6930 | FrameScript_LoadTextTables | Load text tables |
| 0x006e6900 | FrameScript_MemoryCleanup | Memory cleanup |

### Lighting Functions

| Address | Function Name | Description |
|---------|---------------|-------------|
| 0x006c4110 | LoadLightsAndFog | Load LIT file |
| 0x006c43c0 | ReadSingleLightGroup | Read light group |
| 0x006c4da0 | CalcLightColors | Calculate colors |
| 0x006c4680 | CalcIndividualLightColor | Interpolate color |
| 0x0044a240 | CreateGxLight | Create GPU light |
| 0x0044a8b0 | CreateGxLight | Create GPU light (var) |
| 0x00686690 | CreateLight | Create light object |
| 0x0073b310 | CreateLight | Create light object (var) |
| 0x0073c460 | CreateLight | Create light object (var) |
| 0x00690f80 | AllocLight | Allocate light |
| 0x006911a0 | AllocCacheLight | Allocate cached light |
| 0x006a4160 | CreateCacheLight | Create cached light |
| 0x00686570 | CMapLight | Light class |
| 0x005b8820 | CLightList | Light list class |
| 0x006a3d00 | AdjustLightmap | Adjust lightmap |
| 0x0043d9f0 | CopyLights | Copy light data |
| 0x006af820 | CreateLightmapPointers | Create lightmap ptrs |
| 0x0065af30 | CGxLight | GPU light class |

### Timing Functions

| Address | Function Name | Description |
|---------|---------------|-------------|
| 0x0045bc20 | OsGetTime | Get current time |
| 0x0045bd30 | OsGetTime | Get current time (var) |
| 0x0045bbb0 | OsGetTimeStamp | Get timestamp string |
| 0x0045bb70 | OsGetTimeStr | Get time string |
| 0x0045bbf0 | OsGetTimeStr | Get time string (var) |
| 0x00632c20 | WowGetTimeString | Get WoW time string |
| 0x00632c50 | WowGetTimeString | Get WoW time string (var) |
| 0x00541d90 | Script_GetTime | Get script time |

### Windows API Imports

| Address | Function Name | Description |
|---------|---------------|-------------|
| 0x006f0878 | GetTickCount | Milliseconds since boot |
| 0x006f0860 | QueryPerformanceCounter | HPC value |
| 0x006f0854 | QueryPerformanceFrequency | HPC frequency |

---

## 5. Data Structures

### 5.1 LightDataItem (On-Disk Format)

```c
struct DiskLightDataItem {
    // 18 marker groups × 576 markers × 8 bytes = 82944 bytes
    LightMarker m_highlightMarker[18][576];
    uint m_highlightCount[18];
    
    // Sky data: 576 markers × 4 floats × 4 bytes = 9216 bytes
    float m_skyData[576][4];
    
    // Fog data: 576 markers × 2 floats = 4608 bytes
    float m_fogEnd[576];
    float m_fogStartScaler[576];
    
    // Global values
    int m_highlightSky;
    int m_cloudMask;
    
    // Total size: 0x1550 = 5456 bytes
};
```

### 5.2 LightData (Runtime Format)

```c
struct LightData {
    // Header: 64 bytes
    uint type;              // Light type
    uint flags;             // Light flags
    float position[3];      // Position (X, Y, Z)
    float intensity;         // Intensity multiplier
    
    // 4 light groups × 328 bytes = 1312 bytes
    LightGroup groups[4];
    
    // Total per light: 0x560 = 1376 bytes
};
```

### 5.3 LightGroup

```c
struct LightGroup {
    // 18 marker arrays
    TSFixedArray<LightMarker> markers[18];
    
    // Sky data array
    TSFixedArray<LightDataSky> skyData;
    
    // Fog data array  
    TSFixedArray<LightDataFog> fogData;
    
    // Global values
    int highlightSky;
    int cloudMask;
};
```

### 5.4 CMapLight

```c
class CMapLight {
    // Total size: 0xC4 = 196 bytes
    
    TSLink<CMapLight> link;      // Linked list node
    float position[3];           // World position
    float color[3];              // RGB color
    float intensity;            // Intensity (0.0 - 1.0)
    float falloffStart;          // Attenuation start
    float falloffEnd;            // Attenuation end
    uint type;                   // Light type
    uint flags;                  // Render flags
    int parentObject;            // Parent object ID
    
    // Freelist management
    static CMapLight* freeList;
    static int activeCount;
};
```

### 5.5 CurrentLight (Output Format)

```c
struct CurrentLight {
    CImVector DirectColor;        // +0x00: Direct light RGB (4 bytes)
    CImVector AmbientColor;       // +0x04: Ambient light RGB (4 bytes)
    CImVector SkyArray[6];        // +0x08: Sky colors (24 bytes)
    CImVector CloudArray[5];      // +0x20: Cloud colors (20 bytes)
    CImVector WaterArray[4];      // +0x34: Water colors (16 bytes)
    CImVector ShadowOpacity;      // +0x44: Shadow (4 bytes)
    float FogEnd;                // +0x48: Fog distance end
    float FogStartScalar;         // +0x4C: Fog start scalar
    float CloudData[2];          // +0x50: Cloud parameters
    float Darkness;              // +0x58: Darkness level
    
    // Total size: ~96 bytes
};
```

### 5.6 CImVector (Color Format)

```c
struct CImVector {
    uint8_t r;    // Red (0-255)
    uint8_t g;    // Green (0-255)
    uint8_t b;    // Blue (0-255)
    uint8_t a;    // Alpha (0-255)
};
```

---

## Appendix A: Lua 5.0 API Reference

The FrameScript system uses standard Lua 5.0 API functions:

| Function | Purpose |
|----------|---------|
| `lua_open()` | Create new state |
| `lua_close()` | Close state |
| `lua_pcall()` | Protected call |
| `luaL_loadbuffer()` | Load chunk |
| `lua_pushnumber()` | Push number |
| `lua_pushstring()` | Push string |
| `lua_pushcclosure()` | Push C function |
| `lua_settable()` | Set table value |
| `lua_gettable()` | Get table value |
| `lua_rawgeti()` | Get by integer key |
| `lua_insert()` | Insert on stack |
| `lua_settop()` | Set stack top |
| `lua_gettop()` | Get stack top |
| `lua_isnumber()` | Check type |
| `lua_isstring()` | Check type |
| `lua_tonumber()` | Convert to number |
| `lua_tostring()` | Convert to string |
| `luaopen_string()` | Open string lib |
| `luaopen_table()` | Open table lib |
| `luaopen_math()` | Open math lib |

---

## Appendix B: File Formats

### LIT File Format

```
[Header: 8 bytes]
- Version: 0x80000004 (4 bytes)
- Light Count: uint32 (4 bytes)

[Light Data: 1376 bytes each × N]
- Light Header: 64 bytes
- Light Group 1: 328 bytes
- Light Group 2: 328 bytes
- Light Group 3: 328 bytes
- Light Group 4: 328 bytes
```

---

## References

- Lua 5.0 Reference Manual: https://www.lua.org/manual/5.0/
- WoW Alpha 0.5.3 (Build 3368) binary analysis
- Ghidra decompilation results

---

*Document generated from Ghidra analysis of WoWClient.exe (Build 3368)*
