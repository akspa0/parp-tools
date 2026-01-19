# WoW 4.0.0.11927 (Cataclysm Beta) Deep Dive Analysis

**Analysis Date**: Jan 18 2026  
**Binary**: `wow.exe`  
**Build**: 11927 (Apr 23 2010 18:28:42)  
**Tool**: Ghidra

## 1. Build Information

```
WoW [Release] Build 11927 (Apr 23 2010)
Build Server Path: d:\buildserver\wow\5\work\wow-code\trunk\
```

**Key Notes:**
- This is **pre-split ADT** - uses 3.3.5-style monolithic ADTs
- Split ADTs (`_tex0.adt`, `_obj0.adt`) introduced in 4.0.1+
- Build path confirms internal CI/CD pipeline location

## 2. Core Architecture

### 2.1 Audio: FMOD
**Status**: Active (FMOD Ex)

| Component | Evidence |
|-----------|----------|
| Engine | `" - FMOD version %08x detected"` |
| Callbacks | `FMod FSoundAllocCallback`, `FSoundFreeCallback` |
| Threads | `FMOD stream thread`, `FMOD file thread` |
| Source | `..\..\src\fmod.cpp`, `fmod_systemi.cpp` |

Error messages confirm full FMOD integration with reverb, DSP, and voice chat support.

### 2.2 UI: Lua 5.1 + XML
**Status**: Active (Blizzard-modified Lua 5.1)

Embedded compatibility layer for Lua 5.0 APIs:
```lua
foreach = tab.foreach
foreachi = tab.foreachi
getn = tab.getn
tinsert = tab.insert
mod = math.fmod  -- was math.mod in 5.0
```

Key Lua debug APIs:
- `debugstack`, `debuglocals`, `debugdump`
- `debugprofilestart`, `debugprofilestop`, `debugtimestamp`
- `GetDebugStats`, `IsDebugBuild`

### 2.3 Database: DBC
Standard `.dbc` format confirmed:
- `DBFilesClient\TerrainType.dbc`
- `DBFilesClient\TerrainTypeSounds.dbc`
- `DBFilesClient\FootstepTerrainLookup.dbc`

## 3. File Formats

### 3.1 Maps (ADT/WDT)
| Format | Status |
|--------|--------|
| ADT | `%s\%s_%d_%d.adt` - Monolithic (pre-split) |
| WDT | `%s\%s.wdt` via `CMap::LoadWdt()` |
| WDL | Supported (low-res terrain) |

**Alpha Map Formats** (Confirmed via Ghidra decompile):
- 4-bit uncompressed (2048 bytes)
- 8-bit uncompressed (4096 bytes)  
- 8-bit RLE compressed (variable size)

Dispatch function at `0x00674b70` (`CMapChunk::UnpackAlphaBits`)

### 3.2 Terrain Shaders
Confirmed shader programs:
- `Terrain0` through `Terrain3` (base layers)
- `Terrain1w` through `Terrain1w_4` (water layers)
- `TerrainSM` (shadow map)
- `MapObjUTwoLayerTerrain` (WMO terrain blending)

### 3.3 Models (M2/WMO)
| Format | Status |
|--------|--------|
| M2 | Active (`InvisibleStalker.m2`) |
| WMO | Active (`World\wmo\Dungeon\test\missingwmo.wmo`) |
| MDX | Legacy (no strings found) |

### 3.4 Textures (BLP)
BLP2 format confirmed:
- `"BLP Texture failure: "%s" invalid file version\n"`

## 4. Console & CVars

### 4.1 Console Commands
Source file: `.\\ConsoleClient.cpp`

| Command | Description |
|---------|-------------|
| `cvarlist` | List all CVars |
| `cvar_default` | Reset CVar to coded default |
| `cvar_reset` | Reset CVar to startup value |
| `closeconsole` | Close console window |
| `consolelines` | Set console line count |

### 4.2 Key CVars
| CVar | Description |
|------|-------------|
| `terrainAlphaBitDepth` | Alpha map bit depth (4 or 8) |
| `gxTerrainDispl` | Terrain displacement factor |
| `cameraTerrainTilt` | Camera terrain tilt |
| `farclip` | View distance |
| `hwDetect` | Hardware detection |

### 4.3 Lua CVar API
```lua
GetCVar("cvar"), SetCVar("cvar", value)
GetCVarBool("cvar"), GetCVarDefault("cvar")
GetCVarMin/Max(), GetCVarAbsoluteMin/Max()
RegisterCVar(), GetCVarInfo()
```

## 5. Hidden/Debug Features

### 5.1 Godmode
**Status**: Active (Server-Gated)

```
0x00a12488: "Godmode enabled"
0x00a12474: "Godmode disabled"
0x00a12460: "Pet Godmode enabled"
0x00a12448: "Pet Godmode disabled"
```

### 5.2 Debug Zone Maps
Lua API present:
- `TeleportToDebugObject`
- `GetDebugZoneMap`, `HasDebugZoneMap`
- `GetMapDebugObjectInfo`, `GetNumMapDebugObjects`

### 5.3 GM Features
- `CHAT_FLAG_GM` - GM chat flag
- `GMReportLag`, `GMSurveySubmit` - GM tools
- `GM_EMAIL_NAME`, `GM_PLAYER_INFO` - GM info

### 5.4 LFG Debug Messages
Extensive debug logging for LFG system:
```
LFGMessage: LFG_QUEUE_STATUS - LFGID: %d, Wait: %d...
LFGMessage: LFG_PROPOSAL_UPDATE - Slot: %u, State: %d...
```

## 6. Source File Paths (Debug Artifacts)

### 6.1 Client Core
```
.\Client.cpp
.\NetClient.cpp
..\NetInternal.cpp
.\Login.cpp
.\WardenClient.cpp
```

### 6.2 Object System
```
.\Object_C.cpp
.\Player_C.cpp
.\Unit_C.cpp
.\GameObject_C.cpp
.\Vehicle_C.cpp
.\Corpse_C.cpp
```

### 6.3 UI System
```
.\CSimpleFrame.cpp
.\CSimpleFrameScript.cpp
.\CSimpleEditBox.cpp
.\CSimpleHTML.cpp
.\CSimpleMovieFrame.cpp
```

### 6.4 Sound
```
.\SI3.cpp
.\SI3ZoneSounds.cpp
.\SI3VoiceChat.cpp
```

### 6.5 Storm Library
```
d:\buildserver\wow\5\work\wow-code\trunk\storm\h\STPL.H
```

## 7. Comparison: 0.5.3 vs 3.3.5 vs 4.0.0

| Feature | Alpha 0.5.3 | WotLK 3.3.5 | Cata 4.0.0 |
|---------|-------------|-------------|------------|
| **ADT Format** | Monolithic WDT | Monolithic ADT | Pre-split ADT |
| **Audio** | Miles (MSS) | FMOD | FMOD Ex |
| **Alpha Maps** | 4-bit only | 4/8-bit + RLE | 4/8-bit + RLE |
| **Godmode** | Dead | Server-Gated | Server-Gated |
| **Lua** | 5.0 | 5.1 compat | 5.1 compat |
| **Vehicle System** | N/A | Active | Active (enhanced) |
| **LFG System** | N/A | Active | Active (enhanced) |

## 8. Conclusion

WoW 4.0.0.11927 is a transitional build between WotLK (3.3.5) and Cataclysm (4.0.1+). Key characteristics:

1. **Pre-Split ADT**: Uses 3.3.5-style monolithic ADT files, NOT split format
2. **Full RLE Alpha Support**: Same compression as 3.3.5
3. **Enhanced Vehicle/LFG**: New Cataclysm systems with debug logging
4. **Clean Build Artifacts**: Debug paths reveal internal build server layout
5. **FMOD Ex Audio**: Full sound engine with voice chat support

**For LKMapService**: The alpha parsing code is identical to 3.3.5. Focus on implementing RLE decompression using the algorithm at `0x00673230`.
