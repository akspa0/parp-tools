# WoW 3.0.1.8303 (Wrath Beta) Deep Dive Analysis

**Analysis Date**: Jan 18 2026  
**Binary**: `wow.exe`  
**Build**: 8303 (Assertions Enabled)  
**Tool**: Ghidra

## 1. Build Information

```
World of WarCraft: Assertions Enabled Build (build 8303)
Build Server: f:\buildserver\bs1\work\wow-code\trunk\
```

**Key Notes:**
- **Debug build** with assertions enabled
- Source paths reveal internal structure
- Same ADT format as 3.3.5 and 4.0.0 (pre-split)

## 2. Source Code Structure (from Debug Paths)

### Client Core
```
f:\buildserver\bs1\work\wow-code\trunk\wow\source\...
    - ClientCommands.h
    - Object/ObjectClient/Player_C.h
    - DB/WowClientDB.h  
    - WowServices/PatchFiles.h
    - objectalloc/IObjectAlloc.h
```

### World/Map System
```
.\MapRenderChunk.cpp      // Alpha decompression (line refs visible!)
.\worldclient\CMapObj.h
.\worldclient\CMapChunk.h
.\worldclient\CMapArea.h
```

### Engine
```
.\engine\source\base\RLECompress.h    // RLE engine utility!
.\engine\source\model2\M2Model.h
.\engine\source\model2\M2Model.inl
.\engine\source\services\ParticleSystem2.h
.\engine\source\gx\CGxDevice.h
```

## 3. Alpha Map Parsing (identical to 4.0.0)

### Functions Found
| Function | Address | Description |
|----------|---------|-------------|
| `UnpackAlphaBits` | `0x00720260` | Main dispatcher |
| `UnpackAlphaShadowBits` | `0x0071f900` | Shadow+alpha |

### Key Global
- `DAT_00edf9c0` - Controls `do_not_fix_alpha_map` flag (same as 4.0.0)

### Source Reference
```c
// From MapRenderChunk.cpp line 0x7a7:
"layerAlpha->m_alpha"  // Layer alpha data pointer
```

## 4. New in Wrath: Vehicle System

### DBCs
- `DBFilesClient\Vehicle.dbc`
- `DBFilesClient\VehicleSeat.dbc`

### Source Files
- `.\UnitVehicle_C.cpp`

### Commands/Messages
```
CMSG_DISMISS_CONTROLLED_VEHICLE
VEHICLE_POWER_SHOW
VEHICLE_ANGLE_UPDATE
```

### CVars
- `VehiclePower` - Launch speed (0-1)
- `VehicleAngle` - Pitch (0-1)

## 5. M2 Model System

### Source Files
- `.\M2Cache.cpp`
- `.\M2Scene.cpp`
- `.\M2Model.cpp`
- `.\M2Light.cpp`
- `.\M2Shared.cpp`
- `.\M2DataInit.inl`

### CVars (M2 Engine Tuning)
| CVar | Description |
|------|-------------|
| `M2Faster` | Performance mode |
| `M2FasterDebug` | Debug perf mode |
| `M2UseShaders` | Shader support |
| `M2BatchDoodads` | Batch rendering |
| `M2BatchParticles` | Particle batching |
| `M2UseThreads` | Multithreading |
| `M2UseClipPlanes` | Clipping |
| `M2UseZFill` | Z-buffer fill |
| `M2OverrideModel` | Debug override |

## 6. Terrain System

### Shaders (BLS format)
```
Shaders\Pixel\terrain1.bls - terrain4.bls
Shaders\Pixel\terrain1w.bls - terrain4w.bls (water)
Shaders\Pixel\terrain_sm.bls (shadow map)
Shaders\Pixel\terrainpw1.bls - terrainpwN.bls
```

### CVars
| CVar | Description |
|------|-------------|
| `terrainAlphaBitDepth` | 4 or 8 bit alpha |
| `showTerrain` | Toggle terrain |
| `cameraTerrainTilt` | Camera tilt |
| `detailDoodadAlpha` | Doodad fade |

### Debug Commands
```
"Terrain enabled/disabled"
"Terrain chunk batches enabled/disabled"
"Terrain doodads AA Box visuals enabled/disabled"
"Terrain doodads collision visuals enabled/disabled"
```

## 7. Chunk Token Validation

### MPHD (WDT Header)
```c
"iffChunk.token=='MPHD'"  // at 0x00988534
```

### MOBN (WMO BSP)
```c
"pIffChunk->token=='MOBN'"  // at 0x00989a34
```

## 8. DBCs (50+ loaded)

### Achievement (new in WotLK)
- Achievement.dbc
- Achievement_Criteria.dbc
- Achievement_Category.dbc

### Vehicle (new in WotLK)
- Vehicle.dbc
- VehicleSeat.dbc

### Map/World
- AreaTable.dbc
- AreaTrigger.dbc
- DungeonMap.dbc
- DungeonMapChunk.dbc

### Creature
- CreatureModelData.dbc
- CreatureDisplayInfo.dbc
- CreatureSoundData.dbc

## 9. Comparison: 3.0.1 vs 4.0.0

| Feature | WotLK 3.0.1.8303 | Cata 4.0.0.11927 |
|---------|-----------------|------------------|
| **Build Server** | `bs1` on F: | `wow\5` on D: |
| **Alpha Parsing** | Identical | Identical |
| **Vehicle System** | Active | Active (enhanced) |
| **RLECompress.h** | Referenced | Not visible |
| **Assertions** | Enabled | Stripped |
| **ADT Format** | Pre-split | Pre-split |

## 10. Key Findings for Data Pipeline

### Alpha Maps (3.0.1 - 4.0.0)
- **Same format** across all Wrath and early Cata
- RLE decompression algorithm unchanged
- `terrainAlphaBitDepth` CVar controls 4/8 bit mode
- WDT MPHD flag 0x4 controls big alpha maps

### ADT Compatibility  
- 3.0.1, 3.3.5, and 4.0.0.11927 all use **monolithic ADT**
- Split ADT (`_tex0.adt`, `_obj0.adt`) not until 4.0.1+

### M2/WMO
- Same formats across versions
- Chunk tokens validated at load time
- BSP tree (MOBN) structure unchanged

### Recommendation
The `LKMapService` fix (using WDT MPHD flags) will work for **all** versions from 3.0.1 through 4.0.0. No additional format changes needed.
