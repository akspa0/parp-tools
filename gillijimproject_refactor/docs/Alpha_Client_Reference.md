# Alpha Client Reference

**Comprehensive documentation of WoW Alpha client architecture, features, and hidden tools.**

---

## Version Comparison Matrix

| Feature | 0.5.3 (3368) | 0.5.5 (3494) | 0.6.0 (3592) | 3.3.5a (12340) |
|:---|:---|:---|:---|:---|
| **Build Date** | Dec 11 2003 | ~Jan 2004 | ~Feb 2004 | Aug 2010 |
| **Map Format** | Monolithic WDT | Monolithic (ADT latent) | Split ADT | Split ADT |
| **WMO Version** | v14 | v14 | v16 | v17 |
| **Audio Engine** | Miles (MSS) | FMOD | FMOD | FMOD |
| **UI Engine** | XML/Lua 5.0 | XML/Lua 5.0 | XML/Lua 5.1 | XML/Lua 5.1 |
| **Textures** | BLP2 | BLP2 | BLP2 | BLP2 |
| **Database** | WDBC | WDBC | WDBC | WDBC |

---

# Part 1: Supported File Formats

## World & Assets

| Extension | Format | 0.5.3 | 0.5.5 | 0.6.0 | Notes |
|:---|:---|:---|:---|:---|:---|
| .wdt | World Data | ✅ | ✅ | ✅ | Monolithic vs Split |
| .adt | Area Tile | ❌ | Latent | ✅ | First in 0.6.0 |
| .wmo | World Object | ✅ v14 | ✅ v14 | ✅ v16 | |
| .m2 | M2 Model | ❌ | ❌ | ✅ | |
| .mdx | WC3 Model | ✅ | ✅ | ✅ | Legacy support |
| .blp | Texture | BLP2 | BLP2 | BLP2 | Not BLP1 |
| .wdl | Low-res World | ✅ | ✅ | ✅ | |

## Database & Cache

| Extension | Format | Description |
|:---|:---|:---|
| .dbc | Client DB | Static data (Spells, Items, etc.) |
| .wdb | World Cache | Server query cache |

## User Interface

| Extension | Format | Description |
|:---|:---|:---|
| .xml | Layout | Frame definitions |
| .lua | Script | UI logic |
| .toc | TOC | Addon metadata |
| .ttf | Font | TrueType fonts |

## Audio (FMOD in 0.5.5+)
FMOD handles: `.mp3`, `.ogg`, `.wav`, `.mid`, `.dls`

---

# Part 2: Console Variables (CVars)

## Graphics & Display

| CVar | Description | Notes |
|:---|:---|:---|
| `gxColorBits` | Color depth (16/32) | |
| `gxDepthBits` | Z-buffer depth (16/24) | |
| `gxResolution` | Screen resolution | e.g. "640x480" |
| `gxRefresh` | Refresh rate | |
| `gxWindow` | Windowed mode | 0=FS, 1=Window |
| `pixelShaders` | Enable pixel shaders | |
| `hwDetect` | Hardware detection | Debug |

## World & Rendering

| CVar | Description | Range |
|:---|:---|:---|
| `farclip` | View distance | 177.0 - 777.0 |
| `horizonfarclip` | Horizon distance | |
| `DistCull` | Object cull distance | 1.0 - 500.0 |

## Debug

| CVar | Description |
|:---|:---|
| `error` | Error reporting level |
| `scriptErrors` | Lua error reporting |
| `debugobjectpathing` | Object pathing debug |
| `playercombatlogdebug` | Combat log debug |
| `CombatDebugShowFlags` | Combat debug flags |

---

# Part 3: Hidden Tools & Dead Code

## Alpha 0.5.3 - Active Developer Tools

### Sound Zone Editor (SndDebug)

> [!IMPORTANT]
> **SndDebug exists ONLY in Alpha 0.5.3**. It was removed in 0.5.5 and later versions.

Fully functional in-game audio zone creation tool:

| Command | Description |
|:---|:---|
| `SndDebugCreateChunk` | Create audio zone at player position |
| `SndDebugSetChunkProperty` | Modify zone properties (reverb, ambience) |
| `SndDebugSetCurrentChunk` | Select a zone by ID |
| `SndDebugShowCurrentChunk` | Display current zone info |
| `SndDebugDumpChunks` | Dump all zones to log file |
| `SndDebugListChunks` | List all loaded zones |

#### Save Mechanism Analysis (Ghidra Verified)

Based on decompilation of 0.5.3 `wowclient.exe`:

**Output Files**:
- `SndEAXChunkInfo_OUTDOORS_%02d.txt` - Outdoor zone audio data
- `SndEAXChunkInfo_INDOORS_%02d.txt` - Indoor zone audio data

**Implementation** (from `DumpChunksOUTDOORS` @ `004a7d50`):
```c
// Creates numbered text files (00-99)
SStrPrintf(buffer, 0x80, "SndEAXChunkInfo_OUTDOORS_%02d.txt", index);
_File = fopen(buffer, "w+");
fprintf(_File, "%d\n", chunkCount);
// Iterates all chunks, calls PrintInfo
OUTDOORSCHUNKHASHOBJ::PrintInfo(chunk, _File);
fclose(_File);
```

**Chunk Structure**:
- Stored in `OUTDOORSCHUNKHASHOBJ` hash table
- Keyed by `AREAHASHKEY` (continent, area, subArea)
- Contains: chunkNumber, continentID, areaID, EAX reverb properties

**Runtime Flow**:
1. `CreateChunkOUTDOORS` gets player position via `ClntObjMgrGetActivePlayer()`
2. Queries area ID via `CWorld::QueryAreaId()`
3. Allocates `OUTDOORSCHUNKHASHOBJ` and links to hash table
4. `DumpChunksOUTDOORS` serializes all chunks to text file

> [!NOTE]
> **No server communication.** No network opcodes. No MPQ writing.  
> Data exists only in memory until manually dumped to text files.

#### Production Equivalent

In later clients (3.3.5a+), audio zone data is stored in:
- `DBFilesClient\WorldChunkSounds.dbc` - Pre-defined world audio zones
- `SoundInterface2ZoneSounds.cpp` - Runtime zone sound management

The SndDebug tool was likely used by Blizzard developers to **prototype** audio zones, then the results were manually entered into DBC files for production builds.


### Cheat Commands

| Command | Syntax | Description | Opcode |
|:---|:---|:---|:---|
| `speed` | `speed <float>` | Set run speed | - |
| `teleport` | `teleport <x> <y> <z>` | Teleport to coords | 0xC6 |
| `money` | `money <copper>` | Set money | 0x24 |
| `level` | `level <1-100>` | Set level | 0x25 |
| `ci` | `ci <itemId>` | Create item | 0x13 |
| `cm` | `cm <creatureId>` | Create monster | 0x11 |

### Quest Commands

| Command | Syntax | Opcode |
|:---|:---|:---|
| `flagquest` | `flagquest <id>` | 0x2A |
| `finishquest` | `finishquest <id>` | 0x2B |
| `clearquest` | `clearquest <id>` | 0x2C |

### GM Commands
`ghost`, `invis`, `bindplayer`, `summon`, `showlabel`, `setsecurity`, `nuke`

### Dead Code
- **MDL Exporter**: `007b3a7a` - Warcraft 3 model export header writer (unreachable)
- **God Mode**: Logic exists but command stripped

---

## WotLK 3.3.5a - Debug Features

### Godmode System
Server-gated but client has display logic:
- Strings: `"Godmode enabled"`, `"Pet Godmode enabled"`
- Opcode: `SPELL_FAILED_BM_OR_INVISGOD`

### Debug Console

| Command | Description |
|:---|:---|
| `ConsoleExec("cmd")` | Execute console command from Lua |
| `closeconsole` | Close console window |
| `cvarlist` | List all CVars |
| `cvar_reset` | Reset all CVars |
| `consolelines` | Set console line count |

### Debug Lua API

| Function | Description |
|:---|:---|
| `TeleportToDebugObject` | Teleport to debug object |
| `GetDebugZoneMap` | Get debug zone map |
| `HasDebugZoneMap` | Check for debug zone map |
| `GetMapDebugObjectInfo` | Get debug object info |
| `GetNumMapDebugObjects` | Count debug objects |
| `IsDebugBuild` | Check if debug build |
| `GetDebugStats` | Get debug statistics |
| `CommentatorSetMoveSpeed(speed)` | Set spectator speed |

### Internal CVars
- Tutorial completion tracking
- Achievement tracking
- Quest tracking

### Legacy Code
- **MDX/MDL**: `stars.mdl` (WC3 skybox), spell effects use `.mdx`
- **Source Paths**: `NetInternal.h`, `ConsoleClient.cpp`, etc.

---

# Part 4: Key Discoveries

## The "BLP1" Myth
Alpha 0.5.3 does **NOT** use BLP1 (JPEG). It checks for `BLP2` magic (0x32504C42). BLP2 was adopted earlier than commonly believed.

## The UI Revolution
Both 0.5.3 and 0.5.5 use **XML/Lua** UI, not hardcoded. The `GlueXML.toc` and Lua copyrights confirm this.

## The 0.5.5 "Rosetta Stone"
0.5.5 contains a fully functional **Prototype ADT** loader gated by `DAT_008ab3e4`. The loader expects headerless ADTs with implicit MCNK data - a format found nowhere else.

## WMO Evolution
- **v14**: Embedded lightmaps (MOLM/MOLD)
- **v16**: Hybrid (v17 root + v14 batch style)
- **v17**: Modern standard
