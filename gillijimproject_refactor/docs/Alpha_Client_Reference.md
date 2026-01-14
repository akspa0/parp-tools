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
| .tga | Texture/Screenshot | ✅ | ✅ | ✅ | Input & Output |
| .wdl | Low-res World | ✅ | ✅ | ✅ | |

## Client Output Formats
The Alpha client writes the following files to disk:
*   **.tga**: Screenshots (`Screenshots/WoWScrnShot_Date.tga`). Uncompressed or RLE.
*   **.txt**: Debug logs (`Errors.txt`), Audio dumps (`SndEAXChunkInfo`), Config (`Config.wtf`).
*   **.wdb**: Client cache (Item, Quest, NPC data).
*   **.jpg/.png**: **NOT SUPPORTED**. The client does not write or read these formats.

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


#### WMO v14 Format Specification
**Based on Ghidra Analysis of 0.5.3 Client (Build 3368) and Debugging**

#### Group File Structure
Unlike WotLK (v17), Alpha WMO Groups use a different chunk order and header handling.

1.  **MOGP Header Size**:
    *   **Memory / Conversion**: The 0.5.3 client unconditionally adds `0x80` (128 bytes) to the MOGP chunk pointer to find the first subchunk (`MOPY`, `MOVI`, etc.).
    *   **Disk**: Files on disk (like `Ironforge.wmo`) may use a shorter header (68 bytes).
    *   **Handling**: A robust converter must use **Adaptive Header Skipping** by peeking for subchunk signatures (`MOPY`, `MOVI`) at offset 0x44 (68) and 0x80 (128). Blindly skipping 128 bytes on a 68-byte header file leads to "Corrupt Groups" (skipping valid data).

2.  **Chunk Order**:
    Analysis of `CMapObjGroup::ReadRequiredChunks` confirms the parsing order:
    1.  `MOGP` (Header)
    2.  `MOPY` (Material Info)
    3.  `MOVT` (Vertices)
    4.  `MONR` (Normals)
    5.  `MOTV` (UV Coordinates - **Required** if texture used)
    6.  `MOIN` (Indices - **Note Token Change!**)
    7.  `MOBA` (Batches)
    8.  Optional Chunks: `MOLR` (Lights), `MODR` (Doodad Refs), `MOBN/MOBR` (BSP), `MOCV` (Colors).

3.  **Key Chunk Differences**:
    *   **MOIN (Indices)**: v14 uses the token `MOIN` (Little Endian: `NIOM`) instead of `MOVI`. The content is standard `ushort` indices.
    *   **MOTV (UVs)**: Standard `Vector2` (float, float). Coordinate system matches WotLK (no V-flip required for pass-through conversion).
    *   **MOBA (Batches)**:
        *   `StartIndex` is `ushort` (vs `uint32` in v17).
        *   `IndexCount` is `ushort`.
        *   Noggit/WotLK expects `StartIndex` to be an index into the index array.
        *   Converter strategy: Cast `ushort` to `uint` for v17 output.

#### Debugging Setup (MCP-x64dbg)
To debug WMO loading in 0.5.3:
*   **Target**: `wowclient.exe` (0.5.3).
*   **Breakpoint**: `CMapObjGroup::CreateOptionalDataPointers` (`006af4d0` in standard mapping, verify with pattern `55 8B EC 83 EC 08 56 8B F1 8B 06`).
*   **Chunk Inspection**: The chunk data pointer is passed as `Arg1` (`[EBP+0x08]`) to this function. Note that this function processes *optional* chunks (after `MOBA`). Required chunks are parsed by the caller (`CMapObjGroup::ReadRequiredChunks`, near `006af...`).

---

## Cheat Commands
 (Ghidra-Verified)

All commands registered via `ConsoleCommandRegister` in category `DEBUG` or `GAME`.

| Command | Address | Syntax | Description |
|:---|:---|:---|:---|
| `speed` | 00832a9c | `speed <float>` | Set run speed multiplier |
| `walkspeed` | 00832a90 | `walkspeed <float>` | Set walk speed |
| `swimspeed` | 00832a84 | `swimspeed <float>` | Set swim speed |
| `turnspeed` | 00832a78 | `turnspeed <float>` | Set turn speed |
| `teleport` | 008540d0 | `teleport <x> <y> <z> [o]` | Teleport to coordinates |
| `worldport` | 00832a64 | `worldport <continent> [x y z] [facing]` | Change continent |
| `money` | 00832a5c | `money [copper]` | Set player money |
| `level` | 00832998 | `level <1-100>` | Set player level |
| `petlevel` | 0083298c | `petlevel <level>` | Set pet level |
| `beastmaster` | 008329bc | `beastmaster <on/off>` | Toggle beastmaster mode |
| `godmode` | 0085e494 | `godmode` | Toggle god mode |

### Quest Commands

| Command | Address | Description |
|:---|:---|:---|
| `flagquest` | 00832974 | Flag quest as active |
| `finishquest` | 0083291c | Mark quest as finished |
| `clearquest` | 00832980 | Clear quest from log |
| `questquery` | 00832900 | Query quest giver |
| `questaccept` | 008328f4 | Accept quest |
| `questcomplete` | 008328e4 | Complete quest |
| `questcancel` | 008328d8 | Abandon quest |

### GM Commands

| Command | Address | Description |
|:---|:---|:---|
| `ghost` | 00833144 | Enter ghost mode |
| `invis` | 00833138 | GM Invisibility ("Go GM Invis") |
| `nuke` | 0083301c | Forcibly remove player from server |
| `summon` | 008330c8 | Summon player to location |
| `showlabel` | 0083309c | Toggle showing 'GM' label |

### Debug CVars

| CVar | Address | Description |
|:---|:---|:---|
| `debugobjectpathing` | 00846b4c | Object pathing debug |
| `playercombatlogdebug` | 00865254 | Combat log debug |
| `CombatDebugShowFlags` | 0085e28c | Show combat debug flags |
| `debugTargetInfo` | 0083207c | Toggle target tooltips |
| `CombatDebugForceActionOn` | 0085e300 | Force combat action |

---

## Dead Code & Latent Tools
**MDL Exporter**: `007b3a7a` - Warcraft 3 model export header writer (unreachable)
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

# Part 5: Texture Limitations (Ghidra Verified)

## Texture Resolution & Formats
Analysis of `PumpBlpTextureAsync` (`00471a70`) and `CreateBlpTexture` (`004717f0`) reveals the true limitations of the 0.5.3 client.

### Resolution Limits
**There is NO hardcoded resolution limit (e.g. 256x256) in the client code.**

The client dynamically queries the hardware capabilities via `GxCaps`:
1.  **Hardware Query**: `RequestImageDimensions` (`00471a70` calls it) checks the requested texture size against `GxCaps.m_maxTextureSize`.
2.  **Auto-Downscaling**: If the texture exceeds the hardware limit, the client uses a **Box Filter** algorithm to downscale the texture in memory until it fits.
    *   *Logic*: `while (width > max || height > max) { width >>= 1; height >>= 1; bestMip++; }`
    *   *Implication*: You can feed the client 2048x2048 textures. If standard 2003 hardware (e.g. Voodoo3/TNT2) limit is 256, it will display at 256. If modern hardware (limit 16384) runs it, it displays at 2048.

### Texture Format Support
The 0.5.3 client supports two primary image formats:

#### 1. BLP (Blizzard texture)
*   **Version**: **BLP2** (Magic `0x32504C42`).
*   **Compression**: DXT1, DXT3, DXT5.
*   **Uncompressed**: ARGB8888, ARGB1555, ARGB4444, RGB565.
*   **Mipmaps**: Used if present, auto-generated via Box Filter if missing.

#### 2. TGA (Truevision Targa)
*   **Usage**: Supported for textures (e.g. valid `MTEX` path) and Screenshots.
*   **Format**: Uncompressed or RLE.
*   **Priority**: Checked *before* BLP in some loading routines, or explicitly if extension matches.
*   *Note*: No support for BMP, PNG, or JPEG.

### Verified String References
*   `"Textures\UnitSelectTexture.blp"`
*   `"Interface\SpellShadow\Spell-Shadow-Unacceptable.blp"`
*   `"UpdateBlpTextureAsync(): GxTex_Lock loading: %s\n"`
