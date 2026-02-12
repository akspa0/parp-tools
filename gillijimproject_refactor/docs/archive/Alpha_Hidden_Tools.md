# Alpha 0.5.3 Hidden Tools & Dead Code

**Analysis Date**: Jan 09 2026

## 1. Active Developer Tools
The 0.5.3 client contains fully functional developer tools accessible via the Console.

### 1.1 Sound Zone Editor ("SndDebug")
The client includes a suite of commands for creating and modifying Audio Zones (Reverb/Ambience) in-game. These commands are registered dynamically when entering/leaving dungeons.

**Commands:**
*   `SndDebugCreateChunk`: Create a new audio zone at current position.
*   `SndDebugSetChunkProperty`: Modify properties of the current audio zone.
*   `SndDebugSetCurrentChunk`: Select an audio zone.
*   `SndDebugShowCurrentChunk`: Display info about the active audio zone.
*   `SndDebugDumpChunks`: Dump all audio zone data to log.
*   `SndDebugListChunks`: List all loaded audio zones.

**Statue**: **ALIVE**. These commands are actively registered by `SndDebugDungeonTransition`.

### 1.2 Debug CVars
Standard debug variables are available, including:
*   `debugobjectpathing`
*   `playercombatlogdebug`
*   `CombatDebugShowFlags`

### 1.3 Cheat Commands (`InstallGameConsoleCommands`)
These commands are registered in the `DEBUG` or `GAME` categories and often send specific cheat opcodes to the server.

| Command | Syntax | Description | Opcode |
| :--- | :--- | :--- | :--- |
| `speed` | `speed <float>` | **Speed Hack**. Sets run speed locally on the client. | - |
| `teleport` | `teleport <x> <y> <z> [o]` | Teleports to coordinates. | `0xC6` |
| `teleport` | `teleport <name>` | Teleports to a unit by name. | `0x09` |
| `money` | `money <copper>` | Sets player money (in copper). | `0x24` |
| `level` | `level <1-100>` | Sets player level. | `0x25` |
| `beastmaster` | `beastmaster <on/off>` | Toggles Beastmaster mode. | `0x21` |
| `ci` | `ci <itemId>` | Create Item. | `0x13` |
| `cm` | `cm <creatureId>` | Create Monster. | `0x11` |
| `db` | `db <query>` | Looks up database records. | `0x02` |

### 1.4 Quest Commands (`InstallGameConsoleCommands`)
Debug commands for manipulating quest state.

| Command | Syntax | Description | Opcode |
| :--- | :--- | :--- | :--- |
| `flagquest` | `flagquest <questId>` | Flags a quest as active/accepted. | `0x2A` |
| `finishquest` | `finishquest <questId>` | Flags a quest as finished/ready to turn in. | `0x2B` |
| `clearquest` | `clearquest <questId>` | Clears a quest from the log/history. | `0x2C` |
| `questquery` | `questquery <giverGUID> <questId>` | Simulates querying a quest giver. | - |
| `questaccept` | `questaccept <giverGUID> <questId>` | Simulates accepting a quest. | - |
| `questcomplete` | `questcomplete <giverGUID> <questId>` | Simulates completing a quest. | - |
| `questcancel` | `questcancel <questId>` | Abandons a quest locally. | - |

### 1.5 GM Commands
The `InstallGMCommands` function actively registers the following commands:
*   `ghost`: Go to ghost mode.
*   `invis`: GM Invisibility.
*   `bindplayer`: Bind a player to a location.
*   `summon`: Summon a player.
*   `showlabel`: Toggle GM label.
*   `setsecurity`: Set security level.
*   `nuke`: Forcibly remove a player.

## 2. Latent / Dead Code
The binary contains code segments that appear to be statically linked libraries from development tools but are not reachable in the release build.

### 2.1 God Mode & Cheats
**Status**: **Mixed**.
*   **Logic**: The client has logic to display "Godmode enabled" in `OnPlayerEvent`.
*   **Strings**: "God", "godmode", "Cheat", "Fly" strings exist in data tables (`0080xxxx`).
*   **Command**: The actual registration call for `/god` seems to be stripped from the release build, unlike the GM commands above. It remains as a "ghost" of the debug suite.

### 2.2 MDL Exporter
**Location**: `007b3a7a` (`MDL::WriteHeaderComment`)
**Function**: Writes a Warcraft 3 `.mdl` file header (`// MDLFile version Dec 11 2003`).
**Status**: **DEAD**. No cross-references were found calling this function. It appears to be a leftover from the model pipeline library.

### 2.2 Zone Debugger
Strings like `D:\build\buildWoW\WoW\Source\Object\ObjectClient\ZoneDebug.cpp` suggest a Zone Debugging tool, potentially related to the `SndDebug` system or a separate trigger editor.

## 3. Conclusion
Alpha 0.5.3 is "dirty" with development code. Unlike the cleaner 0.5.5/0.6.0 builds, it exposes raw creation tools (Sound Editor) and contains unstripped linker artifacts (MDL Exporter), making it a goldmine for understanding the development pipeline.

---

# WoW 3.3.5a (Wrath of the Lich King) Hidden Tools & Dead Code

**Analysis Date**: Jan 10 2026  
**Binary**: `wow.exe` (3.3.5a Build 12340)

## 4. Active Developer/Debug Features (3.3.5a)

### 4.1 Godmode System
The client contains active Godmode toggle logic, likely server-side controlled:
*   **Strings**: `"Godmode enabled"`, `"Godmode disabled"`, `"Pet Godmode enabled/disabled"`
*   **Opcode**: `SPELL_FAILED_BM_OR_INVISGOD` - confirms server-side validation
*   **Status**: **Server-Gated**. Client displays messages but GM command requires server support.

### 4.2 Debug Console Commands
The console infrastructure is fully active:
*   `ConsoleExec("command")` - Execute console commands from Lua
*   `closeconsole` - Close console window
*   `cvarlist` - List all CVars
*   `cvar_reset` - Reset all CVars to default
*   `consolelines` - Set console line count

### 4.3 Debug Lua API
| Function | Description |
|:---|:---|
| `TeleportToDebugObject` | Teleport to a debug object (requires server) |
| `GetDebugZoneMap` | Get debug zone map info |
| `HasDebugZoneMap` | Check for debug zone map |
| `GetMapDebugObjectInfo` | Get debug object info |
| `GetNumMapDebugObjects` | Count debug objects |
| `IsDebugBuild` | Check if debug build |
| `GetDebugStats` | Get debug statistics |
| `CommentatorSetMoveSpeed(speed)` | Set spectator move speed |

### 4.4 Internal CVars
Hidden CVars for internal tracking:
*   `"Internal cvar for saving completed tutorials in order"`
*   `"Internal cvar for saving tracked achievements in order"`
*   `"Internal cvar for saving tracked quests in order"`

## 5. Legacy Code & Dead Artifacts (3.3.5a)

### 5.1 Warcraft 3 MDX/MDL Support
The client still contains references to Warcraft 3 model formats:
*   **File**: `Environments\Stars\stars.mdl` - Original WC3 skybox
*   **MDX Paths**: `Interface\Minimap\MinimapArrow.mdx`, `Spells\ErrorCube.mdx`
*   **Spell Effects**: Many spells reference `.mdx` files (`TalkToMe.mdx`, `Blizzard_Impact_Base.mdx`)
*   **Status**: **Active but Legacy**. The loader exists but assets are progressively replaced with M2.

### 5.2 SavedVariables System
Full Lua variable persistence:
*   `RegisterForSave` - Register addon for save
*   `RegisterForSavePerCharacter` - Per-character saves
*   `SaveAddOns` - Trigger addon save
*   Path: `\SavedVariables.lua`

## 6. Source File Paths (Debug Artifacts)
The binary contains debug path strings revealing internal structure:
*   `d:\buildserver\wow\...\NetInternal.h` - Network subsystem
*   `.\\ConsoleClient.cpp` - Console implementation
*   `.\\ConsoleVar.cpp` - CVar system
*   `.\\ConsoleDetect.cpp` - Hardware detection
*   `.\\SoundInterface2Internal.cpp` - Sound system

## 7. Comparison: 0.5.3 vs 3.3.5a

| Feature | Alpha 0.5.3 | WotLK 3.3.5a |
|:---|:---|:---|
| **Godmode** | Dead (stripped) | Active (server-gated) |
| **Sound Editor** | Active (SndDebug) | Removed |
| **MDL Exporter** | Dead (latent code) | Absent |
| **MDX Support** | Active | Legacy (still present) |
| **Console** | Full GM suite | Lua API only |
| **CVar System** | Basic | Full API (`GetCVar`, etc.) |
| **Debug Zone Maps** | N/A | Active API |
