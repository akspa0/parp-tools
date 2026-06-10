# Alpha 0.5.3 Client Analysis (Build 3368)

**Date**: Dec 11 2003
**Codename**: Alpha 0.5.3 (Internal 3368)

## 1. Core Architecture
*   **Audio**: **Miles Sound System (MSS)**. (Strings: `miles %d%%`). This is the last client to use MSS before FMOD (0.5.5).
*   **UI System**: **XML / Lua 5.0**. The modern UI engine was already in place, contrary to popular belief that it was "Hardcoded". Strings like `GlueXML.toc` and Lua copyrights confirm this.
*   **Database**: **WDBC** (Standard).

## 2. File Formats

### 2.1 Maps (WDT)
*   **Format**: **Monolithic WDT** (Whole map in one file).
*   **Chunk Sequence**: `MVER` -> `MPHD` -> `MAIN` -> *Data* -> optional `MODF`.
*   **ADT Support**: **None**. The client reads the `MAIN` chunk (Terrain Info) directly into memory and stops. There is no logic to load external `.adt` tiles.
*   **Placement**: It supports a single `MODF` chunk at the end of the file, likely for placing a global WMO (e.g. a Dungeon or Main Menu background).

### 2.2 Textures (BLP)
*   **Format**: **BLP2** (Magic `0x32504C42`).
*   **Type**: Strictly **Type 1** (Uncompressed/DXT).
*   **Legacy**: It does **not** support BLP1 (Jpeg). The loader returns failure if the magic is not BLP2.

### 2.3 Models (MDX/M2)
*   **MDX**: "MDLFile version Dec 11 2003" suggests active development/support for MDX v800 assets. Strings like `Spells\*.mdx` are prevalent.
*   **M2**: Support is implied by the presence of `MDDF` (M2 Placement) chunk logic in the WDT loader, though explicit `.m2` extension strings are missing (likely constructed at runtime or using `.mdx` as a placeholder).

### 2.4 Configuration (.wtf)
*   **WTF**: The client uses `.wtf` files (`Config.wtf`, `autoexec.wtf`, `realmlist.wtf`) for configuration, identical to modern clients.

### 2.5 Other formats
*   **WDL**: Low-res terrain (`.wdl`) is supported.
*   **WMO**: Standard World Map Objects (`.wmo`) supported.
*   **Esoteric**: No evidence found for `.wlw`, `.wlm`, `.wlq` or other exotic extensions.

## 3. Write Operations & Editor Leftovers
The client contains code capable of writing to disk, primarily for logging and configuration, but also contains traces of development tools.

*   **Active Writes**:
    *   **Configuration**: `Config.wtf`, `KB.wtf` (Keybindings).
    *   **Logs**: `GlueXML.log`, `Sound.log`, `calldump.log` (Crash Dumps).
    *   **Debug**: `BaseFileCacheDump.txt` (Memory/Cache dumps).
*   **Latent Code**:
    *   **Model Exporter**: A function at `007b3a7a` (`MDL::WriteHeaderComment`) contains strings like `"// Exported on %s"` and `"// SCENE FILENAME: %s"`. This strongly suggests the client contains code from a Model Exporter (likely for the `.mdl` -> `.mdx` pipeline), though it appears unreachable in the retail build.

## 4. Comparison with 0.5.5
| Feature | Alpha 0.5.3 | Alpha 0.5.5 |
| :--- | :--- | :--- |
| **Audio** | Miles (MSS) | FMOD |
| **Map** | Monolithic WDT | Monolithic WDT (w/ Prototype ADT Code) |
| **Textures** | BLP2 | BLP2 |
| **UI** | XML/Lua 5.0 | XML/Lua 5.0 |

**Conclusion**: 0.5.3 is the foundational "Modern" Alpha. 0.5.5 is a direct iteration that swapped the Audio Engine and secretly prototyped the ADT format.
