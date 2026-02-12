# Supported File Formats (Alpha 0.6.0)

**Based on String Analysis of WoW Alpha 0.6.0 Client**

This document lists all file extensions recognized or used by the Alpha 0.6.0 client, categorized by their subsystem.

## 1. World & Assets
These are the core formats for the game world and rendering.

| Extension | Format | Description | Notes |
| :--- | :--- | :--- | :--- |
| **.adt** | World Chunk | Terrain/Map data (16x16 grid). | **New in 0.6.0**. |
| **.wmo** | World Map Object | Buildings/Dungeons. | v16 Format (Hybrid). |
| **.m2** | M2 Model | Character/Item models. | Native WoW format. |
| **.mdx** | MDX Model | Warcraft 3 Models. | **Legacy Support Verification**:
*   Loader checks for `.mdx`/`.mdl` extension.
*   Validates `MDLX` header token.
*   Parses standard chunks (`MTLS`, `TEXS`, `GEOS`, `LITE`, etc.).
*   `TEXS` entry size (268 bytes) implies **Version 800** (WC3 RoC/TFT). |
| **.blp** | BLP Texture | BLP1 (likely). | Main texture format. |

## 2. Database & Cache
Formats used for static game data and local caching.

| Extension | Format | Description | Notes |
| :--- | :--- | :--- | :--- |
| **.dbc** | Client DB | Static data tables (Spells, Items, etc.). | Massive list found (AreaPOI, CharClasses, etc.). |
| **.wdb** | World DB Cache | Local cache of server query results. | `creaturecache.wdb`, `itemcache.wdb`, etc. |

## 3. User Interface (UI)
Formats used to define the Look & Feel.

| Extension | Format | Description | Notes |
| :--- | :--- | :--- | :--- |
| **.xml** | XML Layout | Frame definitions. | `Bindings.xml`, `GlueXML`. |
| **.lua** | Lua Script | UI Logic. | Lua 5.0 Strings found. |
| **.toc** | Table of Contents | Addon metadata. | `GlueXML.toc`, `FrameXML.toc`. |
| **.ttf** | TrueType Font | Fonts. | `FRIZQT__.TTF`, `ARIALN.ttf`. |
| **.html** | HTML | Rich text/Web content. | Only `tos.html` found (Terms of Service). |

## 4. Miscellaneous / Logs
| Extension | Format | Description |
| :--- | :--- | :--- |
| **.tga** | Targa Image | Screenshots, UI masks (`TempPortraitAlphaMask.tga`). |
| **.txt** | Text File | Logs (`WOWChatLog.txt`, `RenderLog.txt`) and Config (`wtfdir.txt`). |

## Missing / Implied Formats
*   **Audio**: No explicit `.mp3` or `.wav` strings were found because the engine uses **FMOD** (`_FSOUND_Init`, etc.). FMOD internally handles `.mp3`, `.ogg`, `.wav`, `.mid`, and `.dls` formats, explaining their absence from the main binary string table.
*   **WDT**: `.wdt` strings were not explicitly found, but are implied by the WMO/ADT system.
