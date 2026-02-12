# Enhanced WoW Archive Analysis Plan

## Overview
Expand WoWRollback to read directly from WoW client installations (MPQ archives + loose files), export DBCs, parse all maps (including WMO-only instances), and generate detailed terrain overlays.

## Current State
- ✅ Viewer overlays working (coordinates fixed)
- ✅ StormLibWrapper exists for MPQ reading
- ❌ No loose file priority handling
- ❌ No DBC export
- ❌ No WDT parsing (map type detection)
- ❌ Basic MCNK extraction only (header, no subchunks)

---

## Phase 1: Archive Reading Foundation

**Goal:** Read from WoW client folders with correct file resolution priority

### New Module: `WoWRollback.ArchiveReader`

```
WoWRollback.ArchiveReader/
├── IArchiveSource.cs          (Abstraction for MPQ + loose files)
├── MpqArchiveSource.cs        (Wraps existing StormLibWrapper)
├── FileSystemArchiveSource.cs (For extracted files)
├── PrioritizedArchiveSource.cs (Implements priority: loose > patch > base)
└── ArchiveLocator.cs          (Detect WoW installation, find wow.exe)
```

### File Resolution Priority (CRITICAL)
1. **Loose files in Data/ subfolders** (HIGHEST priority)
2. **Patch MPQs** (patch-3.MPQ > patch-2.MPQ > patch.MPQ)
3. **Base MPQs**

**Why:** WoW allows loose file overrides for modding. Players exploited this for model swapping (giant campfire models to escape areas). `md5translate.txt` can exist in both MPQ and `Data/textures/Minimap/md5translate.txt`.

### IArchiveSource Interface
```csharp
public interface IArchiveSource : IDisposable
{
    bool FileExists(string path);
    Stream OpenFile(string path);
    IEnumerable<string> EnumerateFiles(string pattern = "*");
}
```

### Implementation Strategy
- **MpqArchiveSource**: Wraps `StormLibWrapper/MpqArchive.cs`
- **PrioritizedArchiveSource**: Checks filesystem FIRST, then delegates to MpqArchiveSource
- **FileSystemArchiveSource**: Direct file access (for already-extracted files)

**Existing Infrastructure:**
- `StormLibWrapper/MpqArchive.cs` - Open/read MPQ
- `StormLibWrapper/MPQReader.cs` - Extract files  
- `StormLibWrapper/DirectoryReader.cs` - Auto-detect patch chain
- `MpqArchive.AddPatchArchives()` - Automatic patching

---

## Phase 2: DBC Export & Map Discovery

**Goal:** Export all DBCs to JSON, enumerate all valid maps from Map.dbc

### New Module: `WoWRollback.DbcExporter`

```
WoWRollback.DbcExporter/
├── DbcToJsonExporter.cs     (DBCD → JSON per version)
├── MapDbcParser.cs          (Parse Map.dbc entries)
└── Models/
    ├── MapEntry.cs
    └── AreaTableEntry.cs
```

### Key DBCs to Export
1. **Map.dbc** - All maps with internal names, display names, map IDs, instance types
2. **AreaTable.dbc** - Zone names for area overlays
3. **LoadingScreens.dbc** - Map metadata (optional)

### Output Structure
```
output/
└── dbcs/
    └── {version}/
        ├── Map.json
        ├── AreaTable.json
        └── LoadingScreens.json
```

### Map Type Detection
- **ADT-based**: Normal terrain with tiles (Azeroth, Kalimdor, Outland)
- **WMO-only**: Instances with single WMO (Karazhan, Scarlet Monastery, Deadmines)
- **Battlegrounds**: Special handling (Alterac Valley, Warsong Gulch)

---

## Phase 3: WDT Parsing

**Goal:** Detect map types and valid ADT tiles before processing

### New Module: `WoWRollback.WdtParser`

```
WoWRollback.WdtParser/
├── WdtReader.cs             (Parse WDT MPHD/MAIN/MWMO)
├── WmoOnlyMapHandler.cs     (Handle single-WMO maps)
└── Models/
    └── WdtInfo.cs
```

### WDT File Structure
```
WDT File:
├── MPHD chunk (header - flags indicate WMO-only)
├── MAIN chunk (64x64 tile grid - flags indicate which tiles exist)
└── MWMO chunk (WMO filename for WMO-only maps)
```

### Detection Logic
- **MPHD flags**: Check for GlobalWMO flag (WMO-only map)
- **MAIN grid**: Each entry has flags indicating if ADT exists
- **MWMO chunk**: Contains single WMO path for instances

### Benefits
- Prevents scanning for non-existent ADT files
- Handles instances correctly (Karazhan has no ADTs, only WMO)
- Detects partial maps (some tiles missing)

**Current Gap:** `analyze-map-adts` assumes all ADTs exist. Needs WDT pre-check.

---

## Phase 4: Detailed Terrain Analysis

**Goal:** Extract full MCNK data for advanced overlays (replaces `terrain_complete`)

### New Module: `WoWRollback.DetailedAnalysisModule`

```
WoWRollback.DetailedAnalysisModule/
├── McnkDetailedExtractor.cs       (Extract all MCNK subchunks)
├── TerrainDetailOverlayBuilder.cs (Generate detailed overlays)
└── Models/
    ├── McnkDetail.cs              (Heights, normals, textures, etc.)
    └── TerrainDetailOverlay.cs
```

### MCNK Subchunks to Extract
```
MCNK (256 per ADT, 16x16 grid):
├── MCVT (vertex heights - 145 vertices) → Height map overlay
├── MCNR (normal vectors) → Lighting/shading
├── MCLY (texture layers - up to 4) → Texture distribution overlay
├── MCAL (alpha maps) → Texture blending
├── MCLQ (liquid data) → Water/lava/slime overlay
├── MCRF (doodad/WMO references) → Object density
├── MCSH (shadow map) → Baked shadows
└── MCSE (sound emitters) → Ambient sound
```

### New Overlays
1. **Height map overlay** (MCVT → heatmap visualization)
2. **Texture layer overlay** (MCLY → show texture distribution)
3. **Liquid overlay** (MCLQ → water/lava regions)
4. **Impassable terrain overlay** (MCNK flags)
5. **Area boundaries** (MCNK AreaID changes)

**Current Gap:** `AdtTerrainExtractor` only extracts basic MCNK header (AreaID, flags, liquid size). Ignores all subchunk data.

---

## Phase 5: CLI Redesign

### New Command: `analyze-archive`

```bash
# Analyze from WoW client installation
dotnet run -- analyze-archive \
  --client-path "C:\Program Files (x86)\World of Warcraft" \
  --version "3.3.5" \
  --map "Azeroth" \
  --out "output" \
  --export-dbcs \
  --detailed-terrain

# Analyze all maps from a version
dotnet run -- analyze-archive \
  --client-path "C:\Program Files (x86)\World of Warcraft" \
  --version "3.3.5" \
  --all-maps \
  --out "output"
```

### CLI Arguments
- `--client-path`: Path to WoW installation
- `--version`: Version to analyze (detected from client)
- `--map`: Specific map (or `--all-maps`)
- `--export-dbcs`: Export DBCs to JSON
- `--detailed-terrain`: Extract full MCNK data
- `--serve`: Start HTTP server after generation
- `--port`: Server port (default 8080)

---

## Implementation Priority

### Must Do First
1. **Fix terrain extraction bug** (0 chunks issue)
2. **Remove `terrain_complete` viewer code** (broken, will be replaced)
3. **Create `IArchiveSource` + `PrioritizedArchiveSource`** (wrap existing MpqArchive)

### Then Build
4. DBC export (Phase 2)
5. WDT parsing (Phase 3)
6. Detailed MCNK extraction (Phase 4)
7. CLI redesign (Phase 5)

---

## Testing Strategy

### Test Scenarios
1. **All MPQ**: Pure MPQ read (no loose files)
2. **Mixed MPQ + loose**: Loose files override MPQ contents
3. **All loose**: Extracted files only
4. **WMO-only maps**: Karazhan, Scarlet Monastery (no ADTs)
5. **Partial maps**: Some tiles missing

### Test Maps
- **Azeroth**: Large ADT-based map (106 tiles)
- **Karazhan**: WMO-only instance
- **Alterac Valley**: Battleground (special handling)
- **development**: Custom test map

---

## Files to Clean Up

### Viewer (Remove)
- `ViewerAssets/js/overlays/terrainPropertiesLayer.js` - DELETE
- References to `terrain_complete` in `overlayManager.js` - REMOVE

### Backend (Keep but Expand)
- `AdtTerrainExtractor.cs` - Move to `DetailedAnalysisModule`, expand for subchunks

---

## Success Criteria
- ✅ Reads from WoW client installations (MPQ + loose files)
- ✅ Respects loose file priority (critical for accuracy)
- ✅ Exports DBCs to JSON per version
- ✅ Handles all map types (ADT, WMO-only, battlegrounds)
- ✅ Extracts full MCNK data (not just header)
- ✅ Generates detailed terrain overlays
- ✅ No more `terrain_complete` errors
- ✅ Works with Karazhan and other WMO-only maps
