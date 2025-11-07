# Technical Context - WoWRollback.RollbackTool
 
## Hot Update (2025-11-07) – Liquids & Placements diagnostics
- Modules touched: AlphaWdtMonolithicWriter (MDNM/MONM build from referenced union; placements MDDF/MODF; MCRF), AlphaMcnkBuilder (MH2O→MCLQ composition; MCAL pack logging), Program CLI (new flags).
- New flags: `--log-file`, `--log-dir` to tee console output to a timestamped log file near outputs.
- Diagnostics: kept assets CSV, `dropped_assets.csv`, `objects_written.csv` (per-tile MDDF/MODF counts and samples), `mclq_summary.csv`.
- Policies: do not gate placements; build name tables from union of referenced names; normalize `/`→`\\`, `.m2`→`.mdx`; placements axis X,Z,Y.

## Runtime Environment
- **Target Framework**: .NET 9.0
- **Platform**: Windows x64 (primary), cross-platform capable
- **Build System**: dotnet CLI with MSBuild
- **Dependencies**: 
  - `GillijimProject.WowFiles` (Alpha/LK ADT parsing) - **CRITICAL DEPENDENCY**
  - `System.Security.Cryptography` (MD5 checksums)
  - `System.Text.Json` (Analysis data, manifests)
  - `CsvHelper` (robust CSV parsing in GUI)

## Project Structure (Current)
```
WoWRollback/
├── WoWRollback.Cli/                # Primary CLI application
├── WoWDataPlot/                    # Legacy (viz/demo), contains previous rollback code
├── AlphaWDTAnalysisTool/           # Analysis module (reuse)
└── memory-bank/                    # Project documentation
```

## Current Implementation Location
CLI-first: `WoWRollback.Cli` is the primary entrypoint for Alpha→LK, analysis, and viewer serving. Orchestrator is legacy.

## Critical Dependencies (PROVEN WORKING)

### GillijimProject.WowFiles
**Location**: `src/gillijimproject-csharp/WowFiles/`

**Key Classes Used**:
- `WdtAlpha` - Parse Alpha WDT files
  - `GetAdtOffsetsInMain()` - Returns offset array for 64x64 grid
  - Handles MVER, MPHD, MAIN chunks
  
- `AdtAlpha` - Parse embedded ADT data
  - **NEWLY ADDED**: `GetMddf()` - Access M2 placement chunk
  - **NEWLY ADDED**: `GetModf()` - Access WMO placement chunk
  - **NEWLY ADDED**: `GetMddfDataOffset()` - Calculate file position
  - **NEWLY ADDED**: `GetModfDataOffset()` - Calculate file position
  - **NEWLY ADDED**: `_adtFileOffset` field - Track position in file
  
- `Mddf` - M2 placement chunk
  - Inherits from `Chunk` (has `Data` byte array)
  - 36 bytes per entry
  
- `Modf` - WMO placement chunk
  - Inherits from `Chunk` (has `Data` byte array)
  - 64 bytes per entry

- `McnkAlpha` - Terrain chunk (256 per ADT)
  - Contains `McnkAlphaHeader` with `Flags` and `Holes` fields
  - Each MCNK = 33.33 yards square
  - Alpha header offsets used in CLI: `M2Number @ +0x14`, `McrfOffset @ +0x24`, `WmoNumber @ +0x3C`, `Holes @ +0x40`

### LK Writers
**Location**: `src/gillijimproject-csharp/WowFiles/LichKing/`
- `AdtLk` - Construct/write LK ADTs
  - `ToFile(<dir or path>)` treats a directory argument as output folder; writes `<map>_<x>_<y>.adt`
- `McnkLk` - LK MCNK builder
  - `ComputePositionFromAdt(adtNum, idxX, idxY)` used to set `PosX/PosY/PosZ`

### MPQ Access
**Location**: `WoWRollback/WoWRollback.Core/Services/Archive/`
- `PrioritizedArchiveSource` - Union of loose files + MPQs
- `MpqArchiveSource` - Uses `MPQToTACT.MPQ` to open files like `DBFilesClient/AreaTable.dbc`
- Enumeration optional; we can read known paths directly

### CLI (WoWRollback.Cli)
- Burial/terrain:
  - `--max-uniqueid`, `--bury-depth`, `--fix-holes`, `--holes-scope`, `--holes-wmo-preserve`, `--disable-mcsh`
- Crosswalks and mapping:
  - `--auto-crosswalks`, `--copy-crosswalks`, `--report-areaid`, `--strict-areaid` (default strict), `--chain-via-060` (opt-in pivot)
  - Inputs precedence: `--src-dbc-dir` / `--lk-dbc-dir` override `--src-client-path` / `--lk-client-path`
- LK export and WDT:
  - Convert Alpha ADTs → LK ADTs and always write `<Map>.wdt` into LK output folder
- AreaID patching (in-place after write):
  - Load CSV crosswalks; derive `currentMapId` via `Map.dbc` guard
  - Patch `MCNK.AreaId` at `mcnkOffset + 8 + 0x34` using non-pivot order; drop cross-map results

### Planned Commands
- `alpha-to-lk`: orchestrate rollback + area-map generation/usage + LK export
- `lk-to-alpha` (v1): patch LK ADTs with bury/holes/mcsh; (v2) consider reverse to Alpha WDT

## Data Flow Architecture (Implemented)
```
Input WDT → WdtAlpha.Load() → Get ADT offsets
  ↓
For each ADT:
  AdtAlpha(path, offset, adtNum) → Parse MHDR
    ↓
  GetMddf() / GetModf() → Access placement chunks
    ↓
  Modify chunk.Data in memory → Change Z coordinates
    ↓
  GetMddfDataOffset() → Calculate writeback position
    ↓
  Array.Copy(chunk.Data → wdtBytes) → Update file bytes
  ↓
Write wdtBytes → Output file
Generate MD5 → {mapName}.md5
```

## File Format Details (VERIFIED via Testing)

### Alpha 0.5.3 WDT Structure
```
WDT File:
├── MVER chunk (version = 18)
├── MPHD chunk (header flags)
├── MAIN chunk (64x64 grid of offsets)
└── [ADT Data Embedded Inline]
    ├── ADT #0 @ offset[0]
    │   ├── MHDR (64 bytes, offsets to sub-chunks)
    │   ├── MCIN (4096 bytes, MCNK index)
    │   ├── MTEX (texture paths)
    │   ├── MMDX (M2 model paths)
    │   ├── MMID (M2 model indices)
    │   ├── MWMO (WMO paths)
    │   ├── MWID (WMO indices)
    │   ├── MDDF (M2 placements) ← MODIFY THIS
    │   ├── MODF (WMO placements) ← MODIFY THIS
    │   └── MCNK chunks (256 per ADT)
    ├── ADT #1 @ offset[1]
    └── ...
```

### MDDF Entry Layout (36 bytes) - VERIFIED
```
+0x00 (4 bytes): nameId       - Index into MMDX string table
+0x04 (4 bytes): uniqueId     - ✅ FILTER CRITERION
+0x08 (4 bytes): position X   - World coordinate
+0x0C (4 bytes): position Z   - ✅ MODIFY THIS TO BURY
+0x10 (4 bytes): position Y   - World coordinate
+0x14 (4 bytes): rotation X
+0x18 (4 bytes): rotation Y
+0x1C (4 bytes): rotation Z
+0x20 (2 bytes): scale (fixed point 1024)
+0x22 (2 bytes): flags
```

### MODF Entry Layout (64 bytes) - VERIFIED
```
+0x00 (4 bytes): nameId       - Index into MWMO string table
+0x04 (4 bytes): uniqueId     - ✅ FILTER CRITERION
+0x08 (4 bytes): position X   - World coordinate
+0x0C (4 bytes): position Z   - ✅ MODIFY THIS TO BURY
+0x10 (4 bytes): position Y   - World coordinate
+0x14 (4 bytes): rotation X
+0x18 (4 bytes): rotation Y
+0x1C (4 bytes): rotation Z
+0x20-0x3F: bounding box, flags, doodad set, name set, etc
```

### MCNK Header (128 bytes) - FOR HOLE MANAGEMENT
```
+0x00 (4 bytes): Flags        - Various flags
+0x04 (4 bytes): IndexX       - X position in 16x16 grid
+0x08 (4 bytes): IndexY       - Y position in 16x16 grid
...
+0x40 (4 bytes): Holes        - ✅ CLEAR THIS FOR BURIED WMOs
...
+0x64 (4 bytes): McnkChunksSize
```
**Holes Field**: 16 bits representing 4x4 grid of 2x2 hole areas

## Output Formats

### Analysis JSON Schema (Planned)
```json
{
  "mapName": "Azeroth",
  "totalPlacements": 9234567,
  "minUniqueId": 1,
  "maxUniqueId": 982345,
  "percentiles": {
    "p10": 12000,
    "p25": 45000,
    "p50": 150000,
    "p75": 450000,
    "p90": 750000
  },
  "suggestedThresholds": [
    {"version": "0.5.3", "maxUniqueId": 10000},
    {"version": "0.6.0", "maxUniqueId": 50000},
    {"version": "1.0.0", "maxUniqueId": 200000}
  ]
}
```

### Overlay Manifest Schema (Planned)
```json
{
  "mapName": "Azeroth",
  "overlays": [
    {
      "threshold": 10000,
      "imagePath": "azeroth_uid_0-10000.png",
      "keptCount": 1234,
      "buriedCount": 8000
    }
  ]
}
```

### Output Directory Structure
```
output/
├── {mapName}.wdt              # Modified WDT
├── {mapName}.md5              # MD5 checksum
└── analysis/
    ├── {mapName}.json         # Analysis data
    └── overlays/
        ├── {mapName}_uid_0-10000.png
        ├── {mapName}_uid_0-50000.png
        └── overlay-index.json
```

## Proven Implementation (TESTED & WORKING!)

### Core Rollback - ✅ COMPLETE
- Load Alpha 0.5.3 WDT files into memory
- Parse embedded ADT data via offset array
- Modify MDDF/MODF chunk data in-place
- Write modified bytes back to file
- Generate MD5 checksums

### Test Results (Actual)
```
Kalimdor 0.5.3:
  - File size: ~380 MB
  - ADT tiles: 951
  - Total placements: 126,297
  - Processing time: ~30 seconds
  - Result: SUCCESS ✅

Azeroth 0.5.3:
  - Multiple successful tests
  - All modifications verified
  - MD5 generation confirmed
  - Result: SUCCESS ✅
```

### Additional Completed Features
- MCNK Hole Management — COMPLETE (MCRF-gated clearing of Holes only when all referenced placements are buried)
- MCSH Shadow Disabling — COMPLETE (optional zeroing of MCSH payloads)

## Implementation Gaps (TO-DO)

### Overlay Generation - ⏳ PLANNED
- Load minimap BLP tiles
- Plot placements (green/red)
- Save as PNG per threshold
- Generate manifest JSON

### Viewer (analyze + serve)
- `analyze-map-adts --input-dir …` produces viewer artifacts (placements/terrain/meshes/overlays)
- `serve-viewer [--viewer-dir … --port … --no-browser]` serves the viewer; auto-detects common locations

## Performance Characteristics

### Memory Usage
- **Current**: Load entire WDT into memory (~380 MB for Kalimdor)
- **Peak**: ~2x file size during modification
- **Optimization**: Stream processing not needed (WDTs are manageable)

### Processing Speed
- **Small maps** (< 100 tiles): < 5 seconds
- **Large maps** (900+ tiles): ~30 seconds
- **Bottleneck**: File I/O, not computation

### Scalability
- **Tested Maximum**: 951 tiles (Kalimdor)
- **Expected Maximum**: ~1500 tiles (theoretical WDT limit)
- **No Parallel Processing Needed**: Sequential is fast enough

## Cache Reuse & BYOD Policy (Added 2025-10-27)
- Preflight cache checks and skip-if-exists behavior are first-class:
  - Reuse LK ADTs if present and complete; allow “Force rebuild” when necessary.
  - Reuse DBCTool crosswalks (`compare/v2`) when present; they are static.
  - Reuse `tile_layers.csv` and `layers.json` when present; GUI supports fallback from `<map>_tile_layers.csv`.
- BYOD: tooling must not include copyrighted game assets; all DBC/DBD/client paths are user-supplied.

## CASC/DB2 Integration (2025-10-29) — Implemented
- CASC reader: `CascArchiveSource` parses community listfiles (delims: `;`, `,`, `\t`), normalizes case, and supports FDID lookups.
- Map discovery: use DBCD with a CASC-backed provider to load `Map.db2`; fallback to WDT scan via `EnumerateFiles("world/maps/*/*.wdt")`.
- Product detection: `.build.info` parsing (wow, wowt, wow_beta) when not explicitly provided.

## Listfile Services (2025-10-29)
- `ListfileIndex` / `ListfileCatalog`: in-memory indices (path↔FDID) and multi-alias registry.
- `ListfileSnapshot` (JSON): persisted snapshots with `{ path, fdid? }` entries and metadata.
- `AssetGate`: filters asset names against a target (e.g., 3.3.5) listfile; writes `dropped_assets.csv`.
- CLI utilities:
  - `snapshot-listfile --client-path <dir> --alias <major.minor.patch.build> --out <json>`
  - `diff-listfiles --a <fileA> --b <fileB> --out <dir>`
  - `pack-monolithic-alpha-wdt --lk-wdt <file> --out <wdt> [--target-listfile <335>] [--strict-target-assets true|false]`

## Alias Policy (2025-10-29)
- Aliases are the full build string `major.minor.patch.build`.
- Sources: CASC `.build.info`, DBD definitions, or path heuristics.

## CASC Asset Source (Planned)
- Integration: WoWFormatLib/CascLib with listfile support
- Priority: honor Loose > CASC > MPQ via a prioritized source wrapper
- Build detection: infer from `.build.info` or install path; allow explicit override
- DBCD enrichment: use wow.tools.local DBCD (ProjectReference) with provided DBD directory and DBC via DBFilesClient from CASC or loose files
- Outputs: same cache schema as filesystem (placements.csv, tile_layers.csv, layers.json, areas.csv)

## MPQ Archive Infrastructure (Added 2025-10-12)

### Existing Infrastructure
Located in `lib/WoWTools.Minimaps/StormLibWrapper/`:
- **MpqArchive.cs** - C# wrapper for StormLib (open/read MPQ files)
- **MPQReader.cs** - Extract files from MPQ archives
- **DirectoryReader.cs** - Auto-detect and sort patch chain
- **MpqArchive.AddPatchArchives()** - Automatic patch application

### WoW File Resolution Priority (CRITICAL)
WoW reads files in this exact order:
1. **Loose files in Data/ subfolders** (HIGHEST priority)
2. **Patch MPQs** (patch-3.MPQ > patch-2.MPQ > patch.MPQ)
3. **Base MPQs** (lowest priority)

**Why This Matters:**
- Players exploited loose file overrides for model swapping (giant campfire = escape geometry)
- `md5translate.txt` can exist in BOTH MPQ and `Data/textures/Minimap/md5translate.txt`
- **Any archive reader MUST check filesystem BEFORE MPQ**

### Implementation Requirements
```csharp
// Required abstraction (not yet implemented)
interface IArchiveSource {
    bool FileExists(string path);
    Stream OpenFile(string path);
}

// Priority wrapper (MUST implement)
class PrioritizedArchiveSource : IArchiveSource {
    // 1. Check loose file in Data/ folder FIRST
    // 2. If not found, delegate to MpqArchive
}
```

### Patch Chain Handling
- `DirectoryReader.cs` automatically detects and sorts patch MPQs
- Higher-numbered patches override lower (patch-3 > patch-2 > patch-1 > base)
- `MpqArchive.AddPatchArchives()` applies patch chain automatically

### Current Gap
- StormLibWrapper exists but not integrated with WoWRollback
- No loose file priority layer implemented
- No IArchiveSource abstraction

## WDT and Map Type Detection (Added 2025-10-12)

### WDT File Structure
```
WDT File:
├── MPHD chunk (header with flags)
├── MAIN chunk (64x64 tile grid, flags indicate which ADTs exist)
└── MWMO chunk (WMO filename for WMO-only maps)
```

### Map Types
1. **ADT-based**: Normal terrain (Azeroth, Kalimdor, Outland)
2. **WMO-only**: Instances with single WMO (Karazhan, Scarlet Monastery, Deadmines)
3. **Battlegrounds**: Special handling (Alterac Valley, Warsong Gulch)

### Detection Logic
- **MPHD.GlobalWMO flag** - Indicates WMO-only map (no ADTs)
- **MAIN grid entries** - Each has flags indicating if ADT exists
- **MWMO chunk** - Contains WMO path for instances

**Benefit:** Prevents scanning for non-existent ADT files, handles instances correctly

### Current Gap
- `analyze-map-adts` assumes all ADTs exist
- No WDT pre-check before ADT processing
- Can't handle Karazhan and other WMO-only maps

## MCNK Terrain Data (Added 2025-10-12)

### MCNK Structure
Each ADT has 256 MCNK chunks (16x16 grid). Each chunk has subchunks:
- **MCVT** - 145 vertex heights (enables height map overlays)
- **MCNR** - Normal vectors (lighting/shading)
- **MCLY** - Texture layers (up to 4 per chunk with blend modes)
- **MCAL** - Alpha maps (texture blending)
- **MCLQ** - Liquid data (water/lava/slime types, heights, flags)
- **MCRF** - Doodad/WMO references in this chunk
- **MCSH** - Shadow map (baked shadows)
- **MCSE** - Sound emitters (ambient sound)

### Current Implementation
`AdtTerrainExtractor.cs` only extracts basic MCNK header:
- AreaID, Flags, TextureLayers count, HasLiquids, HasHoles, IsImpassible

**Missing:** All subchunk data (MCVT, MCNR, MCLY, MCAL, MCLQ, MCRF, MCSH, MCSE)

### Planned Enhancement
New module `WoWRollback.DetailedAnalysisModule` will extract full MCNK data for:
- Height map overlays (MCVT heatmaps)
- Texture distribution overlays (MCLY)
- Liquid region overlays (MCLQ)
- Impassable terrain overlays (MCNK flags)
- Area boundary overlays (AreaID changes)

## Archaeological Terminology Integration
- **Console Output**: Uses archaeological language ("excavation", "preservation", "artifacts")
- **Variable Naming**: "sedimentary layers", "volumes of work", "ancient developers"
- **Documentation**: Consistently applies archaeological metaphor

---

## 2025-10-30 – New Services & Commands

### StatsService (Global Heatmap)
- Scans `<cacheBuild>/*/tile_layers.csv` to compute global min/max UniqueIDs.
- Writes `heatmap_stats.json` at the build root with `{ minUnique, maxUnique, perMap:{...}, generatedAt }`.
- GUI reads this when Heatmap Scope = Global.

### FdidResolver
- Loads community listfile (CSV) and JSON snapshots from `snapshot-listfile`.
- Normalizes paths; maps path↔FDID; exposes resolver APIs.
- CSV enrichment: add `fdid` and canonical `asset_path` to placements/layers CSVs.
- Diagnostics: emit `unresolved_paths.csv`.

### MCCV Analyzer
- New CLI `analyze-mccv --client-path <dir> (--map <m>|--all-maps) --out <dir>`.
- Outputs `mccv_presence.csv` and optional per-tile PNGs under `<map>/mccv/`.
- PNGs are decoded from MCCV BGRA entries (9*9 + 8*8) into a stitched 256‑chunk image.

### Tile Presence CSV
- `tile_presence.csv` summarizing: `has_minimap, has_placements, has_terrain, has_mccv`.
- Drives GUI empty‑tile gridlines when no placements exist.

### GUI WDT Handling for CASC
- CASC dataset: prefer `LkClient` for WDT lookup; if missing, show file picker.
- Loose/Install: use dataset root WDT scan as before.
