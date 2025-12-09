# Progress – AlphaWdtInspector & Retroporting Tools

## ✅ Completed
- **Format Specification**: Created verified [memory-bank/specs/Alpha-0.5.3-Format.md](memory-bank/specs/Alpha-0.5.3-Format.md) based on battle-tested code and in-game verification.
- Plan and Memory Bank alignment for standalone diagnostic tool focus.
- Implemented `placements-sweep-diff`, `wdt-sanity`, and `mtex-audit` commands.
- Fixed `pack-alpha` MCIN backpatching (offsets are RELATIVE to `MHDR.data`).

## Update 2025-12-05 – BlpResizer & Asset Conversion Tools

### ✅ BlpResizer Tool (COMPLETE)
- **Location**: `BlpResizer/`
- **Purpose**: Downscale BLP tilesets from modern WoW (11.0+) to Alpha-compatible 256×256
- **Features**:
  - CASC extraction support (mounts WoW install directly)
  - Listfile auto-download from wow-listfile repo
  - Pattern filtering (e.g., `--pattern tileset`)
  - BLP2 output with DXT compression (Alpha 0.5.3 compatible)
  - Preserves original directory structure
- **Test Results**: 
  - Extracted 7956 tilesets from WoW 12.x CASC
  - 4156 resized (512/1024/2048 → 256×256)
  - 3832 already ≤256 (copied as-is)
  - Output looks correct, ready for Alpha client testing
- **Usage**: `BlpResizer --casc "I:\wow12\World of Warcraft" --output out/tilesets_12`

### ✅ WMO v17 → v14 Converter (NEW - NEEDS TESTING)
- **Location**: `WoWRollback/WMOv14/WMO2Q3/WmoBspConverter/Wmo/WmoV17ToV14Converter.cs`
- **Purpose**: Convert modern WMO (WotLK+) to Alpha v14 format
- **Status**: Implemented, needs validation against existing v14 parser

### ✅ M2 ↔ MDX Converter Framework (NEW - NEEDS TESTING)
- **Location**: `WoWRollback/WMOv14/MDXSupport/`
- **Files**:
  - `M2ToMdxConverter.cs` — Modern M2 → Alpha MDX
  - `MdxToM2Converter.cs` — Alpha MDX → Modern M2
- **Status**: Framework implemented, needs testing with real assets

### 📝 Historical Context: WMO → Q3 BSP Experiment
- **Location**: `WoWRollback/WMOv14/WMO2Q3/`
- **Discovery**: WMO format is essentially a restructured Quake 3 BSP
- **Attempt**: Tried converting WMO v14 → Q3 BSP to understand format
- **Result**: Almost worked, crashed Q3 on load (BSP tree/VIS data issues)
- **Outcome**: Built solid v14 parser and v14→v17 converter from this work

## 🔄 In Progress
- Port WoWRollback monolithic writer verbatim as `pack-alpha-mono` (MVER → MPHD(128) → MAIN → MDNM → MONM).
- Design a dedicated standalone LK→Alpha converter using reverse crosswalk CSVs with no asset gating.

## 🎯 Next
1. Overlay placements-only write path to swap MDDF/MODF from LK ADTs without touching textures/names.
2. Sketch CLI and core types for `AlphaLkToAlphaStandalone` (inputs: LK WDT/ADTs + crosswalks; outputs: Alpha WDT with full placements).
3. Re-run validations on new WDT: `wdt-sanity`, `mtex-audit`, `placements-sweep-diff`; then client test on WMO-heavy tiles like Deadmines 33_32.
4. Backport learnings to WoWRollback.Cli once the standalone pipeline is trusted.

## Update 2025-11-15 – AlphaLkToAlphaStandalone Roundtrip

### ✅ Completed
- Created `AlphaLkToAlphaStandalone` .NET 9 console app.
- Implemented `convert` command:
  - Supports filesystem LK trees and MPQ-based 3.3.5 client roots.
  - Emits `tiles_summary.csv`, `input_files.csv`, `run_summary.txt`, and a stub Alpha WDT.
- Implemented `roundtrip` command:
  - Auto-detects repo root, DBD dir, Alpha & LK DBC roots, and DBCTool.V2 output root from repo layout + `test_data`.
  - Alpha → LK export:
    - Uses `AlphaWdtAnalyzer.Core.AdtExportPipeline` + DBCTool.V2 crosswalk CSVs to write LK ADTs into `lk_export/World/Maps/<MapName>/`.
  - AreaID diagnostics:
    - Emits `areaid_roundtrip.csv` joining original Alpha MCNK "area IDs" with crosswalked LK `AreaId` values.

### 🔄 In Progress
- Designing a real LK→Alpha writer inside `AlphaLkToAlphaStandalone`:
  - Consume exported LK ADTs from `lk_export`.
  - Rebuild Alpha ADTs and a true monolithic Alpha WDT (not just a stub) while preserving placements and WMO UIDs.

### 🎯 Next
1. Implement the LK→Alpha ADT writer and plug it into the `roundtrip` second leg.
2. Add mismatch-only CSVs derived from `areaid_roundtrip.csv` to highlight problematic MCNKs/tiles.
3. Run golden-file style checks on a Kalidar subset (original Alpha vs roundtrip-reconstructed Alpha).
4. Backport LK→Alpha learnings into WoWRollback.Cli / related tooling once the standalone pipeline is trusted.

## Update 2025-12-09 – PM4 Pathfinding ↔ WMO Correlation & ADT Reconstruction

### 🎯 Current Focus
Reconstruct 3.3.5 ADT files with PM4-derived MODF placement data for Noggit visualization.

### ❌ Failed Approach (DO NOT REPEAT)
- Created `AdtLkFactory` to build ADTs from scratch
- Manually constructed MCNK chunks with minimal data
- Result: Invalid ADT files that crash viewers and Noggit
- **Lesson**: Don't reinvent the wheel - use existing proven code

### ✅ Correct Approach
1. **Read existing development ADT files** (split Cata+ format from `test_data/development/`)
2. **Convert to 3.3.5 monolithic format** using existing WoWRollback converters
3. **Patch MWMO/MWID/MODF chunks** with PM4 reconstruction data
4. **Write back** using existing proven `gillijimproject-csharp` writers

### 📚 Existing Code to Use (DO NOT MODIFY)
- `gillijimproject-csharp/WowFiles/LichKing/AdtLk.cs` - LK ADT reader/writer
- `gillijimproject-csharp/WowFiles/LichKing/McnkLk.cs` - MCNK chunk handling
- `WoWRollback.LkToAlphaModule/` - Format conversion builders
- `Warcraft.NET` - Modern ADT reading

### 🔧 Files Created (May Need Cleanup)
- `WoWRollback.Core/Services/PM4/AdtLkFactory.cs` - **BROKEN, creates invalid ADTs**
- `WoWRollback.Core/Services/PM4/SplitAdtMerger.cs` - Split→Monolithic merger (untested)

### 📝 Key Discoveries
- PM4 pathfinding geometry ≠ WMO render/collision geometry
- MODF.NameId = index into MWID (not byte offset into MWMO)
- MCNR has 13-byte padding after chunk data (handled by existing code)
- FourCC reversed on disk, data NOT reversed

### 🎯 Next Steps
1. **Delete or disable `AdtLkFactory`** - it creates broken files
2. **Implement proper split ADT → monolithic converter** using existing readers
3. **Patch only MWMO/MWID/MODF chunks** in converted ADTs
4. **Test with `development_22_18.adt`** - largest PM4 with most object instances
5. Use `development_29_39` as reference (has complete ADT + PM4 + _obj0.adt)
