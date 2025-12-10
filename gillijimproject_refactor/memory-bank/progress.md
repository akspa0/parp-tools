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

## Update 2025-12-09 – ADT Merger & WDL→ADT Generator ✅ COMPLETE

### ✅ ADT Merger (`WoWRollback.PM4Module`)
Successfully merges split 3.3.5 ADTs into monolithic format:
- **352 tiles** processed from `test_data/development/`
- **Texture path normalization**: backslash → forward slash, uppercase → lowercase
- **MCCV vertex colors**: Generates default neutral colors (0x7F7F7F00)
- **MCNK header flags**: Sets `has_mccv` (0x40) when MCCV present
- **WDT generation**: Correct flags (0x0E = MCCV | BigAlpha | DoodadRefsSorted)

**Output**: `PM4ADTs/clean/` - 352 merged ADTs + WDT

### ✅ WDL→ADT Generator
Generates ADT terrain from WDL low-resolution heights:
- **1144 ADTs** generated to fill gaps (tiles without existing ADT data)
- **352 tiles** skipped (already have real ADT data)
- **Interpolation**: WDL 17×17 grid → ADT 145 vertices per chunk

**Usage**:
```bash
dotnet run --project WoWRollback/WoWRollback.PM4Module -- wdl-to-adt \
  --in <wdl-file> --out <dir> --map <name> --fill-gaps --existing <real-adts>
```

**Output**: `PM4ADTs/wdl_generated/` - 1144 generated ADTs

### Key Files
| File | Purpose |
|------|---------|
| `WoWRollback.PM4Module/AdtPatcher.cs` | ADT merger with MCCV fix |
| `WoWRollback.PM4Module/WdlToAdtTest.cs` | WDL→ADT generator |
| `WoWRollback.PM4Module/WdlToAdtProgram.cs` | CLI for WDL→ADT |
| `WoWRollback.Core/Services/PM4/Wdt335Writer.cs` | WDT generator |

### Technical Notes
- **MCCV format**: 145 entries × 4 bytes (BGRA), 0x7F = 1.0 (neutral)
- **WDL MARE**: 17×17 outer + 16×16 inner heights per tile
- **ADT MCNK**: 145 vertices (9×9 + 8×8 interleaved) per chunk
- **Bilinear interpolation** from WDL grid to ADT resolution

### 🎯 Next Steps
1. **Test in Noggit** - Verify merged ADTs load correctly with textures
2. **PM4 object patching** - Add MODF/MDDF from PM4 reconstruction
3. **Combine outputs** - Merge `clean/` and `wdl_generated/` into complete map
