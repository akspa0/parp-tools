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

## Update 2025-12-09/10 – ADT Merger & WDL→ADT Generator ✅ COMPLETE

### ✅ ADT Merger (`WoWRollback.PM4Module`)
Successfully merges split 3.3.5 ADTs into monolithic format:
- **352 tiles** processed from `test_data/development/`
- **Texture path normalization**: backslash → forward slash, uppercase → lowercase
- **MCCV vertex colors**: Generates default neutral colors (0x7F7F7F00)
- **MCNK header flags**: Sets `has_mccv` (0x40) when MCCV present
- **WDT generation**: Correct flags (0x0E = MCCV | BigAlpha | DoodadRefsSorted)
- **MODF placements**: Preserved from original `_obj0.adt` files

**Output**: `PM4ADTs/clean/` - 352 merged ADTs + WDT (with placements)

### ✅ WDL→ADT Generator (Noggit-compatible)
Generates ADT terrain from WDL low-resolution heights:
- **1144 ADTs** generated to fill gaps (tiles without existing ADT data)
- **352 tiles** skipped (already have real ADT data)
- **Interpolation**: WDL 17×17 grid → ADT 145 vertices per chunk

**Noggit Compatibility Fixes (2025-12-10):**
- Fixed MHDR offsets: Write all chunk offsets at correct positions (ofsMcin at 0x04, etc.)
- Fixed MCNK header: Exactly 128 bytes matching noggit's `MapChunkHeader` struct
- Fixed MCNK subchunk offsets: Relative to MCNK chunk start (ofsHeight = 0x88)
- Fixed WDT tile flags: Only mark tiles with actual ADT files

**Output**: `PM4ADTs/wdl_generated/` - 1144 generated ADTs (terrain only, no placements)

### ✅ Combined Map (`PM4ADTs/combined/`)
- **1496 total ADTs** (352 clean + 1144 WDL gap-fill)
- **IMPORTANT**: Copy order matters - clean ADTs first, then WDL (clean takes priority)
- Ready for Noggit testing

### Noggit Testing Results (2025-12-10)
- ✅ **MCCV working**: Vertex colors from original split ADTs transfer correctly (yellow/green terrain visible)
- ✅ **Textures working on clean ADTs**: Merged ADTs show proper texturing
- ❌ **WDL ADTs have no textures**: Gap-fill terrain shows only default salmon/peach color (no MTEX/MCLY data)
- ❌ **No minimap data on WDL ADTs**: Need to add minimap textures for visual verification of object placements

### 🎯 Next Priority: Add Minimap/Texture Data to WDL ADTs
WDL-generated ADTs need basic texture data so we can:
1. Visually distinguish terrain in Noggit
2. Verify object placements once PM4 MODF patching is working
3. Have a usable base map for further development

Options:
- Generate simple solid-color MTEX/MCLY layers
- Use minimap BLP textures as terrain texture source
- Copy texture layers from nearest clean ADT tiles

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
- **Noggit offset convention**: Adds 0x14 to MHDR offsets (MVER header + MHDR header)
- **MCNK subchunk offsets**: Relative to MCNK chunk start (including 8-byte header)

## 🔄 In Progress – PM4 Placement Reconstruction

### Goal
Patch ADTs with MODF/MDDF placement data from PM4 reconstruction to restore missing WMO/M2 assets.

### Status (2025-12-10)
- **Tooling**: ✅ **COMPLETE**
    - `development-repair` command automates the full pipeline.
    - Smart caching reduces iteration time to seconds.
    - `SplitAdtMerger` fixed to handle 3.3.5 texture chunks correctly.
    - `AdtModfInjector` fixed to inject into merged ADTs correctly.
    - Verification pipeline (`chunk_dump`) proves chunks are being written.
- **Data**: ❌ **BROKEN (Domino Effect)**
    - The source ADTs (`PM4ADTs/combined`) were generated by older, buggy code.
    - Even with the new fixes, `development-repair` is merging garbage data.
    - **BLOCKER**: Source ADTs must be regenerated from scratch using the fixed `wdl-to-adt` and `adt-merge` tools.

### PM4 Reconstruction Data Available
- Location: `test_data/development/.../pm4_15_37_output/pm4faces_batch/development_15_37/modf_reconstruction/`
- `modf_entries.csv` - 1003 MODF entries with PM4-relative coordinates
- `mwmo_names.csv` - 254 unique WMO paths
- `ck_instances.csv` - PM4 object instances with tile assignments (e.g., `t36_24`)

### ✅ RESOLVED: PM4 Coordinate System
The `modf_entries.csv` coordinates are now correctly transformed to ADT world space:
- **Transform Found**: `World.X = PM4.X + 17066.666`, `World.Y = 17066.666 - PM4.Y`, `World.Z = PM4.Z`
- **Implementation**: Wired into `modf-reconstruct` command in `WoWRollback.Cli`
- **Status**: Ready to generate valid placement CSVs for injection

### Next Steps
1.  **Regenerate Source ADTs**: Re-run the entire `wdl-to-adt` and `adt-merge` pipeline with the current fixed codebase.
2.  **Re-run Repair**: Execute `development-repair` on the clean source data.
3.  **Verify**: Check `chunk_dump` output for non-zero `MTEX` and `MODF` chunks.
