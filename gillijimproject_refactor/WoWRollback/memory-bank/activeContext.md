# Active Context - WoWRollback.RollbackTool Development

## Current Focus (2025-10-21)
**Building WDT Rollback Tool with Terrain Hole Management**

Successfully implemented and **TESTED** core rollback functionality that modifies Alpha 0.5.3 WDT files! Proven working on both Azeroth and Kalimdor. Now building terrain hole management and overlay generation to complete the tool.

## What We Just Accomplished (2025-10-21)

### ✅ PROVEN: Core Rollback Works on Alpha 0.5.3!

**Successful Tests:**
```
Kalimdor 0.5.3 (951 ADT tiles)
  - Total Placements: 126,297
  - Kept (UID ≤ 78,000): 635
  - Buried (UID > 78,000): 125,662
  - Status: SUCCESS!

Azeroth 0.5.3
  - Multiple successful rollbacks
  - MD5 checksum generation confirmed
  - Status: SUCCESS!
```

**What Works:**
1. ✅ Load Alpha WDT files
2. ✅ Parse all ADT tiles (MHDR offsets from WDT MAIN chunk)
3. ✅ Extract MDDF/MODF chunks via AdtAlpha
4. ✅ Modify placement Z coordinates (bury at -5000.0)
5. ✅ Write modified data back to wdtBytes array
6. ✅ Output modified WDT file
7. ✅ Generate MD5 checksum for minimap compatibility

### ✅ Technical Breakthroughs

**AdtAlpha Integration**
- AdtAlpha already parses MDDF and MODF chunks!
- Added `GetMddf()` and `GetModf()` accessors
- Added `GetMddfDataOffset()` and `GetModfDataOffset()` to locate chunks in file
- Stored `_adtFileOffset` in constructor for offset calculations

**Placement Format**
```
MDDF (M2 models) - 36 bytes per entry:
  +0x00: nameId (4 bytes)
  +0x04: uniqueId (4 bytes) ← FILTER BY THIS
  +0x08: position X (4 bytes)
  +0x0C: position Z (4 bytes) ← MODIFY THIS TO BURY
  +0x10: position Y (4 bytes)
  
MODF (WMO buildings) - 64 bytes per entry:
  +0x00: nameId (4 bytes)
  +0x04: uniqueId (4 bytes) ← FILTER BY THIS
  +0x08: position X (4 bytes)
  +0x0C: position Z (4 bytes) ← MODIFY THIS TO BURY
  +0x10: position Y (4 bytes)
  ... (rest of entry)
```

**File Structure**
```
WDT File (Alpha 0.5.3):
  MVER chunk
  MPHD chunk (flags)
  MAIN chunk (64x64 grid, offsets to ADT data)
  ... ADT data embedded inline ...
    ADT #0 @ offset XXXX
      MHDR (offsets to MDDF/MODF/etc)
      MDDF chunk
      MODF chunk
      MCNK chunks (256 per ADT)
    ADT #1 @ offset YYYY
    ...
```

## Architecture Decision: New Project Structure

**Problem**: WoWDataPlot was being built as a hybrid analysis+modification+visualization tool, which violates separation of concerns.

**Solution**: Split into three focused tools:

```
AlphaWDTAnalysisTool/     (EXISTS - Analysis Phase)
  └─> Scans WDT/ADTs
  └─> Outputs CSVs with UniqueID data
  └─> Already has complete infrastructure!

WoWRollback.RollbackTool/  (NEW - Modification Phase)
  └─> Reads analysis CSVs
  └─> Modifies WDT files in-place
  └─> Manages terrain holes and shadows
  └─> Generates MD5 checksums

WoWDataPlot/               (REFOCUS - Visualization Phase)
  └─> Reads CSVs from analysis
  └─> Pre-generates overlay images
  └─> Lightweight HTML viewer
  └─> No modification, pure viz
```

**Current Implementation Status:**
- ✅ Rollback code working in `WoWDataPlot/Program.cs` (temporary location)
- ⏳ Need to extract to new `WoWRollback.RollbackTool` project
- ⏳ Need to add MCNK terrain hole management
- ⏳ Need to add MCSH shadow disabling
- ⏳ Need to add overlay generation

## Next Steps (For Fresh Session)

### Phase 1: Create WoWRollback.RollbackTool Project
1. Create new CLI project under `WoWRollback/`
2. Move rollback code from `WoWDataPlot/Program.cs`
3. Reference `gillijimproject-csharp` library
4. Command structure: `analyze`, `generate-overlays`, `rollback`

### Phase 2: MCNK Terrain Hole Management
1. Parse MCNK chunks from each ADT (already in AdtAlpha)
2. For each buried placement, calculate which MCNK(s) it overlaps
   - Each ADT = 533.33 yards square
   - Each MCNK = 33.33 yards square (16x16 grid)
   - Formula: `mcnkX = floor((x - tileX*533.33) / 33.33)`
3. Clear `Holes` field (offset 0x40 in MCNK header)
4. Write modified MCNK headers back to file

### Phase 3: MCSH Shadow Disabling (Optional Feature)
1. Find all MCSH chunks in ADT
2. Option `--disable-shadows` zeros out MCSH chunk data
3. Write modified chunks back to file

### Phase 4: Overlay Generation
1. Pre-generate PNG overlays for each UniqueID threshold
2. Color code: green=kept, red=buried
3. Output naming: `overlays/{mapname}_uid_0-5000.png`
4. Generate `overlay-index.json` manifest

### Phase 5: Lightweight Viewer
1. Pure HTML+JS slider UI
2. Loads pre-generated overlays based on slider position
3. Displays placement stats
4. Generates rollback command for copying

## Git Status
- **Branch**: `wrb-poc5`
- **Last Commit**: `58d0aae` - "WoWDataPlot - now with Rollback support (Tested on 0.5.3, it works!)"
- **Test Data**: `test_data/0.5.3/tree/World/Maps/`
- **Output**: `WoWRollback/rollback_*` directories

## Files Modified This Session (2025-10-21)
- `WoWDataPlot/Program.cs` - Added complete rollback command
- `AdtAlpha.cs` - Added `GetMddf()`, `GetModf()`, `Get*DataOffset()` methods
- `AdtAlpha.cs` - Added `_adtFileOffset` field to track position in file

## What Works Now
✅ Load Alpha 0.5.3 WDT files  
✅ Parse all embedded ADT tiles via offsets  
✅ Extract MDDF/MODF placements  
✅ Modify Z coordinates to bury objects  
✅ Write modified WDT back to disk  
✅ Generate MD5 checksums  
✅ TESTED on Kalimdor (951 tiles, 126K placements!)  

## What's Next
⏳ Create WoWRollback.RollbackTool project  
⏳ Implement MCNK hole flag clearing  
⏳ Implement MCSH shadow disabling  
⏳ Generate overlay images  
⏳ Build lightweight viewer
