# Active Context - WoWRollback.RollbackTool Development

## Current Focus (2025-10-22)
**Unify Alpha→LK pipeline (rollback + area map + export) and add LK patcher**

Successfully implemented and **TESTED** core rollback functionality that modifies Alpha 0.5.3 WDT files! Proven working on both Azeroth and Kalimdor. Now building terrain hole management and overlay generation to complete the tool.

## What We Just Accomplished (2025-10-22)

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
3. ✅ Extract MDDF/MODF chunks via `AdtAlpha`
4. ✅ Modify placement Z coordinates (bury at -5000.0)
5. ✅ Write modified data back to `wdtBytes`
6. ✅ Output modified WDT file + MD5 checksum
7. ✅ Selective hole clearing per MCNK using `MCRF` references (only if all referenced placements are buried)
8. ✅ Optional MCSH zeroing
9. ✅ LK export path via `--export-lk-adts` with `AdtAlpha.ToAdtLk(..., areaRemap)` and `AdtLk.ToFile(<dir>)`
10. ✅ Area mapping hook + `--area-remap-json` loader

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

## Current Implementation Status:**
- ✅ Rollback code working in `WoWDataPlot/Program.cs` (temporary location)
- ⏳ Need to extract to new `WoWRollback.RollbackTool` project
- ⏳ Need to add MCNK terrain hole management
- ⏳ Need to add MCSH shadow disabling
- ⏳ Need to add overlay generation

## Next Steps (For Fresh Session)

### Phase 1: Unified Pipeline Command
1. `alpha-to-lk` implemented (wrapper over rollback with `--export-lk-adts`):
   - Rollback: bury + MCRF-gated hole clear + optional MCSH
   - Area mapping: `--area-remap-json` or auto-fill via `--lk-client-path` (LK `AreaTable.dbc` IDs passthrough; unmapped→`--default-unmapped`)
   - LK export: `AdtAlpha.ToAdtLk(..., areaRemap)` → `AdtLk.ToFile(lkOutDir)`
2. CLI help updated. Added preferred crosswalk flags `--crosswalk-dir`/`--crosswalk-file` (kept legacy `dbctool-*` aliases).

### Phase 2: AreaTable Auto-Mapper
1. Implement minimal `AreaTableDbcReader` (IDs only) via `PrioritizedArchiveSource`/`MpqArchiveSource`.
2. Prefill AlphaAreaId→LKAreaId when IDs exist in LK; else map to `--default-unmapped`.

### Phase 3: LK Patcher Command
1. `lk-to-alpha` (v1) implemented: patches LK ADTs (bury/holes/mcsh) and writes to `--out`.
2. Next: validate counts/logs on Kalimdor and Azeroth directories.

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
