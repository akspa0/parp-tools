# CSV Cleanup - Reduce File Spam

**Date**: 2025-01-08 17:31  
**Status**: Implemented

---

## Problem

Pipeline was spamming disk with per-tile CSV files:
- `asset_fixups_0_0.csv`, `asset_fixups_0_1.csv`, ... (hundreds of files)
- Each tile got its own fixup log
- Excessive file count, hard to navigate

---

## Solution

### ✅ Implemented: Consolidate & Delete
**File**: `ConvertPipelineMT.cs` lines 194-213

**What it does**:
1. Merge all per-tile `asset_fixups_*.csv` into single `asset_fixups.csv`
2. **Delete individual tile files** after merge
3. Single CSV includes `tile_x` and `tile_y` columns to track which tile had fixups

**Output format**:
```csv
type,original,resolved,method,map,tile_x,tile_y
m2,world/azeroth/elwynn/passivedoodads/bush/elwynnbush01.mdx,world/azeroth/elwynn/passivedoodads/bush/elwynnbush01.m2,fuzzy,Azeroth,32,48
wmo,world/azeroth/buildings/stormwind_cathedral.wmo,world/azeroth/buildings/stormwind_cathedral.wmo,exact,Azeroth,32,48
```

**Benefits**:
- ✅ Single file per map instead of hundreds
- ✅ Still tracks which tile had fixups (tile_x, tile_y columns)
- ✅ Easy to grep/analyze
- ✅ Reduces disk clutter

---

## What We're Keeping (Don't Touch!)

### AreaTable Crosswalks ✅
**Reason**: "That whole system is a bear to get working right, and I don't want to break it for a 5th time"

**Files preserved**:
- `Area_patch_crosswalk_*.csv`
- `Area_crosswalk_v*.csv`
- All crosswalk-related CSVs

**Why**: Complex mapping system, working correctly, not worth the risk

---

## Other CSV Consolidation Opportunities

### Current State
These still generate multiple files:

#### 1. UniqueID Analysis
**Current**:
- `{map}_uniqueID_by_tile.csv` - One row per unique ID per tile
- `{map}_tile_layers.csv` - One row per tile-layer with range/count
- `{map}_uniqueID_analysis.csv` - Legacy format

**Recommendation**: Keep as-is for now
- Different purposes (per-ID vs per-layer vs legacy)
- JSON already consolidates (`{map}_uniqueid_layers.json`)
- Not excessive file count

#### 2. Terrain/Shadow CSVs
**Current**:
- `{map}_terrain.csv` - One per map
- `{map}_shadow.csv` - One per map

**Status**: ✅ Already consolidated (one per map)

---

## Future Consolidation Ideas

### Convert to JSON (Future Enhancement)
Instead of multiple CSVs, single JSON per map:

```json
{
  "map": "Azeroth",
  "version": "0.5.3",
  "asset_fixups": [
    {
      "tile": {"x": 32, "y": 48},
      "type": "m2",
      "original": "world/.../bush01.mdx",
      "resolved": "world/.../bush01.m2",
      "method": "fuzzy"
    }
  ],
  "uniqueids": {
    "layers": [...],
    "by_tile": {...}
  },
  "terrain": {...},
  "shadow": {...}
}
```

**Benefits**:
- Single file per map
- Easier to parse programmatically
- Better for web viewer integration
- Hierarchical structure

**Cons**:
- Harder to grep/analyze with command-line tools
- Requires JSON parser for manual inspection
- Breaking change for existing tools

**Recommendation**: Keep CSVs for now, consider JSON in v2.0

---

## File Count Reduction

### Before
```
03_adts/0.5.3/logs/Azeroth/
├── asset_fixups_0_0.csv
├── asset_fixups_0_1.csv
├── asset_fixups_0_2.csv
├── ... (685 files for Azeroth!)
└── asset_fixups.csv (merged, but originals kept)
```

### After
```
03_adts/0.5.3/logs/Azeroth/
└── asset_fixups.csv (single file, tile files deleted)
```

**Reduction**: 685 files → 1 file per map ✅

---

## Testing

### Verify Consolidation Works
```powershell
# Run pipeline
dotnet run --project WoWRollback.Orchestrator -- \
  --maps Azeroth \
  --versions 0.5.3 \
  --alpha-root ..\test_data \
  --lk-dbc-dir ..\test_data\3.3.5\tree\DBFilesClient

# Check only merged file exists
ls parp_out\session_*\03_adts\0.5.3\logs\Azeroth\asset_fixups*.csv
# Should show: asset_fixups.csv (only 1 file)

# Verify tile info preserved
cat parp_out\session_*\03_adts\0.5.3\logs\Azeroth\asset_fixups.csv | Select-String "tile_x,tile_y"
# Should show header with tile columns
```

### Verify Data Integrity
```powershell
# Check merged file has data from all tiles
$csv = Import-Csv "parp_out\session_*\03_adts\0.5.3\logs\Azeroth\asset_fixups.csv"
$csv | Group-Object tile_x, tile_y | Measure-Object
# Should show multiple tile combinations
```

---

## Summary

### ✅ Completed
- Consolidated per-tile asset_fixups CSVs into single file
- Delete individual tile files after merge
- Preserve tile location in merged CSV (tile_x, tile_y columns)

### ✅ Preserved
- AreaTable crosswalk CSVs (don't touch!)
- UniqueID analysis CSVs (reasonable count)
- Terrain/Shadow CSVs (already consolidated)

### 🔮 Future
- Consider JSON format for all analysis data
- Single consolidated file per map
- Better web viewer integration

**Status**: CSV spam reduced, disk clutter minimized! 🎯
