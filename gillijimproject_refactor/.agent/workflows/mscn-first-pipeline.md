---
description: PM4 to ADT pipeline using MSCN collision data as foundation
---

# MSCN-First PM4 to ADT Pipeline

## Key Insight
PM4 files contain **MSCN collision geometry** that precisely defines object boundaries. This is the foundation for:
- Object matching by shape
- Coordinate verification
- Missing object discovery

## Current State (December 22, 2024)
- ✅ MSCN discovery groups by CK24 via MdosIndex
- ✅ Shape-based WMO/M2 library matching using PrincipalExtents
- ✅ Old CK24-based matching disabled
- ✅ Added `MscnToAdtPosition` transform (swaps X↔Y)
- ❌ **COORDINATE OUTPUT STILL WRONG** - see findings below

## Session Findings (Dec 22, 2024)

### Raw MSCN Data Discovery
For tile 22_18, raw MSCN coordinates are:
- **X range: 10006 - 10084** (NOT tile-local 0-533!)
- **Y range: 12040 - 12115** 
- **Z range: 184 - 189** (height, looks correct for floor collision)

This means **MSCN is ALREADY in world coordinates**, not tile-local.

### Axis Swap Finding
MSCN has X and Y swapped vs WoW client convention:
- MSCN (10033, 12054) after swap → (12054, 10033)
- Expected client coords: (12021, 9922)
- **Close match!** (within 30-110 units = object bounding box variance)

### Still Broken
Current output in modf_entries.csv shows values like:
- `pos_x=7833.80, pos_y=19574.03`
- Expected: ~12000, ~10000 range

**The MscnToAdtPosition transform is correct, but something else in the pipeline is still applying the wrong conversion or inverting** 

### PM4 Format Insight
PM4 is like a **Blender project file**:
- Gaps ARE data (define container boundaries)
- Everything interlinks (MSLK→MSPI→MSPV, MSUR→MSVI→MSVT, MdosIndex→MSCN)
- Implicit hierarchies (no explicit markers)
- MSVI gap detection partially works but over-subdivides

## Key Files
- `WoWRollback.PM4Module/PipelineCoordinateService.cs` - Coordinate transforms (MscnToAdtPosition added)
- `WoWRollback.PM4Module/Decoding/MscnObjectDiscovery.cs` - MSCN grouping and matching (uses MscnToAdtPosition)
- `WoWRollback.PM4Module/PipelineService.cs` - Main pipeline orchestration

## Next Steps
1. **RUN PIPELINE** with debug logging to see actual coordinate values
2. Check `[DEBUG COORD]` output to see if `MprlPosition` vs `Stats.Centroid` differ
3. If `MprlPosition` is correct but CSV is wrong → issue in export
4. If both are wrong → issue upstream in MSCN extraction
5. Apply fix based on findings

### Debug Output Added
- `[DEBUG COORD]` lines show `MprlPosition` (transformed) vs `Stats.Centroid` (raw)
- Look for: `MprlPosition ~(12054, 10033)` vs `Stats.Centroid ~(10033, 12054)`


## Expected Values for Tile 22_18
- Raw MSCN: (10033, 12054, 185) 
- After axis swap: (12054, 10033, 185)
- Client coords target: (12021.43, 9922.55, 304.32)
- Server coords: (7144.12, 5045.24, 304.32)

## Pipeline Flow
```
PM4 → MSCN Extraction → CK24 Grouping → MscnToAdtPosition → 
Filter Existing ADT → Shape Match vs WMO/M2 → MODF/MDDF → Patch ADTs
```