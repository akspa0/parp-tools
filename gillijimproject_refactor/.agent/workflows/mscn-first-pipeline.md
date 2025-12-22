---
description: PM4 to ADT pipeline using MSCN collision data as foundation
---

# MSCN-First PM4 to ADT Pipeline

## Key Insight
PM4 files contain **MSCN collision geometry** that precisely defines object boundaries. This is the foundation for:
- Object matching by shape
- Coordinate verification
- Missing object discovery

## Current State (December 2024)
- ✅ MSCN discovery groups by CK24 via MdosIndex
- ✅ Shape-based WMO/M2 library matching using PrincipalExtents
- ✅ Old CK24-based matching disabled
- ❌ **COORDINATE TRANSFORM BROKEN** - produces invalid positions (Z=20672)

## Critical Fix Needed: Phase 1
The coordinate transform in `PipelineCoordinateService.Pm4ToAdtPosition` is wrong.

### Debug Steps
// turbo
1. Add debug output showing raw PM4 MSCN bounds per tile
2. Compare to expected world coords from tile index
3. Determine if PM4 is tile-local (0-533) or already world coords

### Expected Values for Tile 22_18
- Client coords: (12021.43, 9922.55, 304.32)
- Server coords: (7144.12, 5045.24, 304.32)
- MODF should be near these values, NOT (11253, 47, 20672)

## Key Files
- `WoWRollback.PM4Module/PipelineCoordinateService.cs` - Coordinate transforms
- `WoWRollback.PM4Module/Decoding/MscnObjectDiscovery.cs` - MSCN grouping and matching
- `WoWRollback.PM4Module/PipelineService.cs` - Main pipeline orchestration

## Pipeline Flow
```
PM4 → MSCN Extraction → CK24 Grouping → Coordinate Transform → 
Filter Existing ADT → Shape Match vs WMO/M2 → MODF/MDDF → Patch ADTs
```

## What Was NOT Wasted
- CK24 structure → enables per-object MSCN grouping
- MPRL decoding → provides rotation/position hints  
- WMO/M2 library building → reusable for matching
- ADT patching code → works once positions are correct

## Phase 2-4 (After Coordinate Fix)
2. MSCN pipeline verification
3. Visualization tools (MSCN terrain overlay, debug OBJ)
4. Cleanup dead CK24 code
