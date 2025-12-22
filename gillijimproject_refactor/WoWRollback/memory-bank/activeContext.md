# WoWRollback Active Context

## Current Focus: PM4 → ADT Pipeline (Dec 2025)

### Session Summary (2025-12-21)

**Goal**: Match PM4 pathfinding geometry to WMOs and reconstruct ADT placements.

**Fixes Applied**:
1. ✅ CK24 Lookup CSV parsing - strip quotes from values
2. ✅ WMO Full Mesh mode - enabled by default
3. ✅ PM4 MSVT+MSCN combined geometry for matching
4. ✅ Coordinate swap reverted in `Pm4Decoder` (X,Y,Z direct)
5. ✅ **Skip CK24=0x000000** - nav mesh excluded from WMO matching

**Root Cause of Noggit Crash**: CK24=0x000000 (nav mesh terrain) was matched to random WMOs and placed, creating garbage that crashed Noggit.

---

## Critical Insight: PM4 Scene Graph Architecture

The PM4 format is a **hierarchical scene graph**:

```
PM4 Map Object (Global)
├── Global Pools (MSVT, MSVI, MSCN, MPRL, MPRR)
├── CK24 Object Groups (can span multiple tiles)
└── Tile Manifests (which CK24s belong to which tile)
```

### Current Problem
Reading tiles independently is **wrong**. Objects span tiles; vertex data is shared globally.

### Next Step: Implement Global PM4 Reader
1. Load ALL tiles into one unified `Pm4MapObject` 
2. Build global vertex/index/MSCN pools with tile provenance
3. Extract per-tile CK24 objects by referencing global pools
4. Use MSCN as verification layer (compare matched WMO point cloud)

**Implementation Plan**: See `implementation_plan.md` in artifacts.

---

## Key Files

| File | Purpose |
|------|---------|
| `Pm4Decoder.cs` | Decodes single PM4 tile chunks |
| `Pm4ObjectBuilder.cs` | Groups surfaces by CK24, splits by MSVI gaps |
| `Pm4ModfReconstructor.cs` | Matches PM4 objects to WMOs |
| `MuseumAdtPatcher.cs` | Injects MODF/MWMO into ADTs |
| `Pm4Reader/Program.cs` | Original standalone reader (reference) |

## CK24 Structure

```
CK24 = [Type:8bit][ObjectID:16bit]
- 0x00XXXX = Nav mesh (SKIP)
- 0x40XXXX = Has pathfinding data
- 0x42XXXX / 0x43XXXX = WMO-type objects
- 0xC0XXXX+ = Various object types
```

## Do NOT
- Match CK24=0x000000 to WMOs (it's nav mesh)
- Read PM4 tiles independently for cross-tile objects
- Use coordinate swaps in `Pm4Decoder` (original format is X,Y,Z)
