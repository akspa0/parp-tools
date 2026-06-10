# Tasks: 050 PM4 WMO Group Matching

## Phase 1: ADT Placement Lookup

- [x] T001 Add `GetPm4SurfaceGroupClusters(tileX, tileY, ck24)` to WorldScene — returns aggregate bounds of all surfaces with that CK24 in that tile, grouped by GroupKey.
- [x] T002 Add `MatchFromPlacement()` in Pm4WmoGroupMatchService — reads `_obj0.adt`, finds WMO placements overlapping the PM4 object bounds, reads WMO group info, computes Jaccard overlap.
- [ ] T003 Add `TryResolveWmoFromArchive(uint ck24)` — searches staged client for WMO files whose CK24-like identifiers or bounds match. (Deferred: shape fallback handles this)

## Phase 2: WMO Group Display

- [x] T004 Implement `DrawPm4WmoGroupMatch()` in ViewerApp_Pm4Utilities.cs — reads WMO via `WmoGroupInfoSummaryReader`, displays per-group bounds and flags table alongside PM4 ObjectId GroupKey clusters.
- [x] T005 Display Jaccard overlap between each PM4 GroupKey cluster and each WMO group (with world-space transform from placement data).

## Phase 3: Fallback Shape Search

- [x] T006 Add `SearchWmoByShape()` in Pm4WmoGroupMatchService — enumerates WMO files from staged client, compares bounds/volume/footprint, returns top 5 candidates.
- [x] T007 Wire fallback search into the UI — "Shape Fallback" button activates after initial match.

## Phase 4: Match Persistence

- [x] T008 Add "Confirm Match" button to matched placements in the UI. Saves to `wow-viewer/output/pm4_wmo_matches.json`.
- [x] T009 Load saved matches on startup (in ViewerApp settings load), display confirmed status in match panel.
- [x] T010 Add "Clear Saved" button support for saved matches.
- [ ] T011 Add "Match to WMO" button in the scene graph tree for each ObjectId. (Deferred: available via PM4 workbench Selection tab)
- [x] T012 Update memory bank and commit.

## Summary

**Implemented: 9/11 tasks** (T003, T011 deferred as lower priority).

### Files Created
- `src/viewer/WoWViewer/Pm4WmoGroupMatchService.cs` — matching service with ADT placement lookup, WMO group reading, Jaccard overlap, fallback shape search, and match persistence

### Files Modified
- `src/viewer/WoWViewer/Terrain/WorldScene.cs` — added `TryGetPm4ObjectGroupBounds()`, `GetPm4SurfaceGroupClusters()`, `Pm4SurfaceGroupCluster` record
- `src/viewer/WoWViewer/ViewerApp.cs` — added match state fields, store initialization on settings load
- `src/viewer/WoWViewer/ViewerApp_Pm4Utilities.cs` — implemented `DrawPm4WmoGroupMatch()` replacing placeholder with full UI
