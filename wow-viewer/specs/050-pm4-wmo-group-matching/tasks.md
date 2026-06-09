# Tasks: 050 PM4 WMO Group Matching

## Phase 1: ADT Placement Lookup

- [ ] T001 Add `GetPm4ObjectIdBounds(uint regionId, ushort objectId)` to WorldScene — returns aggregate bounds of all surfaces with that ObjectId in that region, grouped by GroupKey.
- [ ] T002 Add `TryResolveWmoFromPlacement(int tileX, int tileY, uint ck24)` — reads `_obj0.adt`, finds WMO placements overlapping the PM4 object bounds, returns WMO path.
- [ ] T003 Add `TryResolveWmoFromArchive(uint ck24)` — searches staged client for WMO files whose CK24-like identifiers or bounds match.

## Phase 2: WMO Group Display

- [ ] T004 Add `DrawPm4WmoGroupMatch()` to ViewerApp_Pm4Utilities.cs — reads WMO via `WmoGroupInfoSummaryReader`, displays per-group bounds and flags table alongside PM4 ObjectId GroupKey clusters.
- [ ] T005 Display Jaccard overlap between each PM4 GroupKey cluster and each WMO group.

## Phase 3: Fallback Shape Search

- [ ] T006 Add `SearchWmoByShape(Pm4Bounds3 bounds)` — enumerates WMO files from staged client, compares bounds, returns top 5 candidates.
- [ ] T007 Wire fallback search into the UI when ADT placement returns no results.

## Phase 4: Match Persistence

- [ ] T008 Add "Confirm Match" context menu to matched pairs in the UI. Saves to `wow-viewer/output/pm4_wmo_matches.json`.
- [ ] T009 Load saved matches on startup, display confirmed status in match panel.
- [ ] T010 Add "Delete Match" and "Override Match" support.
- [ ] T011 Add "Match to WMO" button in the scene graph tree for each ObjectId.
- [ ] T012 Update memory bank and commit.
