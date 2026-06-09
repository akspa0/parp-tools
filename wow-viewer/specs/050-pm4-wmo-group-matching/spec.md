# Spec 050: PM4 WMO Group Matching

**Status**: Draft | **Priority**: P1 | **Owner**: WoWViewer

## Problem

PM4 data contains collision surfaces classified by GroupKey (0x12=exterior, 0x10=floor, 0x13=portal). These surfaces cluster under ObjectId values that correspond to WMO or M2 game objects. There is no automated way to determine WHICH WMO file a given PM4 ObjectId cluster matches — users must manually compare against ADT placement data or guess.

## User Stories

### US1: ADT Placement Lookup (P1)
As a user, when I select an ObjectId in the scene graph, if the tile's `_obj0.adt` contains placement records for that CK24, I want to see the WMO path, group count, and per-group bounds matched against the PM4 surface clusters.

### US2: Shape-Based Match Fallback (P2)
As a user, when no ADT placement exists for the selected ObjectId, I want the viewer to search all WMO files in the staged client for the closest shape match by comparing overall bounds, surface area, and volume.

### US3: Match Result Display (P1)
As a user, I want match results shown as a table: WMO group index, WMO group flags, WMO group bounds, matching PM4 surface count, and overlap ratio.

## Functional Requirements

### FR-001: ADT Detection
- When an ObjectId is selected, check if `{mapDir}/{tileX}_{tileY}_obj0.adt` exists in the data source
- If yes, read placement records via `AdtPlacementReader`
- Filter placements where `modelPath` ends with `.wmo` and the placement bounds overlap the PM4 ObjectId cluster bounds
- Display matching placements with WMO path

### FR-002: WMO Group Reading
- For each matching WMO path, read the WMO root file via `WmoGroupInfoSummaryReader`
- Extract per-group bounds and flags from `WmoGroupInfoSummary.Entries`
- Display group count, individual group bounds, and group type flags

### FR-003: Surface-to-Group Comparison
- Cluster all PM4 surfaces with the selected ObjectId by their GroupKey
- For each cluster, compute aggregate bounds
- Compare each cluster bounds against each WMO group bounds via Jaccard overlap
- Display overlap scores per (PM4 GroupKey, WMO Group) pair

### FR-004: Fallback Shape Matching
- If no ADT placement exists, load all WMO root files from the staged client archive
- For each WMO, compare its overall bounds against the PM4 ObjectId aggregate bounds
- Score by volume ratio, footprint ratio, span ratios (sorted)
- Return top 5 matches with scores

## Data Sources

- ADT placement data: `I:/parp/parp-tools/output/tmp/wowarchive-clients/<build>/World of Warcraft/world/maps/<map>/<tile>_obj0.adt`
- WMO files: same client root, path from ADT placement records
- PM4 data: `_pm4ObjectLookup` and `_pm4ObjectGroupBounds` from WorldScene

## Success Criteria

1. Selecting a WMO-type ObjectId shows ADT placement matches within 1 second
2. WMO group bounds display correctly alongside PM4 surface clusters
3. Overlap scores are meaningful (matched groups >90%, mismatched <30%)
4. Fallback shape search completes within 5 seconds for a limited WMO corpus
5. No frame rate drop — all computation is on-demand, not per-frame

### FR-005: Manual Match Curation
- User can right-click a matched pair (PM4 ObjectId × WMO path) and select "Confirm Match"
- Confirmed matches are saved to `wow-viewer/output/pm4_wmo_matches.json` as a JSON dictionary
- Format: `{ "mapName": { "objectId": "wmoPath" } }` — keyed by map name and ObjectId
- On startup, the viewer loads saved matches and displays them in the match panel
- User can delete/override saved matches
- This builds a ground-truth dataset for training the automatic matcher

## Out of Scope

- M2 model matching (only WMO for now)
- Writing modified ADTs or PM4 files
- Automatic WMO path inference without ADT data
