# Plan: 050 PM4 WMO Group Matching

## Phase 1: ADT Check + WMO Path Resolution
- Add method to check if `_obj0.adt` exists for a given tile
- Read placement catalog via `AdtPlacementReader`
- Filter WMO placements by CK24/position overlap
- Display WMO paths in the UI

## Phase 2: WMO Group Reading + Display
- Read WMO root files via `WmoGroupInfoSummaryReader`
- Extract group bounds from `Entries`
- Display per-group table (index, flags, bounds)

## Phase 3: Overlap Comparison
- Aggregate PM4 surfaces by ObjectId and GroupKey
- Compute per-cluster bounds
- Jaccard overlap against each WMO group bounds
- Display match table sorted by overlap

## Phase 4: Fallback Shape Search
- Enumerate WMO files from staged client
- Compare overall bounds against PM4 ObjectId cluster
- Rank by shape similarity (volume, footprint, spans)
- Return top 5
