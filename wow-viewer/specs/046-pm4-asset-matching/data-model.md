# Data Model: PM4 Asset Matching

## Pm4ObjectSegment

- **Purpose**: One exported PM4 object candidate used as the unit of matching and placement synthesis.
- **Fields**:
  - `SegmentId` - stable identifier for the exported PM4 segment
  - `TileCoordinates` - every tile touched by the segment
  - `Field04Values` - `MSHD.Field04` buckets touched by the segment
  - `Ck24` - full PM4 object key
  - `Ck24Type` - high-byte family classification
  - `Ck24ObjectId` - low-16 object identity slice
  - `SurfaceCount` - number of contributing PM4 surfaces
  - `LinkGroupIds` - observed `MSLK.GroupObjectId` hints
  - `ConfidenceFlags` - segmentation ambiguity and export warnings
- **Relationships**:
  - owns one `Pm4SegmentSignalRecord`
  - can have zero or more `CandidateMatch` rows

## Pm4SegmentSignalRecord

- **Purpose**: Comparable signal payload derived from one `Pm4ObjectSegment`.
- **Fields**:
  - `SegmentId`
  - `Bounds`
  - `FootprintHull`
  - `HeightStats`
  - `SurfaceFamilyHistogram`
  - `TopologyStats`
  - `AnchorSignals`
  - `SignalVersion`
  - `SignalStoreRow`
- **Rules**:
  - must be reproducible from the same PM4 inputs
  - must be comparable to an asset reference signal record

## AssetReferenceSignalRecord

- **Purpose**: One staged WMO/M2 asset reference entry represented in the same scoring space as PM4 segments.
- **Fields**:
  - `AssetId`
  - `AssetPath`
  - `AssetKind` - WMO or M2
  - `ClientBuild`
  - `Bounds`
  - `FootprintHull`
  - `SurfaceFamilyHistogram`
  - `RenderOrCollisionSignals`
  - `SignalVersion`
  - `SignalStoreRow`
- **Relationships**:
  - can appear in zero or more `CandidateMatch` rows

## CandidateMatch

- **Purpose**: Ranked match result between one PM4 segment and one asset reference entry.
- **Fields**:
  - `SegmentId`
  - `AssetId`
  - `Rank`
  - `OverallScore`
  - `ScoreBreakdown`
  - `Status` - matched, ambiguous, unresolved, ineligible
  - `Rationale`
  - `ValidationTags`
- **Rules**:
  - ranks are unique within one segment’s candidate list
  - unresolved or ineligible states still preserve rationale

## ReplacementPlacementProposal

- **Purpose**: Machine-readable proposal for adding one asset placement derived from PM4 evidence.
- **Fields**:
  - `ProposalId`
  - `SegmentId`
  - `AssetId`
  - `TargetTileCoordinates`
  - `WorldPosition`
  - `WorldRotation`
  - `WorldScale`
  - `Confidence`
  - `ReviewRequired`
  - `Provenance`
- **Relationships**:
  - references one source `Pm4ObjectSegment`
  - references one selected `CandidateMatch`

## MatchRunManifest

- **Purpose**: Top-level record for one export/match/synthesis run.
- **Fields**:
  - `RunId`
  - `InputPm4Root`
  - `AssetReferenceCorpus`
  - `SegmentSignalCorpus`
  - `SignalVersion`
  - `StartedAtUtc`
  - `CompletedAtUtc`
  - `SegmentCount`
  - `MatchedCount`
  - `AmbiguousCount`
  - `UnresolvedCount`
  - `PlacementProposalCount`
  - `Warnings`
- **Relationships**:
  - owns or references all emitted `Pm4ObjectSegment`, `CandidateMatch`, and `ReplacementPlacementProposal` records for the run

## SignalCorpusIndexRow

- **Purpose**: Row-level locator tying one PM4 segment or asset reference record to its Zarr-backed signal payload.
- **Fields**:
  - `CorpusKind`
  - `LogicalId`
  - `RowIndex`
  - `TileX`
  - `TileY`
  - `Build`
  - `Kind`
  - `SignalVersion`
  - `EligibilityFlags`

## Relationships Summary

- `Pm4ObjectSegment` owns one `Pm4SegmentSignalRecord`.
- `Pm4ObjectSegment` can have many `CandidateMatch` rows.
- `AssetReferenceSignalRecord` can participate in many `CandidateMatch` rows.
- `ReplacementPlacementProposal` references one `Pm4ObjectSegment` and one selected `CandidateMatch`.
- `MatchRunManifest` describes one bounded automation run and its outputs.
