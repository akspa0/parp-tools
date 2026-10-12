# Data Model: PM4-Guided Object Transfer and Museum Placement Repair

## Design rules

- All coordinates in reconciliation records are world-space ADT placement coordinates unless the field is
  explicitly named `Pm4Space` or `AssetLocal`.
- PM4 identities are immutable guide identities. ADT placement identities are immutable source-row
  identities for the lifetime of a preview; a new clone receives a separate identity.
- Preview records are immutable. A decision references a preview and records the user's disposition rather
  than changing the proposal in place.
- Client asset bytes are never stored in these records. Paths, IDs, build fingerprints, hashes, and numeric
  signals are allowed provenance.

## Entities

### `Pm4GuideIdentity`

The stable identity of one PM4 object used as a guide.

| Field | Type | Rules |
|---|---|---|
| `GuideId` | string | Stable hash/qualified identity; required and unique within a guide corpus |
| `SourcePath` | string | Configured PM4 path; must exist for a live preview |
| `BuildFingerprint` | string | Required for cross-build use; compared before applying |
| `MapName` | string | Required |
| `TileX`, `TileY` | int | Filename-derived tile coordinates |
| `Ck24` | uint | Raw PM4 object key; zero is a valid remainder bucket, not an object-kind claim |
| `ObjectPart` | int | Non-negative part identity |
| `GeometryFingerprint` | string? | Existing fingerprint version/key when available |

### `Pm4GuideObservation`

The measured geometry and placement-space constraints for a guide object.

| Field | Type | Rules |
|---|---|---|
| `Guide` | `Pm4GuideIdentity` | Required |
| `Position` | Vector3 | Converted through the canonical coordinate service |
| `BoundsMin`, `BoundsMax` | Vector3 | Must be finite; min <= max per axis |
| `Footprint` | point list | Optional only when geometry is absent; finite points |
| `HeightSignal` | float? | Placement-height evidence, marked as measured or unavailable |
| `ExpectedAssetKind` | enum | `Model`, `WorldModel`, `Unknown` |
| `SignalVersion` | string | Identifies the PM4 signal schema |
| `Evidence` | list | Named signal, value, source, and confidence; never an opaque score only |

### `PlacementIdentity`

The source placement row in a loaded Museum ADT/WDT.

| Field | Type | Rules |
|---|---|---|
| `SourcePath` | string | Configured source path; never an output path |
| `MapName` | string | Required |
| `TileX`, `TileY` | int | Target tile coordinates |
| `Kind` | enum | `Model` for MDDF or `WorldModel` for MODF |
| `EntryIndex` | int | Non-negative row index in its placement chunk |
| `UniqueId` | int | Must still match the source row when applied |
| `AssetPath` | string | Resolved path from the ADT name table |
| `BuildFingerprint` | string | Source build identity |

The pair `(SourcePath, Kind, EntryIndex, UniqueId)` is the optimistic-concurrency key. An apply must
refuse a preview when the row no longer has the same unique ID or source fingerprint.

### `PlacementSnapshot`

The source values captured for comparison and undo.

| Field | Type | Rules |
|---|---|---|
| `Identity` | `PlacementIdentity` | Required |
| `Position` | Vector3 | Finite |
| `Rotation` | Vector3 | Finite; convention supplied by the target era profile |
| `Scale` | Vector3/float | Preserved when supported; unsupported scale is a refusal, not a fallback |
| `BoundsMin`, `BoundsMax` | Vector3? | Required for world models when present |
| `RawChunkHash` | string | Hash of the source placement-bearing input used for stale-preview detection |

### `AssetMatchCandidate`

An existing game-object corpus candidate, reused from the current PM4 matching concepts.

| Field | Type | Rules |
|---|---|---|
| `AssetId` | string | Required corpus identity |
| `AssetPath` | string | Must resolve in the configured client corpus before cloning |
| `Kind` | enum | Must be compatible with target placement kind |
| `Rank` | int | Positive, lower is better |
| `Score` | double | Finite, clamped to [0, 1] for presentation |
| `ScoreBreakdown` | map<string,double> | Each used signal independently named |
| `Rationale` | list<string> | Human-readable evidence and exclusions |
| `Status` | enum | `Matched`, `Ambiguous`, `Unresolved`, `Ineligible` |

### `ReconciliationProposal`

An immutable, side-effect-free suggestion shown in the viewport.

| Field | Type | Rules |
|---|---|---|
| `ProposalId` | string | Deterministic ID over guide, placement, candidate, and source hashes |
| `Guide` | `Pm4GuideIdentity` | Required |
| `ExistingPlacement` | `PlacementIdentity?` | Present for align/substitute; absent for clone |
| `Current` | `PlacementSnapshot?` | Required when an existing placement is involved |
| `Action` | enum | `Align`, `Substitute`, `Clone` |
| `Candidate` | `AssetMatchCandidate` | Required for substitute/clone; optional for align |
| `ProposedPosition` | Vector3 | Finite and inside target tile bounds unless explicitly transferred |
| `ProposedRotation` | Vector3 | Finite and target-era expressible |
| `ProposedScale` | Vector3/float | Finite and target-era expressible |
| `Residual` | map<string,double> | Position/footprint/height residuals; finite |
| `Confidence` | double | Presentation/sorting value only; never an implicit approval. Align uses `exp(-positionResidual/25)`; clone/substitute keep the scorer score. |
| `Evidence` | list | Signals, values, versions, and provenance paths |
| `Status` | enum | `ReviewRequired`, `Unsupported`, `Conflict`, `AlreadyAligned`, `Accepted`, `Rejected` |

An align proposal retains the existing asset path. A substitute proposal changes only the asset reference.
A clone proposal allocates a new target ID during apply, not during preview.

### `ReviewDecision`

The user's explicit choice for one proposal or reviewed batch.

| Field | Type | Rules |
|---|---|---|
| `DecisionId` | string | Unique |
| `ProposalId` | string | Must refer to an unchanged preview |
| `Disposition` | enum | `Accept`, `Reject` |
| `Reviewer` | string? | Optional local operator label |
| `Reason` | string? | Required for rejection of a matched proposal and for override notes |
| `CreatedUtc` | timestamp | Required |

There is no `AutoAccept` disposition. Ambiguous, unresolved, unsupported, and stale proposals cannot be
accepted without a new explicit user selection that resolves the conflict.

### `ReconciliationBatch`

The atomic editor operation submitted to the session/undo service.

| Field | Type | Rules |
|---|---|---|
| `BatchId` | string | Unique |
| `GuideSources` | list<string> | Read-only PM4 paths and fingerprints |
| `Decisions` | list<`ReviewDecision`> | Accepted decisions only at apply time; rejected decisions remain in report |
| `Targets` | list<string> | Distinct output ADT/WDT paths |
| `SourceHashes` | map<string,string> | Must match before write |
| `AllocatedIds` | map<string,int> | Filled deterministically during validation/apply |
| `OperationState` | enum | `Preview`, `Validated`, `Applied`, `Undone`, `Failed` |

### `ReconciliationReport`

The sidecar artifact written beside successful output and retained for failures.

It contains tool/version, configured source/output roots, build fingerprints, guide/placement identities,
all proposals and decisions, writes attempted/completed, name-table remaps, ID allocations, hashes,
refusals, and validation read-back results. It contains no client asset bytes.

## Relationships and state transitions

```text
Pm4GuideIdentity
    -> Pm4GuideObservation
    -> AssetMatchCandidate (0..n)
    -> ReconciliationProposal (0..n)
PlacementIdentity -> PlacementSnapshot -> ReconciliationProposal (0..1)
ReconciliationProposal -> ReviewDecision (0..1 before apply)
ReviewDecision(s) -> ReconciliationBatch -> ReconciliationReport
```

```text
Preview -> Validated -> Applied -> Undone
   |          |          |
   +------> Rejected   Failed
```

`Applied` requires all target source hashes, supported-era checks, placement IDs, name-table references,
and tile bounds to pass. `Failed` is terminal for that batch; no target is considered changed unless all
target outputs were committed. An undo restores the pre-apply placement/name-table state through the
editor session operation.

