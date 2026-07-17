# Data Model: Minimap Lighting Calibration and Lighting-Aware Terrain Reconstruction

## MinimapShadingMatchCandidate

A transient, per-tile probe rendered during inference only -- not persisted independently.

| Field | Meaning | Validation |
|---|---|---|
| `TimeOfDayHours` | The candidate clock time swept for this tile | `0 <= value < 24` |
| `RenderedTerrainRgb` | Output of `TerrainMinimapCompositor` at that time, same tile/resolution as the authored minimap | produced only through the production compositor path, never a reimplementation |
| `DirectionalStructureScore` | Similarity of this candidate's shading-gradient field to the authored minimap's, tint-independent | `[0, 1]`; higher is a better directional fit |

## Extended MinimapLightingProvenance (additive fields)

Extends the existing tint-based record (`src/core/WowViewer.Core/Maps/MinimapLightingProvenance.cs`)
without removing or renaming any current field.

| Field | Meaning | Validation |
|---|---|---|
| `ShadingMatchStatus` | `matched`, `low_confidence_ambiguous`, `low_confidence_flat_terrain`, or `not_evaluated` | mirrors the existing "inference, not capture-proof" discipline; never a bare guess |
| `ShadingMatchedTimeOfDayHours` | Best-fit candidate time-of-day | present only when `ShadingMatchStatus = matched` |
| `ShadingMatchConfidence` | Confidence derived from the score margin between the best and second-best candidate | `[0, 1]`; low margin forces `low_confidence_ambiguous` regardless of raw score |
| `ShadingMatchEvidence` | Fixed evidence string, e.g. `"directional_structure_match_not_capture_proof"` | always present when `ShadingMatchStatus != not_evaluated`, matching the existing `TimeOfDayEvidence` convention |
| `ShadingMatchExcludedMcshFraction` | Fraction of pixels excluded/down-weighted as likely-baked MCSH shadow before scoring | `[0, 1]`; `0` when no MCSH correlation was available |
| `ShadingMatchBuildFingerprint` | Client build identity the candidates were rendered against | non-empty; always 0.5.3.3368-scoped per this feature's boundary |

**Validation rule**: `ShadingMatchStatus = not_evaluated` whenever the tile lacks either an authored
minimap or ground-truth terrain sufficient to render a candidate (FR-001), or the build is not
0.5.3.3368 (FR-006). This mirrors, and is independent from, the existing tint-based
`InferenceStatus`/`NotEvaluated` path -- a tile can be tint-`matched` and shading-`not_evaluated`, or
vice versa.

## LightingBucketDistributionReport

Derived, not stored as a new artifact format -- computed from the extended `MinimapLightingProvenance`
fields across a build's Zarr store and reported as a Parquet-backed summary (constitution principle V:
no NPZ).

| Field | Meaning |
|---|---|
| `BuildFingerprint` | The 0.5.3.3368 build this report covers |
| `MapName` | Map scope of this row, or `"__all__"` for the whole-build summary |
| `BucketCounts` | Count of tiles per `ShadingMatchedTimeOfDayHours` bucket (fixed bucket edges, documented in the report itself) |
| `NotEvaluatedCount` | Tiles with `ShadingMatchStatus = not_evaluated` |
| `LowConfidenceCount` | Tiles with either low-confidence status |
| `TotalEligibleTiles` | Tiles with both an authored minimap and ground-truth terrain, regardless of outcome |

**Validation rule**: `sum(BucketCounts) + NotEvaluatedCount + LowConfidenceCount == TotalEligibleTiles`
always holds; the report generator fails loudly rather than silently dropping a tile from every
category.

## RebalancedTrainingSamplingPlan

| Field | Meaning | Validation |
|---|---|---|
| `SourceReport` | The `LightingBucketDistributionReport` this plan was derived from | must reference a specific build fingerprint |
| `BucketWeights` | Per-bucket sampling weight applied to synthetic-lighting-variant generation | non-negative; normalized to sum to 1 across buckets with any real coverage |
| `SparseBucketPolicy` | How buckets with zero/near-zero real examples are handled | must be an explicit documented policy (e.g. "retain existing synthetic coverage, flag as `no_real_baseline`"), never silent fabrication of an implied real count |
| `LeakSafetyTagsPreserved` | Confirms `source_group_id`/`lighting_variant_id` tagging is untouched by reweighting | boolean; a plan that would break tagging is rejected, not silently applied |

## ReconstructionCheckpointComparison

| Field | Meaning | Validation |
|---|---|---|
| `BaselineCheckpoint` | Identity of the currently deployed checkpoint | must resolve to a real, currently-deployed artifact |
| `CandidateCheckpoint` | Identity of the checkpoint trained on rebalanced data | produced only after the Phase 3 execution gate is explicitly authorized |
| `HeldOutSplit` | The evaluation split used | reuses the existing Spec 108 group-held-out contract (research.md); never a feature-specific split |
| `Outcome` | `improved`, `regressed`, or `inconclusive` | required before any promotion decision |
| `PromotionDecision` | Whether `CandidateCheckpoint` replaces `BaselineCheckpoint` | MUST be `false` whenever `Outcome = regressed` |
