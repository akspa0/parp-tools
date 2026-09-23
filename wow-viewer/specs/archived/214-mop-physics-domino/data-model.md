# Data Model: 5.0.1 Physics

## `PhysicsEraProfile`

Immutable capability decision for one recognized client-build family.

| Field | Meaning | Validation |
|---|---|---|
| `ProfileId` | Stable profile identity | Non-empty and versioned |
| `BuildIdentity` | Recognized build/family selector | Never inferred for unknown builds |
| `PhysicsAvailability` | `Enabled`, `Disabled`, or `Unknown` | Alpha is disabled; unknown stays unknown |
| `EvidenceSource` | Address-cited evidence reference | Required for every result |
| `Capabilities` | Evidence-verified features | Empty until proven |

## `PhysicsDecision`

Resolution result for a model/build pair. It carries the era profile, activation status, and
diagnostics even if simulation is disabled, so a caller can distinguish known absence from unknown
evidence.

## `PhysicsBudget`

Configuration, not a solver object: `IsEnabled`, `CullDistance`, and `MaximumActiveObjects`.
Validation rejects negative/non-finite distance and negative capacity. Every deferred candidate has a
reason; nothing is silently dropped.

## `PhysicsCandidate`

Current solver-independent admission input: non-empty unique `StableId`, finite non-negative
`Distance`, and integer `Priority`. Lower priority values win, then nearer distance, then ordinal
stable id. Duplicate stable ids are rejected because they would make capacity ownership and
diagnostics ambiguous.

## `PhysicsAdmissionDecision`

Exactly one result is returned for each candidate, in original input order. It carries candidate id,
admission/refusal reason, era activation state, full evidence provenance, and a non-empty diagnostic.
Known-disabled and unknown builds are distinct outcomes; neither can be promoted to admitted.

## `PhysicsAssetDescription`

Future parsed sidecar result: source asset identity, provenance, verified body/shape/joint records,
and diagnostics for malformed/unsupported data. Until Phase 2, record layout is intentionally
unknown.

## State transitions

An unknown build is flagged until a new evidence profile exists. A known-disabled build never reaches
discovery or scheduling. A known-enabled build becomes eligible, then active, distance-culled, or
budget-deferred. A cull-deferred instance retains the state needed for continuous resumption.
