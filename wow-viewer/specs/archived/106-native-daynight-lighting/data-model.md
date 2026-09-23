# Data Model

## NativeWorldLightDirectionModel

| Field | Meaning |
|---|---|
| `buildIdentity` | Exact executable/build identity the model applies to |
| `revision` | Immutable evidence revision |
| `rayAzimuthSamples` | Native theta samples in radians |
| `rayPolarSamples` | Native phi samples in radians |
| `timeMarkers` | Normalized periodic markers in `[0,1)` |
| `semantics` | `native_downward_ray` or `viewer_toward_source` |
| `evidenceReference` | Architecture/research location and runtime proof identity |

Evaluation wraps normalized time, interpolates adjacent samples across midnight, computes the native ray, then inverts only when the consumer requests a source vector.

## NativeWorldLightCoordinateTransform

| Field | Meaning |
|---|---|
| `revision` | Immutable transform revision |
| `buildIdentity` | Direction-model build identity |
| `viewerRenderPath` | Exact viewer/capture path the transform applies to |
| `axisMap` | Signed permutation mapping native vector axes to viewer axes |
| `state` | `unproven` or `calibrated` |
| `comparisonEvidence` | Lock and held-out comparison artifact identities |

An `unproven` transform may support diagnostics only. Client-exact capture requires `calibrated`.

## TerrainLightingProfile

| Field | Meaning |
|---|---|
| `profileRevision` | Immutable whole-profile identity |
| `evidenceState` | `authored`, `partially_proven`, or `client_exact` |
| `directionModel` | Build-scoped direction model reference |
| `coordinateTransform` | Native-to-viewer transform reference |
| `colorSource` | Exactly one LIT global-clear or DBC Light* resolution |
| `mcsh` | Presence, coefficient, and its own evidence state |
| `shadowProjection` | Separate fixed-angle dynamic-shadow identity |

The profile produces one `TerrainLightingSample` plus native/viewer vectors and all source revisions.

## CaptureLightingProvenance

Extends the existing hash-bound sidecar with `build_identity`, `direction_model_revision`, `direction_transform_revision`, `direction_transform_state`, `native_light_ray`, `viewer_light_direction`, `color_source_kind`, `color_source_identity`, `shadow_system`, and `mcsh_evidence_state`.

## DatasetLightingVariant

Each synthetic RGB row records `source_group_id`, partition, time, profile revision, all capture provenance identifiers, source rights class, and input/output hashes. Source-group rows must share one partition.
